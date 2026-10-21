from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import file_hash, read_json, timestamp, write_json
from .generative_models import build_model
from .generators import ConditionSchema


def generate(generator, inventory, requests, output, *, device="cpu", steps=32, seed=42, batch_size=8, capture=()):
    output.mkdir(parents=True, exist_ok=True)
    result = read_json(generator / "result.json")
    checkpoint_path = generator / "best.pt"
    if file_hash(checkpoint_path) != result["best_checkpoint_sha256"]:
        raise RuntimeError("Generator checkpoint checksum mismatch")
    schema = ConditionSchema(read_json(inventory)["records"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != schema.receipt():
        raise RuntimeError("Checkpoint condition schema differs from the supplied inventory")
    method = result.get("job", {}).get("method", "controlled_reference")
    if result.get("network") == "shared_condition_unet_reference":
        from .controlled import ReferenceGenerator

        model = ReferenceGenerator(schema.dimension, "ddpm" if result["variant"] == "ddpm" else "flow", result["prediction"])
    else:
        model = build_model(method, schema.dimension)
    model = model.to(device).eval()
    model.load_state_dict(checkpoint["model"])
    modules = dict(model.named_modules())
    if any(name not in modules for name in capture):
        raise ValueError(f"Unknown modules; available names: {list(modules)}")
    traces = defaultdict(list)
    handles = []

    def recorder(name):
        def record(module, inputs, outputs):
            tensors = [outputs] if isinstance(outputs, torch.Tensor) else [value for value in outputs if isinstance(value, torch.Tensor)]
            for tensor_index, value in enumerate(tensors):
                values = value.detach().float().flatten(start_dim=1) if value.ndim > 1 else value.detach().float()[:, None]
                traces[f"{name}_{tensor_index}_mean"].append(values.mean(dim=1).cpu().numpy())
                traces[f"{name}_{tensor_index}_std"].append(values.std(dim=1, unbiased=False).cpu().numpy())
                traces[f"{name}_{tensor_index}_features"].append(values.cpu().numpy())
        return record

    for name in capture:
        handles.append(modules[name].register_forward_hook(recorder(name)))
    torch.manual_seed(seed)
    signals, conditions, request_ids = [], [], []
    try:
        with torch.inference_mode():
            for request_index, request in enumerate(requests):
                if request.get("length", 1024) != 1024:
                    raise ValueError("These external comparator checkpoints use native 1024-point signals")
                if int(request.get("count", 32)) < 1:
                    raise ValueError("Sample counts must be positive")
                condition = schema.encode(request["device"], request["channel"], request["state"], request["coordinates"])
                for start in range(0, int(request.get("count", 32)), batch_size):
                    count = min(batch_size, int(request.get("count", 32)) - start)
                    batch = torch.as_tensor(np.repeat(condition[None], count, axis=0), device=device)
                    generated = model.sample(batch, steps=steps, length=1024).float().cpu().numpy()
                    if not np.isfinite(generated).all():
                        raise RuntimeError("The frozen generator produced nonfinite signals")
                    signals.append(generated)
                    conditions.append(batch.cpu().numpy())
                    request_ids.extend([request_index] * count)
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(output / "signals.npz", normalized=np.concatenate(signals), conditions=np.concatenate(conditions),
                        request_ids=np.asarray(request_ids))
    for name, values in traces.items():
        np.savez_compressed(output / ("trace_" + name.replace(".", "_") + ".npz"),
                            **{f"call_{index:06d}": value for index, value in enumerate(values)})
    write_json(output / "inference.json", {"created": timestamp(), "requests": requests, "seed": seed,
               "sampling_steps": steps, "batch_size": batch_size, "device": device, "capture_modules": list(capture),
               "checkpoint_sha256": file_hash(checkpoint_path), "code_sha256": file_hash(Path(__file__)),
               "sampling_nfe_per_batch": 2 * (2 * steps - 1) if method == "iccdm" else 1 if method == "ccgan" else steps,
               "signal_scale": "Parent-normalized scale; no target statistics or amplitude estimates are substituted.",
               "weight_parameters": {name: value.detach().cpu().tolist() for name, value in model.named_parameters()
                                     if value.numel() <= 16}})
    return output / "signals.npz"


def main():
    parser = argparse.ArgumentParser(description="Generate from a final comparator checkpoint without training data.")
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--capture-modules", nargs="*", default=[])
    arguments = parser.parse_args()
    if arguments.steps < 2 or arguments.batch_size < 1:
        raise ValueError("Use at least two sampling steps and a positive batch size")
    torch.set_num_threads(4)
    generate(arguments.generator, arguments.inventory, read_json(arguments.requests), arguments.output,
             device=arguments.device, steps=arguments.steps, seed=arguments.seed, batch_size=arguments.batch_size,
             capture=arguments.capture_modules)


if __name__ == "__main__":
    main()
