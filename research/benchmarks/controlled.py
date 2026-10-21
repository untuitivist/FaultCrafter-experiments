from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, wasserstein_distance
from torch import nn

from .classifier import run_classifier, save_torch, state_hash
from .common import file_hash, identity, log, read_json, seed_for, timestamp, write_json
from .costs import model_cost
from .data import CHANNELS
from .generative_models import ConditionUNet
from .generators import generator_data
from .pools import source_pool, target_pool
from .quality import features, frechet, mmd_squared, reference_context, transform


PANELS = {"hdu": ("bearing_upper", "16.6666666667_600"), "dirg": ("A1_y", "200_1000")}
SAMPLING_CALLS = 32
VARIANTS = (("flow_x", "x", "full"), ("flow_epsilon", "epsilon", "full"),
            ("flow_v", "v", "full"), ("ddpm", "epsilon", "full"),
            ("single_channel", "v", "single"), ("equal_multichannel", "v", "equal"))
COMPARISONS = {"prediction": ("flow_x", "flow_epsilon", "flow_v"),
               "mechanism": ("ddpm", "flow_v"),
               "multichannel": ("single_channel", "equal_multichannel", "flow_v")}


class ReferenceGenerator(nn.Module):
    def __init__(self, condition_dimension, mechanism="flow", prediction="v"):
        super().__init__()
        if mechanism not in ("flow", "ddpm") or prediction not in ("x", "epsilon", "v"):
            raise ValueError("Unknown generation mechanism or direct prediction target")
        if mechanism == "ddpm" and prediction != "epsilon":
            raise ValueError("The controlled DDPM comparison uses direct noise prediction")
        self.network = ConditionUNet(condition_dimension)
        self.mechanism = mechanism
        self.prediction = prediction
        self.register_buffer("alpha", (1 - torch.linspace(0.0001, 0.02, 1000)).cumprod(dim=0))

    def forward(self, signal, time, condition):
        return self.network(signal, time * 1000, condition)

    def velocity(self, signal, time, condition):
        clipped = time.clamp(0.001, 0.999)
        output = self(signal, clipped, condition)
        if self.prediction == "x":
            return (output - signal) / (1 - clipped[:, None, None])
        if self.prediction == "epsilon":
            return (signal - output) / clipped[:, None, None]
        return output

    def loss(self, signal, condition):
        noise = torch.randn_like(signal)
        if self.mechanism == "ddpm":
            indices = torch.randint(1000, (len(signal),), device=signal.device)
            alpha = self.alpha[indices, None, None]
            noisy = alpha.sqrt() * signal + (1 - alpha).sqrt() * noise
            prediction = self(noisy, indices / 1000, condition)
            return (prediction - noise).square().mean()
        time_value = torch.rand(len(signal), device=signal.device) * 0.998 + 0.001
        time_tensor = time_value[:, None, None]
        noisy = time_tensor * signal + (1 - time_tensor) * noise
        return (self.velocity(noisy, time_value, condition) - (signal - noise)).square().mean()

    @torch.no_grad()
    def sample(self, condition, steps=SAMPLING_CALLS, length=1024):
        signal = torch.randn(len(condition), 1, length, device=condition.device)
        if self.mechanism == "flow":
            for index in range(steps):
                time_value = torch.full((len(condition),), index / steps, device=condition.device)
                signal = signal + self.velocity(signal, time_value, condition) / steps
            return signal
        schedule = torch.linspace(999, 0, steps, device=condition.device).long()
        for position, index in enumerate(schedule):
            current = self.alpha[index]
            prediction = self(signal, (index / 1000).expand(len(signal)), condition)
            estimate = (signal - (1 - current).sqrt() * prediction) / current.sqrt()
            if position + 1 == len(schedule):
                signal = estimate
            else:
                following = self.alpha[schedule[position + 1]]
                ratio = current / following
                variance = (1 - following) / (1 - current) * (1 - ratio)
                mean = following.sqrt() * (1 - ratio) / (1 - current) * estimate
                mean = mean + ratio.sqrt() * (1 - following) / (1 - current) * signal
                signal = mean + variance.clamp_min(0).sqrt() * torch.randn_like(signal)
        return signal


def selection(dataset, device, reference_channel, group):
    units = dataset.units["train"]
    quota = dataset.quota["train"]
    channels = CHANNELS[device]
    by_cell = defaultdict(list)
    for unit, (current, channel, state, condition) in enumerate(units):
        by_cell[current, state, condition].append((channel, int(dataset.unit_starts["train"][unit])))
    groups = defaultdict(list)
    receipts = []
    for cell_number, ((current, state, condition), entries) in enumerate(sorted(by_cell.items())):
        indices = []
        for channel, start in entries:
            if current != device or group == "full":
                chosen = np.arange(quota)
            elif group == "single":
                chosen = np.arange(quota) if channel == reference_channel else np.empty(0, dtype=int)
            else:
                rank = (channels.index(channel) - cell_number) % len(channels)
                count = quota // len(channels) + (rank < quota % len(channels))
                chosen = np.arange(count)
            indices.extend((start + chosen).tolist())
            if len(chosen):
                original = dataset.receipts["train"][next(index for index, item in enumerate(dataset.receipts["train"])
                    if (item["device"], item["channel"], item["state"], item["condition"]) == (current, channel, state, condition))]
                receipts.append({**original, "samples": [original["samples"][index] for index in chosen]})
        groups[current].append(np.asarray(indices, dtype=np.int64))
    return list(groups.values()), receipts


def batch(dataset, groups, rng, count=8):
    indices = []
    for _ in range(count):
        cells = groups[int(rng.integers(len(groups)))]
        candidates = cells[int(rng.integers(len(cells)))]
        indices.append(int(rng.choice(candidates)))
    return tuple(torch.as_tensor(values[indices], device="cuda") for values in dataset.cpu["train"])


def descriptors(signals):
    values = signals[:, 0].astype(np.float64)
    rms = np.sqrt(np.square(values).mean(axis=-1))
    statistics = np.stack((rms, np.ptp(values, axis=-1), np.abs(values).max(axis=-1) / np.maximum(rms, 1e-12),
                           kurtosis(values, axis=-1, fisher=False)), axis=1)
    spectrum = np.abs(np.fft.rfft(values, axis=-1)) ** 2
    bands = np.stack([band.mean(axis=-1) for band in np.array_split(spectrum, 16, axis=-1)], axis=1)
    return np.log(np.maximum(np.concatenate((statistics, bands), axis=1), 1e-12))


def quality_score(real, generated, scales):
    first, second = descriptors(real), descriptors(generated)
    return float(np.mean([wasserstein_distance(first[:, index], second[:, index]) / scales[index]
                          for index in range(first.shape[1])]))


def run_case(data_run, comparisons, output, device, split, variant, prediction, group, updates, steps, references):
    if steps != SAMPLING_CALLS:
        raise ValueError(f"Controlled comparisons require {SAMPLING_CALLS} network calls per sample")
    if (variant, prediction, group) not in VARIANTS:
        raise ValueError("Unknown controlled comparison configuration")
    mechanism = "ddpm" if variant == "ddpm" else "flow"
    channel, target = PANELS[device]
    specification = {"device": device, "channel": channel, "target": target, "split": split,
                     "variant": variant, "prediction": prediction, "group": group,
                     "updates": updates, "sampling_steps": steps, "sampling_nfe": steps,
                     "mechanism": mechanism, "loss_space": "epsilon" if mechanism == "ddpm" else "v",
                     "sampler": "skipped_reverse_diffusion" if mechanism == "ddpm" else "euler",
                     "comparisons": [name for name, variants in COMPARISONS.items() if variant in variants],
                     "network": "shared_condition_unet_reference"}
    identifier = identity(specification)[:24]
    folder = output / "controlled" / identifier
    code_hash = file_hash(Path(__file__))
    if (folder / "result.json").exists():
        previous = read_json(folder / "result.json")
        if previous["code_sha256"] != code_hash:
            raise RuntimeError("Controlled experiment code changed; choose a fresh analysis directory")
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    base_job = {"kind": "generator", "track": "T1", "device": device, "target": target, "split": split}
    dataset = generator_data(data_run, "T1", device, target, split)
    groups, receipts = selection(dataset, device, channel, group)
    selection_id = identity(receipts)
    torch.manual_seed(split)
    model = ReferenceGenerator(dataset.schema.dimension, mechanism, prediction).cuda()
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, fused=True)
    rng = np.random.default_rng(seed_for(device, split, "controlled_training"))
    validation_rng = np.random.default_rng(seed_for(device, split, "controlled_validation"))
    validation_indices = dataset.indices(32, validation_rng, role="validation")
    validation = tuple(torch.as_tensor(values[validation_indices], device="cuda") for values in dataset.cpu["validation"])
    reference_indices = np.concatenate([np.arange(int(start), int(start) + dataset.quota["train"])
        for start, descriptor in zip(dataset.unit_starts["train"], dataset.units["train"])
        if descriptor[:2] == (device, channel)])
    reference_descriptors = descriptors(dataset.cpu["train"][0][reference_indices[::max(1, len(reference_indices) // 512)]])
    scales = np.maximum(np.subtract(*np.percentile(reference_descriptors, [75, 25], axis=0)), 1e-6)
    history, best, training_seconds = [], float("inf"), 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        signals, conditions = batch(dataset, groups, rng)
        torch.cuda.synchronize()
        training_started = time.perf_counter()
        loss = model.loss(signals, conditions)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite controlled training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - training_started
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed_for(device, split, "controlled_validation_noise"))
            generated_validation = model.sample(validation[1], steps=steps).cpu().numpy()
        if not np.isfinite(generated_validation).all():
            raise RuntimeError("Nonfinite controlled source-validation generation")
        score = quality_score(validation[0].cpu().numpy(), generated_validation, scales)
        if score < best:
            best = score
            save_torch(folder / "best.pt", {"model": model.state_dict(), "specification": specification,
                       "schema": dataset.schema.receipt(), "step": step, "source_quality": score})
        history.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), "source_quality": score})
        log("controlled_progress", device=device, split=split, variant=variant, step=step, loss=float(loss.detach()))
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"])
    model.eval()
    forward_cost = model_cost(model, (validation[0][:1], torch.full((1,), 0.5, device="cuda"), validation[1][:1]))
    sampler_cost = model_cost(model, (validation[1][:1],), operation=lambda condition: model.sample(condition, steps=steps), scope="complete_sampler")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        with torch.no_grad():
            model.sample(validation[1][:1], steps=steps)
            torch.cuda.synchronize()
            sampling_started = time.perf_counter()
            model.sample(validation[1][:1], steps=steps)
            torch.cuda.synchronize()
            sampling_seconds = time.perf_counter() - sampling_started
    job = {**base_job, "kind": "cnn", "method": "controlled_reference", "channel": channel,
           "arm": "real_generated", "parent": identifier, "id": identifier,
           "protocol_sha256": file_hash(Path(__file__))}
    pool = source_pool(data_run, job)
    outputs, labels = [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed_for(device, channel, split, "controlled_generation"))
        for state in pool["states"]:
            encoded = dataset.schema.encode(device, channel, state, dataset.schema.coordinates[device, target])
            conditions = torch.as_tensor(np.repeat(encoded[None], 63, axis=0), device="cuda")
            outputs.extend(model.sample(conditions[start:start + 8], steps=steps).cpu().numpy() for start in range(0, 63, 8))
            labels.extend([[part in pool["labels"][state] for part in pool["components"]]] * 63)
    generated, labels = np.concatenate(outputs), np.asarray(labels, dtype=np.float32)
    if not np.isfinite(generated).all():
        raise RuntimeError("Nonfinite controlled target generation")
    np.savez_compressed(folder / "signals.npz", x=generated, y=labels)
    checkpoint_sha = file_hash(folder / "best.pt")
    generator_peak = torch.cuda.max_memory_allocated()
    receipt = {"checkpoint_sha256": checkpoint_sha, "signals_sha256": file_hash(folder / "signals.npz"),
               "specification": specification, "source_identity": selection_id}
    diagnostic = run_classifier(data_run, job, output_root=output, smoke_updates=updates, generated=(generated, labels, receipt))
    del model, optimizer, validation
    torch.cuda.empty_cache()
    reference = references["T1", device, channel, target, split]
    encoder, fitted, _, encoder_sha = reference_context(data_run, comparisons, reference, "cuda")
    real, real_labels, valid, _ = target_pool(data_run, job, pool, diagnostic["best_checkpoint_sha256"], output_folder=folder)
    real_features, fake_features = features(encoder, real), features(encoder, generated)
    per_state = []
    for state in pool["states"]:
        vector = np.asarray([part in pool["labels"][state] for part in pool["components"]])
        real_indices = np.flatnonzero(np.all(real_labels == vector, axis=1) & valid)[:63]
        fake_indices = np.flatnonzero(np.all(labels == vector, axis=1))[:len(real_indices)]
        first = transform(real_features["embedding"][real_indices], fitted["embedding"])
        second = transform(fake_features["embedding"][fake_indices], fitted["embedding"])
        per_state.append({"state": state, "embedding_mmd2": mmd_squared(first, second, fitted["embedding"][3]),
                          "frechet": frechet(first, second)})
    result = {"id": identifier, **specification, "code_sha256": code_hash,
              "initial_weights_sha256": initial_hash, "source_identity": selection_id,
              "best_checkpoint_sha256": checkpoint_sha, "reference_cnn_sha256": encoder_sha,
              "real_only": read_json(comparisons / "jobs" / reference / "result.json")["target"],
              "diagnostic": diagnostic["target"], "cost": forward_cost, "sampler_cost": sampler_cost,
              "sampling_seconds_batch_one": sampling_seconds,
              "training_seconds": training_seconds, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_bytes": generator_peak, "history": history,
              "metrics": {key: float(np.mean([item[key] for item in per_state])) for key in ("embedding_mmd2", "frechet")},
              "per_state": per_state, "completed": timestamp()}
    write_json(folder / "source_manifest.json", {"train": receipts, "validation": dataset.receipts["validation"]})
    write_json(folder / "result.json", result)
    gc.collect()
    return result
