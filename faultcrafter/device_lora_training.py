from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch

from faultcrafter.compact_inference import folded_device_lora, is_device_lora, load_compact_model, sample_compact_signal
from faultcrafter.condition_guidance import source_weak_direction_projectors
from faultcrafter.device_onboarding import read_signal, sha256_file, write_json
from faultcrafter.unified_generator import RoutedLoRALinear, UnifiedSignalGenerator, sample_unified_generator
from faultcrafter.unified_signal import normalization_statistics
from faultcrafter.training.objectives import UnifiedJointBatch, batch_objective


class FoldedTrainingQuery(RoutedLoRALinear):
    def __init__(self, query: RoutedLoRALinear, mean: np.ndarray, deviation: np.ndarray) -> None:
        torch.nn.Module.__init__(self)
        self.shared = query.shared
        self.lora_down = query.lora_down
        self.lora_up = query.lora_up
        self.rank = query.rank
        self.num_devices = query.num_devices
        dimension = self.shared.in_features
        transform = torch.eye(dimension, dtype=torch.float64)
        for index in range(len(mean)):
            transform[index + 1, 0] = -float(mean[index]) / float(deviation[index])
            transform[index + 1, index + 1] = 1.0 / float(deviation[index])
        self.register_buffer("input_transform", transform, persistent=False)

    def forward(self, signal: torch.Tensor, device_id: torch.Tensor) -> torch.Tensor:
        if self.num_devices != 1 or torch.any(device_id != 0):
            raise ValueError("new-device folded training requires a single device")
        dimension = self.shared.in_features
        shared_weight = self.shared.weight.cpu().double()
        down = self.lora_down[0].cpu()
        up = self.lora_up[0].cpu()
        effective = shared_weight + (up.double() @ down.double()) / float(self.rank)
        delta = effective @ self.input_transform.cpu() - shared_weight
        folded_down = torch.zeros_like(down)
        folded_up = torch.zeros_like(up)
        folded_down[:dimension] = torch.eye(dimension, dtype=down.dtype)
        folded_up[:, :dimension] = (delta * float(self.rank)).to(up)
        shared = self.shared(signal)
        selected_down = folded_down.to(signal).unsqueeze(0).expand(len(signal), -1, -1)
        selected_up = folded_up.to(signal).unsqueeze(0).expand(len(signal), -1, -1)
        latent = torch.einsum("b...i,bri->b...r", signal, selected_down)
        return shared + torch.einsum("b...r,bor->b...o", latent, selected_up) / float(self.rank)


class UploadedCorpus:
    def __init__(self, job_root: Path, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.rows = manifest["records"]
        self.signals = []
        self.cells: dict[str, dict[tuple, list[int]]] = {"train": {}, "validation": {}}
        self.audit = []
        signal_roles: dict[str, str] = {}
        for index, row in enumerate(self.rows):
            path = job_root / "uploads" / row["file"]
            values, audit = read_signal(path, channel=manifest["channel"], array_key=manifest.get("array_key", "signal"))
            digest = audit["signal_sha256"]
            if digest in signal_roles and signal_roles[digest] != row["split"]:
                raise ValueError("同一信号内容出现在训练与验证中，不能作为独立记录")
            signal_roles[digest] = row["split"]
            self.signals.append(values)
            parameters = tuple(row["conditions"][name] for name in manifest["parameter_names"])
            self.cells[row["split"]].setdefault((row["fault"], *parameters), []).append(index)
            self.audit.append({"record_id": row["record_id"], "split": row["split"], "file_sha256": sha256_file(path), **audit})
        self.parameters = np.asarray([[row["conditions"][name] for name in manifest["parameter_names"]] for row in self.rows], dtype=np.float32)
        training_parameters = self.parameters[[row["split"] == "train" for row in self.rows]]
        self.mean = training_parameters.mean(axis=0)
        self.deviation = training_parameters.std(axis=0)
        self.deviation[self.deviation < 1e-6] = 1.0
        self.trace = hashlib.sha256()

    def batch(self, split: str, length: int, random: np.random.Generator, count: int, *, cell_keys: list[tuple] | None = None) -> UnifiedJointBatch:
        keys = list(self.cells[split])
        chosen = cell_keys if cell_keys is not None else [keys[int(random.integers(len(keys)))] for _ in range(count)]
        count = len(chosen)
        normalized = np.empty((count, 1, length), dtype=np.float32)
        amplitudes = np.empty(count, dtype=np.float32)
        conditions = np.zeros((count, 2), dtype=np.float32)
        faults = np.empty(count, dtype=np.int64)
        groups = np.empty(count, dtype=np.int64)
        speeds = np.empty(count, dtype=np.float32)
        parent_ids = []
        record_indices = []
        for index, key in enumerate(chosen):
            record_index = int(random.choice(self.cells[split][key]))
            values = self.signals[record_index]
            start = int(random.integers(0, len(values) - length + 1))
            parent = values[start:start + length]
            _, amplitude = normalization_statistics(parent, "max_abs")
            normalized[index, 0] = parent / amplitude
            amplitudes[index] = amplitude
            conditions[index, :len(self.mean)] = self.parameters[record_index]
            token = self.manifest["fault_tokens"][key[0]]
            faults[index], groups[index] = token["fault_index"], token["fault_group_index"]
            speeds[index] = self.parameters[record_index, self.manifest["speed_parameter_index"]] * np.float32(self.manifest["speed_to_hz"])
            parent_ids.append(f"{self.rows[record_index]['record_id']}:{start}:{length}")
            record_indices.append(record_index)
        if split == "train":
            self.trace.update(json.dumps(parent_ids).encode("utf-8"))
        return UnifiedJointBatch(
            normalized, amplitudes, np.zeros(count, dtype=np.float32), conditions, faults, groups,
            np.zeros(count, dtype=np.int64), speeds,
            np.full(count, self.manifest["sample_rate_hz"], dtype=np.float32),
            np.full(count, length, dtype=np.float32), np.asarray(parent_ids), np.asarray(record_indices),
        )


def export_adapter(model, shared_path, manifest, corpus, job_root, sampling, *, probe_lengths=(512, 4096, 51802)):
    output = job_root / "export"
    output.mkdir(exist_ok=False)
    digest = sha256_file(shared_path)
    profile = {
        "route_id": f"{manifest['device_id']}.{manifest['sensor_location']}.{manifest['channel']}",
        "dataset_id": manifest["dataset_id"], "device_id": manifest["device_id"],
        "sensor_location": manifest["sensor_location"], "channel": manifest["channel"],
        "signal_quantity": "vibration", "physical_unit": manifest["physical_unit"],
        "sample_rate_hz": manifest["sample_rate_hz"], "amplitude_method": "parent_long_signal_max_abs",
        "time_coordinate": "t = sample_index / sample_rate_hz",
        "angle_coordinate": "theta = integral(2*pi*shaft_speed_hz dt)",
        "shared_sha256": digest, "parameter_names": manifest["parameter_names"],
        "speed_parameter_index": manifest["speed_parameter_index"], "speed_to_hz": manifest["speed_to_hz"],
        "faults": manifest["fault_tokens"], "fixed_conditions": manifest["fixed_conditions"],
        "sampling_policy": manifest["sampling_policy"],
    }
    write_json(output / "signal_profile.json", profile)
    torch.save({"device_id": manifest["device_id"], "shared_sha256": digest,
                "state": folded_device_lora(model, 0, corpus.mean, corpus.deviation)}, output / "device_lora.pt")
    compute = next(model.parameters()).device
    compact, loaded_profile, loaded_sampling = load_compact_model(shared_path, output / "device_lora.pt", output / "signal_profile.json", compute_device=compute)
    conditions = sorted(set(tuple(key[1:]) for key in corpus.cells["train"]) | set(tuple(value) for value in manifest["verification_conditions"]))
    probes = []
    model.eval()
    with torch.no_grad():
        for label, token in profile["faults"].items():
            for condition_index, raw in enumerate(conditions):
                for length in probe_lengths:
                    if length == 51802 and condition_index != 0:
                        continue
                    values = torch.as_tensor(raw, dtype=torch.float32, device=compute)
                    raw_parameters = torch.zeros((1, 2), device=compute)
                    raw_parameters[0, :len(raw)] = values
                    route = torch.zeros(1, dtype=torch.long, device=compute)
                    torch.testing.assert_close(compact.style(raw_parameters, route), model.style(raw_parameters, route), rtol=2e-4, atol=2e-4)
                    random = torch.Generator(device=compute).manual_seed(424242)
                    noise = torch.randn(1, 1, length, generator=random, device=compute)
                    amplitude_noise = torch.randn(1, generator=random, device=compute) if sampling["stochastic_amplitude"] else torch.zeros(1, device=compute)
                    expected = sample_unified_generator(
                        model, noise, raw_parameters, torch.tensor([token["fault_index"]], device=compute),
                        torch.tensor([token["fault_group_index"]], device=compute), torch.zeros(1, dtype=torch.long, device=compute),
                        values[profile["speed_parameter_index"]].reshape(1) * profile["speed_to_hz"],
                        torch.tensor([profile["sample_rate_hz"]], device=compute),
                        steps=sampling["sampling_steps"], terminal_time=sampling["terminal_time"], amplitude_noise=amplitude_noise,
                    )
                    actual = sample_compact_signal(compact, loaded_profile, loaded_sampling,
                                                   condition_parameters=raw, fault_label=label, length=length, seed=424242)
                    for reference, observed in zip(expected, actual, strict=True):
                        torch.testing.assert_close(observed, reference, rtol=2e-4, atol=2e-4)
                    probes.append({"fault": label, "conditions": raw, "length": length,
                                   "max_absolute_errors": [float((reference - observed).abs().max()) for reference, observed in zip(expected, actual, strict=True)]})
    write_json(job_root / "export_audit.json", {"passed": True, "rtol": 2e-4, "atol": 2e-4, "probes": probes,
               "files": {path.name: sha256_file(path) for path in output.iterdir()}, "shared_sha256": digest,
               "runtime_dependencies": ["shared_generator.pt", "device_lora.pt", "signal_profile.json"]})
    return output


def train_device_lora(job_root: Path, compact_root: Path, recipe: dict[str, Any], *, compute_device: str | None = None,
                      progress: Callable[[dict[str, Any]], None] | None = None, probe_lengths=(512, 4096, 51802)) -> Path:
    recipe = copy.deepcopy(recipe)
    execution = "differentiate_through_export_folding_before_every_forward"
    if recipe.get("condition_execution", execution) != execution:
        raise ValueError("unsupported new-device conditioning execution")
    recipe["condition_execution"] = execution
    manifest = json.loads((job_root / "manifest.json").read_text(encoding="utf-8"))
    shared_path = compact_root / "shared_generator.pt"
    digest = sha256_file(shared_path)
    shared = torch.load(shared_path, weights_only=True, map_location="cpu")
    if shared.get("release_id") != "experiment32_full_data_v1" or shared["model_arguments"]["condition_dimension"] != 2:
        raise ValueError("新设备适配要求32全数据发布主干及原两维工况接口")
    if recipe["generator"]["fault_identity_guidance_scale"] != shared["model_arguments"]["fault_identity_guidance_scale"]:
        raise ValueError("训练与发布主干的故障引导不一致")
    torch.set_num_threads(4)
    torch.manual_seed(manifest["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    compute = torch.device(compute_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UnifiedSignalGenerator(**shared["model_arguments"])
    missing, unexpected = model.load_state_dict(shared["state"], strict=False)
    if unexpected or set(missing) != {name for name in model.state_dict() if is_device_lora(name)}:
        raise ValueError("冻结主干内容与新设备模型不匹配")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_device_lora(name))
    model.to(compute)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_lora = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if is_device_lora(name)}
    corpus = UploadedCorpus(job_root, manifest)
    model.style_field.query = FoldedTrainingQuery(model.style_field.query, corpus.mean, corpus.deviation)
    ema = copy.deepcopy(model)
    write_json(job_root / "input_audit.json", {"records": corpus.audit, "shared_sha256": digest,
               "recipe": recipe, "parameter_mean_training_only": corpus.mean.tolist(),
               "parameter_deviation_training_only": corpus.deviation.tolist(),
               "trainable_parameters": sum(value.numel() for value in trainable),
               "frozen_fault_tokens": manifest["fault_tokens"], "initialization": "fresh LoRA; minimum frozen token per semantic group",
               "condition_execution": "differentiate_through_export_folding_before_every_forward"})
    generator = recipe["generator"]
    normalized_parameters = (corpus.parameters[[row["split"] == "train" for row in corpus.rows]] - corpus.mean) / corpus.deviation
    projectors = source_weak_direction_projectors([normalized_parameters], condition_dimension=2,
                 retained_variance=generator["condition_retained_variance"]).to(compute)
    optimizer = torch.optim.AdamW(trainable, lr=generator["learning_rate"], weight_decay=generator["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=generator["lr_decay_factor"],
                patience=generator["lr_plateau_evaluations"], min_lr=generator["minimum_learning_rate"])
    data_random = np.random.default_rng(manifest["seed"])
    noise_random = torch.Generator(device=compute).manual_seed(manifest["seed"] + 100000)
    validation_random = np.random.default_rng(recipe["validation_panel_seed"])
    bounds = recipe["validation_length_range"]
    validation_lengths = sorted(set([*bounds, *validation_random.integers(bounds[0], bounds[1] + 1, recipe["validation_panel_length_count"] - 2).tolist()]))
    history = []
    best = math.inf
    selected_step = None
    best_path = job_root / "best_training_lora.pt"
    points = 0
    for step in range(1, recipe["maximum_steps"] + 1):
        model.train()
        if step <= generator["warmup_steps"]:
            optimizer.param_groups[0]["lr"] = generator["learning_rate"] * step / max(1, generator["warmup_steps"])
        minimum, maximum = recipe["training_length_range"]
        length = int(data_random.integers(minimum, maximum + 1))
        count = max(1, min(generator["maximum_batch_size"], generator["point_budget_per_batch"] // length))
        batch = corpus.batch("train", length, data_random, count)
        optimizer.zero_grad(set_to_none=True)
        loss, components = batch_objective(model, batch, generator, compute_device=compute, random=noise_random,
                                          condition_projectors=projectors, training_step=step)
        if not torch.isfinite(loss):
            raise FloatingPointError("LoRA训练出现非有限损失")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, generator["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for (name, target), (_, source) in zip(ema.named_parameters(), model.named_parameters(), strict=True):
                if is_device_lora(name):
                    target.lerp_(source, 1 - generator["ema_decay"])
        points += count * length
        if step != 1 and step % generator["evaluation_interval"] and step != recipe["maximum_steps"]:
            continue
        row = {"step": step, "maximum_steps": recipe["maximum_steps"], "training_loss": float(loss.detach()),
               "training_points": points, "training_batch_sha256": corpus.trace.hexdigest(), "learning_rate": optimizer.param_groups[0]["lr"]}
        if step % generator["evaluation_interval"] == 0 or step == recipe["maximum_steps"]:
            ema.eval()
            panel_random = np.random.default_rng(recipe["validation_panel_seed"])
            panel_noise = torch.Generator(device=compute).manual_seed(recipe["validation_panel_seed"])
            losses = []
            with torch.no_grad():
                for validation_length in validation_lengths:
                    keys = sorted(corpus.cells["validation"])
                    budget = max(1, min(generator["maximum_batch_size"], generator["point_budget_per_batch"] // validation_length))
                    for begin in range(0, len(keys), budget):
                        panel = corpus.batch("validation", validation_length, panel_random, budget, cell_keys=keys[begin:begin + budget])
                        objective, _ = batch_objective(ema, panel, generator, compute_device=compute, random=panel_noise, condition_projectors=projectors)
                        losses.extend([float(objective)] * len(panel.fault))
            validation = float(np.mean(losses))
            if not np.isfinite(validation):
                raise FloatingPointError("LoRA源验证损失非有限")
            row["validation_objective"] = validation
            eligible = step // generator["evaluation_interval"] >= generator["checkpoint_preheat_evaluations"]
            if eligible and (not np.isfinite(best) or validation < best * (1 - generator["relative_validation_improvement"])):
                best, selected_step = validation, step
                torch.save({name: value.detach().cpu() for name, value in ema.state_dict().items() if is_device_lora(name)}, best_path)
            if step > generator["warmup_steps"]:
                scheduler.step(validation)
        history.append(row)
        write_json(job_root / "history.json", history)
        with (job_root / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"device": manifest["device_id"], **row}, ensure_ascii=False), flush=True)
        if progress:
            progress(row)
    if selected_step is None:
        raise RuntimeError("训练预算未达到source选模预热要求")
    selected = torch.load(best_path, weights_only=True, map_location=compute)
    ema.load_state_dict({**ema.state_dict(), **selected})
    for candidate in (model, ema):
        for name, value in candidate.state_dict().items():
            if not is_device_lora(name) and not torch.equal(value.cpu(), shared["state"][name]):
                raise RuntimeError(f"禁止修改的共享参数发生变化：{name}")
    if not any(not torch.equal(initial_lora[name], value.cpu()) for name, value in selected.items()):
        raise RuntimeError("LoRA没有产生实际更新")
    if sha256_file(shared_path) != digest:
        raise RuntimeError("发布主干文件发生变化")
    write_json(job_root / "training_audit.json", {"completed_steps": recipe["maximum_steps"], "selected_step": selected_step,
               "best_source_validation": best, "shared_parameters_unchanged": True, "shared_sha256": digest,
               "only_lora_trained": True, "training_batch_sha256": corpus.trace.hexdigest(), "training_points": points})
    return export_adapter(ema, shared_path, manifest, corpus, job_root, shared["sampling"], probe_lengths=probe_lengths)
