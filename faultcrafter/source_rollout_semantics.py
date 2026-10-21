from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from faultcrafter.identity_guidance import project_unit_peak
from faultcrafter.unified_generator import UnifiedSignalGenerator, integrate_unified_terminal_inputs


ROLLOUT_SEED_OFFSET = 49_000_000
SEMANTIC_INPUT_MODES = frozenset({"denoising_prediction", "source_rollout"})


@dataclass(frozen=True, slots=True)
class SourceRolloutSampling:
    interval: int
    steps: int
    terminal_time: float

    def __post_init__(self) -> None:
        if type(self.interval) is not int or self.interval <= 0 or type(self.steps) is not int or self.steps < 2:
            raise ValueError("source rollout needs a positive integer interval and complete Heun steps")
        if not math.isfinite(self.terminal_time) or not 0.0 < self.terminal_time < 1.0:
            raise ValueError("source rollout terminal time must lie in(0,1)")

    def due(self, training_step: int) -> bool:
        if type(training_step) is not int or training_step <= 0:
            raise ValueError("source rollout requires the actual positive training step")
        return training_step % self.interval == 0

    def metadata(self) -> dict:
        return {"input_mode": "source_rollout", "interval": self.interval,
                "sampling_steps": self.steps, "terminal_time": self.terminal_time,
                "seed_offset": ROLLOUT_SEED_OFFSET, "gradient_scope": "terminal_forward_only",
                "prediction_normalization": "generated_record_max_abs", "time_gate": "none_completed_rollout"}


def source_rollout_sampling(config: dict) -> SourceRolloutSampling | None:
    generator = config["generator"]
    mode = generator.get("source_semantic_input_mode", "denoising_prediction")
    if mode not in SEMANTIC_INPUT_MODES:
        raise ValueError("unknown source semantic input mode")
    if mode == "denoising_prediction":
        if "source_semantic_rollout_interval" in generator:
            raise ValueError("rollout interval requires source_rollout semantic mode")
        return None
    weight = float(generator.get("source_semantic_loss_weight", 0.0))
    if not math.isfinite(weight) or weight <= 0.0 or generator.get("fault_identity_guidance_mode") != "unit_peak_projection":
        raise ValueError("source rollout requires positive semantic weight and common unit-peak projection")
    if generator.get("resumable_training", False):
        raise ValueError("source rollout requires a complete non-resumed training and exposure audit")
    generation = config["generation"]
    return SourceRolloutSampling(generator["source_semantic_rollout_interval"], generation["sampling_steps"], float(generation["terminal_time"]))


def source_rollout_prediction(
    model: UnifiedSignalGenerator,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *, length: int, sampling: SourceRolloutSampling, base_seed: int, training_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not sampling.due(training_step) or length <= 0:
        raise ValueError("source rollout prediction may run only on its declared training steps and length")
    if model.fault_identity_guidance_mode != "unit_peak_projection" or model.affine_restoration:
        raise ValueError("source rollout requires the declared non-affine unit-peak generator")
    seed = int(base_seed) + ROLLOUT_SEED_OFFSET + training_step
    random = torch.Generator(device=condition_parameters.device).manual_seed(seed)
    noise = torch.randn(len(condition_parameters), 1, length, device=condition_parameters.device,
                        dtype=condition_parameters.dtype, generator=random)
    inputs = integrate_unified_terminal_inputs(
        model, noise, condition_parameters, fault, fault_group, device_id,
        shaft_speed_hz, sample_rate_hz, steps=sampling.steps, terminal_time=sampling.terminal_time,
    )
    if inputs[0].requires_grad or inputs[0].grad_fn is not None:
        raise RuntimeError("source rollout integration must detach its trajectory")
    prediction = model(*inputs).normalized_signal
    prediction, _ = project_unit_peak(prediction)
    return prediction, {"source_semantic_rollout_executed": 1.0, "source_semantic_rollout_seed": seed,
                        "source_semantic_rollout_no_grad_forwards": 2 * sampling.steps,
                        "source_semantic_rollout_gradient_forwards": 1}


def rollout_update_record(training_step: int, batch: Any, diagnostics: dict) -> dict:
    if diagnostics.get("source_semantic_rollout_executed") != 1.0 or batch.source_record_index is None:
        raise ValueError("rollout audit requires a genuinely executed source update")
    length = int(batch.normalized_signal.shape[-1])
    if any(int(value) != length for value in batch.normalization_length):
        raise ValueError("rollout audit must describe complete parent views")
    return {"step": training_step, "seed": int(diagnostics["source_semantic_rollout_seed"]),
            "length": length, "batch_size": len(batch.device_id),
            "device_indices": batch.device_id.tolist(), "fault_indices": batch.fault.tolist(),
            "source_record_indices": batch.source_record_index.tolist(),
            "source_parent_ids": [str(value) for value in batch.parent_segment_id],
            "no_grad_forwards": int(diagnostics["source_semantic_rollout_no_grad_forwards"]),
            "gradient_forwards": int(diagnostics["source_semantic_rollout_gradient_forwards"]),
            "semantic_loss": float(diagnostics["source_semantic_loss"]),
            "source_confidence_mean": float(diagnostics["source_semantic_confidence_mean"])}


def source_rollout_audit(
    sampling: SourceRolloutSampling, device_ids: Sequence[str], fault_labels: Sequence[str],
    updates: list[dict], *, completed_steps: int, planned_steps: int, base_seed: int,
) -> dict:
    expected = list(range(sampling.interval, completed_steps + 1, sampling.interval))
    if [row["step"] for row in updates] != expected or not 0 <= completed_steps <= planned_steps:
        raise ValueError("source rollout completed update schedule differs")
    counts = {device: Counter() for device in device_ids}
    samples = 0
    for row in updates:
        if row["seed"] != base_seed + ROLLOUT_SEED_OFFSET + row["step"]:
            raise ValueError("source rollout seed differs from its independent stream")
        if row["no_grad_forwards"] != 2 * sampling.steps or row["gradient_forwards"] != 1:
            raise ValueError("source rollout audit used a shortened trajectory or different gradient scope")
        if any(len(row[field]) != row["batch_size"] for field in ("device_indices", "fault_indices", "source_record_indices", "source_parent_ids")):
            raise ValueError("source rollout exposure arrays differ")
        if not math.isfinite(row["semantic_loss"]) or row["semantic_loss"] < 0 or not 0 <= row["source_confidence_mean"] <= 1:
            raise ValueError("source rollout objective diagnostics differ")
        for device_index, fault_index, record_index in zip(row["device_indices"], row["fault_indices"], row["source_record_indices"], strict=True):
            if not 0 <= device_index < len(device_ids) or not 0 <= fault_index < len(fault_labels) or record_index < 0:
                raise ValueError("source rollout exposure index is invalid")
            device_id = device_ids[device_index]
            label = fault_labels[fault_index]
            if not label.startswith(device_id + ":"):
                raise ValueError("source rollout fault does not belong to its device")
            counts[device_id][label] += 1
            samples += 1
    return {"status": "complete" if completed_steps == planned_steps else "in_progress", **sampling.metadata(),
            "completed_training_steps": completed_steps, "planned_training_steps": planned_steps,
            "completed_rollout_updates": len(updates), "planned_rollout_updates": planned_steps // sampling.interval,
            "generated_source_samples": samples, "device_fault_sample_counts": {device: dict(values) for device, values in counts.items()},
            "source_reference_only": True, "target_waveforms_used": False, "runtime_extra_parameters": False,
            "base_seed": base_seed, "updates": updates}
