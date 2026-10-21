from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


FULL_START_UNIFORM = "full_start_uniform_v1"
FORMER_TIME_MINIMUM = 0.05


def time_coverage_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_time_coverage_policy", "none")
    if not isinstance(policy, str) or policy not in {"none", FULL_START_UNIFORM}:
        raise ValueError("unknown training time coverage policy")
    if policy == "none":
        return policy
    values = (generator.get("minimum_time"), generator.get("maximum_time"), generator.get("minimum_noise_fraction"))
    if any(type(value) not in (float, int) for value in values) or values != (0.0, 0.95, 0.05):
        raise ValueError("full-start coverage requires time0..0.95 and the original0.05 noise floor")
    defaults = {"initial_parameter_anchor_policy": "none", "initial_prediction_anchor_policy": "none", "trajectory_average_policy": "ema",
                "source_generation_validation_policy": "none", "shared_gradient_policy": "joint", "sharpness_training_policy": "standard",
                "direct_x_time_weighting": "inverse_noise_squared", "normalized_energy_policy": "none", "source_semantic_input_mode": "denoising_prediction"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("full-start coverage cannot combine undeclared training mechanisms")
    if any(generator.get(name, False) for name in ("resumable_training", "initial_validation_fallback", "preserve_frozen_ema", "device_balanced_sampling",
           "device_balanced_validation", "relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference")):
        raise ValueError("full-start coverage requires the original full-model sampling and checkpoint rules")
    if generator.get("active_lora_device_index") is not None or any(float(generator.get(name, 0)) != 0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")):
        raise ValueError("full-start coverage requires original single-backward training without teachers")
    return policy


def observe_training_times(generator: Mapping[str, Any], time: torch.Tensor, device_id: torch.Tensor) -> dict:
    if time_coverage_policy(generator) == "none":
        return {}
    if time.ndim != 1 or not len(time) or not time.is_floating_point() or device_id.shape != time.shape or device_id.dtype != torch.long:
        raise ValueError("time coverage requires aligned nonempty time and device vectors")
    values = time.detach().cpu().tolist()
    routes = device_id.detach().cpu().tolist()
    upper = float(torch.as_tensor(generator["maximum_time"], dtype=time.dtype))
    if any(not math.isfinite(value) or not 0 <= value <= upper for value in values) or any(route < 0 for route in routes):
        raise ValueError("observed time or device is outside the declared coverage")
    counts = {}
    for value, route in zip(values, routes, strict=True):
        bucket = counts.setdefault(str(route), {"samples": 0, "low_time_samples": 0, "exact_zero_samples": 0})
        bucket["samples"] += 1
        bucket["low_time_samples"] += int(value < FORMER_TIME_MINIMUM)
        bucket["exact_zero_samples"] += int(value == 0)
    return {"time_coverage_batch_samples": len(values), "time_coverage_batch_low_time_samples": sum(value < FORMER_TIME_MINIMUM for value in values),
            "time_coverage_batch_exact_zero_samples": sum(value == 0 for value in values),
            "time_coverage_batch_minimum": min(values), "time_coverage_batch_maximum": max(values), "time_coverage_batch_device_counts": counts}


class TimeCoverageActivity:
    def __init__(self, generator: Mapping[str, Any], device_ids: Sequence[str]):
        if time_coverage_policy(generator) != FULL_START_UNIFORM or not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("time coverage activity requires an explicit policy and unique devices")
        self.device_ids = tuple(device_ids)
        self.counts = {device: {"samples": 0, "low_time_samples": 0, "exact_zero_samples": 0} for device in device_ids}
        self.completed_steps = 0
        self.minimum = math.inf
        self.maximum = -math.inf

    def metadata(self) -> dict:
        return {"policy": FULL_START_UNIFORM, "minimum_time": 0.0, "maximum_time": 0.95, "minimum_noise_fraction": 0.05,
                "former_minimum_time": FORMER_TIME_MINIMUM, "expected_former_gap_fraction": 1 / 19,
                "exact_zero_sampling_guaranteed": False, "sampling": "original_native_uniform_call_same_random_stream",
                "loss_weighting": "original_inverse_noise_squared", "source_validation": "same_full_start_interval_and_original_objective",
                "device_ids": list(self.device_ids), "observations_change_training": False, "extra_runtime_parameters_or_files": False}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != self.completed_steps + 1:
            raise ValueError("time coverage must record every consecutive source update once")
        batch_counts = components["time_coverage_batch_device_counts"]
        for field in ("samples", "low_time_samples", "exact_zero_samples"):
            if components["time_coverage_batch_" + field] != sum(bucket[field] for bucket in batch_counts.values()):
                raise ValueError("time coverage per-device and global batch counts differ")
        for route, bucket in batch_counts.items():
            index = int(route)
            if str(index) != route or not 0 <= index < len(self.device_ids) or any(type(value) is not int for value in bucket.values()) or not 0 <= bucket["exact_zero_samples"] <= bucket["low_time_samples"] <= bucket["samples"]:
                raise ValueError("time coverage device routing or counts are invalid")
            for field, value in bucket.items():
                self.counts[self.device_ids[index]][field] += value
        self.minimum = min(self.minimum, components["time_coverage_batch_minimum"])
        self.maximum = max(self.maximum, components["time_coverage_batch_maximum"])
        self.completed_steps = step
        return {"time_coverage_total_steps": step, "time_coverage_total_samples": sum(bucket["samples"] for bucket in self.counts.values()),
                "time_coverage_total_low_time_samples": sum(bucket["low_time_samples"] for bucket in self.counts.values()),
                "time_coverage_total_exact_zero_samples": sum(bucket["exact_zero_samples"] for bucket in self.counts.values()),
                "time_coverage_observed_minimum": self.minimum, "time_coverage_observed_maximum": self.maximum,
                "time_coverage_device_counts": {device: dict(bucket) for device, bucket in self.counts.items()}}
