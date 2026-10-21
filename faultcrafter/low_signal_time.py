from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch


IDENTITY_TIME = "identity"
QUADRATIC_LOW_SIGNAL = "quadratic_low_signal_v1"


def time_transform_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_time_transform", IDENTITY_TIME)
    if policy not in (IDENTITY_TIME, QUADRATIC_LOW_SIGNAL):
        raise ValueError("unknown training time transform")
    return policy


def low_signal_time_metadata(generator: Mapping[str, Any]) -> dict:
    if time_transform_policy(generator) != QUADRATIC_LOW_SIGNAL:
        raise ValueError("low-signal time metadata requires its active policy")
    if tuple(generator.get(name) for name in ("minimum_time", "maximum_time", "minimum_noise_fraction")) != (0.05, 0.95, 0.05):
        raise ValueError("low-signal time retains the original0.05..0.95 time bounds and noise floor")
    defaults = {"training_time_sampling_policy": "independent_uniform", "training_noise_pair_policy": "independent",
                "training_time_coverage_policy": "none", "direct_x_time_weighting": "inverse_noise_squared",
                "order_resolution_policy": "unweighted_order_v1", "shared_gradient_policy": "joint",
                "sharpness_training_policy": "standard", "initial_parameter_anchor_policy": "none",
                "initial_prediction_anchor_policy": "none", "trajectory_average_policy": "ema"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("low-signal time cannot combine undeclared training mechanisms")
    if any(generator.get(name, False) for name in ("relation_coupled_noise_time", "relation_balanced_sampling", "device_balanced_sampling", "resumable_training")):
        raise ValueError("low-signal time requires original finite source sampling")
    return {"policy": QUADRATIC_LOW_SIGNAL, "minimum_time": 0.05, "maximum_time": 0.95,
            "transform": "a+(b-a)*((raw_time-a)/(b-a))**2", "exponent": 2,
            "finite_precision_sequence": "subtract_lower_multiply_reciprocal_square_multiply_width_add_lower",
            "scope": "student_training_with_gradients_only", "source_validation": "unchanged_original_uniform_times",
            "original_noise_and_random_draw_count_unchanged": True, "additional_random_draws": 0,
            "continuous_original_mean_time": 0.5, "continuous_transformed_mean_time": 0.35,
            "continuous_probability_below_midpoint": math.sqrt(0.5), "actual_snr_claimed": False,
            "importance_correction": False, "unbiased_original_training_objective_claimed": False,
            "additional_model_forwards_per_update": 0, "cnn_non_degradation_guaranteed": False,
            "extra_runtime_parameters_or_files": False}


def quadratic_time_transform(time: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if time.ndim != 1 or not len(time) or time.dtype not in (torch.float32, torch.float64):
        raise ValueError("time transform requires a nonempty float32/float64 vector")
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper < 1:
        raise ValueError("time transform requires a finite nonempty interval inside[0,1)")
    if time.requires_grad or not torch.isfinite(time).all() or torch.any(time < lower) or torch.any(time > upper):
        raise ValueError("time transform requires detached finite draws inside the declared interval")
    normalized = (time - lower) * (1.0 / (upper - lower))
    return lower + (upper - lower) * normalized.square()


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def low_signal_training_times(generator: Mapping[str, Any], time: torch.Tensor, noise: torch.Tensor, device_id: torch.Tensor,
                              random: torch.Generator, *, active: bool) -> tuple[torch.Tensor, dict]:
    if type(active) is not bool:
        raise ValueError("time transform activity must be a boolean")
    if time_transform_policy(generator) == IDENTITY_TIME or not active:
        return time, {}
    metadata = low_signal_time_metadata(generator)
    if time.dtype != torch.float32 or noise.dtype != time.dtype or noise.device != time.device or noise.ndim != 3 or noise.shape[:2] != (len(time), 1):
        raise ValueError("low-signal training requires original float32 time and[batch,1,length]noise")
    if device_id.shape != time.shape or device_id.dtype != torch.long or device_id.device != time.device or torch.any(device_id < 0):
        raise ValueError("low-signal time requires aligned nonnegative integer device routes")
    if noise.requires_grad or not torch.isfinite(noise).all():
        raise ValueError("low-signal time requires original finite detached noise")
    transformed = quadratic_time_transform(time, metadata["minimum_time"], metadata["maximum_time"])
    raw_values, values = time.detach().cpu().tolist(), transformed.detach().cpu().tolist()
    diagnostics = {"low_signal_time_executed": 1.0, "low_signal_time_noise_shape": list(noise.shape),
                   "low_signal_time_batch_samples": len(time), "low_signal_time_batch_elements": noise.numel(),
                   "low_signal_time_raw_values": raw_values, "low_signal_time_values": values,
                   "low_signal_time_routes": device_id.detach().cpu().tolist(),
                   "low_signal_time_changed_samples": sum(raw != actual for raw, actual in zip(raw_values, values, strict=True)),
                   "low_signal_time_original_low_count": sum(value < 0.5 for value in raw_values),
                   "low_signal_time_low_count": sum(value < 0.5 for value in values),
                   "low_signal_time_raw_sum": math.fsum(raw_values), "low_signal_time_sum": math.fsum(values),
                   "low_signal_time_noise_sha256": _tensor_digest(noise), "low_signal_time_raw_sha256": _tensor_digest(time),
                   "low_signal_time_sha256": _tensor_digest(transformed), "low_signal_time_rng_after_sha256": _tensor_digest(random.get_state())}
    return transformed, diagnostics


def validate_low_signal_row(row: dict) -> None:
    shape = row["low_signal_time_noise_shape"]
    count = row["low_signal_time_batch_samples"]
    if not isinstance(shape, list) or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape) or shape[:2] != [count, 1]:
        raise ValueError("low-signal time trace has invalid original noise shape")
    if row["low_signal_time_batch_elements"] != math.prod(shape):
        raise ValueError("low-signal time exposure differs from its noise shape")
    raw_values, values, routes = (row["low_signal_time_" + field] for field in ("raw_values", "values", "routes"))
    if any(len(sequence) != count for sequence in (raw_values, values, routes)) or any(type(route) is not int or route < 0 for route in routes):
        raise ValueError("low-signal time trace has invalid sample or route counts")
    raw = np.asarray(raw_values, dtype=np.float32)
    actual = np.asarray(values, dtype=np.float32)
    if not np.isfinite(raw).all() or not np.isfinite(actual).all() or np.any(raw < np.float32(0.05)) or np.any(raw > np.float32(0.95)):
        raise ValueError("low-signal time trace contains invalid original time values")
    normalized = (raw - np.float32(0.05)) * np.float32(1.0 / (0.95 - 0.05))
    expected = np.float32(0.05) + np.float32(0.95 - 0.05) * normalized ** 2
    if not np.array_equal(actual, expected):
        raise ValueError("low-signal time trace differs from the independent float32 formula")
    expected_counts = {"changed_samples": sum(old != new for old, new in zip(raw_values, values, strict=True)),
                       "original_low_count": sum(value < 0.5 for value in raw_values), "low_count": sum(value < 0.5 for value in values),
                       "raw_sum": math.fsum(raw_values), "sum": math.fsum(values)}
    if any(row["low_signal_time_" + field] != value for field, value in expected_counts.items()):
        raise ValueError("low-signal time statistics differ from saved time values")
    for field, array in (("raw_sha256", raw), ("sha256", actual)):
        if row["low_signal_time_" + field] != hashlib.sha256(array.tobytes()).hexdigest():
            raise ValueError("low-signal time tensor hash differs from its values")
    for field in ("noise_sha256", "rng_after_sha256"):
        value = row["low_signal_time_" + field]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("low-signal time trace requires complete SHA256 values")


class LowSignalTimeActivity:
    def __init__(self, generator: Mapping[str, Any]):
        self.metadata = low_signal_time_metadata(generator)
        self.rows = []
        self.totals = {field: 0 for field in ("samples", "elements", "changed_samples", "original_low_count", "low_count")}
        self.totals.update({"raw_sum": 0.0, "sum": 0.0})

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1 or components.get("low_signal_time_executed") != 1.0:
            raise ValueError("low-signal time requires every consecutive completed update")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items() if name.startswith("low_signal_time_")})}
        validate_low_signal_row(row)
        self.rows.append(row)
        for field in self.totals:
            self.totals[field] += row["low_signal_time_" + ("batch_" + field if field in ("samples", "elements") else field)]
        return {"low_signal_time_total_steps": step, **{"low_signal_time_total_" + field: value for field, value in self.totals.items()}}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_student_gradients_or_parameter_trajectory_reconstructed": False, "extra_runtime_parameters_or_files": False}
