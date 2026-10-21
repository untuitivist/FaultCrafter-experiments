from __future__ import annotations

import copy
import math

import torch

from faultcrafter.low_signal_time import low_signal_time_metadata


CONSTANT_GUIDANCE = "constant"
SYMMETRIC_TIME_GUIDANCE = "symmetric_time_v1"
MEAN_PRESERVING_TIME_GUIDANCE = "mean_preserving_time_v1"
TIME_GUIDANCE_SCHEDULES = frozenset({CONSTANT_GUIDANCE, SYMMETRIC_TIME_GUIDANCE, MEAN_PRESERVING_TIME_GUIDANCE})
PREFIX = "time_guidance_"
TRACE_FIELDS = ("times", "weights", "routes", "batch_samples", "changed_samples", "weight_sum", "weight_mean", "weight_minimum", "weight_maximum")


def time_guidance_policy(generator):
    policy = generator.get("fault_identity_guidance_schedule", CONSTANT_GUIDANCE)
    if policy not in TIME_GUIDANCE_SCHEDULES:
        raise ValueError("unsupported fault identity guidance schedule")
    return policy


def symmetric_guidance_weight(time, maximum_scale):
    if type(maximum_scale) not in (float, int) or not math.isfinite(maximum_scale) or maximum_scale < 1.0:
        raise ValueError("time guidance maximum scale must be finite and at least one")
    if time.ndim != 1 or not len(time) or time.dtype not in (torch.float32, torch.float64):
        raise ValueError("time guidance requires a nonempty float32 or float64 time vector")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("time guidance requires time inside the original unit interval")
    return (4.0 * time * (1.0 - time)) * (maximum_scale - 1.0)


def time_guidance_metadata(generator):
    policy = time_guidance_policy(generator)
    if policy == CONSTANT_GUIDANCE:
        raise ValueError("time guidance metadata requires an active time schedule")
    maximum = generator.get("fault_identity_guidance_scale")
    if maximum != 1.5 or generator.get("fault_identity_guidance_mode", "linear") != "linear":
        raise ValueError("experiment90 requires the original maximum1.5 and linear mixture")
    metadata = {"policy": policy, "maximum_guidance_scale": maximum,
            "residual_weight": "(maximum_guidance_scale-1)*4*t*(1-t)",
            "prediction": "conditional+residual_weight*(conditional-group)",
            "time_coordinate": "original_noise0_to_data1_without_terminal_remapping",
            "same_forward_in_training_source_validation_and_sampling": True,
            "group_reconstruction_and_both_gradient_paths_retained": True,
            "additional_parameters_or_buffers": 0, "additional_model_forward_backward_optimizer_calls": 0,
            "new_device_specific_runtime_parameters_or_files": False}
    if policy == MEAN_PRESERVING_TIME_GUIDANCE:
        low_signal_time_metadata(generator)
        metadata.pop("maximum_guidance_scale")
        metadata.update({"reference_guidance_scale": maximum,
                         "residual_weight": "(reference_guidance_scale-1)*4*t*(1-t)*(500/311)",
                         "analytic_training_shape_expectation": 311.0 / 500.0,
                         "theoretical_training_mean_residual_weight": 0.5,
                         "maximum_residual_weight": 250.0 / 311.0,
                         "mean_preservation_scope": "continuous_quadratic_training_distribution_only",
                         "finite_batch_device_gradient_validation_or_solver_mean_preserved": False,
                         "normalizer_fitted_from_samples_or_scores": False,
                         "additional_scalar_tensor_multiplications_per_forward": 1})
    return metadata


def scheduled_guidance_weight(time, reference_scale, policy):
    if policy not in (SYMMETRIC_TIME_GUIDANCE, MEAN_PRESERVING_TIME_GUIDANCE):
        raise ValueError("scheduled guidance requires a declared nonconstant policy")
    weight = symmetric_guidance_weight(time, reference_scale)
    return weight * (500.0 / 311.0) if policy == MEAN_PRESERVING_TIME_GUIDANCE else weight


def time_guidance_diagnostics(time, weights, routes, *, active):
    if type(active) is not bool:
        raise ValueError("time guidance activity must be a boolean")
    if not active:
        return {}
    if weights is None or weights.shape != time.shape or routes.shape != time.shape:
        raise ValueError("time guidance requires the actual model weights and aligned routes")
    values = weights.detach().cpu().tolist()
    count = len(values)
    total = math.fsum(values)
    return {PREFIX + "times": time.detach().cpu().tolist(), PREFIX + "weights": values,
            PREFIX + "routes": routes.detach().cpu().tolist(), PREFIX + "batch_samples": count,
            PREFIX + "changed_samples": sum(value != 0.5 for value in values), PREFIX + "weight_sum": total,
            PREFIX + "weight_mean": total / count, PREFIX + "weight_minimum": min(values), PREFIX + "weight_maximum": max(values)}


def validate_time_guidance_row(row, policy=SYMMETRIC_TIME_GUIDANCE):
    if policy not in (SYMMETRIC_TIME_GUIDANCE, MEAN_PRESERVING_TIME_GUIDANCE):
        raise ValueError("time guidance trace has an undeclared policy")
    factor = 500.0 / 311.0 if policy == MEAN_PRESERVING_TIME_GUIDANCE else 1.0
    times, weights, routes = (row[PREFIX + name] for name in ("times", "weights", "routes"))
    count = row[PREFIX + "batch_samples"]
    if type(count) is not int or count <= 0 or any(len(values) != count for values in (times, weights, routes)):
        raise ValueError("time guidance trace has inconsistent sample counts")
    if any(type(route) is not int or route < 0 for route in routes):
        raise ValueError("time guidance routes must be nonnegative integers")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in times):
        raise ValueError("time guidance trace has invalid times")
    if any(not math.isfinite(value) or not 0 <= value <= (0.5 * factor) for value in weights):
        raise ValueError("time guidance trace has invalid weights")
    if any(abs(observed - 2.0 * time * (1.0 - time) * factor) > 4 * torch.finfo(torch.float32).eps for time, observed in zip(times, weights, strict=True)):
        raise ValueError("time guidance model weights differ from the declared schedule")
    total = math.fsum(weights)
    expected = {"changed_samples": sum(value != 0.5 for value in weights), "weight_sum": total,
                "weight_mean": total / count, "weight_minimum": min(weights), "weight_maximum": max(weights)}
    if any(row[PREFIX + name] != value for name, value in expected.items()):
        raise ValueError("time guidance trace summaries disagree with actual weights")


class TimeGuidanceActivity:
    def __init__(self, generator):
        self.metadata = time_guidance_metadata(generator)
        self.rows = []
        self.totals = {"samples": 0, "changed_samples": 0, "weight_sum": 0.0}

    def record(self, components, step):
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("time guidance trace requires consecutive persistent updates")
        row = {PREFIX + name: copy.deepcopy(components[PREFIX + name]) for name in TRACE_FIELDS}
        validate_time_guidance_row(row, self.metadata["policy"])
        self.rows.append({"step": step, **row})
        self.totals["samples"] += row[PREFIX + "batch_samples"]
        self.totals["changed_samples"] += row[PREFIX + "changed_samples"]
        self.totals["weight_sum"] += row[PREFIX + "weight_sum"]
        return {PREFIX + "total_" + name: value for name, value in self.totals.items()}

    def audit(self):
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows),
                "totals": dict(self.totals), "steps": copy.deepcopy(self.rows)}
