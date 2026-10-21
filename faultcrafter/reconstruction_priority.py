from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch


JOINT_AUXILIARY_GRADIENTS = "joint"
RECONSTRUCTION_PRIORITY = "reconstruction_priority_v1"
DIAGNOSTIC_FIELDS = (
    "reconstruction_priority_executed", "reconstruction_priority_primary_objective",
    "reconstruction_priority_primary_norm_squared", "reconstruction_priority_auxiliary_norm_squared_before",
    "reconstruction_priority_auxiliary_norm_squared_after", "reconstruction_priority_dot_before",
    "reconstruction_priority_dot_after", "reconstruction_priority_coefficient",
    "reconstruction_priority_projected", "reconstruction_priority_nonzero_correction",
    "reconstruction_priority_correction_norm", "reconstruction_priority_geometry_tolerance",
    "reconstruction_priority_original_total_norm_squared", "reconstruction_priority_total_norm_squared_after",
)


def auxiliary_gradient_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("auxiliary_gradient_policy", JOINT_AUXILIARY_GRADIENTS)
    if policy not in (JOINT_AUXILIARY_GRADIENTS, RECONSTRUCTION_PRIORITY):
        raise ValueError("unknown auxiliary gradient policy")
    return policy


def gradient_vector(parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    if len(parameters) != len(gradients) or not parameters:
        raise ValueError("gradient vectors require aligned nonempty parameters")
    values = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype or gradient.device != parameter.device):
            raise ValueError("gradient shape, dtype or device differs from parameter")
        values.append((torch.zeros_like(parameter) if gradient is None else gradient.detach()).reshape(-1))
    return torch.cat(values).double()


@torch.no_grad()
def project_auxiliary_gradients(parameters: Sequence[torch.nn.Parameter], primary_gradients: Sequence[torch.Tensor | None]) -> dict:
    parameters = tuple(parameters)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("reconstruction priority requires all original parameters trainable")
    dtype, device = parameters[0].dtype, parameters[0].device
    if dtype not in (torch.float32, torch.float64) or any(parameter.dtype != dtype or parameter.device != device for parameter in parameters):
        raise ValueError("reconstruction priority requires uniform float32 or float64 parameters")
    primary = gradient_vector(parameters, primary_gradients)
    original = gradient_vector(parameters, [parameter.grad for parameter in parameters])
    if not torch.isfinite(primary).all() or not torch.isfinite(original).all():
        raise FloatingPointError("reconstruction priority requires finite original gradients")
    auxiliary = original - primary
    primary_squared = float(primary.square().sum())
    auxiliary_squared = float(auxiliary.square().sum())
    dot_before = float((primary * auxiliary).sum())
    coefficient = -dot_before / primary_squared if primary_squared > 0.0 and dot_before < 0.0 else 0.0
    if not all(math.isfinite(value) for value in (primary_squared, auxiliary_squared, dot_before, coefficient)):
        raise FloatingPointError("reconstruction priority produced nonfinite geometry")
    if coefficient > 0.0:
        corrected = (original + coefficient * primary).to(dtype=dtype)
        if not torch.isfinite(corrected).all():
            raise FloatingPointError("reconstruction priority correction is not representable")
        offset = 0
        for parameter, primary_gradient in zip(parameters, primary_gradients, strict=True):
            value = corrected[offset:offset + parameter.numel()].reshape(parameter.shape)
            offset += parameter.numel()
            if primary_gradient is None:
                continue
            if parameter.grad is None:
                if torch.any(value != 0):
                    parameter.grad = torch.zeros_like(parameter)
                    parameter.grad.copy_(value)
            else:
                parameter.grad.copy_(value)
    updated = gradient_vector(parameters, [parameter.grad for parameter in parameters])
    updated_auxiliary = updated - primary
    dot_after = float((primary * updated_auxiliary).sum())
    tolerance = 16.0 * torch.finfo(dtype).eps * math.sqrt(primary_squared) * (math.sqrt(primary_squared) + math.sqrt(auxiliary_squared))
    if not math.isfinite(dot_after) or dot_after < -tolerance:
        raise FloatingPointError("reconstruction priority violated its declared geometry bound")
    correction_norm = float((updated - original).square().sum().sqrt())
    return {"reconstruction_priority_executed": 1.0,
            "reconstruction_priority_primary_norm_squared": primary_squared,
            "reconstruction_priority_auxiliary_norm_squared_before": auxiliary_squared,
            "reconstruction_priority_auxiliary_norm_squared_after": float(updated_auxiliary.square().sum()),
            "reconstruction_priority_dot_before": dot_before, "reconstruction_priority_dot_after": dot_after,
            "reconstruction_priority_coefficient": coefficient, "reconstruction_priority_projected": float(coefficient > 0.0),
            "reconstruction_priority_nonzero_correction": float(correction_norm > 0.0),
            "reconstruction_priority_correction_norm": correction_norm,
            "reconstruction_priority_geometry_tolerance": tolerance,
            "reconstruction_priority_original_total_norm_squared": float(original.square().sum()),
            "reconstruction_priority_total_norm_squared_after": float(updated.square().sum())}


class ReconstructionPriority:
    def __init__(self):
        self.primary = None
        self.consumed = False

    def capture(self, waveform: torch.Tensor, group_waveform: torch.Tensor, group_weight: float) -> None:
        if self.primary is not None or self.consumed or waveform.ndim != 0 or group_waveform.ndim != 0:
            raise ValueError("priority capture requires one scalar fine/group objective per batch")
        if not math.isfinite(group_weight) or group_weight < 0.0:
            raise ValueError("priority group weight must be finite and nonnegative")
        self.primary = waveform + group_weight * group_waveform
        if not self.primary.requires_grad:
            raise ValueError("priority capture requires a differentiable primary objective")

    def backward(self, model: torch.nn.Module, objective: torch.Tensor) -> dict:
        if self.primary is None or self.consumed or objective.ndim != 0:
            raise ValueError("priority backward requires one captured unconsumed objective")
        parameters = tuple(model.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("priority backward requires the full trainable model")
        primary_value = float(self.primary.detach())
        objective.backward(retain_graph=True)
        primary_gradients = torch.autograd.grad(self.primary, parameters, allow_unused=True)
        self.primary = None
        self.consumed = True
        return {**project_auxiliary_gradients(parameters, primary_gradients), "reconstruction_priority_primary_objective": primary_value}


class ReconstructionPriorityActivity:
    def __init__(self, model: torch.nn.Module, generator: Mapping[str, Any]):
        if auxiliary_gradient_policy(generator) != RECONSTRUCTION_PRIORITY:
            raise ValueError("priority activity requires an explicit active policy")
        parameters = tuple(model.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("priority activity requires all original parameters trainable")
        self.metadata_value = {"policy": RECONSTRUCTION_PRIORITY, "primary": "fine_waveform_plus_original_weighted_group_waveform",
            "group_reconstruction_loss_weight": float(generator["group_reconstruction_loss_weight"]),
            "parameter_tensors": len(parameters), "parameter_elements": sum(parameter.numel() for parameter in parameters),
            "parameter_dtype": str(parameters[0].dtype),
            "scope": "joint_all_original_trainable_parameters_not_device_reweighting",
            "geometry_dtype": "float64", "corrected_gradient_computation": "float64_then_single_cast_to_original_dtype",
            "geometry_tolerance_multiplier": 16.0, "original_total_backward_before_primary_autograd": True,
            "logged_and_validation_objective_changed": False, "global_clip_adamw_ema_recipe_changed": False,
            "scalar_trace_saved": True, "historical_gradient_vectors_saved": False, "student_parameter_trajectory_saved": False,
            "adamw_actual_loss_descent_guaranteed": False, "cnn_non_degradation_guaranteed": False,
            "extra_runtime_parameters_or_files": False}
        self.rows = []
        self.projected = 0
        self.nonzero = 0
        self.zero_primary = 0
        self.coefficient_sum = 0.0
        self.coefficient_maximum = 0.0

    def metadata(self) -> dict:
        return copy.deepcopy(self.metadata_value)

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1 or components.get("reconstruction_priority_executed") != 1.0:
            raise ValueError("priority activity requires every consecutive completed update")
        row = {"step": step, **{name: components[name] for name in DIAGNOSTIC_FIELDS}}
        if any(not math.isfinite(value) for name, value in row.items() if name != "step"):
            raise ValueError("priority activity contains nonfinite diagnostics")
        for field in ("reconstruction_priority_projected", "reconstruction_priority_nonzero_correction"):
            if row[field] not in (0.0, 1.0):
                raise ValueError("priority activity flags must be binary")
        if row["reconstruction_priority_nonzero_correction"] > row["reconstruction_priority_projected"]:
            raise ValueError("priority correction occurred without declared projection")
        self.rows.append(row)
        self.projected += int(row["reconstruction_priority_projected"])
        self.nonzero += int(row["reconstruction_priority_nonzero_correction"])
        self.zero_primary += int(row["reconstruction_priority_primary_norm_squared"] == 0.0)
        self.coefficient_sum += row["reconstruction_priority_coefficient"]
        self.coefficient_maximum = max(self.coefficient_maximum, row["reconstruction_priority_coefficient"])
        return {"reconstruction_priority_total_steps": step, "reconstruction_priority_projected_steps": self.projected,
                "reconstruction_priority_nonzero_steps": self.nonzero, "reconstruction_priority_zero_primary_steps": self.zero_primary,
                "reconstruction_priority_coefficient_sum": self.coefficient_sum,
                "reconstruction_priority_maximum_coefficient": self.coefficient_maximum}

    def audit(self) -> dict:
        return {"metadata": self.metadata(), "completed_steps": len(self.rows), "steps": [dict(row) for row in self.rows],
                "historical_gradient_vectors_independently_reconstructed": False, "extra_runtime_parameters_or_files": False}
