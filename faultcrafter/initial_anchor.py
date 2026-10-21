from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch
from torch import nn


INITIAL_L2_SP = "initial_l2_sp_v1"
INITIAL_ANCHOR_WEIGHT = 0.01


def initial_anchor_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("initial_parameter_anchor_policy", "none")
    if not isinstance(policy, str) or policy not in {"none", INITIAL_L2_SP}:
        raise ValueError("unknown initial parameter anchor policy")
    if policy == "none":
        return policy
    incompatible = {
        "shared_gradient_policy": "joint",
        "sharpness_training_policy": "standard",
        "direct_x_time_weighting": "inverse_noise_squared",
        "normalized_energy_policy": "none",
    }
    if any(generator.get(name, default) != default for name, default in incompatible.items()):
        raise ValueError("initial anchor requires the original single-backward data objective")
    if any(generator.get(name, False) for name in ("relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference", "resumable_training")):
        raise ValueError("initial anchor cannot combine undeclared sampling, guidance or recovery policies")
    if any(float(generator.get(name, 0.0)) != 0.0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")) or generator.get("active_lora_device_index") is not None:
        raise ValueError("initial anchor requires full-model training without teachers or contrast")
    return policy


def parameter_fingerprint(parameters: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in parameters.items():
        digest.update((name + "|" + str(value.dtype) + "|" + str(tuple(value.shape)) + "\n").encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class InitialParameterAnchor:
    def __init__(self, model: nn.Module):
        parameters = dict(model.named_parameters())
        if not parameters or any(not parameter.requires_grad or not parameter.is_floating_point() for parameter in parameters.values()):
            raise ValueError("initial anchor requires every model parameter to remain trainable and floating point")
        self.reference = {name: value.detach().clone() for name, value in parameters.items()}
        self.reference_sha256 = parameter_fingerprint(self.reference)
        self.parameter_count = sum(value.numel() for value in self.reference.values())
        self.total_steps = 0
        self.nonzero_steps = 0
        self.first_penalty = None

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": INITIAL_L2_SP, "weight": INITIAL_ANCHOR_WEIGHT,
            "formula": "half_weight_times_global_squared_parameter_distance",
            "reference_origin": "model_at_train_entry_after_initialization",
            "reference_sha256": self.reference_sha256,
            "reference_tensor_count": len(self.reference), "reference_parameter_count": self.parameter_count,
            "scope": "all_shared_lora_and_fine_coarse_identity_parameters",
            "source_validation": "original_data_objective_without_anchor",
            "backward_passes_per_update": 1, "reference_saved_in_checkpoint": False,
            "extra_runtime_parameters_or_files": False,
        }

    def verify_reference_unchanged(self) -> None:
        if parameter_fingerprint(self.reference) != self.reference_sha256:
            raise RuntimeError("initial parameter anchor reference changed during training")

    def squared_distance(self, model: nn.Module) -> torch.Tensor:
        parameters = dict(model.named_parameters())
        if tuple(parameters) != tuple(self.reference):
            raise ValueError("model parameter identities changed after anchor initialization")
        terms = []
        for name, parameter in parameters.items():
            reference = self.reference[name]
            if not parameter.requires_grad or parameter.shape != reference.shape or parameter.dtype != reference.dtype or parameter.device != reference.device:
                raise ValueError("initial anchor parameter trainability, shape, dtype or device changed")
            terms.append((parameter - reference).square().sum())
        return torch.stack(terms).sum()

    def apply(self, model: nn.Module, data_objective: torch.Tensor, components: dict) -> tuple[torch.Tensor, dict]:
        distance = self.squared_distance(model)
        penalty = 0.5 * INITIAL_ANCHOR_WEIGHT * distance
        objective = data_objective + penalty
        value = float(penalty.detach())
        if self.first_penalty is None:
            self.first_penalty = value
        self.total_steps += 1
        self.nonzero_steps += int(value > 0)
        return objective, {
            **components,
            "objective": float(objective.detach()),
            "parameter_anchor_data_objective": float(data_objective.detach()),
            "parameter_anchor_penalty": value,
            "parameter_anchor_distance_squared": float(distance.detach()),
            "parameter_anchor_initial_penalty": self.first_penalty,
            "parameter_anchor_total_steps": self.total_steps,
            "parameter_anchor_nonzero_steps": self.nonzero_steps,
        }
