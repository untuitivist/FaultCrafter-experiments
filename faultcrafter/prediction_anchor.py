from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn

from faultcrafter.direct_x import direct_x_loss
from faultcrafter.initial_anchor import parameter_fingerprint


INITIAL_PREDICTION_ANCHOR = "initial_prediction_v1"
PREDICTION_ANCHOR_WEIGHT = 0.1
INITIAL_OUTPUT_EPS_MULTIPLIER = 16


def prediction_anchor_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("initial_prediction_anchor_policy", "none")
    if not isinstance(policy, str) or policy not in {"none", INITIAL_PREDICTION_ANCHOR}:
        raise ValueError("unknown initial prediction anchor policy")
    if policy == "none":
        return policy
    defaults = {"initial_parameter_anchor_policy": "none", "shared_gradient_policy": "joint",
                "sharpness_training_policy": "standard", "direct_x_time_weighting": "inverse_noise_squared",
                "normalized_energy_policy": "none"}
    if any(generator.get(name, value) != value for name, value in defaults.items()):
        raise ValueError("prediction anchor cannot combine parameter anchoring or undeclared objective policies")
    if any(generator.get(name, False) for name in ("relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference", "resumable_training")):
        raise ValueError("prediction anchor requires the original sampling and guidance without recovery")
    if any(float(generator.get(name, 0.0)) != 0.0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")) or generator.get("active_lora_device_index") is not None:
        raise ValueError("prediction anchor requires full-student training without other teachers")
    return policy


def prediction_anchor_loss(prediction: torch.Tensor, reference: torch.Tensor, time: torch.Tensor, minimum_noise_fraction: float):
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.dtype != reference.dtype or prediction.device != reference.device:
        raise ValueError("prediction anchor requires matching normalized waveform tensors")
    return direct_x_loss(prediction, reference.detach(), time, minimum_noise_fraction=minimum_noise_fraction)


class InitialPredictionAnchor:
    def __init__(self, model: nn.Module, minimum_noise_fraction: float):
        parameters = dict(model.named_parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters.values()):
            raise ValueError("prediction anchor requires all student parameters to remain trainable")
        if not 0.0 < minimum_noise_fraction <= 1.0:
            raise ValueError("prediction anchor noise floor must be in (0,1]")
        self.reference = copy.deepcopy(model).eval().requires_grad_(False)
        self.reference.zero_grad(set_to_none=True)
        self.reference_sha256 = parameter_fingerprint(self.reference.state_dict())
        self.minimum_noise_fraction = float(minimum_noise_fraction)
        self.parameter_count = sum(parameter.numel() for parameter in parameters.values())
        self.parameter_tensor_count = len(parameters)
        self.reference_dtype_epsilon = torch.finfo(next(iter(parameters.values())).dtype).eps
        self.total_steps = 0
        self.nonzero_steps = 0
        self.first_loss = None
        self.first_output_error = None
        self.first_output_scale = None
        self.first_output_bound = None

    def metadata(self) -> dict[str, Any]:
        return {"policy": INITIAL_PREDICTION_ANCHOR, "weight": PREDICTION_ANCHOR_WEIGHT,
                "reference_origin": "model_at_train_entry_after_initialization", "reference_sha256": self.reference_sha256,
                "reference_parameter_count": self.parameter_count, "reference_tensor_count": self.parameter_tensor_count,
                "reference_dtype_epsilon": self.reference_dtype_epsilon,
                "initial_output_eps_multiplier": INITIAL_OUTPUT_EPS_MULTIPLIER, "initial_tolerance_does_not_clip_loss": True,
                "scope": "guided_normalized_signal_only", "minimum_noise_fraction": self.minimum_noise_fraction,
                "time_weighting": "original_inverse_noise_squared", "source_validation": "original_data_objective_without_anchor",
                "reference_forward_passes_per_update": 1, "student_backward_passes_per_update": 1,
                "same_source_inputs_and_no_new_noise_time": True, "reference_saved_in_checkpoint": False,
                "extra_runtime_parameters_or_files": False}

    def loss(self, prediction: torch.Tensor, time: torch.Tensor, model_inputs: tuple) -> tuple[torch.Tensor, dict]:
        if self.reference.training or any(parameter.requires_grad for parameter in self.reference.parameters()):
            raise RuntimeError("prediction reference must remain frozen and in evaluation mode")
        if len(model_inputs) != 9 or not torch.equal(model_inputs[1], time):
            raise ValueError("prediction reference requires the exact nine student inputs and time")
        with torch.no_grad():
            reference = self.reference(*model_inputs).normalized_signal
        raw_loss, clean_mse = prediction_anchor_loss(prediction, reference, time, self.minimum_noise_fraction)
        raw_value = float(raw_loss.detach())
        if self.first_loss is None:
            self.first_output_error = float((prediction.detach() - reference).abs().max())
            self.first_output_scale = max(1.0, float(prediction.detach().abs().max()), float(reference.abs().max()))
            self.first_output_bound = INITIAL_OUTPUT_EPS_MULTIPLIER * torch.finfo(prediction.dtype).eps * self.first_output_scale
            if self.first_output_error > self.first_output_bound:
                raise RuntimeError("initial student/reference forward mismatch exceeds the declared dtype bound")
            self.first_loss = raw_value
        self.total_steps += 1
        self.nonzero_steps += int(raw_value > 0)
        penalty = PREDICTION_ANCHOR_WEIGHT * raw_loss
        return penalty, {"prediction_anchor_raw_loss": raw_value,
                         "prediction_anchor_clean_mse": float(clean_mse.detach()),
                         "prediction_anchor_penalty": float(penalty.detach()),
                         "prediction_anchor_initial_loss": self.first_loss,
                         "prediction_anchor_initial_output_error": self.first_output_error,
                         "prediction_anchor_initial_output_scale": self.first_output_scale,
                         "prediction_anchor_initial_output_bound": self.first_output_bound,
                         "prediction_anchor_total_steps": self.total_steps,
                         "prediction_anchor_nonzero_steps": self.nonzero_steps,
                         "prediction_anchor_reference_forwards": self.total_steps}

    def verify_reference_unchanged(self) -> None:
        if self.reference.training or any(parameter.requires_grad or parameter.grad is not None for parameter in self.reference.parameters()):
            raise RuntimeError("prediction reference trainability or gradients changed")
        if parameter_fingerprint(self.reference.state_dict()) != self.reference_sha256:
            raise RuntimeError("initial prediction reference weights changed during training")
