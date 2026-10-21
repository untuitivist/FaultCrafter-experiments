from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


POST_WARMUP_UNIFORM = "post_warmup_uniform_v1"


def trajectory_average_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("trajectory_average_policy", "ema")
    if not isinstance(policy, str) or policy not in {"ema", POST_WARMUP_UNIFORM}:
        raise ValueError("unknown trajectory averaging policy")
    if policy == "ema":
        return policy
    defaults = {"initial_parameter_anchor_policy": "none", "initial_prediction_anchor_policy": "none",
                "shared_gradient_policy": "joint", "sharpness_training_policy": "standard",
                "direct_x_time_weighting": "inverse_noise_squared", "normalized_energy_policy": "none"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("trajectory averaging cannot combine undeclared training objectives")
    if any(generator.get(name, False) for name in ("resumable_training", "preserve_frozen_ema", "initial_validation_fallback",
           "device_balanced_sampling", "relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference")):
        raise ValueError("trajectory averaging requires original full-model sampling without recovery or fallback")
    if generator.get("active_lora_device_index") is not None or any(float(generator.get(name, 0)) != 0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")):
        raise ValueError("trajectory averaging requires full-model training without teachers or contrast")
    warmup = generator.get("warmup_steps")
    interval = generator.get("evaluation_interval")
    preheat = generator.get("checkpoint_preheat_evaluations")
    if any(type(value) is not int or value <= 0 for value in (warmup, interval, preheat)) or interval * preheat <= warmup:
        raise ValueError("first selectable checkpoint must follow the positive integer warmup")
    return policy


class PostWarmupAverage:
    def __init__(self, average: nn.Module, model: nn.Module, warmup_steps: int, ema_decay: float):
        if type(warmup_steps) is not int or warmup_steps <= 0 or not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("invalid trajectory averaging warmup or EMA decay")
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.completed_steps = 0
        self.averaged_updates = 0
        self.tensor_count = len(list(model.parameters()))
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self._validate(average, model)

    def _validate(self, average: nn.Module, model: nn.Module) -> None:
        if average is model or any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError("trajectory averaging requires an independent model without BatchNorm calibration")
        average_parameters = dict(average.named_parameters())
        parameters = dict(model.named_parameters())
        if not parameters or average_parameters.keys() != parameters.keys():
            raise ValueError("trajectory averaging parameter names differ")
        for name, parameter in parameters.items():
            target = average_parameters[name]
            if not parameter.requires_grad or target.requires_grad or not parameter.is_floating_point():
                raise ValueError("every student parameter must remain trainable and every average parameter frozen")
            if parameter.shape != target.shape or parameter.dtype != target.dtype or parameter.device != target.device or parameter.data_ptr() == target.data_ptr():
                raise ValueError("trajectory averaging parameter schema or storage mismatch")
        targets = dict(average.named_buffers())
        buffers = dict(model.named_buffers())
        if targets.keys() != buffers.keys() or any(buffer.shape != targets[name].shape or buffer.dtype != targets[name].dtype or buffer.device != targets[name].device for name, buffer in buffers.items()):
            raise ValueError("trajectory averaging buffer schema differs")

    @torch.no_grad()
    def update(self, average: nn.Module, model: nn.Module, step: int) -> dict[str, int]:
        if type(step) is not int or step != self.completed_steps + 1:
            raise ValueError("trajectory averaging requires consecutive post-optimizer updates")
        self._validate(average, model)
        parameters = dict(model.named_parameters())
        count = max(0, step - self.warmup_steps)
        for name, target in average.named_parameters():
            parameter = parameters[name]
            if not count:
                target.mul_(self.ema_decay).add_(parameter, alpha=1 - self.ema_decay)
            elif count == 1:
                target.copy_(parameter)
            else:
                target.add_(parameter - target, alpha=1 / count)
        buffers = dict(model.named_buffers())
        for name, target in average.named_buffers():
            target.copy_(buffers[name])
        self.completed_steps = step
        self.averaged_updates = count
        return {"trajectory_average_completed_steps": step, "trajectory_average_updates": count,
                "trajectory_average_start_step": self.warmup_steps + 1}

    def metadata(self) -> dict[str, Any]:
        return {"policy": POST_WARMUP_UNIFORM, "warmup_steps": self.warmup_steps, "warmup_ema_decay": self.ema_decay,
                "start_step": self.warmup_steps + 1, "completed_steps": self.completed_steps,
                "averaged_updates": self.averaged_updates, "tensor_count": self.tensor_count,
                "parameter_count": self.parameter_count, "formula": "average_plus_student_minus_average_over_count",
                "scope": "all_shared_lora_factors_and_fine_coarse_identity_parameters",
                "buffers": "copy_current_student_without_averaging", "source_validation": "original_objective_on_average",
                "learning_rate_scheduler": "original_rule_on_average_validation", "additional_model_copies": 0,
                "extra_runtime_parameters_or_files": False}
