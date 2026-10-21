from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping

import torch
from torch import nn

from faultcrafter.compact_inference import is_device_lora
from faultcrafter.time_weighting import resolve_time_weighting


JOINT_GRADIENT = "joint"
WEIGHTED_SHARED_PCGRAD = "weighted_pcgrad"
WITHIN_DEVICE_CONDITION_PCGRAD = "within_device_condition_pcgrad_v1"
PRIVATE_IDENTITY_PREFIXES = ("fault_embedding.", "fault_group_embedding.")
UNPARTITIONED_LOSS_WEIGHTS = (
    "fault_contrast_loss_weight", "source_semantic_loss_weight", "location_loss_weight",
    "style_global_loss_weight", "style_local_loss_weight", "style_relation_loss_weight", "content_loss_weight",
    "condition_style_transport_loss_weight", "condition_slope_transport_loss_weight", "fault_content_transport_loss_weight",
    "difference_of_differences_loss_weight", "fault_separation_margin_loss_weight",
)


def shared_gradient_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("shared_gradient_policy", JOINT_GRADIENT)
    if not isinstance(policy, str) or policy not in {JOINT_GRADIENT, WEIGHTED_SHARED_PCGRAD}:
        raise ValueError("unknown shared gradient policy")
    return policy


def validate_partition_recipe(model: nn.Module, generator: Mapping[str, Any], normalization_method: str, *, within_device_conditions: bool = False, for_shared_update: bool = False, for_episode: bool = False) -> None:
    if for_episode:
        from faultcrafter.episode_adaptation import episode_adaptation_metadata
        if not within_device_conditions or for_shared_update:
            raise ValueError("source episode requires only its typed condition partition")
        episode_adaptation_metadata(generator)
    elif for_shared_update:
        if within_device_conditions or generator.get("shared_update_policy") != "source_cone_adamw_v1" or shared_gradient_policy(generator) != JOINT_GRADIENT:
            raise ValueError("shared update partition requires its native-AdamW post-update policy without raw gradient projection")
    elif within_device_conditions:
        if generator.get("condition_gradient_policy") != WITHIN_DEVICE_CONDITION_PCGRAD or shared_gradient_policy(generator) != JOINT_GRADIENT:
            raise ValueError("condition partition requires within-device projection without cross-device projection")
    elif shared_gradient_policy(generator) != WEIGHTED_SHARED_PCGRAD:
        raise ValueError("device objective partition requires weighted_pcgrad")
    resolve_time_weighting(generator)
    if normalization_method != "max_abs" or model.affine_restoration or model.style_mechanism != "legacy_affine":
        raise ValueError("shared gradient partition requires max_abs non-affine legacy attention")
    if any(float(generator.get(name, 0.0)) != 0.0 for name in UNPARTITIONED_LOSS_WEIGHTS):
        raise ValueError("shared gradient partition does not support undeclared auxiliary losses")
    if generator.get("relation_coupled_noise_time", False) or generator.get("relation_balanced_sampling", False):
        raise ValueError("shared gradient partition cannot change relation sampling")
    if generator.get("detach_group_guidance_reference", False) or generator.get("active_lora_device_index") is not None:
        raise ValueError("shared gradient partition requires full joint source training")
    if not model.training or not torch.is_grad_enabled() or any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("shared gradient partition requires trainable full-model training")


@dataclass
class DeviceObjectivePartition:
    device_losses: dict[int, torch.Tensor] = field(default_factory=dict)
    regularizer: torch.Tensor | None = None
    reconstruction_error: float = 0.0
    sample_count: int = 0
    eligible_order_sample_count: int = 0

    def collect(
        self,
        device_ids: torch.Tensor,
        sample_losses: torch.Tensor,
        order_numerators: dict[int, torch.Tensor],
        *,
        eligible_order_samples: int,
        order_weight: float,
        regularizer: torch.Tensor,
        objective: torch.Tensor,
        extra_device_losses: dict[int, torch.Tensor] | None = None,
    ) -> None:
        if self.device_losses or self.regularizer is not None:
            raise ValueError("a device objective partition may only collect one batch")
        if device_ids.ndim != 1 or device_ids.dtype != torch.long or sample_losses.shape != device_ids.shape or not len(device_ids):
            raise ValueError("device partition requires one loss and integer device id per sample")
        if device_ids.device != sample_losses.device or not sample_losses.is_floating_point() or torch.any(device_ids < 0):
            raise ValueError("device partition losses and routes are invalid")
        if type(eligible_order_samples) is not int or not 0 <= eligible_order_samples <= len(device_ids):
            raise ValueError("order partition denominator must be the eligible sample count")
        if bool(order_numerators) != bool(eligible_order_samples) or not math.isfinite(order_weight) or order_weight < 0:
            raise ValueError("order partition numerator, denominator or weight is invalid")
        indices = torch.unique(device_ids, sorted=True).tolist()
        extra_device_losses = extra_device_losses or {}
        if set(extra_device_losses) - set(indices) or any(value.ndim != 0 or not torch.isfinite(value) for value in extra_device_losses.values()):
            raise ValueError("additional source device losses must be finite scalars for present routes")
        if set(order_numerators) - set(indices):
            raise ValueError("order partition contains a device absent from the batch")
        self.sample_count = len(sample_losses)
        self.eligible_order_sample_count = eligible_order_samples
        for index in indices:
            contribution = sample_losses[device_ids == index].sum() / self.sample_count
            if index in order_numerators:
                contribution = contribution + order_weight * order_numerators[index] / eligible_order_samples
            if index in extra_device_losses:
                contribution = contribution + extra_device_losses[index]
            self.device_losses[index] = contribution
        self.regularizer = regularizer
        reconstructed = torch.stack([*self.device_losses.values(), regularizer]).sum()
        if objective.ndim != 0 or not torch.isfinite(reconstructed) or not torch.isfinite(objective):
            raise ValueError("partitioned source objective must be a finite scalar")
        self.reconstruction_error = float((reconstructed.detach() - objective.detach()).abs())
        tolerance = 64 * torch.finfo(objective.dtype).eps * max(1.0, float(objective.detach().abs()), float(reconstructed.detach().abs()))
        if self.reconstruction_error > tolerance:
            raise ValueError("device contributions changed the original weighted source objective")


def shared_gradient_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple((name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad and not is_device_lora(name) and not name.startswith(PRIVATE_IDENTITY_PREFIXES))


def project_weighted_task_gradients(
    gradients: torch.Tensor, *, seed: int, step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if gradients.ndim != 2 or min(gradients.shape) < 1 or not gradients.is_floating_point() or not torch.isfinite(gradients).all():
        raise ValueError("shared task gradients must be a finite floating matrix")
    if type(seed) is not int or seed < 0 or type(step) is not int or step < 1:
        raise ValueError("shared projection requires a nonnegative training seed and positive optimizer step")
    task_count = len(gradients)
    references = gradients.detach().to(torch.float64)
    norms = references.square().sum(dim=1)
    negative_pairs = sum(int(torch.dot(references[first], references[second]) < 0)
                         for first in range(task_count) for second in range(first + 1, task_count))
    projected = gradients.detach().clone()
    random_order = random.Random((seed << 32) + step)
    projection_count = 0
    for task_index in range(task_count):
        order = list(range(task_count))
        random_order.shuffle(order)
        for reference_index in order:
            if reference_index == task_index or norms[reference_index] == 0:
                continue
            inner_product = torch.dot(projected[task_index].to(torch.float64), references[reference_index])
            if inner_product < 0:
                coefficient = inner_product / norms[reference_index]
                projected[task_index] = (projected[task_index].to(torch.float64) - coefficient * references[reference_index]).to(projected)
                projection_count += 1
    correction = projected.sum(dim=0) - gradients.detach().sum(dim=0)
    if not torch.isfinite(projected).all() or not torch.isfinite(correction).all():
        raise ValueError("shared gradient projection produced a nonfinite correction")
    diagnostics = {
        "shared_gradient_device_count": task_count,
        "shared_gradient_original_negative_pairs": negative_pairs,
        "shared_gradient_projection_count": projection_count,
        "shared_gradient_correction_norm": float(correction.to(torch.float64).norm()),
        "shared_gradient_original_sum_norm": float(gradients.detach().sum(dim=0).to(torch.float64).norm()),
        "shared_gradient_projected_sum_norm": float(projected.sum(dim=0).to(torch.float64).norm()),
    }
    return correction, diagnostics


def apply_gradient_correction(parameters: tuple[nn.Parameter, ...], correction: torch.Tensor) -> None:
    if correction.ndim != 1 or sum(parameter.numel() for parameter in parameters) != len(correction) or not torch.isfinite(correction).all():
        raise ValueError("shared gradient correction has invalid coordinates or values")
    increments = []
    offset = 0
    for parameter in parameters:
        if correction.device != parameter.device or correction.dtype != parameter.dtype:
            raise ValueError("shared gradient correction and parameters need matching device and dtype")
        increment = correction[offset:offset + parameter.numel()].reshape_as(parameter)
        if parameter.grad is None and torch.any(increment != 0):
            raise ValueError("projection attempted to activate an unused shared parameter")
        increments.append(increment)
        offset += parameter.numel()
    with torch.no_grad():
        for parameter, increment in zip(parameters, increments, strict=True):
            if parameter.grad is not None:
                parameter.grad.add_(increment)


def apply_shared_gradient_correction(parameters: tuple[nn.Parameter, ...], correction: torch.Tensor) -> None:
    apply_gradient_correction(parameters, correction)


def backward_with_shared_projection(
    model: nn.Module,
    objective: torch.Tensor,
    partition: DeviceObjectivePartition,
    *,
    seed: int,
    step: int,
) -> dict[str, float]:
    if not partition.device_losses or partition.regularizer is None:
        raise ValueError("shared projection requires the actual populated source objective partition")
    if type(seed) is not int or seed < 0 or type(step) is not int or step < 1:
        raise ValueError("shared projection requires a nonnegative training seed and positive optimizer step")
    named_parameters = shared_gradient_parameters(model)
    if not named_parameters:
        raise ValueError("shared projection has no shared trainable parameters")
    diagnostics = {"shared_gradient_policy_executed": 1.0,
                   "shared_gradient_partition_error": partition.reconstruction_error,
                   "shared_gradient_sample_count": partition.sample_count,
                   "shared_gradient_order_sample_count": partition.eligible_order_sample_count}
    if len(partition.device_losses) == 1:
        objective.backward()
        return {**diagnostics, "shared_gradient_device_count": 1, "shared_gradient_original_negative_pairs": 0,
                "shared_gradient_projection_count": 0, "shared_gradient_correction_norm": 0.0}
    parameters = tuple(parameter for _, parameter in named_parameters)
    flattened = []
    for device_index in sorted(partition.device_losses):
        gradients = torch.autograd.grad(partition.device_losses[device_index], parameters, retain_graph=True, allow_unused=True)
        flattened.append(torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1)
                                    for parameter, gradient in zip(parameters, gradients, strict=True)]))
    correction, projected = project_weighted_task_gradients(torch.stack(flattened), seed=seed, step=step)
    objective.backward()
    if projected["shared_gradient_projection_count"]:
        apply_shared_gradient_correction(parameters, correction)
    return {**diagnostics, **projected}
