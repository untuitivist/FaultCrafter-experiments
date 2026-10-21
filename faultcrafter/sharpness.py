from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from faultcrafter.time_weighting import INVERSE_NOISE_SQUARED


STANDARD_BACKWARD = "standard"
SAM_L2 = "sam_l2"
SAM_RADIUS = 0.05


def sharpness_training_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("sharpness_training_policy", STANDARD_BACKWARD)
    if not isinstance(policy, str) or policy not in {STANDARD_BACKWARD, SAM_L2}:
        raise ValueError("unknown sharpness training policy")
    if policy == SAM_L2:
        if generator.get("shared_gradient_policy", "joint") != "joint" or generator.get("direct_x_time_weighting", INVERSE_NOISE_SQUARED) != INVERSE_NOISE_SQUARED:
            raise ValueError("sharpness training cannot combine gradient projection or time weighting")
        if any(generator.get(name, False) for name in ("relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference")):
            raise ValueError("sharpness training cannot combine undeclared sampling or guidance gradients")
        if any(float(generator.get(name, 0.0)) != 0.0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")) or generator.get("active_lora_device_index") is not None:
            raise ValueError("sharpness training requires full-model training without teachers or contrast")
    return policy


def sharpness_metadata() -> dict[str, Any]:
    return {"policy": SAM_L2, "radius": SAM_RADIUS, "norm": "global_l2",
            "scope": "all_trainable_shared_lora_and_identity_parameters", "batch_randomness": "same_batch_replayed_one_draw_two_passes",
            "gradient_clipping": "second_gradient_before_original_adamw", "source_validation": "original_unperturbed_objective",
            "restore_before_optimizer_ema_validation_export": True, "runtime_parameters_added": False}


@dataclass
class RandomSnapshot:
    explicit: torch.Tensor
    cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]
    python: tuple
    numpy: tuple

    @classmethod
    def capture(cls, generator: torch.Generator, cuda_devices: list[int]) -> RandomSnapshot:
        return cls(generator.get_state().clone(), torch.random.get_rng_state().clone(),
                   {index: torch.cuda.get_rng_state(index).clone() for index in cuda_devices}, random.getstate(), np.random.get_state())

    def restore(self, generator: torch.Generator) -> None:
        generator.set_state(self.explicit)
        torch.random.set_rng_state(self.cpu)
        for index, state in self.cuda.items():
            torch.cuda.set_rng_state(state, index)
        random.setstate(self.python)
        np.random.set_state(self.numpy)

    def equivalent(self, other: RandomSnapshot) -> bool:
        return (torch.equal(self.explicit, other.explicit) and torch.equal(self.cpu, other.cpu) and self.cuda.keys() == other.cuda.keys()
                and all(torch.equal(state, other.cuda[index]) for index, state in self.cuda.items()) and self.python == other.python
                and self.numpy[0] == other.numpy[0] and np.array_equal(self.numpy[1], other.numpy[1]) and self.numpy[2:] == other.numpy[2:])


def scaled_global_norm(values: list[torch.Tensor]) -> tuple[float, float]:
    if not values or not torch.stack([torch.isfinite(value).all() for value in values]).all():
        raise ValueError("sharpness gradients must be present and finite")
    maximum = float(torch.stack([value.detach().abs().max() for value in values]).max())
    if maximum == 0.0:
        return 0.0, 0.0
    scaled = float(torch.stack([(value.detach().to(torch.float64) / maximum).square().sum() for value in values]).sum().sqrt())
    if not math.isfinite(maximum * scaled):
        raise ValueError("sharpness gradient norm is not finite")
    return maximum, scaled


def tensors_unchanged(current: tuple[torch.Tensor, ...], original: tuple[torch.Tensor, ...]) -> bool:
    return not current or bool(torch.stack([(value == saved).all() for value, saved in zip(current, original, strict=True)]).all())


def backward_with_sharpness(
    model: nn.Module,
    objective_function: Callable[[], tuple[torch.Tensor, dict[str, float]]],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    parameters = tuple(model.parameters())
    if not parameters or not model.training or not torch.is_grad_enabled() or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("sharpness backward requires a trainable full model in training mode")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1 or any(not parameter.is_floating_point() or parameter.layout != torch.strided for parameter in parameters):
        raise ValueError("sharpness backward requires dense floating parameters on one compute device")
    cuda_devices = sorted(device.index for device in devices if device.type == "cuda")
    original = tuple(parameter.detach().clone() for parameter in parameters)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    before = RandomSnapshot.capture(generator, cuda_devices)
    restore_random = before
    model.zero_grad(set_to_none=True)
    try:
        objective, components = objective_function()
        if objective.ndim != 0 or not torch.isfinite(objective):
            raise ValueError("sharpness objective must be a finite scalar")
        objective.backward()
        objective = objective.detach()
        restore_random = RandomSnapshot.capture(generator, cuda_devices)
        if not tensors_unchanged(parameters, original):
            raise RuntimeError("sharpness objective unexpectedly mutated model parameters")
        current_buffers = dict(model.named_buffers())
        if buffers.keys() != current_buffers.keys() or not tensors_unchanged(tuple(current_buffers.values()), tuple(buffers.values())):
            raise RuntimeError("sharpness backward does not support forward-mutated buffers")
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        maximum, scaled_norm = scaled_global_norm(gradients)
        with torch.no_grad():
            if maximum > 0.0:
                for parameter in parameters:
                    if parameter.grad is not None:
                        direction = parameter.grad.detach().to(torch.float64) / maximum / scaled_norm * SAM_RADIUS
                        parameter.add_(direction.to(parameter.dtype))
            displacements = [parameter.detach().to(torch.float64) - saved.to(torch.float64) for parameter, saved in zip(parameters, original, strict=True)]
            displacement_maximum, displacement_scaled = scaled_global_norm(displacements)
            changed = int(torch.stack([(parameter != saved).any() for parameter, saved in zip(parameters, original, strict=True)]).sum())
        model.zero_grad(set_to_none=True)
        before.restore(generator)
        perturbed, _ = objective_function()
        if perturbed.ndim != 0 or not torch.isfinite(perturbed):
            raise ValueError("sharpness perturbed objective must be a finite scalar")
        perturbed.backward()
        scaled_global_norm([parameter.grad for parameter in parameters if parameter.grad is not None])
        if not restore_random.equivalent(RandomSnapshot.capture(generator, cuda_devices)):
            raise RuntimeError("sharpness second pass consumed a different random stream")
        current_buffers = dict(model.named_buffers())
        if buffers.keys() != current_buffers.keys() or not tensors_unchanged(tuple(current_buffers.values()), tuple(buffers.values())):
            raise RuntimeError("sharpness second pass unexpectedly mutated model buffers")
        components = {**components, "sharpness_policy_executed": 1.0, "sharpness_random_stream_replayed": 1.0,
                      "sharpness_original_gradient_norm": maximum * scaled_norm,
                      "sharpness_actual_perturbation_norm": displacement_maximum * displacement_scaled,
                      "sharpness_perturbed_parameter_tensors": changed, "sharpness_perturbed_objective": float(perturbed.detach()),
                      "sharpness_objective_gap": float(perturbed.detach() - objective)}
        return objective, components
    except BaseException:
        model.zero_grad(set_to_none=True)
        raise
    finally:
        with torch.no_grad():
            for parameter, saved in zip(parameters, original, strict=True):
                parameter.copy_(saved)
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name])
        restore_random.restore(generator)
