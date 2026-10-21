from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from faultcrafter.style_transport import infer_condition_geometry
from faultcrafter.typed_conditions import encode_typed_conditions, physical_slots

if TYPE_CHECKING:
    from faultcrafter.unified_generator import UnifiedSignalGenerator


@dataclass(frozen=True)
class TypedConditionProjectors:
    projectors: torch.Tensor
    continuous_presence: torch.Tensor


def typed_condition_projectors(devices, config, compute_device) -> TypedConditionProjectors:
    projectors = np.zeros((len(devices), 8, 8), dtype=np.float32)
    presence = np.zeros((len(devices), 8), dtype=np.float32)
    for device in devices:
        slots = list(physical_slots(device.parameter_names))
        values = encode_typed_conditions(device.parameter_names, device.corpus.parameter_matrix)
        local = source_weak_direction_projectors(
            [values[:, slots]], condition_dimension=len(slots),
            retained_variance=float(config["generator"].get("condition_retained_variance", 0.99)),
        )[0].numpy()
        projectors[device.index][np.ix_(slots, slots)] = local
        presence[device.index, slots] = 1.0
    return TypedConditionProjectors(
        torch.from_numpy(projectors).to(compute_device),
        torch.from_numpy(presence).to(compute_device),
    )


def source_weak_direction_projectors(
    source_parameters: Sequence[np.ndarray],
    *,
    condition_dimension: int,
    retained_variance: float = 0.99,
) -> torch.Tensor:
    geometry = infer_condition_geometry(
        source_parameters,
        condition_dimension=condition_dimension,
        retained_variance=retained_variance,
    )
    projectors = []
    for basis, rank in zip(geometry.basis, geometry.rank, strict=True):
        weak_basis = basis[int(rank):]
        projectors.append(weak_basis.T @ weak_basis)
    return torch.from_numpy(np.stack(projectors).astype(np.float32))


def condition_slope_penalty(
    model: UnifiedSignalGenerator,
    weak_projectors: torch.Tensor | TypedConditionProjectors,
    *,
    device_index: int | None = None,
) -> torch.Tensor:
    if model.style_mechanism != "legacy_affine":
        raise ValueError("compact slope regularization requires the affine style field")
    presence = None
    if isinstance(weak_projectors, TypedConditionProjectors):
        presence = weak_projectors.continuous_presence
        weak_projectors = weak_projectors.projectors
    if weak_projectors.shape != (
        model.num_devices, model.condition_dimension, model.condition_dimension
    ):
        raise ValueError("source weak projector dimensions differ")
    query = model.style_field.query
    effective = query.shared.weight[None] + (
        query.lora_up @ query.lora_down
    ) / float(query.rank)
    decoder = (
        model.style_field.values.T @ model.style_field.keys
    ) / float(model.style_field.keys.shape[1]) ** 0.5
    jacobian = decoder[None] @ effective[:, :, 1:]
    if presence is not None:
        jacobian = jacobian * presence.to(jacobian)[:, None, :]
    weak_response = jacobian @ weak_projectors.to(jacobian)
    reference = jacobian.detach().square().sum(dim=(1, 2)).clamp_min(1e-8)
    penalties = weak_response.square().sum(dim=(1, 2)) / reference
    if device_index is not None:
        if not 0 <= device_index < model.num_devices:
            raise ValueError("condition regularization device is outside the model")
        return penalties[device_index]
    return penalties.mean()
