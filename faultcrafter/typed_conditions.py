from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from faultcrafter.condition_units import canonicalize_condition_values, condition_field, model_condition_request
from faultcrafter.fixed_condition_scaling import FIXED_PHYSICAL_SCALES


TYPED_CONDITION_PROTOCOL = "physical_typed_mask_v1"
TYPED_QUANTITIES = ("shaft_frequency", "drive_frequency", "force", "power")
TYPED_FIELDS = ("shaft_speed_hz", "drive_frequency_hz", "lateral_load_n", "load_power_w")
TYPED_DIMENSION = 8


def physical_slots(parameter_names: Sequence[str]) -> tuple[int, ...]:
    slots = tuple(TYPED_QUANTITIES.index(condition_field(name).quantity) for name in parameter_names)
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("typed conditions require unique physical quantities")
    return slots


def encode_typed_conditions(
    parameter_names: Sequence[str], values: Any, *, input_units: Any = None,
) -> np.ndarray:
    names = tuple(parameter_names)
    slots = physical_slots(names)
    _, canonical = canonicalize_condition_values(names, values, input_units=input_units)
    coordinates = np.zeros((*canonical.shape[:-1], TYPED_DIMENSION), dtype=np.float32)
    scales = np.asarray([FIXED_PHYSICAL_SCALES[condition_field(name).quantity] for name in names])
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates[..., list(slots)] = canonical / scales
    coordinates[..., [slot + 4 for slot in slots]] = 1.0
    if not np.isfinite(coordinates).all():
        raise ValueError("typed coordinates must be finite in float32")
    return coordinates


def prepare_condition_request(parameter_names, values, *, input_units=None, model_protocol=None):
    parameters, audit = model_condition_request(parameter_names, values, input_units=input_units)
    if model_protocol == TYPED_CONDITION_PROTOCOL:
        encoded = encode_typed_conditions(parameter_names, parameters)
        labels = [f"{quantity}_scaled" for quantity in TYPED_QUANTITIES] + [f"{quantity}_present" for quantity in TYPED_QUANTITIES]
        audit["native_parameters"] = audit.pop("model_parameters")
        audit["native_units"] = audit.pop("model_units")
        audit["model_protocol"] = model_protocol
        audit["model_parameters"] = dict(zip(labels, encoded.tolist(), strict=True))
        audit["model_units"] = {label: "dimensionless" for label in labels}
    elif model_protocol is not None:
        raise ValueError("request model condition protocol is unsupported")
    return parameters, audit


def encode_model_conditions(device: Any, values: Any, condition_dimension: int) -> np.ndarray:
    protocol = getattr(device, "condition_protocol", None)
    if protocol == TYPED_CONDITION_PROTOCOL:
        if condition_dimension != TYPED_DIMENSION:
            raise ValueError("typed conditions require four values and four independent masks")
        return encode_typed_conditions(device.parameter_names, values)
    if protocol is not None:
        raise ValueError(f"unknown condition protocol: {protocol}")
    normalized = ((np.asarray(values) - device.parameter_mean) / device.parameter_standard_deviation).astype(np.float32)
    coordinates = np.zeros((*normalized.shape[:-1], condition_dimension), dtype=np.float32)
    coordinates[..., :normalized.shape[-1]] = normalized
    return coordinates
