from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from faultcrafter.condition_units import canonicalize_condition_values, condition_field


FIXED_SCALING_PROTOCOL = "physical_fixed_scale_v1"
FIXED_PHYSICAL_SCALES = MappingProxyType({
    "shaft_frequency": 100.0,
    "drive_frequency": 100.0,
    "force": 10000.0,
    "power": 1000.0,
})


@dataclass(frozen=True, slots=True)
class FixedScaleConditions:
    canonical_names: tuple[str, ...]
    canonical_units: tuple[str, ...]
    canonical_values: np.ndarray
    model_coordinates: np.ndarray
    protocol: ClassVar[str] = FIXED_SCALING_PROTOCOL


def prepare_fixed_scale_conditions(
    parameter_names: Sequence[str],
    values: Any,
    *,
    input_units: Mapping[str, str] | Sequence[str] | None = None,
    condition_dimension: int | None = None,
) -> FixedScaleConditions:
    names = tuple(parameter_names)
    canonical_names, canonical_values = canonicalize_condition_values(
        names, values, input_units=input_units,
    )
    dimension = len(names) if condition_dimension is None else condition_dimension
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < len(names)
    ):
        raise ValueError("condition dimension must be an integer covering all physical fields")
    fields = tuple(condition_field(name) for name in names)
    scales = np.asarray([
        FIXED_PHYSICAL_SCALES[field.quantity] for field in fields
    ], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (canonical_values / scales).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("fixed-scale condition coordinates must be finite in float32")
    coordinates = np.zeros((*scaled.shape[:-1], dimension), dtype=np.float32)
    coordinates[..., :len(names)] = scaled
    return FixedScaleConditions(
        canonical_names=canonical_names,
        canonical_units=tuple(field.canonical_unit for field in fields),
        canonical_values=canonical_values,
        model_coordinates=coordinates,
    )
