from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


UNIT_PROTOCOL = "physical_si_v1"


@dataclass(frozen=True, slots=True)
class ConditionField:
    quantity: str
    native_unit: str
    canonical_name: str
    canonical_unit: str


FIELDS = {
    "shaft_speed_rpm": ConditionField("shaft_frequency", "rpm", "shaft_speed_hz", "Hz"),
    "shaft_speed_hz": ConditionField("shaft_frequency", "Hz", "shaft_speed_hz", "Hz"),
    "drive_frequency_hz": ConditionField("drive_frequency", "Hz", "drive_frequency_hz", "Hz"),
    "lateral_load_kn": ConditionField("force", "kN", "lateral_load_n", "N"),
    "lateral_load_n": ConditionField("force", "N", "lateral_load_n", "N"),
    "radial_force_n": ConditionField("radial_force", "N", "radial_force_n", "N"),
    "shaft_torque_nm": ConditionField("torque", "Nm", "shaft_torque_nm", "Nm"),
    "motor_load_hp": ConditionField("power", "hp", "motor_load_w", "W"),
    "motor_load_w": ConditionField("power", "W", "motor_load_w", "W"),
    "load_hp": ConditionField("power", "hp", "load_power_w", "W"),
    "load_power_w": ConditionField("power", "W", "load_power_w", "W"),
}

UNIT_FACTORS = {
    "shaft_frequency": {"rpm": 1.0 / 60.0, "Hz": 1.0, "kHz": 1000.0},
    "drive_frequency": {"Hz": 1.0, "kHz": 1000.0},
    "force": {"N": 1.0, "kN": 1000.0},
    "radial_force": {"N": 1.0, "kN": 1000.0},
    "torque": {"Nm": 1.0, "Nmm": 0.001},
    "power": {"W": 1.0, "kW": 1000.0, "hp": 745.6998715822702},
}


def condition_field(name: str) -> ConditionField:
    try:
        return FIELDS[name]
    except KeyError as error:
        raise ValueError(f"physical quantity and unit are not declared for {name}") from error


def unit_factor(name: str, unit: str) -> float:
    field = condition_field(name)
    if not isinstance(unit, str):
        raise ValueError(f"condition unit must be text for {name}")
    aliases = {symbol.lower(): symbol for symbol in UNIT_FACTORS[field.quantity]}
    symbol = aliases.get(unit.strip().lower())
    if symbol is None:
        allowed = ", ".join(UNIT_FACTORS[field.quantity])
        raise ValueError(f"unit {unit!r} is incompatible with {name}; use {allowed}")
    return UNIT_FACTORS[field.quantity][symbol]


def _ordered_units(
    names: tuple[str, ...],
    units: Mapping[str, str] | Sequence[str] | None,
) -> tuple[str, ...]:
    if units is None:
        return tuple(condition_field(name).native_unit for name in names)
    if isinstance(units, Mapping):
        if set(units) != set(names):
            raise ValueError("condition unit fields must match the parameter fields exactly")
        return tuple(units[name] for name in names)
    if isinstance(units, str) or len(units) != len(names):
        raise ValueError("condition units must have one entry per parameter")
    return tuple(units)


def convert_condition_values(
    parameter_names: Sequence[str],
    values: Any,
    *,
    input_units: Mapping[str, str] | Sequence[str] | None = None,
    output_units: Mapping[str, str] | Sequence[str] | None = None,
) -> np.ndarray:
    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("condition parameter names must be nonempty and unique")
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError("condition parameter names differ from the declared fields")
        values = [values[name] for name in names]
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[-1] != len(names):
        raise ValueError("condition values must have shape [parameter] or [row,parameter]")
    if not np.isfinite(array).all():
        raise ValueError("condition parameters must be finite")
    source_units = _ordered_units(names, input_units)
    target_units = _ordered_units(names, output_units)
    factors = np.asarray([
        unit_factor(name, source) / unit_factor(name, target)
        for name, source, target in zip(names, source_units, target_units, strict=True)
    ])
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array * factors
    if not np.isfinite(converted).all():
        raise ValueError("condition unit conversion overflowed")
    return converted


def canonicalize_condition_values(
    parameter_names: Sequence[str],
    values: Any,
    *,
    input_units: Mapping[str, str] | Sequence[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    fields = tuple(condition_field(name) for name in parameter_names)
    canonical_names = tuple(field.canonical_name for field in fields)
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError("multiple input fields describe the same canonical condition")
    converted = convert_condition_values(
        parameter_names,
        values,
        input_units=input_units,
        output_units=tuple(field.canonical_unit for field in fields),
    )
    return canonical_names, converted


def model_condition_request(
    parameter_names: Sequence[str],
    values: Mapping[str, float],
    *,
    input_units: Mapping[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    names = tuple(parameter_names)
    native = convert_condition_values(names, values, input_units=input_units)
    canonical_names, canonical = canonicalize_condition_values(
        names, values, input_units=input_units,
    )
    units = _ordered_units(names, input_units)
    return dict(zip(names, native.tolist(), strict=True)), {
        "protocol": UNIT_PROTOCOL,
        "requested_parameters": {name: float(values[name]) for name in names},
        "requested_units": dict(zip(names, units, strict=True)),
        "canonical_parameters": dict(zip(canonical_names, canonical.tolist(), strict=True)),
        "canonical_units": {
            field.canonical_name: field.canonical_unit
            for field in (condition_field(name) for name in names)
        },
        "model_units": {name: condition_field(name).native_unit for name in names},
        "model_parameters": dict(zip(names, native.tolist(), strict=True)),
    }


def unit_control_metadata(name: str) -> dict[str, Any]:
    field = condition_field(name)
    return {
        "quantity_kind": field.quantity,
        "canonical_name": field.canonical_name,
        "canonical_unit": field.canonical_unit,
        "unit_options": [
            {
                "unit": symbol,
                "label": "hp (mechanical)" if symbol == "hp" else symbol,
                "factor_to_si": factor,
            }
            for symbol, factor in UNIT_FACTORS[field.quantity].items()
        ],
    }
