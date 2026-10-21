from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data import WindowCatalog


_PARAMETER_UNITS = {
    "drive_frequency_hz": "Hz",
    "lateral_load_kn": "kN",
    "shaft_speed_rpm": "rpm",
    "motor_load_hp": "hp",
}


@dataclass(frozen=True, slots=True)
class ConditionParameterDefinition:
    name: str
    unit: str


@dataclass(frozen=True, slots=True)
class FaultDefinition:
    id: str
    group: str


@dataclass(frozen=True, slots=True)
class ModelDataContract:
    """Fixed measurement route and controllable labels for one generator."""

    dataset_id: str
    device_id: str
    sensor_location: str
    channel: str
    sample_rate_hz: int
    signal_quantity: str
    window_length: int
    condition_parameters: tuple[ConditionParameterDefinition, ...]
    faults: tuple[FaultDefinition, ...]

    @property
    def route_id(self) -> str:
        return ".".join(
            (self.dataset_id, self.device_id, self.sensor_location, self.channel)
        )

    @property
    def condition_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.condition_parameters)

    @property
    def fault_labels(self) -> tuple[str, ...]:
        return tuple(fault.id for fault in self.faults)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "dataset_id": self.dataset_id,
            "device_id": self.device_id,
            "sensor_location": self.sensor_location,
            "channel": self.channel,
            "sample_rate_hz": self.sample_rate_hz,
            "signal_quantity": self.signal_quantity,
            "signal": {
                "dtype": "float32",
                "shape": [1, self.window_length],
            },
            "condition_parameters": [
                asdict(parameter) for parameter in self.condition_parameters
            ],
            "faults": [asdict(fault) for fault in self.faults],
        }


@dataclass(frozen=True, slots=True)
class SignalRestorationProfile:
    """Non-signal deployment metadata for one measurement route."""

    route_id: str
    dataset_id: str
    device_id: str
    sensor_location: str
    channel: str
    signal_quantity: str
    physical_unit: str
    sample_rate_hz: int
    amplitude_method: str = "parent_long_signal_max_abs"
    time_coordinate: str = "t = sample_index / sample_rate_hz"
    angle_coordinate: str = "theta = integral(2*pi*shaft_speed_hz dt)"

    def __post_init__(self) -> None:
        if not self.physical_unit.strip():
            raise ValueError("physical unit must be explicitly declared")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")
        if self.amplitude_method not in {
            "parent_long_signal_max_abs",
            "parent_long_signal_rms",
            "parent_long_signal_mean_ac_rms",
        }:
            raise ValueError("unsupported amplitude restoration method")

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def signal_restoration_profile(
    contract: ModelDataContract,
    *,
    physical_unit: str,
    amplitude_method: str = "parent_long_signal_max_abs",
) -> SignalRestorationProfile:
    return SignalRestorationProfile(
        route_id=contract.route_id,
        dataset_id=contract.dataset_id,
        device_id=contract.device_id,
        sensor_location=contract.sensor_location,
        channel=contract.channel,
        signal_quantity=contract.signal_quantity,
        physical_unit=physical_unit,
        sample_rate_hz=contract.sample_rate_hz,
        amplitude_method=amplitude_method,
    )


def model_data_contract(catalog: WindowCatalog) -> ModelDataContract:
    if not catalog.records:
        raise ValueError("a model data contract requires at least one record")

    dataset_ids = {record.dataset_id for record in catalog.records}
    device_ids = {record.device_id for record in catalog.records}
    sample_rates = {record.sample_rate_hz for record in catalog.records}
    parameter_schemas = {
        tuple(name for name, _ in record.condition_parameters)
        for record in catalog.records
    }
    channel_facts = {
        (channel.name, channel.location, channel.quantity)
        for record in catalog.records
        for channel in record.channels
    }
    for name, values in (
        ("dataset", dataset_ids),
        ("device", device_ids),
        ("sample rate", sample_rates),
        ("condition parameter schema", parameter_schemas),
        ("measurement channel", channel_facts),
    ):
        if len(values) != 1:
            raise ValueError(f"one generator requires one fixed {name}, got {values}")

    parameter_names = next(iter(parameter_schemas))
    missing_units = [name for name in parameter_names if name not in _PARAMETER_UNITS]
    if missing_units:
        raise ValueError(f"condition parameters have no declared physical units: {missing_units}")

    fault_groups: dict[str, str] = {}
    for record in catalog.records:
        previous = fault_groups.setdefault(record.fault_label, record.fault_group)
        if previous != record.fault_group:
            raise ValueError(
                f"fault {record.fault_label} maps to both {previous} and "
                f"{record.fault_group}"
            )
    channel, sensor_location, signal_quantity = next(iter(channel_facts))
    return ModelDataContract(
        dataset_id=next(iter(dataset_ids)),
        device_id=next(iter(device_ids)),
        sensor_location=sensor_location,
        channel=channel,
        sample_rate_hz=next(iter(sample_rates)),
        signal_quantity=signal_quantity,
        window_length=catalog.window_length,
        condition_parameters=tuple(
            ConditionParameterDefinition(name=name, unit=_PARAMETER_UNITS[name])
            for name in parameter_names
        ),
        faults=tuple(
            FaultDefinition(id=label, group=fault_groups[label])
            for label in sorted(fault_groups)
        ),
    )
