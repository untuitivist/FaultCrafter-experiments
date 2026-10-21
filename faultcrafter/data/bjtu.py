from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .schema import Channel, SignalRecord


BJTU_SAMPLE_RATE_HZ = 64_000
BJTU_NUM_SAMPLES = 640_000
BJTU_BEARING_STATES = {
    "M0_G0_LA0_RA0": ("normal", "normal"),
    "M3_G0_LA0_RA0": ("motor_bearing", "bearing_fault_unspecified"),
    "M0_G5_LA0_RA0": ("gearbox_bearing_inner_race", "inner_race"),
    "M0_G6_LA0_RA0": ("gearbox_bearing_outer_race", "outer_race"),
    "M0_G7_LA0_RA0": ("gearbox_bearing_rolling_element", "rolling_element"),
    "M0_G8_LA0_RA0": ("gearbox_bearing_cage", "cage"),
    "M0_G0_LA1_RA0": ("left_axlebox_bearing_inner_race", "inner_race"),
    "M0_G0_LA2_RA0": ("left_axlebox_bearing_outer_race", "outer_race"),
    "M0_G0_LA3_RA0": ("left_axlebox_bearing_rolling_element", "rolling_element"),
    "M0_G0_LA4_RA0": ("left_axlebox_bearing_cage", "cage"),
}
BJTU_LEFT_ATOMS = {
    "LA1": ("left_axlebox_bearing_inner_race", "inner_race"),
    "LA2": ("left_axlebox_bearing_outer_race", "outer_race"),
    "LA3": ("left_axlebox_bearing_rolling_element", "rolling_element"),
    "LA4": ("left_axlebox_bearing_cage", "cage"),
}
BJTU_SOURCE_COMPOSITIONS = ("LA1+LA2", "LA2+LA3", "LA2+LA4")
BJTU_TARGET_COMPOSITIONS = ("LA1+LA2+LA3", "LA1+LA2+LA4", "LA1+LA2+LA3+LA4")
BJTU_COMPOUND_COMPONENTS = {
    "left_axlebox_compound_" + code.replace("+", "_"): tuple(BJTU_LEFT_ATOMS[atom] for atom in code.split("+"))
    for code in BJTU_SOURCE_COMPOSITIONS + BJTU_TARGET_COMPOSITIONS
}
BJTU_COMPOUND_STATES = {
    f"M0_G0_{code}_RA0": ("left_axlebox_compound_" + code.replace("+", "_"), "compound")
    for code in BJTU_SOURCE_COMPOSITIONS + BJTU_TARGET_COMPOSITIONS
}
BJTU_ACCELERATION_CHANNELS = (
    *(f"CH{index}" for index in range(1, 7)),
    *(f"CH{index}" for index in range(11, 20)),
    *(f"CH{index}" for index in range(21, 24)),
)
_CHANNELS_BY_SENSOR = {
    "motor": tuple(f"CH{index}" for index in range(1, 7)),
    "gearbox": tuple(f"CH{index}" for index in range(11, 17)),
    "leftaxlebox": tuple(f"CH{index}" for index in range(17, 20)),
    "rightaxlebox": tuple(f"CH{index}" for index in range(21, 24)),
}
_FILE_PATTERN = re.compile(
    r"^data_(motor|gearbox|leftaxlebox|rightaxlebox)_.+?_(20|40|60)Hz_([+-]?10|0)kN\.csv$",
    re.IGNORECASE,
)


class BJTUAdapter:
    adapter_id = "bjtu_rao"

    def __init__(self, root: Path, *, include_left_compounds: bool = False) -> None:
        self.root = root
        self.include_left_compounds = include_left_compounds

    def _data_root(self) -> Path:
        candidates = [
            path
            for path in self.root.rglob("M0_G0_LA0_RA0")
            if (path / "Sample_1").is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one BJTU data root, found {candidates}")
        return candidates[0].parent

    def discover(self) -> tuple[SignalRecord, ...]:
        data_root = self._data_root()
        records: list[SignalRecord] = []
        states = dict(BJTU_BEARING_STATES)
        if self.include_left_compounds:
            states.update(BJTU_COMPOUND_STATES)
        for state, (fault_label, fault_group) in states.items():
            state_root = data_root / state
            if not state_root.is_dir():
                raise FileNotFoundError(state_root)
            for sample_root in sorted(
                state_root.glob("Sample_*"), key=lambda path: int(path.name.split("_")[-1])
            ):
                sample_number = int(sample_root.name.split("_")[-1])
                for path in sorted(sample_root.glob("*.csv")):
                    match = _FILE_PATTERN.fullmatch(path.name)
                    if match is None:
                        raise ValueError(f"unsupported BJTU filename: {path.name}")
                    sensor, frequency, load = match.groups()
                    sensor = sensor.lower()
                    channels = tuple(
                        Channel(
                            name=name,
                            source_key=name,
                            location=sensor,
                            quantity="acceleration_g",
                        )
                        for name in _CHANNELS_BY_SENSOR[sensor]
                    )
                    acquisition_id = f"bjtu_rao/{state}/sample_{sample_number}"
                    records.append(
                        SignalRecord(
                            adapter_id=self.adapter_id,
                            dataset_id="bjtu_rao",
                            device_id="bjtu_rao_bogie",
                            record_id=f"{acquisition_id}/{sensor}",
                            acquisition_id=acquisition_id,
                            path=path,
                            original_sample_rate_hz=BJTU_SAMPLE_RATE_HZ,
                            sample_rate_hz=BJTU_SAMPLE_RATE_HZ,
                            original_num_samples=BJTU_NUM_SAMPLES,
                            num_samples=BJTU_NUM_SAMPLES,
                            channels=channels,
                            condition_label=f"{frequency}Hz_{int(load):+d}kN",
                            condition_parameters=(
                                ("drive_frequency_hz", float(frequency)),
                                ("lateral_load_kn", float(load)),
                            ),
                            fault_label=fault_label,
                            fault_group=fault_group,
                            metadata={
                                "state_code": state,
                                "sample_number": sample_number,
                                "sensor_group": sensor,
                            },
                        )
                    )
        if len(records) != len(states) * 9 * 4:
            raise ValueError(f"expected {len(states) * 36} BJTU records, found {len(records)}")
        return tuple(records)

    def read_channels(self, record: SignalRecord) -> dict[str, np.ndarray]:
        keys = [channel.source_key for channel in record.channels]
        with record.path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = [value.strip() for value in next(csv.reader(stream))]
        if len(header) != len(set(header)):
            raise ValueError(f"duplicate CSV columns in {record.path}")
        missing = sorted(set(keys) - set(header))
        if missing:
            raise ValueError(f"missing CSV columns in {record.path}: {missing}")
        column_indices = tuple(header.index(key) for key in keys)
        matrix = np.loadtxt(
            record.path,
            delimiter=",",
            dtype=np.float32,
            skiprows=1,
            usecols=column_indices,
            ndmin=2,
        )
        if matrix.shape != (record.original_num_samples, len(keys)):
            raise ValueError(
                f"{record.path} has numeric shape {matrix.shape}, "
                f"expected {(record.original_num_samples, len(keys))}"
            )
        result: dict[str, np.ndarray] = {}
        for column_index, channel in enumerate(record.channels):
            values = np.ascontiguousarray(matrix[:, column_index])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in {record.path}:{channel.source_key}")
            result[channel.name] = values
        return result

    def read_channel(self, record: SignalRecord, channel: Channel) -> np.ndarray:
        return self.read_channels(record)[channel.name]
