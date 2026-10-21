from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly

from .schema import Channel, SignalRecord


_SPEEDS = (1797, 1772, 1750, 1730)


@dataclass(frozen=True, slots=True)
class FileMetadata:
    file_id: int
    load_hp: int
    speed_rpm: int
    fault_mode: str | None
    diameter_in: float | None
    outer_position: str | None = None


def _rows(
    ids: tuple[int | None, int | None, int | None, int | None],
    mode: str,
    diameter: float,
    position: str | None = None,
) -> tuple[FileMetadata, ...]:
    return tuple(
        FileMetadata(file_id, load, _SPEEDS[load], mode, diameter, position)
        for load, file_id in enumerate(ids)
        if file_id is not None
    )


_NORMAL = tuple(
    FileMetadata(file_id, load, _SPEEDS[load], None, None)
    for load, file_id in enumerate((97, 98, 99, 100))
)
_DRIVE_12K = (
    _rows((105, 106, 107, 108), "inner_race", 0.007)
    + _rows((169, 170, 171, 172), "inner_race", 0.014)
    + _rows((209, 210, 211, 212), "inner_race", 0.021)
    + _rows((3001, 3002, 3003, 3004), "inner_race", 0.028)
    + _rows((118, 119, 120, 121), "rolling_element", 0.007)
    + _rows((185, 186, 187, 188), "rolling_element", 0.014)
    + _rows((222, 223, 224, 225), "rolling_element", 0.021)
    + _rows((3005, 3006, 3007, 3008), "rolling_element", 0.028)
    + _rows((130, 131, 132, 133), "outer_race", 0.007, "6:00")
    + _rows((197, 198, 199, 200), "outer_race", 0.014, "6:00")
    + _rows((234, 235, 236, 237), "outer_race", 0.021, "6:00")
    + _rows((144, 145, 146, 147), "outer_race", 0.007, "3:00")
    + _rows((246, 247, 248, 249), "outer_race", 0.021, "3:00")
    + _rows((156, 158, 159, 160), "outer_race", 0.007, "12:00")
    + _rows((258, 259, 260, 261), "outer_race", 0.021, "12:00")
)
_DRIVE_48K = (
    _rows((109, 110, 111, 112), "inner_race", 0.007)
    + _rows((174, 175, 176, 177), "inner_race", 0.014)
    + _rows((213, 214, 215, 217), "inner_race", 0.021)
    + _rows((122, 123, 124, 125), "rolling_element", 0.007)
    + _rows((189, 190, 191, 192), "rolling_element", 0.014)
    + _rows((226, 227, 228, 229), "rolling_element", 0.021)
    + _rows((135, 136, 137, 138), "outer_race", 0.007, "6:00")
    + _rows((201, 202, 203, 204), "outer_race", 0.014, "6:00")
    + _rows((238, 239, 240, 241), "outer_race", 0.021, "6:00")
    + _rows((148, 149, 150, 151), "outer_race", 0.007, "3:00")
    + _rows((250, 251, 252, 253), "outer_race", 0.021, "3:00")
    + _rows((161, 162, 163, 164), "outer_race", 0.007, "12:00")
    + _rows((262, 263, 264, 265), "outer_race", 0.021, "12:00")
)
_FAN_12K = (
    _rows((278, 279, 280, 281), "inner_race", 0.007)
    + _rows((274, 275, 276, 277), "inner_race", 0.014)
    + _rows((270, 271, 272, 273), "inner_race", 0.021)
    + _rows((282, 283, 284, 285), "rolling_element", 0.007)
    + _rows((286, 287, 288, 289), "rolling_element", 0.014)
    + _rows((290, 291, 292, 293), "rolling_element", 0.021)
    + _rows((294, 295, 296, 297), "outer_race", 0.007, "6:00")
    + _rows((313, None, None, None), "outer_race", 0.014, "6:00")
    + _rows((315, None, None, None), "outer_race", 0.021, "6:00")
    + _rows((298, 299, 300, 301), "outer_race", 0.007, "3:00")
    + _rows((310, 309, 311, 312), "outer_race", 0.014, "3:00")
    + _rows((None, 316, 317, 318), "outer_race", 0.021, "3:00")
    + _rows((302, 305, 306, 307), "outer_race", 0.007, "12:00")
)


@dataclass(frozen=True, slots=True)
class Group:
    sample_rate_hz: int
    bearing_location: str | None
    entries: tuple[FileMetadata, ...]


GROUPS = {
    "Normal Baseline Data": Group(48_000, None, _NORMAL),
    "12k Drive End Bearing Fault Data": Group(12_000, "drive_end", _DRIVE_12K),
    "48k Drive End Bearing Fault Data": Group(48_000, "drive_end", _DRIVE_48K),
    "12k Fan End Bearing Fault Data": Group(12_000, "fan_end", _FAN_12K),
}
EXPECTED_COUNTS = {
    "Normal Baseline Data": 4,
    "12k Drive End Bearing Fault Data": 60,
    "48k Drive End Bearing Fault Data": 52,
    "12k Fan End Bearing Fault Data": 45,
}


def _layout(path: Path) -> tuple[str, dict[str, tuple[int, ...]], tuple[str, ...]]:
    variables = {
        name: shape
        for name, shape, matlab_class in whosmat(path)
        if matlab_class in {"double", "single"}
    }
    prefix_pattern = re.compile(r"^(X\d+)_(?:DE|FE|BA)_time$")
    prefixes = {
        match.group(1)
        for name in variables
        if (match := prefix_pattern.fullmatch(name)) is not None
    }
    file_id = int(path.stem)
    matches = [prefix for prefix in prefixes if int(prefix[1:]) == file_id]
    if len(matches) == 1:
        prefix = matches[0]
    elif len(prefixes) == 1:
        prefix = next(iter(prefixes))
    else:
        raise ValueError(f"cannot choose acquisition in {path}: {sorted(prefixes)}")
    selected = {
        location: variables[key]
        for location, key in (
            ("drive_end", f"{prefix}_DE_time"),
            ("fan_end", f"{prefix}_FE_time"),
            ("base", f"{prefix}_BA_time"),
        )
        if key in variables
    }
    alternatives = tuple(sorted(item for item in prefixes if item != prefix))
    return prefix, selected, alternatives


class CWRUAdapter:
    adapter_id = "cwru"

    def __init__(self, root: Path, target_sample_rate_hz: int = 12_000) -> None:
        self.root = root
        self.target_sample_rate_hz = target_sample_rate_hz

    def discover(self) -> tuple[SignalRecord, ...]:
        records: list[SignalRecord] = []
        for group_name, group in GROUPS.items():
            if len(group.entries) != EXPECTED_COUNTS[group_name]:
                raise RuntimeError(f"invalid metadata count for {group_name}")
            metadata_by_id = {entry.file_id: entry for entry in group.entries}
            paths = sorted((self.root / group_name).glob("*.mat"), key=lambda p: int(p.stem))
            if len(paths) != EXPECTED_COUNTS[group_name]:
                raise ValueError(f"unexpected file count in {group_name}: {len(paths)}")
            for path in paths:
                file_id = int(path.stem)
                metadata = metadata_by_id.get(file_id)
                if metadata is None:
                    raise ValueError(f"file {file_id} is absent from the official CWRU table")
                prefix, available, alternatives = _layout(path)
                if "drive_end" not in available:
                    raise ValueError(f"drive-end channel is missing from {path}")
                source_key = f"{prefix}_DE_time"
                shape = available["drive_end"]
                source_length = max(shape)
                if min(shape) != 1:
                    raise ValueError(f"invalid vector shape in {path}:{source_key}: {shape}")
                if metadata.fault_mode is None:
                    fault_label = "normal"
                    fault_group = "normal"
                    fault_parameters: tuple[tuple[str, str | float], ...] = ()
                else:
                    diameter_mils = int(round(float(metadata.diameter_in) * 1000))
                    suffix = (
                        f"_{metadata.outer_position.replace(':', '')}"
                        if metadata.outer_position is not None
                        else ""
                    )
                    fault_label = (
                        f"{group.bearing_location}_{metadata.fault_mode}_"
                        f"{diameter_mils:03d}mil{suffix}"
                    )
                    fault_group = metadata.fault_mode
                    fault_parameters = (
                        ("bearing_location", str(group.bearing_location)),
                        ("fault_diameter_in", float(metadata.diameter_in)),
                        *(
                            (("outer_race_position", metadata.outer_position),)
                            if metadata.outer_position is not None
                            else ()
                        ),
                    )
                target_length = (
                    source_length
                    if group.sample_rate_hz == self.target_sample_rate_hz
                    else (source_length + 3) // 4
                )
                record_id = f"cwru/{group_name}/{file_id}"
                records.append(
                    SignalRecord(
                        adapter_id=self.adapter_id,
                        dataset_id="cwru",
                        device_id="cwru_motor_bearing_rig",
                        record_id=record_id,
                        acquisition_id=record_id,
                        path=path,
                        original_sample_rate_hz=group.sample_rate_hz,
                        sample_rate_hz=self.target_sample_rate_hz,
                        original_num_samples=source_length,
                        num_samples=target_length,
                        channels=(
                            Channel(
                                name="drive_end",
                                source_key=source_key,
                                location="drive_end",
                            ),
                        ),
                        condition_label=f"{metadata.speed_rpm}rpm_{metadata.load_hp}hp",
                        condition_parameters=(
                            ("shaft_speed_rpm", float(metadata.speed_rpm)),
                            ("motor_load_hp", float(metadata.load_hp)),
                        ),
                        fault_label=fault_label,
                        fault_group=fault_group,
                        fault_parameters=fault_parameters,
                        metadata={
                            "file_id": file_id,
                            "group": group_name,
                            "fault_bearing_location": group.bearing_location,
                            "available_channels": sorted(available),
                            "alternative_acquisition_prefixes": list(alternatives),
                        },
                    )
                )
        return tuple(records)

    def read_channels(self, record: SignalRecord) -> dict[str, np.ndarray]:
        keys = [channel.source_key for channel in record.channels]
        payload = loadmat(record.path, variable_names=keys)
        result: dict[str, np.ndarray] = {}
        for channel in record.channels:
            values = np.asarray(payload[channel.source_key], dtype=np.float32).reshape(-1)
            if record.original_sample_rate_hz != record.sample_rate_hz:
                values = resample_poly(values, up=1, down=4).astype(np.float32, copy=False)
            if values.size != record.num_samples:
                raise ValueError(f"unexpected resampled length in {record.path}: {values.size}")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in {record.path}:{channel.source_key}")
            result[channel.name] = values
        return result

    def read_channel(self, record: SignalRecord, channel: Channel) -> np.ndarray:
        return self.read_channels(record)[channel.name]
