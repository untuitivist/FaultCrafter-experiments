from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from .schema import Channel, SignalRecord


HITSM_SAMPLE_RATE_HZ = 51_200
_RIGS = {
    "SpectraQuest MFS dataset": "hit_sm_spectraquest_mfs",
    "Self-built dataset": "hit_sm_self_built",
}
_PATTERN = re.compile(r"^(Normal|(?:IR|OR)[258])_(600|900|1200)$")


def _signal_layout(path: Path) -> tuple[str, int]:
    candidates = [
        (name, max(shape))
        for name, shape, matlab_class in whosmat(path)
        if matlab_class in {"double", "single"}
        and len(shape) == 2
        and min(shape) == 1
        and max(shape) > 1
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one vector in {path}, found {candidates}")
    return candidates[0]


class HITSMAdapter:
    adapter_id = "hit_sm"

    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self) -> tuple[SignalRecord, ...]:
        records: list[SignalRecord] = []
        for folder, device_id in _RIGS.items():
            paths = sorted((self.root / folder).glob("*.mat"))
            if len(paths) != 21:
                raise ValueError(f"expected 21 HIT-SM files in {folder}, found {len(paths)}")
            for path in paths:
                match = _PATTERN.fullmatch(path.stem)
                if match is None:
                    raise ValueError(f"unsupported HIT-SM filename: {path.name}")
                source_label, speed = match.groups()
                source_key, length = _signal_layout(path)
                if source_label == "Normal":
                    fault_label = "normal"
                    fault_group = "normal"
                    fault_parameters: tuple[tuple[str, str | float], ...] = ()
                else:
                    fault_group = {
                        "IR": "inner_race",
                        "OR": "outer_race",
                    }[source_label[:2]]
                    extent = float(source_label[2:])
                    fault_label = f"{fault_group}_{int(extent)}deg"
                    fault_parameters = (("fault_angular_extent_degree", extent),)
                record_id = f"hit_sm/{device_id}/{path.stem}"
                records.append(
                    SignalRecord(
                        adapter_id=self.adapter_id,
                        dataset_id="hit_sm",
                        device_id=device_id,
                        record_id=record_id,
                        acquisition_id=record_id,
                        path=path,
                        original_sample_rate_hz=HITSM_SAMPLE_RATE_HZ,
                        sample_rate_hz=HITSM_SAMPLE_RATE_HZ,
                        original_num_samples=length,
                        num_samples=length,
                        channels=(
                            Channel(
                                name="vibration",
                                source_key=source_key,
                                location="bearing_housing_vertical",
                            ),
                        ),
                        condition_label=f"{speed}rpm",
                        condition_parameters=(("shaft_speed_rpm", float(speed)),),
                        fault_label=fault_label,
                        fault_group=fault_group,
                        fault_parameters=fault_parameters,
                        metadata={"source_label": source_label},
                    )
                )
        return tuple(records)

    def read_channels(self, record: SignalRecord) -> dict[str, np.ndarray]:
        keys = [channel.source_key for channel in record.channels]
        payload = loadmat(record.path, variable_names=keys)
        result: dict[str, np.ndarray] = {}
        for channel in record.channels:
            values = np.asarray(payload[channel.source_key], dtype=np.float32).reshape(-1)
            if values.size != record.num_samples:
                raise ValueError(f"unexpected signal length in {record.path}: {values.size}")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in {record.path}:{channel.source_key}")
            result[channel.name] = values
        return result

    def read_channel(self, record: SignalRecord, channel: Channel) -> np.ndarray:
        return self.read_channels(record)[channel.name]
