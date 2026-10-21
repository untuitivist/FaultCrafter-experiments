from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .schema import Channel, SignalRecord, UnifiedSample


class Adapter(Protocol):
    adapter_id: str

    def discover(self) -> tuple[SignalRecord, ...]: ...

    def read_channels(self, record: SignalRecord) -> dict[str, np.ndarray]: ...

    def read_channel(self, record: SignalRecord, channel: Channel) -> np.ndarray: ...


class AdapterCollection:
    def __init__(self, adapters: tuple[Adapter, ...]) -> None:
        self.adapters = {adapter.adapter_id: adapter for adapter in adapters}
        if len(self.adapters) != len(adapters):
            raise ValueError("adapter IDs must be unique")

    def discover(self) -> tuple[SignalRecord, ...]:
        return tuple(record for adapter in self.adapters.values() for record in adapter.discover())

    def read_channel(self, record: SignalRecord, channel: Channel) -> np.ndarray:
        return self.adapters[record.adapter_id].read_channel(record, channel)

    def read_channels(self, record: SignalRecord) -> dict[str, np.ndarray]:
        return self.adapters[record.adapter_id].read_channels(record)


@dataclass(frozen=True, slots=True)
class WindowSource:
    record_index: int
    channel_index: int
    window_count: int


class WindowCatalog:
    def __init__(
        self,
        records: tuple[SignalRecord, ...],
        adapters: AdapterCollection,
        *,
        window_length: int = 1024,
        stride: int = 819,
    ) -> None:
        if window_length <= 0 or stride <= 0:
            raise ValueError("window length and stride must be positive")
        self.records = records
        self.adapters = adapters
        self.window_length = window_length
        self.stride = stride
        sources: list[WindowSource] = []
        ends: list[int] = []
        total = 0
        for record_index, record in enumerate(records):
            count = self.window_count(record.num_samples)
            if count == 0:
                continue
            for channel_index in range(len(record.channels)):
                sources.append(WindowSource(record_index, channel_index, count))
                total += count
                ends.append(total)
        self.sources = tuple(sources)
        self.ends = tuple(ends)

    def window_count(self, num_samples: int) -> int:
        if num_samples < self.window_length:
            return 0
        return (num_samples - self.window_length) // self.stride + 1

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def locate(self, index: int) -> tuple[WindowSource, int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect_right(self.ends, index)
        source = self.sources[source_index]
        previous_end = self.ends[source_index - 1] if source_index else 0
        local_index = index - previous_end
        return source, local_index, local_index * self.stride

    def read(self, index: int) -> UnifiedSample:
        source, _, start = self.locate(index)
        record = self.records[source.record_index]
        channel = record.channels[source.channel_index]
        values = self.adapters.read_channel(record, channel)
        signal = values[start : start + self.window_length].copy().reshape(1, -1)
        signal = signal.astype(np.float32, copy=False)
        return UnifiedSample(
            signal=signal,
            sample_id=f"{record.record_id}/{channel.name}/{start}",
            dataset_id=record.dataset_id,
            device_id=record.device_id,
            record_id=record.record_id,
            acquisition_id=record.acquisition_id,
            channel=channel.name,
            sample_rate_hz=record.sample_rate_hz,
            start=start,
            condition_label=record.condition_label,
            condition_parameters=record.condition_parameters,
            fault_label=record.fault_label,
            fault_group=record.fault_group,
            fault_parameters=record.fault_parameters,
        )
