from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .long_signal import LongSignalCorpus, LongSignalRecord


NORMALIZATION_METHODS = ("max_abs", "rms", "mean_ac_rms")
NORMALIZATION_SCOPES = {
    "max_abs": "parent_long_signal_max_abs",
    "rms": "parent_long_signal_rms",
    "mean_ac_rms": "parent_long_signal_mean_ac_rms",
}


def normalization_statistics(
    signal: np.ndarray,
    method: str,
) -> tuple[float, float]:
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError("normalization signal must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("normalization signal contains non-finite values")
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"unsupported normalization method {method}")
    location = float(values.mean(dtype=np.float64)) if method == "mean_ac_rms" else 0.0
    centered = values.astype(np.float64) - location
    if method == "max_abs":
        scale = float(np.max(np.abs(values)))
    else:
        scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(location) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization statistics are invalid")
    return location, scale


@dataclass(frozen=True, slots=True)
class ParentSignalBatch:
    """Source-free views normalized by an explicitly sampled parent segment."""

    normalized_signal: np.ndarray
    amplitude_scale: np.ndarray
    location_offset: np.ndarray
    fault: np.ndarray
    fault_group: np.ndarray
    condition_label: np.ndarray
    condition_parameters: np.ndarray
    record_id: np.ndarray
    parent_segment_id: np.ndarray
    parent_start: np.ndarray
    parent_start_time_seconds: np.ndarray
    parent_length: np.ndarray
    view_start: np.ndarray
    view_start_time_seconds: np.ndarray
    view_length: np.ndarray
    normalization_scope: np.ndarray
    normalization_method: np.ndarray
    device_id: np.ndarray
    sensor_location: np.ndarray
    channel: np.ndarray
    signal_quantity: np.ndarray
    sample_rate_hz: np.ndarray

    def __post_init__(self) -> None:
        batch_size = len(self.normalized_signal)
        if self.normalized_signal.ndim != 3:
            raise ValueError("normalized signal must have shape [batch,1,length]")
        if self.normalized_signal.shape[1] != 1:
            raise ValueError("parent signal batch requires one physical channel")
        if not np.isfinite(self.normalized_signal).all():
            raise ValueError("normalized signal contains non-finite values")
        max_abs_rows = self.normalization_method == "max_abs"
        if np.any(max_abs_rows) and np.max(
            np.abs(self.normalized_signal[max_abs_rows]), initial=0.0
        ) > 1.0 + 1e-6:
            raise ValueError("normalized signal exceeds its parent max-absolute scale")
        vector_fields = (
            self.amplitude_scale,
            self.location_offset,
            self.fault,
            self.fault_group,
            self.condition_label,
            self.record_id,
            self.parent_segment_id,
            self.parent_start,
            self.parent_start_time_seconds,
            self.parent_length,
            self.view_start,
            self.view_start_time_seconds,
            self.view_length,
            self.normalization_scope,
            self.normalization_method,
            self.device_id,
            self.sensor_location,
            self.channel,
            self.signal_quantity,
            self.sample_rate_hz,
        )
        if any(value.shape != (batch_size,) for value in vector_fields):
            raise ValueError("parent signal metadata does not match batch size")
        if self.condition_parameters.ndim != 2:
            raise ValueError("condition parameters must have shape [batch,parameter]")
        if self.condition_parameters.shape[0] != batch_size:
            raise ValueError("condition parameter batch size differs")
        if np.any(self.amplitude_scale <= 0.0):
            raise ValueError("amplitude scales must be positive")
        if not np.isfinite(self.location_offset).all():
            raise ValueError("location offsets must be finite")
        if not set(self.normalization_method.tolist()) <= set(NORMALIZATION_METHODS):
            raise ValueError("parent signal batch has an unsupported normalization method")
        if np.any(self.parent_length < self.view_length):
            raise ValueError("a model view cannot exceed its normalization parent")
        if np.any(self.view_start < self.parent_start):
            raise ValueError("model view starts before its normalization parent")
        if np.any(
            self.view_start + self.view_length
            > self.parent_start + self.parent_length
        ):
            raise ValueError("model view ends after its normalization parent")
        if np.any(self.sample_rate_hz <= 0):
            raise ValueError("sample rates must be positive")


def restore_physical_signal(
    normalized_signal: np.ndarray,
    amplitude_scale: np.ndarray,
    location_offset: np.ndarray | None = None,
) -> np.ndarray:
    normalized = np.asarray(normalized_signal, dtype=np.float32)
    amplitude = np.asarray(amplitude_scale, dtype=np.float32)
    if normalized.ndim != 3 or normalized.shape[1] != 1:
        raise ValueError("normalized signal must have shape [batch,1,length]")
    if amplitude.shape != (len(normalized),):
        raise ValueError("amplitude scale must have shape [batch]")
    location = (
        np.zeros(len(normalized), dtype=np.float32)
        if location_offset is None
        else np.asarray(location_offset, dtype=np.float32)
    )
    if location.shape != (len(normalized),):
        raise ValueError("location offset must have shape [batch]")
    if not (
        np.isfinite(normalized).all()
        and np.isfinite(amplitude).all()
        and np.isfinite(location).all()
    ):
        raise ValueError("signal restoration inputs must be finite")
    if np.any(amplitude <= 0.0):
        raise ValueError("amplitude scales must be positive")
    return location[:, None, None] + normalized * amplitude[:, None, None]


class SourceFreeLongSignalCorpus:
    """Parent-first long-signal sampling for the unified generator."""

    def __init__(self, corpus: LongSignalCorpus) -> None:
        self.corpus = corpus
        self.records = corpus.records
        self.parameter_names = corpus.parameter_names
        self.fault_labels = corpus.fault_labels
        route_facts = {
            (
                record.record.dataset_id,
                record.record.device_id,
                record.record.channels[0].location,
                record.record.channels[0].name,
                record.record.channels[0].quantity,
                record.record.sample_rate_hz,
            )
            for record in self.records
        }
        if len(route_facts) != 1:
            raise ValueError(
                "one source-free corpus requires one device/sensor/channel route"
            )
        (
            self.dataset_id,
            self.device_id,
            self.sensor_location,
            self.channel,
            self.signal_quantity,
            self.sample_rate_hz,
        ) = next(iter(route_facts))

    @staticmethod
    def _fits(interval: tuple[int, int] | None, length: int) -> bool:
        return interval is not None and interval[1] - interval[0] >= length

    def eligible_indices(self, split: str, parent_length: int) -> np.ndarray:
        if split not in {"train", "validation"}:
            raise ValueError("long-signal split must be train or validation")
        if parent_length <= 0:
            raise ValueError("parent length must be positive")
        eligible = np.asarray(
            [
                index
                for index, record in enumerate(self.records)
                if self._fits(getattr(record, f"{split}_interval"), parent_length)
            ],
            dtype=np.int64,
        )
        if not len(eligible):
            raise ValueError(f"no {split} record supports parent length {parent_length}")
        return eligible

    def sample_batch(
        self,
        batch_size: int,
        *,
        parent_length: int,
        view_length: int,
        split: str,
        rng: np.random.Generator,
        record_indices: np.ndarray | None = None,
        normalization_method: str = "max_abs",
        parent_start_codes: np.ndarray | None = None,
    ) -> ParentSignalBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0 < view_length <= parent_length:
            raise ValueError("view length must be within the parent length")
        if normalization_method not in NORMALIZATION_METHODS:
            raise ValueError(f"unsupported normalization method {normalization_method}")
        if parent_start_codes is not None:
            if split != "train" or record_indices is None or not isinstance(parent_start_codes, np.ndarray) or parent_start_codes.dtype != np.uint32 or parent_start_codes.shape != (batch_size,):
                raise ValueError("explicit crop codes require source training records and one uint32 per parent")
        eligible = self.eligible_indices(split, parent_length)
        if record_indices is None:
            record_indices = eligible[
                rng.integers(0, len(eligible), size=batch_size, dtype=np.int64)
            ]
        else:
            record_indices = np.asarray(record_indices, dtype=np.int64)
            if record_indices.shape != (batch_size,):
                raise ValueError("record indices do not match batch size")

        normalized = np.empty((batch_size, 1, view_length), dtype=np.float32)
        amplitudes = np.empty(batch_size, dtype=np.float32)
        locations = np.empty(batch_size, dtype=np.float32)
        faults = np.empty(batch_size, dtype=np.int64)
        groups = np.empty(batch_size, dtype=object)
        condition_labels = np.empty(batch_size, dtype=object)
        parameters = np.empty(
            (batch_size, len(self.parameter_names)), dtype=np.float32
        )
        record_ids = np.empty(batch_size, dtype=object)
        parent_ids = np.empty(batch_size, dtype=object)
        parent_starts = np.empty(batch_size, dtype=np.int64)
        parent_lengths = np.full(batch_size, parent_length, dtype=np.int64)
        view_starts = np.empty(batch_size, dtype=np.int64)
        view_lengths = np.full(batch_size, view_length, dtype=np.int64)

        for row, record_index in enumerate(record_indices):
            record = self.records[int(record_index)]
            interval = getattr(record, f"{split}_interval")
            if not self._fits(interval, parent_length) or interval is None:
                raise ValueError("explicit record cannot provide requested parent length")
            interval_begin, interval_end = interval
            parent_start = int(
                rng.integers(interval_begin, interval_end - parent_length + 1)
            )
            if parent_start_codes is not None:
                valid_starts = interval_end - interval_begin - parent_length + 1
                parent_start = interval_begin + ((int(parent_start_codes[row]) * valid_starts) >> 32)
            parent = record.signal[parent_start : parent_start + parent_length]
            location, amplitude = normalization_statistics(
                parent,
                normalization_method,
            )
            view_offset = int(rng.integers(0, parent_length - view_length + 1))
            view_start = parent_start + view_offset
            view = record.signal[view_start : view_start + view_length]
            normalized[row, 0] = (view - location) / amplitude
            amplitudes[row] = amplitude
            locations[row] = location
            faults[row] = record.fault_id
            groups[row] = record.record.fault_group
            condition_labels[row] = record.record.condition_label
            parameters[row] = record.parameters
            record_ids[row] = record.record.record_id
            parent_ids[row] = (
                f"{record.record.record_id}:{split}:{parent_start}:{parent_length}"
            )
            parent_starts[row] = parent_start
            view_starts[row] = view_start

        repeated = lambda value: np.full(batch_size, value, dtype=object)
        return ParentSignalBatch(
            normalized_signal=normalized,
            amplitude_scale=amplitudes,
            location_offset=locations,
            fault=faults,
            fault_group=groups.astype(str),
            condition_label=condition_labels.astype(str),
            condition_parameters=parameters,
            record_id=record_ids.astype(str),
            parent_segment_id=parent_ids.astype(str),
            parent_start=parent_starts,
            parent_start_time_seconds=parent_starts.astype(np.float64)
            / float(self.sample_rate_hz),
            parent_length=parent_lengths,
            view_start=view_starts,
            view_start_time_seconds=view_starts.astype(np.float64)
            / float(self.sample_rate_hz),
            view_length=view_lengths,
            normalization_scope=repeated(
                NORMALIZATION_SCOPES[normalization_method]
            ).astype(str),
            normalization_method=repeated(normalization_method).astype(str),
            device_id=repeated(self.device_id).astype(str),
            sensor_location=repeated(self.sensor_location).astype(str),
            channel=repeated(self.channel).astype(str),
            signal_quantity=repeated(self.signal_quantity).astype(str),
            sample_rate_hz=np.full(
                batch_size, self.sample_rate_hz, dtype=np.int64
            ),
        )
