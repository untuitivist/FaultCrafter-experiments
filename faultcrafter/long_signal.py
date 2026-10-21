from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .data import AdapterCollection, SignalRecord


def trajectory_length_for_windows(window: int, stride: int, count: int) -> int:
    if window <= 0 or stride <= 0 or count <= 0:
        raise ValueError("window, stride, and count must be positive")
    return window + (count - 1) * stride


def maximum_length_disjoint_intervals(
    records: Iterable[SignalRecord],
    *,
    maximum_length: int,
    train_fraction: float = 0.6,
    validation_start_fraction: float = 0.8,
) -> tuple[
    dict[str, dict[str, tuple[int, int] | None]],
    dict[str, Any],
]:
    """Allocate full-length train/guard/validation regions where a cell permits it."""

    records = tuple(records)
    if not records:
        raise ValueError("disjoint interval allocation requires records")
    if maximum_length <= 0:
        raise ValueError("maximum length must be positive")
    if not 0.0 < train_fraction < validation_start_fraction < 1.0:
        raise ValueError("invalid train/validation interval fractions")

    intervals: dict[str, dict[str, tuple[int, int] | None]] = {}
    rows: list[dict[str, Any]] = []
    supported_cells: set[tuple[str, str]] = set()
    expected_cells = {
        (record.fault_label, record.condition_label) for record in records
    }
    for record in records:
        sample_count = int(record.num_samples)
        assigned: dict[str, tuple[int, int] | None] = {
            "train": None,
            "validation": None,
        }
        guard_interval: tuple[int, int] | None = None
        if sample_count >= 3 * maximum_length:
            preferred_train_end = int(sample_count * train_fraction)
            train_end = min(preferred_train_end, sample_count - 2 * maximum_length)
            preferred_validation_start = int(
                sample_count * validation_start_fraction
            )
            validation_start = min(
                max(preferred_validation_start, train_end + maximum_length),
                sample_count - maximum_length,
            )
            assigned = {
                "train": (0, train_end),
                "validation": (validation_start, sample_count),
            }
            guard_interval = (train_end, validation_start)
            supported_cells.add((record.fault_label, record.condition_label))
        intervals[record.record_id] = assigned
        rows.append(
            {
                "record_id": record.record_id,
                "fault": record.fault_label,
                "condition": record.condition_label,
                "num_samples": sample_count,
                "supports_requested_maximum": assigned["train"] is not None,
                "train_interval": assigned["train"],
                "guard_interval": guard_interval,
                "validation_interval": assigned["validation"],
            }
        )

    missing = sorted(expected_cells - supported_cells)
    if missing:
        raise ValueError(
            "maximum-length train/validation coverage is unavailable for cells "
            f"{missing}"
        )
    return intervals, {
        "policy": (
            "preferred fractional split adjusted only as needed to preserve one "
            "maximum-length train segment, raw guard, and validation segment; "
            "short duplicate records remain declared but are ineligible"
        ),
        "requested_maximum_length": maximum_length,
        "train_fraction": train_fraction,
        "validation_start_fraction": validation_start_fraction,
        "record_count": len(rows),
        "supported_record_count": sum(
            bool(row["supports_requested_maximum"]) for row in rows
        ),
        "records": rows,
    }


@dataclass(frozen=True, slots=True)
class LongSignalRecord:
    record: SignalRecord
    signal: np.ndarray
    fault_id: int
    parameters: np.ndarray
    train_interval: tuple[int, int] | None
    validation_interval: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class LongSignalBatch:
    target: np.ndarray
    source: np.ndarray
    target_amplitude: np.ndarray
    source_amplitude: np.ndarray
    fault: np.ndarray
    target_parameters: np.ndarray
    source_parameters: np.ndarray
    target_record_id: np.ndarray
    source_record_id: np.ndarray
    target_start: np.ndarray
    source_start: np.ndarray


class LongSignalCorpus:
    """Source-only long segments with raw-sample-disjoint train/validation regions."""

    def __init__(
        self,
        records: Iterable[SignalRecord],
        adapters: AdapterCollection,
        *,
        maximum_length: int,
        train_fraction: float = 0.7,
        validation_start_fraction: float = 0.8,
        intervals: dict[str, dict[str, tuple[int, int] | None]] | None = None,
    ) -> None:
        records = tuple(records)
        if not records:
            raise ValueError("long-signal corpus requires source records")
        if maximum_length <= 0:
            raise ValueError("maximum length must be positive")
        if not 0.0 < train_fraction < validation_start_fraction < 1.0:
            raise ValueError("invalid train/validation interval fractions")

        parameter_schemas = {
            tuple(name for name, _ in record.condition_parameters)
            for record in records
        }
        if len(parameter_schemas) != 1:
            raise ValueError("long-signal records have different parameter schemas")
        self.parameter_names = next(iter(parameter_schemas))
        self.fault_labels = tuple(sorted({record.fault_label for record in records}))
        fault_to_id = {label: index for index, label in enumerate(self.fault_labels)}
        loaded: list[LongSignalRecord] = []
        for record in records:
            if len(record.channels) != 1:
                raise ValueError("long-signal corpus requires one physical channel")
            signal = adapters.read_channel(record, record.channels[0]).astype(
                np.float32,
                copy=False,
            )
            if signal.ndim != 1 or len(signal) != record.num_samples:
                raise ValueError(f"invalid raw signal shape for {record.record_id}")
            if intervals is None:
                train_end = int(len(signal) * train_fraction)
                validation_start = int(len(signal) * validation_start_fraction)
                if (
                    train_end < maximum_length
                    or len(signal) - validation_start < maximum_length
                ):
                    raise ValueError(
                        f"record {record.record_id} is too short for long segments"
                    )
                if validation_start - train_end < maximum_length:
                    raise ValueError(
                        f"record {record.record_id} lacks a maximum-length guard interval"
                    )
                train_interval: tuple[int, int] | None = (0, train_end)
                validation_interval: tuple[int, int] | None = (
                    validation_start,
                    len(signal),
                )
            else:
                assigned = intervals.get(record.record_id)
                if assigned is None:
                    continue
                train_interval = assigned.get("train")
                validation_interval = assigned.get("validation")
                for name, interval in (
                    ("train", train_interval),
                    ("validation", validation_interval),
                ):
                    if interval is None:
                        continue
                    begin, end = interval
                    if not 0 <= begin < end <= len(signal):
                        raise ValueError(
                            f"invalid {name} interval for {record.record_id}: {interval}"
                        )
                if (
                    train_interval is not None
                    and validation_interval is not None
                    and train_interval[1] > validation_interval[0]
                ):
                    raise ValueError(
                        f"train and validation intervals overlap for {record.record_id}"
                    )
            loaded.append(
                LongSignalRecord(
                    record=record,
                    signal=np.ascontiguousarray(signal),
                    fault_id=fault_to_id[record.fault_label],
                    parameters=np.asarray(
                        [value for _, value in record.condition_parameters],
                        dtype=np.float32,
                    ),
                    train_interval=train_interval,
                    validation_interval=validation_interval,
                )
            )
        self.records = tuple(loaded)
        if not self.records:
            raise ValueError("long-signal corpus has no records with assigned intervals")
        self.maximum_length = maximum_length
        self._cell_keys = tuple(
            (record.fault_id, record.record.condition_label) for record in self.records
        )

    @property
    def parameter_matrix(self) -> np.ndarray:
        return np.stack([record.parameters for record in self.records])

    @staticmethod
    def _fits(interval: tuple[int, int] | None, length: int) -> bool:
        return interval is not None and interval[1] - interval[0] >= length

    def representative_indices(self, split: str, length: int) -> np.ndarray:
        if split not in {"train", "validation"}:
            raise ValueError("long-signal split must be train or validation")
        by_cell: dict[tuple[int, str], list[int]] = {}
        for index, (record, cell) in enumerate(zip(self.records, self._cell_keys)):
            if self._fits(getattr(record, f"{split}_interval"), length):
                by_cell.setdefault(cell, []).append(index)
        expected = set(self._cell_keys)
        if set(by_cell) != expected:
            missing = sorted(expected - set(by_cell))
            raise ValueError(
                f"length {length} is unsupported by {split} cells {missing}"
            )
        return np.asarray([by_cell[cell][0] for cell in sorted(by_cell)], dtype=np.int64)

    def supports_pair(self, split: str, target_length: int, source_length: int) -> bool:
        try:
            targets = self.representative_indices(split, target_length)
        except ValueError:
            return False
        for target_index in targets:
            target = self.records[int(target_index)]
            if not any(
                source.fault_id == target.fault_id
                and source.record.condition_label != target.record.condition_label
                and self._fits(
                    getattr(source, f"{split}_interval"),
                    source_length,
                )
                for source in self.records
            ):
                return False
        return True

    def supports_length(self, split: str, length: int) -> bool:
        return self.supports_pair(split, length, length)

    @staticmethod
    def _segment(
        record: LongSignalRecord,
        interval: tuple[int, int],
        length: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, int]:
        begin, end = interval
        if length <= 0 or end - begin < length:
            raise ValueError("requested segment does not fit its raw interval")
        start = int(rng.integers(begin, end - length + 1))
        raw = record.signal[start : start + length]
        amplitude = float(np.max(np.abs(raw)))
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError(f"invalid segment amplitude in {record.record.record_id}")
        normalized = (raw / amplitude).astype(np.float32, copy=False)
        return normalized.reshape(1, -1), amplitude, start

    def sample_batch(
        self,
        batch_size: int,
        *,
        target_length: int,
        source_length: int,
        split: str,
        rng: np.random.Generator,
        target_record_indices: np.ndarray | None = None,
    ) -> LongSignalBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("long-signal split must be train or validation")
        eligible = self.representative_indices(split, target_length)
        if target_record_indices is None:
            target_record_indices = eligible[
                rng.integers(0, len(eligible), size=batch_size, dtype=np.int64)
            ]
        else:
            target_record_indices = np.asarray(target_record_indices, dtype=np.int64)
            if target_record_indices.shape != (batch_size,):
                raise ValueError("target record indices do not match batch size")

        targets = np.empty((batch_size, 1, target_length), dtype=np.float32)
        sources = np.empty((batch_size, 1, source_length), dtype=np.float32)
        target_amplitudes = np.empty(batch_size, dtype=np.float32)
        source_amplitudes = np.empty(batch_size, dtype=np.float32)
        faults = np.empty(batch_size, dtype=np.int64)
        target_parameters = np.empty(
            (batch_size, len(self.parameter_names)),
            dtype=np.float32,
        )
        source_parameters = np.empty_like(target_parameters)
        target_record_ids = np.empty(batch_size, dtype=object)
        source_record_ids = np.empty(batch_size, dtype=object)
        target_starts = np.empty(batch_size, dtype=np.int64)
        source_starts = np.empty(batch_size, dtype=np.int64)
        for row, target_index in enumerate(target_record_indices):
            target = self.records[int(target_index)]
            target_interval = getattr(target, f"{split}_interval")
            if not self._fits(target_interval, target_length):
                raise ValueError("explicit target record cannot provide requested length")
            source_candidates = np.asarray(
                [
                    source_index
                    for source_index, source in enumerate(self.records)
                    if source.fault_id == target.fault_id
                    and source.record.condition_label != target.record.condition_label
                    and self._fits(
                        getattr(source, f"{split}_interval"),
                        source_length,
                    )
                ],
                dtype=np.int64,
            )
            if len(source_candidates) == 0:
                raise ValueError("target record lacks a length-compatible source")
            source_index = int(source_candidates[int(rng.integers(0, len(source_candidates)))])
            source = self.records[source_index]
            source_interval = getattr(source, f"{split}_interval")
            if target_interval is None or source_interval is None:
                raise RuntimeError("eligible long-signal interval unexpectedly missing")
            targets[row], target_amplitudes[row], target_starts[row] = self._segment(
                target,
                target_interval,
                target_length,
                rng,
            )
            sources[row], source_amplitudes[row], source_starts[row] = self._segment(
                source,
                source_interval,
                source_length,
                rng,
            )
            faults[row] = target.fault_id
            target_parameters[row] = target.parameters
            source_parameters[row] = source.parameters
            target_record_ids[row] = target.record.record_id
            source_record_ids[row] = source.record.record_id
        return LongSignalBatch(
            target=targets,
            source=sources,
            target_amplitude=target_amplitudes,
            source_amplitude=source_amplitudes,
            fault=faults,
            target_parameters=target_parameters,
            source_parameters=source_parameters,
            target_record_id=target_record_ids.astype(str),
            source_record_id=source_record_ids.astype(str),
            target_start=target_starts,
            source_start=source_starts,
        )

    def sample_multi_source_batch(
        self,
        batch_size: int,
        *,
        target_length: int,
        source_length: int,
        split: str,
        rng: np.random.Generator,
        target_record_indices: np.ndarray | None = None,
    ) -> LongSignalBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("long-signal split must be train or validation")
        eligible = self.representative_indices(split, target_length)
        if target_record_indices is None:
            target_record_indices = eligible[
                rng.integers(0, len(eligible), size=batch_size, dtype=np.int64)
            ]
        else:
            target_record_indices = np.asarray(target_record_indices, dtype=np.int64)
            if target_record_indices.shape != (batch_size,):
                raise ValueError("target record indices do not match batch size")

        condition_labels = sorted({record.record.condition_label for record in self.records})
        source_count = len(condition_labels) - 1
        if source_count <= 0:
            raise ValueError("multi-source batches require at least two conditions")
        targets = np.empty((batch_size, 1, target_length), dtype=np.float32)
        sources = np.empty(
            (batch_size, source_count, 1, source_length), dtype=np.float32
        )
        target_amplitudes = np.empty(batch_size, dtype=np.float32)
        source_amplitudes = np.empty((batch_size, source_count), dtype=np.float32)
        faults = np.empty(batch_size, dtype=np.int64)
        target_parameters = np.empty(
            (batch_size, len(self.parameter_names)), dtype=np.float32
        )
        source_parameters = np.empty(
            (batch_size, source_count, len(self.parameter_names)), dtype=np.float32
        )
        target_record_ids = np.empty(batch_size, dtype=object)
        source_record_ids = np.empty((batch_size, source_count), dtype=object)
        target_starts = np.empty(batch_size, dtype=np.int64)
        source_starts = np.empty((batch_size, source_count), dtype=np.int64)
        for row, target_index in enumerate(target_record_indices):
            target = self.records[int(target_index)]
            target_interval = getattr(target, f"{split}_interval")
            if not self._fits(target_interval, target_length):
                raise ValueError("explicit target record cannot provide requested length")
            source_by_condition: dict[str, list[LongSignalRecord]] = {}
            for source in self.records:
                if (
                    source.fault_id == target.fault_id
                    and source.record.condition_label != target.record.condition_label
                    and self._fits(getattr(source, f"{split}_interval"), source_length)
                ):
                    source_by_condition.setdefault(
                        source.record.condition_label, []
                    ).append(source)
            expected_conditions = set(condition_labels) - {target.record.condition_label}
            if set(source_by_condition) != expected_conditions:
                raise ValueError("target lacks one or more same-fault source conditions")
            if target_interval is None:
                raise RuntimeError("eligible target interval unexpectedly missing")
            targets[row], target_amplitudes[row], target_starts[row] = self._segment(
                target, target_interval, target_length, rng
            )
            for source_row, condition in enumerate(sorted(source_by_condition)):
                candidates = source_by_condition[condition]
                source = candidates[int(rng.integers(0, len(candidates)))]
                source_interval = getattr(source, f"{split}_interval")
                if source_interval is None:
                    raise RuntimeError("eligible source interval unexpectedly missing")
                (
                    sources[row, source_row],
                    source_amplitudes[row, source_row],
                    source_starts[row, source_row],
                ) = self._segment(source, source_interval, source_length, rng)
                source_parameters[row, source_row] = source.parameters
                source_record_ids[row, source_row] = source.record.record_id
            faults[row] = target.fault_id
            target_parameters[row] = target.parameters
            target_record_ids[row] = target.record.record_id
        return LongSignalBatch(
            target=targets,
            source=sources,
            target_amplitude=target_amplitudes,
            source_amplitude=source_amplitudes,
            fault=faults,
            target_parameters=target_parameters,
            source_parameters=source_parameters,
            target_record_id=target_record_ids.astype(str),
            source_record_id=source_record_ids.astype(str),
            target_start=target_starts,
            source_start=source_starts,
        )


def budgeted_intervals_from_split(
    catalog: Any,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_per_cell: int,
    validation_per_cell: int,
    seed: int,
) -> tuple[dict[str, dict[str, tuple[int, int] | None]], dict[str, Any]]:
    """Allocate disjoint continuous blocks with the split's per-record window budget."""

    source_lookup = {id(source): index for index, source in enumerate(catalog.sources)}
    selected_counts = {
        "train": np.zeros(len(catalog.sources), dtype=np.int64),
        "validation": np.zeros(len(catalog.sources), dtype=np.int64),
    }
    for split, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        for global_index in np.asarray(indices, dtype=np.int64):
            source, _, _ = catalog.locate(int(global_index))
            selected_counts[split][source_lookup[id(source)]] += 1

    validation_fraction = validation_per_cell / (train_per_cell + validation_per_cell)
    random = np.random.default_rng(seed)
    intervals: dict[str, dict[str, tuple[int, int] | None]] = {}
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(catalog.sources):
        record = catalog.records[source.record_index]
        train_count = int(selected_counts["train"][source_index])
        validation_count = int(selected_counts["validation"][source_index])
        if train_count == 0 and validation_count == 0:
            continue
        validation_pool_count = max(1, int(round(source.window_count * validation_fraction)))
        validation_start_window = source.window_count - validation_pool_count
        first_validation_sample = validation_start_window * catalog.stride
        train_stop = (
            (first_validation_sample - catalog.window_length) // catalog.stride + 1
        )
        train_pool_end = (train_stop - 1) * catalog.stride + catalog.window_length
        assigned: dict[str, tuple[int, int] | None] = {
            "train": None,
            "validation": None,
        }
        for split, count, pool in (
            ("train", train_count, (0, train_pool_end)),
            (
                "validation",
                validation_count,
                (first_validation_sample, record.num_samples),
            ),
        ):
            if count == 0:
                continue
            span = catalog.window_length + (count - 1) * catalog.stride
            pool_begin, pool_end = pool
            if span > pool_end - pool_begin:
                raise ValueError(
                    f"{record.record_id} cannot fit its {split} continuous budget"
                )
            slots = (pool_end - pool_begin - span) // catalog.stride + 1
            begin = pool_begin + int(random.integers(0, slots)) * catalog.stride
            assigned[split] = (begin, begin + span)
        intervals[record.record_id] = assigned
        rows.append(
            {
                "record_id": record.record_id,
                "fault": record.fault_label,
                "condition": record.condition_label,
                "train_window_budget": train_count,
                "validation_window_budget": validation_count,
                "train_interval": assigned["train"],
                "validation_interval": assigned["validation"],
            }
        )
    report = {
        "allocation": (
            "one contiguous raw block per allocated record; block span equals the "
            "minimum union span of its selected windows"
        ),
        "train_windows_per_cell": train_per_cell,
        "validation_windows_per_cell": validation_per_cell,
        "records": rows,
    }
    return intervals, report
