from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from .data import SignalRecord, WindowCatalog
from .models import CNN1D


MODEL_SPLITS = ("train", "validation", "test")
MANIFEST_SPLITS = (*MODEL_SPLITS, "guard", "unused")


@dataclass(frozen=True, slots=True)
class ConditionSplit:
    indices: dict[str, np.ndarray]
    digest: str
    catalog_digest: str
    catalog_identity: tuple[dict[str, Any], ...]
    per_cell_counts: dict[str, int]
    per_class_counts: dict[str, int]
    source_conditions: tuple[str, ...]
    target_conditions: tuple[str, ...]
    cells: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class MaterializedData:
    catalog: WindowCatalog
    signals: np.ndarray
    source_indices: np.ndarray
    fault_targets: np.ndarray
    fault_labels: tuple[str, ...]
    fault_groups: tuple[str, ...]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog_identity(catalog: WindowCatalog) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "source_index": source_index,
            "dataset_id": record.dataset_id,
            "device_id": record.device_id,
            "record_id": record.record_id,
            "acquisition_id": record.acquisition_id,
            "condition": record.condition_label,
            "fault": record.fault_label,
            "channel": record.channels[source.channel_index].name,
            "sensor_location": record.channels[source.channel_index].location,
            "signal_quantity": record.channels[source.channel_index].quantity,
            "sample_rate_hz": record.sample_rate_hz,
            "num_samples": record.num_samples,
            "window_count": source.window_count,
        }
        for source_index, source in enumerate(catalog.sources)
        for record in (catalog.records[source.record_index],)
    )


def condition_split_digest(
    indices: dict[str, np.ndarray],
    source_conditions: Iterable[str],
    target_conditions: Iterable[str],
    catalog: Iterable[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"faultcrafter.condition_split.v4\0")
    for name in MANIFEST_SPLITS:
        values = indices[name].astype("<i8", copy=False)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
        digest.update(values.tobytes())
    metadata = {
        "source_conditions": sorted(str(value) for value in source_conditions),
        "target_conditions": sorted(str(value) for value in target_conditions),
        "catalog": tuple(catalog),
    }
    digest.update(canonical_digest(metadata).encode("ascii"))
    return digest.hexdigest()


def select_experiment_records(
    records: Iterable[SignalRecord],
    specification: dict[str, Any],
    *,
    allow_empty_target_conditions: bool = False,
) -> tuple[SignalRecord, ...]:
    device_id = str(specification["device_id"])
    channels = tuple(str(value) for value in specification["channels"])
    if len(channels) != 1:
        raise ValueError(f"Experiment 0 requires one fixed channel, got {channels}")

    source_conditions = tuple(str(value) for value in specification["source_conditions"])
    target_conditions = tuple(str(value) for value in specification["target_conditions"])
    if not source_conditions or (
        not target_conditions and not allow_empty_target_conditions
    ):
        raise ValueError("source and target condition sets must both be non-empty")
    if set(source_conditions) & set(target_conditions):
        raise ValueError("source and target conditions overlap")
    selected_conditions = set(source_conditions) | set(target_conditions)

    candidates: list[SignalRecord] = []
    for record in records:
        if record.device_id != device_id or record.condition_label not in selected_conditions:
            continue
        retained_channels = tuple(channel for channel in record.channels if channel.name in channels)
        if retained_channels:
            candidates.append(replace(record, channels=retained_channels))
    available_conditions = {record.condition_label for record in candidates}
    missing_conditions = sorted(selected_conditions - available_conditions)
    if missing_conditions:
        raise ValueError(f"{device_id} is missing conditions {missing_conditions}")

    fault_specification = specification["faults"]
    if fault_specification == "intersection":
        fault_sets = [
            {record.fault_label for record in candidates if record.condition_label == condition}
            for condition in sorted(selected_conditions)
        ]
        faults = set.intersection(*fault_sets)
    elif isinstance(fault_specification, list):
        faults = {str(value) for value in fault_specification}
    else:
        raise ValueError("faults must be an explicit list or 'intersection'")
    if not faults:
        raise ValueError(f"{device_id} has no shared fault labels")

    selected = tuple(record for record in candidates if record.fault_label in faults)
    cells = {(record.fault_label, record.condition_label) for record in selected}
    missing_cells = sorted(
        (fault, condition)
        for fault in faults
        for condition in selected_conditions
        if (fault, condition) not in cells
    )
    if missing_cells:
        raise ValueError(f"missing fault-condition cells for {device_id}: {missing_cells}")
    if {channel.name for record in selected for channel in record.channels} != set(channels):
        raise ValueError(f"selected records do not provide exactly channel {channels[0]}")

    measurement_routes = {
        (
            record.dataset_id,
            record.device_id,
            channel.location,
            channel.name,
            record.sample_rate_hz,
            channel.quantity,
        )
        for record in selected
        for channel in record.channels
    }
    if len(measurement_routes) != 1:
        raise ValueError(
            "one experiment requires one fixed dataset/device/sensor/channel/"
            f"sample-rate/quantity route, got {sorted(measurement_routes)}"
        )
    return selected


def _source_bases(catalog: WindowCatalog) -> np.ndarray:
    return np.concatenate(
        (np.asarray([0], dtype=np.int64), np.asarray(catalog.ends[:-1], dtype=np.int64))
    )


def _balanced_allocation(
    total: int,
    capacities: Iterable[int],
    rng: np.random.Generator,
) -> np.ndarray:
    capacities = np.asarray(tuple(int(value) for value in capacities), dtype=np.int64)
    if total < 0 or len(capacities) == 0 or np.any(capacities < 0):
        raise ValueError("balanced allocation requires non-negative capacity")
    if total > int(capacities.sum()):
        raise ValueError(f"requested {total} windows from capacity {int(capacities.sum())}")
    allocation = np.zeros(len(capacities), dtype=np.int64)
    order = rng.permutation(len(capacities))
    remaining = total
    while remaining:
        progress = False
        for index in order:
            if allocation[index] >= capacities[index]:
                continue
            allocation[index] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            raise RuntimeError("balanced allocation stopped before satisfying the request")
    return allocation


def _random_selection(
    indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 0 or count > len(indices):
        raise ValueError(f"cannot select {count} values from {len(indices)} candidates")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(indices, size=count, replace=False)).astype(
        np.int64,
        copy=False,
    )


def build_condition_split(
    catalog: WindowCatalog,
    *,
    source_conditions: Iterable[str],
    target_conditions: Iterable[str],
    per_cell_counts: dict[str, int],
    seed: int,
) -> ConditionSplit:
    source_conditions = tuple(sorted({str(value) for value in source_conditions}))
    target_conditions = tuple(sorted({str(value) for value in target_conditions}))
    if set(per_cell_counts) != set(MODEL_SPLITS):
        raise ValueError(f"per-cell counts must define exactly {MODEL_SPLITS}")
    per_cell_counts = {name: int(per_cell_counts[name]) for name in MODEL_SPLITS}
    if any(value <= 0 for value in per_cell_counts.values()):
        raise ValueError("per-cell counts must be positive")
    if set(source_conditions) & set(target_conditions):
        raise ValueError("source and target conditions overlap")

    expected_conditions = set(source_conditions) | set(target_conditions)
    catalog_conditions = {record.condition_label for record in catalog.records}
    if catalog_conditions != expected_conditions:
        raise ValueError(
            f"catalog conditions {sorted(catalog_conditions)} do not match split "
            f"conditions {sorted(expected_conditions)}"
        )

    channel_names = {
        catalog.records[source.record_index].channels[source.channel_index].name
        for source in catalog.sources
    }
    if len(channel_names) != 1:
        raise ValueError(f"condition split requires one channel, got {sorted(channel_names)}")
    channel_name = next(iter(channel_names))
    fault_labels = tuple(sorted({record.fault_label for record in catalog.records}))
    validation_pool_fraction = per_cell_counts["validation"] / (
        per_cell_counts["train"] + per_cell_counts["validation"]
    )

    bases = _source_bases(catalog)
    pools: dict[
        str,
        dict[tuple[str, str, str], list[tuple[int, np.ndarray]]],
    ] = {name: defaultdict(list) for name in MODEL_SPLITS}
    guard_parts: list[np.ndarray] = []
    for source_index, source in enumerate(catalog.sources):
        record = catalog.records[source.record_index]
        channel = record.channels[source.channel_index]
        base = int(bases[source_index])
        count = int(source.window_count)
        key = (record.fault_label, record.condition_label, channel.name)

        if record.condition_label in target_conditions:
            pools["test"][key].append(
                (source_index, np.arange(base, base + count, dtype=np.int64))
            )
            continue

        validation_count = max(1, int(round(count * validation_pool_fraction)))
        validation_start = count - validation_count
        first_validation_sample = validation_start * catalog.stride
        train_stop = (first_validation_sample - catalog.window_length) // catalog.stride + 1
        if train_stop <= 0:
            raise ValueError(
                f"source {record.record_id}/{channel.name} is too short for disjoint train/validation"
            )
        train_stop = min(train_stop, validation_start)
        train_indices = np.arange(base, base + train_stop, dtype=np.int64)
        guard_indices = np.arange(
            base + train_stop,
            base + validation_start,
            dtype=np.int64,
        )
        validation_indices = np.arange(
            base + validation_start,
            base + count,
            dtype=np.int64,
        )
        pools["train"][key].append((source_index, train_indices))
        pools["validation"][key].append((source_index, validation_indices))
        if len(guard_indices):
            guard_parts.append(guard_indices)

        last_train_end = (train_stop - 1) * catalog.stride + catalog.window_length
        first_validation_start = validation_start * catalog.stride
        if last_train_end > first_validation_start:
            raise RuntimeError("source train and validation windows share raw samples")

    rng = np.random.default_rng(seed)
    selected_parts: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_SPLITS}
    split_conditions = {
        "train": source_conditions,
        "validation": source_conditions,
        "test": target_conditions,
    }
    for split_name in MODEL_SPLITS:
        for fault in fault_labels:
            for condition in split_conditions[split_name]:
                key = (fault, condition, channel_name)
                if not pools[split_name].get(key):
                    raise ValueError(f"split {split_name} lacks candidate cell {key}")
                entries = pools[split_name][key]
                requested = per_cell_counts[split_name]
                capacity = sum(len(indices) for _, indices in entries)
                if requested > capacity:
                    raise ValueError(
                        f"split {split_name} cell {key} requests {requested} "
                        f"windows from capacity {capacity}"
                    )
                record_allocation = _balanced_allocation(
                    requested,
                    [len(indices) for _, indices in entries],
                    rng,
                )
                for (_, candidates), record_count in zip(
                    entries,
                    record_allocation,
                    strict=True,
                ):
                    selected = _random_selection(candidates, int(record_count), rng)
                    if len(selected):
                        selected_parts[split_name].append(selected)

    indices = {
        name: (
            np.sort(np.concatenate(selected_parts[name])).astype(
                np.int64, copy=False
            )
            if selected_parts[name]
            else np.empty(0, dtype=np.int64)
        )
        for name in MODEL_SPLITS
    }
    indices["guard"] = (
        np.sort(np.concatenate(guard_parts)).astype(np.int64, copy=False)
        if guard_parts
        else np.empty(0, dtype=np.int64)
    )

    assignment = np.full(len(catalog), -1, dtype=np.int8)
    for split_id, name in enumerate((*MODEL_SPLITS, "guard")):
        selected = indices[name]
        if np.any(assignment[selected] != -1):
            raise RuntimeError(f"split {name} overlaps another split")
        assignment[selected] = split_id
    indices["unused"] = np.flatnonzero(assignment == -1).astype(np.int64, copy=False)
    assignment[indices["unused"]] = MANIFEST_SPLITS.index("unused")
    if np.any(assignment == -1):
        raise RuntimeError("split manifest does not cover every catalog window")

    cell_counts: dict[tuple[str, str, str], dict[str, int | str]] = {}
    for source_index, source in enumerate(catalog.sources):
        record = catalog.records[source.record_index]
        channel = record.channels[source.channel_index]
        base = int(bases[source_index])
        key = (record.fault_label, record.condition_label, channel.name)
        row = cell_counts.setdefault(
            key,
            {
                "fault": record.fault_label,
                "condition": record.condition_label,
                "channel": channel.name,
                "total": 0,
                **{name: 0 for name in MANIFEST_SPLITS},
            },
        )
        row["total"] = int(row["total"]) + source.window_count
        counts = np.bincount(
            assignment[base : base + source.window_count],
            minlength=len(MANIFEST_SPLITS),
        )
        for split_id, name in enumerate(MANIFEST_SPLITS):
            row[name] = int(row[name]) + int(counts[split_id])

    for row in cell_counts.values():
        condition = str(row["condition"])
        for split_name in MODEL_SPLITS:
            expected = (
                per_cell_counts[split_name]
                if condition in split_conditions[split_name]
                else 0
            )
            if int(row[split_name]) != expected:
                raise RuntimeError(
                    f"split {split_name} cell {row['fault']}/{condition} has "
                    f"{row[split_name]} windows, expected {expected}"
                )

    per_class_counts = {
        "train": per_cell_counts["train"] * len(source_conditions),
        "validation": per_cell_counts["validation"] * len(source_conditions),
        "test": per_cell_counts["test"] * len(target_conditions),
    }

    identity = catalog_identity(catalog)
    return ConditionSplit(
        indices=indices,
        digest=condition_split_digest(
            indices,
            source_conditions,
            target_conditions,
            identity,
        ),
        catalog_digest=canonical_digest(identity),
        catalog_identity=identity,
        per_cell_counts=per_cell_counts,
        per_class_counts=per_class_counts,
        source_conditions=source_conditions,
        target_conditions=target_conditions,
        cells=tuple(cell_counts[key] for key in sorted(cell_counts)),
    )


def materialize(catalog: WindowCatalog) -> MaterializedData:
    fault_labels = tuple(sorted({record.fault_label for record in catalog.records}))
    fault_to_id = {label: index for index, label in enumerate(fault_labels)}
    group_by_fault: dict[str, str] = {}
    for record in catalog.records:
        existing = group_by_fault.setdefault(record.fault_label, record.fault_group)
        if existing != record.fault_group:
            raise ValueError(f"fault label {record.fault_label} maps to multiple groups")

    signals = np.empty((len(catalog), 1, catalog.window_length), dtype=np.float32)
    source_indices = np.empty(len(catalog), dtype=np.int32)
    fault_targets = np.empty(len(catalog), dtype=np.int64)
    sources_by_record: list[list[int]] = [[] for _ in catalog.records]
    for source_index, source in enumerate(catalog.sources):
        sources_by_record[source.record_index].append(source_index)

    interval = max(1, len(catalog.records) // 10)
    for record_index, record in enumerate(catalog.records):
        channels = catalog.adapters.read_channels(record)
        for source_index in sources_by_record[record_index]:
            source = catalog.sources[source_index]
            channel = record.channels[source.channel_index]
            values = channels[channel.name]
            windows = sliding_window_view(values, catalog.window_length)[:: catalog.stride]
            if len(windows) != source.window_count:
                raise RuntimeError(
                    f"window count mismatch for {record.record_id}/{channel.name}: "
                    f"{len(windows)} != {source.window_count}"
                )
            begin = 0 if source_index == 0 else catalog.ends[source_index - 1]
            end = catalog.ends[source_index]
            signals[begin:end, 0] = windows
            source_indices[begin:end] = source_index
            fault_targets[begin:end] = fault_to_id[record.fault_label]
        if (record_index + 1) % interval == 0 or record_index + 1 == len(catalog.records):
            print(f"materialized {record_index + 1}/{len(catalog.records)} records", flush=True)

    if not np.isfinite(signals).all():
        raise RuntimeError("materialized signals contain non-finite values")
    return MaterializedData(
        catalog=catalog,
        signals=signals,
        source_indices=source_indices,
        fault_targets=fault_targets,
        fault_labels=fault_labels,
        fault_groups=tuple(group_by_fault[label] for label in fault_labels),
    )


def fit_training_scale(
    signals: np.ndarray,
    train_indices: np.ndarray,
    *,
    batch_size: int = 2048,
) -> dict[str, int | float]:
    value_sum = 0.0
    square_sum = 0.0
    point_count = 0
    for begin in range(0, len(train_indices), batch_size):
        block = signals[train_indices[begin : begin + batch_size]].astype(np.float64)
        value_sum += float(block.sum(dtype=np.float64))
        square_sum += float(np.square(block).sum(dtype=np.float64))
        point_count += int(block.size)
    mean = value_sum / point_count
    variance = max(square_sum / point_count - mean * mean, 0.0)
    standard_deviation = math.sqrt(variance)
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError(f"invalid training standard deviation {standard_deviation}")
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "training_window_count": int(len(train_indices)),
        "training_point_count": point_count,
    }


def _source_metadata(data: MaterializedData, indices: np.ndarray) -> dict[str, np.ndarray]:
    catalog = data.catalog
    source_indices = data.source_indices[indices]
    bases = _source_bases(catalog)
    local_indices = indices - bases[source_indices]
    starts = local_indices * catalog.stride
    metadata: dict[str, np.ndarray] = {
        "sample_id": np.empty(len(indices), dtype=object),
        "record_id": np.empty(len(indices), dtype=object),
        "acquisition_id": np.empty(len(indices), dtype=object),
        "condition": np.empty(len(indices), dtype=object),
        "fault": np.empty(len(indices), dtype=object),
        "fault_group": np.empty(len(indices), dtype=object),
        "channel": np.empty(len(indices), dtype=object),
        "sample_rate_hz": np.empty(len(indices), dtype=np.int32),
        "start": starts.astype(np.int32),
    }
    for source_index in np.unique(source_indices):
        selected = np.flatnonzero(source_indices == source_index)
        source = catalog.sources[int(source_index)]
        record = catalog.records[source.record_index]
        channel = record.channels[source.channel_index]
        metadata["sample_id"][selected] = [
            f"{record.record_id}/{channel.name}/{int(start)}" for start in starts[selected]
        ]
        metadata["record_id"][selected] = record.record_id
        metadata["acquisition_id"][selected] = record.acquisition_id
        metadata["condition"][selected] = record.condition_label
        metadata["fault"][selected] = record.fault_label
        metadata["fault_group"][selected] = record.fault_group
        metadata["channel"][selected] = channel.name
        metadata["sample_rate_hz"][selected] = record.sample_rate_hz
    for key in ("sample_id", "record_id", "acquisition_id", "condition", "fault", "fault_group", "channel"):
        metadata[key] = metadata[key].astype(str)
    return metadata


def metric_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
    *,
    detailed: bool,
) -> dict[str, Any]:
    if len(y_true) == 0:
        raise ValueError("cannot calculate metrics for an empty set")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=precision + recall != 0,
    )
    present = support != 0
    result: dict[str, Any] = {
        "support": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
    }
    if detailed:
        result["per_class"] = [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ]
        result["confusion_matrix"] = matrix.tolist()
    return result


def build_test_metrics(
    data: MaterializedData,
    test_indices: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y_true = data.fault_targets[test_indices]
    metadata = _source_metadata(data, test_indices)
    result: dict[str, Any] = {
        "overall": metric_block(y_true, y_pred, data.fault_labels, detailed=True),
        "by_condition": {},
    }
    for condition in sorted(np.unique(metadata["condition"])):
        selected = metadata["condition"] == condition
        result["by_condition"][str(condition)] = metric_block(
            y_true[selected],
            y_pred[selected],
            data.fault_labels,
            detailed=True,
        )
    return result, metadata


def _normalized_tensor(
    data: MaterializedData,
    indices: np.ndarray,
    scale: dict[str, int | float],
    device: torch.device,
) -> torch.Tensor:
    signal = torch.from_numpy(np.ascontiguousarray(data.signals[indices])).to(device)
    return signal.sub(float(scale["mean"])).div(float(scale["standard_deviation"]))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: MaterializedData,
    indices: np.ndarray,
    scale: dict[str, int | float],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        signal = _normalized_tensor(data, selected, scale, device)
        target = torch.from_numpy(data.fault_targets[selected]).to(device)
        logits = model(signal)
        loss_sum += float(criterion(logits, target).item())
        correct += int((logits.argmax(dim=1) == target).sum().item())
    return loss_sum / len(indices), correct / len(indices)


@torch.no_grad()
def predict(
    model: nn.Module,
    data: MaterializedData,
    indices: np.ndarray,
    scale: dict[str, int | float],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    predictions = np.empty(len(indices), dtype=np.int64)
    probabilities = np.empty((len(indices), len(data.fault_labels)), dtype=np.float32)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    for begin in range(0, len(indices), batch_size):
        end = min(begin + batch_size, len(indices))
        selected = indices[begin:end]
        signal = _normalized_tensor(data, selected, scale, device)
        target = torch.from_numpy(data.fault_targets[selected]).to(device)
        logits = model(signal)
        loss_sum += float(criterion(logits, target).item())
        probability = torch.softmax(logits, dim=1)
        probabilities[begin:end] = probability.cpu().numpy()
        predictions[begin:end] = probability.argmax(dim=1).cpu().numpy()
    return loss_sum / len(indices), predictions, probabilities


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_run_definition(
    result_directory: Path,
    data: MaterializedData | WindowCatalog,
    split: ConditionSplit,
    scale: dict[str, int | float],
    *,
    run_id: str,
    relation: str,
    seed: int,
    config_digest: str,
    preprocessing: dict[str, Any] | None = None,
) -> None:
    catalog = data.catalog if isinstance(data, MaterializedData) else data
    result_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_directory / "split_indices.npz", **split.indices)
    records = [
        {
            "record_id": record.record_id,
            "acquisition_id": record.acquisition_id,
            "condition": record.condition_label,
            "fault": record.fault_label,
            "fault_group": record.fault_group,
            "channels": [channel.name for channel in record.channels],
            "sample_rate_hz": record.sample_rate_hz,
            "num_samples": record.num_samples,
            "path": record.path.as_posix(),
        }
        for record in catalog.records
    ]
    _json_write(
        result_directory / "split_summary.json",
        {
            "run_id": run_id,
            "relation": relation,
            "seed": seed,
            "config_sha256": config_digest,
            "split_sha256": split.digest,
            "catalog_sha256": split.catalog_digest,
            "catalog_identity": split.catalog_identity,
            "window_length": catalog.window_length,
            "stride": catalog.stride,
            "preprocessing": preprocessing,
            "per_cell_window_counts": split.per_cell_counts,
            "per_class_window_counts": split.per_class_counts,
            "source_validation_pool_fraction": (
                split.per_cell_counts["validation"]
                / (split.per_cell_counts["train"] + split.per_cell_counts["validation"])
            ),
            "source_conditions": split.source_conditions,
            "target_conditions": split.target_conditions,
            "counts": {name: int(len(split.indices[name])) for name in MANIFEST_SPLITS},
            "cells": split.cells,
            "records": records,
        },
    )
    _json_write(result_directory / "training_scale.json", scale)


def train_fault_cnn(
    data: MaterializedData,
    split: ConditionSplit,
    scale: dict[str, int | float],
    *,
    run_id: str,
    relation: str,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
    result_directory: Path,
    config_digest: str,
) -> dict[str, Any]:
    for name in MODEL_SPLITS:
        counts = np.bincount(
            data.fault_targets[split.indices[name]],
            minlength=len(data.fault_labels),
        )
        expected = split.per_class_counts[name]
        if np.any(counts != expected):
            invalid = {
                data.fault_labels[index]: int(count)
                for index, count in enumerate(counts)
                if count != expected
            }
            raise ValueError(f"split {name} has invalid per-class counts {invalid}")

    batch_size = int(training_config["batch_size"])
    evaluation_batch_size = int(training_config["evaluation_batch_size"])
    maximum_epochs = int(training_config["maximum_epochs"])
    checkpoint_preheat_epochs = int(training_config["checkpoint_preheat_epochs"])
    patience = int(training_config["early_stopping_patience"])
    minimum_improvement = float(training_config["minimum_accuracy_improvement"])
    if checkpoint_preheat_epochs <= 0 or checkpoint_preheat_epochs > maximum_epochs:
        raise ValueError("checkpoint preheat epochs must be in [1, maximum epochs]")
    if patience <= 0:
        raise ValueError("early stopping patience must be positive")

    set_reproducible_seed(seed)
    model = CNN1D(num_classes=len(data.fault_labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    train_indices = split.indices["train"]
    validation_indices = split.indices["validation"]
    test_indices = split.indices["test"]
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "best_cnn.pt"

    best_accuracy = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = train_indices.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        loss_sum = 0.0
        correct = 0
        for begin in range(0, len(order), batch_size):
            selected = order[begin : begin + batch_size]
            signal = _normalized_tensor(data, selected, scale, device)
            target = torch.from_numpy(data.fault_targets[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(selected)
            correct += int((logits.argmax(dim=1) == target).sum().item())

        validation_loss, validation_accuracy = evaluate(
            model,
            data,
            validation_indices,
            scale,
            batch_size=evaluation_batch_size,
            device=device,
        )
        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": loss_sum / len(order),
            "training_accuracy": correct / len(order),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_result)
        checkpoint_eligible = epoch >= checkpoint_preheat_epochs
        if checkpoint_eligible:
            if validation_accuracy - best_accuracy > minimum_improvement:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                bad_epochs = 0
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save(
                    {
                        "model": state,
                        "labels": data.fault_labels,
                        "epoch": epoch,
                        "validation_accuracy": validation_accuracy,
                        "scale": scale,
                        "split_sha256": split.digest,
                        "config_sha256": config_digest,
                        "seed": seed,
                    },
                    checkpoint_path,
                )
            else:
                bad_epochs += 1
        best_display = "preheat" if not checkpoint_eligible else f"{best_accuracy:.6f}"
        print(
            f"run={run_id} epoch={epoch} "
            f"train_loss={epoch_result['training_loss']:.6f} "
            f"train_acc={epoch_result['training_accuracy']:.6f} "
            f"val_loss={validation_loss:.6f} val_acc={validation_accuracy:.6f} "
            f"best={best_display} bad={bad_epochs}/{patience}",
            flush=True,
        )
        if checkpoint_eligible and bad_epochs >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    test_loss, predictions, probabilities = predict(
        model,
        data,
        test_indices,
        scale,
        batch_size=evaluation_batch_size,
        device=device,
    )
    test_metrics, metadata = build_test_metrics(data, test_indices, predictions)
    test_metrics["overall"]["loss"] = test_loss

    result_directory.mkdir(parents=True, exist_ok=True)
    _json_write(result_directory / "history.json", history)
    np.savez_compressed(
        result_directory / "predictions.npz",
        global_index=test_indices,
        y_true=data.fault_targets[test_indices].astype(np.int16),
        y_pred=predictions.astype(np.int16),
        probabilities=probabilities,
        labels=np.asarray(data.fault_labels),
        **metadata,
    )
    result = {
        "run_id": run_id,
        "relation": relation,
        "seed": seed,
        "device": str(device),
        "model": "CNN1D",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "labels": data.fault_labels,
        "training": training_config,
        "scale": scale,
        "config_sha256": config_digest,
        "split_sha256": split.digest,
        "source_conditions": split.source_conditions,
        "target_conditions": split.target_conditions,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "completed_epochs": len(history),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": _file_digest(checkpoint_path),
        },
        "test": test_metrics,
    }
    _json_write(result_directory / "metrics.json", result)
    return result
