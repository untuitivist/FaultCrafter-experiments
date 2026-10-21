from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


SOURCE_SUPPORTED_COMMON_RANGE = "source_supported_common_range_v1"


def split_reference_length(config: Mapping[str, Any], maximum_training_length: int) -> int:
    value = config.get("long_signal_split_reference_length", maximum_training_length)
    if type(value) is not int or value <= 0:
        raise ValueError("source split reference length must be a positive integer")
    if "long_signal_split_reference_length" in config and config.get("long_signal_interval_mode") != "fractional_disjoint":
        raise ValueError("explicit source split reference requires fractional disjoint intervals")
    return value


def supported_source_ranges(config: Mapping[str, Any], devices: Sequence[Any]) -> dict:
    requested = config["generation"]["target_length"]
    if type(requested) is not int or requested <= 0 or not devices:
        raise ValueError("source-supported lengths require devices and a positive output length")
    if len({device.device_id for device in devices}) != len(devices):
        raise ValueError("source-supported device routes must be unique")
    splits = {}
    for split, field in (("train", "training_length_range"), ("validation", "validation_length_range")):
        original = config.get(field)
        if not isinstance(original, list) or len(original) != 2 or any(type(value) is not int for value in original):
            raise ValueError("source-supported lengths require integer range pairs")
        minimum, original_upper = original
        if not 0 < minimum <= original_upper <= requested:
            raise ValueError("source-supported output cannot be shorter than its original range")
        rows = []
        for device in devices:
            indices = device.corpus.representative_indices(split, original_upper).tolist()
            if not indices:
                raise ValueError("source-supported device has no representative cells")
            for index in indices:
                record = device.corpus.records[index]
                interval = getattr(record, split + "_interval")
                if interval is None or len(interval) != 2 or not 0 <= interval[0] < interval[1] <= len(record.signal):
                    raise ValueError("source representative interval is invalid")
                available = int(interval[1] - interval[0])
                if available < original_upper:
                    raise ValueError("source representative does not support the original range")
                rows.append({"device_id": device.device_id, "record_index": int(index), "record_id": record.record.record_id,
                             "fault_id": int(record.fault_id), "condition_label": record.record.condition_label,
                             "interval": [int(value) for value in interval], "available_length": available})
        upper = min(requested, min(row["available_length"] for row in rows))
        for device in devices:
            previous = device.corpus.representative_indices(split, original_upper).tolist()
            current = device.corpus.representative_indices(split, upper).tolist()
            if current != previous:
                raise ValueError("source-supported extension switched representative records")
        splits[split] = {"original_range": list(original), "range": [minimum, upper], "representative_count": len(rows),
                         "representatives": rows, "records_switched_or_cells_dropped": False}
    return {"policy": SOURCE_SUPPORTED_COMMON_RANGE, "requested_output_length": requested, "splits": splits,
            "rule": "minimum_requested_output_and_all_original_representative_intervals_per_split",
            "per_device_ranges": False, "extra_runtime_parameters_or_files": False}


def source_layout(devices: Sequence[Any]) -> list[dict]:
    return [{"device_id": device.device_id, "run_id": device.run_id, "split_sha256": device.split_sha256,
             "records": [{"record_id": record.record.record_id, "fault_id": int(record.fault_id),
                          "condition_label": record.record.condition_label, "sample_count": len(record.signal),
                          "train_interval": None if record.train_interval is None else list(record.train_interval),
                          "validation_interval": None if record.validation_interval is None else list(record.validation_interval),
                          "signal_sha256": hashlib.sha256(record.signal.tobytes()).hexdigest()}
                         for record in device.corpus.records]} for device in devices]
