from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


SHIFTED_BASE2_CROP = "record_local_shifted_base2_crop_v1"
PREFIX = "source_crop_"


def source_crop_policy(generator: Mapping) -> str:
    policy = generator.get("source_crop_policy", "none")
    if policy not in ("none", SHIFTED_BASE2_CROP):
        raise ValueError("unknown source crop policy")
    return policy


def source_crop_metadata(config: Mapping) -> dict:
    generator = config["generator"]
    if source_crop_policy(generator) != SHIFTED_BASE2_CROP:
        raise ValueError("source crop metadata requires its active policy")
    if any(generator.get(name, False) for name in ("relation_balanced_sampling", "relation_coupled_noise_time", "device_balanced_sampling", "resumable_training", "solver_state_policy")):
        raise ValueError("source crop cannot combine undeclared source sampling or solver policies")
    seed = config.get("split_seed", config["seed"])
    if type(seed) is not int or seed < 0:
        raise ValueError("source crop requires the original nonnegative data seed")
    return {"policy": SHIFTED_BASE2_CROP, "data_seed": seed, "sequence_bits": 32, "visit_origin": 0,
            "identity": "compact_sorted_json_seed_device_record_utf8", "shift": "sha256_first4_bytes_little_endian",
            "code": "reverse32(visit)_xor_shift", "parent_start": "train_begin+((code*valid_starts)>>32)",
            "legacy_parent_start_draw_consumed": True, "original_view_offset_draw_preserved": True,
            "source_validation": "unchanged_original_random_crops", "training_cell_and_length_schedule_unchanged": True,
            "extra_random_draws": 0, "additional_source_waveforms": 0, "additional_model_forwards": 0,
            "nonoverlapping_or_unique_training_points_claimed": False, "extra_runtime_parameters_or_files": False}


def reverse32(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("source crop visit must fit unsigned32")
    result = 0
    for _ in range(32):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def record_shift(seed: int, device_id: str, record_id: str) -> int:
    if type(seed) is not int or seed < 0 or not isinstance(device_id, str) or not device_id or not isinstance(record_id, str) or not record_id:
        raise ValueError("source crop requires explicit source record identity")
    encoded = json.dumps({"seed": seed, "device": device_id, "record": record_id}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")


class SourceCropSchedule:
    def __init__(self, metadata: dict, entries: Sequence[dict]):
        self.metadata = copy.deepcopy(metadata)
        self.entries = copy.deepcopy(list(entries))
        keys = [(row["device"], row["record"]) for row in self.entries]
        if not keys or keys != sorted(set(keys)):
            raise ValueError("source crop requires sorted unique eligible source records")
        self.lookup = dict(zip(keys, self.entries, strict=True))
        for row in self.entries:
            if any(type(row[name]) is not int or row[name] < 0 for name in ("device", "record", "train_begin", "train_end", "shift")):
                raise ValueError("source crop source metadata must be nonnegative integer indices")
            if row["train_end"] <= row["train_begin"] or row["shift"] != record_shift(metadata["data_seed"], row["device_id"], row["record_id"]):
                raise ValueError("source crop interval or identity shift disagrees")
        self.visits = dict.fromkeys(keys, 0)
        self.rows = []
        self.pending = None
        self.totals = {"samples": 0, "elements": 0}

    @classmethod
    def from_source_cells(cls, devices: Sequence[Any], cells: Sequence[tuple[int, int]], config: Mapping):
        metadata = source_crop_metadata(config)
        if [device.index for device in devices] != list(range(len(devices))) or len(set(cells)) != len(cells):
            raise ValueError("source crop requires contiguous device routes and unique source cells")
        entries = []
        for device_index, record_index in sorted(cells):
            if not 0 <= device_index < len(devices) or not 0 <= record_index < len(devices[device_index].corpus.records):
                raise ValueError("source crop source cell is out of bounds")
            device = devices[device_index]
            record = device.corpus.records[record_index]
            if record.train_interval is None:
                raise ValueError("source crop cannot use records without a training interval")
            entries.append({"device": int(device_index), "record": int(record_index), "device_id": device.device_id,
                "record_id": record.record.record_id, "train_begin": int(record.train_interval[0]), "train_end": int(record.train_interval[1]),
                "shift": record_shift(metadata["data_seed"], device.device_id, record.record.record_id)})
        return cls(metadata, entries)

    def prepare(self, cells: Sequence[tuple[int, int]], length: int, step: int) -> np.ndarray:
        if self.pending is not None or type(step) is not int or step != len(self.rows) + 1 or type(length) is not int or length <= 0 or not cells:
            raise ValueError("source crop needs one consecutive pending batch of positive length")
        if any(tuple(cell) not in self.lookup or self.lookup[tuple(cell)]["train_end"] - self.lookup[tuple(cell)]["train_begin"] < length for cell in cells):
            raise ValueError("source crop selected a non-source or insufficient training interval")
        selected = []
        for device, record in cells:
            key = (int(device), int(record))
            entry = self.lookup[key]
            visit = self.visits[key]
            code = reverse32(visit) ^ entry["shift"]
            count = entry["train_end"] - entry["train_begin"] - length + 1
            start = entry["train_begin"] + ((code * count) >> 32)
            selected.append({"device": key[0], "record": key[1], "visit": visit, "code": code, "start": start,
                             "parent_id": f"{entry['record_id']}:train:{start}:{length}"})
            self.visits[key] += 1
        ordered = sorted(enumerate(selected), key=lambda item: (item[1]["device"], item[0]))
        self.pending = {"step": step, "length": length, "selected": [row for _, row in ordered]}
        return np.asarray([row["code"] for row in selected], dtype=np.uint32)

    def record(self, batch: Any, step: int) -> dict:
        if self.pending is None or self.pending["step"] != step or step != len(self.rows) + 1:
            raise ValueError("source crop record lacks its matching pending source batch")
        selected = self.pending["selected"]
        length = self.pending["length"]
        expected_routes = [row["device"] for row in selected]
        expected_records = [row["record"] for row in selected]
        expected_ids = [row["parent_id"] for row in selected]
        if batch.source_record_index is None or batch.device_id.tolist() != expected_routes or batch.source_record_index.tolist() != expected_records or batch.parent_segment_id.tolist() != expected_ids:
            raise ValueError("source crop builder did not execute the prescribed record starts")
        if batch.normalized_signal.shape != (len(selected), 1, length) or batch.normalized_signal.dtype != np.float32 or not np.isfinite(batch.normalized_signal).all() or not np.all(batch.normalization_length == length):
            raise ValueError("source crop builder changed source shape, precision or normalization length")
        row = {"step": step, PREFIX + "executed": 1.0, PREFIX + "samples": len(selected), PREFIX + "length": length,
               PREFIX + "elements": int(batch.normalized_signal.size), PREFIX + "routes": expected_routes, PREFIX + "records": expected_records,
               PREFIX + "visits": [item["visit"] for item in selected], PREFIX + "codes": [item["code"] for item in selected],
               PREFIX + "starts": [item["start"] for item in selected], PREFIX + "parent_ids": expected_ids,
               PREFIX + "clean_sha256": hashlib.sha256(batch.normalized_signal.tobytes()).hexdigest()}
        return self.complete(row)

    def complete(self, row: dict) -> dict:
        if self.pending is None or row.get("step") != self.pending["step"] or type(row.get("step")) is not int:
            raise ValueError("source crop trace lacks its matching pending batch")
        step = row["step"]
        selected, length = self.pending["selected"], self.pending["length"]
        expected = {"step": step, PREFIX + "executed": 1.0, PREFIX + "samples": len(selected), PREFIX + "length": length,
                    PREFIX + "elements": len(selected) * length,
                    **{PREFIX + field: [item[key] for item in selected] for field, key in (("routes", "device"), ("records", "record"), ("visits", "visit"), ("codes", "code"), ("starts", "start"), ("parent_ids", "parent_id"))}}
        if set(row) != set(expected) | {PREFIX + "clean_sha256"} or any(row[name] != value for name, value in expected.items()):
            raise ValueError("source crop trace differs from the deterministic source plan")
        fingerprint = row[PREFIX + "clean_sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("source crop requires a complete source waveform fingerprint")
        self.rows.append(copy.deepcopy(row))
        self.pending = None
        for field in ("samples", "elements"):
            self.totals[field] += row[PREFIX + field]
        return {**{name: value for name, value in row.items() if name != "step"}, PREFIX + "total_steps": step,
                **{PREFIX + "total_" + name: value for name, value in self.totals.items()}}

    def audit(self) -> dict:
        if self.pending is not None:
            raise ValueError("source crop cannot archive an uncompleted pending batch")
        return {"metadata": copy.deepcopy(self.metadata), "entries": copy.deepcopy(self.entries), "completed_steps": len(self.rows),
                "steps": copy.deepcopy(self.rows), "record_visits": [{"device": device, "record": record, "visits": count} for (device, record), count in sorted(self.visits.items())],
                "historical_student_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}
