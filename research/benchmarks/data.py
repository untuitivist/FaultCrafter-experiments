from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import firwin, resample_poly

from faultcrafter.data.cwru import CWRUAdapter, _layout
from faultcrafter.data.hitsm import HITSMAdapter

from .common import DATA_ROOT, ROOT, SPLITS, canonical, file_hash, identity, log, read_json, seed_for, write_json


CHANNELS = {
    "hdu": ["motor", "bearing_base", "bearing_upper", "bearing_side"],
    "dirg": ["A1_x", "A1_y", "A1_z", "A2_x", "A2_y", "A2_z"],
    "hust": ["X", "Y", "Z"], "cwru": ["drive_end", "fan_end"],
    "bjtu": ["CH17", "CH18", "CH19"], "hit_sq": ["vibration"],
    "hit_self": ["vibration"], "pu": ["vibration_1"],
}
T4_STATES = {"hdu": ["I+O+B"], "hust": ["C", "0.5X_C"],
             "bjtu": ["LA1+LA2+LA3", "LA1+LA2+LA4", "LA1+LA2+LA3+LA4"], "pu": ["inner_race+outer_race"]}
GUARD = 1024
WINDOW = 1024
STRIDE = 819
PARENT = 51802


def condition_name(values: list[float]) -> str:
    return "_".join(f"{value:.12g}" for value in values)


def record(device, path, state, components, coordinates, sample_rate, samples, *, reader, keys=None, group=None, checksum=None, extra=None):
    stat = path.stat()
    return {"device": device, "path": path.as_posix(), "state": state, "components": components,
            "condition": condition_name(coordinates), "coordinates": coordinates,
            "sample_rate": sample_rate, "samples": samples, "channels": CHANNELS[device],
            "reader": reader, "keys": keys, "group": group or path.as_posix(),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "receipt_sha256": checksum,
            "extra": extra or {}, "id": identity([device, path.as_posix()])[:24]}


def discover() -> list[dict]:
    records = []
    hdu_root = DATA_ROOT / "PT500min"
    receipts = {entry["path"]: entry for entry in read_json(hdu_root / "inventory.json")}
    seen = set()
    for metadata in read_json(hdu_root / "content_inventory.json")["records"]:
        relative = metadata["path"]
        checksum = receipts[relative]["sha256"]
        if checksum in seen:
            continue
        seen.add(checksum)
        state = relative.split("/")[0]
        speed, force = map(int, re.findall(r"rpm(\d+)_f(\d+)", relative)[0])
        variables = [value for value in metadata["variables"] if len(value[1]) == 2 and value[1][1] in (4, 5)]
        key, shape, _ = max(variables, key=lambda value: value[1][0])
        records.append(record("hdu", hdu_root / "dataset" / relative, state, [] if state == "N" else state.split("+"),
                              [speed / 60, force], 10000, shape[0], reader="hdu", keys=[key], group=checksum,
                              checksum=checksum, extra={"columns": shape[1]}))
    for path in sorted((DATA_ROOT / "DIRG/dataset").rglob("C?A_*.mat")):
        match = re.fullmatch(r"(C\dA)_(\d+)_(\d+)_(\d+)", path.stem)
        state, frequency, voltage, _ = match.groups()
        force = min((0, 1000, 1400, 1800), key=lambda value: abs(value - int(voltage) / 0.499))
        if abs(force - int(voltage) / 0.499) > 100:
            raise ValueError(f"Unrecognized DIRG load: {path}")
        key, shape, _ = whosmat(path)[0]
        records.append(record("dirg", path, state, [] if state == "C0A" else [state], [int(frequency), force],
                              51200, shape[0], reader="matrix", keys=[key], extra={"load_mv": int(voltage)}))
    for path in sorted((DATA_ROOT / "HUSTbearing/dataset").rglob("*.xls")):
        match = re.fullmatch(r"(H|(?:0\.5X_)?[IOBC])_(\d+)Hz", path.stem, flags=re.IGNORECASE)
        if not match:
            continue
        state, frequency = match.groups()
        suffix = "_medium" if state.startswith("0.5X_") else "_severe"
        kind = state.removeprefix("0.5X_")
        components = [] if kind == "H" else (["I" + suffix, "O" + suffix] if kind == "C" else [kind + suffix])
        records.append(record("hust", path, state, components, [int(frequency)], 25600, 262144, reader="hust"))
    cwru_records = CWRUAdapter(DATA_ROOT / "CWRU").discover()
    states_by_condition = defaultdict(set)
    for item in cwru_records:
        if item.fault_parameters and dict(item.fault_parameters).get("fault_diameter_in") == 0.028:
            continue
        prefix, available, _ = _layout(item.path)
        if not {"drive_end", "fan_end"} <= set(available):
            continue
        states_by_condition[item.condition_label].add(item.fault_label)
        hp = dict(item.condition_parameters)["motor_load_hp"]
        speed = dict(item.condition_parameters)["shaft_speed_rpm"]
        records.append(record("cwru", item.path, item.fault_label,
                              [] if item.fault_label == "normal" else [item.fault_label],
                              [speed / 60, hp * 745.6998715822702], 12000, item.num_samples,
                              reader="cwru", keys=[f"{prefix}_DE_time", f"{prefix}_FE_time"],
                              group=f"cwru/{item.fault_label}/{item.condition_label}",
                              extra={"raw_sample_rate": item.original_sample_rate_hz}))
    common_states = set.intersection(*states_by_condition.values())
    if len(common_states) != 24:
        raise ValueError(f"Expected 24 common CWRU states, found {len(common_states)}")
    records = [item for item in records if item["device"] != "cwru" or item["state"] in common_states]
    for item in HITSMAdapter(DATA_ROOT / "Bearing-datasets").discover():
        device = "hit_sq" if "spectraquest" in item.device_id else "hit_self"
        coordinates = [dict(item.condition_parameters)["shaft_speed_rpm"] / 60]
        if device == "hit_sq":
            coordinates.append(49.03325)
        records.append(record(device, item.path, item.fault_label, [] if item.fault_label == "normal" else [item.fault_label],
                              coordinates, 51200, item.num_samples, reader="matrix", keys=[item.channels[0].source_key]))
    bjtu_states = {"LA0": [], **{f"LA{number}": [f"LA{number}"] for number in range(1, 5)},
                   **{state: state.split("+") for state in ["LA1+LA2", "LA2+LA3", "LA2+LA4", *T4_STATES["bjtu"]]}}
    for path in sorted((DATA_ROOT / "BJTU-RAO Bogie Datasets").rglob("data_leftaxlebox_*.csv")):
        match = re.fullmatch(r"data_leftaxlebox_M0_G0_(LA[^_]+)_RA0_(20|40|60)Hz_([+-]?10|0)kN", path.stem)
        if not match or match[1] not in bjtu_states:
            continue
        state, frequency, force = match.groups()
        records.append(record("bjtu", path, state, bjtu_states[state], [int(frequency), int(force) * 1000, 10000],
                              64000, 640000, reader="bjtu"))
    pu_scope = read_json(ROOT / "research/protocols/paderborn_full_scope_v10.json")
    pu_states = {bearing: state for state, bearings in pu_scope["selected"]["groups"].items() for bearing in bearings}
    pu_states.update({bearing: "inner_race+outer_race" for bearing in pu_scope["excluded_from_current_task"]})
    for path in sorted((DATA_ROOT / "Paderborn/dataset").rglob("*.mat")):
        match = re.fullmatch(r"(N\d+_M\d+_F\d+)_(K[A-Z]?\d+)_(\d+)", path.stem)
        condition, bearing, _ = match.groups()
        state = pu_states[bearing]
        records.append(record("pu", path, state, [] if state == "normal" else state.split("+"),
                              pu_scope["conditions"][condition], 64000, 256000, reader="pu", group=bearing))
    expected = {"hdu": (13, 10), "dirg": (17, 7), "hust": (10, 9), "cwru": (4, 24),
                "bjtu": (9, 11), "hit_sq": (3, 7), "hit_self": (3, 7), "pu": (4, 4)}
    for device, counts in expected.items():
        subset = [item for item in records if item["device"] == device]
        observed = len({item["condition"] for item in subset}), len({item["state"] for item in subset})
        if counts != observed:
            raise ValueError(f"Incomplete {device}: expected {counts}, found {observed}")
        cells = {(item["state"], item["condition"]) for item in subset}
        if len(cells) != counts[0] * counts[1]:
            raise ValueError(f"Missing state-condition cells in {device}")
    return records


def read_waveform(item: dict) -> np.ndarray:
    path = Path(item["path"])
    stat = path.stat()
    if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
        raise RuntimeError(f"Raw file changed since inventory: {path}")
    if item["reader"] == "hust":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip() == "Data":
                    break
            values = np.loadtxt(stream, usecols=(2, 3, 4), dtype=np.float32)
    elif item["reader"] == "bjtu":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            names = next(csv.reader(stream))
            columns = [names.index(name) for name in CHANNELS["bjtu"]]
            values = np.loadtxt(stream, delimiter=",", usecols=columns, dtype=np.float32)
    elif item["reader"] == "pu":
        payload = loadmat(path, simplify_cells=True)
        contents = next(value for key, value in payload.items() if not key.startswith("__"))
        candidates = [entry for entry in contents["Y"] if str(entry["Name"]) == "vibration_1"]
        if len(candidates) != 1:
            raise ValueError(f"PU vibration_1 unavailable: {path}")
        values = np.asarray(candidates[0]["Data"], dtype=np.float32).reshape(-1, 1)
    else:
        payload = loadmat(path, variable_names=item["keys"])
        if item["reader"] == "cwru":
            values = np.column_stack([np.asarray(payload[key]).reshape(-1) for key in item["keys"]]).astype(np.float32)
        else:
            values = np.asarray(payload[item["keys"][0]])
            if values.ndim == 1 or min(values.shape) == 1:
                values = values.reshape(-1, 1)
            if item["reader"] == "hdu" and values.shape[1] == 5:
                time_axis = values[:, 0]
                differences = np.diff(time_axis.astype(np.float64))
                interval = np.median(differences)
                seconds = interval * (86400 if time_axis[0] > 1000 else 1)
                if not np.all(differences > 0) or not np.isclose(seconds, 1 / item["sample_rate"], rtol=0.15):
                    raise ValueError(f"HDU first column is not a verified time axis: {path}")
                values = values[:, 1:]
    if values.shape[1] != len(item["channels"]) or not np.isfinite(values).all():
        raise ValueError(f"Invalid waveform shape or values: {path}")
    return np.ascontiguousarray(values, dtype=np.float32)


def raw_cache_path(run: Path, item: dict) -> Path:
    return run / "cache/raw" / item["device"] / (item["id"] + ".npy")


def cache_record(run: Path, item: dict) -> dict:
    destination = raw_cache_path(run, item)
    receipt = destination.with_suffix(".json")
    fingerprint = identity(item)
    if receipt.exists() and destination.exists() and read_json(receipt).get("fingerprint") == fingerprint:
        return read_json(receipt)
    try:
        values = read_waveform(item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, destination)
        result = {"id": item["id"], "status": "ready", "fingerprint": fingerprint, "shape": list(values.shape),
                  "raw_sha256": item["receipt_sha256"] or file_hash(Path(item["path"])),
                  "waveform_sha256": hashlib.sha256(memoryview(values)).hexdigest()}
    except Exception as error:
        result = {"id": item["id"], "status": "failed", "fingerprint": fingerprint,
                  "path": item["path"], "error": f"{type(error).__name__}: {error}"}
    write_json(receipt, result)
    return result


def intervals(length: int, resample: bool) -> dict[str, tuple[int, int]]:
    edge = 20 if resample else 0
    usable = length - 2 * GUARD - 3 * edge
    validation = max(16384, int(0.2 * usable))
    seen = max(7576, int(0.2 * usable))
    training = usable - validation - seen
    if training < 16384:
        raise ValueError(f"Record too short for protected temporal partition: {length}")
    offset = 0
    result = {}
    for role, count in (("train", training), ("validation", validation), ("seen", seen)):
        result[role] = (offset, offset + count + edge)
        offset += count + edge + GUARD
    return result


def build_partitions(records: list[dict], receipts: dict[str, dict], split: int) -> list[dict]:
    cells = defaultdict(list)
    for item in records:
        cells[item["device"], item["condition"], item["state"]].append(item)
    result = []
    for (device, condition, state), members in sorted(cells.items()):
        groups = sorted({item["group"] for item in members if item["samples"] >= 16384})
        if len(groups) >= 5:
            generator = np.random.default_rng(seed_for(device, state, None if device == "pu" else condition, split, "partition"))
            generator.shuffle(groups)
            count = max(1, int(0.2 * len(groups)))
            roles = {group: ("validation" if index < count else "seen" if index < 2 * count else "train")
                     for index, group in enumerate(groups)}
            for item in members:
                length = receipts[item["id"]].get("shape", [item["samples"]])[0]
                result.append({"record": item["id"], "role": roles.get(item["group"], "train"), "start": 0, "stop": length})
        else:
            group_lengths = defaultdict(list)
            for item in members:
                length = receipts[item["id"]].get("shape", [item["samples"]])[0]
                grid_length = (length + 3) // 4 if item["extra"].get("raw_sample_rate") == 48000 else length
                group_lengths[item["group"]].append(grid_length)
            for item in members:
                resample = item["extra"].get("raw_sample_rate") == 48000
                common_length = max(group_lengths[item["group"]])
                own_length = receipts[item["id"]].get("shape", [item["samples"]])[0]
                own_length = (own_length + 3) // 4 if resample else own_length
                reserve_filter_edge = any(member["extra"].get("raw_sample_rate") == 48000
                                          for member in members if member["group"] == item["group"])
                for role, (start, stop) in intervals(common_length, reserve_filter_edge).items():
                    if own_length < stop:
                        if role != "train":
                            continue
                        stop = own_length
                    factor = 4 if resample else 1
                    result.append({"record": item["id"], "role": role, "start": start * factor, "stop": stop * factor})
    return result


def parent_windows(values: np.ndarray, *, resample: bool) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if resample:
        coefficients = firwin(81, 0.25, window=("kaiser", 5.0))
        values = resample_poly(values, 1, 4, axis=0, window=coefficients)[10:-10]
    windows, valid, starts = [], [], []
    for parent_start in range(0, len(values), PARENT):
        parent = values[parent_start:parent_start + PARENT]
        if len(parent) < WINDOW:
            break
        scale = np.max(np.abs(parent), axis=0)
        good = np.isfinite(parent).all(axis=0) & np.isfinite(scale) & (scale > 0)
        normalized = np.where(np.isfinite(parent), parent, 0) / np.where(good, scale, 1)
        for start in range(0, len(parent) - WINDOW + 1, STRIDE):
            windows.append(normalized[start:start + WINDOW].T)
            valid.append(good)
            starts.append(parent_start + start)
    return np.asarray(windows, dtype=np.float32), np.asarray(valid, dtype=np.bool_), starts


def select_balanced(blocks: list[tuple[np.ndarray, np.ndarray, dict]], quota: int, generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    selected = []
    orders = [generator.permutation(len(block[0])).tolist() for block in blocks]
    grouped = defaultdict(list)
    for index, block in enumerate(blocks):
        grouped[block[2]["group"]].append(index)
    groups = list(grouped)
    generator.shuffle(groups)
    while len(selected) < quota:
        progressed = False
        for group in groups:
            available = [index for index in grouped[group] if orders[index]]
            if available and len(selected) < quota:
                index = int(generator.choice(available))
                selected.append((index, orders[index].pop()))
                progressed = True
        if not progressed:
            break
    values = np.stack([blocks[index][0][position] for index, position in selected])
    validity = np.stack([blocks[index][1][position] for index, position in selected])
    identities = [{**{key: value for key, value in blocks[index][2].items() if key != "starts"},
                   "window": int(position), "relative_start": blocks[index][2]["starts"][position]}
                  for index, position in selected]
    return values, validity, identities


def partition_values(run, item, partition):
    values = np.load(raw_cache_path(run, item), mmap_mode="r")[partition["start"]:partition["stop"]]
    if item["extra"].get("raw_sample_rate") == 48000:
        coefficients = firwin(81, 0.25, window=("kaiser", 5.0))
        values = resample_poly(values, 1, 4, axis=0, window=coefficients)[10:-10]
    return values


def index_partition(run, item, partition, receipt):
    values = partition_values(run, item, partition) if receipt["status"] == "ready" else None
    length = len(values) if values is not None else partition["stop"] - partition["start"]
    parents = []
    for start in range(0, length - WINDOW + 1, PARENT):
        count = min(PARENT, length - start)
        if values is None:
            scale = np.ones(len(item["channels"]))
            valid = np.zeros(len(item["channels"]), dtype=bool)
        else:
            parent = values[start:start + count]
            scale = np.max(np.abs(parent), axis=0)
            valid = np.isfinite(parent).all(axis=0) & np.isfinite(scale) & (scale > 0)
        parents.append({"start": start, "length": count, "scale": scale.tolist(), "valid": valid.tolist()})
    return {"record": item["id"], "start": partition["start"], "stop": partition["stop"],
            "group": item["group"], "parents": parents, "readable": values is not None}


def indexed_starts(block, role):
    return [parent["start"] + start for parent in block["parents"]
            if role == "seen" or all(parent["valid"])
            for start in range(0, parent["length"] - WINDOW + 1, STRIDE)]


def select_indexed(blocks, quota, generator, role):
    candidates = [indexed_starts(block, role) for block in blocks]
    orders = [generator.permutation(len(starts)).tolist() for starts in candidates]
    grouped = defaultdict(list)
    for index, block in enumerate(blocks):
        grouped[block["group"]].append(index)
    groups = list(grouped)
    generator.shuffle(groups)
    selected = []
    while len(selected) < quota:
        progressed = False
        for group in groups:
            available = [index for index in grouped[group] if orders[index]]
            if available and len(selected) < quota:
                block_index = int(generator.choice(available))
                position = orders[block_index].pop()
                selected.append((block_index, position, candidates[block_index][position]))
                progressed = True
        if not progressed:
            raise RuntimeError("Index capacity below quota")
    return selected


def materialize_indexed(run, blocks, selected, by_id, role):
    signals = [None] * len(selected)
    validity = [None] * len(selected)
    sample_ids = [None] * len(selected)
    grouped = defaultdict(list)
    for output_index, (block_index, position, start) in enumerate(selected):
        grouped[block_index].append((output_index, position, start))
    for block_index, positions in grouped.items():
        block = blocks[block_index]
        item = by_id[block["record"]]
        values = partition_values(run, item, block) if block["readable"] else None
        for output_index, position, start in positions:
            parent = block["parents"][start // PARENT]
            valid = np.asarray(parent["valid"], dtype=bool)
            scale = np.asarray(parent["scale"], dtype=values.dtype if values is not None else np.float32)
            if values is None:
                signal = np.zeros((len(item["channels"]), WINDOW), dtype=np.float32)
            else:
                signal = np.asarray(values[start:start + WINDOW] / np.where(valid, scale, 1), dtype=np.float32).T
            signals[output_index] = signal
            validity[output_index] = valid
            sample_ids[output_index] = {"record": item["id"], "role": role, "start": block["start"], "stop": block["stop"],
                                        "group": block["group"], "window": position, "relative_start": start}
    return np.stack(signals), np.stack(validity), sample_ids


def prepare(run: Path, workers: int = 4) -> dict:
    inventory_path = run / "inventory.json"
    if inventory_path.exists():
        records = read_json(inventory_path)["records"]
    else:
        records = discover()
        write_json(inventory_path, {"records": records, "identity": identity(records)})
    log("inventory", records=len(records), devices=len(CHANNELS), channels=sum(map(len, CHANNELS.values())))
    receipts = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(cache_record, run, item): item for item in records}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            receipt = future.result()
            receipts[item["id"]] = receipt
            if receipt["status"] != "ready" or index % 100 == 0:
                log("raw_cache", completed=index, total=len(records), latest=receipt)
    failures = [value for value in receipts.values() if value["status"] != "ready"]
    write_json(run / "data_read_errors.json", failures)
    log("raw_cache_complete", failures=len(failures))
    known_readback = read_json(DATA_ROOT / "Paderborn/record_readback_check.json")
    verified_bad_file = Path(known_readback["file"]).resolve()
    unverified_failures = [value for value in failures if Path(value["path"]).resolve() != verified_bad_file
                           or not known_readback["matches_original_archive_member"]
                           or file_hash(Path(value["path"])) != known_readback["archive_member_sha256"]]
    if unverified_failures:
        raise RuntimeError(f"Unverified waveform ingestion failures: {unverified_failures}")
    by_id = {item["id"]: item for item in records}
    index_path = run / "cache/window_index.json"
    catalog = read_json(index_path) if index_path.exists() else {}
    splits = {}
    for split in SPLITS:
        destination = run / "cache/windows" / str(split)
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            splits[split] = read_json(manifest_path)
            if splits[split].get("schema") != "streamed-v1":
                raise RuntimeError("Window manifest schema differs; preserve old cache and choose another run")
            continue
        partitions = build_partitions(records, receipts, split)
        write_json(run / "partitions" / f"{split}.json", partitions)
        cell_blocks = defaultdict(list)
        cell_ids = []
        added_indexes = 0
        for partition_index, partition in enumerate(partitions):
            item = by_id[partition["record"]]
            if item["state"] in T4_STATES.get(item["device"], []):
                continue
            if receipts[item["id"]]["status"] != "ready" and partition["role"] != "seen":
                continue
            cache_key = identity([item["id"], partition["start"], partition["stop"], receipts[item["id"]]["fingerprint"]])
            if cache_key not in catalog:
                catalog[cache_key] = index_partition(run, item, partition, receipts[item["id"]])
                added_indexes += 1
                if added_indexes % 500 == 0:
                    log("index_progress", split=split, added=added_indexes, partitions=len(partitions), position=partition_index)
                    write_json(index_path, catalog)
            key = item["device"], item["condition"], item["state"], partition["role"]
            cell_blocks[key].append(catalog[cache_key])
        write_json(index_path, catalog)
        quotas = {role: min(limit, min(sum(len(indexed_starts(block, role)) for block in blocks) for key, blocks in cell_blocks.items()
                                      if key[3] == role and key[2] not in T4_STATES.get(key[0], [])))
                  for role, limit in (("train", 63), ("validation", 9), ("seen", 9))}
        if min(quotas.values()) < 1:
            raise RuntimeError(f"Empty source cell: {quotas}")
        for key, blocks in sorted(cell_blocks.items()):
            device, condition, state, role = key
            generator = np.random.default_rng(seed_for("windows", device, condition, state, role, split))
            capacity = sum(len(indexed_starts(block, role)) for block in blocks)
            if capacity == 0:
                raise RuntimeError(f"No usable windows: {key}")
            selected = select_indexed(blocks, min(quotas[role], capacity), generator, role)
            values, validity, sample_ids = materialize_indexed(run, blocks, selected, by_id, role)
            name = identity(key)[:24]
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / (name + ".npy"), values, allow_pickle=False)
            np.save(destination / (name + "_valid.npy"), validity, allow_pickle=False)
            cell_ids.append({"device": device, "condition": condition, "state": state, "role": role,
                             "file": name + ".npy", "count": len(values), "samples": sample_ids})
        manifest = {"schema": "streamed-v1", "split": split, "quotas": quotas, "cells": cell_ids, "inventory_identity": identity(records)}
        write_json(manifest_path, manifest)
        splits[split] = manifest
        log("windows_ready", split=split, quotas=quotas, cells=len(cell_ids))
    common_quotas = {role: min(manifest["quotas"][role] for manifest in splits.values()) for role in ("train", "validation", "seen")}
    write_json(run / "data_ready.json", {"inventory_identity": identity(records), "quotas": common_quotas,
                                        "records": len(records), "raw_errors": failures, "splits": list(SPLITS)})
    return records


if __name__ == "__main__":
    import argparse

    from .common import RUN_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    prepare(arguments.run, arguments.workers)
