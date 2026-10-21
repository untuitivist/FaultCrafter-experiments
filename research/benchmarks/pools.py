from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np

from .common import identity, read_json, seed_for, write_json
from .data import CHANNELS, PARENT, STRIDE, T4_STATES, WINDOW, parent_windows, raw_cache_path, select_balanced
from .tasks import label_dictionary
from .recipes import CNN_RECIPE, classifier_window_count


def source_pool(run, job):
    return cached_source_pool(run, job["device"], job["channel"], job["target"], job["split"])


@lru_cache(maxsize=2)
def cached_source_pool(run, device, channel, target, split):
    job = {"device": device, "channel": channel, "target": target, "split": split}
    records = read_json(run / "inventory.json")["records"]
    ready = read_json(run / "data_ready.json")
    manifest = read_json(run / "cache/windows" / str(job["split"]) / "manifest.json")
    labels, components = label_dictionary(records, job["device"])
    allowed = sorted(set(labels) - set(T4_STATES.get(job["device"], [])))
    channel_index = CHANNELS[job["device"]].index(job["channel"])
    payload = {"components": components, "states": allowed, "labels": labels, "roles": {}}
    for role in ("train", "validation"):
        signals, targets, units, states, sample_ids = [], [], [], [], []
        cells = [cell for cell in manifest["cells"] if cell["role"] == role and cell["device"] == job["device"]
                 and cell["condition"] != job["target"] and cell["state"] in allowed]
        conditions = sorted({cell["condition"] for cell in cells})
        expected = {(condition, state) for condition in conditions for state in allowed}
        if {(cell["condition"], cell["state"]) for cell in cells} != expected:
            raise RuntimeError("Incomplete legal source Cartesian product")
        for unit, cell in enumerate(sorted(cells, key=lambda item: (item["state"], item["condition"]))):
            folder = run / "cache/windows" / str(job["split"])
            count = ready["quotas"][role]
            values = np.load(folder / cell["file"], mmap_mode="r")[:count, channel_index:channel_index + 1]
            valid = np.load(folder / cell["file"].replace(".npy", "_valid.npy"), mmap_mode="r")[:count, channel_index]
            if len(values) != count or not valid.all():
                raise RuntimeError(f"Invalid source windows: {cell['file']}")
            signals.append(np.array(values))
            vector = np.asarray([part in labels[cell["state"]] for part in components], dtype=np.float32)
            targets.append(np.repeat(vector[None], count, axis=0))
            units.extend([unit] * count)
            states.extend([allowed.index(cell["state"])] * count)
            sample_ids.extend(cell["samples"][:count])
        payload["roles"][role] = {"x": np.concatenate(signals), "y": np.concatenate(targets),
                                  "unit": np.asarray(units), "state": np.asarray(states),
                                  "conditions": conditions, "samples": sample_ids}
    payload["source_identity"] = identity({"job": {key: job[key] for key in ("device", "channel", "target", "split")},
                                           "quotas": ready["quotas"], "inventory": ready["inventory_identity"],
                                           "train": payload["roles"]["train"]["samples"],
                                           "validation": payload["roles"]["validation"]["samples"]})
    return payload


def update_budget(pool, job):
    states = len(pool["states"])
    additional = (252 if job["track"] == "T3" else 63) * states
    if job["track"] == "T4":
        additional = 63 * len(T4_STATES[job["device"]])
    count = classifier_window_count(len(pool["roles"]["train"]["x"]), additional, job)
    per_epoch = math.ceil(count / CNN_RECIPE["batch_size"])
    return per_epoch, CNN_RECIPE["max_epochs"] * per_epoch, additional


def augmentation_pool(pool, job, count):
    source = pool["roles"]["train"]
    generator = np.random.default_rng(seed_for(job["device"], job["channel"], job["target"], job["split"], "augmentation"))
    per_state = count // len(pool["states"])
    output, labels = [], []
    for state in range(len(pool["states"])):
        candidates = np.flatnonzero(source["state"] == state)
        units = sorted(set(source["unit"][candidates]))
        for index in range(per_state):
            indices = candidates[source["unit"][candidates] == units[index % len(units)]]
            selected = int(generator.choice(indices))
            signal = source["x"][selected].copy()
            transform = int(generator.integers(3))
            if transform == 0:
                signal *= generator.uniform(0.9, 1.1)
            elif transform == 1:
                rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
                signal += generator.normal(0, rms * 10 ** (-generator.uniform(20, 30) / 20), signal.shape).astype(np.float32)
            else:
                signal = np.roll(signal, int(generator.integers(WINDOW)), axis=-1)
            output.append(signal)
            labels.append(source["y"][selected])
    return np.stack(output), np.stack(labels)


def superposition_pool(pool, job):
    source = pool["roles"]["train"]
    generator = np.random.default_rng(seed_for(job["device"], job["channel"], job["target"], job["split"], "superposition"))
    output, targets, identities = [], [], []
    conditions = source["conditions"]
    for state in T4_STATES[job["device"]]:
        parts = pool["labels"][state]
        singles = [next(name for name in pool["states"] if pool["labels"][name] == [part]) for part in parts]
        for index in range(63):
            condition_index = index % len(conditions)
            selected = []
            for name in singles:
                state_index = pool["states"].index(name)
                candidates = np.flatnonzero((source["state"] == state_index) & (source["unit"] % len(conditions) == condition_index))
                selected.append(int(generator.choice(candidates)))
            signal = source["x"][selected].sum(axis=0)
            scale = float(np.abs(signal).max())
            if not np.isfinite(scale) or scale == 0:
                raise RuntimeError("Invalid superposition waveform")
            output.append(signal / scale)
            targets.append(np.asarray([part in parts for part in pool["components"]], dtype=np.float32))
            identities.append(selected)
    return np.stack(output), np.stack(targets), {"rule": "Sum normalized source single-fault windows at matched source condition; normalize combined parent.",
                                               "indices": identities}


@lru_cache(maxsize=1)
def target_quota(run):
    records = read_json(run / "inventory.json")["records"]
    lengths = defaultdict(int)
    for item in records:
        receipt = read_json(raw_cache_path(run, item).with_suffix(".json"))
        length = receipt.get("shape", [item["samples"]])[0]
        if item["extra"].get("raw_sample_rate") == 48000:
            length = (length + 3) // 4 - 20
        count = sum(max(0, 1 + (min(PARENT, length - start) - WINDOW) // STRIDE) for start in range(0, length, PARENT))
        lengths[item["device"], item["condition"], item["state"]] += count
    return min(72, min(lengths.values()))


def seen_pool(run, job, pool, frozen_sha256, *, output_folder=None):
    if not frozen_sha256:
        raise RuntimeError("Seen evaluation requires frozen classifier")
    records = read_json(run / "inventory.json")["records"]
    by_id = {item["id"]: item for item in records}
    manifest = read_json(run / "cache/windows" / str(job["split"]) / "manifest.json")
    quota = read_json(run / "data_ready.json")["quotas"]["seen"]
    channel = CHANNELS[job["device"]].index(job["channel"])
    cells = []
    for cell in manifest["cells"]:
        if cell["role"] != "seen" or cell["device"] != job["device"] or cell["condition"] == job["target"]:
            continue
        folder = run / "cache/windows" / str(job["split"])
        values = np.load(folder / cell["file"], mmap_mode="r")[:quota, channel:channel + 1]
        valid = np.load(folder / cell["file"].replace(".npy", "_valid.npy"), mmap_mode="r")[:quota, channel]
        cells.append((cell["condition"], cell["state"], np.array(values), np.array(valid), cell["samples"][:quota]))
    if job["track"] == "T4":
        blocked = defaultdict(list)
        for partition in read_json(run / "partitions" / f"{job['split']}.json"):
            item = by_id[partition["record"]]
            if (item["device"] != job["device"] or item["state"] not in T4_STATES[job["device"]]
                    or item["condition"] == job["target"] or partition["role"] != "seen"):
                continue
            values = np.load(raw_cache_path(run, item), mmap_mode="r")[partition["start"]:partition["stop"], channel:channel + 1]
            windows, validity, starts = parent_windows(values, resample=False)
            blocked[item["condition"], item["state"]].append((windows, validity,
                                                            {**partition, "group": item["group"], "starts": starts}))
        for (condition, state), blocks in sorted(blocked.items()):
            generator = np.random.default_rng(seed_for(job["device"], condition, state, job["split"], "seen_t4"))
            windows, valid, samples = select_balanced(blocks, quota, generator)
            if len(windows) != quota:
                raise RuntimeError("T4 seen-condition capacity below frozen quota")
            cells.append((condition, state, windows, valid[:, 0], samples))
    signals, targets, valid, groups, receipts = [], [], [], [], []
    for condition, state, values, validity, sample_ids in cells:
        signals.append(values)
        targets.extend([[part in pool["labels"][state] for part in pool["components"]]] * len(values))
        valid.extend(validity.tolist())
        groups.extend([condition + "/" + state] * len(values))
        receipts.append({"condition": condition, "state": state, "samples": sample_ids})
    output_folder = output_folder or run / "jobs" / job["id"]
    write_json(output_folder / "seen_manifest.json", {"classifier_sha256": frozen_sha256, "cells": receipts})
    return np.concatenate(signals), np.asarray(targets, dtype=np.float32), np.asarray(valid), groups


def target_pool(run, job, pool, frozen_sha256, *, output_folder=None):
    if not frozen_sha256:
        raise RuntimeError("Target access requires frozen classifier identity")
    records = read_json(run / "inventory.json")["records"]
    selected = [item for item in records if item["device"] == job["device"] and item["condition"] == job["target"]
                and (item["state"] in pool["states"] or job["track"] == "T4")]
    quota = target_quota(run)
    channel_index = CHANNELS[job["device"]].index(job["channel"])
    output, labels, validity, groups, sample_ids = [], [], [], [], []
    for state in sorted({item["state"] for item in selected}):
        blocks = []
        for item in selected:
            if item["state"] != state:
                continue
            receipt = read_json(raw_cache_path(run, item).with_suffix(".json"))
            resample = item["extra"].get("raw_sample_rate") == 48000
            if receipt["status"] == "ready":
                values = np.load(raw_cache_path(run, item), mmap_mode="r")
                windows, valid, starts = parent_windows(values, resample=resample)
                windows, valid = windows[:, channel_index:channel_index + 1], valid[:, channel_index:channel_index + 1]
            else:
                length = item["samples"]
                starts = [parent + start for parent in range(0, length, PARENT)
                          for start in range(0, min(PARENT, length - parent) - WINDOW + 1, STRIDE)]
                windows = np.zeros((len(starts), 1, WINDOW), dtype=np.float32)
                valid = np.zeros((len(starts), 1), dtype=bool)
            blocks.append((windows, valid, {"record": item["id"], "group": item["group"], "starts": starts}))
        generator = np.random.default_rng(seed_for(job["device"], job["target"], state, job["split"], "target_windows"))
        windows, valid, ids = select_balanced(blocks, quota, generator)
        if len(windows) != quota:
            raise RuntimeError("Target capacity below frozen quota")
        output.append(windows)
        labels.extend([[part in pool["labels"][state] for part in pool["components"]]] * quota)
        validity.extend(valid[:, 0].tolist())
        groups.extend([state] * quota)
        sample_ids.extend(ids)
    receipt = {"classifier_sha256": frozen_sha256, "quota_per_state": quota, "samples": sample_ids,
               "valid": validity, "states": groups}
    output_folder = output_folder or run / "jobs" / job["id"]
    write_json(output_folder / "target_manifest.json", receipt)
    return np.concatenate(output), np.asarray(labels, dtype=np.float32), np.asarray(validity), groups
