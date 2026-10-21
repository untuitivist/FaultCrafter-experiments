from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .common import identity, log, read_json, seed_for, write_json
from .data import CHANNELS, PARENT, T4_STATES, WINDOW, index_partition, materialize_indexed, parent_windows, partition_values, raw_cache_path, select_indexed


def verify(run, plan):
    records = read_json(run / "inventory.json")["records"]
    ready = read_json(run / "data_ready.json")
    by_id = {item["id"]: item for item in records}
    counts = Counter((job["kind"], job["track"], job["method"]) for job in plan["jobs"])
    for method in ("real_only", "augmentation", "mixstyle"):
        if counts["cnn", "T1", method] != 960 or counts["cnn", "T2", method] != 170:
            raise RuntimeError("Comparison matrix differs from registered protocol")
    examples = []
    for device in CHANNELS:
        item = next(record for record in records if record["device"] == device and record["state"] not in T4_STATES.get(device, []))
        receipt = read_json(raw_cache_path(run, item).with_suffix(".json"))
        partition = {"record": item["id"], "start": 0, "stop": receipt["shape"][0], "role": "train"}
        block = index_partition(run, item, partition, receipt)
        selected = select_indexed([block], 7, np.random.default_rng(seed_for("preflight", device)), "train")
        actual, valid, sample_ids = materialize_indexed(run, [block], selected, by_id, "train")
        expected, expected_valid, starts = parent_windows(partition_values(run, item, partition), resample=False)
        positions = [starts.index(sample["relative_start"]) for sample in sample_ids]
        if not np.array_equal(actual, expected[positions]) or not np.array_equal(valid, expected_valid[positions]):
            raise RuntimeError(f"Streamed normalization differs from reference: {device}")
        examples.append({"device": device, "record": item["id"], "windows": len(selected), "reference_equal": True})
    for split in (42, 43, 44, 45, 46):
        partitions = read_json(run / "partitions" / f"{split}.json")
        ranges = defaultdict(list)
        for partition in partitions:
            item = by_id[partition["record"]]
            if item["state"] in T4_STATES.get(item["device"], []):
                continue
            factor = 4 if item["extra"].get("raw_sample_rate") == 48000 else 1
            start, stop = partition["start"] / factor, partition["stop"] / factor
            for role, previous_start, previous_stop in ranges[item["group"]]:
                if role != partition["role"] and max(start, previous_start) < min(stop, previous_stop):
                    raise RuntimeError(f"Acquisition group crosses partitions: {item['group']}")
            ranges[item["group"]].append((partition["role"], start, stop))
        manifest = read_json(run / "cache/windows" / str(split) / "manifest.json")
        if any(cell["state"] in T4_STATES.get(cell["device"], []) for cell in manifest["cells"]):
            raise RuntimeError("Held-out fault combination entered source cache")
        for cell in manifest["cells"]:
            if cell["count"] < ready["quotas"][cell["role"]]:
                raise RuntimeError("Cell below common quota")
    result = {"reference_windows": examples, "split_overlap_checks": 5, "t4_source_exclusion": True,
              "matrix_counts": {"/".join(key): value for key, value in counts.items()}, "identity": identity(plan)}
    write_json(run / "preflight.json", result)
    log("preflight_pass", devices=len(examples), splits=5, source_windows_match_reference=True)
    return result
