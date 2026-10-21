from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import SPLITS, T1_DEVICES, file_hash, identity, read_json, timestamp, write_json
from .data import CHANNELS, PARENT, T4_STATES, WINDOW, raw_cache_path
from .tasks import T3_TARGETS


def audit_execution(run, experiment, records, manifests):
    by_id = {record["id"]: record for record in records}
    errors = []
    ready = read_json(run / "data_ready.json")
    generators = 0
    classifiers = 0
    for path in sorted((experiment / "generators").glob("*/source_manifest.json")):
        job = read_json(path.with_name("job.json"))
        source = read_json(path)
        allowed_devices = T1_DEVICES if job["track"] == "T1" else (job["device"],)
        cells = manifests[job["split"]]["cells"]
        legal_samples = {(cell["role"], cell["device"], cell["state"], cell["condition"]):
                         {identity(sample) for sample in cell["samples"]} for cell in cells}
        if job["method"] == "faultcrafter":
            if identity(source) != job["source_identity"]:
                errors.append({"kind": "generator_raw_source_identity", "path": str(path)})
            partitions = read_json(run / "partitions" / f"{job['split']}.json")
            legal_blocks = {(block["record"], block["role"], block["start"], block["stop"]) for block in partitions}
            legal_units = set()
            block_bounds = set()
            members_by_role = {"train": [], "validation": []}
            for block in source["blocks"]:
                record = by_id[block["record"]]
                block_key = (block["record"], block["role"], block["start"], block["stop"])
                if (block_key not in legal_blocks or block["role"] not in members_by_role
                        or record["device"] not in allowed_devices
                        or record["state"] in T4_STATES.get(record["device"], ())
                        or (record["device"] == job["device"] and record["condition"] == job["target"])):
                    errors.append({"kind": "generator_illegal_raw_source", "path": str(path), "block": block_key})
                else:
                    block_bounds.add(block_key)
                    legal_units.add((block["role"], record["device"], record["state"], record["condition"]))
            for role, units in source["units"].items():
                if role not in members_by_role:
                    errors.append({"kind": "generator_source_roles", "path": str(path)})
                    continue
                for device, channel, state, condition in units:
                    if ((role, device, state, condition) not in legal_units
                            or device not in allowed_devices or channel not in CHANNELS[device]):
                        errors.append({"kind": "generator_illegal_raw_unit", "path": str(path), "unit": [role, device, channel, state, condition]})
            batch_path = path.with_name("batches.jsonl")
            if batch_path.exists():
                for line in batch_path.read_text(encoding="utf-8").splitlines():
                    batch = json.loads(line)
                    for parent in batch.get("parents", []):
                        record_id, role, start, stop, channel, crop, length = parent.rsplit(":", 6)
                        record = by_id[record_id]
                        factor = 4 if record["extra"].get("raw_sample_rate") == 48000 else 1
                        usable = (int(stop) - int(start)) // factor - (20 if factor == 4 else 0)
                        if ((record_id, role, int(start), int(stop)) not in block_bounds or role != "train"
                                or channel not in CHANNELS[record["device"]] or int(crop) < 0
                                or int(crop) + int(length) > usable or not 512 <= int(length) <= 16384):
                            errors.append({"kind": "generator_sample_outside_source", "path": str(path), "parent": parent})
        else:
            members_by_role = source
            if set(source) != {"train", "validation"}:
                errors.append({"kind": "generator_source_roles", "path": str(path)})
        for role, members in members_by_role.items():
            for cell in members:
                key = (role, cell["device"], cell["state"], cell["condition"])
                if (cell["device"] not in allowed_devices
                        or cell["state"] in T4_STATES.get(cell["device"], ())
                        or (cell["device"] == job["device"] and cell["condition"] == job["target"])
                        or any(identity(sample) not in legal_samples.get(key, set()) for sample in cell["samples"])):
                    errors.append({"kind": "generator_illegal_source", "path": str(path), "cell": key})
        if job["track"] == "T2" and job.get("strategy") != "scratch":
            parent_folder = experiment / "generators" / job["parent"]
            parent = read_json(parent_folder / "job.json")
            if (parent["track"] != "T1" or parent["device"] != "hdu" or parent["target"] != T3_TARGETS["hdu"]
                    or parent["split"] != job["split"] or parent["method"] != job["method"]
                    or file_hash(parent_folder / "best.pt") != job["parent_sha256"]):
                errors.append({"kind": "adaptation_parent_mismatch", "path": str(path)})
        generators += 1
    for path in sorted((experiment / "jobs").glob("*/result.json")):
        result = read_json(path)
        job = result["job"]
        cells = manifests[job["split"]]["cells"]
        samples = {}
        for role in ("train", "validation"):
            selected = sorted((cell for cell in cells if cell["role"] == role and cell["device"] == job["device"]
                               and cell["condition"] != job["target"]
                               and cell["state"] not in T4_STATES.get(job["device"], ())),
                              key=lambda cell: (cell["state"], cell["condition"]))
            samples[role] = [sample for cell in selected for sample in cell["samples"][:ready["quotas"][role]]]
        source_identity = identity({"job": {key: job[key] for key in ("device", "channel", "target", "split")},
                                    "quotas": ready["quotas"], "inventory": ready["inventory_identity"], **samples})
        if source_identity != result["source_identity"]:
            errors.append({"kind": "classifier_source_mismatch", "path": str(path)})
        frozen = file_hash(path.with_name("best.pt"))
        target = read_json(path.with_name("target_manifest.json"))
        seen = read_json(path.with_name("seen_manifest.json"))
        if any(value != frozen for value in (result["best_checkpoint_sha256"], target["classifier_sha256"], seen["classifier_sha256"])):
            errors.append({"kind": "evaluation_checkpoint_mismatch", "path": str(path)})
        for sample in target["samples"]:
            record = by_id[sample["record"]]
            if record["device"] != job["device"] or record["condition"] != job["target"]:
                errors.append({"kind": "target_identity_mismatch", "path": str(path), "sample": sample})
        for cell in seen["cells"]:
            if cell["condition"] == job["target"] or any(sample["role"] != "seen" for sample in cell["samples"]):
                errors.append({"kind": "seen_identity_mismatch", "path": str(path)})
        classifiers += 1
    return {"experiment": str(experiment.resolve()), "generator_source_manifests": generators,
            "completed_classifiers": classifiers, "errors": errors}


def audit(run: Path, experiments=()):
    records = read_json(run / "inventory.json")["records"]
    by_id = {record["id"]: record for record in records}
    receipts = {record["id"]: read_json(raw_cache_path(run, record).with_suffix(".json")) for record in records}
    errors = []
    aliases = defaultdict(list)
    for record in records:
        digest = receipts[record["id"]].get("waveform_sha256")
        if digest:
            aliases[digest].append(record)
    for digest, members in aliases.items():
        if len({(record["device"], record["condition"], record["state"]) for record in members}) > 1:
            errors.append({"kind": "identical_waveform_across_cells", "sha256": digest,
                           "records": [record["id"] for record in members]})
    summaries = []
    manifests = {}
    for split in SPLITS:
        partitions = read_json(run / "partitions" / f"{split}.json")
        manifest = read_json(run / "cache/windows" / str(split) / "manifest.json")
        manifests[split] = manifest
        grouped = defaultdict(list)
        bounds = set()
        specimen_roles = defaultdict(set)
        for partition in partitions:
            record = by_id[partition["record"]]
            digest = receipts[record["id"]].get("waveform_sha256")
            if digest:
                grouped[("content", digest)].append(partition)
            if record["device"] == "cwru":
                factor = 4 if record["extra"].get("raw_sample_rate") == 48000 else 1
                grouped[("cwru_grid", record["group"])].append(
                    {**partition, "start": partition["start"] / factor, "stop": partition["stop"] / factor})
            if record["device"] == "pu" and record["state"] not in T4_STATES["pu"]:
                specimen_roles[record["group"]].add(partition["role"])
            bounds.add((partition["record"], partition["role"], partition["start"], partition["stop"]))
        for key, members in grouped.items():
            for index, left in enumerate(members):
                for right in members[index + 1:]:
                    if left["role"] != right["role"] and max(left["start"], right["start"]) < min(left["stop"], right["stop"]):
                        errors.append({"kind": "overlapping_roles", "split": split, "alias": key,
                                       "left": left, "right": right})
        for specimen, roles in specimen_roles.items():
            if len(roles) > 1:
                errors.append({"kind": "source_specimen_roles", "split": split, "specimen": specimen})
        samples = 0
        for cell in manifest["cells"]:
            for sample in cell["samples"]:
                record = by_id[sample["record"]]
                partition = (sample["record"], sample["role"], sample["start"], sample["stop"])
                if partition not in bounds or sample["role"] != cell["role"]:
                    errors.append({"kind": "window_partition_mismatch", "split": split, "sample": sample})
                if (record["device"], record["condition"], record["state"]) != (cell["device"], cell["condition"], cell["state"]):
                    errors.append({"kind": "window_cell_mismatch", "split": split, "sample": sample})
                factor = 4 if record["extra"].get("raw_sample_rate") == 48000 else 1
                usable = (sample["stop"] - sample["start"]) // factor - (20 if factor == 4 else 0)
                relative = sample["relative_start"]
                parent_start = (relative // PARENT) * PARENT
                if relative < 0 or relative + WINDOW > min(parent_start + PARENT, usable):
                    errors.append({"kind": "window_outside_parent", "split": split, "sample": sample})
                samples += 1
        summaries.append({"split": split, "partitions": len(partitions), "windows": samples,
                          "partitions_sha256": file_hash(run / "partitions" / f"{split}.json"),
                          "manifest_sha256": file_hash(run / "cache/windows" / str(split) / "manifest.json")})
    executions = [audit_execution(run, experiment, records, manifests) for experiment in experiments]
    errors.extend(error for execution in executions for error in execution["errors"])
    return {"created_at": timestamp(), "data_run": str(run.resolve()),
            "inventory_sha256": file_hash(run / "inventory.json"), "records": len(records),
            "duplicate_content_groups": sum(len(members) > 1 for members in aliases.values()),
            "splits": summaries, "executions": executions, "errors": errors, "passed": not errors,
            "scope": "Raw content aliases, CWRU derived grids, PU source specimen roles, and cached-window provenance.",
            "task_boundary": "Unseen operating conditions; this is not a claim of unseen-specimen generalization across target conditions.",
            "limitations": "Uses cached content receipts; metadata cannot exclude undocumented partial duplicates or acquisition aliases."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-run", type=Path, action="append", default=[])
    arguments = parser.parse_args()
    result = audit(arguments.data_run, arguments.experiment_run)
    write_json(arguments.output, result)
    print({"passed": result["passed"], "errors": len(result["errors"]), "receipt": str(arguments.output)}, flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
