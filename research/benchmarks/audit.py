from __future__ import annotations

import argparse
import importlib.metadata
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import METHODS, SPLITS, file_hash, identity, read_json, timestamp, write_json
from .data import CHANNELS, T4_STATES
from .tasks import T3_TARGETS
from .recipes import CNN_RECIPE, generator_recipe


def audit(run):
    coverage = read_json(run / "coverage.json")
    generator_coverage = read_json(run / "generator_coverage.json")
    errors = []
    paired = defaultdict(list)
    generators = {}
    counts = defaultdict(int)
    expected_updates = coverage["updates"]
    for job in generator_coverage["selected"]:
        folder = run / "generators" / job["id"]
        result_path = folder / "result.json"
        if not result_path.exists():
            errors.append({"id": job["id"], "error": "Missing generator result"})
            continue
        result = read_json(result_path)
        generators[job["id"]] = result
        counts["generators"] += 1
        if expected_updates is not None:
            actual = result.get("updates_completed", len(result.get("history", [])))
            if result["updates_per_phase"] != expected_updates or actual != expected_updates:
                errors.append({"id": job["id"], "error": "Generator update count differs from the requested budget"})
            for phases in result["auxiliary_training"].values():
                if any((history["updates_completed"] if isinstance(history, dict) else len(history)) != expected_updates for history in phases.values()):
                    errors.append({"id": job["id"], "error": "Auxiliary update count differs from the requested budget"})
        else:
            if result.get("mode") != "formal" or result.get("recipe") != generator_recipe(job["method"]):
                errors.append({"id": job["id"], "error": "Formal generator recipe or mode mismatch"})
            expected_phases = {"main", *(phase["name"] for phase in generator_recipe(job["method"])["auxiliaries"])}
            if set(result.get("phase_records", {})) != expected_phases:
                errors.append({"id": job["id"], "error": "Missing or unregistered training phase"})
            for phase, receipt in result.get("phase_records", {}).items():
                limit = receipt["budget"]["max_updates"]
                actual = receipt["updates_completed"]
                allowed_stop = phase == "main" and bool(result["recipe"].get("early_stopping")) and receipt["stop_reason"] == "early_stopping"
                if not (0 < actual <= limit and (actual == limit or allowed_stop)):
                    errors.append({"id": job["id"], "phase": phase, "error": "Formal phase did not reach its configured stopping rule"})
        receipts = read_json(folder / "source_manifest.json")
        for role, cells in receipts.items():
            for cell in cells:
                if cell["state"] in T4_STATES.get(cell["device"], ()):
                    errors.append({"id": job["id"], "error": "Held fault combination in generator sources", "cell": cell})
                if cell["device"] == job["device"] and cell["condition"] == job["target"]:
                    errors.append({"id": job["id"], "error": "Held operating condition in generator sources", "cell": cell})
                if job["track"] == "T2" and cell["device"] != job["device"]:
                    errors.append({"id": job["id"], "error": "Unregistered T2 adaptation device", "cell": cell})
        if file_hash(folder / "best.pt") != result["best_checkpoint_sha256"]:
            errors.append({"id": job["id"], "error": "Generator checkpoint checksum mismatch"})
    for identifier, result in generators.items():
        job = result["job"]
        if job["track"] == "T2":
            parent = generators.get(job["parent"])
            if (parent is None or parent["job"]["track"] != "T1" or parent["job"]["method"] != job["method"]
                    or parent["job"]["split"] != job["split"] or parent["job"]["device"] != "hdu"
                    or parent["job"]["target"] != T3_TARGETS["hdu"]
                    or result["parent_sha256"] != parent["best_checkpoint_sha256"]):
                errors.append({"id": identifier, "error": "T2 inherited checkpoint does not match its registered T1 method and seed"})
    repeat_groups = defaultdict(list)
    for job in coverage["selected"]:
        folder = run / "jobs" / job["id"]
        if not (folder / "result.json").exists():
            errors.append({"id": job["id"], "error": "Missing diagnostic result"})
            continue
        result = read_json(folder / "result.json")
        settings = read_json(folder / "job.json")
        target = read_json(folder / "target_manifest.json")
        seen = read_json(folder / "seen_manifest.json")
        counts["cnns"] += 1
        if expected_updates is not None and result["updates_completed"] != expected_updates:
            errors.append({"id": job["id"], "error": "CNN update count differs from the requested budget"})
        if expected_updates is None:
            epoch = result.get("epochs_completed", 0)
            stopped = result.get("stop_reason") == "early_stopping" and result.get("early_stopping", {}).get("bad_epochs", 0) >= CNN_RECIPE["early_stopping"]["patience"]
            if (result.get("recipe") != CNN_RECIPE or not (0 < epoch <= CNN_RECIPE["max_epochs"])
                    or result["updates_completed"] != epoch * settings["updates_per_epoch"]
                    or not (epoch == CNN_RECIPE["max_epochs"] or stopped)):
                errors.append({"id": job["id"], "error": "CNN epoch traversal or early-stopping record mismatch"})
        frozen_sha = result["best_checkpoint_sha256"]
        if file_hash(folder / "best.pt") != frozen_sha or target["classifier_sha256"] != frozen_sha or seen["classifier_sha256"] != frozen_sha:
            errors.append({"id": job["id"], "error": "Diagnostic checkpoint or evaluation checksum mismatch"})
        pairing = tuple(job[key] for key in ("track", "device", "channel", "target", "split"))
        paired[pairing].append((result["initial_weights_sha256"], result["source_identity"],
                                identity({key: value for key, value in target.items() if key != "classifier_sha256"}),
                                identity(seen["cells"])))
        repeat = tuple(job[key] for key in ("track", "device", "channel", "method", "arm", "target"))
        repeat_groups[repeat].append(job["split"])
        if job["method"] in METHODS:
            generation = read_json(folder / "generation.json")
            parent = generators.get(job["parent"])
            if parent is None or generation["checkpoint_sha256"] != parent["best_checkpoint_sha256"]:
                errors.append({"id": job["id"], "error": "Synthetic pool does not use the registered frozen generator"})
            if settings["additional_windows"] != generation["count"]:
                errors.append({"id": job["id"], "error": "Synthetic pool count differs from CNN settings"})
            counts["generated_only_cnns"] += int(job.get("arm") == "generated_only")
            counts["synthetic_cnns"] += 1
    for key, values in paired.items():
        if len(set(values)) != 1:
            errors.append({"pair": key, "error": "Compared CNNs differ in initialization, source windows, or real evaluation samples"})
    for key, seeds in repeat_groups.items():
        if sorted(seeds) != list(SPLITS):
            errors.append({"group": key, "error": "Missing or duplicated repeat seed", "seeds": seeds})
    channel_groups = defaultdict(set)
    for job in coverage["selected"]:
        key = tuple(job[field] for field in ("track", "device", "method", "arm", "target", "split"))
        channel_groups[key].add(job["channel"])
    for key, channels in channel_groups.items():
        if channels != set(CHANNELS[key[1]]):
            errors.append({"group": key, "error": "Missing registered channel", "channels": sorted(channels)})
    synthetic = []
    for path in sorted((run / "synthetic").glob("*/manifest.json")):
        receipt = read_json(path)
        arrays_path = path.with_name("signals.npz")
        with np.load(arrays_path) as arrays:
            valid = len(arrays["x"]) == receipt["count"] and len(arrays["y"]) == receipt["count"] and np.isfinite(arrays["x"]).all()
        if not valid or file_hash(arrays_path) != receipt["signals_sha256"]:
            errors.append({"path": str(path), "error": "Synthetic pool count, finiteness or checksum mismatch"})
        synthetic.append({"path": str(path.relative_to(run)), "method": generators[receipt["parent"]]["job"]["method"],
                          "count": receipt["count"], "sampling_seconds": receipt["sampling_seconds"]})
    payload = {"created": timestamp(), "passed": not errors, "scope": coverage["sampling"],
               "seeds": list(SPLITS), "standard_deviation": "Sample SD, ddof=1, across the five registered seeds",
               "counts": dict(counts), "paired_groups": len(paired), "repeat_groups": len(repeat_groups),
               "synthetic_pools": synthetic, "errors": errors, "code_sha256": file_hash(Path(__file__))}
    write_json(run / "audit.json", payload)
    dependencies = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "einops", "tqdm", "pillow"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    write_json(run / "dependencies.json", dependencies)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.run)
    print(f"Execution audit: passed={result['passed']}; counts={result['counts']}; errors={len(result['errors'])}")
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
