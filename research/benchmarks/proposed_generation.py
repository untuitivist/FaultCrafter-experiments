from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from faultcrafter.method_model import load_method

from .common import ROOT, file_hash, identity, read_json, seed_for, write_json
from .proposed_training import method_numerics
from .data import PARENT, STRIDE, T4_STATES, WINDOW
from .generators import ConditionSchema
from .pools import source_pool


def synthetic_pool(run, job, output, *, device="cuda"):
    folder_key = {key: job.get(key) for key in ("parent", "device", "channel", "track", "target", "split")}
    if job["track"] == "T3":
        folder_key["generation_mode"] = job["generation_mode"]
    parent_folder = output / "synthetic_parents" / identity(folder_key)[:24]
    output_key = {**folder_key, "quota": job["generation_quota"]} if job["track"] == "T3" else folder_key
    folder = output / "synthetic" / identity(output_key)[:24]
    parent_path = output / "generators" / job["parent"] / "best.pt"
    parent_sha = file_hash(parent_path)
    checkpoint = torch.load(parent_path, weights_only=False, map_location="cpu")
    if checkpoint["verification"]:
        raise ValueError("Verification weights cannot produce formal diagnostic results")
    calls = checkpoint["recipe"]["sampling_calls"]
    schema = ConditionSchema(read_json(run / "inventory.json")["records"])
    if identity(checkpoint["schema"]) != identity(schema.receipt()):
        raise ValueError("Generation schema changed")
    pool = source_pool(run, job)
    states = T4_STATES[job["device"]] if job["track"] == "T4" else pool["states"]
    if job["track"] == "T3" and job["generation_mode"] == "multi":
        sampling = read_json(ROOT / "research/protocols/t3_condition_sampling_v9.json")
        name = {"hdu": "HDU-BFDS", "dirg": "DIRG", "bjtu": "BJTU"}[job["device"]]
        panel = next(panel for panel in sampling["panels"] if panel["device"] == name)
        coordinates = panel["generated_conditions"]
        if job["device"] == "bjtu":
            coordinates = [list(point) + [10000] for point in coordinates]
        condition_sampling = {"sha256": file_hash(ROOT / "research/protocols/t3_condition_sampling_v9.json"), "root_seed": sampling["root_seed"]}
    elif job["track"] in ("T3", "T4"):
        coordinates = [schema.coordinates[job["device"], name] for name in pool["roles"]["train"]["conditions"]]
        condition_sampling = None
    else:
        coordinates = [schema.coordinates[job["device"], job["target"]]]
        condition_sampling = None
    parent_count = 16 if job["track"] == "T3" else len(coordinates)
    parent_receipt = {"parent_sha256": parent_sha, "calls": calls, "states": states, "coordinates": coordinates,
                      "parents_per_state": parent_count, "length": PARENT, "condition_sampling": condition_sampling,
                      "code_sha256": file_hash(Path(__file__))}
    parent_folder.mkdir(parents=True, exist_ok=True)
    parent_file = parent_folder / "parents.npy"
    if parent_file.exists() and (parent_folder / "manifest.json").exists():
        existing = read_json(parent_folder / "manifest.json")
        if identity(existing["configuration"]) != identity(parent_receipt) or existing["signals_sha256"] != file_hash(parent_file):
            raise ValueError("Long-parent generation inputs changed; use a fresh output directory")
        parents = np.load(parent_file, mmap_mode="r")
    else:
        started = time.perf_counter()
        model = load_method(checkpoint).to(device).eval()
        parents = np.empty((len(states), parent_count, PARENT), dtype=np.float32)
        with method_numerics(), torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith("cuda") else []):
            torch.manual_seed(seed_for(folder_key, "long_parent_generation"))
            for state_index, state in enumerate(states):
                for parent_index in range(parent_count):
                    coordinate = coordinates[parent_index % len(coordinates)]
                    vector = schema.encode(job["device"], job["channel"], state, coordinate)
                    values = model.sample(torch.as_tensor(vector[None], device=device), steps=calls, length=PARENT)
                    parents[state_index, parent_index] = values[0, 0].cpu().numpy()
        if not np.isfinite(parents).all():
            raise FloatingPointError("Nonfinite proposed-method long generation")
        temporary = parent_file.with_name(parent_file.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, parents)
        os.replace(temporary, parent_file)
        write_json(parent_folder / "manifest.json", {"configuration": parent_receipt, "signals_sha256": file_hash(parent_file),
                                                   "generation_seconds": time.perf_counter() - started})
    manifest_path = folder / "manifest.json"
    parent_manifest_sha = file_hash(parent_folder / "manifest.json")
    if manifest_path.exists():
        receipt = read_json(manifest_path)
        if (receipt["parent_manifest_sha256"] != parent_manifest_sha
                or receipt["source_identity"] != pool["source_identity"]
                or receipt["count"] != job["generation_quota"] * len(states)
                or receipt["signals_sha256"] != file_hash(folder / "signals.npz")):
            raise ValueError("Cached diagnostic generation changed")
        with np.load(folder / "signals.npz") as arrays:
            return arrays["x"], arrays["y"], receipt
    signals, labels, requests = [], [], []
    quota = job["generation_quota"]
    for state_index, state in enumerate(states):
        for coordinate_index, coordinate in enumerate(coordinates):
            needed = quota // len(coordinates) + (coordinate_index < quota % len(coordinates))
            candidates = [index for index in range(parent_count) if index % len(coordinates) == coordinate_index]
            for position, parent_index in enumerate(candidates):
                count = needed // len(candidates) + (position < needed % len(candidates))
                random = np.random.default_rng(seed_for(folder_key, state, parent_index, "diagnostic_starts"))
                available = np.arange(0, PARENT - WINDOW + 1, STRIDE)
                starts = random.permutation(available)[:count]
                if len(starts) != count:
                    raise ValueError("Insufficient distinct diagnostic windows in a generated parent")
                for start in starts:
                    signals.append(np.array(parents[state_index, parent_index, start:start + WINDOW])[None])
                    labels.append([part in pool["labels"][state] for part in pool["components"]])
                requests.append({"state": state, "coordinates": coordinate, "parent": parent_index, "starts": starts.tolist()})
    values = np.stack(signals).astype(np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if len(values) != quota * len(states):
        raise ValueError("Proposed-method generation quota mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "signals.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, x=values, y=targets)
    os.replace(temporary, folder / "signals.npz")
    receipt = {"checkpoint_sha256": parent_sha, "sampling_steps": calls, "network_calls": calls,
               "parent_manifest_sha256": parent_manifest_sha, "requests": requests,
               "source_identity": pool["source_identity"], "count": len(values), "signals_sha256": file_hash(folder / "signals.npz")}
    write_json(folder / "manifest.json", receipt)
    return values, targets, receipt
