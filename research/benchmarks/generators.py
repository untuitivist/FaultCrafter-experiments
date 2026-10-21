from __future__ import annotations

import gc
import math
import random
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from .common import ROOT, T1_DEVICES, file_hash, identity, log, read_json, seed_for, timestamp, write_json
from .data import CHANNELS, T4_STATES
from .generative_models import build_model
from .pools import source_pool
from .upstream import provenance


FIELDS = ("shaft_frequency_hz", "drive_frequency_hz", "radial_force_n", "lateral_force_n",
          "longitudinal_force_n", "load_power_w", "load_torque_nm")
SCALES = np.asarray([100, 100, 10000, 10000, 10000, 1000, 100], dtype=np.float32)
FIELD_INDICES = {"hdu": (0, 2), "dirg": (0, 2), "hust": (0,), "cwru": (0, 5),
                 "bjtu": (1, 3, 4), "hit_sq": (0, 2), "hit_self": (0,), "pu": (0, 2, 6)}
MODELS_VERSION = "external-generators-1d-v1"


class ConditionSchema:
    def __init__(self, records):
        self.devices = sorted(CHANNELS)
        self.channels = [(device, channel) for device in self.devices for channel in CHANNELS[device]]
        self.components = sorted({(record["device"], part) for record in records for part in record["components"]})
        self.coordinates = {(record["device"], record["condition"]): record["coordinates"] for record in records}
        self.labels = {(record["device"], record["state"]): record["components"] for record in records}
        self.sample_rates = {record["device"]: record["sample_rate"] for record in records}
        self.sample_rates["cwru"] = 12000
        self.dimension = 14 + len(self.devices) + len(self.channels) + len(self.components) + 1

    def encode(self, device, channel, state, coordinates):
        vector = np.zeros(self.dimension, dtype=np.float32)
        axes = FIELD_INDICES[device]
        if len(coordinates) != len(axes) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid physical coordinates for {device}/{channel}/{state}: {coordinates}; expected fields {axes}")
        vector[list(axes)] = np.asarray(coordinates) / SCALES[list(axes)]
        vector[np.asarray(axes) + 7] = 1
        offset = 14
        vector[offset + self.devices.index(device)] = 1
        offset += len(self.devices)
        vector[offset + self.channels.index((device, channel))] = 1
        offset += len(self.channels)
        for part in self.labels[device, state]:
            vector[offset + self.components.index((device, part))] = 1
        vector[-1] = self.sample_rates[device] / 64000
        return vector

    def receipt(self):
        return {"fields": FIELDS, "scales": SCALES.tolist(), "devices": self.devices,
                "device_field_indices": FIELD_INDICES,
                "channels": self.channels, "components": self.components, "dimension": self.dimension,
                "numeric_coordinates": "Source values in typed physical fields; padding uses a separate presence mask."}


def verify_condition_schema(run):
    records = read_json(run / "inventory.json")["records"]
    schema = ConditionSchema(records)
    checked = set()
    for record in records:
        for channel in CHANNELS[record["device"]]:
            key = (record["device"], channel, record["state"], record["condition"])
            if key in checked:
                continue
            values = schema.encode(*key[:3], record["coordinates"])
            if values.shape != (schema.dimension,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid condition vector: {key}")
            checked.add(key)
    return {"checked_cells": len(checked), "schema": schema.receipt(), "inventory_sha256": file_hash(run / "inventory.json")}


class GeneratorData:
    def __init__(self, run, job):
        records = read_json(run / "inventory.json")["records"]
        self.schema = ConditionSchema(records)
        self.job = job
        devices = T1_DEVICES if job["track"] == "T1" else (job["device"],)
        manifest = read_json(run / "cache/windows" / str(job["split"]) / "manifest.json")
        self.cpu = {}
        self.units = {}
        self.unit_starts = {}
        self.device_units = {}
        receipts = {}
        self.unit_conditions = {}
        for role in ("train", "validation"):
            signals, conditions, units, starts, devices_to_units, sample_receipts = [], [], [], [], defaultdict(list), []
            offset = 0
            for cell in sorted(manifest["cells"], key=lambda value: (value["device"], value["state"], value["condition"])):
                device = cell["device"]
                if (cell["role"] != role or device not in devices or cell["state"] in T4_STATES.get(device, ())
                        or (device == job["device"] and cell["condition"] == job["target"])):
                    continue
                folder = run / "cache/windows" / str(job["split"])
                values = np.load(folder / cell["file"], mmap_mode="r")
                valid = np.load(folder / cell["file"].replace(".npy", "_valid.npy"), mmap_mode="r")
                if not valid.all():
                    raise RuntimeError("Invalid generator source window")
                for channel_index, channel in enumerate(CHANNELS[device]):
                    coordinates = self.schema.coordinates[device, cell["condition"]]
                    encoded = self.schema.encode(device, channel, cell["state"], coordinates)
                    unit = len(units)
                    signals.append(np.array(values[:, channel_index:channel_index + 1]))
                    conditions.append(np.repeat(encoded[None], len(values), axis=0))
                    units.append((device, channel, cell["state"], cell["condition"]))
                    starts.append(offset)
                    offset += len(values)
                    devices_to_units[device].append(unit)
                    sample_receipts.append({"device": device, "channel": channel, "state": cell["state"],
                                            "condition": cell["condition"], "samples": cell["samples"]})
            self.cpu[role] = (np.concatenate(signals), np.concatenate(conditions))
            self.units[role] = units
            self.unit_starts[role] = np.asarray(starts)
            self.device_units[role] = list(devices_to_units.values())
            self.unit_conditions[role] = self.cpu[role][1][starts]
            receipts[role] = sample_receipts
        self.quota = {role: len(self.cpu[role][0]) // len(self.units[role]) for role in self.cpu}
        self.receipts = receipts
        self.identity = identity({"schema": self.schema.receipt(), "samples": receipts})
        self.tensors = {}
        grouped = defaultdict(list)
        for unit, descriptor in enumerate(self.units["train"]):
            grouped[descriptor[:3]].append(unit)
        self.vicinal_groups = {}
        for group, candidates in grouped.items():
            numeric = self.unit_conditions["train"][candidates, :7]
            varying = np.ptp(numeric, axis=0) > 1e-8
            bandwidth = 1.06 * numeric.std(axis=0) * max(len(candidates), 2) ** (-0.2)
            pairs = np.linalg.norm(numeric[:, None] - numeric[None], axis=-1)
            pairs[pairs <= 1e-10] = np.inf
            nearest = pairs.min(axis=1)
            finite = nearest[np.isfinite(nearest)]
            self.vicinal_groups[group] = (np.asarray(candidates), numeric, varying, bandwidth,
                                          float(np.median(finite)) if len(finite) else 1e-3)

    def activate(self):
        self.tensors = {role: tuple(torch.as_tensor(value, device="cuda") for value in arrays)
                        for role, arrays in self.cpu.items()}

    def release(self):
        self.tensors.clear()

    def indices(self, count, rng, role="train"):
        units = [int(rng.choice(self.device_units[role][int(rng.integers(len(self.device_units[role])))]))
                 for _ in range(count)]
        return self.unit_starts[role][units] + rng.integers(self.quota[role], size=count)

    def batch(self, count, rng, role="train"):
        indices = torch.as_tensor(self.indices(count, rng, role), device="cuda")
        return tuple(value[indices] for value in self.tensors[role])

    def vicinal(self, count, rng, *, adaptive):
        selected = self.indices(count, rng)
        targets = self.cpu["train"][1][selected].copy()
        units = self.units["train"]
        chosen, weights = [], []
        for index, position in enumerate(selected):
            unit_index = int(np.searchsorted(self.unit_starts["train"], position, side="right") - 1)
            group = units[unit_index][:3]
            candidates, numeric, varying, bandwidth, base_radius = self.vicinal_groups[group]
            targets[index, :7] += rng.normal(size=7) * bandwidth * varying
            distances = np.linalg.norm(numeric - targets[index, :7], axis=1)
            if adaptive:
                ordered = np.sort(distances)
                radius = ordered[min(49 // self.quota["train"], len(ordered) - 1)] + 1e-8
            else:
                radius = max(base_radius, float(distances.min()) + 1e-8)
            eligible = np.flatnonzero(distances <= radius + 1e-8)
            choice = int(rng.choice(eligible))
            chosen.append(self.unit_starts["train"][candidates[choice]] + int(rng.integers(self.quota["train"])))
            weights.append(math.exp(-float(distances[choice] / radius) ** 2))
        signals = self.tensors["train"][0][torch.as_tensor(chosen, device="cuda")]
        return signals, torch.as_tensor(targets, device="cuda"), torch.tensor(weights, device="cuda", dtype=torch.float32)


@lru_cache(maxsize=1)
def generator_data(run, track, device, target, split):
    return GeneratorData(run, {"track": track, "device": device, "target": target, "split": split})


def generator_identity(job, dataset, parent_sha, configuration):
    code = {name: file_hash(ROOT / "research/benchmarks" / name)
            for name in ("generators.py", "generator_training.py", "recipes.py", "generative_models.py", "upstream.py", "data.py", "costs.py")}
    return identity({"version": MODELS_VERSION, "job": job, "source": dataset.identity,
                     "parent_sha256": parent_sha, "configuration": configuration, "code": code,
                     "upstream": provenance(job["method"])}), code


def run_generator(run, job, output, updates=3, sampling_steps=None):
    from .generator_training import run_generator as train

    return train(run, job, output, updates, sampling_steps)


def synthetic_pool(run, job, output, sampling_steps=None):
    if sampling_steps is None:
        sampling_steps = read_json(output / "generators" / job["parent"] / "result.json")["sampling_steps"]
    condition_sampling = None
    panel = None
    if job["track"] == "T3":
        sampling_path = ROOT / "research/protocols/t3_condition_sampling_v9.json"
        sampling = read_json(sampling_path)
        mapping = {"hdu": "HDU-BFDS", "dirg": "DIRG", "bjtu": "BJTU"}
        panel = next(panel for panel in sampling["panels"] if panel["device"] == mapping[job["device"]])
        condition_sampling = {"root_seed": sampling["root_seed"], "device_seed": panel["seed"],
                              "protocol_sha256": file_hash(sampling_path)}
    folder = output / "synthetic" / identity({key: job.get(key) for key in ("parent", "device", "channel", "track", "target", "split")})[:24]
    folder.mkdir(parents=True, exist_ok=True)
    parent_folder = output / "generators" / job["parent"]
    checkpoint_path = parent_folder / "best.pt"
    frozen_sha = file_hash(checkpoint_path)
    if (folder / "manifest.json").exists():
        receipt = read_json(folder / "manifest.json")
        if receipt["checkpoint_sha256"] != frozen_sha:
            raise RuntimeError("Synthetic pool checkpoint mismatch")
        if receipt["sampling_steps"] != sampling_steps:
            raise RuntimeError("Synthetic pool sampler changed; choose a new output directory")
        if receipt.get("condition_sampling") != condition_sampling:
            raise RuntimeError("Condition sampling changed; choose a new output directory")
        arrays = np.load(folder / "signals.npz")
        return arrays["x"], arrays["y"], receipt
    records = read_json(run / "inventory.json")["records"]
    schema = ConditionSchema(records)
    pool = source_pool(run, job)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(job["method"], schema.dimension).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    states = T4_STATES[job["device"]] if job["track"] == "T4" else pool["states"]
    if job["track"] == "T3":
        coordinates = panel["generated_conditions"]
        if job["device"] == "bjtu":
            coordinates = [list(point) + [10000] for point in coordinates]
        quota = 252
    elif job["track"] == "T4":
        coordinates = [schema.coordinates[job["device"], name] for name in pool["roles"]["train"]["conditions"]]
        quota = 63
    else:
        coordinates = [schema.coordinates[job["device"], job["target"]]]
        quota = 63
    requests, outputs, labels = [], [], []
    generator_seed = seed_for(job["device"], job["channel"], job["target"], job["track"], job["split"], "generation")
    started = time.perf_counter()
    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(generator_seed)
        for state in states:
            needed = math.ceil(quota / 0.7) if job["method"] == "diffucd" else quota
            condition_vectors = np.stack([schema.encode(job["device"], job["channel"], state, coordinates[index % len(coordinates)])
                                          for index in range(needed)])
            generated = []
            for start in range(0, needed, 8):
                conditions = torch.as_tensor(condition_vectors[start:start + 8], device="cuda")
                generated.append(model.sample(conditions, steps=sampling_steps).cpu().numpy())
            signals = np.concatenate(generated)
            if not np.isfinite(signals).all():
                raise RuntimeError("Invalid generated signals")
            selected = np.arange(needed)
            if job["method"] == "diffucd":
                selected = []
                for coordinate_index in range(len(coordinates)):
                    indices = np.flatnonzero(np.arange(needed) % len(coordinates) == coordinate_index)
                    count = quota // len(coordinates) + (coordinate_index < quota % len(coordinates))
                    center = signals[indices].mean(axis=0)
                    distances = np.square(signals[indices] - center).mean(axis=(1, 2))
                    selected.extend(indices[np.argsort(distances, kind="stable")[:count]].tolist())
                selected = np.asarray(selected)
            signals = signals[selected[:quota]]
            outputs.append(signals)
            labels.extend([[part in schema.labels[job["device"], state] for part in pool["components"]]] * quota)
            requests.append({"state": state, "requested": quota, "candidates": needed,
                             "coordinates": coordinates, "selected_candidate_indices": selected[:quota].tolist()})
    elapsed = time.perf_counter() - started
    combined, labels = np.concatenate(outputs), np.asarray(labels, dtype=np.float32)
    np.savez_compressed(folder / "signals.npz", x=combined, y=labels)
    receipt = {"checkpoint_sha256": frozen_sha, "parent": job["parent"], "track": job["track"],
               "device": job["device"], "channel": job["channel"], "split": job["split"],
               "seed": generator_seed, "sampling_steps": sampling_steps, "source_identity": pool["source_identity"],
               "requests": requests, "count": len(combined), "sampling_seconds": elapsed,
               "filter": "Per requested condition/state, retain 70% closest to generated cluster center" if job["method"] == "diffucd" else None,
               "signals_sha256": file_hash(folder / "signals.npz")}
    if condition_sampling is not None:
        receipt["condition_sampling"] = condition_sampling
    write_json(folder / "manifest.json", receipt)
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return combined, labels, receipt
