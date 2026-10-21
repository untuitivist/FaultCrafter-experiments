from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch

from faultcrafter.condition_guidance import TypedConditionProjectors
from faultcrafter.training.objectives import UnifiedJointBatch

from .common import T1_DEVICES, identity, read_json
from .data import CHANNELS, T4_STATES, partition_values, raw_cache_path
from .generators import ConditionSchema


def semantic_group(device, component):
    if device == "hdu":
        return {"I": 1, "B": 2, "O": 3, "T": 4}[component]
    if device == "hust":
        return {"I": 1, "B": 2, "O": 3}[component.split("_")[0]]
    if device == "dirg":
        return 1 if component in ("C1A", "C2A", "C3A") else 2
    if device == "bjtu":
        return {"LA1": 1, "LA2": 3, "LA3": 2, "LA4": 4}[component]
    for name, group in (("inner_race", 1), ("rolling_element", 2), ("outer_race", 3)):
        if name in component:
            return group
    raise ValueError(f"Unregistered fault semantics: {device}/{component}")


def vocabulary(schema):
    components = [list(value) for value in schema.components]
    return {"devices": list(schema.devices), "channels": [list(value) for value in schema.channels],
            "components": components, "component_groups": [semantic_group(*value) for value in components],
            "tokens": [[device, "__normal__"] for device in schema.devices] + components}


class ProposedCorpus:
    def __init__(self, run, job):
        self.run, self.job = run, job
        records = read_json(run / "inventory.json")["records"]
        self.schema = ConditionSchema(records)
        self.vocabulary = vocabulary(self.schema)
        self.records = {record["id"]: record for record in records}
        self.cells = {"train": defaultdict(list), "validation": defaultdict(list)}
        allowed = T1_DEVICES if job["track"] == "T1" else (job["device"],)
        receipts = []
        self.blocks = []
        for partition in read_json(run / "partitions" / f"{job['split']}.json"):
            record = self.records[partition["record"]]
            if (partition["role"] not in self.cells or record["device"] not in allowed
                    or record["state"] in T4_STATES.get(record["device"], ())
                    or (record["device"] == job["device"] and record["condition"] == job["target"])):
                continue
            receipt = read_json(raw_cache_path(run, record).with_suffix(".json"))
            if receipt["status"] != "ready":
                continue
            factor = 4 if record["extra"].get("raw_sample_rate") == 48000 else 1
            length = (partition["stop"] - partition["start"]) // factor - (20 if factor == 4 else 0)
            if length < 512:
                continue
            block_index = len(self.blocks)
            self.blocks.append({**partition, "length": length, "group": record["group"]})
            receipts.append({**self.blocks[-1], "waveform_sha256": receipt["waveform_sha256"]})
            for channel in CHANNELS[record["device"]]:
                key = (record["device"], channel, record["state"], record["condition"])
                self.cells[partition["role"]][key].append(block_index)
        self.keys = {role: sorted(cells) for role, cells in self.cells.items()}
        expected = {(record["device"], channel, record["state"], record["condition"])
                    for record in records if record["device"] in allowed
                    and record["state"] not in T4_STATES.get(record["device"], ())
                    and not (record["device"] == job["device"] and record["condition"] == job["target"])
                    for channel in CHANNELS[record["device"]]}
        if any(set(keys) != expected for keys in self.keys.values()):
            raise ValueError("Source cells are missing from the complete registered source inventory")
        if not self.keys["train"] or self.keys["train"] != self.keys["validation"]:
            raise ValueError("Every generator source cell needs both training and validation partitions")
        for role, cells in self.cells.items():
            if any(max(self.blocks[index]["length"] for index in indices) < 16384 for indices in cells.values()):
                raise ValueError(f"A {role} source cell cannot support the declared maximum training length")
        self.receipt = {"job": job, "blocks": receipts,
                        "units": {role: [list(key) for key in keys] for role, keys in self.keys.items()}}
        self.identity = identity(self.receipt)

    @lru_cache(maxsize=32)
    def values(self, block_index):
        block = self.blocks[block_index]
        return partition_values(self.run, self.records[block["record"]], block)

    def batch(self, model, role, keys, length, random):
        if role not in self.cells or not 512 <= length <= 16384:
            raise ValueError("Only legal source training/validation lengths may be sampled")
        signals, amplitudes, conditions, parents = [], [], [], []
        for key in keys:
            groups = defaultdict(list)
            for block_index in self.cells[role][key]:
                if self.blocks[block_index]["length"] >= length:
                    groups[self.blocks[block_index]["group"]].append(block_index)
            group = sorted(groups)[int(random.integers(len(groups)))]
            block_index = int(random.choice(groups[group]))
            block = self.blocks[block_index]
            values = self.values(block_index)
            start = int(random.integers(len(values) - length + 1))
            channel = CHANNELS[key[0]].index(key[1])
            parent = np.array(values[start:start + length, channel], dtype=np.float32)
            amplitude = float(np.max(np.abs(parent)))
            if not np.isfinite(parent).all() or not np.isfinite(amplitude) or amplitude <= 0:
                raise ValueError("A selected source parent is nonfinite or zero; repair the source inventory before training")
            signals.append((parent / amplitude)[None])
            amplitudes.append(amplitude)
            conditions.append(self.schema.encode(key[0], key[1], key[2], self.schema.coordinates[key[0], key[3]]))
            parents.append(f"{block['record']}:{role}:{block['start']}:{block['stop']}:{key[1]}:{start}:{length}")
        vectors = torch.as_tensor(np.stack(conditions))
        parameters, faults, groups, routes, speed, rates = model.encode(vectors)
        count = len(keys)
        return UnifiedJointBatch(np.stack(signals), np.asarray(amplitudes, dtype=np.float32), np.zeros(count, dtype=np.float32),
                                 parameters.numpy(), faults.numpy(), groups.numpy(), routes.numpy(), speed.numpy(), rates.numpy(),
                                 np.full(count, length, dtype=np.float32), np.asarray(parents))

    def projectors(self, compute_device):
        count = len(self.vocabulary["channels"])
        projectors = np.zeros((count, 14, 14), dtype=np.float32)
        presence = np.zeros((count, 14), dtype=np.float32)
        for route_index, route in enumerate(self.vocabulary["channels"]):
            keys = [key for key in self.keys["train"] if key[:2] == tuple(route)]
            if not keys:
                continue
            values = np.unique(np.stack([self.schema.encode(key[0], key[1], key[2], self.schema.coordinates[key[0], key[3]])[:14]
                                         for key in keys]), axis=0)
            presence[route_index, :7] = values[0, 7:]
            centered = values - values.mean(0)
            _, singular, basis = np.linalg.svd(centered, full_matrices=True)
            energy = singular ** 2
            rank = int(np.searchsorted(np.cumsum(energy), energy.sum() * 0.99) + 1) if energy.sum() > 1e-12 else 0
            weak = basis[rank:]
            projectors[route_index] = weak.T @ weak
        return TypedConditionProjectors(torch.as_tensor(projectors, device=compute_device), torch.as_tensor(presence, device=compute_device))


class BalancedRounds:
    def __init__(self, unit_count, seed, recipe):
        self.random = np.random.default_rng(seed)
        self.unit_count = unit_count
        self.recipe = recipe
        self.round = 0
        self.position = 0
        self.order = []
        self.length = None

    def begin(self, remaining):
        if self.position < len(self.order):
            return True
        state = self.random.bit_generator.state
        minimum, maximum = self.recipe["training_length_range"]
        length = int(self.random.integers(minimum, maximum + 1))
        batch_size = min(self.recipe["maximum_batch_size"], self.recipe["point_budget_per_batch"] // length)
        if (self.unit_count + batch_size - 1) // batch_size > remaining:
            self.random.bit_generator.state = state
            return False
        self.round += 1
        self.position = 0
        self.length = length
        self.order = self.random.permutation(self.unit_count).tolist()
        return True

    def next(self):
        count = min(self.recipe["maximum_batch_size"], self.recipe["point_budget_per_batch"] // self.length)
        indices = self.order[self.position:self.position + count]
        self.position += len(indices)
        return indices

    @property
    def complete(self):
        return self.position == len(self.order)

    def state_dict(self):
        return {"rng": self.random.bit_generator.state, "round": self.round, "position": self.position,
                "order": self.order, "length": self.length}

    def load_state_dict(self, state):
        self.random.bit_generator.state = state["rng"]
        self.round, self.position, self.order, self.length = (state[key] for key in ("round", "position", "order", "length"))
