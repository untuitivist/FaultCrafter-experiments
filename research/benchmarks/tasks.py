from __future__ import annotations

from collections import Counter

from .common import METHODS, PROTOCOL, SPLITS, T1_DEVICES, T2_DEVICES, file_hash, identity, read_json, write_json
from .data import CHANNELS, T4_STATES, condition_name


T3_TARGETS = {"hdu": condition_name([1000 / 60, 600]), "dirg": condition_name([200, 1000]),
              "bjtu": condition_name([40, 0, 10000])}
IMPLEMENTED_CONTROLS = ("real_only", "augmentation", "mixstyle", "superposition")
GENERATED_ONLY_PANELS = {
    "T1": {"device": "hdu", "channel": "bearing_upper", "target": T3_TARGETS["hdu"]},
    "T2": {"device": "bjtu", "channel": "CH17", "target": T3_TARGETS["bjtu"]},
}
ADAPTATION_STRATEGIES = ("LoRA", "全参数微调", "从零训练")
EVALUATION_LENGTHS = (257, 4096, 4097, 16384, 51802)
CONDITION_SPLITS = (42,)
SYNTHETIC_SPLITS = (42,)
LENGTH_SPLITS = (42,)
EXPERIMENT_DESIGN = {"version": "v23", "generated_only_panels": GENERATED_ONLY_PANELS,
                     "adaptation_strategies": ADAPTATION_STRATEGIES, "evaluation_lengths": EVALUATION_LENGTHS}


def generated_only_panel(track, device, channel, target):
    return GENERATED_ONLY_PANELS.get(track) == {"device": device, "channel": channel, "target": target}


def execution_plan(run):
    plan = read_json(run / "plan.json")
    jobs = [job for job in plan["jobs"] if job.get("arm") != "generated_only"
            or generated_only_panel(job["track"], job["device"], job["channel"], job["target"])]
    return {**plan, "jobs": jobs,
            "counts": dict(sorted(Counter(f"{job['kind']}/{job['track']}/{job['method']}" for job in jobs).items())),
            "execution_design": EXPERIMENT_DESIGN, "design_sha256": identity(EXPERIMENT_DESIGN)}


def label_dictionary(records, device):
    mapping = {item["state"]: item["components"] for item in records if item["device"] == device}
    components = sorted({part for parts in mapping.values() for part in parts})
    return mapping, components


def build_plan(run):
    records = read_json(run / "inventory.json")["records"]
    protocol_hash = file_hash(PROTOCOL)
    jobs = []

    def add(kind, track, device, target, split, method, channel=None, parent=None, arm=None):
        payload = {"kind": kind, "track": track, "device": device, "target": target, "split": split,
                   "method": method, "channel": channel, "parent": parent, "arm": arm,
                   "protocol_sha256": protocol_hash}
        payload["id"] = identity(payload)[:24]
        payload["implementation"] = "ready" if method in IMPLEMENTED_CONTROLS else "requires_method_port"
        jobs.append(payload)
        return payload["id"]

    parents = {}
    for track, devices in (("T1", T1_DEVICES), ("T2", T2_DEVICES)):
        for device in devices:
            targets = sorted({item["condition"] for item in records if item["device"] == device})
            if device == "pu":
                targets = [condition_name([25, 1000, 0.7])]
            for target in targets:
                for split in SPLITS:
                    for method in METHODS:
                        parent = parents["hdu", T3_TARGETS["hdu"], split, method] if track == "T2" else None
                        parents[device, target, split, method] = add("generator", track, device, target, split, method, parent=parent)
                    for channel in CHANNELS[device]:
                        for method in ("real_only", "augmentation", "mixstyle"):
                            add("cnn", track, device, target, split, method, channel)
                        for method in METHODS:
                            parent = parents[device, target, split, method]
                            arms = ("real_generated", "generated_only") if generated_only_panel(track, device, channel, target) else ("real_generated",)
                            for arm in arms:
                                add("cnn", track, device, target, split, method, channel, parent, arm)
                        if device in T4_STATES:
                            for method in ("real_only", "superposition", "iccdm", "flowts"):
                                parent = parents[device, target, split, method] if method in METHODS else None
                                add("cnn", "T4", device, target, split, method, channel, parent)
                        if T3_TARGETS.get(device) == target:
                            for method in ("real_only", "augmentation", "diffucd", "ccgan", "flowts"):
                                parent = parents[device, target, split, method] if method in METHODS else None
                                add("cnn", "T3", device, target, split, method, channel, parent,
                                    "multi_252" if method in METHODS else None)
    counts = Counter(f"{job['kind']}/{job['track']}/{job['method']}" for job in jobs)
    plan = {"protocol": "v23", "protocol_sha256": protocol_hash,
            "execution_design": EXPERIMENT_DESIGN, "design_sha256": identity(EXPERIMENT_DESIGN),
            "inventory_identity": identity(records), "jobs": jobs, "counts": dict(sorted(counts.items())),
            "scope": "External comparison matrix and common controls; proposed model and its ablations are not implemented."}
    write_json(run / "plan.json", plan)
    return plan
