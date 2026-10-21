from __future__ import annotations

from .common import METHODS, T1_DEVICES, T2_DEVICES
from .data import CHANNELS
from .tasks import GENERATED_ONLY_PANELS, T3_TARGETS


def dependency_order(selected, all_jobs):
    ordered = []
    generators = []
    visited = set()
    visiting = set()

    def add_generator(identifier):
        if identifier in visited:
            return
        if identifier in visiting:
            raise ValueError("Generator dependency cycle: " + identifier)
        job = all_jobs[identifier]
        if job["kind"] != "generator":
            raise ValueError("A generator dependency refers to a non-generator job")
        visiting.add(identifier)
        if job["parent"]:
            parent = all_jobs[job["parent"]]
            if (parent["method"], parent["split"]) != (job["method"], job["split"]):
                raise ValueError("A generator parent must use the same method and seed")
            add_generator(job["parent"])
        visiting.remove(identifier)
        visited.add(identifier)
        generators.append(job)
        ordered.append(job)

    for job in selected:
        if job["kind"] != "cnn" or job["id"] in visited:
            raise ValueError("Select unique diagnostic jobs")
        if job["parent"]:
            add_generator(job["parent"])
        visited.add(job["id"])
        ordered.append(job)
    return ordered, generators


def representative_selection(plan, splits):
    selected = []
    reasons = {}
    for split in splits:
        candidates = [job for job in plan["jobs"] if job["kind"] == "cnn" and job["split"] == split]
        targets = {}
        for device in (*T1_DEVICES, *T2_DEVICES):
            available = sorted({job["target"] for job in candidates if job["device"] == device and job["track"] in ("T1", "T2")})
            targets[device] = T3_TARGETS.get(device, available[len(available) // 2])

        def choose(track, device, method, reason, arm=None):
            channel = "bearing_base" if device == "hdu" else CHANNELS[device][0]
            if arm == "generated_only":
                panel = GENERATED_ONLY_PANELS[track]
                if panel["device"] != device or panel["target"] != targets[device]:
                    return
                channel = panel["channel"]
            matches = [job for job in candidates if (job["track"], job["device"], job["target"], job["channel"], job["method"], job["arm"])
                       == (track, device, targets[device], channel, method, arm)]
            if len(matches) != 1:
                raise ValueError(f"Expected one representative job, found {len(matches)}: {track}/{device}/{method}/{arm}/{split}")
            job = matches[0]
            if job["id"] in reasons:
                raise ValueError("Duplicate representative diagnostic job")
            selected.append(job)
            reasons[job["id"]] = reason

        for device in (*T1_DEVICES, *T2_DEVICES):
            choose("T1" if device in T1_DEVICES else "T2", device, "real_only", "Full CNN convergence on each of the eight device training pools")
        for method in ("augmentation", "mixstyle"):
            choose("T1", "hdu", method, "Conventional augmentation and feature mixing CNN costs")
        for method in ("flowts", "ccgan", "time_weaver", "iccdm", "diffucd"):
            for arm in ("real_generated", "generated_only"):
                choose("T1", "hdu", method, "Full shared T1 generator and two diagnostic training-pool compositions", arm)
            if method == "flowts":
                choose("T3", "hdu", method, "Random-condition generation and enlarged diagnostic training pool", "multi_252")
                choose("T4", "hdu", method, "Unseen fault-combination inference and diagnostic evaluation")
            if method in ("flowts", "time_weaver", "diffucd"):
                device = "pu" if method == "diffucd" else "bjtu"
                for arm in ("real_generated", "generated_only"):
                    choose("T2", device, method, "Full T2 adaptation: small epoch-based, source-early-stopped, or fixed-update training", arm)
                if method == "flowts":
                    choose("T3", device, method, "T3 inference after new-device adaptation", "multi_252")
                    choose("T4", device, method, "T4 composition after new-device adaptation")
            if method == "iccdm":
                choose("T4", "hdu", method, "Diffusion sampling for unseen fault combinations")
    methods = {job["method"] for job in selected if job["method"] in METHODS}
    if methods != set(METHODS):
        raise ValueError("Representative selection must cover all five external methods")
    return selected, {"version": "full-budget-representatives-v2", "purpose": "Runtime calibration from complete training and inference",
                      "complete_matrix": False, "seeds": list(splits), "reasons": reasons,
                      "description": "Eight device CNN controls, all five full T1 methods, three T2 adaptations, and reused T3/T4 inference; generated-only uses the registered representative panels."}
