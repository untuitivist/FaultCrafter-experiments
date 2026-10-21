from __future__ import annotations

from collections import Counter

from .common import METHODS, T1_DEVICES, identity
from .comparison_design import execution_plan
from .proposed_training import training_recipe


def build_plan(run, splits, scope="all"):
    external = execution_plan(run)
    own = []
    parents = {}
    source_jobs = [job for job in external["jobs"] if job["kind"] == "generator" and job["method"] == "flowts" and job["split"] in splits]

    def add(template, **changes):
        job = {key: value for key, value in template.items() if key not in ("id", "implementation")}
        job.update(method="faultcrafter", design="release32-benchmark-v1", **changes)
        job["id"] = identity(job)[:24]
        job["implementation"] = "ready"
        own.append(job)
        return job

    for template in source_jobs:
        if template["track"] == "T1":
            job = add(template, parent=None, strategy="shared")
            parents[template["id"], "lora"] = job["id"]
    for template in source_jobs:
        if template["track"] == "T2":
            for strategy in ("lora", "full", "scratch"):
                parent = parents[template["parent"], "lora"] if strategy != "scratch" else None
                job = add(template, parent=parent, strategy=strategy)
                parents[template["id"], strategy] = job["id"]
    for template in external["jobs"]:
        if template["kind"] != "cnn" or template["method"] != "flowts" or template["split"] not in splits:
            continue
        parent = parents[template["parent"], "lora"]
        primary_strategy = "shared" if template["device"] in T1_DEVICES else "lora"
        if template["track"] == "T3":
            for mode, quota in (("source", 252), ("multi", 63), ("multi", 252)):
                add(template, parent=parent, strategy=primary_strategy, arm=f"{mode}_{quota}", generation_mode=mode, generation_quota=quota)
        else:
            add(template, parent=parent, strategy=primary_strategy, generation_quota=63)
            if template["track"] == "T2" and template.get("arm") != "generated_only":
                for strategy in ("full", "scratch"):
                    add(template, parent=parents[template["parent"], strategy], strategy=strategy, generation_quota=63)
    external_jobs = [job for job in external["jobs"] if job["split"] in splits]
    controls = [job for job in external_jobs if job["method"] not in METHODS]
    jobs = external_jobs if scope == "external" else own + (external_jobs if scope == "all" else controls)
    if scope not in ("all", "proposed", "external"):
        raise ValueError("Unknown training scope")
    generators = [job for job in jobs if job["kind"] == "generator"]
    generators.sort(key=lambda job: (job["track"] != "T1", job["split"], job["method"], job["device"], job["target"], job.get("strategy", "")))
    cnns = [job for job in jobs if job["kind"] == "cnn"]
    ordered = generators + cnns
    known = set()
    for job in ordered:
        if job.get("parent") and job["parent"] not in known:
            raise ValueError(f"Missing parent in proposed execution plan: {job['id']}")
        known.add(job["id"])
    if len(known) != len(ordered):
        raise ValueError("Duplicate execution identity")
    design = {"version": "release32-benchmark-v1", "scope": scope, "splits": sorted(splits),
              "external_design_sha256": external["design_sha256"], "proposed_recipe": training_recipe()}
    return {"design": design, "design_sha256": identity(design), "jobs": ordered,
            "counts": dict(sorted(Counter(f"{job['kind']}/{job['track']}/{job['method']}/{job.get('strategy', '')}" for job in ordered).items())),
            "generators": len(generators), "classifiers": len(cnns)}
