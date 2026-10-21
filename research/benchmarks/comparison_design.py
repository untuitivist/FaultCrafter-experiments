from __future__ import annotations

from collections import Counter

from .common import identity
from .t4_diffucd import execution_plan as generation_plan


def execution_plan(run):
    plan = generation_plan(run)
    jobs = list(plan["jobs"])
    existing = {job["id"] for job in jobs}
    for template in plan["jobs"]:
        if template["kind"] != "cnn" or template["track"] not in ("T1", "T2") or template["method"] != "real_only":
            continue
        job = {key: value for key, value in template.items() if key not in ("id", "implementation")}
        job["method"] = "coral"
        job["id"] = identity(job)[:24]
        job["implementation"] = "ready"
        if job["id"] not in existing:
            jobs.append(job)
            existing.add(job["id"])
    design = {**plan["execution_design"], "source_alignment": "coral-source-only-v1"}
    return {**plan, "jobs": jobs, "execution_design": design, "design_sha256": identity(design),
            "counts": dict(sorted(Counter(f"{job['kind']}/{job['track']}/{job['method']}" for job in jobs).items()))}
