from __future__ import annotations

from functools import lru_cache

from .common import METHODS, read_json


FILTERS = ("tracks", "devices", "channels", "methods", "splits", "targets", "ids")
FIELDS = ("track", "device", "channel", "method", "split", "target", "id")


@lru_cache(maxsize=2)
def registered_jobs(run):
    return read_json(run / "coverage.json")["selected"]


def add_filters(parser):
    for name in FILTERS:
        parser.add_argument("--" + name, nargs="+", type=int if name == "splits" else str)


def select_jobs(run, arguments):
    selected = [job for job in registered_jobs(run)
                if all(not getattr(arguments, option, None) or job.get(field) in getattr(arguments, option)
                       for option, field in zip(FILTERS, FIELDS))]
    if not selected:
        raise ValueError("No registered jobs match the selection")
    missing = [job["id"] for job in selected if not (run / "jobs" / job["id"] / "result.json").exists()]
    if missing:
        raise RuntimeError(f"Selected jobs are unfinished: {missing[:10]} ({len(missing)} total)")
    return selected


def reference_job(run, job):
    keys = ("track", "device", "channel", "target", "split")
    candidates = [candidate for candidate in registered_jobs(run)
                  if candidate["method"] == "real_only" and all(candidate[key] == job[key] for key in keys)]
    if len(candidates) != 1 or not (run / "jobs" / candidates[0]["id"] / "result.json").exists():
        raise RuntimeError(f"A completed paired Real-only reference is required for {job['id']}")
    return candidates[0]


def dependencies(run, jobs):
    diagnostics = {job["id"]: job for job in jobs}
    for job in jobs:
        reference = reference_job(run, job)
        diagnostics[reference["id"]] = reference
    generators = {}

    def add_parent(identifier):
        if not identifier or identifier in generators:
            return
        result = read_json(run / "generators" / identifier / "result.json")
        generators[identifier] = result["job"]
        add_parent(result["job"].get("parent"))

    for job in jobs:
        if job["method"] in METHODS:
            add_parent(job["parent"])
    return list(diagnostics.values()), list(generators.values())
