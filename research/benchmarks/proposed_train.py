from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
from pathlib import Path

import torch

from .common import METHODS, SPLITS, identity, log, read_json, timestamp, write_json
from .leakage_audit import audit
from .proposed_plan import build_plan
from .queue import QueueLock


def select_jobs(plan, identifiers):
    if not identifiers:
        return plan["jobs"]
    lookup = {job["id"]: job for job in plan["jobs"]}
    required = set()

    def include(identifier):
        if identifier not in lookup:
            raise ValueError(f"Unknown job: {identifier}")
        if identifier in required:
            return
        job = lookup[identifier]
        if job.get("parent"):
            include(job["parent"])
        required.add(identifier)

    for identifier in identifiers:
        include(identifier)
    return [job for job in plan["jobs"] if job["id"] in required]


def retention_cases(output, jobs):
    cases = []
    for job in jobs:
        if (job["kind"] != "generator" or job["method"] != "faultcrafter" or job["track"] != "T2"
                or job["strategy"] not in ("lora", "full")):
            continue
        folder = output / "generators" / job["id"]
        if not (folder / "result.json").exists():
            continue
        result = read_json(folder / "result.json")
        if result.get("verification") or not result.get("frozen_parameters_verified"):
            raise ValueError("Retention requires completed, verified formal adaptation")
        base = "generators/" + job["parent"]
        adapted = "generators/" + job["id"]
        cases.append({"strategy": "lora" if job["strategy"] == "lora" else "full_finetune",
                      "split": job["split"], "base_generator": base, "adapted_generator": adapted,
                      "original_generator": base if job["strategy"] == "lora" else adapted})
    write_json(output / "retention_cases.json", cases)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fresh release32-structure benchmark training; proposed method capped at 5000 updates.")
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "proposed", "external"), default="all")
    parser.add_argument("--splits", nargs="+", type=int, choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--job-id", nargs="+", help="Run these registered jobs and their complete generator dependencies")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--with-retention", action="store_true")
    arguments = parser.parse_args(argv)
    run, output = arguments.data_run.resolve(), arguments.output.resolve()
    if output == run or len(set(arguments.splits)) != len(arguments.splits):
        raise ValueError("Use a separate output directory and unique registered seeds")
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(run, arguments.splits, arguments.scope)
    jobs = select_jobs(plan, arguments.job_id)
    selection = {"design_sha256": plan["design_sha256"], "jobs": jobs,
                 "data_ready_sha256": identity(read_json(run / "data_ready.json"))}
    with QueueLock(output):
        path = output / "selection.json"
        if path.exists() and identity(read_json(path)) != identity(selection):
            raise ValueError("Selected protocol changed; use a fresh output directory")
        write_json(path, selection)
        write_json(output / "plan.json", plan)
        log("proposed_plan", generators=sum(job["kind"] == "generator" for job in jobs),
            classifiers=sum(job["kind"] == "cnn" for job in jobs), training_started=False)
        if arguments.plan_only:
            return
        checked = audit(run)
        write_json(output / "leakage_audit.json", checked)
        if not checked["passed"]:
            raise RuntimeError("Data isolation audit failed; no training started")
        from .classifier import configure_runtime
        from .coral import run_classifier
        from .evaluation_cache import EvaluationCache
        from .generators import run_generator as external_training, synthetic_pool as external_generation
        from .proposed_generation import synthetic_pool
        from .proposed_training import run_generator

        runtime = configure_runtime(arguments.threads)
        cache = EvaluationCache()
        cache.install()
        write_json(output / "runtime.json", {**runtime, "python": sys.version, "platform": platform.platform()})
        cnns = [job for job in jobs if job["kind"] == "cnn"]
        generators = [job for job in jobs if job["kind"] == "generator"]
        write_json(output / "coverage.json", {"mode": "formal", "selected": cnns, "splits": arguments.splits,
                   "design_sha256": plan["design_sha256"], "data_run": str(run), "blocked_methods": {},
                   "complete_matrix": not bool(arguments.job_id), "created": timestamp()})
        write_json(output / "generator_coverage.json", {"selected": generators, "count": len(generators)})
        for position, job in enumerate(jobs, 1):
            if (output / "STOP_AFTER_CURRENT").exists():
                write_json(output / "status.json", {"state": "stopped", "next": job, "updated": timestamp()})
                return
            write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "current": job,
                       "position": position, "total": len(jobs), "updated": timestamp()})
            if job["kind"] == "generator":
                if job["method"] == "faultcrafter":
                    run_generator(run, job, output)
                else:
                    external_training(run, job, output, updates=None, sampling_steps=None)
                retention_cases(output, jobs)
            else:
                generated = (synthetic_pool(run, job, output) if job["method"] == "faultcrafter" else
                             external_generation(run, job, output) if job["method"] in METHODS else None)
                run_classifier(run, job, output_root=output, generated=generated)
            gc.collect()
            torch.cuda.empty_cache()
            log("proposed_job_complete", id=job["id"], position=position, total=len(jobs))
        if cnns:
            from .report import generate_report
            generate_report(output)
        if arguments.with_retention:
            cases = retention_cases(output, jobs)
            if not cases:
                raise ValueError("No completed adaptation cases for retention")
            from .adaptation_retention import main as retention
            retention(["--data-run", str(run), "--cases", str(output / "retention_cases.json"),
                       "--output", str(output / "retention"), "--model-factory", "faultcrafter.method_model:load_method"])
        write_json(output / "evaluation_cache.json", cache.statistics())
        write_json(output / "status.json", {"state": "complete", "jobs_completed": len(jobs), "updated": timestamp()})


if __name__ == "__main__":
    main()
