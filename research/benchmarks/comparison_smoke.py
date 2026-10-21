from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
import traceback

from pathlib import Path

import numpy as np
import torch

from .classifier import configure_runtime
from .coral import run_classifier
from .common import METHODS, ROOT, RUN_ROOT, SPLITS, file_hash, log, read_json, timestamp, write_json
from .generators import run_generator, synthetic_pool, verify_condition_schema
from .evaluation_cache import EvaluationCache
from .queue import QueueLock
from .publish import publish_case
from .report import generate_report
from .tasks import T3_TARGETS
from .comparison_design import execution_plan
from .selection import dependency_order


def select_routes(plan):
    selected = {}
    candidates = sorted(plan["jobs"], key=lambda job: (job["target"] != T3_TARGETS.get(job["device"]), job["target"], job["id"]))
    for job in candidates:
        if job["kind"] != "cnn":
            continue
        key = (job["track"], job["device"], job["channel"], job["method"], job["arm"], job["split"])
        selected.setdefault(key, job)
    return sorted(selected.values(), key=lambda job: (job["split"], job["device"], job["channel"], job["track"], job["method"], job["arm"] or ""))


def main():
    parser = argparse.ArgumentParser(description="Run all external comparison routes with five seeds and a limited update budget.")
    parser.add_argument("--data-run", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/comparisons_compact_v23")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-routes", action="store_true")
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--splits", nargs="+", type=int, choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--publish-to", type=Path)
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.data_run.resolve():
        raise ValueError("Use a separate output directory for limited-budget runs")
    if not 0 < arguments.cuda_memory_fraction <= 1 or len(set(arguments.splits)) != len(arguments.splits):
        raise ValueError("Use unique seeds and a CUDA memory fraction within (0, 1]")
    if (arguments.probe or arguments.probe_routes) and 42 not in arguments.splits:
        raise ValueError("Probes require seed 42")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with QueueLock(arguments.output):
        execute(arguments)


def execute(arguments, *, selected_jobs=None, selection_description=None):
    run, output = arguments.data_run.resolve(), arguments.output.resolve()
    formal = getattr(arguments, "formal", False)
    runtime = configure_runtime(arguments.threads)
    torch.cuda.set_per_process_memory_fraction(arguments.cuda_memory_fraction)
    runtime["cuda_memory_fraction"] = arguments.cuda_memory_fraction
    runtime["publish_to"] = str(arguments.publish_to) if arguments.publish_to else None
    evaluation_cache = EvaluationCache()
    evaluation_cache.install()
    write_json(output / "condition_preflight.json", verify_condition_schema(run))
    plan = execution_plan(run)
    selected = ([job for job in plan["jobs"] if job["kind"] == "cnn"]
                if arguments.all_conditions else select_routes(plan))
    selected = [job for job in selected if job["split"] in arguments.splits]
    if selected_jobs is not None:
        selected = selected_jobs
    all_jobs = {job["id"]: job for job in plan["jobs"]}
    if arguments.probe:
        selected = [next(job for job in selected if job["method"] == method and job["track"] == "T1"
                         and job["device"] == "hdu" and job["arm"] == "real_generated" and job["split"] == 42)
                    for method in METHODS]
    if arguments.probe_routes:
        routes = {}
        for job in selected:
            if job["split"] == 42:
                routes.setdefault((job["track"], job["method"], job["arm"]), job)
        selected = list(routes.values())
    coverage = {"created": timestamp(), "mode": "formal" if formal else "smoke_only", "updates": arguments.updates,
                "splits": sorted(arguments.splits), "sampling_steps": arguments.sampling_steps,
                "sampling": (f"All registered conditions and selected seeds {sorted(arguments.splits)}." if arguments.all_conditions else
                             f"Every task/device/channel/method/data-arm route, selected seeds {sorted(arguments.splits)}; one representative held condition per route."),
                "data_run": str(run), "plan_sha256": file_hash(run / "plan.json"), "formal_jobs": len(plan["jobs"]),
                "selected": selected, "blocked_methods": {},
                "execution_design": plan["execution_design"], "design_sha256": plan["design_sha256"]}
    if selection_description:
        coverage["sampling"] = selection_description["description"]
        coverage["selection"] = selection_description
    if (output / "coverage.json").exists():
        previous = read_json(output / "coverage.json")
        if any(previous.get(key) != coverage[key] for key in ("updates", "sampling_steps", "selected", "plan_sha256", "design_sha256")):
            raise RuntimeError("Coverage changed; use a different output directory")
        coverage["created"] = previous["created"]
    coverage["evaluation_cache_sha256"] = file_hash(Path(__file__).with_name("evaluation_cache.py"))
    write_json(output / "coverage.json", coverage)
    write_json(output / "runtime.json", {**runtime, "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "numpy": np.__version__, "created": timestamp()})
    execution_order, generators = dependency_order(selected, all_jobs)
    write_json(output / "generator_coverage.json", {"selected": generators, "count": len(generators)})
    failures = []
    started = time.perf_counter()
    for index, job in enumerate(execution_order):
        if (output / "STOP_AFTER_CURRENT").exists():
            break
        write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "current": job,
                    "position": index + 1, "total": len(generators) + len(selected), "updated": timestamp()})
        try:
            if job["kind"] == "generator":
                run_generator(run, job, output, arguments.updates, arguments.sampling_steps)
            else:
                generated = synthetic_pool(run, job, output, arguments.sampling_steps) if job["method"] in METHODS else None
                run_classifier(run, job, output_root=output, smoke_updates=arguments.updates, generated=generated)
                if arguments.publish_to:
                    publish_case(output, arguments.publish_to, job)
        except Exception as error:
            folder = output / ("generators" if job["kind"] == "generator" else "jobs") / job["id"]
            write_json(folder / "failure.json", {"job": job, "error": str(error), "traceback": traceback.format_exc(), "updated": timestamp()})
            failures.append(job["id"])
            log("comparison_failure", id=job["id"], error=str(error))
            print(traceback.format_exc(), flush=True)
        finally:
            if job["kind"] == "generator" or index % 50 == 0:
                gc.collect()
            if job["kind"] == "generator":
                torch.cuda.empty_cache()
        if job["kind"] == "generator" and failures:
            break
        if job["kind"] == "cnn" and (selection_description or index % 50 == 0 or index + 1 == len(execution_order)):
            generate_report(output)
        if selection_description:
            from .estimate import estimate

            estimate(output, run)
    report = generate_report(output)
    missing_generators = [job["id"] for job in generators if not (output / "generators" / job["id"] / "result.json").exists()]
    write_json(output / "status.json", {"state": "complete" if not report["missing_selected_ids"] and not missing_generators else "incomplete",
                "updated": timestamp(), "cnn_completed": len(report["records"]), "cnn_expected": len(selected),
                "generators_completed": len(generators) - len(missing_generators), "generators_expected": len(generators),
                "failures": failures, "missing_generators": missing_generators, "wall_seconds": time.perf_counter() - started})
    write_json(output / "evaluation_cache.json", evaluation_cache.statistics())
    if failures or report["missing_selected_ids"] or missing_generators:
        raise SystemExit(1)
    if sorted(arguments.splits) == list(SPLITS) and not (arguments.probe or arguments.probe_routes):
        from .workbook import generate_workbook

        generate_workbook(output, run)


if __name__ == "__main__":
    main()
