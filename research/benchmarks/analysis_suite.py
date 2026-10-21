from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch

from .classifier import configure_runtime
from .common import METHODS, ROOT, RUN_ROOT, SPLITS, file_hash, identity, log, read_json, timestamp, write_json
from .condition_analysis import evaluate_case
from .controlled import COMPARISONS, PANELS, SAMPLING_CALLS, VARIANTS, run_case
from .export import Transcript
from .queue import QueueLock
from .tasks import CONDITION_SPLITS


def execute(data_run, comparisons, output, *, updates=3, steps=3, count=32, threads=4):
    runtime = configure_runtime(threads)
    selected = read_json(comparisons / "coverage.json")["selected"]
    references = {tuple(job[key] for key in ("track", "device", "channel", "target", "split")): job["id"]
                  for job in selected if job["method"] == "real_only"}
    control_jobs = sorted([job for job in selected if job["track"] == "T2" and job["device"] == "bjtu"
                          and job["channel"] == "CH17" and job["method"] in METHODS and job["arm"] == "real_generated"
                          and job["split"] in CONDITION_SPLITS],
                          key=lambda job: (job["split"], job["method"]))
    if len(control_jobs) != len(METHODS) * len(CONDITION_SPLITS):
        raise RuntimeError("The BJTU panel requires all five methods at seed 42")
    for split in CONDITION_SPLITS:
        reference_job = next(job for job in selected if job["track"] == "T2" and job["device"] == "bjtu"
                             and job["channel"] == "CH17" and job["split"] == split and job["method"] == "real_only")
        control_jobs.append({**reference_job, "method": "superposition", "parent": None,
                             "id": identity({"analysis": "bjtu_condition_superposition", "split": split})[:24]})
    configuration = {"data_run": str(data_run), "comparisons": str(comparisons), "updates": updates,
                     "condition_sampling_steps": steps, "controlled_sampling_nfe": SAMPLING_CALLS,
                     "samples_per_request": count, "splits": list(SPLITS), "condition_splits": list(CONDITION_SPLITS),
                     "controlled_comparisons": COMPARISONS, "controlled_panels": PANELS,
                     "variants": VARIANTS, "condition_jobs": control_jobs}
    if (output / "plan.json").exists():
        previous = read_json(output / "plan.json")
        current = read_json_compatible(configuration)
        if any(previous.get(key) != current[key] for key in current if key != "condition_jobs"):
            raise RuntimeError("Analysis plan changed; use a fresh output directory")
        previous_jobs = {job["id"]: job for job in previous["condition_jobs"]}
        current_jobs = {job["id"]: job for job in current["condition_jobs"]}
        if any(current_jobs.get(identifier) != job for identifier, job in previous_jobs.items()):
            raise RuntimeError("Previously registered analysis jobs changed")
    write_json(output / "plan.json", configuration)
    write_json(output / "runtime.json", runtime)
    write_json(output / "status.json", {"state": "running", "updated": timestamp()})
    controlled, conditions = [], []
    for split in SPLITS:
        for device in PANELS:
            for variant, prediction, group in VARIANTS:
                log("analysis_case", kind="controlled", device=device, split=split, variant=variant)
                controlled.append(run_case(data_run, comparisons, output, device, split, variant, prediction, group,
                                           updates, SAMPLING_CALLS, references))
    for job in control_jobs:
        log("analysis_case", kind="condition", method=job["method"], split=job["split"])
        key = tuple(job[name] for name in ("track", "device", "channel", "target", "split"))
        conditions.append(evaluate_case(data_run, comparisons, output, job, references[key], steps, count))
    write_json(output / "summary.json", {"controlled": controlled, "conditions": conditions,
               "configuration": configuration, "completed": timestamp()})
    write_json(output / "status.json", {"state": "complete", "controlled": len(controlled),
               "conditions": len(conditions), "updated": timestamp()})
    write_json(comparisons / "analysis.json", {"directory": str(output), "summary_sha256": file_hash(output / "summary.json")})
    from .workbook import generate_workbook

    artifact_runtime = os.environ.get("FAULTCRAFTER_ARTIFACT_RUNTIME")
    generate_workbook(comparisons, data_run, artifact_runtime=Path(artifact_runtime) if artifact_runtime else None)


def read_json_compatible(value):
    import json

    return json.loads(json.dumps(value))


def main():
    parser = argparse.ArgumentParser(description="Run paired controlled analyses and reuse frozen BJTU T2 generators.")
    parser.add_argument("--data-run", type=Path, default=RUN_ROOT)
    parser.add_argument("--comparisons", type=Path, default=ROOT / "outputs/comparisons_all_methods_5seeds_v3_20261020")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_hdu_dirg_bjtu_nfe32_single_scans")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3,
                        help="BJTU frozen-generator sampling steps; controlled comparisons always use 32 network calls")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.updates < 2 or not 2 <= arguments.steps <= 1000 or arguments.samples < 2:
        raise ValueError("Use positive sample budgets, at least two updates and 2-1000 sampling steps")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with QueueLock(arguments.output), (arguments.output / "execution.log").open("a", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Transcript(sys.stdout, logfile)), contextlib.redirect_stderr(Transcript(sys.stderr, logfile)):
            execute(arguments.data_run.resolve(), arguments.comparisons.resolve(), arguments.output.resolve(),
                    updates=arguments.updates, steps=arguments.steps, count=arguments.samples, threads=arguments.threads)


if __name__ == "__main__":
    main()
