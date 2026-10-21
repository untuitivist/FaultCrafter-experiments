from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

from .common import ROOT, RUN_ROOT, file_hash, read_json, timestamp, write_json
from .queue import QueueLock
from .report import generate_report
from .tasks import IMPLEMENTED_CONTROLS, T3_TARGETS


def select_cases(plan):
    selected = {}
    candidates = sorted(plan["jobs"], key=lambda job: (job["target"] != T3_TARGETS.get(job["device"]), job["split"], job["id"]))
    for job in candidates:
        if job["kind"] != "cnn" or job["method"] not in IMPLEMENTED_CONTROLS:
            continue
        key = (job["track"], job["device"], job["channel"], job["method"], job["arm"])
        selected.setdefault(key, job)
    return sorted(selected.values(), key=lambda job: (job["device"], job["channel"], job["target"], job["split"], job["track"], job["method"]))


def main():
    parser = argparse.ArgumentParser(description="Isolated end-to-end short runs; never starts the full queue.")
    parser.add_argument("--data-run", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, choices=(2, 3), default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-jobs", type=int)
    arguments = parser.parse_args()
    if arguments.data_run.resolve() == arguments.output.resolve():
        raise ValueError("Short-run output must be separate from formal results")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with QueueLock(arguments.output):
        exit_code = execute(arguments)
    if exit_code:
        raise SystemExit(exit_code)


def execute(arguments):
    run, output = arguments.data_run.resolve(), arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(run / "plan.json")
    methods = read_json(Path(__file__).with_name("method_status.json"))
    selected = select_cases(plan)
    if arguments.max_jobs is not None:
        selected = selected[:arguments.max_jobs]
    coverage = {"created": timestamp(), "mode": "smoke_only", "updates": arguments.updates,
                "sampling": "One registered condition/split per task-device-channel-method-arm route; not every condition/split and not a formal experimental result.",
                "data_run": str(run), "plan_sha256": file_hash(run / "plan.json"), "formal_jobs": len(plan["jobs"]),
                "selected": selected, "blocked_methods": {name: entry for name, entry in methods["methods"].items() if entry["implementation"] != "ready"}}
    existing = output / "coverage.json"
    if existing.exists():
        previous = read_json(existing)
        if any(previous.get(key) != coverage[key] for key in ("updates", "plan_sha256", "selected", "blocked_methods")):
            raise ValueError("Existing short-run coverage differs; choose another output directory")
    write_json(existing, coverage)
    from .classifier import configure_runtime, run_classifier
    import numpy
    import scipy
    import sklearn
    import torch

    runtime = configure_runtime(arguments.threads)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    write_json(output / "runtime.json", {**runtime, "python": sys.version, "executable": sys.executable,
                                        "platform": platform.platform(), "git_revision": revision,
                                        "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                                        "cudnn": torch.backends.cudnn.version(),
                                        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "created": timestamp()})
    failures = 0
    try:
        for job in selected:
            if (output / "STOP_AFTER_CURRENT").exists():
                break
            write_json(output / "status.json", {"state": "smoke_running", "pid": os.getpid(), "current": job, "updated": timestamp()})
            try:
                run_classifier(run, job, output_root=output, smoke_updates=arguments.updates)
            except Exception as error:
                failures += 1
                write_json(output / "jobs" / job["id"] / "failure.json", {"job": job, "error": str(error), "traceback": traceback.format_exc(), "updated": timestamp()})
                print(traceback.format_exc(), flush=True)
            finally:
                generate_report(output)
    finally:
        report = generate_report(output)
        write_json(output / "status.json", {"state": "short_runs_complete_methods_pending" if report["executed_short_run_cases_complete"] else "short_runs_incomplete",
                                            "updated": timestamp(), "completed": len(report["records"]), "failures": failures,
                                            "blocked_methods": list(coverage["blocked_methods"])})
    if failures or report["missing_selected_ids"]:
        return 1
    if coverage["blocked_methods"]:
        print("Implemented controls completed; generator methods remain blocked. Full comparison readiness has NOT passed.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    main()
