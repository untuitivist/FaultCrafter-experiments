from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from .common import RUN_ROOT, log, read_json, timestamp, write_json
from .tasks import IMPLEMENTED_CONTROLS, build_plan

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class QueueLock:
    def __init__(self, run):
        self.path = run / "queue.lock"

    def __enter__(self):
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if not self.stream.read(1):
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise
        return self

    def __exit__(self, *arguments):
        if arguments[0] is not None:
            status_path = self.path.parent / "status.json"
            status = read_json(status_path) if status_path.exists() else {}
            write_json(status_path, {**status, "state": "failed", "updated": timestamp(),
                                     "error": str(arguments[1]), "traceback": "".join(traceback.format_exception(*arguments))})
        self.stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def estimate(run, plan, timings):
    from .estimate import training_budgets
    from .recipes import CNN_RECIPE

    lookup = {(entry["device"], entry["method"]): entry for entry in timings if entry.get("recipe") == CNN_RECIPE}
    budgets = training_budgets(run, [job for job in plan["jobs"] if job["kind"] == "cnn" and job["method"] in IMPLEMENTED_CONTROLS])
    scenarios = []
    for epochs in (25, 50, 100):
        training = validation = 0.0
        for budget in budgets:
            timing = lookup.get((budget["device"], budget["method"]), lookup.get((budget["device"], "real_only")))
            if timing is None:
                raise RuntimeError("Missing timing for the current CNN recipe: " + budget["device"])
            training += epochs * budget["updates_per_epoch"] / timing["updates_per_second"]
            validation += epochs * timing["validation_seconds"]
        scenarios.append({"epochs_assumed": epochs, "training_hours": training / 3600,
                          "validation_hours": validation / 3600})
    payload = {"time": timestamp(), "scope": "Non-generative controls; epoch counts are planning assumptions",
               "cnn_jobs": len(budgets), "cnn_recipe": CNN_RECIPE, "scenarios": scenarios,
               "unmeasured": "Method-specific external generators, sampling and dependent CNNs",
               "timings": timings}
    write_json(run / "eta.json", payload)
    log("eta", cnn_jobs=len(budgets), scenarios=scenarios)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    arguments = parser.parse_args()
    run = arguments.run
    run.mkdir(parents=True, exist_ok=True)
    with QueueLock(run):
        write_json(run / "status.json", {"state": "preparing", "pid": os.getpid(), "updated": timestamp()})
        if arguments.prepare:
            from .data import prepare

            prepare(run)
        if not (run / "data_ready.json").exists():
            raise RuntimeError("Source cache not ready")
        plan = build_plan(run)
        from .preflight import verify

        verify(run, plan)
        from .classifier import configure_runtime, run_classifier
        from .report import generate_report

        runtime = configure_runtime(arguments.threads)
        write_json(run / "runtime.json", runtime)
        log("runtime", **runtime)
        ready = [job for job in plan["jobs"] if job["kind"] == "cnn" and job["method"] in IMPLEMENTED_CONTROLS]
        ready_ids = {job["id"] for job in ready}
        blocked = [job for job in plan["jobs"] if job["id"] not in ready_ids]
        write_json(run / "blocked.json", {"reason": "This legacy control-only entry does not run external methods; use comparison_train for the complete matrix", "jobs": blocked})
        timings = []
        for device in ("hdu", "dirg", "hust", "cwru", "bjtu", "hit_sq", "hit_self", "pu"):
            job = next(item for item in ready if item["device"] == device and item["method"] == "real_only" and item["track"] in ("T1", "T2"))
            timings.append(run_classifier(run, job, timing_only=True))
        job = next(item for item in ready if item["device"] == "dirg" and item["method"] == "mixstyle")
        timings.append(run_classifier(run, job, timing_only=True))
        estimate(run, plan, timings)
        if arguments.profile_only:
            return
        completed = 0
        for job in ready:
            if (run / "STOP_AFTER_CURRENT").exists():
                write_json(run / "status.json", {"state": "stopped_after_current", "updated": timestamp(), "completed": completed})
                return
            write_json(run / "status.json", {"state": "running", "pid": os.getpid(), "updated": timestamp(), "current": job,
                                             "ready_jobs": len(ready), "blocked_jobs": len(blocked), "completed_this_session": completed})
            try:
                result = run_classifier(run, job)
            except Exception as error:
                write_json(run / "jobs" / job["id"] / "failure.json", {"job": job, "error": str(error), "traceback": traceback.format_exc(), "updated": timestamp()})
                write_json(run / "status.json", {"state": "failed", "pid": os.getpid(), "updated": timestamp(), "current": job,
                                                 "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                raise
            finally:
                generate_report(run)
            completed += 1
            if arguments.max_jobs is not None and completed >= arguments.max_jobs:
                break
        write_json(run / "status.json", {"state": "controls_complete_external_methods_use_comparison_train", "updated": timestamp(),
                                         "completed_this_session": completed, "blocked_jobs": len(blocked)})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("queue_failed", error=f"{type(error).__name__}: {error}")
        raise
