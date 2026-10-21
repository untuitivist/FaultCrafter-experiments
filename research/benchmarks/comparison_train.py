from __future__ import annotations

import argparse
from pathlib import Path

from .common import RUN_ROOT, SPLITS, timestamp, write_json
from .queue import QueueLock
from .recipes import CNN_RECIPE, GENERATOR_RECIPES
from .selection import dependency_order, representative_selection
from .comparison_design import execution_plan


def main():
    parser = argparse.ArgumentParser(description="Train external comparators with their registered method-specific schedules and epoch-based CNN early stopping.")
    parser.add_argument("--data-run", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", type=int, choices=SPLITS)
    parser.add_argument("--representative", action="store_true", help="Complete representative runs for runtime calibration; defaults to seed 42.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args()
    arguments.splits = arguments.splits or ([42] if arguments.representative else list(SPLITS))
    if arguments.output.resolve() == arguments.data_run.resolve():
        raise ValueError("Use a new output directory for the revised training protocol")
    if not 0 < arguments.cuda_memory_fraction <= 1 or len(set(arguments.splits)) != len(arguments.splits):
        raise ValueError("Use unique seeds and a CUDA memory fraction within (0, 1]")
    arguments.output.mkdir(parents=True, exist_ok=True)
    plan = execution_plan(arguments.data_run)
    selected, selection = (representative_selection(plan, arguments.splits) if arguments.representative else
                           ([job for job in plan["jobs"] if job["kind"] == "cnn" and job["split"] in arguments.splits], None))
    jobs, generators = dependency_order(selected, {job["id"]: job for job in plan["jobs"]})
    arguments.formal, arguments.all_conditions = True, True
    arguments.updates = arguments.sampling_steps = arguments.publish_to = None
    arguments.probe = arguments.probe_routes = False
    from .comparison_smoke import execute

    with QueueLock(arguments.output):
        from .estimate import training_budgets

        write_json(arguments.output / "training_recipes.json", {"created": timestamp(), "generators": GENERATOR_RECIPES, "cnn": CNN_RECIPE})
        write_json(arguments.output / "training_budgets.json", training_budgets(arguments.data_run, jobs))
        write_json(arguments.output / "selection.json", {"created": timestamp(), "selection": selection,
                   "execution_design": plan["execution_design"], "design_sha256": plan["design_sha256"],
                   "cnn_jobs": len(selected), "generator_jobs": len(generators), "execution_order": jobs})
        if arguments.plan_only:
            print({"jobs": len(jobs), "generators": len(generators), "cnns": len(selected),
                   "seeds": arguments.splits, "output": str(arguments.output), "training_started": False}, flush=True)
            return
        execute(arguments, selected_jobs=selected, selection_description=selection)


if __name__ == "__main__":
    main()
