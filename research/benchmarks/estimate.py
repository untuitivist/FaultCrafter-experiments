from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

from .common import T1_DEVICES, read_json, timestamp, write_json
from .data import CHANNELS, T4_STATES
from .recipes import CNN_RECIPE, classifier_window_count, generator_recipe, phase_budget
from .comparison_design import execution_plan


def training_budgets(data_run, jobs):
    manifests = {}
    result = []
    for job in jobs:
        split = job["split"]
        if split not in manifests:
            manifests[split] = read_json(data_run / "cache/windows" / str(split) / "manifest.json")["cells"]
        devices = T1_DEVICES if job["kind"] == "generator" and job["track"] == "T1" else (job["device"],)
        cells = [cell for cell in manifests[split] if cell["role"] == "train" and cell["device"] in devices
                 and cell["state"] not in T4_STATES.get(cell["device"], ())
                 and not (cell["device"] == job["device"] and cell["condition"] == job["target"])]
        real = sum(cell["count"] * (len(CHANNELS[cell["device"]]) if job["kind"] == "generator" else 1) for cell in cells)
        if job["kind"] == "generator":
            recipe = generator_recipe(job["method"])
            unique_conditions = sum(len(CHANNELS[cell["device"]]) for cell in cells)
            phases = {"main": phase_budget(recipe, real)}
            for phase in recipe["auxiliaries"]:
                count = unique_conditions if phase["name"].endswith("label_to_embedding") else real
                phases[phase["name"]] = {**phase_budget(phase, count), "batch_size": phase["batch_size"]}
            result.append({"id": job["id"], "kind": "generator", "method": job["method"], "device": job["device"],
                           "split": split, "source_windows": real, "recipe": recipe, "phases": phases})
        else:
            states = len({cell["state"] for cell in cells})
            additional = (252 if job["track"] == "T3" else 63) * states
            if job["track"] == "T4":
                additional = 63 * len(T4_STATES[job["device"]])
            count = classifier_window_count(real, additional, job)
            result.append({"id": job["id"], "kind": "cnn", "method": job["method"], "device": job["device"],
                           "split": split, "source_windows": real, **phase_budget(CNN_RECIPE, count)})
    return result


def estimate(run, data_run, cnn_epochs=(25, 50, 100)):
    jobs = execution_plan(data_run)["jobs"]
    budgets = training_budgets(data_run, jobs)
    generator_counts = Counter(job["method"] for job in jobs if job["kind"] == "generator")
    cnn_timings = {}
    for path in (run / "timing").glob("*/timing_only.json"):
        entry = read_json(path)
        if entry.get("recipe") == CNN_RECIPE:
            cnn_timings[entry["device"], entry["method"]] = entry
    completed_cnn_timings = {}
    completed_cnns = []
    for path in (run / "jobs").glob("*/result.json"):
        entry = read_json(path)
        if entry.get("mode") != "formal" or entry.get("recipe") != CNN_RECIPE:
            continue
        if entry["training_seconds"] <= 0 or entry["updates_completed"] <= 0 or entry["epochs_completed"] <= 0:
            continue
        job = entry["job"]
        completed_cnn_timings.setdefault((job["device"], job["method"]), []).append(entry)
        completed_cnns.append({"id": job["id"], "device": job["device"], "method": job["method"],
                              "track": job["track"], "arm": job["arm"], "epochs": entry["epochs_completed"],
                              "updates": entry["updates_completed"], "best_epoch": entry["best_epoch"],
                              "stop_reason": entry["stop_reason"], "wall_seconds": entry["wall_seconds"]})
    for key, entries in completed_cnn_timings.items():
        cnn_timings[key] = {"updates_per_second": sum(entry["updates_completed"] for entry in entries) /
                                                 sum(entry["training_seconds"] for entry in entries),
                            "validation_seconds": sum(entry["validation_seconds"] for entry in entries) /
                                                  sum(entry["epochs_completed"] for entry in entries),
                            "observed_epochs": [entry["epochs_completed"] for entry in entries],
                            "source": "completed_full_runs", "job_ids": [entry["job"]["id"] for entry in entries]}
    phase_rates = {}
    observed = {}
    completed_generators = []
    for path in (run / "generators").glob("*/result.json"):
        entry = read_json(path)
        expected = generator_recipe(entry["job"]["method"])
        actual = entry.get("recipe", {})
        if any(actual.get(key) != expected.get(key) for key in ("optimizer", "batch_size", "microbatch_size", "native_length")):
            continue
        if entry.get("mode") == "formal":
            completed_generators.append({"id": entry["job"]["id"], "method": entry["job"]["method"],
                "track": entry["job"]["track"], "device": entry["job"]["device"], "recipe": actual,
                "phase_records": entry["phase_records"], "wall_seconds": entry["wall_seconds"],
                "sampling_seconds_batch_one": entry["sampling_seconds_batch_one"]})
        for phase, receipt in entry.get("phase_records", {}).items():
            if receipt["updates_completed"] > 0:
                key = (entry["job"]["method"], phase)
                observed.setdefault(key, []).append(receipt["training_seconds"] / receipt["updates_completed"])
    phase_rates = {key: mean(values) for key, values in observed.items()}
    missing = set()
    generator_seconds = 0.0
    methods = {}
    for budget in budgets:
        if budget["kind"] != "generator":
            continue
        method = budget["method"]
        record = methods.setdefault(method, {"jobs": generator_counts[method], "main_updates_max": 0, "auxiliary_updates_max": 0,
                                            "effective_batch_size": budget["recipe"]["batch_size"],
                                            "max_epochs": budget["recipe"].get("max_epochs"),
                                            "provenance": budget["recipe"]["provenance"]})
        for phase, plan in budget["phases"].items():
            record["main_updates_max" if phase == "main" else "auxiliary_updates_max"] += plan["max_updates"]
            rate = phase_rates.get((method, phase))
            if rate is None:
                missing.add("generator/" + method + "/" + phase)
            else:
                generator_seconds += plan["max_updates"] * rate
    scenarios = []
    for epochs in (*cnn_epochs, None):
        training = validation = 0.0
        scenario_missing = set(missing)
        for budget in budgets:
            if budget["kind"] != "cnn":
                continue
            timing = cnn_timings.get((budget["device"], budget["method"]), cnn_timings.get((budget["device"], "real_only")))
            if timing is None:
                missing.add("cnn/" + budget["device"])
                scenario_missing.add("cnn/" + budget["device"])
                continue
            applied_epochs = epochs
            if applied_epochs is None:
                if not timing.get("observed_epochs"):
                    scenario_missing.add("cnn_full_run/" + budget["device"])
                    continue
                applied_epochs = median(timing["observed_epochs"])
            training += applied_epochs * budget["updates_per_epoch"] / timing["updates_per_second"]
            validation += applied_epochs * timing["validation_seconds"]
        known = not scenario_missing
        days = (generator_seconds + training + validation) / 86400 if known else None
        scenarios.append({"cnn_epochs_assumed": epochs if epochs is not None else "median_of_completed_full_runs",
                          "missing_measurements": sorted(scenario_missing),
                          "generator_training_seconds_max": generator_seconds if known else None,
                          "cnn_training_seconds": training if known else None, "cnn_validation_seconds": validation if known else None,
                          "training_days_4070ti": days,
                          "training_days_four_3090": [{"single_card_speed_relative_to_4070ti": relative,
                              "parallel_efficiency": 0.85, "days": days / (4 * relative * 0.85) if known else None}
                              for relative in (0.7, 0.85, 1.0)]})
    payload = {"created": timestamp(), "scope": "Registered comparison matrix; method-specific training schedules",
               "generator_jobs": sum(generator_counts.values()), "cnn_jobs": sum(job["kind"] == "cnn" for job in jobs),
               "generator_budgets": methods, "cnn_recipe": CNN_RECIPE,
               "cnn_updates_max": sum(budget["max_updates"] for budget in budgets if budget["kind"] == "cnn"),
               "scenarios": scenarios, "missing_measurements": sorted(missing),
               "completed_full_cnn_runs": completed_cnns, "completed_full_generator_runs": completed_generators,
               "timing_status": "needs_matching_batch_and_recipe_measurements" if missing else "scenario_projection",
               "limits": ["Epoch counts in the scenarios are assumptions, not predictions of early stopping.",
                          "Old batch-eight checks and the retired fixed-300-epoch CNN timings are not valid timings for the revised recipes.",
                          "The generator budgets include separate auxiliary phases and the full published preset; early stopping can shorten Time Weaver.",
                          "The observed-epoch scenario uses device/method medians and falls back to that device's Real-only run when the exact method is unmeasured.",
                          "Sampling, source validation of generators, cost profiling, checkpoint IO, figure export, compression and transfer are additional.",
                          "3090 speeds and parallel efficiency are scenarios until measured on the server."]}
    write_json(run / "eta_projection.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Estimate the revised method-specific comparison schedules without artificial point-count units.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--cnn-epochs", nargs="+", type=int, default=[25, 50, 100])
    arguments = parser.parse_args()
    if any(epochs < 1 or epochs > CNN_RECIPE["max_epochs"] for epochs in arguments.cnn_epochs):
        raise ValueError("CNN epoch scenarios must lie between 1 and the configured maximum")
    result = estimate(arguments.run, arguments.data_run, tuple(arguments.cnn_epochs))
    print({key: result[key] for key in ("generator_jobs", "cnn_jobs", "generator_budgets", "timing_status", "missing_measurements")})


if __name__ == "__main__":
    main()
