from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from .common import RUN_ROOT, SPLITS, read_json, timestamp, write_json


def repeat_summaries(records, dimensions, metrics):
    groups = {}
    for record in records:
        groups.setdefault(tuple(record[field] for field in dimensions), []).append(record)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        entry = dict(zip(dimensions, key))
        observed = sorted(record["split"] for record in group)
        entry.update({"runs": len(group), "splits": observed, "five_runs_complete": observed == list(SPLITS)})
        for metric in metrics:
            values = [record[metric] for record in group]
            entry[metric + "_mean"] = mean(values) if all(value is not None for value in values) else None
            entry[metric + "_std"] = stdev(values) if len(values) > 1 and all(value is not None for value in values) else None
        summaries.append(entry)
    return summaries


def write_csv(path, records):
    if not records:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    path.write_text(stream.getvalue(), encoding="utf-8-sig")


def macro_auc(scores):
    values = [value for value in scores["auc_valid"] if value is not None]
    return mean(values) if values else None


def repeat_cell(entry, metric, scale=1):
    average, deviation = entry[metric + "_mean"], entry[metric + "_std"]
    if not entry["five_runs_complete"] or average is None or deviation is None:
        return "---"
    return f"{average * scale:.3f} ± {deviation * scale:.3f}"


def generate_report(root):
    coverage_path = root / "coverage.json"
    coverage = read_json(coverage_path) if coverage_path.exists() else None
    records = []
    errors = []
    for path in sorted((root / "jobs").glob("*/result.json")):
        try:
            result = read_json(path)
            job = result["job"]
            cost = result.get("cost", {})
            records.append({
                "id": job["id"], "task": job["track"], "device": job["device"], "channel": job["channel"],
                "method": job["method"], "arm": job.get("arm"), "target": job["target"], "split": job["split"],
                "mode": result.get("mode", "formal"), "updates": result.get("updates_completed"),
                "best_step": result["best_step"], "source_validation_acc": result["source_accuracy"],
                "epochs_completed": result.get("epochs_completed"), "best_epoch": result.get("best_epoch"),
                "stop_reason": result.get("stop_reason"), "training_recipe": result.get("recipe"),
                "parameters_total": cost.get("parameters_total"), "parameters_trainable": cost.get("parameters_trainable"),
                "gflops_counted_per_sample": cost.get("gflops_counted_per_sample"),
                "training_seconds": result["training_seconds"], "validation_seconds": result["validation_seconds"],
                "evaluation_seconds": result["evaluation_seconds"], "wall_seconds": result["wall_seconds"],
                "peak_cuda_bytes": result["peak_cuda_bytes"], "peak_cuda_reserved_bytes": result.get("peak_cuda_reserved_bytes"),
                "initial_weights_sha256": result["initial_weights_sha256"],
                "checkpoint_sha256": result["best_checkpoint_sha256"], "source_identity": result["source_identity"],
                "target_evaluation_windows": result["target"]["total"],
                "target_unpredicted": result["target"]["unpredicted"],
                "target_acc": result["target"]["exact_set_accuracy"],
                "target_macro_f1": result["target"]["macro_f1_valid"],
                "target_macro_auc": macro_auc(result["target"]),
                "target_correct": result["target"]["correct"],
                "seen_acc": result["seen_conditions"]["exact_set_accuracy"],
                "seen_macro_f1": result["seen_conditions"]["macro_f1_valid"],
                "seen_macro_auc": macro_auc(result["seen_conditions"]),
                "diagnostic_metrics": result["target"],
                "result_path": str(path.relative_to(root)),
            })
        except (KeyError, ValueError, TypeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    for path in sorted((root / "jobs").glob("*/failure.json")):
        if not path.with_name("result.json").exists():
            errors.append(read_json(path))
    generator_records = []
    for path in sorted((root / "generators").glob("*/result.json")):
        result = read_json(path)
        job = result["job"]
        generator_records.append({
            "id": job["id"], "task": job["track"], "device": job["device"], "target": job["target"],
            "method": job["method"], "split": job["split"], "parent": job["parent"],
            "mode": result["mode"], "updates_per_phase": result["updates_per_phase"],
            "auxiliary_phases": {name: {phase: history["updates_completed"] if isinstance(history, dict) else len(history) for phase, history in phases.items()}
                                  for name, phases in result["auxiliary_training"].items()},
            "updates_completed": result.get("updates_completed", result["updates_per_phase"]),
            "epochs_completed": result.get("epochs_completed"), "stop_reason": result.get("stop_reason"),
            "training_recipe": result.get("recipe"), "phase_records": result.get("phase_records"),
            "parameters_total": result["cost"]["parameters_total"],
            "parameters_trainable": result["cost"]["parameters_trainable"],
            "forward_gflops": result["cost"]["gflops_counted_per_sample"],
            "sampler_gflops": result["sampler_cost"]["gflops_counted_per_sample"],
            "sampling_steps": result["sampling_steps"], "sampling_nfe": result["sampling_nfe"],
            "training_seconds": result["training_seconds"], "auxiliary_seconds": result["auxiliary_seconds"],
            "validation_seconds": result["validation_seconds"], "wall_seconds": result["wall_seconds"],
            "sampling_seconds_batch_one": result["sampling_seconds_batch_one"],
            "peak_cuda_bytes": result["peak_cuda_bytes"],
            "checkpoint_sha256": result["best_checkpoint_sha256"], "parent_sha256": result["parent_sha256"],
            "source_identity": result["source_identity"], "result_path": str(path.relative_to(root)),
        })
    for path in sorted((root / "generators").glob("*/failure.json")):
        if not path.with_name("result.json").exists():
            errors.append(read_json(path))
    counts = Counter((record["task"], record["method"], record["mode"]) for record in records)
    expected = {job["id"] for job in coverage["selected"]} if coverage else set()
    completed = {record["id"] for record in records}
    generator_coverage_path = root / "generator_coverage.json"
    generator_coverage = read_json(generator_coverage_path) if generator_coverage_path.exists() else {"selected": []}
    expected_generators = {job["id"] for job in generator_coverage["selected"]}
    completed_generators = {record["id"] for record in generator_records}
    missing_generators = sorted(expected_generators - completed_generators)
    methods = read_json(Path(__file__).with_name("method_status.json"))["methods"]
    blocked = coverage["blocked_methods"] if coverage else {name: entry for name, entry in methods.items() if entry["implementation"] != "ready"}
    runtime_path = root / "runtime.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else None
    payload = {"generated": timestamp(), "coverage": coverage,
               "runtime": runtime,
               "executed_short_run_cases_complete": bool(expected) and expected <= completed and not errors and not missing_generators,
               "all_comparison_methods_ready": not blocked,
               "missing_selected_ids": sorted(expected - completed), "missing_generator_ids": missing_generators,
               "errors": errors, "records": records, "generator_records": generator_records}
    summaries = repeat_summaries(records, ("task", "device", "channel", "method", "arm", "target"),
                                 ("target_acc", "target_macro_f1", "target_macro_auc", "seen_acc", "seen_macro_f1", "seen_macro_auc"))
    generator_summaries = repeat_summaries(generator_records, ("task", "device", "method", "target"),
        ("parameters_total", "parameters_trainable", "forward_gflops", "sampler_gflops", "training_seconds",
         "auxiliary_seconds", "validation_seconds", "sampling_seconds_batch_one", "peak_cuda_bytes"))
    payload["five_run_summary"] = summaries
    payload["generator_five_run_summary"] = generator_summaries
    group_path = root / "diagnostic_groups.json"
    diagnostic_groups = read_json(group_path)["records"] if group_path.exists() else []
    payload["t4_diagnostic_groups"] = diagnostic_groups
    workers_path = root / "parallel_workers.json"
    parallel_workers = read_json(workers_path) if workers_path.exists() else None
    payload["parallel_workers"] = parallel_workers
    write_json(root / "report.json", payload)
    write_json(root / "summary.json", {"standard_deviation": "sample standard deviation, ddof=1", "records": summaries})
    write_json(root / "generator_summary.json", {"standard_deviation": "sample standard deviation, ddof=1", "records": generator_summaries})
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "generator_summary.csv", generator_summaries)
    write_csv(root / "generator_report.csv", generator_records)
    lines = ["# FaultCrafter experiment execution report", "", f"Generated: {payload['generated']}", "",
             f"Completed CNNs: {len(records)} / {len(expected)}; completed generators: {len(completed_generators)} / {len(expected_generators)}; failures: {len(errors)}.", "",
             "Repeated runs use seeds 42, 43, 44, 45 and 46. Standard deviations use ddof=1, across those five runs rather than across channels or devices. Each summary retains run counts and missing-run status.", "",
             "| Task | Method | Mode | Completed cases |", "| --- | --- | --- | ---: |"]
    for (task, method, mode), count in sorted(counts.items()):
        lines.append(f"| {task} | {method} | {mode} | {count} |")
    lines.extend(["", "## Five-repeat diagnostic summary", "",
                  "| Task | Method | Data arm | Device | Channel | Condition | Repeats | ACC (%) | Macro-F1 (%) | Macro-AUC (%) | Seen ACC (%) |",
                  "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for entry in summaries:
        cells = [repeat_cell(entry, name, 100) for name in ("target_acc", "target_macro_f1", "target_macro_auc", "seen_acc")]
        lines.append(f"| {entry['task']} | {entry['method']} | {entry['arm'] or ''} | {entry['device']} | {entry['channel']} | {entry['target']} | {entry['runs']} / 5 | " + " | ".join(cells) + " |")
    lines.extend(["", "## Diagnostic measurements", "",
                  "| Task | Device | Channel | Method | Condition | Split | Updates | Target correct / total | Target ACC (%) | Target Macro-F1 (%) | Seen ACC (%) |",
                  "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for record in sorted(records, key=lambda entry: (entry["task"], entry["device"], entry["channel"], entry["method"], entry["target"], entry["split"])):
        lines.append(f"| {record['task']} | {record['device']} | {record['channel']} | {record['method']} | {record['target']} | {record['split']} | {record['updates']} | {record['target_correct']} / {record['target_evaluation_windows']} | {100 * record['target_acc']:.2f} | {100 * record['target_macro_f1']:.2f} | {100 * record['seen_acc']:.2f} |")
    if diagnostic_groups:
        lines.extend(["", "## T4 fault-state groups", "",
                      "State-group columns use exact-set accuracy. False-positive and false-negative rates pool valid component-level confusion counts across all evaluated states; unpredicted samples remain errors in exact-set accuracy.", "",
                      "| Method | Device | Channel | Seed | Normal ACC (%) | Single-fault ACC (%) | Seen-composite ACC (%) | Unseen-composite ACC (%) | Component FNR (%) | Component FPR (%) |",
                      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for entry in diagnostic_groups:
            job = entry["job"]
            values = [entry["groups"][name]["exact_set_accuracy"] if entry["groups"][name] else None
                      for name in ("normal", "single_fault", "seen_composite", "unseen_composite")]
            values.extend(entry["overall"][name] for name in ("false_negative_rate", "false_positive_rate"))
            cells = [f"{100 * value:.2f}" if value is not None else "---" for value in values]
            lines.append(f"| {job['method']} | {job['device']} | {job['channel']} | {job['split']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Generator costs", "",
                  "Parameter totals include auxiliary networks and the GAN discriminator where present. Forward and complete-sampler FLOPs are measured separately at batch one, 1024 samples. Main and auxiliary training durations are separate; T2 durations measure adaptation of the registered parent checkpoint.", "",
                  "| Task | Method | Device | Condition | Seed | Parameters (M) | Forward GFLOPs | Sampler GFLOPs | NFE | Main training (s) | Auxiliary training (s) | Sampling (ms) | Peak VRAM (MiB) |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for record in generator_records:
        lines.append(f"| {record['task']} | {record['method']} | {record['device']} | {record['target']} | {record['split']} | {record['parameters_total'] / 1e6:.4f} | {record['forward_gflops']:.6f} | {record['sampler_gflops']:.6f} | {record['sampling_nfe']} | {record['training_seconds']:.3f} | {record['auxiliary_seconds']:.3f} | {record['sampling_seconds_batch_one'] * 1000:.3f} | {record['peak_cuda_bytes'] / 2**20:.1f} |")
    lines.extend(["", "## Diagnostic CNN costs", "",
                  "Short-run wall time includes checkpoint IO and evaluation; it is not converged training time.", "",
                  "| Method | Cases | Parameters (M) | Counted forward GFLOPs / signal | Median wall seconds | Maximum allocated VRAM (MiB) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for method in sorted({record["method"] for record in records}):
        group = [record for record in records if record["method"] == method]
        parameter_counts = [record["parameters_total"] / 1e6 for record in group if record["parameters_total"] is not None]
        flops = [record["gflops_counted_per_sample"] for record in group if record["gflops_counted_per_sample"] is not None]
        parameters = f"{min(parameter_counts):.4f}-{max(parameter_counts):.4f}" if parameter_counts else "not recorded"
        arithmetic = f"{min(flops):.6f}-{max(flops):.6f}" if flops else "not recorded"
        lines.append(f"| {method} | {len(group)} | {parameters} | {arithmetic} | {median(record['wall_seconds'] for record in group):.3f} | {max(record['peak_cuda_bytes'] for record in group) / 2**20:.1f} |")
    if runtime:
        lines.extend(["", "## Runtime", "", *[f"- {key}: {value}" for key, value in runtime.items()]])
    if parallel_workers:
        lines.extend(["", "## Parallel execution", "", str(parallel_workers),
                      "Completed worker directories are published atomically after their synthetic pools; worker_imports.jsonl retains result hashes. Generator costs were measured before parallel CNN execution. CNN elapsed times retain the observed shared-GPU runtime."])
    lines.extend(["", "## Method implementation status", ""])
    for method, status in blocked.items():
        lines.append(f"- {method}: {status['remaining']}")
    if not blocked:
        lines.append("All five external generator ports are connected. method_sources.json records original repositories, fixed revisions and task adaptations. The coverage files distinguish representative-condition execution from the full registered condition matrix.")
    lines.extend(["", "## Cost convention and records", "",
                  "Parameters include total/trainable counts and storage bytes. Counted FLOPs use two operations per multiply-add, include registered convolution/matrix/attention operations, and exclude normalization, activations, bias, pooling and other unregistered operations. CNN counts are one 1x1x1024 forward; generator records must distinguish one network forward from a complete sampler, including actual NFE and guidance.", "",
                  "Each job retains job.json, cost.json, progress/checkpoint files, target and seen manifests, predictions, and result.json; failures retain tracebacks. runtime.json records the environment. Timings separate training, validation, profiling and evaluation; wall time includes startup and IO. Short-run timings include cold-start effects and cannot directly predict converged training duration.", ""])
    if errors:
        lines.extend(["## Errors", "", *[f"- {entry}" for entry in errors], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    flat = [{key: value for key, value in record.items() if key != "diagnostic_metrics"} for record in records]
    write_csv(root / "report.csv", flat)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_ROOT)
    arguments = parser.parse_args()
    payload = generate_report(arguments.run)
    print(f"Report: {arguments.run / 'report.md'}; records={len(payload['records'])}; errors={len(payload['errors'])}")
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
