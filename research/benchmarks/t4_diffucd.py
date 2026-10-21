from __future__ import annotations

import argparse
import gc
import shutil
from collections import Counter
from pathlib import Path

import torch

from .common import file_hash, identity, log, read_json, timestamp, write_json
from .data import T4_STATES
from .tasks import execution_plan as base_execution_plan


def execution_plan(run):
    plan = base_execution_plan(run)
    jobs = list(plan["jobs"])
    parents = {(job["device"], job["target"], job["split"]): job
               for job in jobs if job["kind"] == "generator" and job["method"] == "diffucd"}
    existing = {job["id"] for job in jobs}
    for template in plan["jobs"]:
        if template["kind"] != "cnn" or template["track"] != "T4" or template["method"] != "iccdm":
            continue
        parent = parents[template["device"], template["target"], template["split"]]
        job = {key: value for key, value in template.items() if key not in ("id", "implementation")}
        job.update(method="diffucd", parent=parent["id"])
        job["id"] = identity(job)[:24]
        job["implementation"] = "requires_method_port"
        if job["id"] not in existing:
            jobs.append(job)
            existing.add(job["id"])
    design = {**plan["execution_design"], "t4_generators": ["diffucd", "iccdm", "flowts"]}
    return {**plan, "jobs": jobs, "execution_design": design, "design_sha256": identity(design),
            "counts": dict(sorted(Counter(f"{job['kind']}/{job['track']}/{job['method']}" for job in jobs).items()))}


def reuse_generator(source, output, parent, data_run):
    from .generators import ConditionSchema

    source_folder = source / "generators" / parent
    result = read_json(source_folder / "result.json")
    manifest = read_json(source_folder / "source_manifest.json")
    checkpoint_path = source_folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schema = ConditionSchema(read_json(data_run / "inventory.json")["records"])
    if identity(checkpoint["schema"]) != identity(schema.receipt()):
        raise ValueError("Frozen generator condition schema differs from the current inventory")
    if (checkpoint["identity"] != result["identity"]
            or checkpoint["source_identity"] != result["source_identity"]
            or file_hash(checkpoint_path) != result["best_checkpoint_sha256"]):
        raise ValueError("Frozen checkpoint and result identities differ")
    job = result["job"]
    if job["id"] != parent or job["method"] != "diffucd":
        raise ValueError("Expected the registered DiffUCD generator")
    cells = [cell for role in ("train", "validation") for cell in manifest[role]]
    for cell in cells:
        if cell["state"] in T4_STATES.get(cell["device"], ()):
            raise ValueError("Held-out fault combination present in generator sources")
        if cell["device"] == job["device"] and cell["condition"] == job["target"]:
            raise ValueError("Held-out working condition present in generator sources")
    known = {part for cell in cells if cell["device"] == job["device"]
             for part in schema.labels[cell["device"], cell["state"]]}
    for state in T4_STATES[job["device"]]:
        if not set(schema.labels[job["device"], state]) <= known:
            raise ValueError("Requested combination contains an unseen fault component")
    destination = output / "generators" / parent
    destination.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name in ("best.pt", "result.json", "job.json", "source_manifest.json"):
        original, copied = source_folder / name, destination / name
        original_hash = file_hash(original)
        if copied.exists() and file_hash(copied) != original_hash:
            raise ValueError("Imported generator differs from the frozen source: " + str(copied))
        if not copied.exists():
            shutil.copy2(original, copied)
        if file_hash(copied) != original_hash:
            raise ValueError("Frozen generator copy verification failed")
        receipts.append({"source": str(original), "destination": str(copied), "sha256": original_hash})
    return {"parent": parent, "identity": result["identity"], "mode": result["mode"],
            "held_combinations_excluded": True, "files": receipts}


def main():
    from .classifier import configure_runtime, run_classifier
    from .comparison_smoke import select_routes
    from .diagnostic_groups import evaluate
    from .evaluation_cache import EvaluationCache
    from .generators import synthetic_pool
    from .queue import QueueLock
    from .report import generate_report

    parser = argparse.ArgumentParser(description="Supplement T4 with frozen DiffUCD generators from a registered comparison run.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.35)
    arguments = parser.parse_args()
    source, output = arguments.source_run.resolve(), arguments.output.resolve()
    if output == source or not 0 < arguments.cuda_memory_fraction <= 1:
        raise ValueError("Use a separate output directory and a CUDA memory fraction in (0, 1]")
    original_coverage = read_json(source / "coverage.json")
    data_run = Path(original_coverage["data_run"])
    if output == data_run.resolve():
        raise ValueError("The supplement output must not overwrite prepared data")
    plan = execution_plan(data_run)
    original_routes = {(job["device"], job["channel"], job["target"], job["split"])
                       for job in original_coverage["selected"] if job["track"] == "T4" and job["method"] == "iccdm"}
    selected = [job for job in select_routes(plan) if job["track"] == "T4" and job["method"] == "diffucd"
                and (job["device"], job["channel"], job["target"], job["split"]) in original_routes]
    if len(selected) != len(original_routes) or not selected:
        raise ValueError("Supplement does not exactly match the original T4 device/channel/condition/seed routes")
    coverage = {**original_coverage, "created": timestamp(), "selected": selected,
                "source_run": str(source), "source_coverage_sha256": file_hash(source / "coverage.json"),
                "sampling": "T4 DiffUCD supplement, matching the original iCCDM diagnostic routes; frozen generator reuse.",
                "execution_design": plan["execution_design"], "design_sha256": plan["design_sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    with QueueLock(output):
        if (output / "coverage.json").exists():
            previous = read_json(output / "coverage.json")
            for key in ("selected", "source_coverage_sha256", "updates", "sampling_steps", "design_sha256"):
                if previous[key] != coverage[key]:
                    raise ValueError("Supplement coverage changed; use a new output directory")
            coverage["created"] = previous["created"]
        write_json(output / "coverage.json", coverage)
        runtime = configure_runtime(arguments.threads)
        torch.cuda.set_per_process_memory_fraction(arguments.cuda_memory_fraction)
        write_json(output / "runtime.json", {**runtime, "cuda_memory_fraction": arguments.cuda_memory_fraction,
                   "concurrent_training_possible": True, "created": timestamp()})
        cache = EvaluationCache()
        cache.install()
        imports = [reuse_generator(source, output, parent, data_run) for parent in sorted({job["parent"] for job in selected})]
        write_json(output / "generator_imports.json", {"records": imports, "created": timestamp()})
        for index, job in enumerate(selected):
            write_json(output / "status.json", {"state": "running", "position": index + 1,
                       "total": len(selected), "current": job, "updated": timestamp()})
            generated = synthetic_pool(data_run, job, output, coverage["sampling_steps"])
            run_classifier(data_run, job, output_root=output, smoke_updates=coverage["updates"], generated=generated)
            gc.collect()
            torch.cuda.empty_cache()
        groups = evaluate(output)
        report = generate_report(output)
        if report["missing_selected_ids"] or len(groups) != len(selected):
            raise RuntimeError("Incomplete T4 DiffUCD supplement")
        write_json(output / "evaluation_cache.json", cache.statistics())
        write_json(output / "status.json", {"state": "complete", "cnn_completed": len(selected),
                   "generators_reused": len(imports), "mode": coverage["mode"], "updated": timestamp()})
        log("t4_diffucd_complete", cnns=len(selected), generators_reused=len(imports), mode=coverage["mode"])


if __name__ == "__main__":
    main()
