from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from faultcrafter.models import CNN1D
from . import classifier
from .common import file_hash, identity, log, read_json, seed_for, timestamp, write_json
from .pools import source_pool
from .recipes import CNN_RECIPE, EarlyStopping


CORAL_RECIPE = {
    "version": "coral-source-only-v1", "alignment_weight": 1.0,
    "reference": "https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py",
    "hyperparameters": "https://github.com/facebookresearch/DomainBed/blob/main/domainbed/hparams_registry.py",
    "penalty": "Pairwise source-domain feature mean MSE plus unbiased covariance MSE; equal domain-pair weights",
    "features": "Flattened output of the existing six-block CNN, before its unchanged classification head",
    "sampling": "64 samples per update, allocated equally across source conditions; random within condition",
    "supervision": "Mean component-wise BCE per source condition, then equal condition average",
    "selection": "Source validation exact-set accuracy only; held target signals inaccessible before freezing",
}


def mean_covariance_loss(first, second):
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1] or min(len(first), len(second)) < 2:
        raise ValueError("CORAL requires at least two samples per domain and equal feature dimensions")
    mean_first, mean_second = first.mean(0, keepdim=True), second.mean(0, keepdim=True)
    centered_first, centered_second = first - mean_first, second - mean_second
    first_norm = (centered_first @ centered_first.T).square().sum() / (len(first) - 1) ** 2
    second_norm = (centered_second @ centered_second.T).square().sum() / (len(second) - 1) ** 2
    cross_norm = (centered_first @ centered_second.T).square().sum() / ((len(first) - 1) * (len(second) - 1))
    covariance = (first_norm + second_norm - 2 * cross_norm).clamp_min(0) / first.shape[1] ** 2
    return (mean_first - mean_second).square().mean() + covariance


def source_batches(groups, batch_size, generator):
    if len(groups) < 2 or batch_size < 2 * len(groups):
        raise ValueError("CORAL requires multiple source conditions and at least two samples from each per update")
    order = torch.randperm(len(groups), device=groups[0].device, generator=generator).tolist()
    quota, remainder = divmod(batch_size, len(groups))
    return [groups[domain][torch.randint(len(groups[domain]), (quota + (position < remainder),),
                                       device=groups[domain].device, generator=generator)]
            for position, domain in enumerate(order)]


def features_and_logits(model, signals):
    hidden = signals
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6):
        hidden = layer(hidden)
    features = hidden.flatten(start_dim=1)
    return features, model.classifier(features)


def run_coral_classifier(run, job, *, output_root=None, smoke_updates=None):
    if job["track"] not in ("T1", "T2") or job["parent"] is not None or job["arm"] is not None:
        raise ValueError("CORAL is a source-only T1/T2 diagnostic comparator")
    if smoke_updates is not None and (smoke_updates < 2 or output_root is None or output_root.resolve() == run.resolve()):
        raise ValueError("Short CORAL runs require at least two updates and a separate output directory")
    folder = (output_root or run) / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    pool = source_pool(run, job)
    base_identity, code = classifier.training_identity(job, pool)
    code = {**code, "coral.py": file_hash(Path(__file__))}
    run_identity = identity({"classifier": base_identity, "coral_code": code["coral.py"],
                             "recipe": CORAL_RECIPE, "updates": smoke_updates})
    if (folder / "result.json").exists():
        result = read_json(folder / "result.json")
        if result["identity"] != run_identity:
            raise ValueError("CORAL code or source identity changed; use a new output directory")
        return result
    started = time.perf_counter()
    random.seed(job["split"])
    np.random.seed(job["split"])
    torch.manual_seed(job["split"])
    torch.cuda.manual_seed_all(job["split"])
    model = CNN1D(len(pool["components"])).cuda()
    initial_hash = classifier.state_hash(model)
    train = pool["roles"]["train"]
    conditions = train["conditions"]
    if job["target"] in conditions:
        raise ValueError("Held target condition present in CORAL training sources")
    signals, labels = (torch.as_tensor(train[key], device="cuda") for key in ("x", "y"))
    domains = torch.as_tensor(train["unit"] % len(conditions), device="cuda")
    groups = [torch.nonzero(domains == index, as_tuple=True)[0] for index in range(len(conditions))]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Insufficient source samples for CORAL covariance estimation")
    validation_x, validation_y = (torch.as_tensor(pool["roles"]["validation"][key], device="cuda") for key in ("x", "y"))
    batch_size = CNN_RECIPE["batch_size"]
    per_epoch = math.ceil(len(signals) / batch_size)
    max_epochs = smoke_updates or CNN_RECIPE["max_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_RECIPE["optimizer"]["learning_rate"],
                                 weight_decay=CNN_RECIPE["optimizer"]["weight_decay"], fused=True)
    sampling = torch.Generator(device="cuda")
    sampling.manual_seed(seed_for(job["device"], job["channel"], job["target"], job["track"], job["split"], "cnn_sampling"))
    write_json(folder / "job.json", {**job, "identity": run_identity, "code": code, "recipe": CNN_RECIPE,
               "coral": CORAL_RECIPE, "source_identity": pool["source_identity"], "components": pool["components"],
               "source_conditions": conditions, "source_windows": len(signals), "additional_windows": 0,
               "initial_weights_sha256": initial_hash, "batch_size": batch_size, "updates_per_epoch": per_epoch,
               "max_epochs": max_epochs, "training_windows": len(signals)})
    stopping = EarlyStopping(**CNN_RECIPE["early_stopping"])
    best, completed, start_epoch = -1.0, 0, 0
    training_seconds, validation_seconds = 0.0, 0.0
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != run_identity:
            raise ValueError("CORAL resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        stopping.load_state_dict(checkpoint["early_stopping"])
        best, completed, start_epoch = checkpoint["best"], checkpoint["completed"], checkpoint["epoch"]
        training_seconds, validation_seconds = checkpoint["training_seconds"], checkpoint["validation_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        sampling.set_state(checkpoint["sampling_rng"])
    torch.cuda.reset_peak_memory_stats()
    stop_reason = "short_update_limit" if smoke_updates else "max_epochs"
    completed_epochs = start_epoch
    for epoch in range(start_epoch + 1, max_epochs + 1):
        if stopping.bad_epochs >= stopping.patience:
            stop_reason = "early_stopping"
            break
        model.train()
        epoch_started = time.perf_counter()
        for _ in range(1 if smoke_updates else per_epoch):
            optimizer.zero_grad(set_to_none=True)
            features, objectives = [], []
            for indices in source_batches(groups, batch_size, sampling):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden, logits = features_and_logits(model, signals[indices])
                    objectives.append(functional.binary_cross_entropy_with_logits(logits, labels[indices]))
                features.append(hidden.float())
            objective = torch.stack(objectives).mean()
            with torch.autocast("cuda", enabled=False):
                penalties = [mean_covariance_loss(features[first], features[second])
                             for first in range(len(features)) for second in range(first + 1, len(features))]
                alignment = torch.stack(penalties).mean()
            loss = objective + CORAL_RECIPE["alignment_weight"] * alignment
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite CORAL training objective")
            loss.backward()
            optimizer.step()
            completed += 1
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        logits = classifier.predict(model, validation_x)
        score = float(((logits >= 0) == (validation_y >= 0.5)).all(dim=1).float().mean().item())
        torch.cuda.synchronize()
        validation_seconds += time.perf_counter() - validation_started
        should_stop = stopping.update(score) if smoke_updates is None else False
        completed_epochs = epoch
        history = {"epoch": epoch, "step": completed, "source_accuracy": score, "classification_loss": float(objective.detach()),
                   "alignment_loss": float(alignment.detach()), "loss": float(loss.detach()), "time": timestamp()}
        with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history, allow_nan=False) + "\n")
        if score > best:
            best = score
            classifier.save_torch(folder / "best.pt", {"model": model.state_dict(), "step": completed, "epoch": epoch,
                                  "score": best, "identity": run_identity, "source_identity": pool["source_identity"]})
        classifier.save_torch(resume_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                              "completed": completed, "best": best, "identity": run_identity,
                              "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                              "early_stopping": stopping.state_dict(), "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state(), "sampling_rng": sampling.get_state()})
        write_json(folder / "progress.json", {**history, "identity": run_identity, "updated": timestamp()})
        if should_stop:
            stop_reason = "early_stopping"
            break
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    frozen_hash = file_hash(folder / "best.pt")
    evaluation_started = time.perf_counter()
    target_x, target_y, target_valid, target_states = classifier.target_pool(run, job, pool, frozen_hash, output_folder=folder)
    target_logits = classifier.predict(model, torch.as_tensor(target_x, device="cuda"))
    target = classifier.classification_metrics(target_logits, torch.as_tensor(target_y), target_valid, target_states)
    np.savez_compressed(folder / "target_predictions.npz", logits=target_logits.cpu().numpy(), labels=target_y, valid=target_valid)
    seen_x, seen_y, seen_valid, seen_groups = classifier.seen_pool(run, job, pool, frozen_hash, output_folder=folder)
    seen_logits = classifier.predict(model, torch.as_tensor(seen_x, device="cuda"))
    seen = classifier.classification_metrics(seen_logits, torch.as_tensor(seen_y), seen_valid, seen_groups)
    np.savez_compressed(folder / "seen_predictions.npz", logits=seen_logits.cpu().numpy(), labels=seen_y, valid=seen_valid)
    result = {"identity": run_identity, "job": job, "source_identity": pool["source_identity"], "recipe": CNN_RECIPE,
              "coral": CORAL_RECIPE, "initial_weights_sha256": initial_hash, "best_checkpoint_sha256": frozen_hash,
              "best_step": checkpoint["step"], "best_epoch": checkpoint["epoch"], "source_accuracy": checkpoint["score"],
              "target": target, "seen_conditions": seen, "training_seconds": training_seconds, "validation_seconds": validation_seconds,
              "evaluation_seconds": time.perf_counter() - evaluation_started, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "mode": "smoke_only" if smoke_updates else "formal", "updates_completed": completed, "epochs_completed": completed_epochs,
              "stop_reason": stop_reason, "early_stopping": stopping.state_dict(), "formal_updates": per_epoch * CNN_RECIPE["max_epochs"],
              "performance_claims_allowed": smoke_updates is None, "completed": timestamp()}
    write_json(folder / "result.json", result)
    log("coral_complete", id=job["id"], device=job["device"], channel=job["channel"], split=job["split"], updates=completed)
    return result


def run_classifier(run, job, *arguments, **options):
    if job["method"] == "coral":
        if arguments or options.get("generated") is not None:
            raise ValueError("CORAL does not consume generated training signals")
        options.pop("generated", None)
        return run_coral_classifier(run, job, **options)
    return classifier.run_classifier(run, job, *arguments, **options)


def main():
    from .comparison_design import execution_plan
    from .evaluation_cache import EvaluationCache
    from .queue import QueueLock
    from .report import generate_report

    parser = argparse.ArgumentParser(description="Add source-only CORAL on the existing T1/T2 comparison routes.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.35)
    arguments = parser.parse_args()
    source, output = arguments.source_run.resolve(), arguments.output.resolve()
    original = read_json(source / "coverage.json")
    data_run = Path(original["data_run"])
    if output in (source, data_run.resolve()) or not 0 < arguments.cuda_memory_fraction <= 1:
        raise ValueError("Use a separate output directory and a valid CUDA memory fraction")
    plan = execution_plan(data_run)
    route_fields = ("track", "device", "channel", "target", "split")
    routes = {tuple(job[field] for field in route_fields) for job in original["selected"]
              if job["track"] in ("T1", "T2") and job["method"] == "real_only"}
    selected = sorted([job for job in plan["jobs"] if job["method"] == "coral"
                       and tuple(job[field] for field in route_fields) in routes],
                      key=lambda job: (job["split"], job["device"], job["channel"], job["target"]))
    if len(selected) != len(routes) or not selected:
        raise ValueError("CORAL supplement must match every original T1/T2 Real-only route")
    coverage = {**original, "created": timestamp(), "selected": selected, "source_run": str(source),
                "source_coverage_sha256": file_hash(source / "coverage.json"),
                "sampling": "Source-only CORAL on the original T1/T2 Real-only routes; matched CNN update budgets.",
                "execution_design": plan["execution_design"], "design_sha256": plan["design_sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    with QueueLock(output):
        if (output / "coverage.json").exists():
            previous = read_json(output / "coverage.json")
            for key in ("selected", "source_coverage_sha256", "updates", "design_sha256"):
                if previous[key] != coverage[key]:
                    raise ValueError("CORAL coverage changed; use a new output directory")
            coverage["created"] = previous["created"]
        write_json(output / "coverage.json", coverage)
        runtime = classifier.configure_runtime(arguments.threads)
        torch.cuda.set_per_process_memory_fraction(arguments.cuda_memory_fraction)
        write_json(output / "runtime.json", {**runtime, "cuda_memory_fraction": arguments.cuda_memory_fraction,
                   "concurrent_training_possible": True, "created": timestamp()})
        cache = EvaluationCache()
        cache.install()
        for index, job in enumerate(selected):
            write_json(output / "status.json", {"state": "running", "position": index + 1, "total": len(selected),
                       "current": job, "updated": timestamp()})
            run_coral_classifier(data_run, job, output_root=output, smoke_updates=coverage["updates"])
            gc.collect()
            torch.cuda.empty_cache()
        report = generate_report(output)
        if report["missing_selected_ids"]:
            raise RuntimeError("Incomplete CORAL supplement")
        write_json(output / "evaluation_cache.json", cache.statistics())
        write_json(output / "status.json", {"state": "complete", "cnn_completed": len(selected),
                   "mode": coverage["mode"], "updated": timestamp()})
        log("coral_supplement_complete", cnns=len(selected), mode=coverage["mode"])


if __name__ == "__main__":
    main()
