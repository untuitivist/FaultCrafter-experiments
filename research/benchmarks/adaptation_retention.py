from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import importlib
import inspect
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from faultcrafter.models import CNN1D

from .classifier import classification_metrics, save_torch, state_hash
from .common import SPLITS, T1_DEVICES, file_hash, identity, read_json, seed_for, timestamp, write_json
from .data import CHANNELS
from .generative_models import build_model
from .generators import ConditionSchema
from .pools import source_pool
from .queue import QueueLock
from .recipes import CNN_RECIPE, EarlyStopping


STRATEGIES = ("lora", "full_finetune")
METRICS = ("exact_set_accuracy", "macro_f1_valid")


def load_cases(path):
    cases = read_json(path)
    if not cases:
        raise ValueError("Provide completed T2 adaptation cases")
    identifiers = set()
    bases = {}
    checked = {}
    for case in cases:
        if case["strategy"] not in STRATEGIES or case["split"] not in SPLITS:
            raise ValueError("Retention compares LoRA and full fine-tuning at registered seeds")
        for key in ("base_generator", "adapted_generator", "original_generator"):
            folder = Path(case[key])
            case[key] = str((folder if folder.is_absolute() else path.parent / folder).resolve())
        base = read_json(Path(case["base_generator"]) / "result.json")
        adapted = read_json(Path(case["adapted_generator"]) / "result.json")
        if (base["job"]["track"] != "T1" or adapted["job"]["track"] != "T2"
                or base["job"]["split"] != case["split"] or adapted["job"]["split"] != case["split"]
                or adapted["job"]["parent"] != base["job"]["id"]
                or adapted["parent_sha256"] != base["best_checkpoint_sha256"]):
            raise ValueError("Adaptation must inherit the declared T1 checkpoint at the same seed")
        if base["job"]["method"] != adapted["job"]["method"]:
            raise ValueError("Base and adapted models must belong to the same method")
        if case["strategy"] == "full_finetune" and case["original_generator"] != case["adapted_generator"]:
            raise ValueError("Full fine-tuning must evaluate its updated model on original devices")
        for key in ("base_generator", "adapted_generator", "original_generator"):
            result = read_json(Path(case[key]) / "result.json")
            if case[key] not in checked:
                checked[case[key]] = file_hash(Path(case[key]) / "best.pt")
            if checked[case[key]] != result["best_checkpoint_sha256"]:
                raise ValueError("Generator checkpoint checksum mismatch")
        previous = bases.setdefault(case["split"], base["best_checkpoint_sha256"])
        if previous != base["best_checkpoint_sha256"]:
            raise ValueError("Use the same shared T1 base for every adaptation at a given seed")
        case["new_device"] = adapted["job"]["device"]
        case["new_target"] = adapted["job"]["target"]
        case["new_channel"] = adapted["job"].get("channel")
        case["method"] = base["job"]["method"]
        case["base_sha256"] = base["best_checkpoint_sha256"]
        case["adapted_sha256"] = adapted["best_checkpoint_sha256"]
        case["original_sha256"] = read_json(Path(case["original_generator"]) / "result.json")["best_checkpoint_sha256"]
        case["original_result_sha256"] = file_hash(Path(case["original_generator"]) / "result.json")
        if case["original_sha256"] not in (case["base_sha256"], case["adapted_sha256"]):
            raise ValueError("Declare either the frozen original route or the adapted model for original devices")
        key = (case["strategy"], case["split"], case["new_device"], case["new_channel"], case["new_target"])
        if key in identifiers:
            raise ValueError("Duplicate adaptation case")
        identifiers.add(key)
        case["id"] = identity(case)[:24]
    if len({case["method"] for case in cases}) != 1:
        raise ValueError("Compare adaptation strategies of one generator architecture at a time")
    if set(case["strategy"] for case in cases) == set(STRATEGIES):
        templates = [{(case["split"], case["new_device"], case["new_channel"], case["new_target"])
                      for case in cases if case["strategy"] == strategy} for strategy in STRATEGIES]
        if templates[0] != templates[1]:
            raise ValueError("LoRA and full fine-tuning must cover the same new-device adaptation tasks")
    return cases


def original_job(case, device, channel):
    base = read_json(Path(case["base_generator"]) / "result.json")["job"]
    target = base["target"] if device == base["device"] else "__all_original_source_conditions__"
    return {"device": device, "channel": channel, "target": target, "split": case["split"]}


def synthetic_signals(data_run, job, folder, *, steps, device, factory=None):
    pool = source_pool(data_run, job)
    schema = ConditionSchema(read_json(data_run / "inventory.json")["records"])
    result = read_json(folder / "result.json")
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    if identity(checkpoint["schema"]) != identity(schema.receipt()):
        raise ValueError("Condition schema differs from the prepared data")
    if factory:
        model = factory(checkpoint, result, schema.dimension)
    else:
        model = build_model(result["job"]["method"], schema.dimension)
        model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    conditions = pool["roles"]["train"]["conditions"]
    generated, labels = [], []
    seed = seed_for(job, "adaptation_retention_generation")
    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(seed)
        for state in pool["states"]:
            vectors = np.stack([schema.encode(job["device"], job["channel"], state,
                                schema.coordinates[job["device"], conditions[index % len(conditions)]])
                                for index in range(63)])
            for start in range(0, len(vectors), 8):
                batch = torch.as_tensor(vectors[start:start + 8], device=device)
                signals = model.sample(batch, steps=steps, length=1024).float().cpu().numpy()
                if signals.shape != (len(batch), 1, 1024) or not np.isfinite(signals).all():
                    raise RuntimeError("Invalid original-device generation")
                generated.append(signals)
            vector = [part in pool["labels"][state] for part in pool["components"]]
            labels.extend([vector] * 63)
    return np.concatenate(generated), np.asarray(labels, dtype=np.float32)


def original_test(data_run, job, pool):
    manifest = read_json(data_run / "cache/windows" / str(job["split"]) / "manifest.json")
    quota = read_json(data_run / "data_ready.json")["quotas"]["seen"]
    channel = CHANNELS[job["device"]].index(job["channel"])
    signals, labels, validity, receipts = [], [], [], []
    source_ids = {identity(sample) for role in pool["roles"].values() for sample in role["samples"]}
    for cell in sorted(manifest["cells"], key=lambda item: (item["device"], item["condition"], item["state"], item["role"])):
        if (cell["role"] != "seen" or cell["device"] != job["device"]
                or cell["condition"] == job["target"] or cell["state"] not in pool["states"]):
            continue
        samples = cell["samples"][:quota]
        if any(identity(sample) in source_ids for sample in samples):
            raise ValueError("Original-device evaluation overlaps classifier training or validation")
        directory = data_run / "cache/windows" / str(job["split"])
        values = np.load(directory / cell["file"], mmap_mode="r")[:quota, channel:channel + 1]
        valid = np.load(directory / cell["file"].replace(".npy", "_valid.npy"), mmap_mode="r")[:quota, channel]
        if len(values) != quota or len(samples) != quota:
            raise ValueError("Original-device test quota mismatch")
        signals.append(np.array(values))
        labels.extend([[part in pool["labels"][cell["state"]] for part in pool["components"]]] * quota)
        validity.extend(valid.tolist())
        receipts.append({"condition": cell["condition"], "state": cell["state"], "samples": samples})
    if not signals:
        raise ValueError("No independent original-device test data")
    return np.concatenate(signals), np.asarray(labels, dtype=np.float32), np.asarray(validity), receipts


def diagnose(data_run, job, generated, folder, *, device):
    pool = source_pool(data_run, job)
    run_id = identity({"job": job, "source": pool["source_identity"], "recipe": CNN_RECIPE,
                       "signals": [hashlib.sha256(value.tobytes()).hexdigest() for value in generated],
                       "code": file_hash(Path(__file__)), "cnn": file_hash(Path(inspect.getfile(CNN1D))),
                       "classifier": file_hash(Path(inspect.getfile(classification_metrics))), "device": device})
    if (folder / "result.json").exists():
        result = read_json(folder / "result.json")
        if result["identity"] != run_id:
            raise ValueError("Retention inputs changed; use a fresh output directory")
        return result
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(job["split"])
    model = CNN1D(len(pool["components"])).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_RECIPE["optimizer"]["learning_rate"],
                                 weight_decay=CNN_RECIPE["optimizer"]["weight_decay"], fused=device == "cuda")
    train = pool["roles"]["train"]
    train_x = torch.as_tensor(np.concatenate((train["x"], generated[0])), device=device)
    train_y = torch.as_tensor(np.concatenate((train["y"], generated[1])), device=device)
    validation = pool["roles"]["validation"]
    validation_x = torch.as_tensor(validation["x"], device=device)
    validation_y = torch.as_tensor(validation["y"], device=device)
    sampling = torch.Generator(device=device).manual_seed(seed_for(job, "retention_cnn_sampling"))
    stopping = EarlyStopping(**CNN_RECIPE["early_stopping"])
    best = -1.0
    completed = 0
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume["identity"] != run_id:
            raise ValueError("Retention resume inputs changed; use a fresh output directory")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        sampling.set_state(resume["sampling_rng"].cpu())
        torch.set_rng_state(resume["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"].cpu())
        stopping.load_state_dict(resume["early_stopping"])
        completed, best = resume["epoch"], resume["best"]

    def prediction(signals):
        model.eval()
        with torch.inference_mode():
            return torch.cat([model(signals[start:start + 256]).float().cpu()
                              for start in range(0, len(signals), 256)])

    epoch = completed
    for epoch in range(completed + 1, CNN_RECIPE["max_epochs"] + 1):
        if stopping.bad_epochs >= stopping.patience:
            break
        model.train()
        order = torch.randperm(len(train_x), generator=sampling, device=device)
        for indices in order.split(CNN_RECIPE["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext():
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite retention classifier loss")
            loss.backward()
            optimizer.step()
        score = classification_metrics(prediction(validation_x), validation_y.cpu())["exact_set_accuracy"]
        if score > best:
            best = score
            save_torch(folder / "best.pt", {"model": model.state_dict(), "identity": run_id, "epoch": epoch})
        stop = stopping.update(score)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, "source_accuracy": score, "loss": float(loss.detach())}) + "\n")
        save_torch(resume_path, {"identity": run_id, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "sampling_rng": sampling.get_state(), "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                                "early_stopping": stopping.state_dict(), "epoch": epoch, "best": best})
        if stop:
            break
    model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=False)["model"])
    test_x, test_y, valid, receipts = original_test(data_run, job, pool)
    logits = prediction(torch.as_tensor(test_x, device=device))
    np.savez_compressed(folder / "predictions.npz", logits=logits.numpy(), labels=test_y, valid=valid)
    write_json(folder / "test_manifest.json", {"role": "seen", "cells": receipts})
    result = {"identity": run_id, "job": job, "source_identity": pool["source_identity"],
              "initial_weights_sha256": initial_hash, "test_identity": identity(receipts),
              "metrics": classification_metrics(logits, torch.as_tensor(test_y), valid),
              "best_checkpoint_sha256": file_hash(folder / "best.pt"), "epochs_completed": epoch,
              "recipe": CNN_RECIPE, "completed": timestamp()}
    write_json(folder / "result.json", result)
    return result


def summarize(records, cases):
    output = {}
    expected_units = {(device, channel) for device in T1_DEVICES for channel in CHANNELS[device]}
    for strategy in STRATEGIES:
        selected = [record for record in records if record["strategy"] == strategy]
        seed_values = {scope: {metric: [] for metric in METRICS} for scope in ("all", *T1_DEVICES)}
        complete = True
        templates = {(case["new_device"], case["new_channel"], case["new_target"])
                     for case in cases if case["strategy"] == strategy}
        for split in SPLITS:
            current_templates = {(case["new_device"], case["new_channel"], case["new_target"])
                                 for case in cases if case["strategy"] == strategy and case["split"] == split}
            if current_templates != templates or not templates:
                complete = False
                break
            expected_cases = {case["id"] for case in cases if case["split"] == split and case["strategy"] == strategy}
            current = [record for record in selected if record["job"]["split"] == split]
            if ({(record["case_id"], record["job"]["device"], record["job"]["channel"]) for record in current}
                    != {(case_id, *unit) for case_id in expected_cases for unit in expected_units} or not current):
                complete = False
                break
            for scope, metrics in seed_values.items():
                for metric in METRICS:
                    by_case = defaultdict(list)
                    for original_device in T1_DEVICES if scope == "all" else (scope,):
                        for case_id in expected_cases:
                            values = [record["metrics"][metric] for record in current
                                      if record["case_id"] == case_id and record["job"]["device"] == original_device]
                            by_case[case_id].append(mean(values))
                    by_new_device = defaultdict(list)
                    for case_id, values in by_case.items():
                        new_device = next(case["new_device"] for case in cases if case["id"] == case_id)
                        by_new_device[new_device].append(mean(values))
                    metrics[metric].append(mean(mean(values) for values in by_new_device.values()))
        measurements = {scope: {metric: [mean(values), stdev(values)] if complete else None
                                for metric, values in metrics.items()} for scope, metrics in seed_values.items()}
        output[strategy] = {**measurements["all"], "by_device": {device: measurements[device] for device in T1_DEVICES}}
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate original-device R+G diagnosis after T2 adaptation; T1 remains the reference.")
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-factory", help="module:function returning a loaded generator from (checkpoint, result, condition_dim)")
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.data_run.resolve() or arguments.steps < 2:
        raise ValueError("Use a separate output directory and at least two sampling steps")
    cases = load_cases(arguments.cases.resolve())
    factory = None
    factory_sha = None
    if arguments.model_factory:
        module, name = arguments.model_factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        factory_sha = file_hash(Path(inspect.getfile(factory)))
    plan = {"cases": cases, "steps": arguments.steps, "device": arguments.device,
            "factory": arguments.model_factory, "factory_sha256": factory_sha,
            "data_sha256": file_hash(arguments.data_run / "data_ready.json"),
            "inventory_sha256": file_hash(arguments.data_run / "inventory.json"),
            "window_manifests": {str(split): file_hash(arguments.data_run / "cache/windows" / str(split) / "manifest.json")
                                 for split in sorted({case["split"] for case in cases})},
            "code_sha256": file_hash(Path(__file__)), "recipe": CNN_RECIPE,
            "original_devices": T1_DEVICES, "original_channels": sum(len(CHANNELS[device]) for device in T1_DEVICES)}
    arguments.output.mkdir(parents=True, exist_ok=True)
    with QueueLock(arguments.output):
        if (arguments.output / "plan.json").exists() and identity(read_json(arguments.output / "plan.json")) != identity(plan):
            raise ValueError("Retention plan changed; use a fresh output directory")
        write_json(arguments.output / "plan.json", plan)
        if arguments.plan_only:
            print({"adaptations": len(cases), "original_channels": plan["original_channels"], "training_started": False})
            return
        torch.set_num_threads(4)
        cache, paired, records = {}, {}, []
        for position, case in enumerate(cases, 1):
            write_json(arguments.output / "status.json", {"state": "running", "case": case["id"],
                       "position": position, "total": len(cases), "updated": timestamp()})
            for device in T1_DEVICES:
                for channel in CHANNELS[device]:
                    job = original_job(case, device, channel)
                    key = identity({"original_checkpoint": case["original_sha256"],
                                    "original_result": case["original_result_sha256"], "job": job})[:24]
                    reused = key in cache
                    if not reused:
                        generated = synthetic_signals(arguments.data_run, job, Path(case["original_generator"]), steps=arguments.steps,
                                                      device=arguments.device, factory=factory)
                        folder = arguments.output / "jobs" / key
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(folder / "generated.npz", x=generated[0], y=generated[1])
                        cache[key] = diagnose(arguments.data_run, job, generated, folder, device=arguments.device)
                    result = cache[key]
                    pairing = tuple(result[name] for name in ("source_identity", "test_identity", "initial_weights_sha256"))
                    if paired.setdefault(identity(job), pairing) != pairing:
                        raise RuntimeError("Adaptation strategies must use paired original-device diagnosis")
                    record = {**result, "strategy": case["strategy"], "case_id": case["id"], "reused_original_route": reused,
                              "adapted_sha256": case["adapted_sha256"], "original_sha256": case["original_sha256"]}
                    write_json(arguments.output / "comparisons" / case["id"] / (key + ".json"), record)
                    records.append(record)
                    print(f"Retention: {case['strategy']} seed={case['split']} {device}/{channel}; reused={reused}", flush=True)
            if (arguments.output / "STOP_AFTER_CURRENT").exists():
                break
        summary = {"schema": "t2-original-device-retention-v1", "plan_sha256": identity(plan),
                   "method": cases[0]["method"],
                   "summary": summarize(records, cases), "records": records,
                   "complete": len(records) == plan["original_channels"] * len(cases),
                   "aggregation": "Equal original channels/devices and new-device cases within seed; mean and sample SD over seeds 42-46.",
                   "created": timestamp()}
        write_json(arguments.output / "retention_summary.json", summary)
        write_json(arguments.output / "status.json", {"state": "complete" if summary["complete"] else "incomplete",
                   "records": len(records), "updated": timestamp()})


if __name__ == "__main__":
    main()
