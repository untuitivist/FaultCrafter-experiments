from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from faultcrafter.method_model import AdaptationParameters, MethodGenerator, initialize_new_tokens, load_method
from faultcrafter.physical_reference import MASKED_MECHANICAL_REFERENCE
from faultcrafter.training.objectives import batch_objective

from .common import ROOT, T1_DEVICES, canonical, file_hash, identity, log, read_json, timestamp, write_json
from .costs import model_cost
from .proposed_data import BalancedRounds, ProposedCorpus
from .tasks import T3_TARGETS


def training_recipe():
    released = read_json(ROOT / "faultcrafter/resources/device_lora_training_v1.json")
    generator = released["generator"]
    return {"maximum_updates": 5000, "training_length_range": [512, 16384],
            "maximum_batch_size": 32, "point_budget_per_batch": 65536,
            "validation_seed": 424242, "validation_length_count": 16,
            "validation_interval": 500, "checkpoint_preheat_updates": 2500,
            "early_stopping_evaluations": 4, "relative_improvement": 0.001,
            "save_interval": 50, "sampling_calls": 32, "terminal_time": 0.95,
            "generator": {**generator, "physics_reference_policy": MASKED_MECHANICAL_REFERENCE},
            "sampling_solver": "Euler integration with a final clean-signal prediction; exactly sampling_calls network forwards",
            "rounds": "One common length per complete round, each source cell visited once; stop before a round exceeding the update ceiling.",
            "new_device_labels": "Initialize new atomic-label embeddings from the source semantic-group mean and train only new rows alongside new-route LoRA."}


def save_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scientific_code():
    files = [*sorted((ROOT / "faultcrafter").glob("*.py")), *sorted((ROOT / "faultcrafter/training").glob("*.py")),
             Path(__file__), Path(__file__).with_name("proposed_data.py"), Path(__file__).with_name("data.py"),
             Path(__file__).with_name("generators.py")]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}


def validation(model, corpus, recipe, projectors, compute):
    random = np.random.default_rng(recipe["validation_seed"])
    lengths = [512, 16384, *random.integers(512, 16385, recipe["validation_length_count"] - 2).tolist()]
    if recipe.get("verification_lengths"):
        lengths = recipe["verification_lengths"]
    random = np.random.default_rng(recipe["validation_seed"])
    noise = torch.Generator(device=compute).manual_seed(recipe["validation_seed"])
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for length in lengths:
            count = min(recipe["maximum_batch_size"], recipe["point_budget_per_batch"] // length)
            keys = corpus.keys["validation"]
            for start in range(0, len(keys), count):
                batch = corpus.batch(model, "validation", keys[start:start + count], length, random)
                loss, _ = batch_objective(model.network, batch, recipe["generator"], compute_device=compute,
                                          random=noise, condition_projectors=projectors)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source validation objective")
                total += float(loss) * len(batch.fault)
                samples += len(batch.fault)
    return total / samples


def generation_cost(model, corpus, calls, compute, trainable_count):
    key = corpus.keys["train"][0]
    vector = corpus.schema.encode(key[0], key[1], key[2], corpus.schema.coordinates[key[0], key[3]])
    condition = torch.as_tensor(vector[None], device=compute)
    parameters, faults, groups, routes, speed, rates = model.encode(condition)
    inputs = (torch.zeros(1, 1, 1024, device=compute), torch.full((1,), 0.5, device=compute),
              parameters, faults, groups, routes, speed, rates, torch.full((1,), 1024, device=compute))
    cost = model_cost(model.network, inputs)
    cost["parameters_trainable"] = trainable_count
    sampler = model_cost(model, (condition,), operation=lambda batch: model.sample(batch, steps=calls), scope="complete_sampler")
    sampler["parameters_trainable"] = trainable_count
    count = [0]

    def observe(module, inputs, output):
        count[0] += 1

    with torch.random.fork_rng(devices=[compute.index or 0] if compute.type == "cuda" else []):
        handle = model.network.register_forward_hook(observe)
        try:
            model.sample(condition, steps=calls)
            if compute.type == "cuda":
                torch.cuda.synchronize(compute)
            count[0] = 0
            started = time.perf_counter()
            model.sample(condition, steps=calls)
            if compute.type == "cuda":
                torch.cuda.synchronize(compute)
            duration = time.perf_counter() - started
        finally:
            handle.remove()
    if count[0] != calls:
        raise RuntimeError("Actual sampler calls differ from the declared NFE")
    return cost, sampler, duration


def run_generator(run, job, output, *, device="cuda", recipe=None, verification=False, stop_after_updates=None, corpus_factory=ProposedCorpus):
    with method_numerics():
        return train_generator(run, job, output, device=device, recipe=recipe, verification=verification,
                               stop_after_updates=stop_after_updates, corpus_factory=corpus_factory)


@contextlib.contextmanager
def method_numerics():
    benchmark = torch.backends.cudnn.benchmark
    precision = torch.get_float32_matmul_precision()
    matmul_tf32, convolution_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = convolution_tf32


def train_generator(run, job, output, *, device="cuda", recipe=None, verification=False, stop_after_updates=None, corpus_factory=ProposedCorpus):
    recipe = copy.deepcopy(recipe or training_recipe())
    if not 0 < recipe["maximum_updates"] <= 5000:
        raise ValueError("The proposed method has a hard ceiling of 5000 optimizer updates")
    if not verification and recipe != training_recipe():
        raise ValueError("Formal training requires the declared recipe; use verification mode for reduced checks")
    if stop_after_updates is not None and not verification:
        raise ValueError("Artificial interruptions are only allowed in verification mode")
    if not verification and corpus_factory is not ProposedCorpus:
        raise ValueError("Formal training requires the complete registered source corpus")
    if (job["method"] != "faultcrafter" or job["track"] not in ("T1", "T2")
            or (job["track"] == "T1" and (job["strategy"] != "shared" or job.get("parent")))
            or (job["strategy"] == "scratch" and job.get("parent"))):
        raise ValueError("Invalid proposed-method initialization or adaptation strategy")
    folder = output / "generators" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    compute = torch.device(device)
    corpus = corpus_factory(run, job)
    torch.manual_seed(job["split"])
    if compute.type == "cuda":
        torch.cuda.manual_seed_all(job["split"])
        torch.cuda.reset_peak_memory_stats(compute)
    model = MethodGenerator(corpus.vocabulary)
    parent_path = output / "generators" / str(job.get("parent")) / "best.pt" if job.get("parent") else None
    parent_sha = file_hash(parent_path) if parent_path else None
    parent = None
    if parent_path:
        parent = torch.load(parent_path, weights_only=False, map_location="cpu")
        parent_job = parent["job"]
        if (job["track"] != "T2" or parent_job["track"] != "T1" or parent_job["device"] != "hdu"
                or parent_job["target"] != T3_TARGETS["hdu"] or parent_job["split"] != job["split"]
                or parent_job["method"] != job["method"] or parent_job.get("strategy") != "shared"
                or parent.get("verification") != verification or identity(parent["vocabulary"]) != identity(corpus.vocabulary)):
            raise ValueError("T2 requires the registered same-seed held-center benchmark parent")
        model = load_method(parent)
        initialize_new_tokens(model, T1_DEVICES, job["device"])
    if job["track"] == "T2" and job["strategy"] != "scratch" and parent_path is None:
        raise ValueError("Adaptation requires a freshly trained benchmark base")
    model.to(compute)
    ema = copy.deepcopy(model).eval()
    routes = [index for index, route in enumerate(corpus.vocabulary["channels"]) if route[0] == job["device"]]
    tokens = [index for index, token in enumerate(corpus.vocabulary["tokens"]) if token[0] == job["device"]]
    strategy = "full" if job["track"] == "T1" else job["strategy"]
    constrained = AdaptationParameters(model, strategy, routes, tokens)
    source_recipe = {"recipe": recipe, "job": job, "source": corpus.identity, "parent_sha256": parent_sha,
                     "code": scientific_code(), "verification": verification}
    run_id = identity(source_recipe)
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["identity"] != run_id or file_hash(folder / "best.pt") != result["best_checkpoint_sha256"]:
            raise ValueError("Completed method inputs changed; use a fresh output directory")
        return result
    resume_path = folder / "resume.pt"
    resume = torch.load(resume_path, weights_only=False, map_location=compute) if resume_path.exists() else None
    if resume is not None and resume["identity"] != run_id:
        raise ValueError("Resume source, code, parent or recipe differs")
    write_json(folder / "job.json", {**job, "identity": run_id, "source_identity": corpus.identity,
                                     "recipe": recipe, "verification": verification, "parent_sha256": parent_sha,
                                     "trainable_parameters": constrained.count, "code": source_recipe["code"],
                                     "native_length": "512–16384", "generator_batch_size": recipe["maximum_batch_size"],
                                     "optimizer": {"name": "AdamW", "learning_rate": recipe["generator"]["learning_rate"],
                                                   "weight_decay": recipe["generator"]["weight_decay"]},
                                     "sampling_steps": recipe["sampling_calls"], "updates_per_phase": recipe["maximum_updates"],
                                     "precision": "FP32; TF32 disabled", "implementation": "release32_backbone_fresh_training"})
    write_json(folder / "source_manifest.json", corpus.receipt)
    generator = recipe["generator"]
    optimizer = torch.optim.AdamW(constrained.parameters, lr=generator["learning_rate"], weight_decay=generator["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=generator["lr_decay_factor"],
                                                         patience=generator["lr_plateau_evaluations"], min_lr=generator["minimum_learning_rate"])
    projectors = corpus.projectors(compute)
    rounds = BalancedRounds(len(corpus.keys["train"]), job["split"], recipe)
    random = np.random.default_rng(job["split"] + 10000)
    noise = torch.Generator(device=compute).manual_seed(job["split"] + 100000)
    completed = points = 0
    best = math.inf
    anchor = None
    bad = 0
    next_validation = recipe["validation_interval"]
    last_validation = 0
    trace = ""
    elapsed = 0.0
    training_seconds = validation_seconds = 0.0
    initial = state_digest(model)
    if resume is not None:
        model.load_state_dict(resume["model"])
        ema.load_state_dict(resume["ema"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rounds.load_state_dict(resume["rounds"])
        random.bit_generator.state = resume["data_rng"]
        noise.set_state(resume["noise_rng"].cpu())
        torch.set_rng_state(resume["torch_rng"].cpu())
        if compute.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"].cpu(), compute)
        completed, points, best, anchor, bad, next_validation, trace, elapsed, initial = (
            resume[key] for key in ("updates", "points", "best", "anchor", "bad_evaluations", "next_validation", "batch_trace", "elapsed_seconds", "initial_sha256"))
        last_validation = resume["last_validation"]
        training_seconds, validation_seconds = resume["training_seconds"], resume["validation_seconds"]
        constrained.check(model)
        constrained.check(ema)
        event = {"event": "resume", "committed_update": completed, "committed_trace": trace, "created": timestamp()}
        for name in ("batches.jsonl", "history.jsonl"):
            with (folder / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
    started = time.perf_counter()
    log("proposed_training_start", id=job["id"], source_cells=len(corpus.keys["train"]),
        validation_cells=len(corpus.keys["validation"]), maximum_updates=recipe["maximum_updates"],
        resumed_updates=completed, verification=verification)

    def checkpoint(selected=False, score=None):
        common = {"format": "faultcrafter_benchmark32_v1", "open_set_evaluation_eligible": True,
                  "deployment_only": False, "verification": verification, "identity": run_id, "job": job,
                  "model": ema.state_dict() if selected else model.state_dict(), "backbone": model.backbone,
                  "vocabulary": corpus.vocabulary, "schema": corpus.schema.receipt(), "updates": completed,
                  "parent_sha256": parent_sha, "source_identity": corpus.identity, "score": score,
                  "recipe": recipe, "initial_sha256": initial}
        if selected:
            save_torch(folder / "best.pt", common)
            return
        save_torch(resume_path, {**common, "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(), "rounds": rounds.state_dict(),
                                "data_rng": random.bit_generator.state, "noise_rng": noise.get_state(),
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(compute) if compute.type == "cuda" else None,
                                "points": points, "best": best, "anchor": anchor, "bad_evaluations": bad,
                                "next_validation": next_validation, "batch_trace": trace,
                                "last_validation": last_validation,
                                "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                                "elapsed_seconds": elapsed + time.perf_counter() - started})

    stop_reason = "maximum_updates_at_complete_round"
    while completed < recipe["maximum_updates"] and bad < recipe["early_stopping_evaluations"]:
        if not rounds.begin(recipe["maximum_updates"] - completed):
            break
        indices = rounds.next()
        keys = [corpus.keys["train"][index] for index in indices]
        batch = corpus.batch(model, "train", keys, rounds.length, random)
        step_started = time.perf_counter()
        model.train()
        if completed < generator["warmup_steps"]:
            for group in optimizer.param_groups:
                group["lr"] = generator["learning_rate"] * (completed + 1) / max(1, generator["warmup_steps"])
        optimizer.zero_grad(set_to_none=True)
        loss, components = batch_objective(model.network, batch, generator, compute_device=compute, random=noise,
                                          condition_projectors=projectors, training_step=completed + 1)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite proposed-method training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(constrained.parameters, generator["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        constrained.restore_frozen()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters(), strict=True):
                target.lerp_(source, 1 - generator["ema_decay"])
            for target, source in zip(ema.buffers(), model.buffers(), strict=True):
                target.copy_(source)
        constrained.restore_frozen(ema)
        if compute.type == "cuda":
            torch.cuda.synchronize(compute)
        training_seconds += time.perf_counter() - step_started
        completed += 1
        points += int(np.prod(batch.normalized_signal.shape))
        trace = hashlib.sha256(trace.encode() + canonical(batch.parent_segment_id.tolist())).hexdigest()
        with (folder / "batches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"update": completed, "round": rounds.round, "length": rounds.length,
                                     "parents": batch.parent_segment_id.tolist(), "trace": trace}) + "\n")
        if completed % recipe["save_interval"] == 0:
            checkpoint()
            log("proposed_training", id=job["id"], update=completed, maximum_updates=recipe["maximum_updates"],
                length=rounds.length, loss=float(loss.detach()), round=rounds.round)
        evaluate = rounds.complete and (completed >= next_validation or completed == recipe["maximum_updates"])
        if evaluate:
            validation_started = time.perf_counter()
            score = validation(ema, corpus, recipe, projectors, compute)
            validation_seconds += time.perf_counter() - validation_started
            last_validation = completed
            if completed >= recipe["checkpoint_preheat_updates"]:
                if score < best:
                    best = score
                    checkpoint(selected=True, score=score)
                if anchor is None or score < anchor - max(abs(anchor), 1e-8) * recipe["relative_improvement"]:
                    anchor, bad = score, 0
                else:
                    bad += 1
            if completed >= generator["warmup_steps"]:
                scheduler.step(score)
            next_validation = completed + recipe["validation_interval"]
            row = {"update": completed, "source_validation_objective": score, "training_loss": float(loss.detach()),
                   "bad_evaluations": bad, "round": rounds.round, "points": points, "components": components}
            with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            log("proposed_validation", id=job["id"], **{key: value for key, value in row.items() if key != "components"})
            checkpoint()
        if stop_after_updates is not None and completed >= stop_after_updates:
            checkpoint()
            return {"interrupted_for_verification": True, "updates": completed}
    if completed == 0 or not rounds.complete:
        raise RuntimeError("The update ceiling cannot cover a complete balanced source round")
    if bad >= recipe["early_stopping_evaluations"]:
        stop_reason = "source_validation_stagnation"
    if last_validation != completed or not (folder / "best.pt").exists():
        validation_started = time.perf_counter()
        score = validation(ema, corpus, recipe, projectors, compute)
        validation_seconds += time.perf_counter() - validation_started
        last_validation = completed
        if score < best:
            best = score
            checkpoint(selected=True, score=score)
    checkpoint()
    constrained.check(model)
    constrained.check(ema)
    if state_digest(model) == initial:
        raise RuntimeError("No actual proposed-method parameter update occurred")
    selected = torch.load(folder / "best.pt", weights_only=False, map_location="cpu")
    reloaded = load_method(selected).to(compute).eval()
    constrained.check(reloaded)
    cost, sampler_cost, sampling_seconds = generation_cost(reloaded, corpus, recipe["sampling_calls"], compute, constrained.count)
    result = {"identity": run_id, "job": job, "source_identity": corpus.identity, "recipe": recipe, "schema": corpus.schema.receipt(),
              "mode": "verification_only" if verification else "formal", "verification": verification,
              "updates_completed": completed, "best_step": selected["updates"], "source_validation_loss": best,
              "parent_sha256": parent_sha, "best_checkpoint_sha256": file_hash(folder / "best.pt"),
              "sampling_steps": recipe["sampling_calls"], "network_calls": recipe["sampling_calls"],
              "training_points": points, "balanced_rounds": rounds.round, "batch_trace": trace,
              "stop_reason": stop_reason, "trainable_parameters": constrained.count,
              "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "frozen_parameters_verified": True, "initial_weights_sha256": initial,
              "cost": cost, "sampler_cost": sampler_cost, "sampling_nfe": recipe["sampling_calls"],
              "sampling_seconds_batch_one": sampling_seconds, "training_seconds": training_seconds,
              "validation_seconds": validation_seconds, "auxiliary_seconds": 0.0,
              "updates_per_phase": completed, "auxiliary_training": {},
              "wall_seconds": elapsed + time.perf_counter() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(compute) if compute.type == "cuda" else None,
              "completed": timestamp(), "performance_claims_allowed": not verification}
    write_json(result_path, result)
    corpus.values.cache_clear()
    return result
