from __future__ import annotations

import gc
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as functional

from .classifier import save_torch
from .common import canonical, file_hash, log, seed_for, timestamp, write_json
from .costs import model_cost
from .generative_models import CcGAN, ImprovedCCDM, build_model
from .recipes import EarlyStopping, epoch_learning_rate, generator_recipe, phase_budget
from .upstream import provenance


def make_optimizer(parameters, specification):
    options = {"lr": specification["learning_rate"], "weight_decay": specification.get("weight_decay", 0.0)}
    if specification["name"] == "SGD":
        options["momentum"] = specification.get("momentum", 0.9)
    else:
        options["betas"] = tuple(specification.get("betas", (0.9, 0.999)))
    return getattr(torch.optim, specification["name"])(parameters, **options)


def freeze_auxiliaries(model):
    for name in ("regressor", "embedding", "cov_regressor", "covariance"):
        module = getattr(model, name, None)
        if module is not None and isinstance(model, (CcGAN, ImprovedCCDM)):
            module.requires_grad_(False).eval()


def source_validation(model, dataset, recipe, job):
    signals, conditions = dataset.tensors["validation"]
    total = len(signals) if recipe["checkpoint_selection"] == "best_source_loss" else min(len(signals), recipe["batch_size"])
    total = min(total, recipe.get("validation_samples", total))
    accumulated = 0.0
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed_for(job["id"], "fixed_validation_noise"))
        for start in range(0, total, recipe["microbatch_size"]):
            real = signals[start:min(start + recipe["microbatch_size"], total)]
            condition = conditions[start:start + len(real)]
            if isinstance(model, CcGAN):
                generated = model.sample(condition)
                loss = functional.mse_loss(generated.mean(dim=-1), real.mean(dim=-1))
                loss += functional.mse_loss(generated.std(dim=-1), real.std(dim=-1))
            else:
                loss = model.loss(real, condition)
            accumulated += float(loss) * len(real)
    score = accumulated / total
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite source validation loss")
    return score


def train_batch(model, dataset, phase, recipe, indices, rng, optimizer, discriminator_optimizer):
    main = phase["name"] == "main"
    if main and isinstance(model, CcGAN):
        model.discriminator.requires_grad_(True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        real, condition, weights = dataset.vicinal(recipe["discriminator_batch_size"], rng, adaptive=False)
        embedding = model.embedding(condition)
        with torch.no_grad():
            fake = model(torch.randn(len(real), 128, device="cuda"), condition)
        discriminator_loss = (functional.relu(1 - model.discriminator(real, embedding)) * weights).mean()
        discriminator_loss += functional.relu(1 + model.discriminator(fake, embedding)).mean()
        if not torch.isfinite(discriminator_loss):
            raise RuntimeError("Nonfinite discriminator loss")
        discriminator_loss.backward()
        discriminator_optimizer.step()
        model.discriminator.requires_grad_(False)
        _, condition, _ = dataset.vicinal(recipe["batch_size"], rng, adaptive=False)
        loss = -model.discriminator(model(torch.randn(len(condition), 128, device="cuda"), condition),
                                    model.embedding(condition)).mean()
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite generator loss")
        loss.backward()
        optimizer.step()
        model.discriminator.requires_grad_(True)
        return float(loss.detach()), float(discriminator_loss.detach()), len(condition)
    mapper_phase = phase["name"].endswith("label_to_embedding")
    if mapper_phase:
        signals = None
        conditions = dataset.unique_conditions[indices]
        weights = None
    elif main and isinstance(model, ImprovedCCDM):
        signals, conditions, weights = dataset.vicinal(len(indices), rng, adaptive=True)
    else:
        signals, conditions = (values[indices] for values in dataset.tensors["train"])
        weights = None
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    microbatch = min(recipe["microbatch_size"], len(conditions))
    for start in range(0, len(conditions), microbatch):
        selected = slice(start, start + microbatch)
        condition = conditions[selected]
        if main:
            loss = model.loss(signals[selected], condition, None if weights is None else weights[selected])
        else:
            covariance = phase["name"].startswith("covariance")
            regressor = model.cov_regressor if covariance else model.regressor
            mapper = model.covariance if covariance else model.embedding
            if mapper_phase:
                perturbed = condition.clone()
                perturbed[:, :7] += 0.2 * torch.randn_like(perturbed[:, :7]) * perturbed[:, 7:14]
                loss = functional.mse_loss(regressor.head(mapper(perturbed)), perturbed)
            else:
                loss = functional.mse_loss(regressor(signals[selected]), condition)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {phase['name']}")
        fraction = len(condition) / len(conditions)
        (loss * fraction).backward()
        loss_sum += float(loss.detach()) * fraction
    if recipe.get("gradient_clip") and main:
        torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]],
                                      recipe["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return loss_sum, None, len(conditions)


def run_generator(run, job, output, updates=3, sampling_steps=None):
    from .generators import generator_data, generator_identity

    folder = output / "generators" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    recipe = generator_recipe(job["method"], smoke_updates=updates)
    sampling_steps = sampling_steps if sampling_steps is not None else recipe["sampling_steps"]
    if sampling_steps < 1 or (job["method"] != "ccgan" and sampling_steps < 2):
        raise ValueError("Use at least two sampler steps for diffusion and flow models")
    mode = "smoke_only" if updates is not None else "formal"
    dataset = generator_data(run, job["track"], job["device"], job["target"], job["split"])
    parent_path = output / "generators" / str(job["parent"]) / "best.pt" if job["parent"] else None
    parent_sha = file_hash(parent_path) if parent_path else None
    run_id, code = generator_identity(job, dataset, parent_sha, {"recipe": recipe, "sampling_steps": sampling_steps, "mode": mode})
    if (folder / "result.json").exists():
        from .common import read_json

        result = read_json(folder / "result.json")
        if result["identity"] != run_id:
            raise RuntimeError("Generator identity changed; choose a new output directory")
        return result
    random.seed(job["split"])
    np.random.seed(job["split"])
    torch.manual_seed(job["split"])
    torch.cuda.manual_seed_all(job["split"])
    rng = np.random.default_rng(seed_for(job["id"], "generator_batches"))
    model = build_model(job["method"], dataset.schema.dimension).cuda()
    if parent_path:
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if parent["schema"] != dataset.schema.receipt() or parent.get("mode") != mode:
            raise RuntimeError("Parent schema or formal/short training mode mismatch")
        model.load_state_dict(parent["model"])
        del parent
    budget = phase_budget(recipe, len(dataset.cpu["train"][0]))
    write_json(folder / "source_manifest.json", dataset.receipts)
    write_json(folder / "job.json", {**job, "identity": run_id, "source_identity": dataset.identity,
               "schema": dataset.schema.receipt(), "code": code, "upstream": provenance(job["method"]),
               "parent_sha256": parent_sha, "updates_per_phase": budget["max_updates"], "sampling_steps": sampling_steps,
               "native_length": 1024, "generator_batch_size": recipe["batch_size"], "microbatch_size": recipe["microbatch_size"],
               "optimizer": recipe["optimizer"], "recipe": recipe, "training_budget": budget, "mode": mode,
               "implementation": recipe["provenance"],
               "precision": "FP32 parameters and accumulated gradients; TF32 enabled"})
    dataset.activate()
    dataset.unique_conditions = torch.as_tensor(np.unique(dataset.unit_conditions["train"], axis=0), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    resume_path = folder / "resume.pt"
    resumed = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path.exists() else None
    receipts = {}
    if resumed:
        if resumed["identity"] != run_id:
            raise RuntimeError("Generator resume identity changed")
        model.load_state_dict(resumed["model"])
        receipts = resumed["receipts"]
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state(resumed["cuda_rng"])
        rng.bit_generator.state = resumed["batch_rng"]
        random.setstate(resumed["python_rng"])
        np.random.set_state(resumed["numpy_rng"])
    phases = [*recipe["auxiliaries"], {**recipe, "name": "main"}]
    for phase_index, phase in enumerate(phases):
        if resumed and phase_index < resumed["phase_index"]:
            continue
        main = phase["name"] == "main"
        model.requires_grad_(False).eval()
        if main:
            model.requires_grad_(True).train()
            freeze_auxiliaries(model)
            parameters = model.generator.parameters() if isinstance(model, CcGAN) else (parameter for parameter in model.parameters() if parameter.requires_grad)
        else:
            covariance = phase["name"].startswith("covariance")
            mapper_phase = phase["name"].endswith("label_to_embedding")
            module = (model.covariance if covariance else model.embedding) if mapper_phase else (model.cov_regressor if covariance else model.regressor)
            module.requires_grad_(True).train()
            parameters = module.parameters()
        optimizer = make_optimizer(parameters, phase["optimizer"])
        discriminator_optimizer = make_optimizer(model.discriminator.parameters(), phase["optimizer"]) if main and isinstance(model, CcGAN) else None
        count = len(dataset.unique_conditions) if phase["name"].endswith("label_to_embedding") else len(dataset.cpu["train"][0])
        phase_plan = phase_budget(phase, count)
        stopping = EarlyStopping(**recipe["early_stopping"]) if main and recipe.get("early_stopping") else None
        completed = epoch = 0
        best = None
        elapsed = validation_seconds = 0.0
        exposures = 0
        stop_reason = "max_epochs" if "max_epochs" in phase else "max_updates"
        if resumed and phase_index == resumed["phase_index"]:
            optimizer.load_state_dict(resumed["optimizer"])
            if discriminator_optimizer:
                discriminator_optimizer.load_state_dict(resumed["discriminator_optimizer"])
            completed, epoch, best = resumed["completed"], resumed["epoch"], resumed["best"]
            elapsed, validation_seconds = resumed["training_seconds"], resumed["validation_seconds"]
            exposures, stop_reason = resumed["samples_seen"], resumed["stop_reason"]
            if stopping:
                stopping.load_state_dict(resumed["early_stopping"])
        while completed < phase_plan["max_updates"] and stop_reason != "early_stopping":
            epoch += 1
            rate = epoch_learning_rate(phase, epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            if "max_epochs" in phase:
                order = rng.permutation(count)
                batches = (order[start:start + phase["batch_size"]] for start in range(0, count, phase["batch_size"]))
            else:
                batch_count = min(phase_plan["updates_per_epoch"], phase_plan["max_updates"] - completed)
                batches = (dataset.indices(phase["batch_size"], rng) if not phase["name"].endswith("label_to_embedding")
                           else rng.integers(count, size=phase["batch_size"]) for _ in range(batch_count))
            for positions in batches:
                indices = torch.as_tensor(positions, device="cuda")
                if main:
                    model.train()
                    freeze_auxiliaries(model)
                torch.cuda.synchronize()
                train_started = time.perf_counter()
                loss, discriminator_loss, exposure = train_batch(model, dataset, phase, recipe, indices, rng, optimizer, discriminator_optimizer)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - train_started
                completed += 1
                exposures += exposure
                epoch_end = completed % phase_plan["updates_per_epoch"] == 0
                due = (epoch_end if "max_epochs" in phase else completed % phase.get("validation_every_updates", 500) == 0)
                due = due or completed == phase_plan["max_updates"]
                if not due:
                    continue
                score = None
                if main:
                    validation_started = time.perf_counter()
                    score = source_validation(model, dataset, recipe, job)
                    torch.cuda.synchronize()
                    validation_seconds += time.perf_counter() - validation_started
                    if stopping and stopping.update(score):
                        stop_reason = "early_stopping"
                    if recipe["checkpoint_selection"] == "last" or best is None or score < best:
                        best = score
                        save_torch(folder / "best.pt", {"model": model.state_dict(), "schema": dataset.schema.receipt(),
                                   "identity": run_id, "source_identity": dataset.identity, "step": completed,
                                   "epoch": epoch if "max_epochs" in phase else None, "score": score, "mode": mode,
                                   "recipe": recipe})
                receipt = {"updates_completed": completed, "epochs_completed": epoch if "max_epochs" in phase else None,
                           "samples_seen": exposures, "training_seconds": elapsed, "validation_seconds": validation_seconds,
                           "stop_reason": stop_reason, "batch_size": phase["batch_size"], "budget": phase_plan}
                receipts[phase["name"]] = receipt
                write_json(folder / "progress.json", {"identity": run_id, "updated": timestamp(), "phase": phase["name"],
                           **receipt, "max_updates": phase_plan["max_updates"], "source_validation_loss": score})
                history = {"phase": phase["name"], "step": completed, "epoch": receipt["epochs_completed"], "loss": loss,
                           "discriminator_loss": discriminator_loss, "source_validation_loss": score,
                           "learning_rate": rate, "training_seconds": elapsed, "validation_seconds": validation_seconds,
                           "bad_epochs": stopping.bad_epochs if stopping else None, "samples_seen": exposures}
                with (folder / "history.jsonl").open("ab") as stream:
                    stream.write(canonical(history) + b"\n")
                save_torch(resume_path, {"identity": run_id, "model": model.state_dict(), "phase_index": phase_index,
                           "completed": completed, "epoch": epoch, "best": best, "receipts": receipts,
                           "optimizer": optimizer.state_dict(),
                           "discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer else None,
                           "training_seconds": elapsed, "validation_seconds": validation_seconds, "samples_seen": exposures,
                           "stop_reason": stop_reason, "early_stopping": stopping.state_dict() if stopping else None,
                           "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                           "batch_rng": rng.bit_generator.state, "python_rng": random.getstate(), "numpy_rng": np.random.get_state()})
                log("generator_progress", id=job["id"], method=job["method"], phase=phase["name"],
                    step=completed, max_updates=phase_plan["max_updates"], epoch=receipt["epochs_completed"],
                    loss=loss, source_validation_loss=score, stop_reason=stop_reason)
                if stop_reason == "early_stopping":
                    break
        if not main:
            module.requires_grad_(False).eval()
        resumed = None
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    condition = dataset.tensors["validation"][1][:1]
    inputs = (torch.zeros(1, 128, device="cuda"), condition) if isinstance(model, CcGAN) else (
        torch.zeros(1, 1, 1024, device="cuda"), torch.ones(1, device="cuda"), condition)
    forward_cost = model_cost(model, inputs)
    sample_cost = model_cost(model, (condition,), operation=lambda value: model.sample(value, steps=sampling_steps), scope="complete_sampler")
    with torch.inference_mode():
        model.sample(condition, steps=sampling_steps)
        torch.cuda.synchronize()
        sampling_started = time.perf_counter()
        probe = model.sample(condition, steps=sampling_steps)
        torch.cuda.synchronize()
        sample_seconds = time.perf_counter() - sampling_started
    if not torch.isfinite(probe).all():
        raise RuntimeError("Nonfinite generated sample after checkpoint reload")
    auxiliaries = {}
    for phase_name, receipt in receipts.items():
        if phase_name != "main":
            group, name = phase_name.split("_", 1)
            auxiliaries.setdefault(group, {})[name] = receipt
    main_receipt = receipts["main"]
    result = {"identity": run_id, "job": job, "schema": dataset.schema.receipt(), "source_identity": dataset.identity,
              "best_checkpoint_sha256": file_hash(folder / "best.pt"), "best_step": checkpoint["step"],
              "parent_sha256": parent_sha, "source_validation_loss": checkpoint["score"],
              "updates_per_phase": main_receipt["updates_completed"], "updates_completed": main_receipt["updates_completed"],
              "epochs_completed": main_receipt["epochs_completed"], "stop_reason": main_receipt["stop_reason"],
              "recipe": recipe, "training_budget": budget, "phase_records": receipts,
              "auxiliary_training": auxiliaries, "training_seconds": main_receipt["training_seconds"],
              "auxiliary_seconds": sum(value["training_seconds"] for name, value in receipts.items() if name != "main"),
              "validation_seconds": main_receipt["validation_seconds"], "wall_seconds": time.perf_counter() - started,
              "sampling_seconds_batch_one": sample_seconds, "sampling_steps": sampling_steps,
              "sampling_nfe": (2 * (2 * sampling_steps - 1) if job["method"] == "iccdm" else 1 if job["method"] == "ccgan" else sampling_steps),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "cost": forward_cost, "sampler_cost": sample_cost,
              "completed": timestamp(), "mode": mode, "performance_claims_allowed": mode == "formal"}
    write_json(folder / "auxiliary_training.json", auxiliaries)
    write_json(folder / "cost.json", {"forward": forward_cost, "sampler": sample_cost})
    write_json(folder / "result.json", result)
    dataset.release()
    del dataset.unique_conditions
    del model, optimizer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result
