from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from faultcrafter.models import CNN1D

from .common import canonical, file_hash, identity, log, read_json, seed_for, timestamp, write_json
from .common import METHODS
from .costs import model_cost
from .pools import augmentation_pool, seen_pool, source_pool, superposition_pool, target_pool, update_budget
from .recipes import CNN_RECIPE, EarlyStopping


def configure_runtime(threads=4):
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison queue requires CUDA")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "precision": "bfloat16 autocast; FP32 parameters and Adam", "cpu_threads": threads,
            "tf32": True, "cudnn_benchmark": True}


def save_torch(path, payload):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MixStyleCNN(CNN1D):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.domains = None

    def mix(self, hidden):
        if not self.training or self.domains is None or self.domains.unique().numel() < 2 or random.random() > 0.5:
            return hidden
        mean = hidden.mean(dim=-1, keepdim=True).detach()
        deviation = (hidden.var(dim=-1, keepdim=True) + 1e-6).sqrt().detach()
        probabilities = (self.domains[:, None] != self.domains[None, :]).float()
        partners = torch.multinomial(probabilities, 1).flatten()
        mixing = torch.distributions.Beta(0.1, 0.1).sample((len(hidden), 1, 1)).to(hidden.device)
        return (hidden - mean) / deviation * (mixing * deviation + (1 - mixing) * deviation[partners]) + mixing * mean + (1 - mixing) * mean[partners]

    def forward(self, signal):
        hidden = self.mix(self.layer1(signal))
        hidden = self.mix(self.layer2(hidden))
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)
        hidden = self.layer5(hidden)
        hidden = self.layer6(hidden)
        return self.classifier(hidden.flatten(start_dim=1))


@torch.inference_mode()
def predict(model, signals):
    model.eval()
    output = []
    for start in range(0, len(signals), 256):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output.append(model(signals[start:start + 256]).float())
    return torch.cat(output)


def classification_metrics(logits, labels, valid=None, groups=None):
    from sklearn.metrics import roc_auc_score

    probabilities = torch.sigmoid(logits).cpu().numpy()
    truth = labels.cpu().numpy().astype(bool)
    decisions = probabilities >= 0.5
    if valid is None:
        valid = np.ones(len(truth), dtype=bool)
    exact = np.all(decisions == truth, axis=1) & valid
    true_positive = ((decisions & truth) & valid[:, None]).sum(axis=0)
    false_positive = ((decisions & ~truth) & valid[:, None]).sum(axis=0)
    false_negative = ((~decisions & truth) & valid[:, None]).sum(axis=0)
    true_negative = ((~decisions & ~truth) & valid[:, None]).sum(axis=0)
    precision = true_positive / np.maximum(1, true_positive + false_positive)
    recall = true_positive / np.maximum(1, true_positive + false_negative)
    f1 = 2 * true_positive / np.maximum(1, 2 * true_positive + false_positive + false_negative)
    auc = [float(roc_auc_score(truth[valid, index], probabilities[valid, index]))
           if len(np.unique(truth[valid, index])) == 2 else None for index in range(truth.shape[1])]
    result = {"correct": int(exact.sum()), "total": len(truth), "exact_set_accuracy": float(exact.mean()),
              "unpredicted": int((~valid).sum()), "macro_f1_valid": float(f1.mean()),
              "precision_valid": precision.tolist(), "recall_valid": recall.tolist(), "f1_valid": f1.tolist(),
              "auc_valid": auc, "confusion_valid": np.stack([true_negative, false_positive, false_negative, true_positive], axis=1).tolist()}
    if groups is not None:
        result["groups"] = {str(group): {"correct": int(exact[np.asarray(groups) == group].sum()),
                                        "total": int((np.asarray(groups) == group).sum())} for group in sorted(set(groups))}
    return result


def training_identity(job, pool):
    folder = Path(__file__).parent
    sources = {name: file_hash(folder / name) for name in ("classifier.py", "common.py", "costs.py", "data.py", "pools.py", "tasks.py", "recipes.py")}
    sources["CNN1D"] = file_hash(folder.parents[1] / "faultcrafter/models.py")
    return identity({"job": job, "source": pool["source_identity"], "code": sources,
                     "recipe": CNN_RECIPE, "numerics": "bf16-tf32-fused-adam-v1"}), sources


def run_classifier(run, job, timing_only=False, *, output_root=None, smoke_updates=None, generated=None):
    if smoke_updates is not None and (smoke_updates < 2 or timing_only or output_root is None or output_root.resolve() == run.resolve()):
        raise ValueError("Smoke runs require at least two updates and a separate output root")
    folder = (output_root or run) / ("timing" if timing_only else "jobs") / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pool = source_pool(run, job)
    run_identity, code = training_identity(job, pool)
    if job["method"] in METHODS or generated is not None:
        if generated is None:
            raise RuntimeError("Generator comparison requires its registered synthetic pool")
        run_identity = identity({"classifier": run_identity, "generation": generated[2]})
    if smoke_updates is not None:
        run_identity = identity({"formal_identity": run_identity, "mode": "smoke_only", "updates": smoke_updates})
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["identity"] != run_identity:
            raise RuntimeError("Completed job has different code or source identity; use a new run directory")
        return result
    per_epoch, total_updates, added = update_budget(pool, job)
    formal_updates = total_updates
    if smoke_updates is not None:
        per_epoch, total_updates = 1, smoke_updates
    random.seed(job["split"])
    np.random.seed(job["split"])
    torch.manual_seed(job["split"])
    torch.cuda.manual_seed_all(job["split"])
    model = (MixStyleCNN if job["method"] == "mixstyle" else CNN1D)(len(pool["components"])).cuda()
    initial_hash = state_hash(model)
    cost_started = time.perf_counter()
    cost = model_cost(model, (torch.zeros(1, 1, pool["roles"]["train"]["x"].shape[-1], device="cuda"),))
    torch.cuda.synchronize()
    cost_seconds = time.perf_counter() - cost_started
    write_json(folder / "cost.json", cost)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_RECIPE["optimizer"]["learning_rate"],
                                 weight_decay=CNN_RECIPE["optimizer"]["weight_decay"], fused=True)
    criterion = nn.BCEWithLogitsLoss()
    train = pool["roles"]["train"]
    real_x = torch.as_tensor(train["x"], device="cuda")
    real_y = torch.as_tensor(train["y"], device="cuda")
    domains = torch.as_tensor(train["unit"] % len(train["conditions"]), device="cuda")
    validation_x = torch.as_tensor(pool["roles"]["validation"]["x"], device="cuda")
    validation_y = torch.as_tensor(pool["roles"]["validation"]["y"], device="cuda")
    new_x = new_y = None
    if job["method"] == "augmentation":
        synthetic_x, synthetic_y = augmentation_pool(pool, job, added)
        new_x, new_y = torch.as_tensor(synthetic_x, device="cuda"), torch.as_tensor(synthetic_y, device="cuda")
    elif job["method"] == "superposition":
        synthetic_x, synthetic_y, receipt = superposition_pool(pool, job)
        new_x, new_y = torch.as_tensor(synthetic_x, device="cuda"), torch.as_tensor(synthetic_y, device="cuda")
        write_json(folder / "superposition.json", receipt)
    elif job["method"] in METHODS or generated is not None:
        synthetic_x, synthetic_y, receipt = generated
        new_x, new_y = torch.as_tensor(synthetic_x, device="cuda"), torch.as_tensor(synthetic_y, device="cuda")
        if len(new_x) != added:
            raise RuntimeError(f"Generated count differs from protocol: {len(new_x)} != {added}")
        write_json(folder / "generation.json", receipt)
    sampling = torch.Generator(device="cuda")
    sampling.manual_seed(seed_for(job["device"], job["channel"], job["target"], job["track"], job["split"], "cnn_sampling"))
    if new_x is None:
        training_x, training_y = real_x, real_y
    elif job.get("arm") == "generated_only":
        training_x, training_y = new_x, new_y
    else:
        training_x, training_y = torch.cat((real_x, new_x)), torch.cat((real_y, new_y))
    batch_size = CNN_RECIPE["batch_size"]
    max_epochs = smoke_updates or CNN_RECIPE["max_epochs"]
    write_json(folder / "job.json", {**job, "identity": run_identity, "code": code, "initial_weights_sha256": initial_hash,
                                     "source_identity": pool["source_identity"], "components": pool["components"],
                                     "source_windows": len(real_x), "additional_windows": 0 if new_x is None else len(new_x),
                                     "updates_per_epoch": per_epoch, "updates": total_updates,
                                     "formal_updates": formal_updates, "mode": "smoke_only" if smoke_updates else "formal",
                                     "optimizer": {**CNN_RECIPE["optimizer"], "fused": True},
                                     "recipe": CNN_RECIPE, "training_windows": len(training_x),
                                     "max_epochs": max_epochs, "early_stopping": CNN_RECIPE["early_stopping"],
                                     "batch_size": batch_size, "loss": "BCEWithLogitsLoss", "cost": cost})
    best = -1.0
    completed = 0
    completed_epochs = 0
    stopping = EarlyStopping(**CNN_RECIPE["early_stopping"])
    stop_reason = "short_update_limit" if smoke_updates else "max_epochs"
    cumulative_training = 0.0
    cumulative_validation = 0.0
    last_path = folder / "resume.pt"
    if last_path.exists() and not timing_only:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != run_identity:
            raise RuntimeError("Resume code/source identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed, best = checkpoint["completed"], checkpoint["best"]
        completed_epochs = checkpoint["epoch"]
        stopping.load_state_dict(checkpoint["early_stopping"])
        stop_reason = checkpoint["stop_reason"]
        cumulative_training = checkpoint["training_seconds"]
        cumulative_validation = checkpoint["validation_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        sampling.set_state(checkpoint["sampling_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    log("cnn_start", id=job["id"], track=job["track"], device=job["device"], channel=job["channel"],
        method=job["method"], updates=total_updates, resumed_at=completed, samples=len(real_x))
    torch.cuda.reset_peak_memory_stats()
    warmup = min(32, per_epoch)
    measurement = 128
    throughput_start = None
    for epoch in range(completed_epochs + 1, max_epochs + 1):
        if stop_reason == "early_stopping":
            break
        model.train()
        epoch_started = time.perf_counter()
        order = torch.randperm(len(training_x), device="cuda", generator=sampling)
        batches = (order[start:start + batch_size] for start in range(0, len(order), batch_size))
        if smoke_updates or timing_only:
            count = warmup + measurement if timing_only else 1
            batches = (torch.randint(len(training_x), (batch_size,), device="cuda", generator=sampling) for _ in range(count))
        epoch_loss = 0.0
        samples_seen = 0
        for indices in batches:
            if job["method"] == "mixstyle":
                model.domains = domains[indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(training_x[indices]), training_y[indices])
            if not torch.isfinite(loss.detach()).item():
                raise RuntimeError(f"Nonfinite training loss at update {completed + 1}")
            loss.backward()
            optimizer.step()
            completed += 1
            epoch_loss += float(loss.detach()) * len(indices)
            samples_seen += len(indices)
            if timing_only and completed == warmup:
                torch.cuda.synchronize()
                throughput_start = time.perf_counter()
        torch.cuda.synchronize()
        if timing_only:
            training_seconds = time.perf_counter() - throughput_start
            validation_started = time.perf_counter()
            predict(model, validation_x)
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - validation_started
            timing = {"time": timestamp(), "job": job["id"], "method": job["method"], "device": job["device"],
                      "updates_per_second": measurement / training_seconds, "validation_seconds": validation_seconds,
                      "updates": total_updates, "max_epochs": CNN_RECIPE["max_epochs"], "updates_per_epoch": per_epoch,
                      "recipe": CNN_RECIPE, "training_windows": len(training_x), "batch_size": batch_size,
                      "estimated_training_seconds_max": total_updates * training_seconds / measurement,
                      "estimated_validation_seconds_max": CNN_RECIPE["max_epochs"] * validation_seconds,
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "timing_only": True}
            write_json(folder / "timing_only.json", timing)
            log("cnn_throughput", **timing)
            return timing
        cumulative_training += time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation_logits = predict(model, validation_x)
        score = float(((validation_logits >= 0) == (validation_y >= 0.5)).all(dim=1).float().mean().item())
        torch.cuda.synchronize()
        cumulative_validation += time.perf_counter() - validation_started
        should_stop = stopping.update(score) if smoke_updates is None else False
        if should_stop:
            stop_reason = "early_stopping"
        completed_epochs = epoch
        history = {"time": timestamp(), "identity": run_identity, "step": completed, "epoch": epoch,
                   "loss": epoch_loss / samples_seen, "source_accuracy": score,
                   "samples_seen": samples_seen, "bad_epochs": stopping.bad_epochs,
                   "training_seconds": cumulative_training, "validation_seconds": cumulative_validation}
        with (folder / "history.jsonl").open("ab") as stream:
            stream.write(canonical(history) + b"\n")
        if score > best:
            best = score
            save_torch(folder / "best.pt", {"model": model.state_dict(), "step": completed, "epoch": epoch, "score": score,
                                           "identity": run_identity, "source_identity": pool["source_identity"]})
        if epoch % 10 == 0 or epoch == 1 or should_stop or epoch == max_epochs:
            log("cnn_progress", id=job["id"], epoch=epoch, updates=completed, source_accuracy=score,
                best=best, loss=float(loss.detach()), training_seconds=cumulative_training)
            write_json(folder / "progress.json", {"identity": run_identity, "updated": timestamp(), "epoch": epoch,
                                                  "updates": completed, "total_updates": total_updates, "source_accuracy": score,
                                                  "training_seconds": cumulative_training, "validation_seconds": cumulative_validation})
        save_torch(last_path, {"identity": run_identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "completed": completed, "epoch": epoch, "best": best, "early_stopping": stopping.state_dict(),
                               "stop_reason": stop_reason, "training_seconds": cumulative_training,
                               "validation_seconds": cumulative_validation, "torch_rng": torch.get_rng_state(),
                               "cuda_rng": torch.cuda.get_rng_state(), "sampling_rng": sampling.get_state(),
                               "python_rng": random.getstate(), "numpy_rng": np.random.get_state()})
        if should_stop:
            break
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    frozen_sha256 = file_hash(folder / "best.pt")
    evaluation_start = time.perf_counter()
    target_x, target_y, target_valid, target_states = target_pool(run, job, pool, frozen_sha256, output_folder=folder)
    target_logits = predict(model, torch.as_tensor(target_x, device="cuda"))
    metrics = classification_metrics(target_logits, torch.as_tensor(target_y), target_valid, target_states)
    np.savez_compressed(folder / "target_predictions.npz", logits=target_logits.cpu().numpy(), labels=target_y, valid=target_valid)
    seen_x, seen_y, seen_valid, seen_groups = seen_pool(run, job, pool, frozen_sha256, output_folder=folder)
    seen_logits = predict(model, torch.as_tensor(seen_x, device="cuda"))
    seen_metrics = classification_metrics(seen_logits, torch.as_tensor(seen_y), seen_valid, seen_groups)
    np.savez_compressed(folder / "seen_predictions.npz", logits=seen_logits.cpu().numpy(), labels=seen_y, valid=seen_valid)
    result = {"identity": run_identity, "job": job, "source_identity": pool["source_identity"],
              "initial_weights_sha256": initial_hash, "best_checkpoint_sha256": frozen_sha256,
              "best_step": checkpoint["step"], "source_accuracy": checkpoint["score"],
              "target": metrics, "seen_conditions": seen_metrics, "training_seconds": cumulative_training, "validation_seconds": cumulative_validation,
              "evaluation_seconds": time.perf_counter() - evaluation_start, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "completed": timestamp(), "cost": cost, "cost_profiling_seconds": cost_seconds,
              "mode": "smoke_only" if smoke_updates else "formal", "updates_completed": completed,
              "epochs_completed": completed_epochs, "best_epoch": checkpoint["epoch"], "stop_reason": stop_reason,
              "early_stopping": stopping.state_dict(), "recipe": CNN_RECIPE,
              "formal_updates": formal_updates, "performance_claims_allowed": smoke_updates is None}
    write_json(result_path, result)
    log("cnn_complete", id=job["id"], target_accuracy=metrics["exact_set_accuracy"], epochs=completed_epochs,
        updates=completed, stop_reason=stop_reason, wall_seconds=result["wall_seconds"])
    return result
