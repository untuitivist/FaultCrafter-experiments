from __future__ import annotations

import argparse
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert, welch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import kurtosis, wasserstein_distance
from sklearn.decomposition import PCA

from faultcrafter.models import CNN1D

from .classifier import configure_runtime
from .common import ROOT, file_hash, identity, read_json, seed_for, timestamp, write_json
from .pools import source_pool, target_pool


@torch.inference_mode()
def features(model, signals):
    content, style, embeddings, predictions = [], [], [], []
    device = next(model.parameters()).device
    for start in range(0, len(signals), 128):
        inputs = torch.as_tensor(signals[start:start + 128], device=device)
        hidden = inputs
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4, model.layer5, model.layer6):
            hidden = layer(hidden)
        centers = hidden.mean(dim=-1, keepdim=True)
        deviations = hidden.var(dim=-1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        content.append(((hidden - centers) / deviations).flatten(start_dim=1).cpu().numpy())
        style.append(torch.cat((centers[..., 0], deviations[..., 0]), dim=1).cpu().numpy())
        embeddings.append(hidden.flatten(start_dim=1).cpu().numpy())
        predictions.append((model.classifier(hidden.flatten(start_dim=1)) >= 0).cpu().numpy())
    return {"content": np.concatenate(content), "style": np.concatenate(style),
            "embedding": np.concatenate(embeddings), "predictions": np.concatenate(predictions)}


def fit_transform(source):
    center = source.mean(axis=0)
    deviation = np.maximum(source.std(axis=0), 1e-6)
    standardized = (source - center) / deviation
    projection = PCA(n_components=min(16, len(source) - 1, source.shape[1]), svd_solver="randomized", random_state=42,
                     n_oversamples=8, iterated_power=2).fit(standardized)
    projected = projection.transform(standardized)
    distances = pdist(projected)
    bandwidth = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    return center, deviation, projection, max(bandwidth, 1e-6), projected


def transform(values, fitted):
    center, deviation, projection, bandwidth, source = fitted
    return projection.transform((values - center) / deviation)


def mmd_squared(real, generated, bandwidth):
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    real_kernel = np.exp(-cdist(real, real, "sqeuclidean") / (2 * bandwidth**2))
    fake_kernel = np.exp(-cdist(generated, generated, "sqeuclidean") / (2 * bandwidth**2))
    cross_kernel = np.exp(-cdist(real, generated, "sqeuclidean") / (2 * bandwidth**2))
    real_mean = (real_kernel.sum() - np.trace(real_kernel)) / (len(real) * (len(real) - 1))
    fake_mean = (fake_kernel.sum() - np.trace(fake_kernel)) / (len(generated) * (len(generated) - 1))
    return float(real_mean + fake_mean - 2 * cross_kernel.mean())


def frechet(real, generated):
    real_cov = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * 1e-8
    fake_cov = np.cov(generated, rowvar=False) + np.eye(real.shape[1]) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(real_cov)
    square_root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = square_root @ fake_cov @ square_root
    covariance_distance = np.trace(real_cov + fake_cov) - 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(middle), 0)).sum()
    return float(max(0, np.square(real.mean(axis=0) - generated.mean(axis=0)).sum() + covariance_distance))


def signal_statistics(real, generated, sample_rate):
    real = real[:, 0].astype(np.float64)
    generated = generated[:, 0].astype(np.float64)
    edges = np.r_[-np.inf, np.linspace(-1, 1, 65)[1:-1], np.inf]
    real_hist = np.histogram(real, bins=edges)[0].astype(float) + 1e-12
    fake_hist = np.histogram(generated, bins=edges)[0].astype(float) + 1e-12
    real_hist /= real_hist.sum()
    fake_hist /= fake_hist.sum()
    midpoint = (real_hist + fake_hist) / 2
    divergence = 0.5 * (np.sum(real_hist * np.log(real_hist / midpoint)) + np.sum(fake_hist * np.log(fake_hist / midpoint)))
    rms_real, rms_fake = np.sqrt(np.mean(real**2, axis=1)), np.sqrt(np.mean(generated**2, axis=1))
    _, real_psd = welch(real, fs=sample_rate, nperseg=1024, noverlap=512, axis=-1)
    _, fake_psd = welch(generated, fs=sample_rate, nperseg=1024, noverlap=512, axis=-1)
    envelope_real = np.abs(hilbert(real - real.mean(axis=-1, keepdims=True), axis=-1))
    envelope_fake = np.abs(hilbert(generated - generated.mean(axis=-1, keepdims=True), axis=-1))
    _, real_envelope = welch(envelope_real, fs=sample_rate, nperseg=1024, noverlap=512, axis=-1)
    _, fake_envelope = welch(envelope_fake, fs=sample_rate, nperseg=1024, noverlap=512, axis=-1)
    real_kurtosis = kurtosis(real, axis=-1, fisher=False)
    fake_kurtosis = kurtosis(generated, axis=-1, fisher=False)
    kurtosis_error = (float(abs(real_kurtosis.mean() - fake_kurtosis.mean()))
                      if np.isfinite(real_kurtosis).all() and np.isfinite(fake_kurtosis).all() else None)
    return {
        "wasserstein": float(wasserstein_distance(real.flatten(), generated.flatten())),
        "js": float(divergence), "rms_error": float(abs(rms_real.mean() - rms_fake.mean())),
        "kurtosis_error": kurtosis_error,
        "psd_error": float(np.abs(real_psd.mean(axis=0) - fake_psd.mean(axis=0)).sum() / max(real_psd.mean(axis=0).sum(), 1e-12)),
        "envelope_error": float(np.abs(real_envelope.mean(axis=0) - fake_envelope.mean(axis=0)).sum() /
                                max(real_envelope.mean(axis=0).sum(), 1e-12)),
    }


@lru_cache(maxsize=1)
def reference_context(run, output, reference_id, device):
    result = read_json(output / "jobs" / reference_id / "result.json")
    pool = source_pool(run, result["job"])
    model = CNN1D(len(pool["components"])).to(device).eval()
    checkpoint = torch.load(output / "jobs" / reference_id / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rng = np.random.default_rng(seed_for(reference_id, "quality_source_features"))
    count = min(512, len(pool["roles"]["train"]["x"]))
    indices = rng.choice(len(pool["roles"]["train"]["x"]), count, replace=False)
    source = pool["roles"]["train"]["x"][indices]
    source_features = features(model, source)
    fitted = {name: fit_transform(source_features[name]) for name in ("embedding", "content", "style")}
    return model, fitted, source, result["best_checkpoint_sha256"]


def evaluate(run, output, device="cuda", *, completed_only=False):
    sample_rates = {record["device"]: record["sample_rate"] for record in read_json(run / "inventory.json")["records"]}
    sample_rates["cwru"] = 12000
    results = [read_json(path) for path in (output / "jobs").glob("*/result.json")]
    references = {tuple(result["job"][key] for key in ("track", "device", "channel", "target", "split")): result["job"]["id"]
                  for result in results if result["job"]["method"] == "real_only"}
    selected = [result for result in results if result["job"]["track"] in ("T1", "T2") and
                result["job"].get("arm") == "real_generated"]
    if completed_only:
        selected = [result for result in selected if tuple(result["job"][key] for key in
                    ("track", "device", "channel", "target", "split")) in references]
    selected.sort(key=lambda result: tuple(str(result["job"][key]) for key in ("split", "track", "device", "channel", "method")))
    completed = []
    for result in selected:
        job = result["job"]
        destination = output / "quality" / (job["id"] + ".json")
        if destination.exists():
            previous = read_json(destination)
            if previous["code_sha256"] != file_hash(Path(__file__)) or previous["feature_device"] != device:
                raise RuntimeError("Quality-analysis code or device changed; use a separate output directory")
            completed.append(previous)
            continue
        reference_key = tuple(job[key] for key in ("track", "device", "channel", "target", "split"))
        model, fitted, source_signals, reference_sha = reference_context(run, output, references[reference_key], device)
        pool = source_pool(run, job)
        real, labels, valid, states = target_pool(run, job, pool, result["best_checkpoint_sha256"],
                                                 output_folder=output / "jobs" / job["id"])
        synthetic_id = identity({key: job.get(key) for key in ("parent", "device", "channel", "track", "target", "split")})[:24]
        synthetic_folder = output / "synthetic" / synthetic_id
        with np.load(synthetic_folder / "signals.npz") as arrays:
            generated, generated_labels = arrays["x"], arrays["y"]
        real_features, fake_features = features(model, real), features(model, generated)
        state_metrics = []
        for state in pool["states"]:
            vector = np.asarray([part in pool["labels"][state] for part in pool["components"]])
            real_indices = np.flatnonzero(np.all(labels == vector, axis=1) & valid)
            fake_indices = np.flatnonzero(np.all(generated_labels == vector, axis=1))
            count = min(63, len(real_indices), len(fake_indices))
            if count < 2:
                raise RuntimeError("Insufficient samples for distribution metrics")
            real_indices, fake_indices = real_indices[:count], fake_indices[:count]
            statistics = signal_statistics(real[real_indices], generated[fake_indices], sample_rates[job["device"]])
            projected = {}
            for name in fitted:
                real_projected = transform(real_features[name][real_indices], fitted[name])
                fake_projected = transform(fake_features[name][fake_indices], fitted[name])
                statistics[name + "_mmd2"] = mmd_squared(real_projected, fake_projected, fitted[name][3])
                projected[name] = (real_projected, fake_projected)
            real_projected, fake_projected = projected["embedding"]
            statistics["frechet"] = frechet(real_projected, fake_projected)
            statistics["pairwise_distance"] = float(pdist(fake_projected).mean())
            statistics["nearest_source_distance"] = float(cdist(fake_projected, fitted["embedding"][4]).min(axis=1).mean())
            statistics["duplicate_rate"] = float(1 - len(np.unique(generated[fake_indices, 0], axis=0)) / count)
            statistics["label_consistency"] = float(np.all(fake_features["predictions"][fake_indices] == vector, axis=1).mean())
            state_metrics.append({"state": state, "count_per_distribution": count, **statistics})
        metrics = {name: (float(np.mean([entry[name] for entry in state_metrics]))
                          if all(entry[name] is not None for entry in state_metrics) else None)
                   for name in state_metrics[0] if name not in ("state", "count_per_distribution")}
        entry = {"job": job, "metrics": metrics, "per_state": state_metrics, "created": timestamp(),
                 "reference_cnn_sha256": reference_sha, "generator_sha256": read_json(synthetic_folder / "manifest.json")["checkpoint_sha256"],
                 "window_length": 1024, "scale": "Native parent-normalized waveform scale",
                 "feature_device": device, "feature_precision": "FP32",
                 "feature_projection": "Source-only standardization and 16-component PCA; bandwidth is the median source pairwise distance",
                 "mmd": "Unbiased squared MMD; finite-sample estimates may be negative",
                 "undefined_statistics": "Constant-window kurtosis is undefined and is stored as null, never replaced by zero",
                 "code_sha256": file_hash(Path(__file__))}
        write_json(destination, entry)
        completed.append(entry)
        print(f"quality {len(completed)}/{len(selected)} {job['method']} {job['device']} {job['channel']} seed={job['split']}", flush=True)
    coverage_path = output / "coverage.json"
    expected = (sum(job["track"] in ("T1", "T2") and job.get("arm") == "real_generated"
                    for job in read_json(coverage_path)["selected"]) if coverage_path.exists() else len(selected))
    write_json(output / "quality_summary.json", {"completed": len(completed), "expected": expected,
                                               "complete": len(completed) == expected, "feature_device": device, "records": completed})
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    if arguments.watch and arguments.device != "cpu":
        raise ValueError("Concurrent quality analysis uses CPU so generator and CNN execution retain the GPU")
    if arguments.device == "cuda":
        configure_runtime()
    else:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    while True:
        evaluate(arguments.data_run, arguments.output, arguments.device, completed_only=arguments.watch)
        if not arguments.watch:
            break
        status = read_json(arguments.output / "status.json")
        summary = read_json(arguments.output / "quality_summary.json")
        if status["state"] == "complete":
            if not summary["complete"]:
                evaluate(arguments.data_run, arguments.output, arguments.device)
                summary = read_json(arguments.output / "quality_summary.json")
                if not summary["complete"]:
                    raise RuntimeError("Training completed but quality-analysis cases are missing")
            break
        if status["state"] in ("failed", "incomplete"):
            raise RuntimeError("Comparison execution stopped before all quality-analysis cases became available")
        time.sleep(30)


if __name__ == "__main__":
    main()
