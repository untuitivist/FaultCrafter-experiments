from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert, welch

from .common import file_hash, read_json, seed_for, timestamp, write_json
from .data import T4_STATES
from .generative_models import build_model
from .generators import ConditionSchema, FIELD_INDICES, SCALES
from .inference import indicators
from .pools import seen_pool, source_pool, target_pool
from .quality import features, mmd_squared, reference_context, signal_statistics, transform


def responses(signals, sample_rate):
    values = signals[:, 0].astype(np.float64)
    frequencies, power = welch(values, fs=sample_rate, nperseg=1024, axis=-1)
    envelopes = np.abs(hilbert(values - values.mean(axis=-1, keepdims=True), axis=-1))
    _, envelope_power = welch(envelopes, fs=sample_rate, nperseg=1024, axis=-1)
    return np.asarray([np.sqrt(np.square(values).mean(axis=-1)).mean(),
                       frequencies[1 + power[:, 1:].mean(axis=0).argmax()],
                       frequencies[1 + envelope_power[:, 1:].mean(axis=0).argmax()]])


def bicoherence(signals):
    values = signals[:, 0].astype(np.float64)
    segments = np.stack([values[:, start:start + 512] for start in (0, 256, 512)], axis=1).reshape(-1, 512)
    segments -= segments.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(segments * np.hanning(512), axis=-1)
    bins = np.arange(1, 256, 4)
    first, second = np.meshgrid(bins, bins, indexing="ij")
    valid = first + second < 256
    pair = spectrum[:, first[valid]] * spectrum[:, second[valid]]
    combined = spectrum[:, (first + second)[valid]]
    denominator = np.square(np.abs(pair)).mean(axis=0) * np.square(np.abs(combined)).mean(axis=0)
    result = np.zeros(first.shape)
    result[valid] = np.square(np.abs((pair * combined.conj()).mean(axis=0))) / np.maximum(denominator, 1e-30)
    return result, valid, bins


def real_grid(data_run, comparisons, output, job, encoder_sha):
    pool = source_pool(data_run, job)
    all_states_job = {**job, "track": "T4"}
    seen, _, seen_valid, seen_groups = seen_pool(data_run, all_states_job, pool, encoder_sha, output_folder=output)
    target, _, target_valid, target_groups = target_pool(data_run, all_states_job, pool, encoder_sha, output_folder=output)
    grid = {}
    for group in sorted(set(seen_groups)):
        condition, state = group.split("/", 1)
        indices = np.flatnonzero((np.asarray(seen_groups) == group) & seen_valid)
        grid[condition, state] = seen[indices]
    for state in sorted(set(target_groups)):
        indices = np.flatnonzero((np.asarray(target_groups) == state) & target_valid)
        grid[job["target"], state] = target[indices]
    if any(len(values) < 2 for values in grid.values()):
        raise RuntimeError("Condition analysis requires independent real evaluation samples in every cell")
    return pool, grid


def evaluate_case(data_run, comparisons, output, job, reference_id, steps, count):
    destination = output / "conditions" / job["id"]
    code_hash = file_hash(Path(__file__))
    if (destination / "result.json").exists():
        previous = read_json(destination / "result.json")
        expected_steps = 0 if job["method"] == "superposition" else steps
        if previous["code_sha256"] != code_hash or previous["sampling_steps"] != expected_steps or previous["samples_per_request"] != count:
            raise RuntimeError("Condition analysis configuration changed; choose a fresh output directory")
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    schema = ConditionSchema(read_json(data_run / "inventory.json")["records"])
    encoder, fitted, _, encoder_sha = reference_context(data_run, comparisons, reference_id, "cuda")
    pool, real = real_grid(data_run, comparisons, destination, job, encoder_sha)
    coordinates = sorted({condition for condition, state in real}, key=lambda condition: schema.coordinates["bjtu", condition])
    source_conditions = [condition for condition in coordinates if condition != job["target"]]
    superposition = job["method"] == "superposition"
    selected_states = (T4_STATES["bjtu"] if superposition else
                       pool["states"] + (T4_STATES["bjtu"] if job["method"] in ("iccdm", "flowts") else []))
    checkpoint_sha, model = None, None
    if not superposition:
        generator_folder = comparisons / "generators" / job["parent"]
        generator_record = read_json(generator_folder / "result.json")
        checkpoint_sha = file_hash(generator_folder / "best.pt")
        if checkpoint_sha != generator_record["best_checkpoint_sha256"]:
            raise RuntimeError("Frozen T2 generator checksum mismatch")
        model = build_model(job["method"], schema.dimension).cuda().eval()
        model.load_state_dict(torch.load(generator_folder / "best.pt", map_location="cpu", weights_only=False)["model"])
    generated, nearest, arrays, per_cell = {}, {}, {}, []
    for state in selected_states:
        for condition in coordinates:
            point = schema.coordinates["bjtu", condition]
            scaled_point = np.asarray(point) / SCALES[list(FIELD_INDICES["bjtu"])]
            nearest[condition] = min(source_conditions, key=lambda candidate: (
                float(np.square(np.asarray(schema.coordinates["bjtu", candidate]) / SCALES[list(FIELD_INDICES["bjtu"])] - scaled_point).sum()),
                tuple(schema.coordinates["bjtu", candidate])))
            if superposition:
                rng = np.random.default_rng(seed_for("bjtu", job["split"], state, "condition_superposition"))
                source = pool["roles"]["train"]
                condition_index = source["conditions"].index(nearest[condition])
                singles = [next(name for name in pool["states"] if pool["labels"][name] == [part])
                           for part in pool["labels"][state]]
                pieces, selected_indices = [], []
                for single in singles:
                    candidates = np.flatnonzero((source["state"] == pool["states"].index(single)) &
                        (source["unit"] % len(source["conditions"]) == condition_index))
                    indices = rng.choice(candidates, count, replace=True)
                    pieces.append(source["x"][indices])
                    selected_indices.append(indices)
                values = np.sum(pieces, axis=0)
                values = values / np.maximum(np.abs(values).max(axis=(1, 2), keepdims=True), 1e-12)
                arrays[f"cell_{len(per_cell):03d}_source_indices"] = np.stack(selected_indices)
            else:
                encoded = schema.encode("bjtu", job["channel"], state, point)
                torch.manual_seed(seed_for("bjtu", job["channel"], job["split"], state, "matched_condition_noise"))
                with torch.inference_mode():
                    conditions = torch.as_tensor(np.repeat(encoded[None], count, axis=0), device="cuda")
                    values = np.concatenate([model.sample(conditions[start:start + 8], steps=steps).cpu().numpy()
                                             for start in range(0, count, 8)])
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite frozen T2 condition response")
            generated[condition, state] = values
            cell_index = len(per_cell)
            key = f"cell_{cell_index:03d}"
            arrays[key + "_generated"] = values
            arrays[key + "_real"] = real[condition, state]
            arrays[key + "_indicators_generated"] = indicators(values)
            arrays[key + "_indicators_real"] = indicators(real[condition, state])
            real_features, fake_features = features(encoder, real[condition, state]), features(encoder, values)
            size = min(len(real[condition, state]), len(values))
            metrics = signal_statistics(real[condition, state][:size], values[:size], 64000)
            for name in ("content", "style", "embedding"):
                first = transform(real_features[name], fitted[name])
                second = transform(fake_features[name], fitted[name])
                arrays[key + "_" + name + "_real"] = first
                arrays[key + "_" + name + "_generated"] = second
                metrics[name + "_mmd2"] = mmd_squared(first[:size], second[:size], fitted[name][3])
            vector = np.asarray([part in pool["labels"][state] for part in pool["components"]])
            metrics["label_consistency"] = float(np.all(fake_features["predictions"] == vector, axis=-1).mean())
            entry = {"array_key": key, "condition": condition, "coordinates": point, "state": state,
                     "unseen_composite": state in T4_STATES["bjtu"], "count_real": len(real[condition, state]),
                     "count_generated": len(values), "real_response": responses(real[condition, state], 64000).tolist(),
                     "generated_response": responses(values, 64000).tolist(), "metrics": metrics}
            if state in T4_STATES["bjtu"]:
                first, mask, bins = bicoherence(real[condition, state][:size])
                second, _, _ = bicoherence(values[:size])
                entry["metrics"]["interaction_error"] = float(np.square(first[mask] - second[mask]).mean())
                arrays[key + "_bicoherence_real"] = first
                arrays[key + "_bicoherence_generated"] = second
                arrays["bicoherence_frequency_hz"] = bins * 64000 / 512
            per_cell.append(entry)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    by_cell = {(entry["condition"], entry["state"]): entry for entry in per_cell}
    edges = []
    for state in (() if superposition else pool["states"]):
        for axis in (0, 1):
            for first_condition in coordinates:
                point = schema.coordinates["bjtu", first_condition]
                candidates = [condition for condition in coordinates
                              if all(schema.coordinates["bjtu", condition][index] == point[index] for index in range(3) if index != axis)
                              and schema.coordinates["bjtu", condition][axis] > point[axis]]
                if not candidates:
                    continue
                second_condition = min(candidates, key=lambda condition: schema.coordinates["bjtu", condition][axis])
                first, second = by_cell[first_condition, state], by_cell[second_condition, state]
                real_delta = np.subtract(second["real_response"], first["real_response"])
                for arm in ("requested", "nearest_source"):
                    left = first if arm == "requested" else by_cell[nearest[first_condition], state]
                    right = second if arm == "requested" else by_cell[nearest[second_condition], state]
                    fake_delta = np.subtract(right["generated_response"], left["generated_response"])
                    error = np.abs(fake_delta - real_delta)
                    edges.append({"state": state, "axis": axis, "first": first_condition, "second": second_condition,
                                  "arm": arm, "real_delta": real_delta.tolist(), "generated_delta": fake_delta.tolist(),
                                  "rms_response_error": float(error[0]), "peak_frequency_response_error": float(error[1]),
                                  "envelope_peak_response_error": float(error[2])})
    summaries = []
    for arm in (() if superposition else ("requested", "nearest_source")):
        cells = [entry for entry in per_cell if not entry["unseen_composite"]]
        metrics = {}
        for name in ("label_consistency", "content_mmd2", "style_mmd2"):
            values = []
            for cell in cells:
                other = cell if arm == "requested" else by_cell[nearest[cell["condition"]], cell["state"]]
                if name == "label_consistency":
                    values.append(other["metrics"][name])
                else:
                    feature_name = name.removesuffix("_mmd2")
                    first = arrays[cell["array_key"] + "_" + feature_name + "_real"]
                    second = arrays[other["array_key"] + "_" + feature_name + "_generated"]
                    size = min(len(first), len(second))
                    values.append(mmd_squared(first[:size], second[:size], fitted[feature_name][3]))
            metrics[name] = float(np.mean(values))
        metrics.update({name: float(np.mean([edge[name] for edge in edges if edge["arm"] == arm])) for name in
                        ("rms_response_error", "peak_frequency_response_error", "envelope_peak_response_error")})
        summaries.append({"arm": arm, "metrics": metrics})
    np.savez_compressed(destination / "arrays.npz", **arrays)
    result = {"job": job, "generator_sha256": checkpoint_sha, "reference_cnn_sha256": encoder_sha,
              "code_sha256": code_hash, "sampling_steps": 0 if superposition else steps, "samples_per_request": count,
              "source_identity": pool["source_identity"],
              "window_length": 1024, "sample_rate_hz": 64000, "signal_scale": "parent_normalized",
              "nearest_source": nearest, "per_cell": per_cell, "responses": edges, "summaries": summaries,
              "completed": timestamp(), "arrays_sha256": file_hash(destination / "arrays.npz")}
    write_json(destination / "result.json", result)
    return result
