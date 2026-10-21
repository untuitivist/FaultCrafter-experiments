from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis, skew

from faultcrafter.models import CNN1D

from .artifact_selection import add_filters, reference_job, select_jobs
from .common import METHODS, file_hash, identity, read_json, seed_for, timestamp, write_json
from .data import CHANNELS, PARENT, partition_values
from .pools import augmentation_pool, seen_pool, source_pool, superposition_pool, target_pool, update_budget
from .quality import features, fit_transform, transform


INDICATORS = ("mean", "absolute_mean", "standard_deviation", "rms", "maximum_absolute", "peak_to_peak",
              "skewness", "kurtosis", "crest_factor", "shape_factor", "impulse_factor", "clearance_factor")


def balanced_indices(groups, count, seed):
    generator = np.random.default_rng(seed)
    groups = np.asarray(groups)
    return np.concatenate([np.sort(generator.choice(np.flatnonzero(groups == group),
                           min(count, int((groups == group).sum())), replace=False)) for group in sorted(set(groups))])


def indicators(signals):
    values = signals[:, 0].astype(np.float64)
    absolute = np.abs(values)
    maximum = absolute.max(axis=1)
    mean_absolute = absolute.mean(axis=1)
    root_mean_square = np.sqrt(np.square(values).mean(axis=1))
    root_amplitude = np.square(np.sqrt(absolute).mean(axis=1))
    divide = lambda numerator, denominator: np.divide(numerator, denominator,
              out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.stack((values.mean(axis=1), mean_absolute, values.std(axis=1), root_mean_square, maximum,
                     np.ptp(values, axis=1), skew(values, axis=1), kurtosis(values, axis=1, fisher=False),
                     divide(maximum, root_mean_square), divide(root_mean_square, mean_absolute),
                     divide(maximum, mean_absolute), divide(maximum, root_amplitude)), axis=1)


def physical_scales(data_run, job, samples, records):
    channel = CHANNELS[job["device"]].index(job["channel"])
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample["record"], sample.get("start", 0), sample.get("stop")].append((index, sample))
    scales = np.full(len(samples), np.nan)
    for (record_id, start, stop), entries in grouped.items():
        item = records[record_id]
        try:
            values = partition_values(data_run, item, {"start": start, "stop": stop})
        except (OSError, ValueError):
            continue
        for index, sample in entries:
            parent_start = sample["relative_start"] // PARENT * PARENT
            parent = values[parent_start:parent_start + PARENT, channel]
            if len(parent) and np.isfinite(parent).all():
                scale = float(np.abs(parent).max())
                if scale > 0:
                    scales[index] = scale
    return scales


def save_role(folder, name, values, labels, valid, groups, sample_ids, model, fitted, sample_rate,
              waveforms_per_cell, seed, scales=None, representation=None):
    valid = np.asarray(valid, dtype=bool)
    groups = np.asarray(groups, dtype=str)
    all_scales = np.full(len(values), np.nan) if scales is None else scales
    np.savez_compressed(folder / (name + "_waveforms.npz"), normalized=values, labels=labels, valid=valid,
                        groups=groups, parent_scale=all_scales, sample_rate_hz=np.asarray(sample_rate))
    selected = balanced_indices(groups, waveforms_per_cell, seed)
    representation = features(model, values[valid]) if representation is None else representation
    projected = transform(representation["embedding"], fitted)
    np.savez_compressed(folder / (name + "_features.npz"), **representation, projected=projected,
                        labels=labels[valid], groups=groups[valid], row_indices=np.flatnonzero(valid))
    waveforms = values[selected]
    frequencies, power = welch(waveforms[:, 0], fs=sample_rate, nperseg=min(4096, values.shape[-1]))
    envelope = np.abs(hilbert(waveforms[:, 0] - waveforms[:, 0].mean(axis=1, keepdims=True), axis=1))
    envelope_frequencies, envelope_power = welch(envelope, fs=sample_rate, nperseg=min(4096, values.shape[-1]))
    raw_scales = np.full(len(selected), np.nan) if scales is None else scales[selected]
    np.savez_compressed(folder / (name + "_signals.npz"), normalized=waveforms,
                        original_scale=waveforms * raw_scales[:, None, None], parent_scale=raw_scales,
                        labels=labels[selected], valid=valid[selected], groups=groups[selected], row_indices=selected,
                        time_seconds=np.arange(values.shape[-1]) / sample_rate,
                        fft_hz=np.fft.rfftfreq(values.shape[-1], 1 / sample_rate),
                        fft_magnitude=np.abs(np.fft.rfft(waveforms[:, 0], axis=1)) / values.shape[-1],
                        psd_hz=frequencies, psd=power, envelope_hz=envelope_frequencies, envelope_psd=envelope_power)
    statistics = indicators(values)
    original_statistics = statistics.copy()
    original_statistics[:, :6] *= all_scales[:, None]
    np.savez_compressed(folder / (name + "_statistics.npz"), indicators=statistics,
                        original_scale_indicators=original_statistics, parent_scale=all_scales,
                        indicator_names=np.asarray(INDICATORS), labels=labels, valid=valid, groups=groups)
    write_json(folder / (name + "_samples.json"), {"samples": sample_ids, "waveform_indices": selected.tolist(),
               "sample_rate_hz": sample_rate, "feature_rows": np.flatnonzero(valid).tolist(),
               "original_scale_available": scales is not None,
               "normalization": "Each real window inherits the maximum absolute value of its parent segment.",
               "generated_scale": "External method ports output normalized signals and have no learned amplitude head." if scales is None else None})


def collect(run, data_run, output, jobs, *, device="cpu", waveforms_per_cell=32):
    output.mkdir(parents=True, exist_ok=True)
    inventory = read_json(data_run / "inventory.json")
    records = {item["id"]: item for item in inventory["records"]}
    sample_rates = {item["device"]: item["sample_rate"] for item in inventory["records"]}
    sample_rates["cwru"] = 12000
    references = {}
    generated_sets = set()
    receipts = []
    current_reference = None
    ordered_jobs = sorted(jobs, key=lambda job: (reference_job(run, job)["id"], job["id"]))
    for job in ordered_jobs:
        reference = reference_job(run, job)
        reference_id = reference["id"]
        pool = source_pool(data_run, reference)
        reference_result = read_json(run / "jobs" / reference_id / "result.json")
        checkpoint_path = run / "jobs" / reference_id / "best.pt"
        if file_hash(checkpoint_path) != reference_result["best_checkpoint_sha256"]:
            raise RuntimeError("Reference checkpoint checksum mismatch")
        source = pool["roles"]["train"]
        if current_reference != reference_id:
            model = CNN1D(len(pool["components"])).to(device).eval()
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
            source_features = features(model, source["x"])
            fitted = fit_transform(source_features["embedding"])
            current_reference = reference_id
        if reference_id not in references:
            folder = output / "references" / reference_id
            folder.mkdir(parents=True, exist_ok=True)
            train_groups = [pool["states"][state] + "/" + source["conditions"][unit % len(source["conditions"]) ]
                            for state, unit in zip(source["state"], source["unit"])]
            scales = physical_scales(data_run, reference, source["samples"], records)
            save_role(folder, "source", source["x"], source["y"], np.ones(len(source["x"]), dtype=bool), train_groups,
                      source["samples"], model, fitted, sample_rates[job["device"]], waveforms_per_cell,
                      seed_for(reference_id, "artifact_source"), scales, source_features)
            real, labels, valid, states = target_pool(data_run, reference, pool, reference_result["best_checkpoint_sha256"],
                                                      output_folder=folder)
            target_ids = read_json(folder / "target_manifest.json")["samples"]
            save_role(folder, "target", real, labels, valid, states, target_ids, model, fitted,
                      sample_rates[job["device"]], waveforms_per_cell, seed_for(reference_id, "artifact_target"),
                      physical_scales(data_run, reference, target_ids, records))
            seen, seen_labels, seen_valid, seen_groups = seen_pool(data_run, reference, pool,
                 reference_result["best_checkpoint_sha256"], output_folder=folder)
            seen_ids = [sample for cell in read_json(folder / "seen_manifest.json")["cells"] for sample in cell["samples"]]
            save_role(folder, "seen", seen, seen_labels, seen_valid, seen_groups, seen_ids, model, fitted,
                      sample_rates[job["device"]], waveforms_per_cell, seed_for(reference_id, "artifact_seen"),
                      physical_scales(data_run, reference, seen_ids, records))
            np.savez_compressed(folder / "feature_transform.npz", center=fitted[0], scale=fitted[1],
                                pca_components=fitted[2].components_, pca_mean=fitted[2].mean_, bandwidth=fitted[3])
            write_json(folder / "reference.json", {"job": reference, "checkpoint_sha256": file_hash(checkpoint_path),
                       "components": pool["components"], "sample_rate_hz": sample_rates[job["device"]],
                       "fit_data": "Source training windows only", "feature_definition": "Frozen paired Real-only CNN; layer6 embedding, standardized temporal content, per-channel mean/std style."})
            references[reference_id] = folder
        folder = output / "jobs" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        job_result = read_json(run / "jobs" / job["id"] / "result.json")
        added = None
        generated_id = job["id"]
        if job["method"] in METHODS:
            generated_id = identity({key: job.get(key) for key in ("parent", "device", "channel", "track", "target", "split")})[:24]
            synthetic = run / "synthetic" / generated_id
            generation = read_json(synthetic / "manifest.json")
            if file_hash(synthetic / "signals.npz") != generation["signals_sha256"]:
                raise RuntimeError("Generated pool checksum mismatch")
            with np.load(synthetic / "signals.npz") as arrays:
                added = np.array(arrays["x"]), np.array(arrays["y"])
            write_json(folder / "generation.json", generation)
        elif job["method"] == "augmentation":
            added = augmentation_pool(pool, job, update_budget(pool, job)[2])
        elif job["method"] == "superposition":
            signals, labels, receipt = superposition_pool(pool, job)
            added = signals, labels
            write_json(folder / "generation.json", receipt)
        data_folder = output / "pools" / generated_id
        if added is not None and generated_id not in generated_sets:
            data_folder.mkdir(parents=True, exist_ok=True)
            groups = ["+".join(part for part, present in zip(pool["components"], row) if present) or "normal" for row in added[1]]
            if job["method"] in METHODS:
                groups = [request["state"] + "@" + ",".join(map(str, request["coordinates"][index % len(request["coordinates"])]))
                          for request in generation["requests"] for index in request["selected_candidate_indices"]]
                if len(groups) != len(added[0]):
                    raise RuntimeError("Generated waveform request count mismatch")
            save_role(data_folder, "generated", added[0], added[1], np.ones(len(added[0]), dtype=bool), groups, [], model,
                      fitted, sample_rates[job["device"]], waveforms_per_cell, seed_for(job["id"], "artifact_generated"))
            generated_sets.add(generated_id)
        entry = {"job": job, "reference": reference_id, "diagnostic_checkpoint_sha256": job_result["best_checkpoint_sha256"],
                 "reference_checkpoint_sha256": reference_result["best_checkpoint_sha256"], "created": timestamp(),
                 "generated_data": data_folder.relative_to(output).as_posix() if added is not None else None}
        write_json(folder / "inference.json", entry)
        receipts.append(entry)
        print(f"inference {len(receipts)}/{len(jobs)} {job['method']} {job['device']} {job['channel']} seed={job['split']}", flush=True)
    manifest = {"created": timestamp(), "device": device, "waveforms_per_cell": waveforms_per_cell,
                "jobs": receipts, "code_sha256": file_hash(Path(__file__)), "feature_precision": "float32",
                "statistics_scope": "All retained windows; waveform panels use a deterministic balanced subset; invalid real windows remain identified."}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Collect figure data using frozen models, after training.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--waveforms-per-cell", type=int, default=32)
    add_filters(parser)
    arguments = parser.parse_args()
    if arguments.waveforms_per_cell < 1:
        raise ValueError("A positive waveform count is required")
    torch.set_num_threads(4)
    collect(arguments.run, arguments.data_run, arguments.output, select_jobs(arguments.run, arguments),
            device=arguments.device, waveforms_per_cell=arguments.waveforms_per_cell)


if __name__ == "__main__":
    main()
