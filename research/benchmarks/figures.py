from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .common import file_hash, read_json, write_json


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def diagnostic_curves(run, job, folder, component_names):
    with np.load(run / "jobs" / job["id"] / "target_predictions.npz") as arrays:
        logits, labels, valid = arrays["logits"], arrays["labels"].astype(bool), arrays["valid"]
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    truth = labels[valid]
    prediction = probabilities[valid] >= 0.5
    curves = {}
    figure, axes = plt.subplots(1, 3, figsize=(12, max(3.3, len(component_names) * 0.17)), constrained_layout=True)
    for component in range(labels.shape[1]):
        if len(np.unique(truth[:, component])) != 2:
            continue
        false_positive, true_positive, thresholds = roc_curve(truth[:, component], probabilities[valid, component])
        precision, recall, pr_thresholds = precision_recall_curve(truth[:, component], probabilities[valid, component])
        curves.update({f"roc_fpr_{component}": false_positive, f"roc_tpr_{component}": true_positive,
                       f"roc_thresholds_{component}": thresholds, f"pr_precision_{component}": precision,
                       f"pr_recall_{component}": recall, f"pr_thresholds_{component}": pr_thresholds})
        axes[0].plot(false_positive, true_positive, label=component_names[component])
        axes[1].plot(recall, precision, label=component_names[component])
    confusion = np.stack([confusion_matrix(truth[:, component], prediction[:, component], labels=[False, True]).ravel()
                          for component in range(labels.shape[1])])
    curves["component_confusion_tn_fp_fn_tp"] = confusion
    truth_sets = np.asarray(["+".join(name for name, present in zip(component_names, row) if present) or "normal" for row in truth])
    predicted_sets = np.asarray(["+".join(name for name, present in zip(component_names, row) if present) or "normal" for row in prediction])
    sets = sorted(set(truth_sets) | set(predicted_sets))
    curves["fault_set_labels"] = np.asarray(sets)
    curves["fault_set_confusion"] = confusion_matrix(truth_sets, predicted_sets, labels=sets)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[2].imshow(confusion, aspect="auto", cmap="Blues")
    axes[2].set(xticks=np.arange(4), xticklabels=["TN", "FP", "FN", "TP"],
                yticks=np.arange(len(component_names)), yticklabels=component_names, title="Component confusion")
    axes[2].tick_params(axis="y", labelsize=7)
    if len(component_names) <= 8:
        axes[0].legend(fontsize=6)
    np.savez_compressed(folder / "diagnostic_curves.npz", **curves)
    save_figure(figure, folder / "diagnostics.pdf")


def qualitative(reference, generated, folder):
    with np.load(reference / "target_signals.npz") as arrays:
        real = {name: np.array(arrays[name]) for name in arrays.files}
    with np.load(generated / "generated_signals.npz") as arrays:
        fake = {name: np.array(arrays[name]) for name in arrays.files}
    real_valid = np.flatnonzero(real["valid"])
    if not len(real_valid):
        return
    real_index = int(real_valid[0])
    matching = np.flatnonzero(np.all(fake["labels"] == real["labels"][real_index], axis=1))
    fake_index = int(matching[0]) if len(matching) else 0
    figure, axes = plt.subplots(2, 3, figsize=(12, 5.8))
    for values, index, label in ((real, real_index, "Real"), (fake, fake_index, "Generated")):
        signal = values["normalized"][index, 0]
        axes[0, 0].plot(values["time_seconds"], signal, label=label, alpha=0.75)
        axes[0, 1].plot(values["fft_hz"], values["fft_magnitude"][index], label=label)
        axes[0, 2].hist(signal, bins=48, density=True, alpha=0.45, label=label)
        axes[1, 0].semilogy(values["psd_hz"], np.maximum(values["psd"][index], 1e-20), label=label)
        axes[1, 1].semilogy(values["envelope_hz"], np.maximum(values["envelope_psd"][index], 1e-20), label=label)
    with np.load(reference / "target_statistics.npz") as arrays:
        real_statistics = arrays["indicators"][arrays["valid"] & np.all(arrays["labels"] == real["labels"][real_index], axis=1)]
        names = arrays["indicator_names"]
    with np.load(generated / "generated_statistics.npz") as arrays:
        fake_statistics = arrays["indicators"][arrays["valid"] & np.all(arrays["labels"] == real["labels"][real_index], axis=1)]
    median_real = np.nanmedian(real_statistics, axis=0)
    median_fake = np.nanmedian(fake_statistics, axis=0)
    scale = np.maximum(np.abs(median_real), 1e-6)
    displayed = [3, 5, 7, 8]
    axes[1, 2].plot(np.arange(len(displayed)), (median_real / scale)[displayed], marker="o", label="Real")
    axes[1, 2].plot(np.arange(len(displayed)), (median_fake / scale)[displayed], marker="o", label="Generated")
    axes[1, 2].set_xticks(np.arange(len(displayed)), [names[index].replace("_", " ") for index in displayed], rotation=30, fontsize=7)
    axes[1, 2].set_ylabel("Relative to real median")
    for axis, title, horizontal in zip(axes.flat,
            ("Waveform", "FFT", "Amplitude distribution", "Power spectrum", "Envelope spectrum", "Time-domain indicators"),
            ("Time / s", "Frequency / Hz", "Normalized amplitude", "Frequency / Hz", "Frequency / Hz", "")):
        axis.set(title=title, xlabel=horizontal)
    axes[0, 0].legend()
    figure.tight_layout()
    save_figure(figure, folder / "signal_comparison.pdf")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    transforms = []
    for values, index, title in ((real, real_index, "Real"), (fake, fake_index, "Generated")):
        sample_rate = 1 / np.diff(values["time_seconds"][:2])[0]
        frequency, time, transform = stft(values["normalized"][index, 0], fs=sample_rate, nperseg=128, noverlap=96)
        transforms.append((frequency, time, transform, title))
        np.savez_compressed(folder / (title.lower() + "_time_frequency.npz"), frequency_hz=frequency,
                            time_seconds=time, complex_stft=transform)
    upper = max(float((20 * np.log10(np.abs(transform) + 1e-12)).max()) for frequency, time, transform, title in transforms)
    for axis, (frequency, time, transform, title) in zip(axes, transforms):
        graphic = axis.pcolormesh(time, frequency, 20 * np.log10(np.abs(transform) + 1e-12),
                                 vmin=upper - 80, vmax=upper, shading="auto", rasterized=True)
        axis.set(title=title, xlabel="Time / s", ylabel="Frequency / Hz")
    figure.colorbar(graphic, ax=list(axes), label="Magnitude / dB", shrink=0.8)
    save_figure(figure, folder / "time_frequency.pdf")


def embedding_plot(artifacts, reference_id, entries, destination):
    vectors, roles, groups = [], [], []
    reference = artifacts / "references" / reference_id
    for name in ("source", "target"):
        with np.load(reference / (name + "_features.npz")) as arrays:
            vectors.append(arrays["projected"])
            roles.extend([name] * len(arrays["projected"]))
            groups.extend(arrays["groups"].tolist())
    for entry in entries:
        path = artifacts / entry["generated_data"] / "generated_features.npz" if entry["generated_data"] else None
        if path is None or not path.exists():
            continue
        with np.load(path) as arrays:
            vectors.append(arrays["projected"])
            roles.extend([entry["job"]["method"] + ":" + str(entry["job"].get("arm"))] * len(arrays["projected"]))
            groups.extend(arrays["groups"].tolist())
    values = np.concatenate(vectors)
    try:
        from umap import UMAP

        coordinates = UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(values)
        method = "UMAP"
    except ImportError:
        coordinates = PCA(n_components=2, random_state=42).fit_transform(values)
        method = "PCA"
    roles = np.asarray(roles)
    figure, axis = plt.subplots(figsize=(7, 4.7))
    for role in sorted(set(roles)):
        indices = roles == role
        axis.scatter(*coordinates[indices].T, s=3, alpha=0.5, label=role, rasterized=True)
    axis.set(xlabel=method + " 1", ylabel=method + " 2", title="One common embedding")
    axis.legend(markerscale=3, fontsize=6)
    save_figure(figure, destination / "joint_embedding.pdf")
    np.savez_compressed(destination / "joint_embedding.npz", coordinates=coordinates, roles=roles,
                        groups=np.asarray(groups), method=np.asarray(method))


def render(artifacts, run):
    manifest = read_json(artifacts / "manifest.json")
    selected = {}
    by_reference = defaultdict(list)
    for entry in manifest["jobs"]:
        job = entry["job"]
        key = job["track"], job["device"], job["method"], job.get("arm")
        rank = job["split"], job["channel"], job["target"]
        if key not in selected or rank < selected[key][0]:
            selected[key] = rank, entry
        by_reference[entry["reference"]].append(entry)
    plots = []
    rendered_references = set()
    for rank, entry in selected.values():
        job = entry["job"]
        folder = artifacts / "figures" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        generated = artifacts / entry["generated_data"] if entry["generated_data"] else None
        reference = artifacts / "references" / entry["reference"]
        diagnostic_curves(run, job, folder, read_json(reference / "reference.json")["components"])
        if generated is not None and (generated / "generated_signals.npz").exists():
            qualitative(reference, generated, folder)
        if entry["reference"] not in rendered_references:
            embedding_plot(artifacts, entry["reference"], by_reference[entry["reference"]], folder)
            rendered_references.add(entry["reference"])
            with np.load(reference / "source_statistics.npz") as arrays:
                statistics = arrays["indicators"][arrays["valid"]]
                names = arrays["indicator_names"]
            with np.errstate(invalid="ignore", divide="ignore"):
                correlations = np.corrcoef(statistics, rowvar=False)
            np.savez_compressed(folder / "indicator_correlation.npz", correlation=correlations, indicator_names=names)
            figure, axis = plt.subplots(figsize=(6, 5.4))
            graphic = axis.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=6)
            axis.set_yticks(np.arange(len(names)), names, fontsize=6)
            figure.colorbar(graphic, ax=axis)
            save_figure(figure, folder / "indicator_correlation.pdf")
        plots.append({"job": job, "reference": entry["reference"], "folder": folder.relative_to(artifacts).as_posix()})
    write_json(artifacts / "figure_index.json", {"selection": "Lowest seed, channel and condition per task/device/method/training group; all cases retain data.",
               "figures": plots, "code_sha256": file_hash(Path(__file__))})
    return plots


def main():
    parser = argparse.ArgumentParser(description="Redraw representative paper figures from saved arrays, without GPU inference.")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    print(f"Rendered {len(render(arguments.artifacts, arguments.run))} representative cases")


if __name__ == "__main__":
    main()
