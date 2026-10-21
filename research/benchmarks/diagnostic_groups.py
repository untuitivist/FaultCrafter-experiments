from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .classifier import classification_metrics
from .common import file_hash, read_json, timestamp, write_json
from .data import T4_STATES
from .report import repeat_summaries, write_csv


def summarize(logits, labels, valid):
    scores = classification_metrics(torch.from_numpy(logits), torch.from_numpy(labels), valid)
    confusion = np.asarray(scores["confusion_valid"])
    true_negative, false_positive, false_negative, true_positive = confusion.sum(axis=0)
    scores["false_positive_rate"] = float(false_positive / (false_positive + true_negative)) if false_positive + true_negative else None
    scores["false_negative_rate"] = float(false_negative / (false_negative + true_positive)) if false_negative + true_positive else None
    return scores


def evaluate(run):
    completed = []
    for result_path in sorted((run / "jobs").glob("*/result.json")):
        result = read_json(result_path)
        job = result["job"]
        if job["track"] != "T4":
            continue
        folder = result_path.parent
        manifest = read_json(folder / "target_manifest.json")
        states = np.asarray(manifest["states"])
        with np.load(folder / "target_predictions.npz") as arrays:
            logits, labels, valid = arrays["logits"], arrays["labels"], arrays["valid"]
        parts = (labels >= 0.5).sum(axis=1)
        unseen = np.isin(states, T4_STATES[job["device"]])
        masks = {"normal": parts == 0, "single_fault": parts == 1,
                 "seen_composite": (parts > 1) & ~unseen, "unseen_composite": unseen}
        if not np.all(np.stack(list(masks.values())).sum(axis=0) == 1):
            raise RuntimeError("T4 diagnostic groups do not form a disjoint evaluation partition")
        groups = {name: summarize(logits[mask], labels[mask], valid[mask]) if mask.any() else None
                  for name, mask in masks.items()}
        overall = summarize(logits, labels, valid)
        if overall["correct"] != result["target"]["correct"] or overall["total"] != result["target"]["total"]:
            raise RuntimeError("Saved T4 predictions differ from the diagnostic result")
        entry = {"job": job, "groups": groups, "overall": overall,
                 "checkpoint_sha256": result["best_checkpoint_sha256"],
                 "prediction_sha256": file_hash(folder / "target_predictions.npz"),
                 "rate_convention": "False-positive and false-negative rates pool component-level valid confusion counts. Exact-set accuracy includes unpredicted samples as errors.",
                 "created": timestamp(), "code_sha256": file_hash(Path(__file__))}
        write_json(run / "diagnostic_groups" / (job["id"] + ".json"), entry)
        completed.append(entry)
    write_json(run / "diagnostic_groups.json", {"completed": len(completed), "records": completed})
    flat = []
    for entry in completed:
        job = entry["job"]
        row = {key: job[key] for key in ("track", "device", "channel", "method", "arm", "target", "split")}
        row.update({name + "_acc": scores["exact_set_accuracy"] if scores else None for name, scores in entry["groups"].items()})
        row.update({name: entry["overall"][name] for name in ("false_positive_rate", "false_negative_rate")})
        flat.append(row)
    metrics = ("normal_acc", "single_fault_acc", "seen_composite_acc", "unseen_composite_acc", "false_positive_rate", "false_negative_rate")
    summaries = repeat_summaries(flat, ("track", "device", "channel", "method", "arm", "target"), metrics)
    write_json(run / "diagnostic_group_summary.json", {"standard_deviation": "sample standard deviation, ddof=1", "records": summaries})
    write_csv(run / "diagnostic_group_summary.csv", summaries)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    records = evaluate(arguments.run)
    print(f"T4 diagnostic groups: {len(records)} evaluated CNNs; all predictions reused.")


if __name__ == "__main__":
    main()
