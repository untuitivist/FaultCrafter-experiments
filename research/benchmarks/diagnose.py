from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from faultcrafter.models import CNN1D

from .common import file_hash, timestamp, write_json
from .quality import features


def diagnose(checkpoint_path, signals_path, output, *, device="cpu", batch_size=128):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    classes = state["classifier.6.weight"].shape[0]
    model = CNN1D(classes).to(device).eval()
    model.load_state_dict(state)
    with np.load(signals_path) as arrays:
        signals = np.array(arrays["normalized"])
        labels = np.array(arrays["labels"]) if "labels" in arrays else None
        valid = np.array(arrays["valid"]) if "valid" in arrays else np.ones(len(signals), dtype=bool)
    if signals.ndim != 3 or signals.shape[1] != 1 or not np.isfinite(signals).all():
        raise ValueError("Expected finite parent-normalized signals with shape [samples,1,length]")
    logits = []
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            logits.append(model(torch.as_tensor(signals[start:start + batch_size], device=device)).cpu().numpy())
    logits = np.concatenate(logits)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    output.mkdir(parents=True, exist_ok=True)
    payload = {"logits": logits, "probabilities": probabilities, "decisions": probabilities >= 0.5,
               "valid": valid, **features(model, signals)}
    if labels is not None:
        payload["labels"] = labels
    np.savez_compressed(output / "diagnostics.npz", **payload)
    write_json(output / "inference.json", {"created": timestamp(), "checkpoint_sha256": file_hash(checkpoint_path),
               "input_sha256": file_hash(signals_path), "device": device, "threshold": 0.5, "head": "Independent sigmoid fault components",
               "code_sha256": file_hash(Path(__file__))})
    return output / "diagnostics.npz"


def main():
    parser = argparse.ArgumentParser(description="Run a final diagnostic CNN on saved normalized waveforms.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    print(diagnose(arguments.checkpoint, arguments.signals, arguments.output, device=arguments.device))


if __name__ == "__main__":
    main()
