from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from faultcrafter.experiment0 import canonical_digest


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_convergence(history: list[dict[str, Any]], generator: dict[str, Any]) -> dict[str, Any]:
    reference = float("inf")
    stale = 0
    stopped = False
    preheat = int(generator["checkpoint_preheat_evaluations"])
    patience = int(generator["early_stopping_evaluations"])
    threshold = float(generator["relative_validation_improvement"])
    interval = int(generator["evaluation_interval"])
    if min(preheat, patience, interval) <= 0 or threshold <= 0:
        raise ValueError("source convergence needs positive preheat, patience, interval and improvement")
    for evaluation, row in enumerate(history, start=1):
        if stopped:
            raise ValueError("training continued after the source convergence boundary")
        if row["evaluation"] != evaluation or row.get("phase_step", row["step"]) != evaluation * interval:
            raise ValueError("source convergence history is incomplete")
        validation = float(row["validation_objective"])
        if not np.isfinite(validation):
            raise ValueError("source convergence requires finite validation objectives")
        if evaluation >= preheat:
            improvement = (reference - validation) / max(abs(reference), 1e-12) if np.isfinite(reference) else float("inf")
            if improvement >= threshold:
                reference, stale = validation, 0
            else:
                stale += 1
            stopped = stale >= patience
    return {
        "converged": stopped,
        "evaluation_count": len(history),
        "stale_evaluations": stale,
        "patience_reference": reference,
        "termination": "source_relative_improvement" if stopped else "not_converged",
    }


class TrainingRecovery:
    def __init__(
        self, path: Path, config: dict[str, Any], metadata: dict[str, Any] | None,
        model: torch.nn.Module, ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, scheduler: Any,
        selection_rng: np.random.Generator, torch_random: torch.Generator,
        samplers: dict[str, Any],
    ) -> None:
        self.path = path
        self.identity = canonical_digest({"config": config, "metadata": metadata})
        self.model, self.ema_model = model, ema_model
        self.optimizer, self.scheduler = optimizer, scheduler
        self.selection_rng, self.torch_random = selection_rng, torch_random
        self.samplers = samplers

    def save(self, progress: dict[str, Any], checkpoint_path: Path) -> None:
        numpy_state = np.random.get_state()
        payload = {
            "format": "faultcrafter_training_recovery_v1",
            "identity": self.identity,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "progress": progress,
            "samplers": {name: copy.deepcopy(vars(sampler)) for name, sampler in self.samplers.items()},
            "selection_rng": self.selection_rng.bit_generator.state,
            "torch_random": self.torch_random.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "selected_checkpoint": torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.is_file() else None,
        }
        atomic_torch_save(payload, self.path)

    def restore(self, checkpoint_path: Path) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("format") != "faultcrafter_training_recovery_v1" or payload["identity"] != self.identity:
            raise ValueError("training recovery belongs to different frozen inputs")
        if payload["samplers"].keys() != self.samplers.keys():
            raise ValueError("training recovery sampler set differs")
        for name, sampler in self.samplers.items():
            saved = payload["samplers"][name]
            current = vars(sampler)
            if saved.keys() != current.keys() or any(saved[key] != current[key] for key in current if key not in {"cursor", "device_cursors", "exposure"}):
                raise ValueError("training recovery sampler identity differs")
            current.update(copy.deepcopy(saved))
        self.model.load_state_dict(payload["model"], strict=True)
        self.ema_model.load_state_dict(payload["ema_model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.selection_rng.bit_generator.state = payload["selection_rng"]
        self.torch_random.set_state(payload["torch_random"])
        random.setstate(payload["python_rng"])
        numpy_state = payload["numpy_rng"]
        np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            if not torch.cuda.is_available() or len(payload["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("training recovery CUDA topology differs")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        selected = payload["selected_checkpoint"]
        if selected is not None:
            atomic_torch_save(selected, checkpoint_path)
        print(f"training_recovery_restored={self.path} completed_step={payload['progress']['completed_step']}", flush=True)
        return payload["progress"]
