from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch


SORTED_WAVEFORM_L1 = "sorted_waveform_l1_v1"


def waveform_distribution_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("waveform_distribution_policy", "none")
    if policy not in ("none", SORTED_WAVEFORM_L1):
        raise ValueError("unknown waveform distribution reconstruction policy")
    return policy


def waveform_distribution_metadata(generator: Mapping[str, Any]) -> dict:
    if waveform_distribution_policy(generator) != SORTED_WAVEFORM_L1:
        raise ValueError("waveform distribution metadata requires its active policy")
    weight = generator.get("max_abs_loss_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("waveform distribution requires the existing positive finite peak-loss weight")
    if any(float(generator.get(name, 0.0)) != 0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")):
        raise ValueError("waveform distribution does not combine source teachers or contrast")
    if any(generator.get(name, False) for name in ("resumable_training", "device_balanced_sampling", "relation_balanced_sampling", "relation_coupled_noise_time")):
        raise ValueError("waveform distribution retains original finite source sampling")
    return {"policy": SORTED_WAVEFORM_L1, "weight": float(weight), "weight_source": "existing_max_abs_loss_weight",
            "formula": "mean_samples(mean_time(abs(stable_sort(prediction)-stable_sort(clean))))",
            "scope": "guided_fine_parent_waveform", "source_target_detached": True,
            "source_validation": "unchanged_original_objective", "additional_model_forwards": 0,
            "additional_source_samples": 0, "extra_runtime_parameters_or_files": False}


def sorted_waveform_l1(prediction: torch.Tensor, clean: torch.Tensor, routes: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if prediction.shape != clean.shape or prediction.ndim != 3 or prediction.shape[1] != 1 or not len(prediction) or not prediction.shape[-1]:
        raise ValueError("waveform distribution requires equal nonempty [batch,1,length] signals")
    if prediction.dtype not in (torch.float32, torch.float64) or prediction.dtype != clean.dtype or prediction.device != clean.device:
        raise ValueError("waveform distribution requires shared-device float32/64 signals")
    if routes.shape != (len(prediction),) or routes.dtype != torch.long or torch.any(routes < 0):
        raise ValueError("waveform distribution requires one nonnegative int64 route per source parent")
    if not torch.isfinite(prediction).all() or not torch.isfinite(clean).all():
        raise ValueError("waveform distribution requires finite waveforms")
    predicted = torch.sort(prediction, dim=-1, stable=True).values
    target = torch.sort(clean.detach(), dim=-1, stable=True).values
    errors = (predicted - target).abs().mean(dim=(1, 2))
    loss = errors.mean()
    if not torch.isfinite(loss):
        raise ValueError("waveform distribution reconstruction overflowed")
    diagnostics = {"waveform_distribution_executed": 1.0, "waveform_distribution_loss": float(loss.detach()),
                   "waveform_distribution_sample_losses": errors.detach().cpu().tolist(),
                   "waveform_distribution_samples": len(prediction), "waveform_distribution_elements": prediction.numel(),
                   "waveform_distribution_length": prediction.shape[-1], "waveform_distribution_dtype": str(prediction.dtype),
                   "waveform_distribution_routes": routes.detach().cpu().tolist(),
                   "waveform_distribution_source_sorted_sha256": hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
                   "waveform_distribution_prediction_sorted_sha256": hashlib.sha256(predicted.detach().cpu().numpy().tobytes()).hexdigest()}
    return loss, diagnostics


def validate_distribution_row(row: dict) -> None:
    prefix = "waveform_distribution_"
    if row.get(prefix + "executed") != 1.0:
        raise ValueError("waveform distribution objective was not executed")
    count, length, elements = (row.get(prefix + field) for field in ("samples", "length", "elements"))
    if any(type(value) is not int or value <= 0 for value in (count, length, elements)) or elements != count * length:
        raise ValueError("waveform distribution source sample or element counts differ")
    routes, values = row.get(prefix + "routes", []), row.get(prefix + "sample_losses", [])
    if len(routes) != count or len(values) != count or any(type(route) is not int or route < 0 for route in routes):
        raise ValueError("waveform distribution routes or sample losses differ")
    loss = row.get(prefix + "loss", float("nan"))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in [loss, *values]):
        raise ValueError("waveform distribution requires finite nonnegative errors")
    dtype = row.get(prefix + "dtype")
    if dtype not in ("torch.float32", "torch.float64"):
        raise ValueError("waveform distribution trace has unsupported precision")
    epsilon = np.finfo(np.float32 if dtype == "torch.float32" else np.float64).eps
    if abs(loss - math.fsum(values) / count) > 8 * epsilon * max(1.0, max(values)):
        raise ValueError("waveform distribution batch loss differs from saved per-parent errors")
    for field in ("source_sorted_sha256", "prediction_sorted_sha256"):
        value = row.get(prefix + field)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("waveform distribution trace requires complete sorted-waveform hashes")


class WaveformDistributionActivity:
    def __init__(self, generator: Mapping[str, Any]):
        self.metadata = waveform_distribution_metadata(generator)
        self.rows = []
        self.totals = {"samples": 0, "elements": 0, "nonzero_steps": 0, "loss_sum": 0.0}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("waveform distribution requires every consecutive completed update")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items() if name.startswith("waveform_distribution_")})}
        validate_distribution_row(row)
        self.rows.append(row)
        for field in ("samples", "elements"):
            self.totals[field] += row["waveform_distribution_" + field]
        self.totals["nonzero_steps"] += int(row["waveform_distribution_loss"] > 0)
        self.totals["loss_sum"] += row["waveform_distribution_loss"]
        return {"waveform_distribution_total_steps": step, **{"waveform_distribution_total_" + name: value for name, value in self.totals.items()}}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_predictions_or_student_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}
