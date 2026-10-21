from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
import torch


ORIGINAL_ORDER = "unweighted_order_v1"
RESOLUTION_AWARE_ORDER = "rotation_grid_reliability_v1"


def order_resolution_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("order_resolution_policy", ORIGINAL_ORDER)
    if policy not in (ORIGINAL_ORDER, RESOLUTION_AWARE_ORDER):
        raise ValueError("unknown order resolution policy")
    return policy


def order_resolution_metadata(generator: Mapping[str, Any]) -> dict:
    maximum = generator["maximum_order"]
    count = generator["order_feature_count"]
    minimum = generator["minimum_order_rotations"]
    weight = generator["order_loss_weight"]
    if order_resolution_policy(generator) != RESOLUTION_AWARE_ORDER:
        raise ValueError("order resolution metadata requires the active policy")
    if type(count) is not int or count < 2 or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (maximum, minimum, weight)):
        raise ValueError("order resolution settings must be positive and finite")
    if generator.get("physics_reference_policy") != "known_mechanical_only":
        raise ValueError("order resolution requires explicit mechanical reference masks")
    spacing = float(maximum) / (count - 1)
    return {"policy": RESOLUTION_AWARE_ORDER, "maximum_order": float(maximum), "order_feature_count": count,
            "order_grid_spacing": spacing, "minimum_order_rotations": float(minimum), "order_loss_weight": float(weight),
            "weight_formula": "min(1,window_rotations*order_grid_spacing)",
            "reduction": "sum_weighted_errors_divided_by_original_eligible_sample_count",
            "feature_and_eligibility_rules_unchanged": True, "train_and_source_validation_same_objective": True,
            "actual_hann_peak_resolvability_claimed": False, "statistical_confidence_claimed": False,
            "additional_model_forwards_per_update": 0, "cnn_non_degradation_guaranteed": False,
            "extra_runtime_parameters_or_files": False}


def rotation_resolution_weights(rotations: torch.Tensor, spacing: float) -> torch.Tensor:
    if rotations.ndim != 1 or rotations.dtype not in (torch.float32, torch.float64):
        raise ValueError("rotations must be a one-dimensional float32/float64 tensor")
    if not math.isfinite(spacing) or spacing <= 0 or not torch.isfinite(rotations).all() or torch.any(rotations < 0):
        raise ValueError("rotation weights require finite nonnegative rotations and positive spacing")
    return (rotations.detach() * spacing).clamp(max=1.0)


class OrderResolutionBatch:
    def __init__(self, generator: Mapping[str, Any], length: int, batch_size: int):
        self.metadata = order_resolution_metadata(generator)
        if type(length) is not int or length < 4 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("order resolution requires a nonempty signal batch")
        self.length = length
        self.batch_size = batch_size
        self.samples = []
        self.original_terms = []

    def weight(self, errors: torch.Tensor, rotations: np.ndarray, indices: np.ndarray, device_index: int) -> torch.Tensor:
        if errors.ndim != 1 or errors.shape != rotations.shape or errors.shape != indices.shape or not len(errors) or not torch.isfinite(errors).all():
            raise ValueError("order resolution requires matching nonempty finite sample vectors")
        if np.any(rotations < self.metadata["minimum_order_rotations"]) or np.any(indices < 0) or np.any(indices >= self.batch_size):
            raise ValueError("order resolution received an ineligible sample")
        rotation_tensor = torch.as_tensor(rotations, dtype=errors.dtype, device=errors.device)
        weights = rotation_resolution_weights(rotation_tensor, self.metadata["order_grid_spacing"])
        weighted = errors * weights
        self.original_terms.append(errors.detach().sum())
        for index, rotation, weight, original, adjusted in zip(indices.tolist(), rotation_tensor.detach().cpu().tolist(), weights.cpu().tolist(), errors.detach().cpu().tolist(), weighted.detach().cpu().tolist(), strict=True):
            self.samples.append({"sample_index": int(index), "device_index": int(device_index), "rotations": rotation,
                                 "weight": weight, "original_error": original, "weighted_error": adjusted})
        return weighted

    def diagnostics(self, order: torch.Tensor) -> dict:
        count = len(self.samples)
        original = torch.stack(self.original_terms).sum() / count if count else order.detach() * 0.0
        return {"order_resolution_executed": 1.0, "order_resolution_length": self.length,
                "order_resolution_batch_size": self.batch_size, "order_resolution_eligible_samples": count,
                "order_resolution_attenuated_samples": sum(sample["weight"] < 1.0 for sample in self.samples),
                "order_resolution_weight_sum": math.fsum(sample["weight"] for sample in self.samples),
                "order_resolution_original_order": float(original), "order_resolution_weighted_order": float(order.detach()),
                "order_resolution_samples": copy.deepcopy(self.samples)}


class OrderResolutionActivity:
    def __init__(self, generator: Mapping[str, Any]):
        self.metadata = order_resolution_metadata(generator)
        self.rows = []
        self.eligible = 0
        self.attenuated = 0
        self.weight_sum = 0.0

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1 or components.get("order_resolution_executed") != 1.0:
            raise ValueError("order resolution requires every consecutive completed update")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items() if name.startswith("order_resolution_")})}
        validate_order_resolution_row(row, self.metadata)
        self.rows.append(row)
        self.eligible += row["order_resolution_eligible_samples"]
        self.attenuated += row["order_resolution_attenuated_samples"]
        self.weight_sum += row["order_resolution_weight_sum"]
        return {"order_resolution_total_steps": step, "order_resolution_total_eligible_samples": self.eligible,
                "order_resolution_total_attenuated_samples": self.attenuated, "order_resolution_total_weight_sum": self.weight_sum}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_student_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}


def validate_order_resolution_row(row: dict, metadata: dict) -> None:
    samples = row["order_resolution_samples"]
    count = row["order_resolution_eligible_samples"]
    batch_size = row["order_resolution_batch_size"]
    if type(count) is not int or type(batch_size) is not int or not 0 <= count <= batch_size or len(samples) != count:
        raise ValueError("order resolution sample count differs from its batch")
    if len({sample["sample_index"] for sample in samples}) != count:
        raise ValueError("order resolution samples must be unique")
    for sample in samples:
        if type(sample["sample_index"]) is not int or not 0 <= sample["sample_index"] < batch_size or type(sample["device_index"]) is not int or sample["device_index"] < 0:
            raise ValueError("order resolution sample routing is invalid")
        if any(not math.isfinite(sample[name]) for name in ("rotations", "weight", "original_error", "weighted_error")):
            raise ValueError("order resolution trace contains a nonfinite value")
        if sample["rotations"] < metadata["minimum_order_rotations"]:
            raise ValueError("order resolution trace contains an ineligible rotation count")
        expected = min(1.0, sample["rotations"] * metadata["order_grid_spacing"])
        if not math.isclose(sample["weight"], expected, rel_tol=2e-7, abs_tol=2e-8) or not math.isclose(sample["weighted_error"], sample["weight"] * sample["original_error"], rel_tol=2e-7, abs_tol=2e-8):
            raise ValueError("order resolution trace does not reproduce its declared formula")
    if row["order_resolution_attenuated_samples"] != sum(sample["weight"] < 1.0 for sample in samples) or row["order_resolution_weight_sum"] != math.fsum(sample["weight"] for sample in samples):
        raise ValueError("order resolution weight statistics differ from individual samples")
    for field, value in (("original", "original_error"), ("weighted", "weighted_error")):
        expected = math.fsum(sample[value] for sample in samples) / count if count else 0.0
        if not math.isclose(row[f"order_resolution_{field}_order"], expected, rel_tol=3e-7, abs_tol=3e-8):
            raise ValueError("order resolution must divide by the original eligible sample count")
