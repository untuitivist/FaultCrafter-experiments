from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn

from faultcrafter.shared_gradient import JOINT_GRADIENT, WITHIN_DEVICE_CONDITION_PCGRAD, apply_gradient_correction, project_weighted_task_gradients, shared_gradient_policy


def condition_gradient_policy(generator: dict) -> str:
    policy = generator.get("condition_gradient_policy", JOINT_GRADIENT)
    if policy not in (JOINT_GRADIENT, WITHIN_DEVICE_CONDITION_PCGRAD):
        raise ValueError("unknown source condition gradient policy")
    if policy != JOINT_GRADIENT and shared_gradient_policy(generator) != JOINT_GRADIENT:
        raise ValueError("within-device condition projection cannot combine cross-device projection")
    return policy


def condition_gradient_metadata() -> dict:
    return {"policy": WITHIN_DEVICE_CONDITION_PCGRAD, "tasks": "source_record_condition_labels_within_each_device",
            "projection_scope": "all_existing_trainable_parameters_including_device_lora_and_fine_coarse_identity",
            "cross_device_projection": False, "original_global_sample_and_eligible_order_denominators": True,
            "regularizer": "original_condition_slope_once_outside_projection",
            "reference_gradients": "original_not_already_projected", "inner_product_dtype": "float64",
            "reference_order": "sorted_condition_labels_local_python_rng_same_global_seed_and_step_for_each_device",
            "model_forward_and_source_loss_changed": False, "optimizer_updates_equal_backward_pass_count_claimed": False,
            "extra_runtime_parameters_or_files": False}


@dataclass(frozen=True)
class SourceConditionRegistry:
    labels: dict[tuple[int, int], str]
    device_ids: tuple[str, ...]

    @classmethod
    def from_source_cells(cls, devices: list[Any], cells: tuple[tuple[int, int], ...]):
        if not cells or [device.index for device in devices] != list(range(len(devices))):
            raise ValueError("source condition registry requires eligible source cells and contiguous device routes")
        labels = {}
        for device_index, record_index in cells:
            if not 0 <= device_index < len(devices) or not 0 <= record_index < len(devices[device_index].corpus.records):
                raise ValueError("source condition registry cell is out of range")
            key = (int(device_index), int(record_index))
            if key in labels:
                raise ValueError("source condition registry contains duplicate cells")
            label = devices[device_index].corpus.records[record_index].record.condition_label
            if not isinstance(label, str) or not label:
                raise ValueError("source condition labels must be nonempty strings")
            labels[key] = label
        return cls(labels, tuple(device.device_id for device in devices))

    def partition(self, batch):
        if batch.source_record_index is None or len(batch.source_record_index) != len(batch.device_id):
            raise ValueError("condition projection requires actual source record indices")
        routes = []
        for device, record in zip(batch.device_id, batch.source_record_index, strict=True):
            key = (int(device), int(record))
            if key not in self.labels:
                raise ValueError("condition projection received a cell outside registered source training")
            routes.append((key[0], self.labels[key]))
        return ConditionObjectivePartition(tuple(routes))

    def audit(self) -> dict:
        return {"source_training_metadata_only": True, "target_waveforms_or_target_classifier_labels_used": False,
                "source_condition_counts": {name: len({label for (device, _), label in self.labels.items() if device == index}) for index, name in enumerate(self.device_ids)},
                "cells": [{"device_index": device, "device_id": self.device_ids[device], "source_record_index": record, "condition_label": label}
                          for (device, record), label in sorted(self.labels.items())], "extra_runtime_parameters_or_files": False}


@dataclass
class ConditionObjectivePartition:
    routes: tuple[tuple[int, str], ...]
    task_losses: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    regularizer: torch.Tensor | None = None
    reconstruction_error: float = 0.0
    sample_count: int = 0
    eligible_order_sample_count: int = 0

    def collect(self, device_ids: torch.Tensor, sample_losses: torch.Tensor, order_samples: list[tuple[torch.Tensor, torch.Tensor]],
                *, eligible_order_samples: int, order_weight: float, regularizer: torch.Tensor, objective: torch.Tensor) -> None:
        if self.task_losses or self.regularizer is not None:
            raise ValueError("a condition partition may only collect one batch")
        if not self.routes or len(self.routes) != len(device_ids) or device_ids.ndim != 1 or device_ids.dtype != torch.long or sample_losses.shape != device_ids.shape:
            raise ValueError("condition partition requires one registered route and loss per sample")
        if any(type(device) is not int or device < 0 or not isinstance(label, str) or not label for device, label in self.routes):
            raise ValueError("condition partition routes are invalid")
        if [device for device, _ in self.routes] != device_ids.tolist() or sample_losses.device != device_ids.device or not sample_losses.is_floating_point():
            raise ValueError("condition partition device routing disagrees with actual batch")
        if type(eligible_order_samples) is not int or not 0 <= eligible_order_samples <= len(device_ids) or not math.isfinite(order_weight) or order_weight < 0:
            raise ValueError("condition order denominator or weight is invalid")
        visited = []
        for indices, losses in order_samples:
            if indices.ndim != 1 or indices.dtype != torch.long or losses.shape != indices.shape or indices.device != device_ids.device or losses.device != device_ids.device:
                raise ValueError("condition order samples require matching per-sample indices and losses")
            visited.extend(indices.tolist())
        if len(visited) != eligible_order_samples or len(set(visited)) != len(visited) or any(not 0 <= value < len(device_ids) for value in visited):
            raise ValueError("condition order samples are missing, duplicated or out of range")
        self.sample_count = len(device_ids)
        self.eligible_order_sample_count = eligible_order_samples
        for route in sorted(set(self.routes)):
            mask = torch.tensor([current == route for current in self.routes], device=device_ids.device, dtype=torch.bool)
            contribution = sample_losses[mask].sum() / self.sample_count
            for indices, losses in order_samples:
                selected = mask[indices]
                if torch.any(selected):
                    contribution = contribution + order_weight * losses[selected].sum() / eligible_order_samples
            self.task_losses[route] = contribution
        self.regularizer = regularizer
        reconstructed = torch.stack([*self.task_losses.values(), regularizer]).sum()
        if objective.numel() != 1 or reconstructed.numel() != 1 or not torch.isfinite(objective) or not torch.isfinite(reconstructed):
            raise ValueError("condition partition requires a finite scalar source objective")
        self.reconstruction_error = float((reconstructed.detach() - objective.detach()).abs())
        tolerance = 64 * torch.finfo(objective.dtype).eps * max(1.0, float(objective.detach().abs()), float(reconstructed.detach().abs()))
        if self.reconstruction_error > tolerance:
            raise ValueError("condition contributions changed the original weighted source objective")


def backward_with_condition_projection(model: nn.Module, objective: torch.Tensor, partition: ConditionObjectivePartition, *, seed: int, step: int) -> dict:
    if not partition.task_losses or partition.regularizer is None:
        raise ValueError("condition projection requires the populated actual source partition")
    if type(seed) is not int or seed < 0 or type(step) is not int or step <= 0:
        raise ValueError("condition projection requires a valid global training seed and step")
    named = tuple(model.named_parameters())
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("condition projection requires all existing model parameters trainable")
    parameters = tuple(parameter for _, parameter in named)
    corrections = []
    device_reports = []
    backward_calls = 0
    counts = Counter(partition.routes)
    for device in sorted({route[0] for route in partition.task_losses}):
        tasks = sorted(route for route in partition.task_losses if route[0] == device)
        report = {"device_index": device, "source_condition_count": len(tasks),
                  "condition_sample_counts": {route[1]: counts[route] for route in tasks},
                  "original_negative_pairs": 0, "projection_count": 0, "correction_norm": 0.0}
        if len(tasks) >= 2:
            flattened = []
            for route in tasks:
                gradients = torch.autograd.grad(partition.task_losses[route], parameters, retain_graph=True, allow_unused=True)
                flattened.append(torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1)
                                            for parameter, gradient in zip(parameters, gradients, strict=True)]))
                backward_calls += 1
            correction, projected = project_weighted_task_gradients(torch.stack(flattened), seed=seed, step=step)
            report.update({"original_negative_pairs": projected["shared_gradient_original_negative_pairs"],
                           "projection_count": projected["shared_gradient_projection_count"],
                           "correction_norm": projected["shared_gradient_correction_norm"]})
            if report["projection_count"]:
                corrections.append(correction)
        device_reports.append(report)
    objective.backward()
    correction_norm = 0.0
    if corrections:
        combined = torch.stack(corrections).sum(dim=0)
        correction_norm = float(combined.to(torch.float64).norm())
        apply_gradient_correction(parameters, combined)
    return {"condition_gradient_policy_executed": 1.0, "condition_gradient_partition_error": partition.reconstruction_error,
            "condition_gradient_sample_count": partition.sample_count, "condition_gradient_order_sample_count": partition.eligible_order_sample_count,
            "condition_gradient_task_count": len(partition.task_losses), "condition_gradient_task_backward_calls": backward_calls,
            "condition_gradient_multiple_condition_devices": sum(row["source_condition_count"] >= 2 for row in device_reports),
            "condition_gradient_original_negative_pairs": sum(row["original_negative_pairs"] for row in device_reports),
            "condition_gradient_projection_count": sum(row["projection_count"] for row in device_reports),
            "condition_gradient_correction_norm": correction_norm, "condition_gradient_devices": device_reports}
