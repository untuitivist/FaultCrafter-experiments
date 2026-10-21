from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SOURCE_SUPPORT_POLICY = "rank_deficiency_projection"
SPECTRAL_SOURCE_SUPPORT_POLICY = "source_spectral_weak_axis_projection"
SOURCE_SUPPORT_POLICIES = {
    DEFAULT_SOURCE_SUPPORT_POLICY,
    "all_axes_projection",
    SPECTRAL_SOURCE_SUPPORT_POLICY,
}


@dataclass(frozen=True, slots=True)
class ConditionGeometry:
    origin: np.ndarray
    basis: np.ndarray
    rank: np.ndarray
    transform_enabled: np.ndarray
    support_minimum: np.ndarray
    support_maximum: np.ndarray
    spectral_diagnostics: dict[str, object] | None = None


def infer_condition_geometry(
    normalized_parameter_sets: Sequence[np.ndarray],
    *,
    condition_dimension: int,
    requested_ranks: Sequence[int] | None = None,
    retained_variance: float | None = None,
) -> ConditionGeometry:
    """Infer a deterministic source-only coordinate basis for every device."""
    if condition_dimension <= 0:
        raise ValueError("condition dimension must be positive")
    adaptive = retained_variance is not None
    if adaptive and requested_ranks is not None:
        raise ValueError("source spectral rank cannot use manually requested ranks")
    if adaptive and not 0.0 < float(retained_variance) <= 1.0:
        raise ValueError("retained source variance must be in (0,1]")
    if requested_ranks is None:
        if not adaptive:
            raise ValueError("condition geometry requires a rank selection rule")
        requested_ranks = [condition_dimension] * len(normalized_parameter_sets)
    if len(normalized_parameter_sets) != len(requested_ranks):
        raise ValueError("condition parameter sets and requested ranks differ")
    device_count = len(normalized_parameter_sets)
    origin = np.zeros((device_count, condition_dimension), dtype=np.float32)
    basis = np.repeat(
        np.eye(condition_dimension, dtype=np.float32)[None],
        device_count,
        axis=0,
    )
    rank = np.zeros(device_count, dtype=np.int64)
    transform_enabled = np.zeros(device_count, dtype=np.bool_)
    support_minimum = np.zeros(
        (device_count, condition_dimension), dtype=np.float32
    )
    support_maximum = np.zeros(
        (device_count, condition_dimension), dtype=np.float32
    )
    variance_ratios = np.zeros((device_count, condition_dimension), dtype=np.float64)
    unique_source_conditions: list[list[list[float]]] = []

    for device_index, (values, requested) in enumerate(
        zip(normalized_parameter_sets, requested_ranks, strict=True)
    ):
        parameters = np.asarray(values, dtype=np.float64)
        if parameters.ndim != 2 or not 0 < parameters.shape[1] <= condition_dimension:
            raise ValueError("normalized source parameter matrix shape differs")
        if not len(parameters) or not np.isfinite(parameters).all():
            raise ValueError("normalized source parameters must be finite and non-empty")
        if not 0 <= int(requested) <= condition_dimension:
            raise ValueError("requested condition rank is outside the supported range")
        transform_enabled[device_index] = adaptive or int(requested) < parameters.shape[1]
        padded = np.zeros((len(parameters), condition_dimension), dtype=np.float64)
        padded[:, : parameters.shape[1]] = parameters
        unique = np.unique(padded, axis=0)
        unique_source_conditions.append(unique.tolist())
        center = unique.mean(axis=0)
        centered = unique - center
        origin[device_index] = center.astype(np.float32)
        if len(unique) < 2 or not np.any(centered):
            continue

        _, singular_values, right = np.linalg.svd(centered, full_matrices=True)
        tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
        numerical_rank = int(np.sum(singular_values > tolerance))
        effective_rank = min(int(requested), len(unique) - 1, numerical_rank)
        if adaptive:
            spectrum = np.square(singular_values)
            ratios = spectrum / spectrum.sum()
            variance_ratios[device_index, : len(ratios)] = ratios
            effective_rank = min(
                numerical_rank,
                int(np.searchsorted(np.cumsum(ratios), retained_variance)) + 1,
            )
        for row in range(effective_rank):
            vector = right[row]
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                right[row] *= -1.0
        basis[device_index] = right.astype(np.float32)
        rank[device_index] = effective_rank
        coordinates = centered @ right.T
        support_minimum[device_index] = coordinates.min(axis=0).astype(np.float32)
        support_maximum[device_index] = coordinates.max(axis=0).astype(np.float32)
    return ConditionGeometry(
        origin=origin,
        basis=basis,
        rank=rank,
        transform_enabled=transform_enabled,
        support_minimum=support_minimum,
        support_maximum=support_maximum,
        spectral_diagnostics=(
            {
                "selection_rule": "smallest_rank_reaching_source_variance_fraction",
                "retained_variance": float(retained_variance),
                "explained_variance_ratio": variance_ratios.tolist(),
                "unique_source_conditions": unique_source_conditions,
                "source_condition_counts": [len(values) for values in unique_source_conditions],
                "source_only": True,
                "target_parameters_used": False,
                "device_specific_rules": False,
            }
            if adaptive
            else None
        ),
    )


def _analysis_features(signal: torch.Tensor) -> torch.Tensor:
    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError("style transport analysis expects [B,1,T]")
    length = signal.shape[-1]
    features: list[torch.Tensor] = []
    for requested_kernel in (3, 9, 33, 129):
        kernel = min(requested_kernel, length if length % 2 else length - 1)
        kernel = max(kernel, 1)
        padding = kernel // 2
        smoothed = F.avg_pool1d(
            F.pad(signal, (padding, padding), mode="replicate"),
            kernel,
            stride=1,
        )
        detail = signal - smoothed
        envelope = F.avg_pool1d(
            F.pad(detail.square(), (padding, padding), mode="replicate"),
            kernel,
            stride=1,
        ).add(1e-6).sqrt()
        features.extend((detail, envelope))
    return torch.cat(features, dim=1)


def style_descriptor(signal: torch.Tensor) -> torch.Tensor:
    features = _analysis_features(signal)
    mean = features.mean(dim=-1)
    log_deviation = features.var(dim=-1, unbiased=False).add(1e-6).log() * 0.5
    return torch.cat((mean, log_deviation), dim=1)


def paired_relative_transport_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition_parameters: torch.Tensor,
    device_id: torch.Tensor,
    fault: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Separate fault residuals from source-visible condition-style slopes."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("paired transport signals must have matching [B,1,T] shape")
    batch = len(prediction)
    if condition_parameters.ndim != 2 or condition_parameters.shape[0] != batch:
        raise ValueError("paired transport conditions differ from signal batch")
    for name, value in (("device", device_id), ("fault", fault)):
        if value.shape != (batch,) or value.dtype != torch.long:
            raise ValueError(f"paired transport {name} must be long with shape [B]")

    predicted = style_descriptor(prediction)
    reference = style_descriptor(target).detach()
    style_terms: list[torch.Tensor] = []
    content_terms: list[torch.Tensor] = []
    slope_terms: list[torch.Tensor] = []
    condition_count = 0
    identifiable_pairs = 0

    for device in torch.unique(device_id, sorted=True):
        device_indices = torch.nonzero(device_id == device, as_tuple=False).flatten()
        device_parameters = condition_parameters.index_select(0, device_indices)
        unique_conditions, inverse = torch.unique(
            device_parameters,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        predicted_means: list[torch.Tensor] = []
        reference_means: list[torch.Tensor] = []
        retained_conditions: list[torch.Tensor] = []
        for condition_index in range(len(unique_conditions)):
            local = torch.nonzero(inverse == condition_index, as_tuple=False).flatten()
            indices = device_indices.index_select(0, local)
            if len(torch.unique(fault.index_select(0, indices))) < 2:
                continue
            predicted_group = predicted.index_select(0, indices)
            reference_group = reference.index_select(0, indices)
            predicted_mean = predicted_group.mean(dim=0)
            reference_mean = reference_group.mean(dim=0)
            style_terms.append(F.smooth_l1_loss(predicted_mean, reference_mean))
            content_terms.append(
                F.smooth_l1_loss(
                    predicted_group - predicted_mean,
                    reference_group - reference_mean,
                )
            )
            predicted_means.append(predicted_mean)
            reference_means.append(reference_mean)
            retained_conditions.append(unique_conditions[condition_index])
            condition_count += 1

        for left in range(len(retained_conditions)):
            for right_index in range(left + 1, len(retained_conditions)):
                distance = torch.linalg.vector_norm(
                    retained_conditions[right_index] - retained_conditions[left]
                ).clamp_min(1e-6)
                predicted_slope = (
                    predicted_means[right_index] - predicted_means[left]
                ) / distance
                reference_slope = (
                    reference_means[right_index] - reference_means[left]
                ) / distance
                slope_terms.append(
                    F.smooth_l1_loss(predicted_slope, reference_slope)
                )
                identifiable_pairs += 1

    zero = prediction.sum() * 0.0
    return {
        "condition_style": torch.stack(style_terms).mean() if style_terms else zero,
        "fault_content": torch.stack(content_terms).mean() if content_terms else zero,
        "transport_slope": torch.stack(slope_terms).mean() if slope_terms else zero,
        "difference_of_differences": zero,
        "fault_separation_margin": zero,
        "condition_count": prediction.new_tensor(float(condition_count)),
        "identifiable_pair_count": prediction.new_tensor(float(identifiable_pairs)),
        "relation_group_count": zero,
    }


def four_cell_relation_transport_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition_parameters: torch.Tensor,
    device_id: torch.Tensor,
    fault: torch.Tensor,
    *,
    fault_separation_reference_ratio: float,
) -> dict[str, torch.Tensor]:
    """Supervise condition transport on ordered two-condition/two-fault groups."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("four-cell signals must have matching [B,1,T] shape")
    batch = len(prediction)
    if batch == 0 or batch % 4:
        raise ValueError("four-cell relation batch must be a non-empty multiple of four")
    if condition_parameters.ndim != 2 or condition_parameters.shape[0] != batch:
        raise ValueError("four-cell conditions differ from signal batch")
    for name, value in (("device", device_id), ("fault", fault)):
        if value.shape != (batch,) or value.dtype != torch.long:
            raise ValueError(f"four-cell {name} must be long with shape [B]")
    ratio = float(fault_separation_reference_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("fault-separation reference ratio must be in (0,1]")

    predicted = style_descriptor(prediction)
    reference = style_descriptor(target).detach()
    condition_terms: list[torch.Tensor] = []
    slope_terms: list[torch.Tensor] = []
    fault_terms: list[torch.Tensor] = []
    difference_terms: list[torch.Tensor] = []
    margin_terms: list[torch.Tensor] = []
    descriptor_scale = math.sqrt(float(predicted.shape[1]))

    for begin in range(0, batch, 4):
        indices = slice(begin, begin + 4)
        local_device = device_id[indices]
        local_fault = fault[indices]
        local_condition = condition_parameters[indices]
        if not torch.all(local_device == local_device[0]):
            raise ValueError("four-cell group crosses device routes")
        if not (
            local_fault[0] == local_fault[1]
            and local_fault[2] == local_fault[3]
            and local_fault[0] != local_fault[2]
        ):
            raise ValueError("four-cell group fault ordering differs")
        if not (
            torch.allclose(local_condition[0], local_condition[2])
            and torch.allclose(local_condition[1], local_condition[3])
            and not torch.allclose(local_condition[0], local_condition[1])
        ):
            raise ValueError("four-cell group condition ordering differs")

        predicted_group = predicted[indices]
        reference_group = reference[indices]
        condition_distance = torch.linalg.vector_norm(
            local_condition[1] - local_condition[0]
        ).clamp_min(1e-6)

        predicted_condition_left = 0.5 * (
            predicted_group[0] + predicted_group[2]
        )
        predicted_condition_right = 0.5 * (
            predicted_group[1] + predicted_group[3]
        )
        reference_condition_left = 0.5 * (
            reference_group[0] + reference_group[2]
        )
        reference_condition_right = 0.5 * (
            reference_group[1] + reference_group[3]
        )
        condition_terms.extend(
            (
                F.smooth_l1_loss(
                    predicted_condition_left,
                    reference_condition_left,
                ),
                F.smooth_l1_loss(
                    predicted_condition_right,
                    reference_condition_right,
                ),
            )
        )
        slope_terms.append(
            F.smooth_l1_loss(
                (predicted_condition_right - predicted_condition_left)
                / condition_distance,
                (reference_condition_right - reference_condition_left)
                / condition_distance,
            )
        )

        predicted_fault_left = predicted_group[2] - predicted_group[0]
        predicted_fault_right = predicted_group[3] - predicted_group[1]
        reference_fault_left = reference_group[2] - reference_group[0]
        reference_fault_right = reference_group[3] - reference_group[1]
        fault_terms.extend(
            (
                F.smooth_l1_loss(predicted_fault_left, reference_fault_left),
                F.smooth_l1_loss(predicted_fault_right, reference_fault_right),
            )
        )

        predicted_transport_a = (
            predicted_group[1] - predicted_group[0]
        ) / condition_distance
        predicted_transport_b = (
            predicted_group[3] - predicted_group[2]
        ) / condition_distance
        reference_transport_a = (
            reference_group[1] - reference_group[0]
        ) / condition_distance
        reference_transport_b = (
            reference_group[3] - reference_group[2]
        ) / condition_distance
        difference_terms.append(
            F.smooth_l1_loss(
                predicted_transport_a - predicted_transport_b,
                reference_transport_a - reference_transport_b,
            )
        )

        for predicted_fault, reference_fault in (
            (predicted_fault_left, reference_fault_left),
            (predicted_fault_right, reference_fault_right),
        ):
            predicted_distance = torch.linalg.vector_norm(
                predicted_fault
            ) / descriptor_scale
            reference_distance = torch.linalg.vector_norm(
                reference_fault
            ) / descriptor_scale
            margin_terms.append(
                F.relu(ratio * reference_distance - predicted_distance)
            )

    return {
        "condition_style": torch.stack(condition_terms).mean(),
        "fault_content": torch.stack(fault_terms).mean(),
        "transport_slope": torch.stack(slope_terms).mean(),
        "difference_of_differences": torch.stack(difference_terms).mean(),
        "fault_separation_margin": torch.stack(margin_terms).mean(),
        "condition_count": prediction.new_tensor(float(batch // 2)),
        "identifiable_pair_count": prediction.new_tensor(float(batch // 4)),
        "relation_group_count": prediction.new_tensor(float(batch // 4)),
    }
