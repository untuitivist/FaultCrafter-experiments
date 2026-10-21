from __future__ import annotations

import itertools

import numpy as np
import torch


def _validated_inputs(
    source_conditions: np.ndarray,
    target_condition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_conditions, dtype=np.float64)
    target = np.asarray(target_condition, dtype=np.float64)
    if source.ndim != 2 or target.shape != (source.shape[1],):
        raise ValueError("condition dimensions do not match")
    if len(source) == 0 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("conditions must be finite and non-empty")
    return source, target


def nearest_condition_weights(
    source_conditions: np.ndarray,
    target_condition: np.ndarray,
) -> np.ndarray:
    source, target = _validated_inputs(source_conditions, target_condition)
    distances = np.linalg.norm(source - target[None, :], axis=1)
    nearest = np.flatnonzero(np.isclose(distances, distances.min()))
    weights = np.zeros(len(source), dtype=np.float64)
    weights[nearest] = 1.0 / len(nearest)
    return weights


def softmax_condition_weights(
    source_conditions: np.ndarray,
    target_condition: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    source, target = _validated_inputs(source_conditions, target_condition)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    squared_distance = np.square(source - target[None, :]).sum(axis=1)
    logits = -0.5 * squared_distance / bandwidth**2
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def local_linear_condition_weights(
    source_conditions: np.ndarray,
    target_condition: np.ndarray,
    *,
    bandwidth: float,
    rank_tolerance: float = 1e-10,
) -> np.ndarray:
    """Equivalent-kernel weights for multivariate local linear regression.

    The returned weights reproduce affine condition responses: their sum is one
    and their source-condition weighted mean equals the requested condition.
    Negative weights are intentionally permitted for parameter extrapolation.
    """

    source, target = _validated_inputs(source_conditions, target_condition)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    centered = source - target[None, :]
    design = np.column_stack((np.ones(len(source)), centered))
    if np.linalg.matrix_rank(design, tol=rank_tolerance) != design.shape[1]:
        raise ValueError("source conditions do not span an affine condition model")
    kernel = np.exp(-0.5 * np.square(centered).sum(axis=1) / bandwidth**2)
    moment = design.T @ (kernel[:, None] * design)
    intercept = np.zeros(design.shape[1], dtype=np.float64)
    intercept[0] = 1.0
    coefficients = np.linalg.solve(moment, intercept)
    weights = kernel * (design @ coefficients)
    if not np.allclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("local linear weights do not preserve constants")
    if not np.allclose(weights @ source, target, atol=1e-8):
        raise RuntimeError("local linear weights do not preserve affine conditions")
    return weights


def median_pairwise_bandwidth(conditions: np.ndarray) -> float:
    source = np.asarray(conditions, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2:
        raise ValueError("at least two condition rows are required")
    differences = source[:, None, :] - source[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    positive = distances[np.triu_indices(len(source), k=1)]
    positive = positive[positive > 0.0]
    if len(positive) == 0:
        raise ValueError("condition rows must not all be identical")
    return float(np.median(positive))


def torch_condition_weight_family(
    source_conditions: torch.Tensor,
    target_condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian-kernel and local-linear weights for each batch row."""

    if source_conditions.ndim != 3:
        raise ValueError("source conditions must have shape [batch,source,parameter]")
    batch_size, source_count, parameter_count = source_conditions.shape
    if target_condition.shape != (batch_size, parameter_count):
        raise ValueError("target condition shape differs from source conditions")
    if source_count == 0:
        raise ValueError("at least one source condition is required")
    if not torch.isfinite(source_conditions).all() or not torch.isfinite(
        target_condition
    ).all():
        raise ValueError("conditions must be finite")
    if source_count == 1:
        unit = source_conditions.new_ones((batch_size, 1))
        return unit, unit

    pairwise = torch.linalg.vector_norm(
        source_conditions[:, :, None, :] - source_conditions[:, None, :, :],
        dim=-1,
    )
    triangle = torch.triu_indices(source_count, source_count, offset=1)
    positive = pairwise[:, triangle[0], triangle[1]]
    bandwidth = positive.median(dim=1).values.clamp_min(1e-6)
    centered = source_conditions - target_condition[:, None, :]
    kernel = torch.exp(
        -0.5 * centered.square().sum(dim=-1) / bandwidth[:, None].square()
    )
    softmax = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)

    design = torch.cat(
        (
            torch.ones(
                (batch_size, source_count, 1),
                device=source_conditions.device,
                dtype=source_conditions.dtype,
            ),
            centered,
        ),
        dim=2,
    )
    weighted_design = kernel[:, :, None] * design
    moment = design.transpose(1, 2) @ weighted_design
    intercept = source_conditions.new_zeros((batch_size, parameter_count + 1, 1))
    intercept[:, 0, 0] = 1.0
    coefficients = torch.linalg.pinv(moment, hermitian=True) @ intercept
    local_linear = kernel * (design @ coefficients).squeeze(2)
    return softmax, local_linear


def torch_convex_hull_projection(
    source_conditions: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    feasibility_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Project each target onto the convex hull of its source conditions.

    Condition dimensions are small in the fault-generation protocol. Enumerating
    faces up to the Caratheodory bound gives a deterministic Euclidean projection
    without introducing an external optimization dependency.
    """

    if source_conditions.ndim != 3:
        raise ValueError("source conditions must have shape [batch,source,parameter]")
    batch_size, source_count, parameter_count = source_conditions.shape
    if target_condition.shape != (batch_size, parameter_count):
        raise ValueError("target condition shape differs from source conditions")
    if source_count == 0:
        raise ValueError("at least one source condition is required")
    if feasibility_tolerance < 0.0:
        raise ValueError("feasibility tolerance cannot be negative")
    if not torch.isfinite(source_conditions).all() or not torch.isfinite(
        target_condition
    ).all():
        raise ValueError("conditions must be finite")

    projected_rows: list[torch.Tensor] = []
    maximum_face_size = min(source_count, parameter_count + 1)
    for batch_index in range(batch_size):
        source = source_conditions[batch_index]
        target = target_condition[batch_index]
        best_projection = source[0]
        best_distance = (best_projection - target).square().sum()
        for face_size in range(1, maximum_face_size + 1):
            for indices in itertools.combinations(range(source_count), face_size):
                vertices = source[list(indices)]
                gram = vertices @ vertices.T
                ones = torch.ones(
                    (face_size, 1), device=source.device, dtype=source.dtype
                )
                kkt = torch.cat(
                    (
                        torch.cat((gram, ones), dim=1),
                        torch.cat(
                            (
                                ones.T,
                                torch.zeros(
                                    (1, 1), device=source.device, dtype=source.dtype
                                ),
                            ),
                            dim=1,
                        ),
                    ),
                    dim=0,
                )
                right_hand_side = torch.cat(
                    (vertices @ target, source.new_ones(1)), dim=0
                )
                solution = torch.linalg.pinv(kkt) @ right_hand_side
                weights = solution[:face_size]
                if torch.any(weights < -feasibility_tolerance):
                    continue
                weights = weights.clamp_min(0.0)
                weights = weights / weights.sum().clamp_min(1e-12)
                candidate = weights @ vertices
                distance = (candidate - target).square().sum()
                if distance < best_distance:
                    best_projection = candidate
                    best_distance = distance
        projected_rows.append(best_projection)
    return torch.stack(projected_rows)


def torch_projected_condition_weight_family(
    source_conditions: torch.Tensor,
    target_condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hull-supported kernel weights and a linear target residual.

    The affine weights use a kernel frozen at the hull projection. Consequently,
    their change from the projected condition to the requested condition is
    exactly linear in the out-of-hull residual instead of saturating it.
    """

    projected = torch_convex_hull_projection(source_conditions, target_condition)
    softmax, _ = torch_condition_weight_family(source_conditions, projected)
    batch_size, source_count, parameter_count = source_conditions.shape
    if source_count == 1:
        return softmax, softmax, projected, target_condition - projected

    pairwise = torch.linalg.vector_norm(
        source_conditions[:, :, None, :] - source_conditions[:, None, :, :],
        dim=-1,
    )
    triangle = torch.triu_indices(source_count, source_count, offset=1)
    positive = pairwise[:, triangle[0], triangle[1]]
    bandwidth = positive.median(dim=1).values.clamp_min(1e-6)
    centered = source_conditions - projected[:, None, :]
    kernel = torch.exp(
        -0.5 * centered.square().sum(dim=-1) / bandwidth[:, None].square()
    )
    design = torch.cat(
        (
            torch.ones(
                (batch_size, source_count, 1),
                device=source_conditions.device,
                dtype=source_conditions.dtype,
            ),
            centered,
        ),
        dim=2,
    )
    moment = design.transpose(1, 2) @ (kernel[:, :, None] * design)
    residual = target_condition - projected
    query = torch.cat(
        (source_conditions.new_ones((batch_size, 1)), residual), dim=1
    )[:, :, None]
    coefficients = torch.linalg.pinv(moment, hermitian=True) @ query
    affine = kernel * (design @ coefficients).squeeze(2)
    return softmax, affine, projected, residual
