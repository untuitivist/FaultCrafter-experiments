from __future__ import annotations

import torch


IDENTITY_GUIDANCE_MODES = frozenset({"linear", "conditional_rms_cap", "unit_peak_projection"})


def project_unit_peak(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim < 2 or min(prediction.shape) <= 0 or not prediction.is_floating_point():
        raise ValueError("peak projection requires nonempty floating batched signals")
    if prediction.dtype not in (torch.float32, torch.float64):
        raise ValueError("peak projection requires float32 or float64 for its fixed-epsilon gradients")
    peak = prediction.abs().flatten(start_dim=1).amax(dim=1)
    denominator = peak.clamp_min(1e-8)
    shape = (-1,) + (1,) * (prediction.ndim - 1)
    projected = prediction / denominator.reshape(shape)
    return projected, denominator.reciprocal()


def cap_guidance_rms(
    prediction: torch.Tensor, conditional: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != conditional.shape or prediction.ndim < 2:
        raise ValueError("guidance cap requires matching batched signal shapes")
    if not prediction.is_floating_point() or not conditional.is_floating_point():
        raise ValueError("guidance cap requires floating-point signals")
    prediction_norm = torch.linalg.vector_norm(
        prediction.flatten(start_dim=1), dim=1, dtype=torch.float64
    )
    conditional_norm = torch.linalg.vector_norm(
        conditional.flatten(start_dim=1), dim=1, dtype=torch.float64
    )
    maximum = torch.maximum(prediction_norm, conditional_norm)
    nonzero = maximum > 0
    denominator = torch.where(nonzero, maximum, torch.ones_like(maximum))
    multiplier = torch.where(
        nonzero, conditional_norm / denominator, torch.ones_like(maximum)
    ).to(prediction.dtype)
    shape = (-1,) + (1,) * (prediction.ndim - 1)
    return prediction * multiplier.reshape(shape), multiplier
