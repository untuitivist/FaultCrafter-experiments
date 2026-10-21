from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch


INVERSE_NOISE_SQUARED = "inverse_noise_squared"
MEAN_PRESERVING_P2 = "mean_preserving_p2"


@dataclass(frozen=True)
class DirectXTimeWeighting:
    minimum_time: float
    maximum_time: float
    minimum_noise_fraction: float
    reference_mean: float = field(init=False)
    raw_mean: float = field(init=False)
    normalizer: float = field(init=False)

    def __post_init__(self) -> None:
        values = (self.minimum_time, self.maximum_time, self.minimum_noise_fraction)
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
            raise ValueError("time weighting bounds and noise floor must be finite numbers")
        if not 0.0 <= self.minimum_time < self.maximum_time <= 1.0 or not 0.0 < self.minimum_noise_fraction <= 1.0:
            raise ValueError("time weighting requires ordered [0,1] bounds and a noise floor in (0,1]")
        lower, upper, floor = values
        breakpoint = 1.0 - floor
        reference_parts, raw_parts = [], []
        regular_end = min(upper, breakpoint)
        if lower < regular_end:
            width = regular_end - lower
            reference_parts.append(width / (1.0 - regular_end) / (1.0 - lower))
            raw_parts.append(math.atan2(2.0 * width, 1.0 + (2.0 * lower - 1.0) * (2.0 * regular_end - 1.0)))
        tail_start = max(lower, breakpoint)
        if tail_start < upper:
            width = upper - tail_start
            reference_parts.append(width / floor / floor)
            raw_parts.append(math.atan2(floor * width, floor * floor + tail_start * upper) / floor)
        reference_mean = math.fsum(reference_parts) / (upper - lower)
        raw_mean = math.fsum(raw_parts) / (upper - lower)
        normalizer = reference_mean / raw_mean
        if not all(math.isfinite(value) and value > 0.0 for value in (reference_mean, raw_mean, normalizer)):
            raise ValueError("time weighting integrals must be positive and finite")
        object.__setattr__(self, "reference_mean", reference_mean)
        object.__setattr__(self, "raw_mean", raw_mean)
        object.__setattr__(self, "normalizer", normalizer)

    def weights(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1 or not time.is_floating_point() or not len(time):
            raise ValueError("time weighting requires a nonempty floating point time vector")
        if not torch.isfinite(time).all() or torch.any((time < self.minimum_time) | (time > self.maximum_time)):
            raise ValueError("time weighting samples must lie within the declared uniform bounds")
        noise_fraction = (1.0 - time).clamp_min(self.minimum_noise_fraction)
        return self.normalizer / (noise_fraction.square() + time.square())

    def metadata(self) -> dict[str, Any]:
        return {"policy": MEAN_PRESERVING_P2, "p2_k": 1.0, "p2_gamma": 1.0,
                "minimum_time": self.minimum_time, "maximum_time": self.maximum_time,
                "minimum_noise_fraction": self.minimum_noise_fraction, "reference_mean": self.reference_mean,
                "raw_mean": self.raw_mean, "normalizer": self.normalizer,
                "normalization": "analytic_uniform_time_global_mean", "scope": "fine_and_group_direct_x",
                "source_validation_uses_same_weighting": True, "runtime_parameters_added": False}


def resolve_time_weighting(generator: Mapping[str, Any]) -> DirectXTimeWeighting | None:
    policy = generator.get("direct_x_time_weighting", INVERSE_NOISE_SQUARED)
    if not isinstance(policy, str) or policy not in {INVERSE_NOISE_SQUARED, MEAN_PRESERVING_P2}:
        raise ValueError("unknown direct-x time weighting policy")
    if policy == INVERSE_NOISE_SQUARED:
        return None
    if generator.get("shared_gradient_policy", "joint") != "joint":
        raise ValueError("time weighting does not support the legacy-only shared gradient partition")
    if generator.get("relation_coupled_noise_time", False) or generator.get("relation_balanced_sampling", False):
        raise ValueError("time weighting requires the declared independent uniform time sampler")
    try:
        return DirectXTimeWeighting(generator["minimum_time"], generator["maximum_time"], generator["minimum_noise_fraction"])
    except KeyError as error:
        raise ValueError("time weighting requires all declared bounds and noise floor") from error
