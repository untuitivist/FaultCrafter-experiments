from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from faultcrafter.jit import normalized_log_power_spectrum


@dataclass(frozen=True, slots=True)
class SourceFaultContrastRoutes:
    starts: torch.Tensor
    counts: torch.Tensor
    groups: torch.Tensor

    @classmethod
    def from_groups(cls, source_groups: Sequence[Sequence[int]], compute_device: torch.device):
        if not source_groups or any(not groups for groups in source_groups):
            raise ValueError("fault contrast requires nonempty source fault vocabularies")
        if any(int(group) < 0 for groups in source_groups for group in groups):
            raise ValueError("fault contrast group indices must be nonnegative")
        counts = torch.tensor([len(groups) for groups in source_groups], dtype=torch.long, device=compute_device)
        starts = counts.cumsum(dim=0) - counts
        groups = torch.tensor([group for groups in source_groups for group in groups], dtype=torch.long, device=compute_device)
        return cls(starts, counts, groups)

    def negatives(self, fault: torch.Tensor, device_id: torch.Tensor, training_step: int):
        if training_step < 1 or fault.ndim != 1 or fault.shape != device_id.shape:
            raise ValueError("fault contrast needs scalar single-fault labels and a positive training step")
        if fault.dtype != torch.long or device_id.dtype != torch.long:
            raise ValueError("fault contrast indices must be int64")
        if torch.any((device_id < 0) | (device_id >= len(self.counts))):
            raise ValueError("fault contrast device is outside source vocabulary")
        starts, counts = self.starts[device_id], self.counts[device_id]
        local = fault - starts
        if torch.any((local < 0) | (local >= counts)):
            raise ValueError("fault contrast label belongs to another device")
        eligible = counts > 1
        positions = torch.arange(len(fault), device=fault.device)
        offset = 1 + (training_step + positions) % (counts - 1).clamp_min(1)
        negative = starts + (local + offset) % counts
        return negative, self.groups[negative], eligible


def source_spectral_contrast(
    positive: torch.Tensor, negative: torch.Tensor, clean: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if positive.shape != negative.shape or positive.shape != clean.shape or time.shape != (len(clean),):
        raise ValueError("source spectral contrast shapes differ")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("source spectral contrast time must lie in[0,1]")
    reference = normalized_log_power_spectrum(clean)
    positive_error = (normalized_log_power_spectrum(positive) - reference).square().mean(dim=1)
    negative_error = (normalized_log_power_spectrum(negative) - reference).square().mean(dim=1)
    scale = (positive_error.detach() + negative_error.detach()).clamp_min(1e-8)
    relative = (positive_error - negative_error) / scale
    loss = (F.softplus(relative) * (1.0 - time).square()).mean()
    return loss, positive_error, negative_error
