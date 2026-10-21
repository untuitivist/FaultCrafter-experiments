from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


COMPOSITION_RULE = "normal_anchored_atomic_sum_v1"
MAXIMUM_COMPONENTS = 4


def composition_weights(
    components: Sequence[Sequence[int]],
    normal_indices: Sequence[int],
    vocabulary_size: int,
    *,
    allow_repeated_components: bool = False,
) -> np.ndarray:
    if vocabulary_size <= 0 or len(components) != len(normal_indices):
        raise ValueError("composition vocabulary or batch dimensions differ")
    weights = np.zeros((len(components), vocabulary_size), dtype=np.float32)
    for row, (members, normal) in enumerate(zip(components, normal_indices, strict=True)):
        members = tuple(members)
        if not 0 <= normal < vocabulary_size or any(not 0 <= member < vocabulary_size for member in members):
            raise ValueError("composition index is outside the vocabulary")
        if len(members) > MAXIMUM_COMPONENTS:
            raise ValueError("composition exceeds the frozen component limit")
        if not allow_repeated_components and len(set(members)) != len(members):
            raise ValueError("duplicate fault components are not allowed")
        if normal in members and len(members) != 1:
            raise ValueError("normal cannot be mixed with fault components")
        if not members:
            weights[row, normal] = 1
        else:
            weights[row, normal] = 1 - len(members)
            for member in members:
                weights[row, member] += 1
    return weights


def validate_fault_condition(value: torch.Tensor, batch_size: int, vocabulary_size: int, name: str) -> None:
    if value.ndim == 1:
        if value.shape != (batch_size,) or value.dtype != torch.long:
            raise ValueError(f"{name} must be a long tensor with shape [batch]")
        if torch.any(value < 0) or torch.any(value >= vocabulary_size):
            raise ValueError(f"{name} index is outside the configured range")
        return
    if value.shape != (batch_size, vocabulary_size) or not value.is_floating_point():
        raise ValueError(f"{name} composition must have shape [batch,vocabulary]")
    if not torch.isfinite(value).all() or not torch.equal(value, value.round()):
        raise ValueError(f"{name} composition weights must be finite integers")
    if not torch.all(value.sum(dim=1) == 1):
        raise ValueError(f"{name} composition must preserve the normal anchor")
    if torch.any((value < 0).sum(dim=1) > 1):
        raise ValueError(f"{name} composition has multiple negative anchors")
    if torch.any(value.clamp_min(0).sum(dim=1) > MAXIMUM_COMPONENTS):
        raise ValueError(f"{name} composition exceeds the component limit")


def embed_fault_condition(embedding: nn.Embedding, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return embedding(value)
    singleton = (value != 0).sum(dim=1) == 1
    selected = embedding(value.argmax(dim=1))
    if bool(singleton.all()):
        return selected
    composed = value.to(dtype=embedding.weight.dtype) @ embedding.weight
    return torch.where(singleton[:, None], selected, composed)
