from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from faultcrafter.condition_units import condition_field, unit_factor
from faultcrafter.fixed_condition_scaling import FIXED_PHYSICAL_SCALES
from faultcrafter.typed_conditions import TYPED_DIMENSION, encode_typed_conditions, physical_slots


def balanced_lora_factors(up: torch.Tensor, down: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0] or min(*up.shape, *down.shape) <= 0:
        raise ValueError("balanced LoRA requires compatible nonempty matrix factors")
    if not torch.is_floating_point(up) or not torch.is_floating_point(down) or up.dtype != down.dtype:
        raise ValueError("balanced LoRA factors must share a floating dtype")
    if not torch.isfinite(up).all() or not torch.isfinite(down).all():
        raise ValueError("balanced LoRA factors must be finite")
    rank = up.shape[1]
    residual = up.detach().to(device="cpu", dtype=torch.float64) @ down.detach().to(device="cpu", dtype=torch.float64) / rank
    if not torch.isfinite(residual).all():
        raise ValueError("balanced LoRA residual overflowed")
    left, singular, right = torch.linalg.svd(residual, full_matrices=False)
    threshold = singular[0] * max(residual.shape) * torch.finfo(torch.float64).eps
    active_rank = int((singular > threshold).sum())
    if active_rank > rank:
        raise ValueError("balanced LoRA numerical rank exceeds its adapter rank")
    new_up = torch.zeros_like(up, device="cpu")
    new_down = torch.empty_like(down, device="cpu")
    random = torch.Generator().manual_seed(seed)
    bound = 1.0 / math.sqrt(down.shape[1])
    new_down.uniform_(-bound, bound, generator=random)
    scale = (singular[:active_rank] * rank).sqrt()
    new_up[:, :active_rank] = left[:, :active_rank] * scale
    new_down[:active_rank] = scale[:, None] * right[:active_rank]
    reconstructed = new_up.double() @ new_down.double() / rank
    torch.testing.assert_close(reconstructed, residual, rtol=2e-6, atol=2e-6)
    up_norm = new_up[:, :active_rank].double().norm(dim=0)
    down_norm = new_down[:active_rank].double().norm(dim=1)
    balance_ratio = torch.maximum(up_norm / down_norm, down_norm / up_norm)
    return new_up, new_down, {
        "mode": "balanced_svd", "active_rank": active_rank, "adapter_rank": rank,
        "spare_rank": rank - active_rank, "spare_up_zero_down_local_random": True,
        "rank_threshold": float(threshold), "local_seed": seed,
        "max_effective_weight_error": float((reconstructed - residual).abs().max()),
        "active_up_norms": up_norm.tolist(), "active_down_norms": down_norm.tolist(),
        "max_active_factor_norm_ratio": float(balance_ratio.max()) if active_rank else 1.0,
    }


def migrate_condition_projection(
    model: Any, devices: Any, checkpoint: dict, *, factorization: str = "coordinate",
) -> tuple[dict, dict]:
    if factorization not in {"coordinate", "balanced_svd"}:
        raise ValueError("unsupported typed condition migration factorization")
    source = checkpoint["model"]
    shared_name = "style_field.query.shared.weight"
    down_name = "style_field.query.lora_down"
    up_name = "style_field.query.lora_up"
    old_shared = source[shared_name].double()
    old_dimension = old_shared.shape[1]
    query = model.style_field.query
    if model.condition_dimension != TYPED_DIMENSION or old_dimension != 3 or query.rank < old_dimension:
        raise ValueError("condition migration requires the original two-coordinate projection and sufficient LoRA rank")
    new_shared = torch.zeros_like(query.shared.weight, device="cpu")
    new_shared[:, 0] = old_shared[:, 0]
    new_down = torch.zeros_like(query.lora_down, device="cpu")
    new_up = torch.zeros_like(query.lora_up, device="cpu")
    audits = []
    for device in devices:
        index = device.index
        names = tuple(device.parameter_names)
        if tuple(checkpoint["parameter_names"][index]) != names:
            raise ValueError("initial checkpoint physical fields differ")
        mean = np.asarray(checkpoint["parameter_mean"][index], dtype=np.float64)
        deviation = np.asarray(checkpoint["parameter_standard_deviation"][index], dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(deviation).all() or (deviation <= 0).any():
            raise ValueError("invalid initialization-only coordinate statistics")
        transform = torch.zeros(old_dimension, TYPED_DIMENSION + 1, dtype=torch.float64)
        transform[0, 0] = 1.0
        for offset, (name, slot) in enumerate(zip(names, physical_slots(names), strict=True)):
            field = condition_field(name)
            factor = unit_factor(name, field.native_unit)
            transform[offset + 1, slot + 1] = FIXED_PHYSICAL_SCALES[field.quantity] / (factor * deviation[offset])
            transform[offset + 1, slot + 5] = -mean[offset] / deviation[offset]
        effective = old_shared + source[up_name][index].double() @ source[down_name][index].double() / query.rank
        random = torch.Generator().manual_seed(41000 + index)
        new_down[index].uniform_(-1 / math.sqrt(TYPED_DIMENSION + 1), 1 / math.sqrt(TYPED_DIMENSION + 1), generator=random)
        new_down[index, :old_dimension] = transform
        new_up[index, :, :old_dimension] = effective * query.rank
        new_up[index, :, 0] -= new_shared[:, 0] * query.rank
        factor_audit = None
        if factorization == "balanced_svd":
            balanced_up, balanced_down, factor_audit = balanced_lora_factors(new_up[index], new_down[index], seed=41000 + index)
            new_up[index] = balanced_up
            new_down[index] = balanced_down
        raw = np.vstack((device.corpus.parameter_matrix, device.target_parameters))
        old_values = torch.zeros(len(raw), old_dimension)
        old_values[:, 0] = 1.0
        old_values[:, 1:len(names) + 1] = torch.from_numpy(((raw - mean) / deviation).astype(np.float32))
        new_values = torch.cat((torch.ones(len(raw), 1), torch.from_numpy(encode_typed_conditions(names, raw))), dim=1)
        expected = old_values @ effective.float().T
        migrated = new_shared + new_up[index] @ new_down[index] / query.rank
        actual = new_values @ migrated.T
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        audits.append({"device_id": device.device_id, "max_query_error": float((actual - expected).abs().max()),
                       **({"factor_balance": factor_audit} if factor_audit is not None else {})})
    migrated_state = dict(source)
    migrated_state.update({shared_name: new_shared, down_name: new_down, up_name: new_up})
    return migrated_state, {
        "strategy": "function_preserving_typed_coordinate_migration",
        "old_dimension": 2, "new_dimension": TYPED_DIMENSION,
        "source_statistics_used_only_for_initial_weight_migration": True,
        "source_statistics_required_at_inference": False,
        "changed_tensors": [shared_name, down_name, up_name], "devices": audits,
        **({"factorization_mode": factorization} if factorization != "coordinate" else {}),
    }
