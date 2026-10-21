from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from faultcrafter.fixed_condition_scaling import FIXED_PHYSICAL_SCALES
from faultcrafter.typed_conditions import TYPED_CONDITION_PROTOCOL, TYPED_DIMENSION, encode_typed_conditions


LEGACY_CYCLIC_REFERENCE = "legacy_cyclic"
KNOWN_MECHANICAL_REFERENCE = "known_mechanical_only"
MASKED_MECHANICAL_REFERENCE = "seven_field_mechanical_only"
PHYSICS_REFERENCE_POLICIES = frozenset({LEGACY_CYCLIC_REFERENCE, KNOWN_MECHANICAL_REFERENCE, MASKED_MECHANICAL_REFERENCE})
MECHANICAL_ANGLE_COORDINATE = "theta = 2*pi*shaft_speed_hz*t only when shaft_frequency_present=1; otherwise no shaft-angle prior"


def validate_reference_policy(policy: str, condition_dimension: int) -> str:
    if not isinstance(policy, str) or policy not in PHYSICS_REFERENCE_POLICIES:
        raise ValueError("unknown physical reference policy")
    if policy == KNOWN_MECHANICAL_REFERENCE and condition_dimension != TYPED_DIMENSION:
        raise ValueError("known mechanical reference requires the eight typed condition coordinates")
    if policy == MASKED_MECHANICAL_REFERENCE and condition_dimension != 14:
        raise ValueError("seven-field reference requires seven values and seven presence masks")
    return policy


def reference_policy_for_config(config: dict) -> str:
    policy = validate_reference_policy(config.get("generator", {}).get("physics_reference_policy", LEGACY_CYCLIC_REFERENCE), int(config.get("condition_dimension", 0)))
    if policy == KNOWN_MECHANICAL_REFERENCE and config.get("condition_protocol") != TYPED_CONDITION_PROTOCOL:
        raise ValueError("known mechanical reference requires the explicit typed condition protocol")
    return policy


def resolve_physical_reference(
    parameters: torch.Tensor, cyclic_reference_hz: torch.Tensor, policy: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    validate_reference_policy(policy, parameters.shape[-1] if parameters.ndim else 0)
    if policy == LEGACY_CYCLIC_REFERENCE:
        return cyclic_reference_hz, None
    if parameters.ndim != 2 or cyclic_reference_hz.shape != (len(parameters),):
        raise ValueError("physical reference requires batched conditions and [batch] Hz")
    if not parameters.dtype.is_floating_point or not cyclic_reference_hz.dtype.is_floating_point or parameters.device != cyclic_reference_hz.device:
        raise ValueError("physical reference conditions and Hz need floating tensors on the same device")
    if not torch.isfinite(parameters).all() or not torch.isfinite(cyclic_reference_hz).all():
        raise ValueError("physical reference inputs must be finite")
    mask_offset = 7 if policy == MASKED_MECHANICAL_REFERENCE else 4
    masks = parameters[:, mask_offset:]
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError("typed presence masks must be independent binary values")
    present = parameters[:, mask_offset] == 1
    comparison_dtype = torch.promote_types(parameters.dtype, cyclic_reference_hz.dtype)
    expected = parameters[:, 0].to(comparison_dtype) * FIXED_PHYSICAL_SCALES["shaft_frequency"]
    tolerance = 16 * max(torch.finfo(parameters.dtype).eps, torch.finfo(cyclic_reference_hz.dtype).eps, torch.finfo(torch.float32).eps)
    if not torch.isclose(cyclic_reference_hz[present].to(comparison_dtype), expected[present], rtol=tolerance, atol=tolerance).all():
        raise ValueError("physical shaft Hz disagrees with the explicitly supplied typed shaft frequency")
    return torch.where(present, cyclic_reference_hz, torch.zeros_like(cyclic_reference_hz)), present


def condition_reference_rows(parameter_names: Sequence[str], values: Any, cyclic_index: int, cyclic_to_hz: float, policy: str) -> dict:
    raw = np.asarray(values, dtype=np.float32).reshape(-1, len(parameter_names))
    coordinates = encode_typed_conditions(parameter_names, raw)
    cyclic = np.asarray(raw[:, cyclic_index] * cyclic_to_hz, dtype=np.float32)
    effective, present = resolve_physical_reference(torch.from_numpy(coordinates), torch.from_numpy(cyclic), policy)
    return {"native_parameters": raw.tolist(), "typed_coordinates": coordinates.tolist(),
            "legacy_cyclic_hz": cyclic.tolist(), "effective_shaft_hz": effective.tolist(),
            "shaft_reference_present": None if present is None else present.tolist()}


def physical_reference_audit(devices: Sequence[Any], policy: str) -> dict:
    if policy != KNOWN_MECHANICAL_REFERENCE:
        raise ValueError("mechanical-reference audit requires the explicit known-only policy")
    rows = []
    for device in devices:
        source = np.unique(device.corpus.parameter_matrix, axis=0)
        rows.append({"device_id": device.device_id, "run_id": device.run_id, "parameter_names": list(device.parameter_names),
                     "source_condition_references": condition_reference_rows(device.parameter_names, source, device.speed_parameter_index, device.speed_to_hz, policy),
                     "target_condition_reference": condition_reference_rows(device.parameter_names, device.target_parameters, device.speed_parameter_index, device.speed_to_hz, policy)})
    return {"policy": policy, "scope": "condition_metadata_audit_not_an_inference_dependency", "target_waveforms_read": False,
            "extra_device_calibration_parameters": False, "rows": rows}
