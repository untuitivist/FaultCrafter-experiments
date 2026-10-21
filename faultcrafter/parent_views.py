from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from faultcrafter.config import load_config


MINIMAL_NONOVERLAP_PARENTS = "minimal_nonoverlap_parents_v1"
PARENT_FIELDS = frozenset({"parent_view_policy", "parent_view_count", "cnn_view_parent", "cnn_view_start",
                          "extra_normalized_signal", "extra_predicted_amplitude", "extra_predicted_scale",
                          "extra_predicted_location", "extra_physical_signal"})


def make_parent_view_plan(window_length: int, stride: int, windows_per_fault: int, target_length: int) -> dict:
    if any(type(value) is not int or value <= 0 for value in (window_length, stride, windows_per_fault, target_length)):
        raise ValueError("parent views require positive integer dimensions")
    if target_length != window_length + stride * (windows_per_fault - 1):
        raise ValueError("parent length must retain exactly the original CNN window grid")
    count = min(windows_per_fault, (window_length + stride - 1) // stride)
    return {"policy": MINIMAL_NONOVERLAP_PARENTS, "window_length": window_length, "stride": stride,
            "windows_per_fault": windows_per_fault, "target_length": target_length, "parent_count": count,
            "view_starts": [index * stride for index in range(windows_per_fault)],
            "view_parents": [index % count for index in range(windows_per_fault)]}


def resolve_parent_view_plan(config: Mapping[str, Any], experiment0: Mapping[str, Any] | None = None) -> dict | None:
    policy = config["generation"].get("parent_view_policy", "none")
    if policy == "none":
        return None
    if policy != MINIMAL_NONOVERLAP_PARENTS:
        raise ValueError("unknown generation parent view policy")
    if config.get("normalization_method", "max_abs") != "max_abs" or config["generator"].get("affine_restoration", False):
        raise ValueError("parent views require the declared max-abs zero-location representation")
    experiment0 = experiment0 if experiment0 is not None else load_config(Path(config["experiment0_config"]))
    if int(config["cnn_window_length"]) != experiment0["window_length"]:
        raise ValueError("parent view and CNN window dimensions disagree")
    return make_parent_view_plan(experiment0["window_length"], experiment0["stride"],
                                 experiment0["per_cell_window_counts"]["train"], config["generation"]["target_length"])


def parent_archive_payload(plan: dict, normalized: np.ndarray, amplitude: np.ndarray, location: np.ndarray, physical: np.ndarray) -> dict:
    if plan != make_parent_view_plan(plan["window_length"], plan["stride"], plan["windows_per_fault"], plan["target_length"]):
        raise ValueError("parent archive plan differs from the canonical minimum-parent rule")
    if normalized.ndim != 4 or normalized.shape[0] != plan["parent_count"] or normalized.shape[2:] != (1, plan["target_length"]):
        raise ValueError("parent generation shape differs from its complete plan")
    if physical.shape != normalized.shape or amplitude.shape != normalized.shape[:2] or location.shape != amplitude.shape:
        raise ValueError("parent restoration arrays do not align")
    return {"parent_view_policy": np.asarray([plan["policy"]]), "parent_view_count": np.asarray([plan["parent_count"]], dtype=np.int64),
            "cnn_view_parent": np.asarray(plan["view_parents"], dtype=np.int64), "cnn_view_start": np.asarray(plan["view_starts"], dtype=np.int64),
            "extra_normalized_signal": normalized[1:], "extra_predicted_amplitude": amplitude[1:], "extra_predicted_scale": amplitude[1:],
            "extra_predicted_location": location[1:], "extra_physical_signal": physical[1:]}


def validate_parent_archive(archive: Mapping[str, Any], plan: dict | None) -> dict:
    keys = set(archive)
    if plan is None:
        if keys & PARENT_FIELDS:
            raise ValueError("extra generated parents cannot be ignored by a legacy view policy")
        return {}
    if plan != make_parent_view_plan(plan["window_length"], plan["stride"], plan["windows_per_fault"], plan["target_length"]):
        raise ValueError("parent archive plan differs from the canonical minimum-parent rule")
    if not PARENT_FIELDS <= keys or np.asarray(archive["parent_view_policy"]).tolist() != [plan["policy"]]:
        raise ValueError("generated parent archive lacks its declared policy or complete data")
    for name, values in (("parent_view_count", [plan["parent_count"]]), ("cnn_view_parent", plan["view_parents"]), ("cnn_view_start", plan["view_starts"])):
        actual = np.asarray(archive[name])
        if actual.dtype.kind not in "iu" or not np.array_equal(actual, np.asarray(values, dtype=np.int64)):
            raise ValueError("generated parent view provenance differs from the fixed plan")
    fault_count = len(archive["fault_label"])
    if fault_count == 0 or len(set(str(value) for value in archive["fault_label"])) != fault_count:
        raise ValueError("generated parent primary fault vocabulary must be nonempty and unique")
    joined = {}
    for field in ("normalized_signal", "predicted_amplitude", "predicted_scale", "predicted_location", "physical_signal"):
        primary = np.asarray(archive[field])
        extra = np.asarray(archive["extra_" + field])
        expected = (fault_count, 1, plan["target_length"]) if field.endswith("signal") else (fault_count,)
        if primary.shape != expected or extra.shape != (plan["parent_count"] - 1, *expected):
            raise ValueError("generated parent array shape differs from its fault and parent counts")
        if primary.dtype != np.float32 or extra.dtype != np.float32 or not np.isfinite(primary).all() or not np.isfinite(extra).all():
            raise ValueError("generated parent arrays must be finite float32 values")
        joined[field] = np.concatenate((primary[None], extra), axis=0)
    if np.any(joined["predicted_amplitude"] <= 0) or not np.array_equal(joined["predicted_scale"], joined["predicted_amplitude"]) or np.any(joined["predicted_location"] != 0):
        raise ValueError("generated parent amplitude or zero-location format is invalid")
    if not np.allclose(np.abs(joined["normalized_signal"]).max(axis=(2, 3)), 1.0, rtol=0, atol=1e-5):
        raise ValueError("every generated parent must retain full-parent max-abs normalization")
    restored = joined["normalized_signal"] * joined["predicted_amplitude"][:, :, None, None]
    if not np.allclose(joined["physical_signal"], restored, rtol=1e-6, atol=1e-6):
        raise ValueError("generated parent physical restoration differs")
    return {"policy": plan["policy"], "parents_per_fault": plan["parent_count"], "generated_long_signal_count": fault_count * plan["parent_count"],
            "windows_per_fault": plan["windows_per_fault"], "view_starts": plan["view_starts"], "view_parents": plan["view_parents"],
            "views_per_parent_per_fault": np.bincount(plan["view_parents"], minlength=plan["parent_count"]).tolist(),
            "no_direct_sample_overlap_within_parent": True, "all_views_statistically_independent_claimed": False,
            "extra_runtime_parameters_or_files": False}


def cut_parent_views(archive: Mapping[str, Any], plan: dict) -> tuple[np.ndarray, dict]:
    evidence = validate_parent_archive(archive, plan)
    count = len(archive["fault_label"])
    windows = np.empty((count * plan["windows_per_fault"], 1, plan["window_length"]), dtype=np.float32)
    for fault_index in range(count):
        for view_index, (parent_index, start) in enumerate(zip(plan["view_parents"], plan["view_starts"], strict=True)):
            parent = archive["normalized_signal"][fault_index] if parent_index == 0 else archive["extra_normalized_signal"][parent_index - 1, fault_index]
            windows[fault_index * plan["windows_per_fault"] + view_index] = parent[:, start:start + plan["window_length"]]
    return windows, {**evidence, "generated_views_shape": list(windows.shape), "generated_views_sha256": hashlib.sha256(windows.tobytes()).hexdigest()}
