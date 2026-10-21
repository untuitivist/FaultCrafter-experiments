from __future__ import annotations

import math
from typing import Mapping


SOURCE_RELATIVE_MINIMAX = "relative_device_minimax_v1"
DENOMINATOR_FLOOR = 1e-8


def source_selection_policy(generator: dict) -> str:
    policy = generator.get("source_checkpoint_selection", "original")
    if policy not in ("original", SOURCE_RELATIVE_MINIMAX):
        raise ValueError("unknown source checkpoint selection policy")
    return policy


def device_validation_scores(details: Mapping[str, float], device_ids: list[str]) -> dict[str, float]:
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError("source selection requires unique nonempty device routes")
    expected = {f"device_{index}_{field}" for index in range(len(device_ids)) for field in ("objective", "count")}
    if {key for key in details if key.startswith("device_")} != expected:
        raise ValueError("source validation device routes are incomplete or unexpected")
    scores = {}
    for index, name in enumerate(device_ids):
        count = details[f"device_{index}_count"]
        value = float(details[f"device_{index}_objective"])
        if not math.isfinite(count) or count <= 0 or int(count) != count or not math.isfinite(value):
            raise ValueError("source device scores require finite losses and positive integer counts")
        scores[name] = value
    return scores


class SourceMinimaxSelection:
    def __init__(self, reference: Mapping[str, float]):
        if not reference or any(not isinstance(name, str) or not name for name in reference):
            raise ValueError("source minimax reference needs named devices")
        self.reference = {name: float(value) for name, value in reference.items()}
        if not all(math.isfinite(value) for value in self.reference.values()):
            raise ValueError("source minimax reference must be finite")

    def score(self, current: Mapping[str, float]) -> dict:
        if set(current) != set(self.reference):
            raise ValueError("source minimax cannot omit or add device routes")
        scores = {name: float(current[name]) for name in self.reference}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("source minimax device losses must be finite")
        excess = {name: (scores[name] - initial) / max(abs(initial), DENOMINATOR_FLOOR)
                  for name, initial in self.reference.items()}
        if not all(math.isfinite(value) for value in excess.values()):
            raise ValueError("source minimax relative score must be finite")
        return {"checkpoint_selection_objective": max(excess.values()),
                "source_minimax_device_objectives": scores, "source_minimax_relative_excess": excess}

    def metadata(self) -> dict:
        return {"policy": SOURCE_RELATIVE_MINIMAX, "denominator_floor": DENOMINATOR_FLOOR,
                "reference_device_objectives": dict(self.reference),
                "reference_model": "initial_migrated13_frozen_ema_not_a_checkpoint_candidate",
                "selection": "minimum_worst_device_relative_excess_single_shared_checkpoint_earliest_exact_tie",
                "validation": "original_source_intervals_cells_lengths_grouped_by_device",
                "training_loss_and_sampling_changed": False, "scheduler_and_patience_use_original_validation": True,
                "extra_runtime_parameters_or_files": False}
