from __future__ import annotations

from typing import Any, Mapping


EXPANDED_SOURCE_BATCH = "expanded_source_batch_v1"
VALIDATION_BUDGET_KEY = "validation_point_budget_per_batch"


def source_validation_point_budget(generator: Mapping[str, Any]) -> int:
    if VALIDATION_BUDGET_KEY not in generator:
        return int(generator["point_budget_per_batch"])
    value = generator[VALIDATION_BUDGET_KEY]
    if type(value) is not int or value <= 0:
        raise ValueError("explicit source validation point budget must be a positive integer")
    return value


def expanded_source_batch_metadata(generator: Mapping[str, Any]) -> dict | None:
    policy = generator.get("training_batch_budget_policy", "none")
    if policy == "none":
        return None
    if policy != EXPANDED_SOURCE_BATCH:
        raise ValueError("unknown source training batch budget policy")
    expected = {"maximum_batch_size": 64, "point_budget_per_batch": 131072, VALIDATION_BUDGET_KEY: 65536}
    if any(type(generator.get(name)) is not int or generator[name] != value for name, value in expected.items()):
        raise ValueError("expanded source batch requires64 signals/131072 training points/65536 validation points")
    if generator.get("device_balanced_sampling", False) or generator.get("relation_balanced_sampling", False) or generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None:
        raise ValueError("expanded source batch preserves original joint source sampling")
    return {"policy": EXPANDED_SOURCE_BATCH, **expected, "original_maximum_batch_size": 32, "original_point_budget_per_batch": 65536,
            "batch_formula": "min(maximum_batch_size,max(1,point_budget_per_batch//parent_length))",
            "source_sampling": "original_length_and_cell_samplers_more_cells_consumed_per_update",
            "time_sampling": "original_independent_uniform", "source_validation": "original65536_point_partition_and_batch_seeds",
            "source_exposure_and_compute_equal_to_reference_claimed": False, "learning_rate_scaled_with_batch_size": False,
            "gradient_accumulation": False, "backwards_per_optimizer_update": 1, "device_balancing": False,
            "extra_runtime_parameters_or_files": False}
