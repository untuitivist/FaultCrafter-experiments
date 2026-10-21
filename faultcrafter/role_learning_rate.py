from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import torch

from faultcrafter.compact_inference import is_device_lora


UNIFORM_LEARNING_RATE = "uniform"
SHARED_SLOW_ADAPTATION = "shared_slow_adaptation_v1"
ADAPTATION = "adaptation"
SHARED_BACKBONE = "shared_backbone"
SHARED_MULTIPLIER = 0.1
IDENTITIES = ("fault_embedding.weight", "fault_group_embedding.weight")


def role_learning_rate_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_learning_rate_policy", UNIFORM_LEARNING_RATE)
    if policy not in (UNIFORM_LEARNING_RATE, SHARED_SLOW_ADAPTATION):
        raise ValueError("unknown parameter-role learning rate policy")
    return policy


def role_parameter_groups(named_parameters: Iterable[tuple[str, torch.nn.Parameter]], generator: Mapping[str, Any]) -> list[dict]:
    if role_learning_rate_policy(generator) != SHARED_SLOW_ADAPTATION:
        raise ValueError("parameter role partition requires an explicit active policy")
    named = tuple(named_parameters)
    names = [name for name, _ in named]
    if not named or len(set(names)) != len(named) or len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("role partition requires nonempty unique parameter names and tensors")
    if any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("role partition requires every original parameter trainable")
    if any(name not in names for name in IDENTITIES) or not any(is_device_lora(name) for name in names):
        raise ValueError("role partition requires device LoRA and both original identity banks")
    base_rate = float(generator["learning_rate"])
    weight_decay = float(generator["weight_decay"])
    if not math.isfinite(base_rate) or base_rate <= 0.0 or not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("role partition requires finite positive LR and nonnegative weight decay")
    groups = []
    for role, multiplier in ((ADAPTATION, 1.0), (SHARED_BACKBONE, SHARED_MULTIPLIER)):
        selected = [(name, parameter) for name, parameter in named if (is_device_lora(name) or name in IDENTITIES) == (role == ADAPTATION)]
        if not selected:
            raise ValueError("both parameter roles must be nonempty")
        groups.append({"params": [parameter for _, parameter in selected], "parameter_names": [name for name, _ in selected],
                       "parameter_role": role, "learning_rate_multiplier": multiplier,
                       "lr": base_rate * multiplier, "weight_decay": weight_decay})
    return groups


def role_learning_rate_floors(generator: Mapping[str, Any]) -> float | list[float]:
    minimum = float(generator["minimum_learning_rate"])
    if role_learning_rate_policy(generator) == UNIFORM_LEARNING_RATE:
        return minimum
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("role LR floors must be finite and positive")
    return [minimum, minimum * SHARED_MULTIPLIER]


def apply_warmup_learning_rate(optimizer: torch.optim.Optimizer, warmup_rate: float, generator: Mapping[str, Any]) -> None:
    if role_learning_rate_policy(generator) == UNIFORM_LEARNING_RATE:
        for group in optimizer.param_groups:
            group["lr"] = warmup_rate
        return
    if not math.isfinite(warmup_rate) or warmup_rate <= 0.0:
        raise ValueError("role warmup LR must be finite and positive")
    if [(group.get("parameter_role"), group.get("learning_rate_multiplier")) for group in optimizer.param_groups] != [(ADAPTATION, 1.0), (SHARED_BACKBONE, SHARED_MULTIPLIER)]:
        raise ValueError("role warmup requires the declared ordered optimizer groups")
    for group in optimizer.param_groups:
        group["lr"] = warmup_rate * group["learning_rate_multiplier"]


class RoleLearningRateActivity:
    def __init__(self, optimizer: torch.optim.Optimizer, generator: Mapping[str, Any]):
        if role_learning_rate_policy(generator) != SHARED_SLOW_ADAPTATION or not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("role LR activity requires its declared native AdamW policy")
        if [(group.get("parameter_role"), group.get("learning_rate_multiplier")) for group in optimizer.param_groups] != [(ADAPTATION, 1.0), (SHARED_BACKBONE, SHARED_MULTIPLIER)]:
            raise ValueError("role LR activity requires the declared ordered optimizer groups")
        roles = {}
        for group in optimizer.param_groups:
            if len(group["parameter_names"]) != len(group["params"]) or any(not parameter.requires_grad for parameter in group["params"]):
                raise ValueError("role LR activity parameter names or trainability differ")
            roles[group["parameter_role"]] = {"parameter_names": list(group["parameter_names"]), "parameter_tensors": len(group["params"]),
                "parameter_elements": sum(parameter.numel() for parameter in group["params"]), "learning_rate_multiplier": group["learning_rate_multiplier"]}
        self.metadata_value = {"policy": SHARED_SLOW_ADAPTATION, "shared_backbone_multiplier": SHARED_MULTIPLIER, "adaptation_multiplier": 1.0,
            "roles": roles, "warmup_and_plateau_floors_preserve_relative_rates": True, "weight_decay_coefficient_changed": False,
            "shared_decoupled_decay_movement_also_scaled_by_lr": True, "all_original_parameters_trainable": True,
            "original_global_adamw_clock": True, "source_sampling_forward_loss_formulas_or_inference_changed": False,
            "identical_model_and_batch_gradients_unchanged": True, "post_update_gradient_trajectory_unchanged_claimed": False,
            "legacy_learning_rate_log_is_post_validation_adaptation_rate": True, "activity_rates_are_applied_before_validation_scheduler": True,
            "base_learning_rate": float(generator["learning_rate"]), "minimum_learning_rates": role_learning_rate_floors(generator),
            "extra_runtime_parameters_or_files": False}
        self.steps = 0
        self.sums = {ADAPTATION: 0.0, SHARED_BACKBONE: 0.0}
        self.minimums = {ADAPTATION: float("inf"), SHARED_BACKBONE: float("inf")}
        self.maximums = {ADAPTATION: 0.0, SHARED_BACKBONE: 0.0}
        self.schedule_hash = hashlib.sha256()

    def metadata(self) -> dict:
        return copy.deepcopy(self.metadata_value)

    def record(self, optimizer: torch.optim.Optimizer, step: int) -> dict:
        if type(step) is not int or step != self.steps + 1:
            raise ValueError("role LR observations require every consecutive optimizer update")
        if [group.get("parameter_role") for group in optimizer.param_groups] != [ADAPTATION, SHARED_BACKBONE]:
            raise ValueError("role LR optimizer groups changed")
        rates = {group["parameter_role"]: float(group["lr"]) for group in optimizer.param_groups}
        if any(not math.isfinite(value) or value <= 0.0 for value in rates.values()):
            raise ValueError("both parameter roles must have positive finite applied LR")
        if not math.isclose(rates[SHARED_BACKBONE], rates[ADAPTATION] * SHARED_MULTIPLIER, rel_tol=1e-14, abs_tol=0.0):
            raise ValueError("applied role learning rate ratio differs from the declaration")
        for role, value in rates.items():
            self.sums[role] += value
            self.minimums[role] = min(self.minimums[role], value)
            self.maximums[role] = max(self.maximums[role], value)
        self.schedule_hash.update(json.dumps([step, rates[ADAPTATION].hex(), rates[SHARED_BACKBONE].hex()], separators=(",", ":")).encode("ascii") + b"\n")
        self.steps = step
        return {"role_learning_rate_executed": 1.0, "role_learning_rate_total_steps": step,
                "role_learning_rate_schedule_sha256": self.schedule_hash.hexdigest(),
                **{"role_learning_rate_" + role: value for role, value in rates.items()},
                **{"role_learning_rate_" + role + "_sum": value for role, value in self.sums.items()},
                **{"role_learning_rate_" + role + "_minimum_applied": value for role, value in self.minimums.items()},
                **{"role_learning_rate_" + role + "_maximum_applied": value for role, value in self.maximums.items()},
                **{"role_learning_rate_" + role + "_positive_steps": step for role in rates}}
