from __future__ import annotations

import copy

import torch

from faultcrafter.low_signal_time import low_signal_time_metadata
from faultcrafter.shared_update_cone import value_digest


NATIVE_UPDATE = "native"
LOOKAHEAD_UPDATE = "all_parameters_k5_half_v1"


def lookahead_policy(generator):
    policy = generator.get("lookahead_update_policy", NATIVE_UPDATE)
    if policy not in (NATIVE_UPDATE, LOOKAHEAD_UPDATE):
        raise ValueError("unknown Lookahead update policy")
    return policy


def lookahead_metadata(generator):
    if lookahead_policy(generator) != LOOKAHEAD_UPDATE:
        raise ValueError("Lookahead metadata requires the declared active policy")
    low_signal_time_metadata(generator)
    defaults = {"fault_identity_guidance_schedule": "constant", "trajectory_average_policy": "ema",
                "shared_update_policy": "native", "gradient_clipping_policy": "global",
                "route_optimizer_policy": "native", "role_learning_rate_policy": "uniform",
                "source_episode_policy": "none", "condition_contrast_policy": "none"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("Lookahead cannot combine undeclared optimization mechanisms")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None or generator.get("resumable_training", False):
        raise ValueError("Lookahead requires full-model finite nonresumable training")
    return {"policy": LOOKAHEAD_UPDATE, "interval": 5, "interpolation": 0.5,
            "formula": "delta=fast-slow;interpolated=slow+delta*0.5;slow=fast=interpolated",
            "scope": "all_original_trainable_parameters_including_all_lora_factors_and_identities",
            "ordering": "native_adamw_then_optional_sync_then_original_ema",
            "native_optimizer_state_gradients_and_buffers_unchanged_by_sync": True,
            "ema_and_source_validation_recipe_unchanged": True,
            "additional_optimizer_forward_backward_random_calls": 0,
            "extra_training_parameter_copies": 1, "extra_runtime_parameters_or_files": False,
            "slow_parameters_are_not_checkpoint_or_runtime_state": True,
            "equal_compute_claimed": False}


class LookaheadSynchronizer:
    def __init__(self, model, generator):
        self.metadata = lookahead_metadata(generator)
        self.parameters = tuple(model.named_parameters())
        self.buffers = tuple(model.named_buffers())
        if not self.parameters or any(not value.requires_grad or value.dtype not in (torch.float32, torch.float64) for _, value in self.parameters):
            raise ValueError("Lookahead requires all original float32/64 parameters trainable")
        self.slow = tuple(value.detach().clone() for _, value in self.parameters)
        self.schema = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in self.parameters]
        self.initial_sha256 = value_digest(self.slow)
        self.steps = []
        self.synchronizations = 0

    @torch.no_grad()
    def apply(self, optimizer, step, routes):
        if type(optimizer) is not torch.optim.AdamW or type(step) is not int or step != len(self.steps) + 1:
            raise ValueError("Lookahead requires consecutive native AdamW updates")
        actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len(self.parameters) or {id(value) for value in actual} != {id(value) for _, value in self.parameters}:
            raise ValueError("Lookahead optimizer must contain exactly the full model")
        row = {"step": step, "source_routes": routes.tolist(), "synchronized": step % 5 == 0}
        if row["synchronized"]:
            if any(int(optimizer.state[value].get("step", -1)) != step for _, value in self.parameters):
                raise ValueError("Lookahead must follow the matching native AdamW update")
            row.update({"native_fast_sha256": value_digest(tuple(value for _, value in self.parameters)),
                        "slow_before_sha256": value_digest(self.slow)})
            changed = 0
            for (_, fast), slow in zip(self.parameters, self.slow, strict=True):
                interpolated = slow + (fast - slow) * 0.5
                if not torch.isfinite(interpolated).all():
                    raise FloatingPointError("nonfinite Lookahead interpolation")
                changed += int(not torch.equal(fast, interpolated))
                slow.copy_(interpolated)
                fast.copy_(interpolated)
            self.synchronizations += 1
            row.update({"written_fast_sha256": value_digest(tuple(value for _, value in self.parameters)),
                        "slow_after_sha256": value_digest(self.slow), "changed_parameter_tensors": changed})
            if row["written_fast_sha256"] != row["slow_after_sha256"]:
                raise RuntimeError("Lookahead fast and slow writes disagree")
        row["completed_synchronizations"] = self.synchronizations
        self.steps.append(row)
        return {"lookahead_synchronized": row["synchronized"], "lookahead_completed_synchronizations": self.synchronizations,
                "lookahead_step_evidence": copy.deepcopy(row)}

    def audit(self):
        return {"metadata": copy.deepcopy(self.metadata), "schema": copy.deepcopy(self.schema),
                "initial_slow_sha256": self.initial_sha256, "completed_steps": len(self.steps),
                "completed_synchronizations": self.synchronizations, "steps": copy.deepcopy(self.steps)}
