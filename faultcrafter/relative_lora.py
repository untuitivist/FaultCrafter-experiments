from __future__ import annotations

import copy
import math

import torch

from faultcrafter.shared_update_cone import value_digest
from faultcrafter.unified_generator import RoutedLoRALinear


RELATIVE_LORA = "factor_relative_adamw_v1"


def relative_lora_policy(generator):
    policy = generator.get("relative_lora_update_policy", "none")
    if policy not in ("none", RELATIVE_LORA):
        raise ValueError("unknown relative LoRA update policy")
    return policy


def relative_lora_metadata(generator):
    if relative_lora_policy(generator) != RELATIVE_LORA:
        raise ValueError("relative LoRA update is not enabled")
    defaults = {"lookahead_update_policy": "none", "shared_update_policy": "native",
                "gradient_clipping_policy": "global", "route_optimizer_policy": "native",
                "role_learning_rate_policy": "uniform", "source_episode_policy": "none",
                "condition_contrast_policy": "none", "trajectory_average_policy": "ema"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("relative LoRA cannot combine undeclared update mechanisms")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None or generator.get("resumable_training", False):
        raise ValueError("relative LoRA requires finite full-model training")
    return {"policy": RELATIVE_LORA, "norm": "per_device_per_factor_frobenius",
            "ratio": "parameter_norm/adam_direction_norm_if_both_positive_else_one",
            "formula": "new=old*(1-lr*weight_decay)-lr*ratio*adam_direction",
            "ordering": "capture_then_native_adamw_then_lora_writeback_then_original_ema",
            "decay_outside_trust_ratio": True, "ratio_clipping": False,
            "original_moments_epsilon_clipping_and_shared_updates_retained": True,
            "full_history_or_general_transformation_invariance_claimed": False,
            "extra_runtime_parameters_or_files": False, "extra_training_lora_copies": 1,
            "equal_compute_claimed": False}


class RelativeLoRAUpdate:
    def __init__(self, model, generator):
        self.metadata = relative_lora_metadata(generator)
        self.parameters = tuple(model.named_parameters())
        self.buffers = tuple(model.named_buffers())
        if not self.parameters or any(not value.requires_grad or value.dtype not in (torch.float32, torch.float64) for _, value in self.parameters):
            raise ValueError("relative LoRA requires all original float32/64 parameters trainable")
        self.factors = tuple((f"{name}.{factor}", getattr(module, factor))
                             for name, module in model.named_modules() if isinstance(module, RoutedLoRALinear)
                             for factor in ("lora_down", "lora_up"))
        if not self.factors or any(value.ndim != 3 or value.shape[0] != model.num_devices for _, value in self.factors):
            raise ValueError("relative LoRA requires explicit device factor matrices")
        self.factor_ids = {id(value) for _, value in self.factors}
        self.schema = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in self.factors]
        self.initial_sha256 = value_digest(tuple(value for _, value in self.factors))
        self.steps = []
        self.before = None

    def groups(self, optimizer):
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError("relative LoRA requires native AdamW")
        parameters = [value for group in optimizer.param_groups for value in group["params"]]
        if len(parameters) != len(self.parameters) or {id(value) for value in parameters} != {id(value) for _, value in self.parameters}:
            raise ValueError("relative LoRA optimizer must cover exactly the full model")
        if any(group["amsgrad"] or group["maximize"] or group.get("differentiable", False) for group in optimizer.param_groups):
            raise ValueError("unsupported AdamW mode for relative LoRA")
        return {id(value): group for group in optimizer.param_groups for value in group["params"]}

    @torch.no_grad()
    def capture(self, optimizer, step):
        self.groups(optimizer)
        if type(step) is not int or step != len(self.steps) + 1 or self.before is not None:
            raise ValueError("relative LoRA requires one capture before each consecutive update")
        if any(int(optimizer.state.get(value, {}).get("step", 0)) != step - 1 for _, value in self.parameters):
            raise ValueError("relative LoRA capture must precede the matching native update")
        self.before = tuple(value.detach().clone() for _, value in self.factors)

    @torch.no_grad()
    def apply(self, optimizer, step, routes):
        groups = self.groups(optimizer)
        if self.before is None or type(step) is not int or step != len(self.steps) + 1:
            raise ValueError("relative LoRA has no matching captured update")
        if any(int(optimizer.state.get(value, {}).get("step", -1)) != step for _, value in self.parameters):
            raise ValueError("relative LoRA must follow exactly one matching native update")
        row = {"step": step, "source_routes": routes.tolist(),
               "captured_lora_sha256": value_digest(self.before),
               "native_lora_sha256": value_digest(tuple(value for _, value in self.factors))}
        ratios, writes, zero_norms = [], [], 0
        for (_, parameter), previous in zip(self.factors, self.before, strict=True):
            group, state = groups[id(parameter)], optimizer.state[parameter]
            beta_first, beta_second = group["betas"]
            direction = (state["exp_avg"] / (1 - beta_first ** step)) / (state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta_second ** step) + group["eps"])
            previous_norm = torch.linalg.vector_norm(previous, dim=(-2, -1), keepdim=True)
            direction_norm = torch.linalg.vector_norm(direction, dim=(-2, -1), keepdim=True)
            positive = (previous_norm > 0) & (direction_norm > 0)
            ratio = torch.where(positive, previous_norm / torch.where(direction_norm > 0, direction_norm, torch.ones_like(direction_norm)), torch.ones_like(previous_norm))
            updated = previous * (1 - group["lr"] * group["weight_decay"]) - direction * ratio * group["lr"]
            if not all(torch.isfinite(value).all() for value in (previous, direction, previous_norm, direction_norm, ratio, updated)):
                raise FloatingPointError("nonfinite relative LoRA update")
            writes.append(updated)
            ratios.extend(ratio.flatten().tolist())
            zero_norms += int((~positive).sum())
        for (_, parameter), updated in zip(self.factors, writes, strict=True):
            parameter.copy_(updated)
        row.update({"ratios": ratios, "ratio_min": min(ratios), "ratio_max": max(ratios), "zero_norm_fallbacks": zero_norms,
                    "written_lora_sha256": value_digest(tuple(value for _, value in self.factors))})
        self.before = None
        self.steps.append(row)
        return {"relative_lora_completed_steps": step, "relative_lora_step_evidence": copy.deepcopy(row)}

    def audit(self):
        if self.before is not None:
            raise ValueError("cannot audit an incomplete relative LoRA update")
        return {"metadata": self.metadata, "schema": copy.deepcopy(self.schema), "initial_lora_sha256": self.initial_sha256,
                "completed_steps": len(self.steps), "steps": copy.deepcopy(self.steps)}
