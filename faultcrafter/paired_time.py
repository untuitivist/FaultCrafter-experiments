from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping

import torch

from faultcrafter.antithetic_noise import AntitheticNoisePair, tensor_digest
from faultcrafter.low_signal_time import low_signal_time_metadata


REFLECTED_RAW_TIME = "reflected_raw_time_pair_v1"
TRACE_PREFIX = "paired_time_"


def paired_time_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_time_pair_policy", "independent")
    if policy not in ("independent", REFLECTED_RAW_TIME):
        raise ValueError("unknown training time pair policy")
    return policy


def paired_time_metadata(generator: Mapping[str, Any]) -> dict:
    if paired_time_policy(generator) != REFLECTED_RAW_TIME:
        raise ValueError("paired time requires its active policy")
    low_signal_time_metadata(generator)
    defaults = {"matrix_ema_policy": "none", "relative_lora_update_policy": "none",
                "fault_identity_guidance_schedule": "constant", "route_optimizer_policy": "dense_adamw",
                "training_batch_budget_policy": None, "training_device_ids": None,
                "active_lora_device_index": None}
    for name, default in defaults.items():
        if generator.get(name, default) != default:
            raise ValueError("paired time cannot combine undeclared training mechanisms: " + name)
    for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight", "condition_contrast_loss_weight"):
        if generator.get(name, 0):
            raise ValueError("paired time retains the original complete objective")
    return {"policy": REFLECTED_RAW_TIME, "minimum_time": 0.05, "maximum_time": 0.95,
            "paired_raw_time_formula": "float32(1)-raw_time", "transform_after_reflection": "quadratic_low_signal_v1",
            "branch_weights": [0.5, 0.5], "same_source_noise_condition_identity": True,
            "objective_calls_per_update": 2, "backward_calls_per_update": 2, "optimizer_calls_per_update": 1,
            "additional_random_draws": 0, "source_validation": "unchanged_original_single_uniform_branch",
            "finite_discrete_marginal_equality_claimed": False, "gradient_variance_reduction_guaranteed": False,
            "equal_compute_claimed": False, "cnn_non_degradation_guaranteed": False,
            "extra_runtime_parameters_or_files": False}


class PairedTime(AntitheticNoisePair):
    def capture(self, clean, noise, time, random):
        if clean.dtype != torch.float32 or clean.ndim != 3 or clean.shape[1] != 1:
            raise ValueError("paired time requires original float32 [batch,1,length] source")
        super().capture(clean, noise, time, random)
        self.noise_sha256 = tensor_digest(noise)
        self.clean_sha256 = tensor_digest(clean)

    def replay(self, clean, random):
        if self.stage != 1 or clean.dtype != self.clean.dtype or clean.device != self.clean.device or clean.shape != self.clean.shape or not torch.equal(clean, self.clean):
            raise ValueError("paired time requires exactly one identical original source branch")
        if not torch.equal(random.get_state(), self.rng_after):
            raise ValueError("paired time cannot consume extra random draws")
        self.reflected = 1.0 - self.time
        self.stage = 2
        return self.noise, self.reflected

    def finish(self, random, primary, reflected, mean, primary_metrics, reflected_metrics):
        if self.stage != 2 or not torch.equal(random.get_state(), self.rng_after):
            raise ValueError("paired time requires two branches and unchanged RNG")
        if tensor_digest(self.noise) != self.noise_sha256 or tensor_digest(self.clean) != self.clean_sha256:
            raise ValueError("paired time source or original noise was mutated")
        if not torch.equal(self.reflected, 1.0 - self.time):
            raise ValueError("paired raw time was mutated")
        for metrics, time in ((primary_metrics, self.time), (reflected_metrics, self.reflected)):
            if metrics["low_signal_time_raw_sha256"] != tensor_digest(time) or metrics["low_signal_time_noise_sha256"] != self.noise_sha256:
                raise ValueError("paired time must transform the declared raw time with identical noise")
            if metrics["low_signal_time_rng_after_sha256"] != tensor_digest(self.rng_after):
                raise ValueError("paired time diagnostic RNG differs")
        values = [float(value.detach()) for value in (primary, reflected, mean)]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("paired time objectives must be finite")
        self.stage = 3
        return {TRACE_PREFIX + "evidence": {"primary_objective": values[0], "reflected_objective": values[1],
                "mean_objective": values[2], "source_sha256": self.clean_sha256,
                "primary": {name: copy.deepcopy(value) for name, value in primary_metrics.items() if name.startswith("low_signal_time_")},
                "reflected": {name: copy.deepcopy(value) for name, value in reflected_metrics.items() if name.startswith("low_signal_time_")},
                "objective_calls": 2, "backward_calls": 2, "additional_random_draws": 0}}


def backward_with_paired_time(model: torch.nn.Module, evaluate: Callable, pair: PairedTime, random: torch.Generator):
    parameters = tuple(model.parameters())
    if not parameters or any(not parameter.requires_grad or parameter.grad is not None for parameter in parameters) or pair.stage != 0:
        raise ValueError("paired time requires fresh pair and cleared full-model gradients")
    primary, primary_metrics = evaluate()
    if primary.ndim != 0 or pair.stage != 1:
        raise ValueError("paired time primary objective did not capture its draws")
    primary_value = primary.detach()
    (primary * 0.5).backward()
    del primary
    reflected, reflected_metrics = evaluate()
    if reflected.ndim != 0 or pair.stage != 2 or primary_metrics.keys() != reflected_metrics.keys():
        raise ValueError("paired time second objective schema differs")
    reflected_value = reflected.detach()
    (reflected * 0.5).backward()
    del reflected
    mean = (primary_value + reflected_value) * 0.5
    metrics = {}
    for name, value in primary_metrics.items():
        if name.startswith("low_signal_time_"):
            metrics[name] = copy.deepcopy(value)
        elif all(type(item) in (float, int) and math.isfinite(item) for item in (value, reflected_metrics[name])):
            metrics[name] = (value + reflected_metrics[name]) * 0.5
        else:
            raise ValueError("paired time cannot silently average undeclared diagnostic metadata")
    metrics.update(pair.finish(random, primary_value, reflected_value, mean, primary_metrics, reflected_metrics))
    if "objective" in metrics:
        metrics["objective"] = float(mean)
    return mean, metrics


class PairedTimeActivity:
    def __init__(self, model, generator):
        if not tuple(model.parameters()) or any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("paired time requires all model parameters trainable")
        self.metadata_value = paired_time_metadata(generator)
        self.rows = []
        self.samples = self.elements = 0

    def metadata(self):
        return copy.deepcopy(self.metadata_value)

    def record(self, components, step):
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("paired time requires consecutive completed updates")
        row = {"step": step, **copy.deepcopy(components[TRACE_PREFIX + "evidence"])}
        if row["objective_calls"] != 2 or row["backward_calls"] != 2 or row["additional_random_draws"] != 0:
            raise ValueError("paired time requires exactly two original objectives and backwards")
        self.rows.append(row)
        self.samples += row["primary"]["low_signal_time_batch_samples"]
        self.elements += row["primary"]["low_signal_time_batch_elements"]
        return {TRACE_PREFIX + "total_steps": step, TRACE_PREFIX + "total_objective_calls": 2 * step,
                TRACE_PREFIX + "total_backward_calls": 2 * step, TRACE_PREFIX + "total_source_samples": self.samples,
                TRACE_PREFIX + "total_source_elements": self.elements, TRACE_PREFIX + "total_denoising_samples": 2 * self.samples,
                TRACE_PREFIX + "total_denoising_elements": 2 * self.elements}

    def audit(self):
        return {"metadata": self.metadata(), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_gradient_trajectory_reconstruction_claimed": False, "extra_runtime_parameters_or_files": False}
