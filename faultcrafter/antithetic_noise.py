from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Callable, Mapping

import torch


ANTITHETIC_PAIR = "antithetic_pair_v1"
DIAGNOSTIC_FIELDS = (
    "antithetic_noise_executed", "antithetic_noise_shape", "antithetic_noise_batch_samples",
    "antithetic_noise_batch_elements", "antithetic_noise_positive_sha256", "antithetic_noise_negative_sha256",
    "antithetic_noise_time_sha256", "antithetic_noise_rng_after_sha256", "antithetic_noise_positive_objective",
    "antithetic_noise_negative_objective", "antithetic_noise_mean_objective",
    "antithetic_noise_objective_evaluations", "antithetic_noise_backward_passes",
)


def noise_pair_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_noise_pair_policy", "independent")
    if policy not in ("independent", ANTITHETIC_PAIR):
        raise ValueError("unknown training noise pair policy")
    return policy


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class AntitheticNoisePair:
    def __init__(self):
        self.stage = 0
        self.clean = self.noise = self.time = self.rng_after = self.negative = None

    def capture(self, clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor, random: torch.Generator) -> None:
        if self.stage != 0 or clean.ndim < 2 or not clean.numel() or clean.dtype not in (torch.float32, torch.float64):
            raise ValueError("antithetic capture requires one nonempty original batch")
        if noise.shape != clean.shape or noise.dtype != clean.dtype or noise.device != clean.device or time.shape != (len(clean),) or time.device != clean.device or time.dtype != clean.dtype:
            raise ValueError("antithetic noise/time must match the original batch")
        if any(value.requires_grad or not torch.isfinite(value).all() for value in (clean, noise, time)):
            raise ValueError("antithetic inputs must be finite detached tensors")
        self.clean, self.noise, self.time = (value.detach().clone() for value in (clean, noise, time))
        self.rng_after = random.get_state().clone()
        self.stage = 1

    def replay(self, clean: torch.Tensor, random: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stage != 1:
            raise ValueError("antithetic replay requires exactly one original branch")
        if clean.shape != self.clean.shape or clean.dtype != self.clean.dtype or clean.device != self.clean.device or not torch.equal(clean, self.clean):
            raise ValueError("antithetic branches must use the identical source batch")
        if not torch.equal(random.get_state(), self.rng_after):
            raise ValueError("antithetic replay cannot consume additional random draws")
        self.negative = -self.noise
        self.stage = 2
        return self.negative, self.time

    def finish(self, random: torch.Generator, positive: torch.Tensor, negative: torch.Tensor, mean: torch.Tensor) -> dict:
        if self.stage != 2 or not torch.equal(random.get_state(), self.rng_after):
            raise ValueError("antithetic finish requires exactly two branches and unchanged RNG")
        if not torch.equal(self.negative, -self.noise):
            raise ValueError("antithetic negative noise was modified")
        values = [float(value.detach()) for value in (positive, negative, mean)]
        if any(not math.isfinite(value) for value in values):
            raise FloatingPointError("antithetic objectives must be finite")
        self.stage = 3
        return {"antithetic_noise_executed": 1.0, "antithetic_noise_shape": list(self.clean.shape),
                "antithetic_noise_batch_samples": len(self.clean), "antithetic_noise_batch_elements": self.clean.numel(),
                "antithetic_noise_positive_sha256": tensor_digest(self.noise), "antithetic_noise_negative_sha256": tensor_digest(self.negative),
                "antithetic_noise_time_sha256": tensor_digest(self.time), "antithetic_noise_rng_after_sha256": tensor_digest(self.rng_after),
                "antithetic_noise_positive_objective": values[0], "antithetic_noise_negative_objective": values[1],
                "antithetic_noise_mean_objective": values[2], "antithetic_noise_objective_evaluations": 2,
                "antithetic_noise_backward_passes": 2}


def backward_with_antithetic_noise(model: torch.nn.Module, evaluate: Callable, pair: AntitheticNoisePair, random: torch.Generator):
    parameters = tuple(model.parameters())
    if not parameters or any(not parameter.requires_grad or parameter.grad is not None for parameter in parameters) or pair.stage != 0:
        raise ValueError("antithetic backward requires a fresh pair and cleared full-model gradients")
    positive, positive_metrics = evaluate()
    if positive.ndim != 0 or pair.stage != 1:
        raise ValueError("antithetic first evaluation did not capture its original noise")
    positive_value = positive.detach()
    (positive * 0.5).backward()
    del positive
    negative, negative_metrics = evaluate()
    if negative.ndim != 0 or pair.stage != 2 or positive_metrics.keys() != negative_metrics.keys():
        raise ValueError("antithetic second evaluation or metric schema is invalid")
    negative_value = negative.detach()
    (negative * 0.5).backward()
    mean = (positive_value + negative_value) * 0.5
    metrics = {name: (positive_metrics[name] + negative_metrics[name]) * 0.5 for name in positive_metrics}
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError("antithetic original metrics must be finite scalars")
    metrics.update(pair.finish(random, positive_value, negative_value, mean))
    if "objective" in metrics:
        metrics["objective"] = float(mean)
    return mean, metrics


class AntitheticNoiseActivity:
    def __init__(self, model: torch.nn.Module, generator: Mapping[str, Any]):
        parameters = tuple(model.parameters())
        if noise_pair_policy(generator) != ANTITHETIC_PAIR or not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("antithetic activity requires an active full-model policy")
        self.metadata_value = {"policy": ANTITHETIC_PAIR, "parameter_tensors": len(parameters),
            "parameter_elements": sum(parameter.numel() for parameter in parameters), "parameter_dtype": str(parameters[0].dtype),
            "branches": ["original_noise", "negated_original_noise"], "branch_backward_weights": [0.5, 0.5],
            "same_source_condition_identity_and_time": True, "additional_torch_random_draws": 0,
            "clip_optimizer_and_ema_updates_per_pair": 1, "source_validation_objective_changed": False,
            "noise_time_rng_sha256_trace_saved": True, "historical_gradient_vectors_saved": False,
            "student_parameter_trajectory_saved": False, "equal_compute_claimed": False,
            "cnn_non_degradation_guaranteed": False, "extra_runtime_parameters_or_files": False}
        self.rows = []
        self.source_samples = self.source_elements = 0

    def metadata(self) -> dict:
        return copy.deepcopy(self.metadata_value)

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1 or components.get("antithetic_noise_executed") != 1.0:
            raise ValueError("antithetic activity requires every consecutive completed update")
        row = {"step": step, **copy.deepcopy({name: components[name] for name in DIAGNOSTIC_FIELDS})}
        shape = row["antithetic_noise_shape"]
        if not isinstance(shape, list) or len(shape) < 2 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("antithetic trace requires a nonempty batch shape")
        if shape[0] != row["antithetic_noise_batch_samples"] or math.prod(shape) != row["antithetic_noise_batch_elements"]:
            raise ValueError("antithetic trace exposure differs from its batch shape")
        if row["antithetic_noise_objective_evaluations"] != 2 or row["antithetic_noise_backward_passes"] != 2:
            raise ValueError("antithetic trace requires two complete original objectives and backwards")
        for name, value in row.items():
            if name.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError("antithetic trace contains invalid SHA256")
            if name.endswith("_objective") and not math.isfinite(value):
                raise ValueError("antithetic trace contains a nonfinite objective")
        self.rows.append(row)
        self.source_samples += row["antithetic_noise_batch_samples"]
        self.source_elements += row["antithetic_noise_batch_elements"]
        return {"antithetic_noise_total_steps": step, "antithetic_noise_total_objective_evaluations": 2 * step,
                "antithetic_noise_total_backward_passes": 2 * step,
                "antithetic_noise_total_source_samples": self.source_samples, "antithetic_noise_total_source_elements": self.source_elements,
                "antithetic_noise_total_denoising_samples": 2 * self.source_samples, "antithetic_noise_total_denoising_elements": 2 * self.source_elements}

    def audit(self) -> dict:
        return {"metadata": self.metadata(), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_gradient_vectors_independently_reconstructed": False, "extra_runtime_parameters_or_files": False}
