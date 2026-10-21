from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
import torch


DETACHED_SOLVER_STATE = "detached_source_predictor_reconstruction_v1"
PREFIX = "solver_state_"


def solver_state_policy(generator: Mapping) -> str:
    policy = generator.get("solver_state_policy", "none")
    if policy not in ("none", DETACHED_SOLVER_STATE):
        raise ValueError("unknown source solver state policy")
    return policy


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceSolverStep:
    sampling_steps: int
    terminal_time: float
    minimum_noise_fraction: float
    weight: float

    @classmethod
    def from_config(cls, config: Mapping):
        generator = config["generator"]
        if solver_state_policy(generator) != DETACHED_SOLVER_STATE:
            return None
        generation = config["generation"]
        steps, terminal = generation["sampling_steps"], generation["terminal_time"]
        floor, weight = generator["minimum_noise_fraction"], generator["group_reconstruction_loss_weight"]
        if type(steps) is not int or steps < 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (terminal, floor, weight)):
            raise ValueError("solver state requires a valid finite original generation recipe")
        if not 0 < generator["minimum_time"] < generator["maximum_time"] <= terminal < 1 or not 0 < floor <= 1 or weight <= 0:
            raise ValueError("solver state requires source times within the original terminal time and positive weights")
        if any(generator.get(name, False) for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight", "relation_balanced_sampling", "resumable_training", "waveform_distribution_policy")):
            raise ValueError("solver state cannot combine undeclared auxiliary mechanisms")
        return cls(steps, float(terminal), float(floor), float(weight))

    def metadata(self) -> dict:
        return {"policy": DETACHED_SOLVER_STATE, "sampling_steps": self.sampling_steps, "terminal_time": self.terminal_time,
                "nominal_increment": self.terminal_time / self.sampling_steps, "minimum_noise_fraction": self.minimum_noise_fraction,
                "weight": self.weight, "weight_source": "existing_group_reconstruction_loss_weight",
                "proposal": "detach(state+(min(time+terminal/steps,terminal)-time)*(detach(prediction)-state)/(1-time))",
                "loss": "mean(sample_clean_mse/max(1-next_time,minimum_noise_fraction)^2)",
                "gradient_scope": "future_forward_only_no_predictor_state_gradient", "source_validation": "unchanged_original_objective",
                "extra_random_draws": 0, "additional_source_waveforms": 0, "extra_forward_per_training_batch": 1,
                "activation_checkpoint": "non_reentrant", "full_heun_trajectory_or_corrector_state_claimed": False,
                "extra_runtime_parameters_or_files": False}

    def propose(self, state: torch.Tensor, prediction: torch.Tensor, time: torch.Tensor, routes: torch.Tensor):
        if state.shape != prediction.shape or state.ndim != 3 or state.shape[1] != 1 or not len(state) or not state.shape[-1]:
            raise ValueError("solver state requires equal nonempty [batch,1,length] tensors")
        if state.dtype not in (torch.float32, torch.float64) or state.dtype != prediction.dtype or state.dtype != time.dtype or state.device != prediction.device or state.device != time.device:
            raise ValueError("solver state precision or device differs")
        if time.shape != (len(state),) or routes.shape != time.shape or routes.dtype != torch.long or torch.any(routes < 0):
            raise ValueError("solver state times or device routes are invalid")
        if any(not torch.isfinite(value).all() for value in (state, prediction, time)) or torch.any((time < 0) | (time > self.terminal_time)):
            raise ValueError("solver state inputs must be finite and inside the generation interval")
        with torch.no_grad():
            current_time = time.detach()
            next_time = (current_time + self.terminal_time / self.sampling_steps).clamp_max(self.terminal_time)
            velocity = (prediction.detach() - state.detach()) / (1.0 - current_time[:, None, None])
            proposal = state.detach() + (next_time - current_time)[:, None, None] * velocity
        if not torch.isfinite(proposal).all():
            raise ValueError("solver state proposal is not finite")
        return proposal, next_time, {PREFIX + "executed": 1.0, PREFIX + "samples": len(state), PREFIX + "elements": state.numel(),
                PREFIX + "length": state.shape[-1], PREFIX + "routes": routes.detach().cpu().tolist(), PREFIX + "dtype": str(state.dtype),
                PREFIX + "times": current_time.cpu().tolist(), PREFIX + "next_times": next_time.cpu().tolist(),
                PREFIX + "advanced_samples": int((next_time > current_time).sum()), PREFIX + "input_sha256": tensor_digest(state),
                PREFIX + "first_prediction_sha256": tensor_digest(prediction), PREFIX + "proposal_sha256": tensor_digest(proposal)}

    def loss(self, prediction: torch.Tensor, clean: torch.Tensor, time: torch.Tensor):
        if prediction.shape != clean.shape or prediction.ndim != 3 or prediction.shape[1] != 1 or time.shape != (len(prediction),):
            raise ValueError("solver state reconstruction shapes differ")
        if any(not torch.isfinite(value).all() for value in (prediction, clean, time)) or torch.any((time < 0) | (time > self.terminal_time)):
            raise ValueError("solver state reconstruction inputs are invalid")
        clean_mse = (prediction - clean.detach()).square().flatten(1).mean(dim=1)
        noise = (1 - time.detach()).clamp_min(self.minimum_noise_fraction)
        value = (clean_mse / noise.square()).mean()
        if not torch.isfinite(value):
            raise ValueError("solver state loss is not finite")
        return value, {PREFIX + "loss": float(value.detach()), PREFIX + "sample_clean_mse": clean_mse.detach().cpu().tolist(),
                       PREFIX + "clean_sha256": tensor_digest(clean), PREFIX + "future_prediction_sha256": tensor_digest(prediction)}


def validate_solver_state_row(row: dict, metadata: dict) -> None:
    count, length, elements = (row.get(PREFIX + name) for name in ("samples", "length", "elements"))
    if any(type(value) is not int or value <= 0 for value in (count, length, elements)) or elements != count * length or row.get(PREFIX + "executed") != 1.0:
        raise ValueError("solver state requires complete positive source exposure")
    routes = row.get(PREFIX + "routes", [])
    if len(routes) != count or any(type(value) is not int or value < 0 for value in routes):
        raise ValueError("solver state source routes are invalid")
    dtype_name = row.get(PREFIX + "dtype")
    if dtype_name not in ("torch.float32", "torch.float64"):
        raise ValueError("solver state precision is not supported")
    dtype = np.float32 if dtype_name == "torch.float32" else np.float64
    arrays = [np.asarray(row.get(PREFIX + name, []), dtype=dtype) for name in ("times", "next_times", "sample_clean_mse")]
    if any(values.shape != (count,) or not np.isfinite(values).all() for values in arrays):
        raise ValueError("solver state numerical trace is invalid")
    times, next_times, clean_mse = arrays
    expected_time = np.minimum(times + dtype(metadata["nominal_increment"]), dtype(metadata["terminal_time"]))
    if not np.array_equal(expected_time, next_times) or np.any((times < 0) | (times > metadata["terminal_time"])) or np.any(clean_mse < 0):
        raise ValueError("solver state time progression or reconstruction errors disagree")
    if row.get(PREFIX + "advanced_samples") != int(np.sum(next_times > times)):
        raise ValueError("solver state advance counter disagrees")
    expected_loss = np.mean(clean_mse.astype(np.float64) / np.maximum(1 - next_times.astype(np.float64), metadata["minimum_noise_fraction"]) ** 2)
    loss = row.get(PREFIX + "loss", float("nan"))
    if isinstance(loss, bool) or not math.isfinite(loss) or loss < 0 or abs(loss - expected_loss) > 16 * np.finfo(dtype).eps * max(1.0, expected_loss):
        raise ValueError("solver state loss differs from recorded source errors")
    for name in ("input_sha256", "first_prediction_sha256", "proposal_sha256", "clean_sha256", "future_prediction_sha256"):
        value = row.get(PREFIX + name)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("solver state requires complete waveform hashes")


class SolverStateActivity:
    def __init__(self, recipe: SourceSolverStep):
        self.metadata = recipe.metadata()
        self.rows = []
        self.totals = {"samples": 0, "elements": 0, "advanced_samples": 0, "nonzero_steps": 0, "loss_sum": 0.0}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("solver state requires consecutive complete optimizer updates")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items() if name.startswith(PREFIX)})}
        validate_solver_state_row(row, self.metadata)
        self.rows.append(row)
        for name in ("samples", "elements", "advanced_samples"):
            self.totals[name] += row[PREFIX + name]
        self.totals["nonzero_steps"] += int(row[PREFIX + "loss"] > 0)
        self.totals["loss_sum"] += row[PREFIX + "loss"]
        return {PREFIX + "total_steps": step, **{PREFIX + "total_" + name: value for name, value in self.totals.items()}}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_predictions_states_or_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}
