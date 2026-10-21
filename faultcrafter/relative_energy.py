from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch


RELATIVE_SOURCE_ENERGY = "batch_normalized_inverse_source_energy_v1"
ROUTE_RELATIVE_SOURCE_ENERGY = "route_mean_inverse_source_energy_v1"
RELATIVE_ENERGY_POLICIES = (RELATIVE_SOURCE_ENERGY, ROUTE_RELATIVE_SOURCE_ENERGY)
PREFIX = "relative_energy_"


def relative_energy_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("relative_energy_reconstruction_policy", "none")
    if policy not in ("none", *RELATIVE_ENERGY_POLICIES):
        raise ValueError("unknown relative source energy reconstruction policy")
    return policy


def relative_energy_metadata(generator: Mapping[str, Any]) -> dict:
    policy = relative_energy_policy(generator)
    if policy == "none":
        raise ValueError("relative source energy metadata requires the active policy")
    if generator.get("group_reconstruction_loss_weight") != 0.1 or generator.get("minimum_noise_fraction") != 0.05:
        raise ValueError("relative source energy retains the original group weight and noise floor")
    if generator.get("direct_x_time_weighting", "inverse_noise_squared") != "inverse_noise_squared" or generator.get("direct_x_residual_policy", "squared") != "squared":
        raise ValueError("relative source energy retains squared residuals and inverse-noise time weights")
    if any(generator.get(name, "none") != "none" for name in (
        "waveform_distribution_policy",
        "source_crop_policy", "solver_state_policy", "training_noise_pair_policy",
    )):
        raise ValueError("relative source energy cannot combine undeclared reconstruction or sampling policies")
    if any(generator.get(name, False) for name in ("resumable_training", "device_balanced_sampling", "relation_balanced_sampling", "relation_coupled_noise_time")):
        raise ValueError("relative source energy retains original finite source exposure")
    metadata = {"policy": policy,
            "formula": "inverse=reciprocal(max(mean_time(clean**2),1/length));weight=detach(inverse/mean_batch(inverse))",
            "scope": "fine_and_group_original_velocity_reconstruction", "group_weight": 0.1,
            "minimum_noise_fraction": 0.05, "clean_target_and_weights_detached": True,
            "energy_statistics_scope": "current_source_parent_only", "batch_weights_mean": 1.0,
            "source_validation": "unchanged_original_objective", "device_sampling_changed": False,
            "effective_sample_and_device_gradient_weights_changed": True,
            "original_flow_matching_risk_equivalence_claimed": False,
            "additional_model_forwards": 0, "additional_source_samples": 0,
            "extra_runtime_parameters_or_files": False}
    if policy == ROUTE_RELATIVE_SOURCE_ENERGY:
        metadata.update({"formula": "inverse=reciprocal(max(mean_time(clean**2),1/length));weight=detach(inverse/mean_same_current_device_route(inverse))",
                         "normalization_scope": "current_batch_device_route", "per_route_weights_mean": 1.0,
                         "per_route_multiplier_mass_preserved": True, "final_reduction": "mean_all_batch_not_mean_devices",
                         "per_route_loss_or_gradient_values_preserved": False, "cross_batch_statistics_used": False})
    return metadata


def relative_energy_weights(clean: torch.Tensor, *, routes: torch.Tensor | None = None,
                            policy: str = RELATIVE_SOURCE_ENERGY) -> tuple[torch.Tensor, torch.Tensor]:
    if policy not in RELATIVE_ENERGY_POLICIES:
        raise ValueError("relative source energy weights require a declared active policy")
    if clean.ndim != 3 or clean.shape[1] != 1 or not len(clean) or not clean.shape[-1]:
        raise ValueError("relative source energy requires nonempty [batch,1,length] clean targets")
    if clean.dtype not in (torch.float32, torch.float64) or not torch.isfinite(clean).all():
        raise ValueError("relative source energy requires finite float32/64 clean targets")
    energy = clean.detach().square().flatten(1).mean(dim=1)
    if not torch.isfinite(energy).all():
        raise ValueError("source energy overflowed")
    inverse = energy.clamp_min(1.0 / clean.shape[-1]).reciprocal()
    if policy == RELATIVE_SOURCE_ENERGY:
        weights = inverse / inverse.mean()
    else:
        if not isinstance(routes, torch.Tensor) or routes.shape != (len(clean),) or routes.device != clean.device or routes.dtype != torch.long or torch.any(routes < 0):
            raise ValueError("route-local energy weights require nonnegative int64 routes on the source device")
        weights = torch.empty_like(inverse)
        for route in torch.unique(routes, sorted=True):
            selected = routes == route
            local_inverse = inverse[selected]
            weights[selected] = local_inverse / local_inverse.mean()
    if not torch.isfinite(weights).all() or not torch.all(weights > 0):
        raise ValueError("relative source energy weights must be finite and positive")
    return weights, energy


def relative_energy_losses(prediction: torch.Tensor, group_prediction: torch.Tensor, clean: torch.Tensor,
                           time: torch.Tensor, routes: torch.Tensor, *,
                           policy: str = RELATIVE_SOURCE_ENERGY) -> tuple[torch.Tensor, torch.Tensor, dict]:
    weights, energy = relative_energy_weights(clean, routes=routes, policy=policy)
    if any(value.shape != clean.shape or value.device != clean.device or value.dtype != clean.dtype or not torch.isfinite(value).all()
           for value in (prediction, group_prediction)):
        raise ValueError("relative source energy requires matching finite predictions")
    if time.shape != (len(clean),) or time.device != clean.device or time.dtype != clean.dtype or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("relative source energy requires matching finite times in [0,1]")
    if routes.shape != time.shape or routes.device != clean.device or routes.dtype != torch.long or torch.any(routes < 0):
        raise ValueError("relative source energy requires one nonnegative int64 device route per parent")
    target = clean.detach()
    sigma = (1.0 - time).clamp_min(0.05)
    fine_errors = (prediction - target).square().flatten(1).mean(dim=1) / sigma.square()
    group_errors = (group_prediction - target).square().flatten(1).mean(dim=1) / sigma.square()
    fine_loss = (weights * fine_errors).mean()
    group_loss = (weights * group_errors).mean()
    if not torch.isfinite(fine_loss) or not torch.isfinite(group_loss):
        raise ValueError("relative source energy reconstruction overflowed")
    diagnostics = {PREFIX + "executed": 1.0, PREFIX + "samples": len(clean), PREFIX + "elements": clean.numel(),
                   PREFIX + "length": clean.shape[-1], PREFIX + "dtype": str(clean.dtype),
                   PREFIX + "routes": routes.detach().cpu().tolist(), PREFIX + "energy_floor": 1.0 / clean.shape[-1]}
    for name, value in (("energies", energy), ("weights", weights), ("times", time),
                        ("fine_errors", fine_errors), ("group_errors", group_errors)):
        diagnostics[PREFIX + name] = value.detach().cpu().tolist()
    for name, value in (("fine_loss", fine_loss), ("group_loss", group_loss),
                        ("original_fine_loss", fine_errors.mean()), ("original_group_loss", group_errors.mean()),
                        ("weight_min", weights.min()), ("weight_max", weights.max()), ("weight_mean", weights.mean())):
        diagnostics[PREFIX + name] = float(value.detach())
    for name, value in (("clean", target), ("fine_prediction", prediction), ("group_prediction", group_prediction)):
        diagnostics[PREFIX + name + "_sha256"] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    if policy == ROUTE_RELATIVE_SOURCE_ENERGY:
        identifiers = torch.unique(routes, sorted=True)
        diagnostics.update({PREFIX + "normalization_scope": "device_route",
                            PREFIX + "normalization_routes": identifiers.detach().cpu().tolist(),
                            PREFIX + "route_counts": [int((routes == route).sum()) for route in identifiers],
                            PREFIX + "route_weight_sums": [float(weights[routes == route].sum()) for route in identifiers]})
    return fine_loss, group_loss, diagnostics


def validate_relative_energy_row(row: dict, *, policy: str = RELATIVE_SOURCE_ENERGY) -> None:
    if policy not in RELATIVE_ENERGY_POLICIES:
        raise ValueError("relative source energy trace requires a declared active policy")
    if row.get(PREFIX + "executed") != 1.0:
        raise ValueError("relative source energy reconstruction was not executed")
    count, length, elements = (row.get(PREFIX + name) for name in ("samples", "length", "elements"))
    if any(type(value) is not int or value <= 0 for value in (count, length, elements)) or elements != count * length:
        raise ValueError("relative source energy sample or element counts differ")
    dtype = row.get(PREFIX + "dtype")
    if dtype not in ("torch.float32", "torch.float64") or row.get(PREFIX + "energy_floor") != 1.0 / length:
        raise ValueError("relative source energy dtype or floor differs")
    epsilon = np.finfo(np.float32 if dtype == "torch.float32" else np.float64).eps
    routes = row.get(PREFIX + "routes", [])
    if len(routes) != count or any(type(route) is not int or route < 0 for route in routes):
        raise ValueError("relative source energy device routes differ")
    arrays = {}
    for name in ("energies", "weights", "times", "fine_errors", "group_errors"):
        values = np.asarray(row.get(PREFIX + name, []), dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("relative source energy trace has invalid per-parent values")
        arrays[name] = values
    if np.any(arrays["times"] > 1) or np.any(arrays["weights"] <= 0):
        raise ValueError("relative source energy times or weights are out of range")
    inverse = np.reciprocal(np.maximum(arrays["energies"], 1.0 / length))
    if policy == RELATIVE_SOURCE_ENERGY:
        if row.get(PREFIX + "normalization_scope", "batch") != "batch":
            raise ValueError("batch-global relative energy cannot accept a route-local trace")
        expected = inverse / inverse.mean()
    else:
        identifiers = sorted(set(routes))
        if row.get(PREFIX + "normalization_scope") != "device_route" or row.get(PREFIX + "normalization_routes") != identifiers:
            raise ValueError("route-local relative energy trace lacks the declared scope or routes")
        if any(type(value) is not int for value in row[PREFIX + "normalization_routes"]):
            raise ValueError("route-local normalization identifiers must be integers")
        counts = [routes.count(route) for route in identifiers]
        if row.get(PREFIX + "route_counts") != counts or any(type(value) is not int for value in row[PREFIX + "route_counts"]):
            raise ValueError("route-local relative energy changes route sample counts")
        sums = np.asarray(row.get(PREFIX + "route_weight_sums", []), dtype=np.float64)
        if sums.shape != (len(identifiers),) or not np.isfinite(sums).all():
            raise ValueError("route-local relative energy trace requires all route weight sums")
        expected = np.empty_like(inverse)
        for position, route in enumerate(identifiers):
            selected = np.asarray(routes) == route
            expected[selected] = inverse[selected] / inverse[selected].mean()
            actual_sum = math.fsum(arrays["weights"][selected].tolist())
            if abs(sums[position] - actual_sum) > 8 * epsilon * counts[position] or abs(sums[position] - counts[position]) > 8 * epsilon * counts[position]:
                raise ValueError("route-local weights do not preserve original route sample mass")
    if not np.allclose(arrays["weights"], expected, rtol=8 * epsilon, atol=0):
        raise ValueError("relative source energy weights differ from their declared formula")
    scalars = {"fine_loss": np.mean(arrays["weights"] * arrays["fine_errors"]),
               "group_loss": np.mean(arrays["weights"] * arrays["group_errors"]),
               "original_fine_loss": arrays["fine_errors"].mean(), "original_group_loss": arrays["group_errors"].mean(),
               "weight_min": arrays["weights"].min(), "weight_max": arrays["weights"].max(), "weight_mean": arrays["weights"].mean()}
    for name, expected_value in scalars.items():
        actual = row.get(PREFIX + name)
        if not isinstance(actual, (float, int)) or not math.isfinite(actual) or abs(actual - expected_value) > 8 * epsilon * max(1.0, abs(expected_value)):
            raise ValueError("relative source energy saved loss or weight summary differs")
    for name in ("clean_sha256", "fine_prediction_sha256", "group_prediction_sha256"):
        value = row.get(PREFIX + name)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("relative source energy trace requires complete waveform hashes")


class RelativeEnergyActivity:
    def __init__(self, generator: Mapping[str, Any]):
        self.metadata = relative_energy_metadata(generator)
        self.rows = []
        self.totals = {"samples": 0, "elements": 0, "nonuniform_steps": 0,
                       "fine_loss_sum": 0.0, "group_loss_sum": 0.0,
                       "original_fine_loss_sum": 0.0, "original_group_loss_sum": 0.0}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("relative source energy requires consecutive complete optimizer updates")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items()
               if name.startswith(PREFIX) and not name.startswith(PREFIX + "total_")})}
        validate_relative_energy_row(row, policy=self.metadata["policy"])
        self.rows.append(row)
        for name in ("samples", "elements"):
            self.totals[name] += row[PREFIX + name]
        self.totals["nonuniform_steps"] += int(row[PREFIX + "weight_min"] != row[PREFIX + "weight_max"])
        for name in ("fine_loss", "group_loss", "original_fine_loss", "original_group_loss"):
            self.totals[name + "_sum"] += row[PREFIX + name]
        return {PREFIX + "total_steps": step, **{PREFIX + "total_" + name: value for name, value in self.totals.items()}}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_student_predictions_or_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}
