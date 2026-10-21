from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


SQUARED_RESIDUAL = "squared"
SCALED_HUBER_VELOCITY = "scaled_huber_velocity_v1"


def robust_reconstruction_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("direct_x_residual_policy", SQUARED_RESIDUAL)
    if policy not in (SQUARED_RESIDUAL, SCALED_HUBER_VELOCITY):
        raise ValueError("unknown direct-x residual policy")
    if policy == SCALED_HUBER_VELOCITY and generator.get("direct_x_time_weighting", "inverse_noise_squared") != "inverse_noise_squared":
        raise ValueError("scaled Huber velocity reconstruction requires original time weighting")
    return policy


def robust_reconstruction_metadata(generator: Mapping[str, Any], device_ids: Sequence[str]) -> dict:
    if robust_reconstruction_policy(generator) != SCALED_HUBER_VELOCITY:
        raise ValueError("robust reconstruction metadata requires an explicit active policy")
    return {"policy": SCALED_HUBER_VELOCITY, "velocity_residual_threshold": 1.0, "minimum_noise_fraction": float(generator["minimum_noise_fraction"]),
            "residual": "prediction_minus_clean_over_original_noise_fraction", "small_error_loss": "squared_velocity_residual",
            "large_error_loss": "twice_absolute_velocity_residual_minus_one", "implementation": "clean_unit_threshold_then_original_sample_mean_over_sigma_squared",
            "scope": "fine_and_group_reconstruction_in_training_and_source_validation", "clean_mse_diagnostic_changed": False,
            "source_noise_time_sampling_changed": False, "other_loss_formulas_or_coefficients_changed": False,
            "signal_clipping_or_sample_selection": False, "mse_conditional_mean_optimality_preserved_claimed": False,
            "device_ids": list(device_ids), "extra_runtime_parameters_or_files": False}


@torch.no_grad()
def observe_robust_reconstruction(generator: Mapping[str, Any], prediction: torch.Tensor, group_prediction: torch.Tensor | None,
                                  clean: torch.Tensor, time: torch.Tensor, device_id: torch.Tensor) -> dict:
    if robust_reconstruction_policy(generator) != SCALED_HUBER_VELOCITY:
        return {}
    if group_prediction is None or group_prediction.shape != prediction.shape or clean.shape != prediction.shape or prediction.ndim < 2 or not len(prediction):
        raise ValueError("robust observations require aligned nonempty fine/group/clean predictions")
    if time.shape != (len(prediction),) or device_id.shape != time.shape or device_id.dtype != torch.long:
        raise ValueError("robust observations require aligned time and device routing")
    sigma = (1.0 - time).clamp_min(float(generator["minimum_noise_fraction"]))
    threshold = sigma.reshape((-1,) + (1,) * (prediction.ndim - 1))
    fine_error, group_error = prediction - clean, group_prediction - clean
    if not torch.isfinite(fine_error).all() or not torch.isfinite(group_error).all() or not torch.isfinite(sigma).all():
        raise ValueError("robust observations require finite errors and noise fractions")
    fine_tail = (fine_error.abs() > threshold).flatten(1).sum(1).cpu().tolist()
    group_tail = (group_error.abs() > threshold).flatten(1).sum(1).cpu().tolist()
    routes = device_id.detach().cpu().tolist()
    if any(route < 0 for route in routes):
        raise ValueError("robust observations contain negative device routes")
    elements = prediction[0].numel()
    counts = {}
    for route, fine, group in zip(routes, fine_tail, group_tail, strict=True):
        bucket = counts.setdefault(str(route), {"samples": 0, "elements": 0, "fine_tail": 0, "group_tail": 0})
        bucket["samples"] += 1
        bucket["elements"] += elements
        bucket["fine_tail"] += fine
        bucket["group_tail"] += group
    return {"robust_reconstruction_executed": 1.0, "robust_reconstruction_batch_samples": len(prediction),
            "robust_reconstruction_batch_elements": prediction.numel(), "robust_reconstruction_batch_fine_tail": sum(fine_tail),
            "robust_reconstruction_batch_group_tail": sum(group_tail), "robust_reconstruction_batch_device_counts": counts,
            "robust_reconstruction_reference_waveform_mse": float((fine_error.square().flatten(1).mean(1) / sigma.square()).mean()),
            "robust_reconstruction_reference_group_mse": float((group_error.square().flatten(1).mean(1) / sigma.square()).mean())}


class RobustReconstructionActivity:
    def __init__(self, generator: Mapping[str, Any], device_ids: Sequence[str]):
        if not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("robust reconstruction activity requires unique devices")
        self.metadata_value = robust_reconstruction_metadata(generator, device_ids)
        self.device_ids = tuple(device_ids)
        self.counts = {device: {"samples": 0, "elements": 0, "fine_tail": 0, "group_tail": 0} for device in device_ids}
        self.steps = 0

    def metadata(self) -> dict:
        return {**self.metadata_value, "device_ids": list(self.device_ids)}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != self.steps + 1 or components.get("robust_reconstruction_executed") != 1.0:
            raise ValueError("robust activity requires every consecutive executed source update")
        counts = components["robust_reconstruction_batch_device_counts"]
        if not counts:
            raise ValueError("robust activity contains no device observations")
        for route, bucket in counts.items():
            index = int(route)
            if str(index) != route or not 0 <= index < len(self.device_ids) or set(bucket) != {"samples", "elements", "fine_tail", "group_tail"}:
                raise ValueError("robust activity has invalid device routes or fields")
            if any(type(value) is not int or value < 0 for value in bucket.values()) or not 0 < bucket["samples"] <= bucket["elements"]:
                raise ValueError("robust activity has invalid sample or element counts")
            if max(bucket["fine_tail"], bucket["group_tail"]) > bucket["elements"]:
                raise ValueError("robust tail counts exceed element counts")
        for field in ("samples", "elements", "fine_tail", "group_tail"):
            if sum(bucket[field] for bucket in counts.values()) != components["robust_reconstruction_batch_" + field]:
                raise ValueError("robust per-device and global counts differ")
        for route, bucket in counts.items():
            destination = self.counts[self.device_ids[int(route)]]
            for field, value in bucket.items():
                destination[field] += value
        self.steps = step
        return {"robust_reconstruction_total_steps": step,
                **{"robust_reconstruction_total_" + field: sum(bucket[field] for bucket in self.counts.values()) for field in ("samples", "elements", "fine_tail", "group_tail")},
                "robust_reconstruction_device_counts": {device: dict(bucket) for device, bucket in self.counts.items()}}
