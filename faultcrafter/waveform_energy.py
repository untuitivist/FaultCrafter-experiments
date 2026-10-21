from __future__ import annotations

import math
from typing import Any, Mapping

import torch


MATCH_NORMALIZED_LOG_RMS = "match_normalized_log_rms"
RMS_FLOOR = 1e-8


def normalized_energy_weight(generator: Mapping[str, Any]) -> float | None:
    policy = generator.get("normalized_energy_policy", "none")
    if not isinstance(policy, str) or policy not in {"none", MATCH_NORMALIZED_LOG_RMS}:
        raise ValueError("unknown normalized waveform energy policy")
    if policy == "none":
        return None
    if generator.get("shared_gradient_policy", "joint") != "joint" or generator.get("direct_x_time_weighting", "inverse_noise_squared") != "inverse_noise_squared" or generator.get("sharpness_training_policy", "standard") != "standard":
        raise ValueError("normalized energy cannot combine projection, time weighting or sharpness policies")
    if any(generator.get(name, False) for name in ("relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference")):
        raise ValueError("normalized energy requires the original sampling and guidance gradients")
    if any(float(generator.get(name, 0.0)) != 0.0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")) or generator.get("active_lora_device_index") is not None:
        raise ValueError("normalized energy requires full-model training without teachers or contrast")
    weight = generator.get("amplitude_loss_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("normalized energy requires the existing positive finite amplitude loss weight")
    return float(weight)


def normalized_energy_metadata(weight: float) -> dict[str, Any]:
    return {"policy": MATCH_NORMALIZED_LOG_RMS, "weight": weight, "weight_source": "existing_amplitude_loss_weight",
            "rms_floor": RMS_FLOOR, "scope": "guided_fine_waveform_only", "target": "same_normalized_source_waveform",
            "formula": "mean_squared_log_rms_error", "source_validation_uses_same_objective": True,
            "forward_and_inference_unchanged": True, "runtime_parameters_added": False}


def normalized_log_rms_loss(prediction: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != clean.shape or prediction.ndim != 3 or prediction.shape[1] != 1 or prediction.shape[-1] < 4 or not len(prediction):
        raise ValueError("normalized energy requires matching nonempty [batch,1,length>=4] signals")
    if prediction.dtype != clean.dtype or prediction.dtype not in (torch.float32, torch.float64) or prediction.device != clean.device:
        raise ValueError("normalized energy requires shared-device float32 or float64 signals")
    if not torch.isfinite(prediction).all() or not torch.isfinite(clean).all():
        raise ValueError("normalized energy signals must be finite")
    predicted_energy = prediction.square().flatten(1).mean(1)
    real_energy = clean.square().flatten(1).mean(1)
    if not torch.isfinite(predicted_energy).all() or not torch.isfinite(real_energy).all():
        raise ValueError("normalized energy computation overflowed")
    predicted_log_rms = 0.5 * predicted_energy.clamp_min(RMS_FLOOR ** 2).log()
    real_log_rms = 0.5 * real_energy.clamp_min(RMS_FLOOR ** 2).log()
    error = predicted_log_rms - real_log_rms
    loss = error.square().mean()
    return loss, {"normalized_energy_loss": float(loss.detach()), "normalized_energy_log_rms_error_mean": float(error.detach().mean()),
                  "normalized_energy_predicted_rms_mean": float(predicted_energy.detach().sqrt().mean()),
                  "normalized_energy_real_rms_mean": float(real_energy.detach().sqrt().mean())}
