from __future__ import annotations

import math

import torch

from faultcrafter.jit import normalized_log_power_spectrum


NORMALIZED_SPECTRAL_DISTANCE = "normalized_spectrum_half_squared_l2_v1"


def normalized_spectral_metadata(weight: float) -> dict:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("normalized spectral distance requires a positive finite weight")
    return {"policy": NORMALIZED_SPECTRAL_DISTANCE, "spectral_loss_weight": weight,
            "features": "unchanged_full_parent_hann_normalized_log_power_spectrum",
            "distance": "half_squared_l2_sum_frequency_then_mean_samples",
            "cosine_equivalence": "only_when_both_feature_vectors_have_unit_norm",
            "zero_vectors": "half_squared_distance_not_one_minus_dot_product",
            "relative_to_original_reduction": "frequency_bin_count_divided_by_two",
            "same_effective_loss_scale_as_original_claimed": False,
            "input_length_invariance_claimed": False,
            "train_and_source_validation_same_objective": True,
            "additional_model_forwards_per_update": 0,
            "extra_runtime_parameters_or_files": False}


def half_squared_spectral_distance(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if prediction.shape != reference.shape or prediction.ndim != 2 or min(prediction.shape) <= 0:
        raise ValueError("spectral vectors require matching nonempty [batch,features] shapes")
    if prediction.dtype != reference.dtype or prediction.dtype not in (torch.float32, torch.float64) or prediction.device != reference.device:
        raise ValueError("spectral vectors require matching float32/float64 dtype and device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("spectral vectors must be finite")
    return 0.5 * (prediction - reference).square().sum(dim=1, keepdim=True)


def normalized_spectral_errors(prediction: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[1] != 1 or not len(prediction) or prediction.shape[-1] < 4:
        raise ValueError("normalized spectral distance needs matching [batch,1,length>=4] signals")
    if prediction.dtype != reference.dtype or prediction.dtype not in (torch.float32, torch.float64) or prediction.device != reference.device:
        raise ValueError("normalized spectral distance requires matching float32/float64 dtype and device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("normalized spectral distance requires finite signals")
    predicted_features = normalized_log_power_spectrum(prediction)
    real_features = normalized_log_power_spectrum(reference)
    per_sample = half_squared_spectral_distance(predicted_features, real_features)
    original_error = (predicted_features - real_features).square().mean()
    bin_count = predicted_features.shape[1]
    diagnostics = {"normalized_spectrum_policy_executed": 1.0,
                   "normalized_spectrum_bin_count": bin_count,
                   "normalized_spectrum_feature_values": len(prediction) * bin_count,
                   "normalized_spectrum_reduction_factor": bin_count / 2.0,
                   "normalized_spectrum_original_mean_error": float(original_error.detach()),
                   "normalized_spectrum_prediction_zero_vectors": int((predicted_features == 0).all(dim=1).sum()),
                   "normalized_spectrum_reference_zero_vectors": int((real_features == 0).all(dim=1).sum())}
    return per_sample, diagnostics
