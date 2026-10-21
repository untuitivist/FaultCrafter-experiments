from __future__ import annotations

import math

import torch

from faultcrafter.normalized_spectrum import NORMALIZED_SPECTRAL_DISTANCE


MULTIRESOLUTION_STFT = "multi_resolution_stft_v1"
ORIGINAL_SPECTRUM = "full_parent_normalized_log_power"
FFT_WINDOWS = (128, 512, 2048)
MAGNITUDE_FLOOR = 1e-7


def spectral_reconstruction_policy(generator: dict) -> str:
    policy = generator.get("spectral_reconstruction_policy", ORIGINAL_SPECTRUM)
    if policy not in (ORIGINAL_SPECTRUM, MULTIRESOLUTION_STFT, NORMALIZED_SPECTRAL_DISTANCE):
        raise ValueError("unknown spectral reconstruction policy")
    return policy


def resolution_frames(length: int, batch_size: int) -> dict[int, int]:
    if type(length) is not int or type(batch_size) is not int or length < FFT_WINDOWS[0] or batch_size <= 0:
        raise ValueError("STFT requires positive batch size and at least128 real samples")
    return {window: batch_size * (1 + (length - window) // (window // 4)) if window <= length else 0 for window in FFT_WINDOWS}


def multiresolution_spectral_metadata(weight: float) -> dict:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("multiresolution spectral supervision requires a positive finite original weight")
    return {"policy": MULTIRESOLUTION_STFT, "fft_windows_samples": list(FFT_WINDOWS), "hop_divisor": 4,
            "window": "periodic_hann", "center": False, "pad_to_longer_window": False,
            "magnitude_floor": MAGNITUDE_FLOOR, "spectral_loss_weight": weight,
            "loss": "per_example_frobenius_convergence_plus_log_magnitude_l1_then_resolution_and_batch_mean",
            "resolution_selection": "only_windows_not_longer_than_real_parent",
            "replaces_original_spectral_term": True, "train_and_source_validation_same_objective": True,
            "same_effective_gradient_scale_as_original_claimed": False, "extra_runtime_parameters_or_files": False}


def multiresolution_spectral_errors(prediction: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[1] != 1:
        raise ValueError("multiresolution spectrum needs matching [batch,1,length] signals")
    if prediction.dtype != reference.dtype or prediction.dtype not in (torch.float32, torch.float64) or prediction.device != reference.device:
        raise ValueError("multiresolution spectrum requires matching float32/float64 dtype and device")
    frames = resolution_frames(prediction.shape[-1], len(prediction))
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("multiresolution spectrum requires finite real and predicted signals")
    errors = []
    convergences = []
    logarithms = []
    for window_length, frame_count in frames.items():
        if not frame_count:
            continue
        window = torch.hann_window(window_length, periodic=True, dtype=prediction.dtype, device=prediction.device)
        settings = {"n_fft": window_length, "hop_length": window_length // 4, "win_length": window_length,
                    "window": window, "center": False, "normalized": False, "onesided": True, "return_complex": True}
        predicted = torch.stft(prediction[:, 0], **settings).abs()
        real = torch.stft(reference[:, 0], **settings).abs()
        convergence = torch.linalg.vector_norm((predicted - real).flatten(1), dim=1) / torch.linalg.vector_norm(real.flatten(1), dim=1).clamp_min(MAGNITUDE_FLOOR)
        logarithm = (predicted.clamp_min(MAGNITUDE_FLOOR).log() - real.clamp_min(MAGNITUDE_FLOOR).log()).abs().flatten(1).mean(dim=1)
        errors.append(convergence + logarithm)
        convergences.append(convergence)
        logarithms.append(logarithm)
    per_example = torch.stack(errors, dim=1).mean(dim=1, keepdim=True)
    diagnostics = {"multiresolution_stft_policy_executed": 1.0, "multiresolution_stft_resolution_count": len(errors),
                   "multiresolution_stft_convergence": float(torch.stack(convergences).mean().detach()),
                   "multiresolution_stft_log_magnitude": float(torch.stack(logarithms).mean().detach()),
                   **{f"multiresolution_stft_frames_{window}": count for window, count in frames.items()}}
    return per_example, diagnostics
