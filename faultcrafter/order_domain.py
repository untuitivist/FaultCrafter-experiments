from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def envelope_spectrum(
    signals: np.ndarray,
    *,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-sided spectrum of the centered Hilbert envelope."""
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("signals must have shape [batch,length] or [batch,1,length]")
    if not np.isfinite(values).all() or sample_rate_hz <= 0:
        raise ValueError("signals and sample rate must be finite and valid")

    envelope = np.abs(hilbert(values, axis=-1))
    envelope -= envelope.mean(axis=-1, keepdims=True)
    window = np.hanning(values.shape[-1])[None, :]
    magnitude = np.abs(np.fft.rfft(envelope * window, axis=-1))
    frequency_hz = np.fft.rfftfreq(values.shape[-1], d=1.0 / sample_rate_hz)
    return frequency_hz, magnitude


def interpolate_spectra(
    frequency_hz: np.ndarray,
    spectra: np.ndarray,
    query_frequency_hz: np.ndarray,
) -> np.ndarray:
    """Linearly sample each spectrum on an individual frequency grid."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    values = np.asarray(spectra, dtype=np.float64)
    query = np.asarray(query_frequency_hz, dtype=np.float64)
    if frequency.ndim != 1 or values.ndim != 2:
        raise ValueError("frequency must be 1D and spectra must be 2D")
    if values.shape[1] != len(frequency):
        raise ValueError("spectrum width does not match the frequency grid")
    if query.ndim == 1:
        query = np.broadcast_to(query[None, :], (len(values), len(query)))
    if query.shape[0] != len(values):
        raise ValueError("query frequencies must have one row per spectrum")
    if np.any(np.diff(frequency) <= 0) or np.any(query < 0):
        raise ValueError("frequency grids must be non-negative and increasing")

    upper = np.searchsorted(frequency, query, side="right")
    upper = np.clip(upper, 1, len(frequency) - 1)
    lower = upper - 1
    left = frequency[lower]
    right = frequency[upper]
    weight = (query - left) / (right - left)
    rows = np.arange(len(values))[:, None]
    sampled = values[rows, lower] * (1.0 - weight) + values[rows, upper] * weight
    outside = query > frequency[-1]
    sampled[outside] = 0.0
    return sampled


def normalized_log_spectrum(spectra: np.ndarray) -> np.ndarray:
    """Remove overall energy while retaining relative spectral shape."""
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("spectra must be a non-negative 2D array")
    floor = np.maximum(np.median(values, axis=1, keepdims=True) * 1e-6, 1e-12)
    features = np.log(values + floor)
    features -= features.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def absolute_frequency_features(
    frequency_hz: np.ndarray,
    spectra: np.ndarray,
    *,
    maximum_frequency_hz: float,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_frequency_hz <= 0 or feature_count < 2:
        raise ValueError("maximum frequency and feature count must be positive")
    grid = np.linspace(0.0, maximum_frequency_hz, feature_count, dtype=np.float64)
    sampled = interpolate_spectra(frequency_hz, spectra, grid)
    return grid, normalized_log_spectrum(sampled)


def order_features(
    frequency_hz: np.ndarray,
    spectra: np.ndarray,
    speed_hz: np.ndarray,
    *,
    maximum_order: float,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Express envelope spectra on a shared frequency/speed order grid."""
    speed = np.asarray(speed_hz, dtype=np.float64)
    if speed.shape != (len(spectra),) or np.any(speed <= 0):
        raise ValueError("speed_hz must contain one positive value per spectrum")
    if maximum_order <= 0 or feature_count < 2:
        raise ValueError("maximum order and feature count must be positive")
    grid = np.linspace(0.0, maximum_order, feature_count, dtype=np.float64)
    query = speed[:, None] * grid[None, :]
    sampled = interpolate_spectra(frequency_hz, spectra, query)
    return grid, normalized_log_spectrum(sampled)
