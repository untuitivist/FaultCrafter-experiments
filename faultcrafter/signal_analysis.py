from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DEFAULT_MAXIMUM_FREQUENCY_HZ = 2_000.0
BJTU_BEARING_DIAGNOSTICS = {
    "bearing_model": "HRB 352213",
    "shaft_frequency_label": "轴箱转频估计",
    "source_parameter": "drive_frequency_hz",
    "shaft_frequency_ratio": 16.0 / 107.0,
    "maximum_frequency_hz": 800,
    "orders": {"BPFI": 10.929, "BPFO": 8.071, "BSF": 3.164, "FTF": 0.425},
    "fault_order_map": {
        "inner_race": "BPFI", "outer_race": "BPFO",
        "rolling_element": "BSF", "cage": "FTF",
    },
    "assumption": "按说明书 16/107 齿数比，由驱动频率估算轴箱转频；精确值应由 CH10 实测转速校核。",
}


def _amplitude_spectrum(
    values: np.ndarray,
    sample_rate_hz: float,
    maximum_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values)
    window = np.hanning(values.size)
    scale = float(np.sum(window))
    spectrum = np.abs(np.fft.rfft(centered * window)) * (2.0 / scale)
    if spectrum.size:
        spectrum[0] *= 0.5
        if values.size % 2 == 0:
            spectrum[-1] *= 0.5
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate_hz)
    keep = frequencies <= min(maximum_frequency_hz, sample_rate_hz / 2.0)
    return frequencies[keep], spectrum[keep]


def _envelope(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    spectrum = np.fft.fft(centered)
    multiplier = np.zeros(values.size, dtype=np.float64)
    multiplier[0] = 1.0
    if values.size % 2 == 0:
        multiplier[1 : values.size // 2] = 2.0
        multiplier[values.size // 2] = 1.0
    else:
        multiplier[1 : (values.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def _spectrogram(
    values: np.ndarray,
    sample_rate_hz: float,
    maximum_frequency_hz: float,
) -> dict[str, Any]:
    target = max(128, values.size // 4)
    segment_length = min(2_048, 2 ** int(np.floor(np.log2(target))))
    segment_length = min(segment_length, values.size)
    hop = max(1, segment_length // 4)
    starts = list(range(0, values.size - segment_length + 1, hop))
    final_start = values.size - segment_length
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    window = np.hanning(segment_length)
    spectra = []
    for start in starts:
        segment = values[start : start + segment_length]
        segment = (segment - np.mean(segment)) * window
        spectra.append(np.abs(np.fft.rfft(segment)))
    matrix = np.stack(spectra, axis=1)
    frequencies = np.fft.rfftfreq(segment_length, d=1.0 / sample_rate_hz)
    keep = frequencies <= min(maximum_frequency_hz, sample_rate_hz / 2.0)
    selected = np.flatnonzero(keep)
    if selected.size > 128:
        positions = np.linspace(0, selected.size - 1, 128).astype(np.int64)
        selected = selected[positions]
    matrix = matrix[selected]
    reference = max(float(np.max(matrix)), np.finfo(np.float64).eps)
    power_db = 20.0 * np.log10(np.maximum(matrix, reference * 1e-6) / reference)
    times = (np.asarray(starts, dtype=np.float64) + segment_length / 2.0) / sample_rate_hz
    return {
        "time_seconds": times.tolist(),
        "frequency_hz": frequencies[selected].tolist(),
        "amplitude_db": power_db.tolist(),
        "segment_length": segment_length,
    }


def _bearing_markers(
    diagnostics: Mapping[str, Any] | None,
    fault_id: str,
    condition_parameters: Mapping[str, float],
    envelope_frequencies: np.ndarray,
    envelope_amplitudes: np.ndarray,
    resolution_hz: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not diagnostics:
        return [], None
    source_parameter = str(diagnostics["source_parameter"])
    shaft_frequency_hz = (
        float(condition_parameters[source_parameter])
        * float(diagnostics["shaft_frequency_ratio"])
    )
    orders = {str(key): float(value) for key, value in diagnostics["orders"].items()}
    selected_code = None
    for fragment, code in diagnostics["fault_order_map"].items():
        if str(fragment) in fault_id:
            selected_code = str(code)
            break
    markers: list[dict[str, Any]] = []
    for code, order in orders.items():
        markers.append(
            {
                "code": code,
                "order": order,
                "harmonic": 1,
                "frequency_hz": order * shaft_frequency_hz,
                "selected": code == selected_code,
            }
        )
    selected_frequency_hz = None
    match = None
    if selected_code is not None:
        selected_order = orders[selected_code]
        selected_frequency_hz = selected_order * shaft_frequency_hz
        for harmonic in range(2, 5):
            frequency = selected_frequency_hz * harmonic
            if frequency <= float(envelope_frequencies[-1]):
                markers.append(
                    {
                        "code": selected_code,
                        "order": selected_order * harmonic,
                        "harmonic": harmonic,
                        "frequency_hz": frequency,
                        "selected": True,
                    }
                )
        tolerance_hz = max(2.0 * resolution_hz, 0.05 * selected_frequency_hz)
        local = np.abs(envelope_frequencies - selected_frequency_hz) <= tolerance_hz
        if np.any(local):
            local_indices = np.flatnonzero(local)
            peak_index = int(local_indices[np.argmax(envelope_amplitudes[local])])
            peak_frequency_hz = float(envelope_frequencies[peak_index])
            wider = np.abs(envelope_frequencies - selected_frequency_hz) <= 5.0 * tolerance_hz
            background = envelope_amplitudes[wider & ~local]
            if not background.size:
                background = envelope_amplitudes
            floor = max(float(np.median(background)), np.finfo(np.float64).eps)
            prominence_db = 20.0 * np.log10(
                max(float(envelope_amplitudes[peak_index]), floor) / floor
            )
            match = {
                "peak_frequency_hz": peak_frequency_hz,
                "offset_hz": peak_frequency_hz - selected_frequency_hz,
                "tolerance_hz": tolerance_hz,
                "prominence_db": float(prominence_db),
            }
    return markers, {
        "bearing_model": str(diagnostics["bearing_model"]),
        "shaft_frequency_hz": shaft_frequency_hz,
        "shaft_frequency_label": str(diagnostics["shaft_frequency_label"]),
        "selected_code": selected_code,
        "selected_frequency_hz": selected_frequency_hz,
        "match": match,
        "assumption": str(diagnostics["assumption"]),
    }


def analyze_signal(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    fault_id: str,
    condition_parameters: Mapping[str, float],
    bearing_diagnostics: Mapping[str, Any] | None = None,
    maximum_frequency_hz: float = DEFAULT_MAXIMUM_FREQUENCY_HZ,
) -> dict[str, Any]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("signal analysis requires at least eight one-dimensional samples")
    if not np.isfinite(values).all() or not sample_rate_hz > 0.0:
        raise ValueError("signal analysis input must be finite with a positive sample rate")
    frequencies, amplitudes = _amplitude_spectrum(
        values, sample_rate_hz, maximum_frequency_hz
    )
    envelope_frequencies, envelope_amplitudes = _amplitude_spectrum(
        _envelope(values), sample_rate_hz, maximum_frequency_hz
    )
    resolution_hz = sample_rate_hz / values.size
    markers, bearing = _bearing_markers(
        bearing_diagnostics,
        fault_id,
        condition_parameters,
        envelope_frequencies,
        envelope_amplitudes,
        resolution_hz,
    )
    return {
        "frequency_spectrum": {
            "frequency_hz": frequencies.tolist(),
            "amplitude": amplitudes.tolist(),
        },
        "envelope_spectrum": {
            "frequency_hz": envelope_frequencies.tolist(),
            "amplitude": envelope_amplitudes.tolist(),
        },
        "spectrogram": _spectrogram(values, sample_rate_hz, maximum_frequency_hz),
        "diagnostics": {
            "duration_seconds": values.size / sample_rate_hz,
            "frequency_resolution_hz": resolution_hz,
            "recommended_minimum_length": 8_192,
            "maximum_frequency_hz": min(maximum_frequency_hz, sample_rate_hz / 2.0),
            "bearing": bearing,
            "markers": markers,
        },
    }
