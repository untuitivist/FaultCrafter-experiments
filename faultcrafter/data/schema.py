from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Channel:
    name: str
    source_key: str
    location: str
    quantity: str = "acceleration"


@dataclass(frozen=True, slots=True)
class SignalRecord:
    adapter_id: str
    dataset_id: str
    device_id: str
    record_id: str
    acquisition_id: str
    path: Path
    original_sample_rate_hz: int
    sample_rate_hz: int
    original_num_samples: int
    num_samples: int
    channels: tuple[Channel, ...]
    condition_label: str
    condition_parameters: tuple[tuple[str, float], ...]
    fault_label: str
    fault_group: str
    fault_parameters: tuple[tuple[str, str | float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UnifiedSample:
    signal: np.ndarray
    sample_id: str
    dataset_id: str
    device_id: str
    record_id: str
    acquisition_id: str
    channel: str
    sample_rate_hz: int
    start: int
    condition_label: str
    condition_parameters: tuple[tuple[str, float], ...]
    fault_label: str
    fault_group: str
    fault_parameters: tuple[tuple[str, str | float], ...]

    def __post_init__(self) -> None:
        if self.signal.dtype != np.float32 or self.signal.ndim != 2:
            raise ValueError("signal must be a two-dimensional float32 array")
        if self.signal.shape[0] != 1:
            raise ValueError("a unified sample must contain exactly one channel")
        if not np.isfinite(self.signal).all():
            raise ValueError("signal contains non-finite values")
