from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from faultcrafter.unified_generator import UnifiedSignalGenerator, sample_unified_generator


@dataclass(frozen=True, slots=True)
class ParameterControl:
    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float
    initial: float
    measured: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParameterControl:
        result = cls(
            name=str(value["name"]),
            label=str(value["label"]),
            unit=str(value["unit"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
            step=float(value["step"]),
            initial=float(value["initial"]),
            measured=tuple(float(item) for item in value.get("measured", ())),
        )
        if not result.minimum < result.maximum or result.step <= 0.0:
            raise ValueError(f"invalid parameter control: {result.name}")
        if not result.minimum <= result.initial <= result.maximum:
            raise ValueError(f"parameter initial value is outside limits: {result.name}")
        return result


@dataclass(frozen=True, slots=True)
class FaultControl:
    id: str
    label: str
    group: str
    global_index: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultControl:
        return cls(
            id=str(value["id"]),
            label=str(value["label"]),
            group=str(value["group"]),
            global_index=int(value["global_index"]),
        )


@dataclass(frozen=True, slots=True)
class DeviceControl:
    index: int
    id: str
    label: str
    sensor_label: str
    channel: str
    quantity_label: str
    physical_unit: str
    sample_rate_hz: float
    faults: tuple[FaultControl, ...]
    parameters: tuple[ParameterControl, ...]
    parameter_mean: tuple[float, ...]
    parameter_standard_deviation: tuple[float, ...]
    speed_parameter_index: int
    speed_to_hz: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeviceControl:
        result = cls(
            index=int(value["index"]),
            id=str(value["id"]),
            label=str(value["label"]),
            sensor_label=str(value["sensor_label"]),
            channel=str(value["channel"]),
            quantity_label=str(value["quantity_label"]),
            physical_unit=str(value["physical_unit"]),
            sample_rate_hz=float(value["sample_rate_hz"]),
            faults=tuple(FaultControl.from_dict(item) for item in value["faults"]),
            parameters=tuple(
                ParameterControl.from_dict(item) for item in value["parameters"]
            ),
            parameter_mean=tuple(float(item) for item in value["parameter_mean"]),
            parameter_standard_deviation=tuple(
                float(item) for item in value["parameter_standard_deviation"]
            ),
            speed_parameter_index=int(value["speed_parameter_index"]),
            speed_to_hz=float(value["speed_to_hz"]),
        )
        parameter_count = len(result.parameters)
        if not result.faults or not result.parameters:
            raise ValueError(f"device controls are incomplete: {result.id}")
        if len(result.parameter_mean) != parameter_count or len(
            result.parameter_standard_deviation
        ) != parameter_count:
            raise ValueError(f"device parameter scaling is invalid: {result.id}")
        if any(value <= 0.0 for value in result.parameter_standard_deviation):
            raise ValueError(f"device parameter scaling is non-positive: {result.id}")
        if not 0 <= result.speed_parameter_index < parameter_count:
            raise ValueError(f"device speed parameter index is invalid: {result.id}")
        return result

    @property
    def fault_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.faults)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)


@dataclass(frozen=True, slots=True)
class UnifiedDeploymentManifest:
    model_id: str
    model_label: str
    checkpoint_asset: str
    checkpoint_sha256: str
    condition_dimension: int
    devices: tuple[DeviceControl, ...]
    runtime: Mapping[str, Any]
    architecture: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UnifiedDeploymentManifest:
        if int(value["schema_version"]) != 2:
            raise ValueError("unsupported unified deployment manifest")
        result = cls(
            model_id=str(value["model_id"]),
            model_label=str(value["model_label"]),
            checkpoint_asset=str(value["checkpoint_asset"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]).lower(),
            condition_dimension=int(value["condition_dimension"]),
            devices=tuple(DeviceControl.from_dict(item) for item in value["devices"]),
            runtime=dict(value["runtime"]),
            architecture=dict(value["architecture"]),
            raw=dict(value),
        )
        if len(result.devices) != 4:
            raise ValueError("unified deployment requires exactly four device routes")
        if tuple(device.index for device in result.devices) != tuple(range(4)):
            raise ValueError("device route indices must be contiguous")
        if len({device.id for device in result.devices}) != 4:
            raise ValueError("device route IDs must be unique")
        if len(result.checkpoint_sha256) != 64:
            raise ValueError("checkpoint SHA-256 is invalid")
        minimum = int(result.runtime["minimum_length"])
        maximum = int(result.runtime["maximum_length"])
        default = int(result.runtime["default_length"])
        if not 0 < minimum <= default <= maximum:
            raise ValueError("runtime signal length limits are invalid")
        return result

    @classmethod
    def load(cls, path: str | Path) -> UnifiedDeploymentManifest:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def device(self, device_id: str) -> DeviceControl:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise ValueError(f"unsupported device: {device_id}")


@dataclass(frozen=True, slots=True)
class UnifiedGenerationResult:
    normalized_signal: np.ndarray
    physical_signal: np.ndarray
    predicted_amplitude: float
    sample_rate_hz: int
    physical_unit: str
    device_id: str
    fault_id: str
    condition_parameters: Mapping[str, float]
    seed: int
    elapsed_seconds: float
    compute_device: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fault_group(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "healthy": "normal",
        "ball": "rolling_element",
        "roller": "rolling_element",
        "inner": "inner_race",
        "inner_race_fault": "inner_race",
        "outer": "outer_race",
        "outer_race_fault": "outer_race",
        "cage_fault": "cage",
    }
    return aliases.get(normalized, normalized)


def create_model(
    manifest: UnifiedDeploymentManifest, checkpoint: Mapping[str, Any]
) -> UnifiedSignalGenerator:
    architecture = manifest.architecture
    model = UnifiedSignalGenerator(
        num_faults=len(checkpoint["fault_labels"]),
        num_fault_groups=len(checkpoint["fault_groups"]),
        condition_dimension=manifest.condition_dimension,
        num_devices=len(manifest.devices),
        base_channels=int(architecture["base_channels"]),
        context_dimension=int(architecture["context_dimension"]),
        attention_heads=int(architecture["attention_heads"]),
        style_attention_rank=int(architecture["style_attention_rank"]),
        style_token_count=int(architecture["style_token_count"]),
        lora_rank=int(architecture["device_lora_rank"]),
        attention_radius=int(architecture["attention_radius"]),
        attention_chunk_size=int(architecture["attention_chunk_size"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class UnifiedJiTRuntime:
    def __init__(
        self,
        manifest: UnifiedDeploymentManifest,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        verify_checkpoint: bool = True,
    ) -> None:
        self.manifest = manifest
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if verify_checkpoint and sha256_file(self.checkpoint_path) != manifest.checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 differs from the deployment manifest")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        expected_devices = tuple(item.id for item in manifest.devices)
        if tuple(checkpoint["device_ids"]) != expected_devices:
            raise ValueError("checkpoint device routes differ from the manifest")
        expected_faults = tuple(
            f"{item.id}:{fault.id}"
            for item in manifest.devices
            for fault in item.faults
        )
        if tuple(checkpoint["fault_labels"]) != expected_faults:
            raise ValueError("checkpoint fault routes differ from the manifest")
        self.fault_groups = tuple(str(value) for value in checkpoint["fault_groups"])
        requested_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = requested_device
        self.model = create_model(manifest, checkpoint).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._lock = threading.Lock()

    @property
    def compute_device(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return str(self.device)

    def _condition_vector(
        self, device: DeviceControl, values: Mapping[str, float]
    ) -> np.ndarray:
        if set(values) != set(device.parameter_names):
            raise ValueError("condition parameter names differ from the device route")
        raw = np.asarray([float(values[name]) for name in device.parameter_names], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("condition parameters must be finite")
        for control, value in zip(device.parameters, raw, strict=True):
            if not control.minimum <= float(value) <= control.maximum:
                raise ValueError(
                    f"{control.label} must lie in [{control.minimum}, {control.maximum}] {control.unit}"
                )
        mean = np.asarray(device.parameter_mean, dtype=np.float32)
        deviation = np.asarray(device.parameter_standard_deviation, dtype=np.float32)
        normalized = (raw - mean) / deviation
        padded = np.zeros(self.manifest.condition_dimension, dtype=np.float32)
        padded[: len(normalized)] = normalized
        return padded

    @torch.inference_mode()
    def generate(
        self,
        *,
        device_id: str,
        fault_id: str,
        condition_parameters: Mapping[str, float],
        length: int,
        seed: int,
    ) -> UnifiedGenerationResult:
        return self.generate_many(
            device_id=device_id,
            fault_id=fault_id,
            condition_parameters=condition_parameters,
            length=length,
            seeds=(seed,),
        )[0]

    @torch.inference_mode()
    def generate_many(
        self,
        *,
        device_id: str,
        fault_id: str,
        condition_parameters: Mapping[str, float],
        length: int,
        seeds: Sequence[int],
    ) -> tuple[UnifiedGenerationResult, ...]:
        route = self.manifest.device(device_id)
        if fault_id not in route.fault_ids:
            raise ValueError(f"unsupported fault for {device_id}: {fault_id}")
        minimum = int(self.manifest.runtime["minimum_length"])
        maximum = int(self.manifest.runtime["maximum_length"])
        if not isinstance(length, int) or not minimum <= length <= maximum:
            raise ValueError(f"signal length must lie in [{minimum}, {maximum}]")
        seeds = tuple(seeds)
        if not seeds or len(seeds) > 64:
            raise ValueError("batch seeds must contain between 1 and 64 values")
        if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("batch seeds must be unique")
        normalized = self._condition_vector(route, condition_parameters)
        fault = route.faults[route.fault_ids.index(fault_id)]
        group_name = _canonical_fault_group(fault.group)
        if group_name not in self.fault_groups:
            raise ValueError(f"unsupported fault group: {group_name}")
        batch_size = len(seeds)
        noise_rows = []
        amplitude_noise_rows = []
        for seed in seeds:
            random = torch.Generator(device=self.device).manual_seed(seed)
            noise_rows.append(
                torch.randn((1, 1, length), device=self.device, generator=random)
            )
            amplitude_noise_rows.append(
                torch.randn((1,), device=self.device, generator=random)
            )
        noise = torch.cat(noise_rows, dim=0)
        amplitude_noise = torch.cat(amplitude_noise_rows, dim=0)
        condition_tensor = torch.from_numpy(
            np.repeat(normalized[None], batch_size, axis=0)
        ).to(self.device)
        fault_tensor = torch.full(
            (batch_size,), fault.global_index, dtype=torch.long, device=self.device
        )
        group_tensor = torch.full(
            (batch_size,),
            self.fault_groups.index(group_name),
            dtype=torch.long,
            device=self.device,
        )
        device_tensor = torch.full(
            (batch_size,), route.index, dtype=torch.long, device=self.device
        )
        shaft_speed_hz = float(
            condition_parameters[route.parameter_names[route.speed_parameter_index]]
        ) * route.speed_to_hz
        with self._lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            normalized_tensor, amplitude_tensor, physical_tensor = sample_unified_generator(
                self.model,
                noise,
                condition_tensor,
                fault_tensor,
                group_tensor,
                device_tensor,
                torch.full(
                    (batch_size,),
                    shaft_speed_hz,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.full(
                    (batch_size,),
                    route.sample_rate_hz,
                    dtype=torch.float32,
                    device=self.device,
                ),
                steps=int(self.manifest.runtime["sampling_steps"]),
                terminal_time=float(self.manifest.runtime["terminal_time"]),
                amplitude_noise=amplitude_noise,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
        amplitudes = amplitude_tensor.cpu().numpy().astype(np.float64)
        normalized_signals = normalized_tensor[:, 0].cpu().numpy().astype(np.float32)
        physical_signals = physical_tensor[:, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0.0):
            raise RuntimeError("model predicted an invalid physical amplitude")
        if not np.isfinite(physical_signals).all():
            raise RuntimeError("generated signal contains non-finite values")
        return tuple(
            UnifiedGenerationResult(
                normalized_signal=normalized_signals[index],
                physical_signal=physical_signals[index],
                predicted_amplitude=float(amplitudes[index]),
                sample_rate_hz=int(route.sample_rate_hz),
                physical_unit=route.physical_unit,
                device_id=route.id,
                fault_id=fault.id,
                condition_parameters=dict(condition_parameters),
                seed=seed,
                elapsed_seconds=elapsed / batch_size,
                compute_device=self.compute_device,
            )
            for index, seed in enumerate(seeds)
        )
