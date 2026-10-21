from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from faultcrafter.diffphysinet import PhysiUNet1D
from faultcrafter.direct_x import heun_direct_x_sample


@dataclass(frozen=True, slots=True)
class ParameterControl:
    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float
    initial: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParameterControl:
        control = cls(
            name=str(value["name"]),
            label=str(value["label"]),
            unit=str(value["unit"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
            step=float(value["step"]),
            initial=float(value["initial"]),
        )
        if not control.name:
            raise ValueError("parameter name must not be empty")
        if not control.minimum < control.maximum:
            raise ValueError(f"invalid parameter interval: {control.name}")
        if control.step <= 0.0:
            raise ValueError(f"parameter step must be positive: {control.name}")
        if not control.minimum <= control.initial <= control.maximum:
            raise ValueError(f"initial parameter is outside its interval: {control.name}")
        return control


@dataclass(frozen=True, slots=True)
class FaultControl:
    id: str
    label: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultControl:
        control = cls(id=str(value["id"]), label=str(value["label"]))
        if not control.id or not control.label:
            raise ValueError("fault ID and label must not be empty")
        return control


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    schema_version: int
    model_id: str
    model_label: str
    checkpoint_asset: str
    source_checkpoint_path: str
    checkpoint_sha256: str
    device: Mapping[str, Any]
    parameters: tuple[ParameterControl, ...]
    faults: tuple[FaultControl, ...]
    runtime: Mapping[str, Any]
    architecture: Mapping[str, Any]
    selection: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeploymentManifest:
        manifest = cls(
            schema_version=int(value["schema_version"]),
            model_id=str(value["model_id"]),
            model_label=str(value["model_label"]),
            checkpoint_asset=str(value["checkpoint_asset"]),
            source_checkpoint_path=str(value["source_checkpoint_path"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]).lower(),
            device=dict(value["device"]),
            parameters=tuple(
                ParameterControl.from_dict(item) for item in value["parameters"]
            ),
            faults=tuple(FaultControl.from_dict(item) for item in value["faults"]),
            runtime=dict(value["runtime"]),
            architecture=dict(value["architecture"]),
            selection=dict(value["selection"]),
        )
        if manifest.schema_version != 1:
            raise ValueError(f"unsupported deployment manifest: {manifest.schema_version}")
        if not manifest.model_id or not manifest.model_label:
            raise ValueError("model ID and label must not be empty")
        if len(manifest.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in manifest.checkpoint_sha256
        ):
            raise ValueError("checkpoint SHA-256 is invalid")
        parameter_names = [item.name for item in manifest.parameters]
        if len(parameter_names) != len(set(parameter_names)) or not parameter_names:
            raise ValueError("parameter names must be unique and non-empty")
        fault_ids = [item.id for item in manifest.faults]
        if len(fault_ids) != len(set(fault_ids)) or not fault_ids:
            raise ValueError("fault IDs must be unique and non-empty")
        minimum_length = int(manifest.runtime["minimum_length"])
        maximum_length = int(manifest.runtime["maximum_length"])
        default_length = int(manifest.runtime["default_length"])
        if not 0 < minimum_length <= default_length <= maximum_length:
            raise ValueError("runtime signal length limits are invalid")
        if int(manifest.runtime["sampling_steps"]) < 2:
            raise ValueError("JiT sampling requires at least two steps")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> DeploymentManifest:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    @property
    def fault_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.faults)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    normalized_signal: np.ndarray
    physical_signal: np.ndarray
    predicted_amplitude: float
    sample_rate_hz: int
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


def _source_support(checkpoint: Mapping[str, Any]) -> np.ndarray:
    initialization = checkpoint.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("checkpoint has no source-free initialization metadata")
    support = np.asarray(
        initialization.get("source_conditions_normalized"), dtype=np.float32
    )
    if support.ndim != 2 or support.shape[0] < 2 or not np.isfinite(support).all():
        raise ValueError("checkpoint source-condition support is invalid")
    return support


def create_source_free_model(
    manifest: DeploymentManifest,
    checkpoint: Mapping[str, Any],
) -> PhysiUNet1D:
    if tuple(checkpoint["fault_labels"]) != manifest.fault_ids:
        raise ValueError("checkpoint fault labels differ from the deployment manifest")
    if tuple(checkpoint["parameter_names"]) != manifest.parameter_names:
        raise ValueError(
            "checkpoint condition parameters differ from the deployment manifest"
        )
    parameter_mean = np.asarray(checkpoint["parameter_mean"], dtype=np.float32)
    parameter_standard_deviation = np.asarray(
        checkpoint["parameter_standard_deviation"], dtype=np.float32
    )
    parameter_count = len(manifest.parameters)
    if parameter_mean.shape != (parameter_count,) or (
        parameter_standard_deviation.shape != (parameter_count,)
    ):
        raise ValueError("checkpoint parameter scaling has an invalid shape")
    if np.any(parameter_standard_deviation <= 0.0):
        raise ValueError("checkpoint parameter standard deviations must be positive")
    support = _source_support(checkpoint)
    if support.shape[1] != parameter_count:
        raise ValueError("checkpoint source support has the wrong parameter count")

    architecture = manifest.architecture
    runtime = manifest.runtime
    model = PhysiUNet1D(
        window_length=int(runtime["window_length"]),
        num_faults=len(manifest.faults),
        num_condition_parameters=parameter_count,
        base_channels=int(architecture["base_channels"]),
        embedding_dimension=int(architecture["embedding_dimension"]),
        attention_heads=int(architecture["attention_heads"]),
        fourier_top_k=int(architecture["fourier_top_k"]),
        parameter_transform="identity",
        parameter_support_minimum=support.min(axis=0),
        parameter_support_maximum=support.max(axis=0),
        source_conditioned=False,
        parameterized_qkv=False,
        conditioning_rank=int(architecture["conditioning_rank"]),
        adaptive_attention_normalization=False,
        variable_length=True,
        query_chunk_size=int(architecture["query_chunk_size"]),
        target_context_radius=int(architecture["target_context_radius"]),
        amplitude_prediction=False,
        source_free_amplitude_prediction=True,
        relative_condition_attention=bool(
            architecture.get("relative_condition_attention", False)
        ),
        preserve_source_fourier=False,
        speed_parameter_index=architecture.get("speed_parameter_index"),
        speed_to_hz=float(architecture.get("speed_to_hz", 1.0)),
        parameter_mean=parameter_mean,
        parameter_standard_deviation=parameter_standard_deviation,
        factorized_multi_source_attention=False,
        transport_time_scale=float(architecture["time_embedding_scale"]),
        style_spectral_coordinate=str(
            architecture.get("style_spectral_coordinate", "fixed_frequency")
        ),
        sample_rate_hz=float(manifest.device["sample_rate_hz"]),
        style_maximum_order=float(architecture.get("style_maximum_order", 32.0)),
        condition_affine_residual_capacity=int(
            architecture["condition_affine_residual_capacity"]
        ),
        local_style_statistics=bool(
            architecture.get("local_style_statistics", False)
        ),
    )
    return model


def build_source_free_model(
    manifest: DeploymentManifest,
    checkpoint: Mapping[str, Any],
) -> PhysiUNet1D:
    model = create_source_free_model(manifest, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class SourceFreeJiTRuntime:
    """Dataset-independent runtime for one frozen Source-free JiT checkpoint."""

    def __init__(
        self,
        manifest: DeploymentManifest,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        verify_checkpoint: bool = True,
    ) -> None:
        self.manifest = manifest
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint_path}")
        if verify_checkpoint:
            actual_sha256 = sha256_file(self.checkpoint_path)
            if actual_sha256 != manifest.checkpoint_sha256:
                raise ValueError(
                    "checkpoint SHA-256 differs from the deployment manifest: "
                    f"{actual_sha256}"
                )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        requested_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = requested_device
        self.parameter_mean = np.asarray(
            checkpoint["parameter_mean"], dtype=np.float32
        )
        self.parameter_standard_deviation = np.asarray(
            checkpoint["parameter_standard_deviation"], dtype=np.float32
        )
        self.model = build_source_free_model(manifest, checkpoint).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._lock = threading.Lock()

    @property
    def compute_device(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        return torch.cuda.get_device_name(self.device)

    def _condition_vector(self, values: Mapping[str, float]) -> np.ndarray:
        if set(values) != set(self.manifest.parameter_names):
            raise ValueError(
                "condition parameter names must exactly match the deployment manifest"
            )
        ordered = np.asarray(
            [float(values[name]) for name in self.manifest.parameter_names],
            dtype=np.float32,
        )
        if not np.isfinite(ordered).all():
            raise ValueError("condition parameters must be finite")
        for control, value in zip(self.manifest.parameters, ordered, strict=True):
            if not control.minimum <= float(value) <= control.maximum:
                raise ValueError(
                    f"{control.label} must lie in "
                    f"[{control.minimum}, {control.maximum}] {control.unit}"
                )
        return ordered

    @torch.inference_mode()
    def generate(
        self,
        *,
        fault_id: str,
        condition_parameters: Mapping[str, float],
        length: int,
        seed: int,
        sampling_steps: int | None = None,
    ) -> GenerationResult:
        if fault_id not in self.manifest.fault_ids:
            raise ValueError(f"unsupported fault: {fault_id}")
        minimum_length = int(self.manifest.runtime["minimum_length"])
        maximum_length = int(self.manifest.runtime["maximum_length"])
        if not isinstance(length, int) or not minimum_length <= length <= maximum_length:
            raise ValueError(
                f"signal length must lie in [{minimum_length}, {maximum_length}]"
            )
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        steps = (
            int(self.manifest.runtime["sampling_steps"])
            if sampling_steps is None
            else int(sampling_steps)
        )
        if steps < 2:
            raise ValueError("JiT sampling requires at least two steps")
        raw_condition = self._condition_vector(condition_parameters)
        normalized_condition = (
            (raw_condition - self.parameter_mean)
            / self.parameter_standard_deviation
        ).astype(np.float32)
        fault_index = self.manifest.fault_ids.index(fault_id)
        random = torch.Generator(device=self.device).manual_seed(seed)

        with self._lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            condition_tensor = torch.from_numpy(normalized_condition[None, :]).to(
                self.device
            )
            fault_tensor = torch.tensor(
                [fault_index], dtype=torch.long, device=self.device
            )
            noise = torch.randn(
                (1, 1, length),
                device=self.device,
                generator=random,
            )
            generated = heun_direct_x_sample(
                self.model,
                noise,
                condition_tensor,
                fault_tensor,
                steps=steps,
                terminal_time=float(self.manifest.runtime["terminal_time"]),
                time_embedding_scale=float(
                    self.manifest.architecture["time_embedding_scale"]
                ),
            )
            predicted_log_amplitude = self.model.predict_source_free_log_amplitude(
                condition_tensor,
                fault_tensor,
                torch.tensor([length], dtype=torch.long, device=self.device),
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_seconds = time.perf_counter() - started

        generated_numpy = generated[0, 0].detach().cpu().numpy().astype(np.float32)
        maximum = max(float(np.max(np.abs(generated_numpy))), 1e-8)
        normalized_signal = generated_numpy / maximum
        predicted_amplitude = float(predicted_log_amplitude.exp().item())
        if not math.isfinite(predicted_amplitude) or predicted_amplitude <= 0.0:
            raise RuntimeError("model predicted an invalid physical amplitude")
        physical_signal = normalized_signal * predicted_amplitude
        if not np.isfinite(physical_signal).all():
            raise RuntimeError("generated signal contains non-finite values")
        return GenerationResult(
            normalized_signal=normalized_signal,
            physical_signal=physical_signal.astype(np.float32, copy=False),
            predicted_amplitude=predicted_amplitude,
            sample_rate_hz=int(self.manifest.device["sample_rate_hz"]),
            fault_id=fault_id,
            condition_parameters=dict(condition_parameters),
            seed=seed,
            elapsed_seconds=elapsed_seconds,
            compute_device=self.compute_device,
        )
