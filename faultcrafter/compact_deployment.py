from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from faultcrafter.compact_inference import load_compact_model, sample_compact_signal
from faultcrafter.condition_units import condition_field, convert_condition_values, model_condition_request
from faultcrafter.typed_conditions import TYPED_CONDITION_PROTOCOL
from faultcrafter.signal_analysis import BJTU_BEARING_DIAGNOSTICS
from faultcrafter.unified_jit_deployment import FaultControl, ParameterControl, UnifiedGenerationResult


PROFILE_FIELDS = frozenset({
    "route_id", "dataset_id", "device_id", "sensor_location", "channel",
    "signal_quantity", "physical_unit", "sample_rate_hz", "amplitude_method",
    "time_coordinate", "angle_coordinate", "shared_sha256", "parameter_names",
    "speed_parameter_index", "speed_to_hz", "faults",
    "condition_protocol",
    "fixed_conditions", "sampling_policy",
})

INPUT_CONTROLS = {
    "shaft_frequency": ("轴转速", 0.001, 10000.0, 30.0, 1.0),
    "drive_frequency": ("驱动频率", 0.001, 10000.0, 40.0, 1.0),
    "force": ("力载荷", -1e9, 1e9, 0.0, 1000.0),
    "radial_force": ("径向力", 0.0, 1e9, 1000.0, 100.0),
    "torque": ("轴扭矩", -1e9, 1e9, 0.0, 1.0),
    "power": ("标注载荷功率", 0.0, 1e9, 0.0, 1000.0),
}


def _parameter_control(name: str) -> ParameterControl:
    field = condition_field(name)
    label, minimum, maximum, initial, step = INPUT_CONTROLS[field.quantity]
    converted = convert_condition_values(
        [name], [[minimum], [maximum], [initial], [step]],
        input_units=[field.canonical_unit],
    )[:, 0]
    return ParameterControl(
        name=name, label=label, unit=field.native_unit, minimum=float(converted[0]),
        maximum=float(converted[1]), initial=float(converted[2]), step=float(converted[3]), measured=(),
    )


@dataclass(frozen=True, slots=True)
class CompactDeviceControl:
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
    speed_parameter_index: int
    speed_to_hz: float

    @property
    def fault_ids(self) -> tuple[str, ...]:
        return tuple(fault.id for fault in self.faults)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)


@dataclass(frozen=True, slots=True)
class CompactDeploymentManifest:
    compact_root: Path
    model_id: str
    model_label: str
    devices: tuple[CompactDeviceControl, ...]
    runtime: Mapping[str, Any]
    raw: Mapping[str, Any]
    device_roots: Mapping[str, Path] | None = None

    @classmethod
    def load(cls, compact_root: Path, *, extra_device_root: Path | None = None) -> CompactDeploymentManifest:
        root = compact_root.resolve()
        shared_path = root / "shared_generator.pt"
        digest = hashlib.sha256(shared_path.read_bytes()).hexdigest()
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        if shared.get("format") != "faultcrafter_compact_v1":
            raise ValueError("unsupported compact deployment format")
        devices = []
        public_devices = []
        device_roots = {}
        profiles = sorted(root.glob("*/signal_profile.json"))
        if extra_device_root is not None:
            profiles.extend(sorted(extra_device_root.glob("*/signal_profile.json")))
        for profile_path in profiles:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if set(profile) - PROFILE_FIELDS:
                raise ValueError("compact identity contains undeclared runtime state")
            if profile["shared_sha256"] != digest or profile["device_id"] != profile_path.parent.name:
                raise ValueError("compact identity does not match its shared model or directory")
            if not (profile_path.parent / "device_lora.pt").is_file():
                raise ValueError("compact device LoRA is missing")
            if profile["device_id"] in device_roots:
                raise ValueError("duplicate compact device identity")
            device_roots[profile["device_id"]] = profile_path.parent
            parameters = tuple(_parameter_control(name) for name in profile["parameter_names"])
            faults = tuple(FaultControl(
                label, label, str(identity["fault_group_index"]), int(identity["fault_index"]),
            ) for label, identity in profile["faults"].items())
            route = CompactDeviceControl(
                len(devices), profile["device_id"], profile["device_id"],
                profile["sensor_location"], profile["channel"], profile["signal_quantity"],
                profile["physical_unit"], float(profile["sample_rate_hz"]), faults, parameters,
                int(profile["speed_parameter_index"]), float(profile["speed_to_hz"]),
            )
            devices.append(route)
            public_devices.append({
                "index": route.index, "id": route.id, "label": route.label,
                "sensor_label": route.sensor_label, "channel": route.channel,
                "quantity_label": route.quantity_label, "physical_unit": route.physical_unit,
                "sample_rate_hz": route.sample_rate_hz,
                "fixed_conditions": profile.get("fixed_conditions", {}),
                "sampling_policy": profile.get("sampling_policy"),
                "speed_parameter_index": route.speed_parameter_index, "speed_to_hz": route.speed_to_hz,
                "faults": [{"id": fault.id, "label": fault.label, "group": fault.group} for fault in faults],
                "parameters": [{
                    "name": control.name, "label": control.label, "unit": control.unit,
                    "minimum": control.minimum, "maximum": control.maximum, "initial": control.initial,
                    "step": control.step, "measured": [],
                    "range_note": "界面数值限制，不是已验证工况范围；本包不携带源工况统计。",
                } for control in parameters],
            })
            if route.id == "bjtu_rao_bogie" and route.channel == "CH17" and "drive_frequency_hz" in route.parameter_names:
                public_devices[-1]["bearing_diagnostics"] = copy.deepcopy(BJTU_BEARING_DIAGNOSTICS)
        if not devices:
            raise ValueError("compact deployment contains no device identities")
        runtime = {
            "minimum_length": 512, "maximum_length": 131072, "default_length": 4096,
            **shared["sampling"],
        }
        model_id = f"compact_{digest[:16]}"
        model_label = f"Source-free compact JiT · {root.parent.parent.name}/{root.parent.name}"
        raw = {
            "model_id": model_id, "model_label": model_label,
            "shared_sha256": digest,
            "release_id": shared.get("release_id"),
            "device_lora_rank": shared["model_arguments"]["lora_rank"],
            "devices": public_devices, "runtime": runtime,
            "selection": {"training_lengths": []},
            "runtime_contract": "shared_generator + current_device_lora + signal_profile",
            "model_scope": "单个冻结生成器；不按请求工况切换模型。单位等价不证明物理外推正确。",
            **({"condition_model_protocol": TYPED_CONDITION_PROTOCOL} if shared["input_condition_coordinates"] == TYPED_CONDITION_PROTOCOL else {}),
        }
        return cls(root, model_id, model_label, tuple(devices), runtime, raw, device_roots)

    @property
    def device_ids(self) -> tuple[str, ...]:
        return tuple(device.id for device in self.devices)

    def device(self, device_id: str) -> CompactDeviceControl:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise ValueError(f"unsupported device: {device_id}")


class CompactJiTRuntime:
    def __init__(self, manifest: CompactDeploymentManifest, *, device: str | None = None,
                 compute_lock: threading.Lock | None = None) -> None:
        self.manifest = manifest
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = compute_lock or threading.Lock()
        self._active_device: str | None = None
        self._loaded: tuple[Any, dict[str, Any], dict[str, Any]] | None = None

    @property
    def compute_device(self) -> str:
        return torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"

    def generate(self, *, seed: int, **request: Any) -> UnifiedGenerationResult:
        return self.generate_many(seeds=(seed,), **request)[0]

    def generate_many(
        self, *, device_id: str, fault_id: str, condition_parameters: Mapping[str, float],
        length: int, seeds: Sequence[int], condition_units: Mapping[str, str] | None = None,
    ) -> tuple[UnifiedGenerationResult, ...]:
        route = self.manifest.device(device_id)
        parameters, _ = model_condition_request(route.parameter_names, condition_parameters, input_units=condition_units)
        if fault_id not in route.fault_ids:
            raise ValueError(f"unsupported fault for {device_id}: {fault_id}")
        for control in route.parameters:
            if not control.minimum <= parameters[control.name] <= control.maximum:
                raise ValueError(f"{control.label} is outside the declared numeric input limits")
        if not isinstance(length, int) or not self.manifest.runtime["minimum_length"] <= length <= self.manifest.runtime["maximum_length"]:
            raise ValueError("signal length is outside the declared input limits")
        seeds = tuple(seeds)
        if not seeds or len(seeds) > 64 or len(set(seeds)) != len(seeds):
            raise ValueError("batch seeds must contain between 1 and 64 unique values")
        if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative integers")
        results = []
        with self._lock:
            if self._active_device != device_id:
                root = self.manifest.compact_root
                device_root = (self.manifest.device_roots or {}).get(device_id, root / device_id)
                self._loaded = load_compact_model(
                    root / "shared_generator.pt", device_root / "device_lora.pt",
                    device_root / "signal_profile.json", compute_device=self.device,
                )
                self._active_device = device_id
            model, profile, sampling = self._loaded
            for seed in seeds:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                normalized, amplitude, physical = sample_compact_signal(
                    model, profile, sampling,
                    condition_parameters=[parameters[name] for name in route.parameter_names],
                    fault_label=fault_id, length=length, seed=seed,
                )
                if not all(torch.isfinite(value).all().item() for value in (normalized, amplitude, physical)):
                    raise RuntimeError("model returned nonfinite values; the request was not clipped or repaired")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                results.append(UnifiedGenerationResult(
                    normalized[0, 0].cpu().numpy(), physical[0, 0].cpu().numpy(),
                    float(amplitude[0]), int(route.sample_rate_hz), route.physical_unit,
                    device_id, fault_id, parameters, seed, time.perf_counter() - started,
                    self.compute_device,
                ))
        return tuple(results)
