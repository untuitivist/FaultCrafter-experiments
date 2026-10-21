from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from faultcrafter.unified_generator import UnifiedSignalGenerator, sample_unified_generator
from faultcrafter.generator_architecture import create_signal_generator
from faultcrafter.fault_composition import composition_weights
from faultcrafter.condition_units import convert_condition_values
from faultcrafter.typed_conditions import TYPED_CONDITION_PROTOCOL, TYPED_DIMENSION, encode_typed_conditions
from faultcrafter.physical_reference import KNOWN_MECHANICAL_REFERENCE, MECHANICAL_ANGLE_COORDINATE


def is_device_lora(name: str) -> bool:
    return name.endswith((".lora_down", ".lora_up"))


def folded_device_lora(
    model: UnifiedSignalGenerator,
    device_index: int,
    mean: Sequence[float],
    deviation: Sequence[float],
) -> dict[str, torch.Tensor]:
    if model.style_mechanism != "legacy_affine":
        raise ValueError("compact normalization folding requires affine conditioning")
    if len(mean) != len(deviation) or not 0 < len(mean) <= model.condition_dimension:
        raise ValueError("normalization dimensions differ")
    query = model.style_field.query
    dimension = model.condition_dimension + 1
    if query.rank < dimension:
        raise ValueError("existing device LoRA rank cannot contain the affine input map")
    center = torch.as_tensor(mean, dtype=torch.float64)
    scale = torch.as_tensor(deviation, dtype=torch.float64)
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("normalization statistics must be finite with positive scale")
    state = {
        name: tensor[device_index].detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if is_device_lora(name)
    }
    transform = torch.eye(dimension, dtype=torch.float64)
    for parameter_index in range(len(mean)):
        transform[parameter_index + 1, 0] = -center[parameter_index] / scale[parameter_index]
        transform[parameter_index + 1, parameter_index + 1] = 1.0 / scale[parameter_index]
    shared = query.shared.weight.detach().cpu().double()
    effective = shared + (
        state["style_field.query.lora_up"].double()
        @ state["style_field.query.lora_down"].double()
    ) / float(query.rank)
    delta = effective @ transform - shared
    down = torch.zeros_like(state["style_field.query.lora_down"])
    up = torch.zeros_like(state["style_field.query.lora_up"])
    down[:dimension] = torch.eye(dimension, dtype=down.dtype)
    up[:, :dimension] = (delta * float(query.rank)).to(up)
    state["style_field.query.lora_down"] = down
    state["style_field.query.lora_up"] = up
    return state


def load_compact_model(
    shared_path: Path,
    lora_path: Path,
    profile_path: Path,
    *,
    compute_device: str | torch.device = "cpu",
) -> tuple[UnifiedSignalGenerator, dict[str, Any], dict[str, Any]]:
    shared = torch.load(shared_path, map_location="cpu", weights_only=True)
    precision = shared.get("inference_precision")
    if precision not in (None, "float32_full_precision"):
        raise ValueError("unsupported compact inference precision")
    if precision == "float32_full_precision":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    adapter = torch.load(lora_path, map_location="cpu", weights_only=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(shared_path.read_bytes()).hexdigest()
    if shared["format"] != "faultcrafter_compact_v1" or adapter["shared_sha256"] != digest:
        raise ValueError("compact LoRA and shared generator are not paired")
    if profile["shared_sha256"] != digest or adapter["device_id"] != profile["device_id"]:
        raise ValueError("compact signal identity and device LoRA differ")
    protocol = shared["input_condition_coordinates"]
    if protocol not in ("physical_raw_folded_into_device_lora", TYPED_CONDITION_PROTOCOL):
        raise ValueError("compact conditioning contract differs")
    if protocol == TYPED_CONDITION_PROTOCOL:
        if profile.get("condition_protocol") != protocol or shared["model_arguments"]["condition_dimension"] != TYPED_DIMENSION:
            raise ValueError("typed shared generator and signal input identity differ")
    elif profile.get("condition_protocol") is not None:
        raise ValueError("legacy compact model cannot use a typed signal profile")
    if shared["model_arguments"].get("physics_reference_policy") == KNOWN_MECHANICAL_REFERENCE:
        if protocol != TYPED_CONDITION_PROTOCOL or profile.get("angle_coordinate") != MECHANICAL_ANGLE_COORDINATE:
            raise ValueError("known mechanical compact reference requires matching typed inputs and missing-angle semantics")
    model = create_signal_generator(**shared["model_arguments"])
    expected = model.state_dict()
    if set(shared["state"]) != {name for name in expected if not is_device_lora(name)}:
        raise ValueError("compact shared model contains unexpected device state")
    if set(adapter["state"]) != {name for name in expected if is_device_lora(name)}:
        raise ValueError("compact adapter may contain only the existing LoRA tensors")
    state = dict(shared["state"])
    state.update({name: tensor.unsqueeze(0) for name, tensor in adapter["state"].items()})
    model.load_state_dict(state, strict=True)
    model.to(compute_device).eval()
    return model, profile, shared["sampling"]


@torch.no_grad()
def sample_compact_signal(
    model: UnifiedSignalGenerator,
    profile: dict[str, Any],
    sampling: dict[str, Any],
    *,
    condition_parameters: Sequence[float],
    fault_label: str | Sequence[str],
    length: int,
    seed: int,
    condition_units: Mapping[str, str] | Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if length <= 0 or len(condition_parameters) != len(profile["parameter_names"]):
        raise ValueError("requested length or physical condition dimensions differ")
    compute_device = next(model.parameters()).device
    values = condition_parameters
    if condition_units is not None:
        values = convert_condition_values(
            profile["parameter_names"], condition_parameters, input_units=condition_units,
        )
    raw = torch.as_tensor(values, dtype=torch.float32, device=compute_device)
    if not torch.isfinite(raw).all():
        raise ValueError("physical conditions must be finite")
    parameters = torch.zeros(1, model.condition_dimension, device=compute_device)
    if profile.get("condition_protocol") == TYPED_CONDITION_PROTOCOL:
        encoded = encode_typed_conditions(profile["parameter_names"], raw.detach().cpu().numpy())
        parameters[0] = torch.from_numpy(encoded).to(parameters)
    else:
        parameters[0, :len(raw)] = raw
    if isinstance(fault_label, str):
        identity = profile["faults"][fault_label]
        fault = torch.tensor([identity["fault_index"]], device=compute_device, dtype=torch.long)
        group = torch.tensor([identity["fault_group_index"]], device=compute_device, dtype=torch.long)
    else:
        members = tuple(fault_label)
        if any(label not in profile["faults"] for label in members):
            raise ValueError("fault component is absent from the device signal identity")
        normal = profile["faults"]["normal"]
        fault = torch.from_numpy(composition_weights(
            [[profile["faults"][label]["fault_index"] for label in members]], [normal["fault_index"]], model.num_faults,
        )).to(compute_device)
        group = torch.from_numpy(composition_weights(
            [[profile["faults"][label]["fault_group_index"] for label in members]], [normal["fault_group_index"]], model.num_fault_groups,
            allow_repeated_components=True,
        )).to(compute_device)
    route = torch.zeros(1, device=compute_device, dtype=torch.long)
    speed = raw[int(profile["speed_parameter_index"])].reshape(1) * float(profile["speed_to_hz"])
    sample_rate = torch.tensor([profile["sample_rate_hz"]], device=compute_device)
    random = torch.Generator(device=compute_device).manual_seed(seed)
    noise = torch.randn(1, 1, length, device=compute_device, generator=random)
    amplitude_noise = (
        torch.randn(1, device=compute_device, generator=random)
        if sampling["stochastic_amplitude"] else torch.zeros(1, device=compute_device)
    )
    return sample_unified_generator(
        model, noise, parameters, fault, group, route, speed, sample_rate,
        steps=int(sampling["sampling_steps"]),
        terminal_time=float(sampling["terminal_time"]),
        amplitude_noise=amplitude_noise,
    )
