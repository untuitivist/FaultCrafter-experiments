from __future__ import annotations

import copy

import torch
from torch import nn

from faultcrafter.fault_composition import composition_weights
from faultcrafter.physical_reference import MASKED_MECHANICAL_REFERENCE
from faultcrafter.unified_generator import UnifiedSignalGenerator


RELEASE32_BACKBONE = {
    "base_channels": 24, "context_dimension": 96, "attention_heads": 4,
    "style_attention_rank": 24, "style_token_count": 8, "attention_radius": 16,
    "attention_chunk_size": 256, "lora_rank": 8, "affine_restoration": False,
    "style_mechanism": "legacy_affine", "fault_identity_guidance_scale": 1.5,
    "fault_identity_guidance_mode": "linear",
}


class MethodGenerator(nn.Module):
    def __init__(self, vocabulary, *, backbone=None):
        super().__init__()
        self.vocabulary = copy.deepcopy(vocabulary)
        self.backbone = {**RELEASE32_BACKBONE, **(backbone or {})}
        self.arguments = {**self.backbone, "num_faults": len(vocabulary["tokens"]),
                          "num_fault_groups": 5, "num_devices": len(vocabulary["channels"]),
                          "condition_dimension": 14, "physics_reference_policy": MASKED_MECHANICAL_REFERENCE}
        self.network = UnifiedSignalGenerator(**self.arguments)
        self.token_indices = {tuple(value): index for index, value in enumerate(vocabulary["tokens"])}

    def encode(self, condition):
        device_offset = 14
        route_offset = device_offset + len(self.vocabulary["devices"])
        component_offset = route_offset + len(self.vocabulary["channels"])
        expected = component_offset + len(self.vocabulary["components"]) + 1
        if condition.ndim != 2 or condition.shape[1] != expected or not torch.isfinite(condition).all():
            raise ValueError("Condition schema differs from the trained generator")
        route_values = condition[:, route_offset:component_offset]
        device_values = condition[:, device_offset:route_offset]
        active = condition[:, component_offset:-1]
        if (not torch.all((route_values == 0) | (route_values == 1)) or not torch.all(route_values.sum(1) == 1)
                or not torch.all((device_values == 0) | (device_values == 1)) or not torch.all(device_values.sum(1) == 1)
                or not torch.all((active == 0) | (active == 1))):
            raise ValueError("Device, channel and atomic-fault inputs must be explicit binary indicators")
        device_indices = device_values.argmax(1).cpu().tolist()
        route_indices = route_values.argmax(1).cpu().tolist()
        fault_components, normal_indices, group_components = [], [], []
        for row, (device_index, route_index) in enumerate(zip(device_indices, route_indices, strict=True)):
            device = self.vocabulary["devices"][device_index]
            if self.vocabulary["channels"][route_index][0] != device:
                raise ValueError("Channel route does not belong to the requested device")
            parts = active[row].nonzero().flatten().cpu().tolist()
            if any(self.vocabulary["components"][part][0] != device for part in parts):
                raise ValueError("Fault component belongs to a different device")
            fault_components.append([self.token_indices[tuple(self.vocabulary["components"][part])] for part in parts])
            normal_indices.append(self.token_indices[device, "__normal__"])
            group_components.append([self.vocabulary["component_groups"][part] for part in parts])
        fault = composition_weights(fault_components, normal_indices, len(self.vocabulary["tokens"]))
        group = composition_weights(group_components, [0] * len(condition), 5, allow_repeated_components=True)
        parameters = condition[:, :14]
        speed = parameters[:, 0] * 100 * parameters[:, 7]
        rates = condition[:, -1] * 64000
        if torch.any(rates <= 0):
            raise ValueError("Sample rates must be positive")
        return (parameters, torch.as_tensor(fault, device=condition.device), torch.as_tensor(group, device=condition.device),
                torch.as_tensor(route_indices, device=condition.device), speed, rates)

    @torch.no_grad()
    def sample(self, condition, steps=32, length=1024, *, initial_noise=None, amplitude_noise=None, physical=False):
        if steps < 2 or length < 16:
            raise ValueError("Generation requires at least two calls and a valid signal length")
        parameters, fault, group, routes, speed, rates = self.encode(condition)
        shape = (len(condition), 1, length)
        state = torch.randn(shape, device=condition.device) if initial_noise is None else initial_noise.clone()
        if state.shape != shape or not torch.isfinite(state).all():
            raise ValueError("Initial noise has the wrong shape or contains nonfinite values")
        times = torch.linspace(0, 0.95, steps, device=condition.device, dtype=condition.dtype)
        lengths = torch.full((len(condition),), length, device=condition.device)
        for index in range(steps - 1):
            time = times[index].expand(len(condition))
            prediction = self.network(state, time, parameters, fault, group, routes, speed, rates, lengths)
            state = state + (times[index + 1] - times[index]) * (prediction.normalized_signal - state) / (1 - time[:, None, None])
        final = self.network(state, times[-1].expand(len(condition)), parameters, fault, group, routes, speed, rates, lengths)
        normalized = final.normalized_signal
        normalized = normalized / normalized.abs().amax((1, 2), keepdim=True).clamp_min(1e-8)
        if not physical:
            return normalized
        amplitude_noise = torch.randn(len(condition), device=condition.device) if amplitude_noise is None else amplitude_noise
        if amplitude_noise.shape != (len(condition),):
            raise ValueError("Amplitude noise must contain one value per generated parent")
        amplitude = (final.log_amplitude_mean + final.log_amplitude_log_scale.exp() * amplitude_noise).exp()
        return normalized, amplitude, normalized * amplitude[:, None, None]


class AdaptationParameters:
    def __init__(self, model, strategy, routes, tokens):
        if strategy not in ("lora", "full", "scratch"):
            raise ValueError("Unknown adaptation strategy")
        self.model = model
        self.masks = {}
        self.reference = {name: value.detach().clone() for name, value in model.named_parameters()}
        self.hooks = []
        for name, parameter in model.named_parameters():
            mask = torch.ones_like(parameter, dtype=torch.bool)
            if strategy == "lora":
                mask.zero_()
                if name.endswith(("lora_up", "lora_down")):
                    mask[list(routes)] = True
                elif name == "network.fault_embedding.weight":
                    mask[list(tokens)] = True
            parameter.requires_grad_(bool(mask.any()))
            self.masks[name] = mask
            if parameter.requires_grad and not bool(mask.all()):
                self.hooks.append(parameter.register_hook(lambda gradient, selected=mask: gradient * selected))
        self.parameters = [value for value in model.parameters() if value.requires_grad]
        self.count = sum(int(mask.sum()) for mask in self.masks.values())

    @torch.no_grad()
    def restore_frozen(self, model=None):
        for name, parameter in (model or self.model).named_parameters():
            mask = self.masks[name]
            if not bool(mask.all()):
                parameter[~mask] = self.reference[name][~mask]

    def check(self, model=None):
        for name, parameter in (model or self.model).named_parameters():
            mask = self.masks[name]
            if not torch.equal(parameter.detach()[~mask], self.reference[name][~mask]):
                raise RuntimeError(f"Frozen original-device parameters changed: {name}")


def initialize_new_tokens(model, parent_devices, new_device):
    vocabulary = model.vocabulary
    groups = {tuple(component): group for component, group in zip(vocabulary["components"], vocabulary["component_groups"], strict=True)}
    with torch.no_grad():
        weights = model.network.fault_embedding.weight
        for index, (device, label) in enumerate(vocabulary["tokens"]):
            if device != new_device:
                continue
            group = 0 if label == "__normal__" else groups[device, label]
            candidates = [position for position, token in enumerate(vocabulary["tokens"])
                          if token[0] in parent_devices and (0 if token[1] == "__normal__" else groups[tuple(token)]) == group]
            if not candidates:
                raise ValueError(f"No source semantic group for new fault {device}/{label}")
            weights[index].copy_(weights[candidates].mean(0))


def load_method(checkpoint, result=None, condition_dimension=None):
    if checkpoint.get("format") != "faultcrafter_benchmark32_v1":
        raise ValueError("A fresh held-out benchmark checkpoint is required")
    if checkpoint.get("deployment_only") or checkpoint.get("open_set_evaluation_eligible") is not True:
        raise ValueError("Deployment-only weights cannot be used for held-out evaluation")
    model = MethodGenerator(checkpoint["vocabulary"], backbone=checkpoint["backbone"])
    model.load_state_dict(checkpoint["model"])
    return model
