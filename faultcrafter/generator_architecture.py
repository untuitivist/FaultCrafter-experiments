from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as functional

from faultcrafter.unified_generator import (
    RoutedLoRALinear,
    UnifiedSignalGenerator,
    timestep_embedding,
)


ARCHITECTURE_KEYS = (
    "waveform_backbone", "prediction_parameterization", "transformer_patch_size",
    "transformer_depth", "transformer_mlp_ratio",
)
UNET_MODULES = (
    "lifting", "encoder_0", "down_0", "encoder_1", "down_1", "middle",
    "attention", "decoder_1", "decoder_0", "output_norm", "output",
)


def architecture_arguments(generator: dict[str, Any]) -> dict[str, Any]:
    return {name: generator[name] for name in ARCHITECTURE_KEYS if name in generator}


class WaveformTransformerBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, mlp_ratio: float,
        num_devices: int, lora_rank: int,
    ) -> None:
        super().__init__()
        if dimension % heads or mlp_ratio <= 0:
            raise ValueError("invalid waveform Transformer dimensions")
        self.heads = heads
        self.dimension = dimension
        self.attention_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.qkv = RoutedLoRALinear(dimension, 3 * dimension, num_devices, lora_rank)
        self.attention_output = RoutedLoRALinear(dimension, dimension, num_devices, lora_rank)
        hidden = int(dimension * mlp_ratio)
        self.mlp_input = RoutedLoRALinear(dimension, hidden, num_devices, lora_rank)
        self.mlp_output = RoutedLoRALinear(hidden, dimension, num_devices, lora_rank)
        self.modulation = RoutedLoRALinear(dimension, 6 * dimension, num_devices, lora_rank)
        self.log_order = nn.Parameter(torch.zeros(heads))
        self.angle_bias_strength = nn.Parameter(torch.zeros(heads))
        self.log_time_scale = nn.Parameter(torch.zeros(heads))
        nn.init.zeros_(self.modulation.shared.weight)
        nn.init.zeros_(self.modulation.shared.bias)

    def forward(
        self, tokens: torch.Tensor, context: torch.Tensor, device_id: torch.Tensor,
        delta_time: torch.Tensor, shaft_speed_hz: torch.Tensor,
    ) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(functional.silu(context), device_id).chunk(6, dim=-1)
        )
        normalized = self.attention_norm(tokens) * (1.0 + scale_attention[:, None]) + shift_attention[:, None]
        batch_size, token_count, _ = tokens.shape
        projections = self.qkv(normalized, device_id).reshape(
            batch_size, token_count, 3, self.heads, self.dimension // self.heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = projections.unbind(0)
        order = functional.softplus(self.log_order) + 0.1
        angle = 2.0 * math.pi * shaft_speed_hz[:, None, None] * delta_time
        bias = torch.cos(angle[:, None] * order[None, :, None, None]) * self.angle_bias_strength[None, :, None, None]
        bias = bias - delta_time[:, None].abs() / (functional.softplus(self.log_time_scale)[None, :, None, None] + 1e-4)
        attended = functional.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, self.dimension)
        tokens = tokens + gate_attention[:, None] * self.attention_output(attended, device_id)
        normalized = self.mlp_norm(tokens) * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden = functional.gelu(self.mlp_input(normalized, device_id), approximate="tanh")
        return tokens + gate_mlp[:, None] * self.mlp_output(hidden, device_id)


class WaveformTransformer(nn.Module):
    def __init__(
        self, dimension: int, heads: int, patch_size: int, depth: int,
        mlp_ratio: float, num_devices: int, lora_rank: int,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or depth <= 0:
            raise ValueError("waveform patch size and Transformer depth must be positive")
        self.patch_size = patch_size
        self.dimension = dimension
        self.patch_embedding = nn.Linear(patch_size, dimension)
        self.blocks = nn.ModuleList([
            WaveformTransformerBlock(dimension, heads, mlp_ratio, num_devices, lora_rank)
            for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.output_modulation = RoutedLoRALinear(dimension, 2 * dimension, num_devices, lora_rank)
        self.output = nn.Linear(dimension, patch_size)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, signal: torch.Tensor, context: torch.Tensor, device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor, sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        original_length = signal.shape[-1]
        padded = functional.pad(signal, (0, (-original_length) % self.patch_size))
        patches = padded[:, 0].unfold(-1, self.patch_size, self.patch_size)
        positions = torch.arange(patches.shape[1], device=signal.device, dtype=signal.dtype)
        tokens = self.patch_embedding(patches) + timestep_embedding(positions, self.dimension)[None]
        offsets = (positions[None, :] - positions[:, None]) * self.patch_size
        delta_time = offsets[None] / sample_rate_hz[:, None, None]
        for block in self.blocks:
            tokens = block(tokens, context, device_id, delta_time, shaft_speed_hz)
        shift, scale = self.output_modulation(functional.silu(context), device_id).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1.0 + scale[:, None]) + shift[:, None]
        return self.output(tokens).reshape(len(signal), 1, -1)[..., :original_length]


class ControlledSignalGenerator(UnifiedSignalGenerator):
    def __init__(
        self, *, waveform_backbone: str = "unet", prediction_parameterization: str = "clean",
        transformer_patch_size: int = 256, transformer_depth: int = 2,
        transformer_mlp_ratio: float = 2.0, **arguments: Any,
    ) -> None:
        if waveform_backbone not in ("unet", "transformer"):
            raise ValueError("unknown waveform backbone")
        if prediction_parameterization not in ("clean", "velocity"):
            raise ValueError("unknown prediction parameterization")
        if arguments.get("style_mechanism", "legacy_affine") != "legacy_affine":
            raise ValueError("controlled architecture comparison requires the common affine condition path")
        super().__init__(**arguments)
        self.waveform_backbone = waveform_backbone
        self.prediction_parameterization = prediction_parameterization
        if waveform_backbone == "transformer":
            for name in UNET_MODULES:
                delattr(self, name)
            self.waveform_transformer = WaveformTransformer(
                self.context_dimension, arguments.get("attention_heads", 4), transformer_patch_size,
                transformer_depth, transformer_mlp_ratio, self.num_devices, arguments.get("lora_rank", 8),
            )

    def _waveform_prediction(
        self, signal: torch.Tensor, context: torch.Tensor, style: torch.Tensor,
        condition_parameters: torch.Tensor, device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor, sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        if self.waveform_backbone == "unet":
            return super()._waveform_prediction(
                signal, context, style, condition_parameters, device_id, shaft_speed_hz, sample_rate_hz,
            )
        return self.waveform_transformer(signal, context, device_id, shaft_speed_hz, sample_rate_hz)

    def _clean_prediction(
        self, signal: torch.Tensor, prediction: torch.Tensor, time: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_parameterization == "velocity":
            return signal + (1.0 - time[:, None, None]) * prediction
        return prediction


def create_signal_generator(**arguments: Any) -> UnifiedSignalGenerator:
    settings = {name: arguments.pop(name) for name in ARCHITECTURE_KEYS if name in arguments}
    if not settings:
        return UnifiedSignalGenerator(**arguments)
    return ControlledSignalGenerator(**settings, **arguments)
