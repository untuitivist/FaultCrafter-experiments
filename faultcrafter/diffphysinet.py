from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .condition_attention import (
    torch_condition_weight_family,
    torch_convex_hull_projection,
)


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor,
    dimension: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError(f"timestep must have shape [batch], got {tuple(timestep.shape)}")
    if dimension < 2:
        raise ValueError("embedding dimension must be at least two")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / half
    )
    angles = timestep.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=1)
    return embedding


def topk_fourier_component(signal: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the largest Fourier coefficients independently in every channel."""

    if signal.ndim != 3:
        raise ValueError("signal must have shape [batch,channels,length]")
    spectrum = torch.fft.rfft(signal, dim=-1)
    frequency_count = spectrum.shape[-1]
    if not 0 < top_k <= frequency_count:
        raise ValueError(f"top_k must be in [1,{frequency_count}]")
    indices = torch.topk(spectrum.abs(), k=top_k, dim=-1, sorted=False).indices
    selected = torch.zeros_like(spectrum)
    selected.scatter_(-1, indices, spectrum.gather(-1, indices))
    return torch.fft.irfft(selected, n=signal.shape[-1], dim=-1)


class LocalLinearConditionField(nn.Module):
    """Continuous condition field using local-linear attention over prototypes."""

    def __init__(
        self,
        condition_dimension: int,
        context_dimension: int,
        prototype_capacity: int,
        local_linear_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if condition_dimension <= 0 or context_dimension <= 0:
            raise ValueError("condition and context dimensions must be positive")
        if prototype_capacity < 2:
            raise ValueError("condition attention requires at least two prototypes")
        if not 0.0 <= local_linear_fraction <= 1.0:
            raise ValueError("local-linear fraction must lie in [0,1]")
        self.condition_dimension = condition_dimension
        self.context_dimension = context_dimension
        self.prototype_capacity = prototype_capacity
        self.register_buffer(
            "prototype_conditions",
            torch.zeros(prototype_capacity, condition_dimension),
        )
        self.register_buffer("active_prototype_count", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "local_linear_fraction", torch.tensor(float(local_linear_fraction))
        )
        self.prototype_contexts = nn.Parameter(
            torch.zeros(prototype_capacity, context_dimension)
        )

    def attention_weights(self, conditions: torch.Tensor) -> torch.Tensor:
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dimension:
            raise ValueError("condition matrix does not match the prototype field")
        count = int(self.active_prototype_count)
        if count < 2:
            raise RuntimeError("condition prototypes have not been initialized")
        prototypes = self.prototype_conditions[:count]
        source = prototypes[None, :, :].expand(len(conditions), -1, -1)
        softmax, local_linear = torch_condition_weight_family(source, conditions)
        return softmax + self.local_linear_fraction * (local_linear - softmax)

    @torch.no_grad()
    def initialize_from_contexts(
        self,
        conditions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, float | int]:
        conditions = torch.as_tensor(
            conditions,
            dtype=self.prototype_conditions.dtype,
            device=self.prototype_conditions.device,
        )
        contexts = torch.as_tensor(
            contexts,
            dtype=self.prototype_contexts.dtype,
            device=self.prototype_contexts.device,
        )
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dimension:
            raise ValueError("prototype condition dimensions differ")
        if contexts.shape != (len(conditions), self.context_dimension):
            raise ValueError("prototype contexts do not align with conditions")
        if not 2 <= len(conditions) <= self.prototype_capacity:
            raise ValueError("prototype count exceeds the configured capacity")
        self.prototype_conditions.zero_()
        self.prototype_conditions[: len(conditions)].copy_(conditions)
        self.active_prototype_count.fill_(len(conditions))
        knot_weights = self.attention_weights(conditions)
        rank = int(torch.linalg.matrix_rank(knot_weights))
        if rank != len(conditions):
            raise ValueError("condition-attention knot matrix is singular")
        solution = torch.linalg.solve(
            knot_weights.to(torch.float64), contexts.to(torch.float64)
        ).to(contexts.dtype)
        self.prototype_contexts.zero_()
        self.prototype_contexts[: len(conditions)].copy_(solution)
        reconstructed = self(conditions)
        maximum_error = float((reconstructed - contexts).abs().max())
        if maximum_error > 2e-5:
            raise RuntimeError("condition prototypes do not reproduce source contexts")
        return {
            "prototype_count": len(conditions),
            "knot_matrix_rank": rank,
            "knot_matrix_condition_number": float(
                torch.linalg.cond(knot_weights.to(torch.float64))
            ),
            "maximum_context_error": maximum_error,
        }

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        count = int(self.active_prototype_count)
        weights = self.attention_weights(conditions)
        return weights @ self.prototype_contexts[:count]


class AffineResidualConditionField(nn.Module):
    """Affine condition coordinates plus bounded data-driven residual attention."""

    def __init__(
        self,
        condition_dimension: int,
        context_dimension: int,
        prototype_capacity: int,
    ) -> None:
        super().__init__()
        if condition_dimension <= 0 or context_dimension <= 0:
            raise ValueError("condition and context dimensions must be positive")
        if prototype_capacity < 2:
            raise ValueError("residual attention requires at least two prototypes")
        self.condition_dimension = condition_dimension
        self.context_dimension = context_dimension
        self.prototype_capacity = prototype_capacity
        self.affine_context = nn.Linear(condition_dimension, context_dimension)
        self.residual_prototypes = nn.Parameter(
            torch.zeros(prototype_capacity, context_dimension)
        )
        self.register_buffer(
            "prototype_conditions",
            torch.zeros(prototype_capacity, condition_dimension),
        )
        self.register_buffer("active_prototype_count", torch.zeros((), dtype=torch.long))

    def attention_weights(
        self,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dimension:
            raise ValueError("condition matrix does not match the residual field")
        count = int(self.active_prototype_count)
        if count < 2:
            raise RuntimeError("condition prototypes have not been initialized")
        prototypes = self.prototype_conditions[:count]
        source = prototypes[None, :, :].expand(len(conditions), -1, -1)
        distance_to_prototype = torch.linalg.vector_norm(
            source - conditions[:, None, :],
            dim=2,
        )
        source_knot_query = distance_to_prototype.min(dim=1).values <= 1e-6
        projected = conditions.clone()
        if bool((~source_knot_query).any()):
            unmatched, inverse = torch.unique(
                conditions[~source_knot_query],
                dim=0,
                return_inverse=True,
            )
            unmatched_source = prototypes[None].expand(len(unmatched), -1, -1)
            unique_projection = torch_convex_hull_projection(
                unmatched_source,
                unmatched,
            )
            projected[~source_knot_query] = unique_projection[inverse]
        weights, _ = torch_condition_weight_family(source, projected)
        return weights, projected

    @torch.no_grad()
    def initialize_from_contexts(
        self,
        conditions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> dict[str, float | int]:
        conditions = torch.as_tensor(
            conditions,
            dtype=self.prototype_conditions.dtype,
            device=self.prototype_conditions.device,
        )
        contexts = torch.as_tensor(
            contexts,
            dtype=self.residual_prototypes.dtype,
            device=self.residual_prototypes.device,
        )
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dimension:
            raise ValueError("prototype condition dimensions differ")
        if contexts.shape != (len(conditions), self.context_dimension):
            raise ValueError("prototype contexts do not align with conditions")
        if not 2 <= len(conditions) <= self.prototype_capacity:
            raise ValueError("prototype count exceeds the configured capacity")
        design = torch.cat(
            (conditions, torch.ones(len(conditions), 1, device=conditions.device)),
            dim=1,
        )
        solution = torch.linalg.lstsq(
            design.to(torch.float64), contexts.to(torch.float64)
        ).solution.to(contexts.dtype)
        self.affine_context.weight.copy_(solution[:-1].transpose(0, 1))
        self.affine_context.bias.copy_(solution[-1])
        self.prototype_conditions.zero_()
        self.prototype_conditions[: len(conditions)].copy_(conditions)
        self.active_prototype_count.fill_(len(conditions))

        affine_values = self.affine_context(conditions)
        residual_values = contexts - affine_values
        knot_weights, _ = self.attention_weights(conditions)
        rank = int(torch.linalg.matrix_rank(knot_weights))
        if rank != len(conditions):
            raise ValueError("residual-attention knot matrix is singular")
        residual_solution = torch.linalg.solve(
            knot_weights.to(torch.float64), residual_values.to(torch.float64)
        ).to(contexts.dtype)
        self.residual_prototypes.zero_()
        self.residual_prototypes[: len(conditions)].copy_(residual_solution)
        reconstructed = self(conditions)
        maximum_error = float((reconstructed - contexts).abs().max())
        if maximum_error > 2e-5:
            raise RuntimeError("affine-residual field does not reproduce source contexts")
        return {
            "prototype_count": len(conditions),
            "affine_design_rank": int(torch.linalg.matrix_rank(design)),
            "residual_knot_matrix_rank": rank,
            "residual_knot_matrix_condition_number": float(
                torch.linalg.cond(knot_weights.to(torch.float64))
            ),
            "maximum_context_error": maximum_error,
            "maximum_source_residual_norm": float(
                torch.linalg.vector_norm(residual_values, dim=1).max()
            ),
        }

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        count = int(self.active_prototype_count)
        weights, _ = self.attention_weights(conditions)
        residual = weights @ self.residual_prototypes[:count]
        return self.affine_context(conditions) + residual


class DeviceRoutedLowRankCondition(nn.Module):
    """Shared low-rank condition basis with data-learned device routing."""

    def __init__(
        self,
        condition_dimension: int,
        context_dimension: int,
        num_devices: int,
        rank: int,
    ) -> None:
        super().__init__()
        if condition_dimension <= 0 or context_dimension <= 0:
            raise ValueError("condition and context dimensions must be positive")
        if num_devices <= 1:
            raise ValueError("device routing requires at least two devices")
        if not 0 < rank <= context_dimension:
            raise ValueError("device condition rank must lie in [1,context_dimension]")
        self.condition_dimension = condition_dimension
        self.context_dimension = context_dimension
        self.num_devices = num_devices
        self.rank = rank
        self.device_embedding = nn.Embedding(num_devices, context_dimension)
        self.device_coefficients = nn.Embedding(num_devices, rank)
        self.condition_down = nn.Linear(condition_dimension, rank, bias=False)
        self.condition_up = nn.Linear(rank, context_dimension, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.device_embedding.weight, std=0.02)
        nn.init.ones_(self.device_coefficients.weight)
        nn.init.normal_(self.condition_down.weight, std=0.02)
        nn.init.zeros_(self.condition_up.weight)

    def forward(
        self,
        context: torch.Tensor,
        conditions: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(conditions)
        if conditions.shape != (batch_size, self.condition_dimension):
            raise ValueError("condition matrix does not match device routing")
        if context.shape != (batch_size, self.context_dimension):
            raise ValueError("context matrix does not match device routing")
        if device_id.shape != (batch_size,) or device_id.dtype != torch.long:
            raise ValueError("device_id must be a long tensor with shape [batch]")
        if torch.any(device_id < 0) or torch.any(device_id >= self.num_devices):
            raise ValueError("device_id is outside the configured device vocabulary")
        coefficients = self.device_coefficients(device_id)
        low_rank = self.condition_up(
            self.condition_down(conditions) * coefficients
        )
        return context + self.device_embedding(device_id) + low_rank


class RoutedLoRALinear(nn.Module):
    """Linear layer with a shared low-rank basis and per-sample route weights."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        rank: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("LoRA dimensions must be positive")
        if not 0 < rank <= min(input_dimension, output_dimension):
            raise ValueError("LoRA rank exceeds the linear layer dimensions")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(output_dimension, input_dimension))
        self.bias = nn.Parameter(torch.empty(output_dimension)) if bias else None
        self.lora_a = nn.Parameter(torch.empty(input_dimension, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, output_dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.input_dimension)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.zeros_(self.lora_b)

    def forward(
        self,
        values: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[-1] != self.input_dimension:
            raise ValueError("LoRA input dimension differs from the configured layer")
        if route.shape != (len(values), self.rank):
            raise ValueError("device route must have shape [batch,rank]")
        base = F.linear(values, self.weight, self.bias)
        delta = ((values @ self.lora_a) * route) @ self.lora_b
        return base + delta / float(self.rank)


class IndependentDeviceLoRALinear(nn.Module):
    """Linear layer with an independent low-rank update for each device."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        rank: int,
        num_devices: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or output_dimension <= 0 or num_devices <= 1:
            raise ValueError("independent device LoRA dimensions must be valid")
        if not 0 < rank <= min(input_dimension, output_dimension):
            raise ValueError("LoRA rank exceeds the linear layer dimensions")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.rank = rank
        self.num_devices = num_devices
        self.weight = nn.Parameter(torch.empty(output_dimension, input_dimension))
        self.bias = nn.Parameter(torch.empty(output_dimension)) if bias else None
        self.lora_a = nn.Parameter(
            torch.empty(num_devices, input_dimension, rank)
        )
        self.lora_b = nn.Parameter(
            torch.empty(num_devices, rank, output_dimension)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.input_dimension)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.zeros_(self.lora_b)

    def forward(
        self,
        values: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[-1] != self.input_dimension:
            raise ValueError("LoRA input dimension differs from the configured layer")
        if device_id.shape != (len(values),) or device_id.dtype != torch.long:
            raise ValueError("device_id must be a long tensor with shape [batch]")
        if torch.any(device_id < 0) or torch.any(device_id >= self.num_devices):
            raise ValueError("device_id is outside the LoRA adapter vocabulary")
        base = F.linear(values, self.weight, self.bias)
        first = self.lora_a[device_id]
        second = self.lora_b[device_id]
        delta = torch.bmm(torch.bmm(values[:, None, :], first), second)[:, 0, :]
        return base + delta / float(self.rank)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        embedding_dimension: int,
        device_lora_rank: int = 0,
        independent_device_lora_count: int = 0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.input_convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )
        self.output_norm = nn.GroupNorm(_group_count(output_channels), output_channels)
        condition_linear: nn.Module
        if independent_device_lora_count:
            condition_linear = IndependentDeviceLoRALinear(
                embedding_dimension,
                2 * output_channels,
                device_lora_rank,
                independent_device_lora_count,
            )
        elif device_lora_rank:
            condition_linear = RoutedLoRALinear(
                embedding_dimension,
                2 * output_channels,
                device_lora_rank,
            )
        else:
            condition_linear = nn.Linear(embedding_dimension, 2 * output_channels)
        self.condition_projection = nn.Sequential(nn.SiLU(), condition_linear)
        self.device_lora_rank = device_lora_rank
        self.independent_device_lora_count = independent_device_lora_count
        self.output_convolution = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(
        self,
        signal: torch.Tensor,
        conditioning: torch.Tensor,
        device_adapter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.input_convolution(F.silu(self.input_norm(signal)))
        activated_condition = self.condition_projection[0](conditioning)
        projection = self.condition_projection[1]
        if self.device_lora_rank:
            if device_adapter is None:
                raise ValueError("device LoRA residual block requires a device route")
            projected_condition = projection(activated_condition, device_adapter)
        else:
            if device_adapter is not None:
                raise ValueError("device route was provided to a shared residual block")
            projected_condition = projection(activated_condition)
        shift, scale = projected_condition.chunk(2, dim=1)
        hidden = self.output_norm(hidden)
        hidden = hidden * (1.0 + scale[:, :, None]) + shift[:, :, None]
        hidden = self.output_convolution(F.silu(hidden))
        return self.skip(signal) + hidden


class ConditionalSelfAttention1D(nn.Module):
    def __init__(
        self,
        channels: int,
        embedding_dimension: int,
        num_heads: int,
        parameterized_qkv: bool = False,
        conditioning_rank: int = 0,
        adaptive_attention_normalization: bool = False,
        query_chunk_size: int = 0,
        target_context_radius: int = 0,
        relative_condition_dimension: int = 0,
        factorized_multi_source_attention: bool = False,
        anchor_content_multi_source_style: bool = False,
        residual_multi_source_style_attention: bool = False,
        spectral_style_token_attention: bool = False,
        frequency_resolved_style_attention: bool = False,
        style_spectral_coordinate: str = "fixed_frequency",
        style_sample_rate_hz: float = 0.0,
        style_maximum_order: float = 32.0,
        direct_adaattn_style_modulation: bool = False,
        fourier_value_style_attention: bool = False,
        hull_residual_dimension: int = 0,
        parameter_conditioned_self_attention: bool = False,
        style_condition_dimension: int = 0,
        style_condition_rank: int = 0,
        local_style_statistics: bool = False,
        device_lora_rank: int = 0,
        independent_device_lora_count: int = 0,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("attention channels must be divisible by num_heads")
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.condition_projection = nn.Linear(embedding_dimension, channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            batch_first=True,
        )
        self.output_projection = nn.Linear(channels, channels)
        self.parameterized_qkv = parameterized_qkv
        self.conditioning_rank = conditioning_rank
        self.adaptive_attention_normalization = adaptive_attention_normalization
        self.query_chunk_size = query_chunk_size
        self.target_context_radius = target_context_radius
        self.relative_condition_dimension = relative_condition_dimension
        self.factorized_multi_source_attention = factorized_multi_source_attention
        self.anchor_content_multi_source_style = anchor_content_multi_source_style
        self.residual_multi_source_style_attention = residual_multi_source_style_attention
        self.spectral_style_token_attention = spectral_style_token_attention
        self.frequency_resolved_style_attention = frequency_resolved_style_attention
        self.style_spectral_coordinate = style_spectral_coordinate
        self.style_sample_rate_hz = float(style_sample_rate_hz)
        self.style_maximum_order = float(style_maximum_order)
        self.direct_adaattn_style_modulation = direct_adaattn_style_modulation
        self.fourier_value_style_attention = fourier_value_style_attention
        self.hull_residual_dimension = hull_residual_dimension
        self.parameter_conditioned_self_attention = parameter_conditioned_self_attention
        self.style_condition_dimension = style_condition_dimension
        self.local_style_statistics = local_style_statistics
        self.device_lora_rank = device_lora_rank
        self.independent_device_lora_count = independent_device_lora_count
        if independent_device_lora_count and not device_lora_rank:
            raise ValueError("independent device LoRA requires a positive rank")
        if independent_device_lora_count == 1:
            raise ValueError("independent device LoRA requires multiple devices")
        if query_chunk_size < 0 or target_context_radius < 0:
            raise ValueError("attention chunk size and context radius cannot be negative")
        if bool(query_chunk_size) != bool(target_context_radius):
            raise ValueError("attention chunking requires both chunk size and context radius")
        if adaptive_attention_normalization and not parameterized_qkv:
            raise ValueError("adaptive attention normalization requires parameterized QKV")
        if relative_condition_dimension < 0:
            raise ValueError("relative condition dimension cannot be negative")
        if relative_condition_dimension and not parameterized_qkv:
            raise ValueError("relative condition modulation requires parameterized QKV")
        if factorized_multi_source_attention and not parameterized_qkv:
            raise ValueError("multi-source attention requires parameterized QKV")
        if anchor_content_multi_source_style and not factorized_multi_source_attention:
            raise ValueError("multi-source style requires factorized multi-source attention")
        if residual_multi_source_style_attention and not factorized_multi_source_attention:
            raise ValueError("multi-source residual requires factorized attention memory")
        if spectral_style_token_attention and not residual_multi_source_style_attention:
            raise ValueError("spectral style tokens require multi-source residual attention")
        if spectral_style_token_attention and embedding_dimension % num_heads:
            raise ValueError("spectral style embedding must divide evenly across heads")
        if frequency_resolved_style_attention and not residual_multi_source_style_attention:
            raise ValueError("frequency-resolved style requires multi-source residual attention")
        if frequency_resolved_style_attention and spectral_style_token_attention:
            raise ValueError("frequency-resolved and flattened spectral style are exclusive")
        if frequency_resolved_style_attention and embedding_dimension % num_heads:
            raise ValueError("frequency-resolved style embedding must divide across heads")
        if style_spectral_coordinate not in {"fixed_frequency", "order"}:
            raise ValueError("style spectral coordinate must be fixed_frequency or order")
        if style_spectral_coordinate == "order" and not frequency_resolved_style_attention:
            raise ValueError("order style coordinates require frequency-resolved attention")
        if style_spectral_coordinate == "order" and (
            style_sample_rate_hz <= 0.0 or style_maximum_order <= 0.0
        ):
            raise ValueError("order style coordinates require positive rate and maximum order")
        if direct_adaattn_style_modulation and not (
            frequency_resolved_style_attention and adaptive_attention_normalization
        ):
            raise ValueError(
                "direct AdaAttN style modulation requires resolved style and AdaAttN"
            )
        if fourier_value_style_attention and not residual_multi_source_style_attention:
            raise ValueError("Fourier Value style requires multi-source residual attention")
        if fourier_value_style_attention and (
            spectral_style_token_attention or frequency_resolved_style_attention
        ):
            raise ValueError("Fourier Value style is exclusive with prior spectral styles")
        if hull_residual_dimension < 0:
            raise ValueError("hull residual dimension cannot be negative")
        if hull_residual_dimension and not (
            parameterized_qkv
            and factorized_multi_source_attention
            and residual_multi_source_style_attention
            and adaptive_attention_normalization
        ):
            raise ValueError(
                "hull residual modulation requires residual multi-source QKV-AdaAttN"
            )
        if parameter_conditioned_self_attention:
            if parameterized_qkv:
                raise ValueError(
                    "source-free style attention and source QKV attention are exclusive"
                )
            if style_condition_dimension <= 0:
                raise ValueError("source-free style attention requires a style context")
            if style_condition_rank <= 0 or style_condition_rank > channels:
                raise ValueError("source-free style rank must lie in [1,channels]")
            self.style_q_a = nn.Parameter(torch.empty(channels, style_condition_rank))
            self.style_q_b = nn.Parameter(torch.empty(style_condition_rank, channels))
            self.style_k_a = nn.Parameter(torch.empty(channels, style_condition_rank))
            self.style_k_b = nn.Parameter(torch.empty(style_condition_rank, channels))
            self.style_v_a = nn.Parameter(torch.empty(channels, style_condition_rank))
            self.style_v_b = nn.Parameter(torch.empty(style_condition_rank, channels))
            self.style_q_coefficients = nn.Linear(
                style_condition_dimension, style_condition_rank, bias=False
            )
            self.style_k_coefficients = nn.Linear(
                style_condition_dimension, style_condition_rank, bias=False
            )
            self.style_v_coefficients = nn.Linear(
                style_condition_dimension, style_condition_rank, bias=False
            )
            self.style_mean = nn.Linear(
                style_condition_dimension, channels, bias=False
            )
            self.style_log_scale = nn.Linear(
                style_condition_dimension, channels, bias=False
            )
            self.style_normalization_gate = nn.Parameter(torch.zeros(channels))
            if device_lora_rank:
                if device_lora_rank > channels:
                    raise ValueError("device LoRA rank exceeds attention channels")
                leading = (independent_device_lora_count,) if independent_device_lora_count else ()
                self.device_q_a = nn.Parameter(
                    torch.empty(*leading, channels, device_lora_rank)
                )
                self.device_q_b = nn.Parameter(
                    torch.empty(*leading, device_lora_rank, channels)
                )
                self.device_k_a = nn.Parameter(
                    torch.empty(*leading, channels, device_lora_rank)
                )
                self.device_k_b = nn.Parameter(
                    torch.empty(*leading, device_lora_rank, channels)
                )
                self.device_v_a = nn.Parameter(
                    torch.empty(*leading, channels, device_lora_rank)
                )
                self.device_v_b = nn.Parameter(
                    torch.empty(*leading, device_lora_rank, channels)
                )
        elif local_style_statistics:
            raise ValueError(
                "local style statistics require source-free condition attention"
            )
        elif device_lora_rank:
            raise ValueError("device QKV LoRA requires source-free style attention")
        if parameterized_qkv:
            if conditioning_rank <= 0 or conditioning_rank > channels:
                raise ValueError("conditioning rank must lie in [1,channels]")
            self.q_a = nn.Parameter(torch.empty(channels, conditioning_rank))
            self.q_b = nn.Parameter(torch.empty(conditioning_rank, channels))
            self.k_a = nn.Parameter(torch.empty(channels, conditioning_rank))
            self.k_b = nn.Parameter(torch.empty(conditioning_rank, channels))
            self.v_a = nn.Parameter(torch.empty(channels, conditioning_rank))
            self.v_b = nn.Parameter(torch.empty(conditioning_rank, channels))
            self.q_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dimension, conditioning_rank),
            )
            self.k_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dimension, conditioning_rank),
            )
            self.v_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dimension, conditioning_rank),
            )
            if adaptive_attention_normalization:
                self.adaattn_gate = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(embedding_dimension, channels),
                )
            if hull_residual_dimension:
                self.hull_residual_q = nn.Linear(
                    hull_residual_dimension, conditioning_rank, bias=False
                )
                self.hull_residual_k = nn.Linear(
                    hull_residual_dimension, conditioning_rank, bias=False
                )
                self.hull_residual_v = nn.Linear(
                    hull_residual_dimension, conditioning_rank, bias=False
                )
                self.hull_residual_adaattn = nn.Linear(
                    hull_residual_dimension, channels, bias=False
                )
            if relative_condition_dimension:
                self.relative_q = nn.Linear(
                    relative_condition_dimension,
                    conditioning_rank,
                    bias=False,
                )
                self.relative_source_k = nn.Linear(
                    relative_condition_dimension,
                    conditioning_rank,
                    bias=False,
                )
                self.relative_source_v = nn.Linear(
                    relative_condition_dimension,
                    conditioning_rank,
                    bias=False,
                )
                if adaptive_attention_normalization:
                    self.relative_adaattn = nn.Linear(
                        relative_condition_dimension,
                        channels,
                        bias=False,
                    )
            if factorized_multi_source_attention:
                self.condition_shrinkage_logits = nn.Parameter(
                    torch.full((num_heads,), math.log(0.25 / 0.75))
                )
                self.source_mix_logits = nn.Parameter(torch.zeros(num_heads))
                if residual_multi_source_style_attention:
                    self.multi_source_style_gate = nn.Parameter(torch.zeros(num_heads))
                    if fourier_value_style_attention:
                        self.fourier_value_style_gate = nn.Parameter(
                            torch.full((num_heads,), math.atanh(0.75))
                        )
                    if spectral_style_token_attention:
                        style_input_dimension = channels * 16
                        self.spectral_style_encoder = nn.Sequential(
                            nn.LayerNorm(style_input_dimension),
                            nn.Linear(style_input_dimension, embedding_dimension),
                            nn.SiLU(),
                            nn.Linear(embedding_dimension, embedding_dimension),
                        )
                        self.spectral_style_context = nn.Linear(
                            embedding_dimension, embedding_dimension, bias=False
                        )
                    if frequency_resolved_style_attention:
                        self.frequency_style_encoder = nn.Sequential(
                            nn.LayerNorm(channels),
                            nn.Linear(channels, embedding_dimension),
                            nn.SiLU(),
                            nn.Linear(embedding_dimension, embedding_dimension),
                        )
                        self.frequency_style_shrinkage_logits = nn.Parameter(
                            torch.full(
                                (16, num_heads),
                                math.log(0.25 / 0.75),
                            )
                        )
                        self.frequency_style_pool_logits = nn.Parameter(
                            torch.zeros(num_heads, 16)
                        )
                        self.frequency_style_context = nn.Linear(
                            embedding_dimension, embedding_dimension, bias=False
                        )
                        if direct_adaattn_style_modulation:
                            self.frequency_style_adaattn = nn.Linear(
                                embedding_dimension,
                                2 * channels,
                                bias=False,
                            )

    def reset_conditioning(self) -> None:
        if self.parameter_conditioned_self_attention:
            for parameter in (
                self.style_q_a,
                self.style_q_b,
                self.style_k_a,
                self.style_k_b,
                self.style_v_a,
                self.style_v_b,
            ):
                nn.init.normal_(parameter, std=0.02)
            nn.init.zeros_(self.style_q_coefficients.weight)
            nn.init.zeros_(self.style_k_coefficients.weight)
            nn.init.zeros_(self.style_v_coefficients.weight)
            nn.init.zeros_(self.style_mean.weight)
            nn.init.zeros_(self.style_log_scale.weight)
            nn.init.zeros_(self.style_normalization_gate)
            if self.device_lora_rank:
                for parameter in (
                    self.device_q_a,
                    self.device_k_a,
                    self.device_v_a,
                ):
                    nn.init.normal_(parameter, std=0.02)
                for parameter in (
                    self.device_q_b,
                    self.device_k_b,
                    self.device_v_b,
                ):
                    nn.init.zeros_(parameter)
        if not self.parameterized_qkv:
            return
        for parameter in (
            self.q_a,
            self.q_b,
            self.k_a,
            self.k_b,
            self.v_a,
            self.v_b,
        ):
            nn.init.normal_(parameter, std=0.02)
        for gate in (self.q_gate, self.k_gate, self.v_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
        if self.adaptive_attention_normalization:
            nn.init.zeros_(self.adaattn_gate[-1].weight)
            nn.init.zeros_(self.adaattn_gate[-1].bias)
        if self.hull_residual_dimension:
            nn.init.zeros_(self.hull_residual_q.weight)
            nn.init.zeros_(self.hull_residual_k.weight)
            nn.init.zeros_(self.hull_residual_v.weight)
            nn.init.zeros_(self.hull_residual_adaattn.weight)
        if self.relative_condition_dimension:
            nn.init.zeros_(self.relative_q.weight)
            nn.init.zeros_(self.relative_source_k.weight)
            nn.init.zeros_(self.relative_source_v.weight)
            if self.adaptive_attention_normalization:
                nn.init.zeros_(self.relative_adaattn.weight)
        if self.factorized_multi_source_attention:
            nn.init.constant_(
                self.condition_shrinkage_logits,
                math.log(0.25 / 0.75),
            )
            nn.init.zeros_(self.source_mix_logits)
            if self.residual_multi_source_style_attention:
                nn.init.zeros_(self.multi_source_style_gate)
                if self.fourier_value_style_attention:
                    nn.init.constant_(
                        self.fourier_value_style_gate,
                        math.atanh(0.75),
                    )
                if self.spectral_style_token_attention:
                    nn.init.zeros_(self.spectral_style_context.weight)
                if self.frequency_resolved_style_attention:
                    nn.init.constant_(
                        self.frequency_style_shrinkage_logits,
                        math.log(0.25 / 0.75),
                    )
                    nn.init.zeros_(self.frequency_style_pool_logits)
                    nn.init.zeros_(self.frequency_style_context.weight)
                    if self.direct_adaattn_style_modulation:
                        nn.init.zeros_(self.frequency_style_adaattn.weight)

    def _spectral_style_context(
        self,
        source_memory: torch.Tensor,
        source_conditions: torch.Tensor,
        target_condition: torch.Tensor,
        anchor_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, source_length, channels = source_memory.shape
        centered = source_memory - source_memory.mean(dim=2, keepdim=True)
        log_magnitude = torch.log1p(torch.fft.rfft(centered, dim=2).abs())
        frequency_count = log_magnitude.shape[2]
        pooled = F.adaptive_avg_pool1d(
            log_magnitude.permute(0, 1, 3, 2).reshape(
                batch_size * source_count * channels, 1, frequency_count
            ),
            16,
        ).reshape(batch_size, source_count, channels * 16)
        tokens = self.spectral_style_encoder(pooled)
        head_count = self.attention.num_heads
        head_dimension = tokens.shape[-1] // head_count
        tokens = tokens.reshape(batch_size, source_count, head_count, head_dimension)
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_conditions, target_condition
        )
        shrinkage = torch.sigmoid(self.condition_shrinkage_logits)[None, :, None]
        condition_weights = softmax_weights[:, None, :] + shrinkage * (
            local_linear_weights[:, None, :] - softmax_weights[:, None, :]
        )
        requested = torch.einsum(
            "bhk,bkhd->bhd", condition_weights, tokens
        )
        batch_index = torch.arange(batch_size, device=source_memory.device)
        anchor = tokens[batch_index, anchor_index]
        return self.spectral_style_context((requested - anchor).reshape(batch_size, -1))

    def _order_resolved_style_tokens(
        self,
        log_magnitude: torch.Tensor,
        source_speed_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, _, channels = log_magnitude.shape
        if source_speed_hz.shape != (batch_size, source_count):
            raise ValueError("source speeds do not align with multi-source memory")
        samples_per_band = 4
        order = (
            torch.arange(
                16 * samples_per_band,
                device=log_magnitude.device,
                dtype=log_magnitude.dtype,
            )
            + 0.5
        ) * (self.style_maximum_order / (16 * samples_per_band))
        frequency = source_speed_hz[..., None] * order
        normalized_frequency = 2.0 * frequency / (self.style_sample_rate_hz / 2.0) - 1.0
        grid = torch.stack(
            (normalized_frequency, torch.zeros_like(normalized_frequency)), dim=-1
        ).reshape(batch_size * source_count, 1, -1, 2)
        sampled = F.grid_sample(
            log_magnitude.permute(0, 1, 3, 2).reshape(
                batch_size * source_count, channels, 1, -1
            ),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(
            batch_size, source_count, channels, 16, samples_per_band
        ).mean(dim=-1).permute(0, 1, 3, 2)

    def _resolved_spectral_style_context(
        self,
        source_memory: torch.Tensor,
        source_conditions: torch.Tensor,
        target_condition: torch.Tensor,
        anchor_index: torch.Tensor,
        source_speed_hz: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, source_count, _, channels = source_memory.shape
        centered = source_memory - source_memory.mean(dim=2, keepdim=True)
        log_magnitude = torch.log1p(torch.fft.rfft(centered, dim=2).abs())
        if self.style_spectral_coordinate == "order":
            if source_speed_hz is None:
                raise ValueError("order style coordinates require source speeds")
            pooled = self._order_resolved_style_tokens(log_magnitude, source_speed_hz)
        else:
            frequency_count = log_magnitude.shape[2]
            pooled = F.adaptive_avg_pool1d(
                log_magnitude.permute(0, 1, 3, 2).reshape(
                    batch_size * source_count * channels, 1, frequency_count
                ),
                16,
            ).reshape(batch_size, source_count, channels, 16).permute(0, 1, 3, 2)
        tokens = self.frequency_style_encoder(pooled)
        head_count = self.attention.num_heads
        head_dimension = tokens.shape[-1] // head_count
        tokens = tokens.reshape(
            batch_size, source_count, 16, head_count, head_dimension
        )
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_conditions, target_condition
        )
        shrinkage = torch.sigmoid(self.frequency_style_shrinkage_logits)
        condition_weights = softmax_weights[:, None, None, :] + shrinkage[
            None, :, :, None
        ] * (
            local_linear_weights[:, None, None, :]
            - softmax_weights[:, None, None, :]
        )
        requested = torch.einsum("bfhs,bsfhd->bfhd", condition_weights, tokens)
        batch_index = torch.arange(batch_size, device=source_memory.device)
        anchor = tokens[batch_index, anchor_index]
        frequency_weights = torch.softmax(self.frequency_style_pool_logits, dim=1)
        difference = torch.einsum(
            "hf,bfhd->bhd", frequency_weights, requested - anchor
        )
        difference = difference.reshape(batch_size, -1)
        context = self.frequency_style_context(difference)
        adaattn = (
            self.frequency_style_adaattn(difference)
            if self.direct_adaattn_style_modulation
            else None
        )
        return context, adaattn

    def _fourier_styled_anchor_value(
        self,
        source_memory: torch.Tensor,
        source_conditions: torch.Tensor,
        target_condition: torch.Tensor,
        anchor_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, source_length, channels = source_memory.shape
        head_count = self.attention.num_heads
        head_dimension = channels // head_count
        centered = source_memory - source_memory.mean(dim=2, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=2)
        log_magnitude = torch.log(spectrum.abs() + 1e-6).reshape(
            batch_size,
            source_count,
            spectrum.shape[2],
            head_count,
            head_dimension,
        )
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_conditions, target_condition
        )
        shrinkage = torch.sigmoid(self.condition_shrinkage_logits)[None, :, None]
        condition_weights = softmax_weights[:, None, :] + shrinkage * (
            local_linear_weights[:, None, :] - softmax_weights[:, None, :]
        )
        requested_log_magnitude = torch.einsum(
            "bhs,bsfhd->bfhd", condition_weights, log_magnitude
        )
        batch_index = torch.arange(batch_size, device=source_memory.device)
        anchor_memory = source_memory[batch_index, anchor_index]
        anchor_spectrum = torch.fft.rfft(
            anchor_memory - anchor_memory.mean(dim=1, keepdim=True),
            dim=1,
        ).reshape(
            batch_size,
            spectrum.shape[2],
            head_count,
            head_dimension,
        )
        anchor_log_magnitude = log_magnitude[batch_index, anchor_index]
        gain = torch.tanh(self.fourier_value_style_gate)[None, None, :, None]
        log_gain = (gain * (requested_log_magnitude - anchor_log_magnitude)).clamp(
            min=-2.0,
            max=2.0,
        )
        log_gain[:, 0] = 0.0
        styled_spectrum = anchor_spectrum * log_gain.exp()
        styled = torch.fft.irfft(
            styled_spectrum.reshape(batch_size, spectrum.shape[2], channels),
            n=source_length,
            dim=1,
        )
        return styled + anchor_memory.mean(dim=1, keepdim=True)

    def _residual_multi_source_style_attention(
        self,
        query_tokens: torch.Tensor,
        target_memory: torch.Tensor,
        source_memory: torch.Tensor,
        query_conditions: torch.Tensor,
        source_memory_conditions: torch.Tensor,
        source_conditions: torch.Tensor,
        target_condition: torch.Tensor,
        source_speed_hz: torch.Tensor | None,
        precomputed_anchor_value: torch.Tensor | None,
        hull_residual: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, source_count, source_length, channels = source_memory.shape
        query_length = query_tokens.shape[1]
        head_count = self.attention.num_heads
        head_dimension = channels // head_count
        expected_context = (batch_size, source_count, query_conditions.shape[-1])
        if query_conditions.shape != expected_context:
            raise ValueError("multi-source query contexts do not align with memory")
        if source_memory_conditions.shape != expected_context:
            raise ValueError("multi-source key/value contexts do not align with memory")
        anchor_index = torch.argmin(
            (source_conditions - target_condition[:, None, :])
            .square()
            .sum(dim=2),
            dim=1,
        )
        if self.spectral_style_token_attention:
            style_context = self._spectral_style_context(
                source_memory,
                source_conditions,
                target_condition,
                anchor_index,
            )
            query_conditions = query_conditions + style_context[:, None, :]
        resolved_style_adaattn = None
        if self.frequency_resolved_style_attention:
            style_context, resolved_style_adaattn = self._resolved_spectral_style_context(
                source_memory,
                source_conditions,
                target_condition,
                anchor_index,
                source_speed_hz,
            )
            query_conditions = query_conditions + style_context[:, None, :]
        batch_index = torch.arange(batch_size, device=query_tokens.device)
        anchor_memory = source_memory[batch_index, anchor_index]
        if precomputed_anchor_value is not None:
            if precomputed_anchor_value.shape != anchor_memory.shape:
                raise ValueError("precomputed anchor Value shape differs from anchor memory")
            anchor_value_memory = precomputed_anchor_value
        elif self.fourier_value_style_attention:
            anchor_value_memory = self._fourier_styled_anchor_value(
                source_memory,
                source_conditions,
                target_condition,
                anchor_index,
            )
        else:
            anchor_value_memory = anchor_memory
        anchor_query_condition = query_conditions[batch_index, anchor_index]
        anchor_source_condition = source_memory_conditions[batch_index, anchor_index]
        base = self._parameterized_attention(
            query_tokens,
            target_memory,
            anchor_memory,
            anchor_query_condition,
            anchor_query_condition,
            anchor_source_condition,
            None,
            anchor_value_memory,
            hull_residual,
        )
        if not self.adaptive_attention_normalization:
            return base

        q_weight, _, v_weight = self.attention.in_proj_weight.chunk(3, dim=0)
        _, k_weight, _ = self.attention.in_proj_weight.chunk(3, dim=0)
        if self.attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attention.in_proj_bias.chunk(3, dim=0)
        repeated_query = query_tokens[:, None].expand(-1, source_count, -1, -1)
        flat_query = repeated_query.reshape(
            batch_size * source_count, query_length, channels
        )
        flat_source = source_memory.reshape(
            batch_size * source_count, source_length, channels
        )
        flat_query_condition = query_conditions.reshape(batch_size * source_count, -1)
        flat_source_condition = source_memory_conditions.reshape(
            batch_size * source_count, -1
        )
        query = F.linear(flat_query, q_weight, q_bias) + self._low_rank_delta(
            flat_query,
            flat_query_condition,
            self.q_a,
            self.q_b,
            self.q_gate,
        )
        source_key = F.linear(flat_source, k_weight, k_bias) + self._low_rank_delta(
            flat_source,
            flat_source_condition,
            self.k_a,
            self.k_b,
            self.k_gate,
        )
        source_value = F.linear(flat_source, v_weight, v_bias) + self._low_rank_delta(
            flat_source,
            flat_source_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
        )
        target_value = F.linear(target_memory, v_weight, v_bias) + self._low_rank_delta(
            target_memory,
            anchor_query_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
        )
        query_heads = query.reshape(
            batch_size,
            source_count,
            query_length,
            head_count,
            head_dimension,
        ).permute(0, 3, 1, 2, 4)
        source_key_heads = source_key.reshape(
            batch_size,
            source_count,
            source_length,
            head_count,
            head_dimension,
        ).permute(0, 3, 1, 2, 4)
        source_value_heads = source_value.reshape(
            batch_size,
            source_count,
            source_length,
            head_count,
            head_dimension,
        ).permute(0, 3, 1, 2, 4)
        target_value_heads = target_value.reshape(
            batch_size, query_length, head_count, head_dimension
        ).transpose(1, 2)
        target_mean = target_value_heads.mean(dim=2, keepdim=True)
        target_variance = target_value_heads.var(dim=2, correction=0, keepdim=True)
        normalized_target = (target_value_heads - target_mean) / (
            target_variance + 1e-6
        ).sqrt()
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_conditions, target_condition
        )
        shrinkage = torch.sigmoid(self.condition_shrinkage_logits)[None, :, None]
        condition_weights = softmax_weights[:, None, :] + shrinkage * (
            local_linear_weights[:, None, :] - softmax_weights[:, None, :]
        )
        anchor_weights = F.one_hot(anchor_index, num_classes=source_count).to(
            source_memory.dtype
        )
        residual_gate = torch.tanh(self.multi_source_style_gate)[
            None, :, None, None
        ]
        residual_blocks: list[torch.Tensor] = []
        block_size = self.query_chunk_size or query_length
        for begin in range(0, query_length, block_size):
            end = min(begin + block_size, query_length)
            logits = torch.einsum(
                "bhkqd,bhkmd->bhkqm",
                query_heads[:, :, :, begin:end],
                source_key_heads,
            )
            attention = torch.softmax(logits / math.sqrt(head_dimension), dim=-1)
            source_mean = torch.einsum(
                "bhkqm,bhkmd->bhkqd", attention, source_value_heads
            )
            source_second_moment = torch.einsum(
                "bhkqm,bhkmd->bhkqd", attention, source_value_heads.square()
            )
            source_standard_deviation = (
                source_second_moment - source_mean.square()
            ).clamp_min(1e-6).sqrt()
            style_log_standard_deviation = torch.einsum(
                "bhk,bhkqd->bhqd",
                condition_weights,
                source_standard_deviation.log(),
            )
            anchor_log_standard_deviation = torch.einsum(
                "bk,bhkqd->bhqd",
                anchor_weights,
                source_standard_deviation.log(),
            )
            scale_residual = (
                style_log_standard_deviation.exp()
                - anchor_log_standard_deviation.exp()
            )
            if resolved_style_adaattn is None:
                residual_blocks.append(
                    residual_gate
                    * scale_residual
                    * normalized_target[:, :, begin:end]
                )
            else:
                mean_gate, scale_gate = resolved_style_adaattn.chunk(2, dim=1)
                mean_gate = torch.tanh(mean_gate).reshape(
                    batch_size, head_count, 1, head_dimension
                )
                scale_gate = torch.tanh(scale_gate).reshape(
                    batch_size, head_count, 1, head_dimension
                )
                style_mean = torch.einsum(
                    "bhk,bhkqd->bhqd", condition_weights, source_mean
                )
                anchor_mean = torch.einsum(
                    "bk,bhkqd->bhqd", anchor_weights, source_mean
                )
                residual_blocks.append(
                    mean_gate * (style_mean - anchor_mean)
                    + scale_gate
                    * scale_residual
                    * normalized_target[:, :, begin:end]
                )
        residual = torch.cat(residual_blocks, dim=2).transpose(1, 2).reshape(
            batch_size, query_length, channels
        )
        residual = F.linear(
            residual,
            self.attention.out_proj.weight,
            None,
        )
        return base + residual

    def _factorized_multi_source_attention(
        self,
        query_tokens: torch.Tensor,
        target_memory: torch.Tensor,
        source_memory: torch.Tensor,
        query_condition: torch.Tensor,
        source_memory_condition: torch.Tensor,
        source_conditions: torch.Tensor,
        target_condition: torch.Tensor,
        transport_time: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, source_count, source_length, channels = source_memory.shape
        query_length = query_tokens.shape[1]
        head_count = self.attention.num_heads
        head_dimension = channels // head_count
        if source_memory_condition.shape[:2] != (batch_size, source_count):
            raise ValueError("source memory conditions do not align with source memory")
        if transport_time.shape != (batch_size,) or torch.any(
            (transport_time < 0.0) | (transport_time > 1.0)
        ):
            raise ValueError("transport time must lie in [0,1]")

        q_weight, k_weight, v_weight = self.attention.in_proj_weight.chunk(3, dim=0)
        if self.attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attention.in_proj_bias.chunk(3, dim=0)
        query = F.linear(query_tokens, q_weight, q_bias) + self._low_rank_delta(
            query_tokens,
            query_condition,
            self.q_a,
            self.q_b,
            self.q_gate,
        )
        target_key = F.linear(target_memory, k_weight, k_bias) + self._low_rank_delta(
            target_memory,
            query_condition,
            self.k_a,
            self.k_b,
            self.k_gate,
        )
        target_value = F.linear(target_memory, v_weight, v_bias) + self._low_rank_delta(
            target_memory,
            query_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
        )
        flat_source = source_memory.reshape(
            batch_size * source_count, source_length, channels
        )
        flat_condition = source_memory_condition.reshape(batch_size * source_count, -1)
        source_key = F.linear(flat_source, k_weight, k_bias) + self._low_rank_delta(
            flat_source,
            flat_condition,
            self.k_a,
            self.k_b,
            self.k_gate,
        )
        source_value = F.linear(flat_source, v_weight, v_bias) + self._low_rank_delta(
            flat_source,
            flat_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
        )
        source_key = source_key.reshape(
            batch_size, source_count, source_length, channels
        )
        source_value = source_value.reshape(
            batch_size, source_count, source_length, channels
        )

        def heads(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.reshape(
                batch_size, tokens.shape[1], head_count, head_dimension
            ).transpose(1, 2)

        def source_heads(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.reshape(
                batch_size,
                source_count,
                source_length,
                head_count,
                head_dimension,
            ).permute(0, 3, 1, 2, 4)

        query_heads = heads(query)
        target_key_heads = heads(target_key)
        target_value_heads = heads(target_value)
        source_key_heads = source_heads(source_key)
        source_value_heads = source_heads(source_value)
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_conditions, target_condition
        )
        shrinkage = torch.sigmoid(self.condition_shrinkage_logits)[None, :, None]
        condition_weights = softmax_weights[:, None, :] + shrinkage * (
            local_linear_weights[:, None, :] - softmax_weights[:, None, :]
        )
        anchor_index = torch.argmin(
            (source_conditions - target_condition[:, None, :])
            .square()
            .sum(dim=2),
            dim=1,
        )
        anchor_weights = F.one_hot(
            anchor_index, num_classes=source_count
        ).to(source_memory.dtype)
        source_prior = (1.0 - transport_time).clamp(0.05, 0.95)
        source_prior_logit = torch.logit(source_prior)[:, None]
        source_mix = torch.sigmoid(
            source_prior_logit + self.source_mix_logits[None, :]
        )[:, :, None, None]

        normalized_target = None
        adaattn_gate = None
        if self.adaptive_attention_normalization:
            target_mean = target_value_heads.mean(dim=2, keepdim=True)
            target_variance = target_value_heads.var(
                dim=2, correction=0, keepdim=True
            )
            normalized_target = (target_value_heads - target_mean) / (
                target_variance + 1e-6
            ).sqrt()
            adaattn_gate = torch.tanh(self.adaattn_gate(query_condition)).reshape(
                batch_size, head_count, 1, head_dimension
            )

        def attend_block(
            query_block: torch.Tensor,
            target_key_block: torch.Tensor,
            target_value_block: torch.Tensor,
            normalized_target_block: torch.Tensor | None,
        ) -> torch.Tensor:
            target_logits = query_block @ target_key_block.transpose(-2, -1)
            target_attention = torch.softmax(
                target_logits / math.sqrt(head_dimension), dim=-1
            )
            target_output = target_attention @ target_value_block
            source_logits = torch.einsum(
                "bhqd,bhkmd->bhkqm", query_block, source_key_heads
            )
            source_attention = torch.softmax(
                source_logits / math.sqrt(head_dimension), dim=-1
            )
            source_output = torch.einsum(
                "bhkqm,bhkmd->bhkqd", source_attention, source_value_heads
            )
            source_standard_deviation = None
            if self.adaptive_attention_normalization:
                if normalized_target_block is None or adaattn_gate is None:
                    raise RuntimeError("multi-source AdaAttN state is unavailable")
                source_second_moment = torch.einsum(
                    "bhkqm,bhkmd->bhkqd",
                    source_attention,
                    source_value_heads.square(),
                )
                source_standard_deviation = (
                    source_second_moment - source_output.square()
                ).clamp_min(1e-6).sqrt()
            if self.anchor_content_multi_source_style:
                combined_source = torch.einsum(
                    "bk,bhkqd->bhqd", anchor_weights, source_output
                )
                if self.adaptive_attention_normalization:
                    if source_standard_deviation is None:
                        raise RuntimeError("multi-source style statistics are unavailable")
                    log_style_standard_deviation = torch.einsum(
                        "bhk,bhkqd->bhqd",
                        condition_weights,
                        source_standard_deviation.log(),
                    ).clamp(-6.0, 6.0)
                    combined_source = combined_source + (
                        adaattn_gate
                        * log_style_standard_deviation.exp()
                        * normalized_target_block
                    )
            else:
                if self.adaptive_attention_normalization:
                    if source_standard_deviation is None:
                        raise RuntimeError("multi-source style statistics are unavailable")
                    source_output = source_output + (
                        adaattn_gate[:, :, None]
                        * source_standard_deviation
                        * normalized_target_block[:, :, None]
                    )
                combined_source = torch.einsum(
                    "bhk,bhkqd->bhqd", condition_weights, source_output
                )
            return (1.0 - source_mix) * target_output + source_mix * combined_source

        if self.query_chunk_size:
            blocks: list[torch.Tensor] = []
            for begin in range(0, query_length, self.query_chunk_size):
                end = min(begin + self.query_chunk_size, query_length)
                left = max(0, begin - self.target_context_radius)
                right = min(query_length, end + self.target_context_radius)
                blocks.append(
                    attend_block(
                        query_heads[:, :, begin:end],
                        target_key_heads[:, :, left:right],
                        target_value_heads[:, :, left:right],
                        (
                            normalized_target[:, :, begin:end]
                            if normalized_target is not None
                            else None
                        ),
                    )
                )
            attended = torch.cat(blocks, dim=2)
        else:
            attended = attend_block(
                query_heads,
                target_key_heads,
                target_value_heads,
                normalized_target,
            )
        attended = attended.transpose(1, 2).reshape(
            batch_size, query_length, channels
        )
        return F.linear(
            attended,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )

    @staticmethod
    def _low_rank_delta(
        tokens: torch.Tensor,
        context: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        gate: nn.Module,
        relative_coefficients: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coefficients = gate(context)
        if relative_coefficients is not None:
            coefficients = coefficients + relative_coefficients
        coefficients = coefficients[:, None, :]
        return ((tokens @ first) * coefficients) @ second

    def _parameter_conditioned_self_attention(
        self,
        tokens: torch.Tensor,
        style_condition: torch.Tensor,
        device_adapter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        if style_condition.shape != (batch_size, self.style_condition_dimension):
            raise ValueError("style condition does not match the configured dimension")
        if self.device_lora_rank:
            if self.independent_device_lora_count:
                if (
                    device_adapter is None
                    or device_adapter.shape != (batch_size,)
                    or device_adapter.dtype != torch.long
                ):
                    raise ValueError(
                        "independent device QKV LoRA requires [batch] device IDs"
                    )
                if torch.any(device_adapter < 0) or torch.any(
                    device_adapter >= self.independent_device_lora_count
                ):
                    raise ValueError("device ID is outside the QKV LoRA vocabulary")
            elif device_adapter is None or device_adapter.shape != (
                batch_size,
                self.device_lora_rank,
            ):
                raise ValueError("device QKV LoRA requires a [batch,rank] route")
        elif device_adapter is not None:
            raise ValueError("device route was provided to shared QKV attention")
        exact_global_context = (
            not self.query_chunk_size
            or token_count
            <= self.query_chunk_size + 2 * self.target_context_radius
        )
        if (
            not self.training
            and self._style_modulation_is_zero()
            and (
                not self.device_lora_rank
                or self._device_qkv_modulation_is_zero()
            )
            and exact_global_context
        ):
            return self.attention(tokens, tokens, tokens, need_weights=False)[0]
        q_weight, k_weight, v_weight = self.attention.in_proj_weight.chunk(3, dim=0)
        if self.attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attention.in_proj_bias.chunk(3, dim=0)

        def projected(
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            first: torch.Tensor,
            second: torch.Tensor,
            coefficients: nn.Linear,
            device_first: torch.Tensor | None = None,
            device_second: torch.Tensor | None = None,
        ) -> torch.Tensor:
            base = F.linear(tokens, weight, bias)
            condition = coefficients(style_condition)[:, None, :]
            output = base + ((tokens @ first) * condition) @ second
            if self.device_lora_rank:
                if device_first is None or device_second is None or device_adapter is None:
                    raise RuntimeError("device QKV LoRA parameters are unavailable")
                if self.independent_device_lora_count:
                    selected_first = device_first[device_adapter]
                    selected_second = device_second[device_adapter]
                    delta = torch.bmm(
                        torch.bmm(tokens, selected_first), selected_second
                    )
                else:
                    route = device_adapter[:, None, :]
                    delta = ((tokens @ device_first) * route) @ device_second
                output = output + delta / float(self.device_lora_rank)
            return output

        query = projected(
            q_weight,
            q_bias,
            self.style_q_a,
            self.style_q_b,
            self.style_q_coefficients,
            getattr(self, "device_q_a", None),
            getattr(self, "device_q_b", None),
        )
        key = projected(
            k_weight,
            k_bias,
            self.style_k_a,
            self.style_k_b,
            self.style_k_coefficients,
            getattr(self, "device_k_a", None),
            getattr(self, "device_k_b", None),
        )
        value = projected(
            v_weight,
            v_bias,
            self.style_v_a,
            self.style_v_b,
            self.style_v_coefficients,
            getattr(self, "device_v_a", None),
            getattr(self, "device_v_b", None),
        )
        head_count = self.attention.num_heads
        head_dimension = channels // head_count

        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(
                batch_size, token_count, head_count, head_dimension
            ).transpose(1, 2)

        query_heads = heads(query)
        key_heads = heads(key)
        value_heads = heads(value)
        blocks: list[torch.Tensor] = []
        requested_mean = self.style_mean(style_condition).reshape(
            batch_size, head_count, 1, head_dimension
        )
        requested_scale = self.style_log_scale(style_condition).clamp(
            min=-2.0, max=2.0
        ).exp().reshape(batch_size, head_count, 1, head_dimension)
        normalization_gate = self.style_normalization_gate.reshape(
            1, head_count, 1, head_dimension
        )
        if self.local_style_statistics:
            normalization_gate = torch.tanh(normalization_gate)
        block_size = self.query_chunk_size or token_count
        for begin in range(0, token_count, block_size):
            end = min(begin + block_size, token_count)
            if self.query_chunk_size:
                left = max(0, begin - self.target_context_radius)
                right = min(token_count, end + self.target_context_radius)
            else:
                left, right = 0, token_count
            logits = (
                query_heads[:, :, begin:end]
                @ key_heads[:, :, left:right].transpose(-2, -1)
            )
            attention = torch.softmax(logits / math.sqrt(head_dimension), dim=-1)
            local_value = value_heads[:, :, left:right]
            attended = attention @ local_value
            if self.local_style_statistics:
                second_moment = attention @ local_value.square()
                local_scale = (
                    second_moment - attended.square()
                ).clamp_min(1e-6).sqrt()
                query_value = value_heads[:, :, begin:end]
                normalized_query = (query_value - attended) / local_scale
                styled = requested_scale * normalized_query + requested_mean
                attended = attended + normalization_gate * (styled - attended)
            blocks.append(attended)
        attended_heads = torch.cat(blocks, dim=2)
        if not self.local_style_statistics:
            temporal_mean = attended_heads.mean(dim=2, keepdim=True)
            temporal_variance = attended_heads.var(dim=2, correction=0, keepdim=True)
            normalized = (attended_heads - temporal_mean) / (
                temporal_variance + 1e-6
            ).sqrt()
            styled = requested_scale * normalized + requested_mean
            attended_heads = attended_heads + normalization_gate * (
                styled - attended_heads
            )
        attended = attended_heads.transpose(1, 2).reshape(
            batch_size, token_count, channels
        )
        return F.linear(
            attended,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )

    def _style_modulation_is_zero(self) -> bool:
        return all(
            bool(torch.count_nonzero(parameter).item() == 0)
            for parameter in (
                self.style_q_coefficients.weight,
                self.style_k_coefficients.weight,
                self.style_v_coefficients.weight,
                self.style_normalization_gate,
            )
        )

    def _device_qkv_modulation_is_zero(self) -> bool:
        if not self.device_lora_rank:
            return True
        return all(
            bool(torch.count_nonzero(parameter).item() == 0)
            for parameter in (
                self.device_q_b,
                self.device_k_b,
                self.device_v_b,
            )
        )

    def _parameterized_attention(
        self,
        query_tokens: torch.Tensor,
        target_memory: torch.Tensor,
        source_memory: torch.Tensor,
        query_condition: torch.Tensor,
        target_memory_condition: torch.Tensor,
        source_memory_condition: torch.Tensor,
        relative_condition: torch.Tensor | None,
        source_value_memory: torch.Tensor | None = None,
        hull_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_value_memory is None:
            source_value_memory = source_memory
        if source_value_memory.shape != source_memory.shape:
            raise ValueError("source Value memory must match source key memory")
        if self.relative_condition_dimension:
            expected = (len(query_tokens), self.relative_condition_dimension)
            if relative_condition is None or relative_condition.shape != expected:
                raise ValueError(
                    "relative condition must match the configured attention dimension"
                )
            relative_q = self.relative_q(relative_condition)
            relative_source_k = self.relative_source_k(relative_condition)
            relative_source_v = self.relative_source_v(relative_condition)
        else:
            if relative_condition is not None:
                raise ValueError("relative condition was provided to legacy attention")
            relative_q = relative_source_k = relative_source_v = None
        if self.hull_residual_dimension:
            expected_residual = (len(query_tokens), self.hull_residual_dimension)
            if hull_residual is None or hull_residual.shape != expected_residual:
                raise ValueError("hull residual does not match the configured dimension")
            hull_q = self.hull_residual_q(hull_residual)
            hull_k = self.hull_residual_k(hull_residual)
            hull_v = self.hull_residual_v(hull_residual)
        else:
            if hull_residual is not None:
                raise ValueError("hull residual was provided to legacy attention")
            hull_q = hull_k = hull_v = None

        def combined_coefficients(
            first: torch.Tensor | None,
            second: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if first is None:
                return second
            if second is None:
                return first
            return first + second

        q_weight, k_weight, v_weight = self.attention.in_proj_weight.chunk(3, dim=0)
        if self.attention.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attention.in_proj_bias.chunk(3, dim=0)
        query = F.linear(query_tokens, q_weight, q_bias) + self._low_rank_delta(
            query_tokens,
            query_condition,
            self.q_a,
            self.q_b,
            self.q_gate,
            combined_coefficients(relative_q, hull_q),
        )
        target_key = F.linear(target_memory, k_weight, k_bias) + self._low_rank_delta(
            target_memory,
            target_memory_condition,
            self.k_a,
            self.k_b,
            self.k_gate,
            hull_k,
        )
        source_key = F.linear(source_memory, k_weight, k_bias) + self._low_rank_delta(
            source_memory,
            source_memory_condition,
            self.k_a,
            self.k_b,
            self.k_gate,
            combined_coefficients(relative_source_k, hull_k),
        )
        target_value = F.linear(target_memory, v_weight, v_bias) + self._low_rank_delta(
            target_memory,
            target_memory_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
            hull_v,
        )
        source_value = F.linear(
            source_value_memory, v_weight, v_bias
        ) + self._low_rank_delta(
            source_value_memory,
            source_memory_condition,
            self.v_a,
            self.v_b,
            self.v_gate,
            combined_coefficients(relative_source_v, hull_v),
        )
        key = torch.cat((target_key, source_key), dim=1)
        value = torch.cat((target_value, source_value), dim=1)
        batch_size, query_length, channels = query.shape
        head_count = self.attention.num_heads
        head_dimension = channels // head_count

        def heads(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.reshape(
                batch_size,
                tokens.shape[1],
                head_count,
                head_dimension,
            ).transpose(1, 2)

        query_heads = heads(query)
        key_heads = heads(key)
        value_heads = heads(value)
        if self.adaptive_attention_normalization:
            target_value_heads = heads(target_value)
            target_mean = target_value_heads.mean(dim=2, keepdim=True)
            target_variance = target_value_heads.var(
                dim=2,
                correction=0,
                keepdim=True,
            )
            normalized_target = (target_value_heads - target_mean) / (
                target_variance + 1e-6
            ).sqrt()
            gate_logits = self.adaattn_gate(query_condition)
            if self.relative_condition_dimension:
                if relative_condition is None:
                    raise RuntimeError("relative AdaAttN condition is unavailable")
                gate_logits = gate_logits + self.relative_adaattn(relative_condition)
            gate = torch.tanh(gate_logits)
            if self.hull_residual_dimension:
                gate = gate + self.hull_residual_adaattn(hull_residual)
            gate = gate.reshape(
                batch_size,
                head_count,
                1,
                head_dimension,
            )

        def attend(
            query_block: torch.Tensor,
            key_block: torch.Tensor,
            value_block: torch.Tensor,
            source_value_block: torch.Tensor,
            target_key_count: int,
            normalized_target_block: torch.Tensor | None,
        ) -> torch.Tensor:
            logits = query_block @ key_block.transpose(-2, -1)
            attention = torch.softmax(logits / math.sqrt(head_dimension), dim=-1)
            block = attention @ value_block
            if not self.adaptive_attention_normalization:
                return block
            if normalized_target_block is None:
                raise RuntimeError("AdaAttN target normalization was not prepared")
            source_attention = attention[..., target_key_count:]
            source_mass = source_attention.sum(dim=-1, keepdim=True)
            source_weights = source_attention / source_mass.clamp_min(1e-6)
            source_mean = source_weights @ source_value_block
            source_second_moment = source_weights @ source_value_block.square()
            source_std = (
                source_second_moment - source_mean.square()
            ).clamp_min(1e-6).sqrt()
            return block + gate * source_mass * source_std * normalized_target_block

        if self.query_chunk_size:
            source_key_heads = heads(source_key)
            source_value_heads = heads(source_value)
            target_key_heads = heads(target_key)
            target_value_heads = heads(target_value)
            blocks: list[torch.Tensor] = []
            for begin in range(0, query_length, self.query_chunk_size):
                end = min(begin + self.query_chunk_size, query_length)
                left = max(0, begin - self.target_context_radius)
                right = min(query_length, end + self.target_context_radius)
                local_key = torch.cat(
                    (target_key_heads[:, :, left:right], source_key_heads),
                    dim=2,
                )
                local_value = torch.cat(
                    (target_value_heads[:, :, left:right], source_value_heads),
                    dim=2,
                )
                blocks.append(
                    attend(
                        query_heads[:, :, begin:end],
                        local_key,
                        local_value,
                        source_value_heads,
                        right - left,
                        (
                            normalized_target[:, :, begin:end]
                            if self.adaptive_attention_normalization
                            else None
                        ),
                    )
                )
            attended = torch.cat(blocks, dim=2)
        elif self.adaptive_attention_normalization:
            attended = attend(
                query_heads,
                key_heads,
                value_heads,
                heads(source_value),
                target_memory.shape[1],
                normalized_target,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                dropout_p=0.0,
            )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            channels,
        )
        return F.linear(
            attended,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )

    def forward(
        self,
        signal: torch.Tensor,
        conditioning: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_conditioning: torch.Tensor | None = None,
        query_qkv_conditioning: torch.Tensor | None = None,
        source_qkv_conditioning: torch.Tensor | None = None,
        relative_qkv_conditioning: torch.Tensor | None = None,
        source_condition_parameters: torch.Tensor | None = None,
        target_condition_parameters: torch.Tensor | None = None,
        transport_time: torch.Tensor | None = None,
        source_speed_hz: torch.Tensor | None = None,
        memory_value: torch.Tensor | None = None,
        hull_condition_residual: torch.Tensor | None = None,
        style_conditioning: torch.Tensor | None = None,
        device_route: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.norm(signal).transpose(1, 2)
        tokens = tokens + self.condition_projection(conditioning)[:, None, :]
        key_value = tokens
        memory_tokens = None
        memory_value_tokens = None
        if memory is not None:
            if memory_conditioning is None:
                raise ValueError("memory conditioning is required with memory")
            if self.factorized_multi_source_attention:
                if (
                    memory.ndim != 4
                    or memory.shape[0] != signal.shape[0]
                    or memory.shape[2] != signal.shape[1]
                ):
                    raise ValueError(
                        "multi-source memory must have shape [batch,source,channel,length]"
                    )
                batch_size, source_count, channels, source_length = memory.shape
                if memory_conditioning.shape[:2] != (batch_size, source_count):
                    raise ValueError("multi-source memory conditioning shape differs")
                flat_memory = memory.reshape(
                    batch_size * source_count, channels, source_length
                )
                memory_tokens = self.norm(flat_memory).transpose(1, 2)
                memory_tokens = memory_tokens + self.condition_projection(
                    memory_conditioning.reshape(batch_size * source_count, -1)
                )[:, None, :]
                memory_tokens = memory_tokens.reshape(
                    batch_size, source_count, source_length, channels
                )
                if memory_value is not None:
                    if memory_value.shape != (
                        batch_size,
                        channels,
                        source_length,
                    ):
                        raise ValueError("precomputed memory Value shape differs")
                    if (
                        source_condition_parameters is None
                        or target_condition_parameters is None
                    ):
                        raise ValueError("precomputed Value requires condition parameters")
                    anchor_index = torch.argmin(
                        (
                            source_condition_parameters
                            - target_condition_parameters[:, None, :]
                        )
                        .square()
                        .sum(dim=2),
                        dim=1,
                    )
                    batch_index = torch.arange(batch_size, device=memory.device)
                    anchor_conditioning = memory_conditioning[
                        batch_index, anchor_index
                    ]
                    memory_value_tokens = self.norm(memory_value).transpose(1, 2)
                    memory_value_tokens = memory_value_tokens + self.condition_projection(
                        anchor_conditioning
                    )[:, None, :]
            else:
                if memory_value is not None:
                    raise ValueError("precomputed Value requires multi-source attention")
                if (
                    memory.shape[0] != signal.shape[0]
                    or memory.shape[1] != signal.shape[1]
                ):
                    raise ValueError("attention memory batch or channel shape differs")
                memory_tokens = self.norm(memory).transpose(1, 2)
                memory_tokens = memory_tokens + self.condition_projection(
                    memory_conditioning
                )[:, None, :]
                key_value = torch.cat((tokens, memory_tokens), dim=1)
        if self.parameter_conditioned_self_attention:
            if memory_tokens is not None:
                raise ValueError("source-free style attention does not accept memory")
            if style_conditioning is None:
                raise ValueError("source-free style attention requires style conditioning")
            attended = self._parameter_conditioned_self_attention(
                tokens,
                style_conditioning,
                device_route,
            )
        elif self.parameterized_qkv:
            if (
                memory_tokens is None
                or query_qkv_conditioning is None
                or source_qkv_conditioning is None
            ):
                raise ValueError("parameterized QKV requires source and condition contexts")
            if self.factorized_multi_source_attention:
                if source_condition_parameters is None or target_condition_parameters is None:
                    raise ValueError("multi-source attention requires condition parameters")
                if transport_time is None:
                    raise ValueError("multi-source attention requires transport time")
                if self.residual_multi_source_style_attention:
                    attended = self._residual_multi_source_style_attention(
                        tokens,
                        tokens,
                        memory_tokens,
                        query_qkv_conditioning,
                        source_qkv_conditioning,
                        source_condition_parameters,
                        target_condition_parameters,
                        source_speed_hz,
                        memory_value_tokens,
                        hull_condition_residual,
                    )
                else:
                    attended = self._factorized_multi_source_attention(
                        tokens,
                        tokens,
                        memory_tokens,
                        query_qkv_conditioning,
                        source_qkv_conditioning,
                        source_condition_parameters,
                        target_condition_parameters,
                        transport_time,
                    )
            else:
                attended = self._parameterized_attention(
                    tokens,
                    tokens,
                    memory_tokens,
                    query_qkv_conditioning,
                    query_qkv_conditioning,
                    source_qkv_conditioning,
                    relative_qkv_conditioning,
                    hull_residual=hull_condition_residual,
                )
        else:
            if style_conditioning is not None:
                raise ValueError("style conditioning was provided to legacy attention")
            attended = self.attention(tokens, key_value, key_value, need_weights=False)[0]
        attended = self.output_projection(attended).transpose(1, 2)
        return signal + attended


class PhysiUNet1D(nn.Module):
    """Paper-constrained 1D reconstruction of the DiffPhysiNet denoiser.

    The paper specifies continuous WCE, conditional attention, Fourier Top-K
    decomposition, a residual U-Net, reweighting and projection. It does not
    disclose the layer widths or Top-K value; those remain explicit constructor
    arguments here.
    """

    def __init__(
        self,
        *,
        window_length: int,
        num_faults: int,
        num_condition_parameters: int,
        base_channels: int,
        embedding_dimension: int,
        attention_heads: int,
        fourier_top_k: int,
        parameter_transform: str,
        parameter_support_minimum: np.ndarray | torch.Tensor | None = None,
        parameter_support_maximum: np.ndarray | torch.Tensor | None = None,
        source_conditioned: bool = False,
        parameterized_qkv: bool = False,
        conditioning_rank: int = 0,
        adaptive_attention_normalization: bool = False,
        variable_length: bool = False,
        query_chunk_size: int = 0,
        target_context_radius: int = 0,
        amplitude_prediction: bool = False,
        source_free_amplitude_prediction: bool = False,
        relative_condition_attention: bool = False,
        preserve_source_fourier: bool = False,
        speed_parameter_index: int | None = None,
        speed_to_hz: float = 1.0,
        parameter_mean: np.ndarray | torch.Tensor | None = None,
        parameter_standard_deviation: np.ndarray | torch.Tensor | None = None,
        factorized_multi_source_attention: bool = False,
        transport_time_scale: float = 1.0,
        anchor_content_multi_source_style: bool = False,
        residual_multi_source_style_attention: bool = False,
        spectral_style_token_attention: bool = False,
        frequency_resolved_style_attention: bool = False,
        style_spectral_coordinate: str = "fixed_frequency",
        sample_rate_hz: float = 0.0,
        style_maximum_order: float = 32.0,
        direct_adaattn_style_modulation: bool = False,
        fourier_value_style_attention: bool = False,
        signal_fourier_style_attention: bool = False,
        hull_residual_conditioning: bool = False,
        condition_style_order: int = 0,
        condition_prototype_capacity: int = 0,
        condition_attention_local_linear_fraction: float = 1.0,
        condition_affine_residual_capacity: int = 0,
        local_style_statistics: bool = False,
        num_devices: int = 0,
        device_condition_rank: int = 0,
        device_lora_rank: int = 0,
        independent_device_lora: bool = False,
    ) -> None:
        super().__init__()
        if window_length % 4:
            raise ValueError("window length must be divisible by four")
        if num_faults <= 0 or num_condition_parameters <= 0:
            raise ValueError("fault and condition dimensions must be positive")
        if fourier_top_k > window_length // 2 + 1:
            raise ValueError("fourier_top_k exceeds the rFFT frequency count")
        if parameter_transform not in {"identity", "tanh", "tanh_linear_tail"}:
            raise ValueError(
                "parameter_transform must be 'identity', 'tanh', or "
                "'tanh_linear_tail'"
            )
        self.window_length = window_length
        self.variable_length = variable_length
        self.amplitude_prediction = amplitude_prediction
        self.source_free_amplitude_prediction = source_free_amplitude_prediction
        self.base_channels = base_channels
        self.num_condition_parameters = num_condition_parameters
        self.fourier_top_k = fourier_top_k
        self.parameter_transform = parameter_transform
        self.source_conditioned = source_conditioned
        self.parameterized_qkv = parameterized_qkv
        self.relative_condition_attention = relative_condition_attention
        self.preserve_source_fourier = preserve_source_fourier
        self.factorized_multi_source_attention = factorized_multi_source_attention
        self.transport_time_scale = transport_time_scale
        self.anchor_content_multi_source_style = anchor_content_multi_source_style
        self.residual_multi_source_style_attention = residual_multi_source_style_attention
        self.spectral_style_token_attention = spectral_style_token_attention
        self.frequency_resolved_style_attention = frequency_resolved_style_attention
        self.style_spectral_coordinate = style_spectral_coordinate
        self.direct_adaattn_style_modulation = direct_adaattn_style_modulation
        self.fourier_value_style_attention = fourier_value_style_attention
        self.signal_fourier_style_attention = signal_fourier_style_attention
        self.hull_residual_conditioning = hull_residual_conditioning
        self.condition_style_order = condition_style_order
        self.condition_prototype_capacity = condition_prototype_capacity
        self.condition_affine_residual_capacity = condition_affine_residual_capacity
        self.local_style_statistics = local_style_statistics
        self.num_devices = num_devices
        self.device_condition_rank = device_condition_rank
        self.device_lora_rank = device_lora_rank
        self.independent_device_lora = independent_device_lora
        if bool(num_devices) != bool(device_condition_rank):
            raise ValueError(
                "num_devices and device_condition_rank must be enabled together"
            )
        if num_devices == 1:
            raise ValueError("a single-device model does not need device routing")
        if num_devices < 0 or device_condition_rank < 0 or device_lora_rank < 0:
            raise ValueError("device routing dimensions cannot be negative")
        if independent_device_lora and (num_devices <= 1 or device_lora_rank <= 0):
            raise ValueError(
                "independent device LoRA requires multiple devices and a positive rank"
            )
        if device_lora_rank and (
            device_lora_rank != device_condition_rank
            or source_conditioned
            or not condition_style_order
        ):
            raise ValueError(
                "device LoRA requires source-free linear style conditioning and "
                "the shared device route rank"
            )
        independent_device_lora_count = num_devices if independent_device_lora else 0
        if hull_residual_conditioning and not (
            parameter_transform == "identity"
            and parameterized_qkv
            and factorized_multi_source_attention
            and residual_multi_source_style_attention
        ):
            raise ValueError(
                "hull residual conditioning requires identity-parameter residual "
                "multi-source QKV-AdaAttN"
            )
        if condition_style_order not in {0, 1, 2}:
            raise ValueError("condition style order must be zero, one or two")
        if condition_prototype_capacity < 0:
            raise ValueError("condition prototype capacity cannot be negative")
        if condition_affine_residual_capacity < 0:
            raise ValueError("affine-residual prototype capacity cannot be negative")
        if sum(
            bool(value)
            for value in (
                condition_style_order,
                condition_prototype_capacity,
                condition_affine_residual_capacity,
            )
        ) > 1:
            raise ValueError("condition field variants are mutually exclusive")
        if condition_style_order and not (
            parameter_transform == "identity"
            and not source_conditioned
            and not parameterized_qkv
            and conditioning_rank > 0
        ):
            raise ValueError(
                "condition style field requires a source-free identity-parameter model"
            )
        if condition_affine_residual_capacity and not (
            parameter_transform == "identity" and not source_conditioned
        ):
            raise ValueError(
                "affine-residual condition field requires a source-free "
                "identity-parameter model"
            )
        if condition_prototype_capacity and not (
            parameter_transform == "identity"
            and not source_conditioned
            and not parameterized_qkv
        ):
            raise ValueError(
                "prototype condition field requires a source-free identity-parameter model"
            )
        if parameter_transform == "tanh_linear_tail":
            support_minimum = torch.as_tensor(
                parameter_support_minimum, dtype=torch.float32
            )
            support_maximum = torch.as_tensor(
                parameter_support_maximum, dtype=torch.float32
            )
            expected = (num_condition_parameters,)
            if support_minimum.shape != expected or support_maximum.shape != expected:
                raise ValueError("parameter support does not match the condition vector")
            if not torch.isfinite(support_minimum).all() or not torch.isfinite(
                support_maximum
            ).all():
                raise ValueError("parameter support must be finite")
            if torch.any(support_minimum >= support_maximum):
                raise ValueError("each parameter support interval must have positive width")
        else:
            support_minimum = torch.full(
                (num_condition_parameters,), -torch.inf, dtype=torch.float32
            )
            support_maximum = torch.full(
                (num_condition_parameters,), torch.inf, dtype=torch.float32
            )
        self.register_buffer(
            "parameter_support_minimum", support_minimum, persistent=False
        )
        self.register_buffer(
            "parameter_support_maximum", support_maximum, persistent=False
        )
        if parameterized_qkv and not source_conditioned:
            raise ValueError("parameterized QKV requires source conditioning")
        if adaptive_attention_normalization and not parameterized_qkv:
            raise ValueError("adaptive attention normalization requires parameterized QKV")
        if amplitude_prediction and source_free_amplitude_prediction:
            raise ValueError("source-conditioned and source-free amplitude heads are exclusive")
        if source_free_amplitude_prediction and source_conditioned:
            raise ValueError("source-free amplitude prediction requires a source-free model")
        if factorized_multi_source_attention and not parameterized_qkv:
            raise ValueError("multi-source attention requires parameterized QKV")
        if factorized_multi_source_attention and relative_condition_attention:
            raise ValueError("multi-source and legacy relative attention are exclusive")
        if factorized_multi_source_attention and transport_time_scale <= 0.0:
            raise ValueError("multi-source transport time scale must be positive")
        if anchor_content_multi_source_style and not factorized_multi_source_attention:
            raise ValueError("multi-source style requires factorized multi-source attention")
        if residual_multi_source_style_attention and not factorized_multi_source_attention:
            raise ValueError("multi-source residual requires factorized attention memory")
        if spectral_style_token_attention and not residual_multi_source_style_attention:
            raise ValueError("spectral style tokens require multi-source residual attention")
        if frequency_resolved_style_attention and not residual_multi_source_style_attention:
            raise ValueError("frequency-resolved style requires multi-source residual attention")
        if fourier_value_style_attention and not residual_multi_source_style_attention:
            raise ValueError("Fourier Value style requires multi-source residual attention")
        if signal_fourier_style_attention and not residual_multi_source_style_attention:
            raise ValueError("signal Fourier style requires multi-source residual attention")
        if signal_fourier_style_attention and fourier_value_style_attention:
            raise ValueError("signal and latent Fourier styles are exclusive")
        needs_physical_speed = relative_condition_attention or (
            frequency_resolved_style_attention and style_spectral_coordinate == "order"
        )
        if needs_physical_speed:
            if not parameterized_qkv or speed_parameter_index is None:
                raise ValueError(
                    "speed-aware attention requires parameterized QKV and speed"
                )
            if not 0 <= speed_parameter_index < num_condition_parameters:
                raise ValueError("speed parameter index is outside the condition vector")
            if speed_to_hz <= 0:
                raise ValueError("speed conversion must be positive")
            if (
                frequency_resolved_style_attention
                and style_spectral_coordinate == "order"
                and sample_rate_hz <= 0
            ):
                raise ValueError("order style attention requires a positive sample rate")
            mean = torch.as_tensor(parameter_mean, dtype=torch.float32)
            deviation = torch.as_tensor(
                parameter_standard_deviation,
                dtype=torch.float32,
            )
            expected = (num_condition_parameters,)
            if mean.shape != expected or deviation.shape != expected:
                raise ValueError("parameter scaling does not match the condition vector")
            if not torch.isfinite(mean).all() or not torch.isfinite(deviation).all():
                raise ValueError("parameter scaling must be finite")
            if torch.any(deviation <= 0):
                raise ValueError("parameter standard deviations must be positive")
        else:
            mean = torch.zeros(num_condition_parameters, dtype=torch.float32)
            deviation = torch.ones(num_condition_parameters, dtype=torch.float32)
        self.speed_parameter_index = speed_parameter_index
        self.speed_to_hz = float(speed_to_hz)
        self.register_buffer("physical_parameter_mean", mean, persistent=False)
        self.register_buffer("physical_parameter_deviation", deviation, persistent=False)

        self.time_encoder = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(num_condition_parameters, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        if condition_style_order:
            polynomial_dimension = num_condition_parameters
            if condition_style_order == 2:
                polynomial_dimension += (
                    num_condition_parameters * (num_condition_parameters + 1) // 2
                )
            self.condition_style_field = nn.Linear(
                polynomial_dimension,
                embedding_dimension,
            )
        if condition_prototype_capacity:
            self.condition_prototype_field = LocalLinearConditionField(
                num_condition_parameters,
                embedding_dimension,
                condition_prototype_capacity,
                condition_attention_local_linear_fraction,
            )
        if condition_affine_residual_capacity:
            self.condition_affine_residual_field = AffineResidualConditionField(
                num_condition_parameters,
                embedding_dimension,
                condition_affine_residual_capacity,
            )
        if num_devices:
            self.device_condition_field = DeviceRoutedLowRankCondition(
                num_condition_parameters,
                embedding_dimension,
                num_devices,
                device_condition_rank,
            )
        if hull_residual_conditioning:
            self.hull_residual_context = nn.Linear(
                num_condition_parameters,
                embedding_dimension,
                bias=False,
            )
        self.fault_embedding = nn.Embedding(num_faults, embedding_dimension)
        if signal_fourier_style_attention:
            self.source_spectrum_shrinkage_logit = nn.Parameter(
                torch.tensor(math.log(0.25 / 0.75))
            )
            self.source_spectrum_style_gate = nn.Parameter(
                torch.tensor(math.atanh(0.75))
            )

        self.lifting = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        if preserve_source_fourier:
            self.source_fourier_gate = nn.Parameter(
                torch.zeros(1, base_channels, 1)
            )
        self.encoder_0 = ConditionalResidualBlock(
            base_channels,
            base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.down_0 = nn.Conv1d(
            base_channels,
            2 * base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_1 = ConditionalResidualBlock(
            2 * base_channels,
            2 * base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.down_1 = nn.Conv1d(
            2 * base_channels,
            4 * base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.middle_0 = ConditionalResidualBlock(
            4 * base_channels,
            4 * base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.middle_attention = ConditionalSelfAttention1D(
            4 * base_channels,
            embedding_dimension,
            attention_heads,
            parameterized_qkv,
            conditioning_rank,
            adaptive_attention_normalization,
            query_chunk_size,
            target_context_radius,
            num_condition_parameters + 1 if relative_condition_attention else 0,
            factorized_multi_source_attention,
            anchor_content_multi_source_style,
            residual_multi_source_style_attention,
            spectral_style_token_attention,
            frequency_resolved_style_attention,
            style_spectral_coordinate,
            sample_rate_hz / 4.0,
            style_maximum_order,
            direct_adaattn_style_modulation,
            fourier_value_style_attention,
            num_condition_parameters if hull_residual_conditioning else 0,
            bool(
                condition_style_order
                or condition_prototype_capacity
                or condition_affine_residual_capacity
            )
            and conditioning_rank > 0,
            (
                embedding_dimension
                if (
                    condition_style_order
                    or condition_prototype_capacity
                    or condition_affine_residual_capacity
                )
                and conditioning_rank > 0
                else 0
            ),
            (
                conditioning_rank
                if (
                    condition_style_order
                    or condition_prototype_capacity
                    or condition_affine_residual_capacity
                )
                and conditioning_rank > 0
                else 0
            ),
            local_style_statistics,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.middle_1 = ConditionalResidualBlock(
            4 * base_channels,
            4 * base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.up_1 = nn.Conv1d(4 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.decoder_1 = ConditionalResidualBlock(
            4 * base_channels,
            2 * base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.up_0 = nn.Conv1d(2 * base_channels, base_channels, kernel_size=3, padding=1)
        self.decoder_0 = ConditionalResidualBlock(
            2 * base_channels,
            base_channels,
            embedding_dimension,
            device_lora_rank,
            independent_device_lora_count,
        )
        self.fault_reweight = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=1),
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.projection = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)
        if amplitude_prediction:
            self.amplitude_head = nn.Sequential(
                nn.Linear(8 * base_channels + embedding_dimension + 2, embedding_dimension),
                nn.SiLU(),
                nn.Linear(embedding_dimension, 1),
            )
        if source_free_amplitude_prediction:
            self.amplitude_fault_coefficients = nn.Embedding(num_faults, 2)
            self.amplitude_condition_slope = nn.Linear(
                num_condition_parameters,
                1,
                bias=False,
            )
            if num_devices:
                self.amplitude_device_bias = nn.Embedding(num_devices, 1)
                self.amplitude_device_slope = nn.Embedding(
                    num_devices,
                    num_condition_parameters,
                )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.fault_embedding.weight, std=0.02)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if self.amplitude_prediction:
            nn.init.zeros_(self.amplitude_head[-1].weight)
            nn.init.zeros_(self.amplitude_head[-1].bias)
        if self.source_free_amplitude_prediction:
            nn.init.zeros_(self.amplitude_fault_coefficients.weight)
            nn.init.zeros_(self.amplitude_condition_slope.weight)
            if self.num_devices:
                nn.init.zeros_(self.amplitude_device_bias.weight)
                nn.init.zeros_(self.amplitude_device_slope.weight)
        if self.num_devices:
            self.device_condition_field.reset_parameters()
        self.middle_attention.reset_conditioning()
        if self.preserve_source_fourier:
            nn.init.zeros_(self.source_fourier_gate)
        if self.signal_fourier_style_attention:
            nn.init.constant_(
                self.source_spectrum_shrinkage_logit,
                math.log(0.25 / 0.75),
            )
            nn.init.constant_(self.source_spectrum_style_gate, math.atanh(0.75))
        if self.hull_residual_conditioning:
            nn.init.zeros_(self.hull_residual_context.weight)

    def _source_encoder_signal(self, lifted: torch.Tensor) -> torch.Tensor:
        periodic = topk_fourier_component(lifted, self.fourier_top_k)
        residual = lifted - periodic
        if self.preserve_source_fourier:
            residual = residual + torch.tanh(self.source_fourier_gate) * periodic
        return residual

    def _styled_source_signal(
        self,
        source_signal: torch.Tensor,
        source_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, source_count, _, source_length = source_signal.shape
        anchor_index = torch.argmin(
            (source_parameters - target_parameters[:, None, :])
            .square()
            .sum(dim=2),
            dim=1,
        )
        centered = source_signal - source_signal.mean(dim=3, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=3).squeeze(2)
        log_magnitude = torch.log(spectrum.abs() + 1e-6)
        softmax_weights, local_linear_weights = torch_condition_weight_family(
            source_parameters, target_parameters
        )
        shrinkage = torch.sigmoid(self.source_spectrum_shrinkage_logit)
        condition_weights = softmax_weights + shrinkage * (
            local_linear_weights - softmax_weights
        )
        requested_log_magnitude = torch.einsum(
            "bs,bsf->bf", condition_weights, log_magnitude
        )
        batch_index = torch.arange(batch_size, device=source_signal.device)
        anchor_signal = source_signal[batch_index, anchor_index]
        anchor_centered = anchor_signal - anchor_signal.mean(dim=2, keepdim=True)
        anchor_spectrum = torch.fft.rfft(anchor_centered, dim=2).squeeze(1)
        anchor_log_magnitude = log_magnitude[batch_index, anchor_index]
        log_gain = (
            torch.tanh(self.source_spectrum_style_gate)
            * (requested_log_magnitude - anchor_log_magnitude)
        ).clamp(min=-2.0, max=2.0)
        log_gain[:, 0] = 0.0
        styled = torch.fft.irfft(
            anchor_spectrum * log_gain.exp(),
            n=source_length,
            dim=1,
        )[:, None, :]
        return styled + anchor_signal.mean(dim=2, keepdim=True), anchor_index

    def physical_speed_hz(self, normalized_parameters: torch.Tensor) -> torch.Tensor:
        if (
            not self.relative_condition_attention
            and self.style_spectral_coordinate != "order"
        ) or self.speed_parameter_index is None:
            raise ValueError("physical speed requires relative condition attention")
        physical = (
            normalized_parameters * self.physical_parameter_deviation
            + self.physical_parameter_mean
        )
        speed = physical[:, self.speed_parameter_index] * self.speed_to_hz
        if torch.any(speed <= 0):
            raise ValueError("physical speed must be positive")
        return speed

    def _relative_condition_features(
        self,
        source_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
    ) -> torch.Tensor:
        source_speed = self.physical_speed_hz(source_parameters)
        target_speed = self.physical_speed_hz(target_parameters)
        return torch.cat(
            (
                target_parameters - source_parameters,
                torch.log(target_speed / source_speed)[:, None],
            ),
            dim=1,
        )

    def predict_log_amplitude(
        self,
        source_signal: torch.Tensor,
        source_log_amplitude: torch.Tensor,
        source_condition_parameters: torch.Tensor,
        target_condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not self.factorized_multi_source_attention:
            return self._predict_log_amplitude_single(
                source_signal,
                source_log_amplitude,
                source_condition_parameters,
                target_condition_parameters,
                fault,
                target_lengths,
            )
        if source_signal.ndim != 4 or source_signal.shape[2] != 1:
            raise ValueError(
                "multi-source signal must have shape [batch,source,1,length]"
            )
        batch_size, source_count = source_signal.shape[:2]
        if source_log_amplitude.shape != (batch_size, source_count):
            raise ValueError("multi-source log amplitude shape differs")
        if source_condition_parameters.shape != (
            batch_size,
            source_count,
            self.num_condition_parameters,
        ):
            raise ValueError("multi-source condition parameter shape differs")
        if target_condition_parameters.shape != (
            batch_size,
            self.num_condition_parameters,
        ):
            raise ValueError("target condition parameter shape differs")
        projected_target = target_condition_parameters
        target_residual = None
        if self.hull_residual_conditioning:
            projected_target = torch_convex_hull_projection(
                source_condition_parameters,
                target_condition_parameters,
            )
            target_residual = target_condition_parameters - projected_target
        flat_prediction = self._predict_log_amplitude_single(
            source_signal.reshape(
                batch_size * source_count, 1, source_signal.shape[-1]
            ),
            source_log_amplitude.reshape(batch_size * source_count),
            source_condition_parameters.reshape(
                batch_size * source_count, self.num_condition_parameters
            ),
            projected_target[:, None, :]
            .expand(-1, source_count, -1)
            .reshape(batch_size * source_count, self.num_condition_parameters),
            fault[:, None].expand(-1, source_count).reshape(batch_size * source_count),
            target_lengths[:, None]
            .expand(-1, source_count)
            .reshape(batch_size * source_count),
            (
                target_residual[:, None, :]
                .expand(-1, source_count, -1)
                .reshape(batch_size * source_count, self.num_condition_parameters)
                if target_residual is not None
                else None
            ),
        ).reshape(batch_size, source_count)
        positive_weights, _ = torch_condition_weight_family(
            source_condition_parameters, projected_target
        )
        return (positive_weights * flat_prediction).sum(dim=1)

    def predict_source_free_log_amplitude(
        self,
        condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        target_lengths: torch.Tensor,
        device_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict segment max-absolute amplitude without a source waveform."""

        if not self.source_free_amplitude_prediction:
            raise ValueError("this model has no source-free amplitude field")
        batch_size = len(condition_parameters)
        if condition_parameters.shape != (
            batch_size,
            self.num_condition_parameters,
        ):
            raise ValueError("condition parameter shape differs")
        if fault.shape != (batch_size,):
            raise ValueError("fault must have shape [batch]")
        if target_lengths.shape != (batch_size,) or torch.any(target_lengths <= 0):
            raise ValueError("target lengths must be positive and have shape [batch]")
        if self.num_devices:
            if device_id is None or device_id.shape != (batch_size,):
                raise ValueError("device-routed amplitude requires device_id")
            if device_id.dtype != torch.long:
                raise ValueError("device_id must use torch.long")
        elif device_id is not None:
            raise ValueError("device_id was provided to a single-device model")
        coefficients = self.amplitude_fault_coefficients(fault)
        relative_log_length = torch.log(
            target_lengths.to(condition_parameters.dtype) / float(self.window_length)
        )
        prediction = (
            coefficients[:, 0]
            + self.amplitude_condition_slope(condition_parameters).squeeze(1)
            + coefficients[:, 1] * relative_log_length
        )
        if self.num_devices:
            prediction = (
                prediction
                + self.amplitude_device_bias(device_id).squeeze(1)
                + (
                    self.amplitude_device_slope(device_id)
                    * condition_parameters
                ).sum(dim=1)
            )
        return prediction

    def _predict_log_amplitude_single(
        self,
        source_signal: torch.Tensor,
        source_log_amplitude: torch.Tensor,
        source_condition_parameters: torch.Tensor,
        target_condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        target_lengths: torch.Tensor,
        hull_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.amplitude_prediction:
            raise ValueError("this model was not configured to predict amplitude")
        batch_size = len(source_signal)
        if source_signal.ndim != 3 or source_signal.shape[:2] != (batch_size, 1):
            raise ValueError("source signal must have shape [batch,1,length]")
        if source_log_amplitude.shape != (batch_size,):
            raise ValueError("source log amplitude must have shape [batch]")
        if target_lengths.shape != (batch_size,) or torch.any(target_lengths <= 0):
            raise ValueError("target lengths must be positive and have shape [batch]")
        expected_parameters = (batch_size, self.num_condition_parameters)
        if source_condition_parameters.shape != expected_parameters:
            raise ValueError("source condition parameter shape differs")
        if target_condition_parameters.shape != expected_parameters:
            raise ValueError("target condition parameter shape differs")
        if fault.shape != (batch_size,):
            raise ValueError("fault must have shape [batch]")

        padding = (-source_signal.shape[-1]) % 4
        if padding:
            source_signal = F.pad(source_signal, (0, padding), mode="reflect")
        source_conditioning = self._conditioning(
            torch.zeros(batch_size, device=source_signal.device, dtype=source_signal.dtype),
            source_condition_parameters,
            fault,
        )
        lifted = self.lifting(source_signal)
        residual = self._source_encoder_signal(lifted)
        encoder_0 = self.encoder_0(residual, source_conditioning)
        encoder_1 = self.encoder_1(self.down_0(encoder_0), source_conditioning)
        middle = self.middle_0(self.down_1(encoder_1), source_conditioning)
        pooled = torch.cat(
            (
                middle.mean(dim=-1),
                middle.std(dim=-1, correction=0),
            ),
            dim=1,
        )
        transfer = self._parameter_context(
            target_condition_parameters - source_condition_parameters
        )
        target_context = (
            self._parameter_context(target_condition_parameters)
            + transfer
            + self.fault_embedding(fault)
        )
        if self.hull_residual_conditioning:
            if hull_residual is None or hull_residual.shape != expected_parameters:
                raise ValueError("amplitude hull residual shape differs")
            target_context = target_context + self.hull_residual_context(hull_residual)
        elif hull_residual is not None:
            raise ValueError("amplitude residual was provided to a legacy model")
        relative_log_length = torch.log(
            target_lengths.to(source_signal.dtype) / float(self.window_length)
        )
        features = torch.cat(
            (
                pooled,
                target_context,
                source_log_amplitude[:, None],
                relative_log_length[:, None],
            ),
            dim=1,
        )
        residual_log_amplitude = self.amplitude_head(features).squeeze(1)
        return source_log_amplitude + residual_log_amplitude

    def _transform_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        if self.parameter_transform == "identity":
            return parameters
        transformed = torch.tanh(parameters)
        if self.parameter_transform == "tanh":
            return transformed
        lower = self.parameter_support_minimum.to(parameters)
        upper = self.parameter_support_maximum.to(parameters)
        lower_value = torch.tanh(lower)
        upper_value = torch.tanh(upper)
        lower_slope = 1.0 - lower_value.square()
        upper_slope = 1.0 - upper_value.square()
        lower_tail = lower_value + lower_slope * (parameters - lower)
        upper_tail = upper_value + upper_slope * (parameters - upper)
        return torch.where(
            parameters < lower,
            lower_tail,
            torch.where(parameters > upper, upper_tail, transformed),
        )

    def _parameter_context(
        self,
        parameters: torch.Tensor,
        device_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_affine_residual_capacity:
            context = self.condition_affine_residual_field(parameters)
        elif self.condition_prototype_capacity:
            context = self.condition_prototype_field(parameters)
        elif self.condition_style_order:
            features = [parameters]
            if self.condition_style_order == 2:
                products = [
                    parameters[:, left] * parameters[:, right]
                    for left in range(self.num_condition_parameters)
                    for right in range(left, self.num_condition_parameters)
                ]
                features.append(torch.stack(products, dim=1))
            context = self.condition_style_field(torch.cat(features, dim=1))
        else:
            context = self.parameter_encoder(self._transform_parameters(parameters))
        if self.num_devices:
            if device_id is None:
                raise ValueError("device-routed condition field requires device_id")
            context = self.device_condition_field(context, parameters, device_id)
        elif device_id is not None:
            raise ValueError("device_id was provided to a single-device model")
        return context

    def _conditioning(
        self,
        timestep: torch.Tensor,
        condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        hull_residual: torch.Tensor | None = None,
        device_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dimension = self.fault_embedding.embedding_dim
        context = (
            self.time_encoder(sinusoidal_timestep_embedding(timestep, dimension))
            + self._parameter_context(condition_parameters, device_id)
            + self.fault_embedding(fault)
        )
        if self.hull_residual_conditioning:
            if hull_residual is None:
                hull_residual = torch.zeros_like(condition_parameters)
            if hull_residual.shape != condition_parameters.shape:
                raise ValueError("hull residual shape differs from condition parameters")
            context = context + self.hull_residual_context(hull_residual)
        elif hull_residual is not None:
            raise ValueError("hull residual was provided to a legacy condition encoder")
        return context

    def forward(
        self,
        noised_signal: torch.Tensor,
        timestep: torch.Tensor,
        condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        *,
        source_signal: torch.Tensor | None = None,
        source_condition_parameters: torch.Tensor | None = None,
        projected_condition_parameters: torch.Tensor | None = None,
        condition_residual_parameters: torch.Tensor | None = None,
        device_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = len(noised_signal)
        if noised_signal.ndim != 3 or noised_signal.shape[1] != 1:
            raise ValueError("signal must have shape [batch,1,length]")
        target_length = noised_signal.shape[-1]
        if not self.variable_length and target_length != self.window_length:
            raise ValueError("signal shape does not match the configured window length")
        if target_length < 4:
            raise ValueError("signal length must be at least four")
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must have shape [batch]")
        if condition_parameters.shape != (
            batch_size,
            self.num_condition_parameters,
        ):
            raise ValueError("condition parameter shape does not match the model")
        if fault.shape != (batch_size,):
            raise ValueError("fault must have shape [batch]")
        if self.num_devices:
            if device_id is None or device_id.shape != (batch_size,):
                raise ValueError("device-routed model requires device_id with shape [batch]")
            if device_id.dtype != torch.long:
                raise ValueError("device_id must use torch.long")
        elif device_id is not None:
            raise ValueError("device_id was provided to a single-device model")
        if self.source_conditioned:
            if source_signal is None or source_condition_parameters is None:
                raise ValueError("source-conditioned model requires source inputs")
            if self.factorized_multi_source_attention:
                if (
                    source_signal.ndim != 4
                    or source_signal.shape[0] != batch_size
                    or source_signal.shape[2] != 1
                ):
                    raise ValueError(
                        "multi-source signal must have shape [batch,source,1,length]"
                    )
                if source_condition_parameters.shape != (
                    batch_size,
                    source_signal.shape[1],
                    self.num_condition_parameters,
                ):
                    raise ValueError("multi-source condition parameter shape differs")
                if not self.variable_length and source_signal.shape[-1] != target_length:
                    raise ValueError("source and target signal lengths differ")
            else:
                if source_signal.ndim != 3 or source_signal.shape[:2] != (batch_size, 1):
                    raise ValueError("source signal must have shape [batch,1,length]")
                if not self.variable_length and source_signal.shape != noised_signal.shape:
                    raise ValueError("source and target signal shapes differ")
                if source_condition_parameters.shape != condition_parameters.shape:
                    raise ValueError("source and target parameter shapes differ")
            if source_signal.shape[-1] < 4:
                raise ValueError("source signal length must be at least four")
        elif source_signal is not None or source_condition_parameters is not None:
            raise ValueError("unconditioned model does not accept source inputs")

        content_condition_parameters = condition_parameters
        hull_condition_residual = None
        if self.hull_residual_conditioning:
            if (projected_condition_parameters is None) != (
                condition_residual_parameters is None
            ):
                raise ValueError("projected condition and residual must be provided together")
            if projected_condition_parameters is None:
                content_condition_parameters = torch_convex_hull_projection(
                    source_condition_parameters,
                    condition_parameters,
                )
                hull_condition_residual = (
                    condition_parameters - content_condition_parameters
                )
            else:
                if (
                    projected_condition_parameters.shape != condition_parameters.shape
                    or condition_residual_parameters.shape != condition_parameters.shape
                ):
                    raise ValueError("projected condition tensors have invalid shapes")
                if not torch.allclose(
                    projected_condition_parameters + condition_residual_parameters,
                    condition_parameters,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError("projected condition and residual do not recover target")
                content_condition_parameters = projected_condition_parameters
                hull_condition_residual = condition_residual_parameters
        elif (
            projected_condition_parameters is not None
            or condition_residual_parameters is not None
        ):
            raise ValueError("projected conditions were provided to a legacy model")

        target_padding = (-target_length) % 4
        if target_padding:
            noised_signal = F.pad(noised_signal, (0, target_padding), mode="reflect")
        if source_signal is not None:
            source_padding = (-source_signal.shape[-1]) % 4
            if source_padding:
                if self.factorized_multi_source_attention:
                    source_shape = source_signal.shape
                    source_signal = F.pad(
                        source_signal.reshape(-1, 1, source_shape[-1]),
                        (0, source_padding),
                        mode="reflect",
                    ).reshape(
                        source_shape[0],
                        source_shape[1],
                        source_shape[2],
                        source_shape[3] + source_padding,
                    )
                else:
                    source_signal = F.pad(
                        source_signal, (0, source_padding), mode="reflect"
                    )

        conditioning = self._conditioning(
            timestep,
            content_condition_parameters,
            fault,
            hull_condition_residual,
            device_id,
        )
        device_adapter = None
        if self.device_lora_rank:
            device_adapter = (
                device_id
                if self.independent_device_lora
                else self.device_condition_field.device_coefficients(device_id)
            )
        style_conditioning = (
            self._parameter_context(condition_parameters, device_id)
            if (
                (
                    self.condition_style_order
                    or self.condition_prototype_capacity
                    or self.condition_affine_residual_capacity
                )
                and self.middle_attention.parameter_conditioned_self_attention
            )
            else None
        )
        lifted = self.lifting(noised_signal)
        fault_component = topk_fourier_component(lifted, self.fourier_top_k)
        residual = lifted - fault_component

        encoder_0 = self.encoder_0(residual, conditioning, device_adapter)
        encoder_1 = self.encoder_1(
            self.down_0(encoder_0), conditioning, device_adapter
        )
        middle = self.middle_0(
            self.down_1(encoder_1), conditioning, device_adapter
        )
        source_middle = None
        source_value_middle = None
        source_conditioning = None
        target_qkv_conditioning = None
        source_qkv_conditioning = None
        relative_qkv_conditioning = None
        if self.source_conditioned:
            if self.factorized_multi_source_attention:
                source_count = source_signal.shape[1]
                styled_source_signal = None
                styled_anchor_index = None
                if self.signal_fourier_style_attention:
                    styled_source_signal, styled_anchor_index = self._styled_source_signal(
                        source_signal,
                        source_condition_parameters,
                        condition_parameters,
                    )
                flat_source_signal = source_signal.reshape(
                    batch_size * source_count, 1, source_signal.shape[-1]
                )
                flat_source_parameters = source_condition_parameters.reshape(
                    batch_size * source_count, self.num_condition_parameters
                )
                flat_fault = fault[:, None].expand(-1, source_count).reshape(-1)
                flat_source_conditioning = self._conditioning(
                    torch.zeros(
                        batch_size * source_count,
                        device=timestep.device,
                        dtype=timestep.dtype,
                    ),
                    flat_source_parameters,
                    flat_fault,
                )
                source_lifted = self.lifting(flat_source_signal)
                source_encoder_0 = self.encoder_0(
                    self._source_encoder_signal(source_lifted),
                    flat_source_conditioning,
                )
                source_encoder_1 = self.encoder_1(
                    self.down_0(source_encoder_0), flat_source_conditioning
                )
                flat_source_middle = self.middle_0(
                    self.down_1(source_encoder_1), flat_source_conditioning
                )
                source_middle = flat_source_middle.reshape(
                    batch_size,
                    source_count,
                    flat_source_middle.shape[1],
                    flat_source_middle.shape[2],
                )
                source_conditioning = flat_source_conditioning.reshape(
                    batch_size, source_count, -1
                )
                if styled_source_signal is not None:
                    if styled_anchor_index is None:
                        raise RuntimeError("styled source anchor index is unavailable")
                    batch_index = torch.arange(batch_size, device=source_signal.device)
                    styled_conditioning = source_conditioning[
                        batch_index, styled_anchor_index
                    ]
                    styled_lifted = self.lifting(styled_source_signal)
                    styled_encoder_0 = self.encoder_0(
                        self._source_encoder_signal(styled_lifted),
                        styled_conditioning,
                    )
                    styled_encoder_1 = self.encoder_1(
                        self.down_0(styled_encoder_0),
                        styled_conditioning,
                    )
                    source_value_middle = self.middle_0(
                        self.down_1(styled_encoder_1),
                        styled_conditioning,
                    )
                target_qkv_conditioning = (
                    self.time_encoder(
                        sinusoidal_timestep_embedding(
                            timestep, self.fault_embedding.embedding_dim
                        )
                    )
                    + self._parameter_context(content_condition_parameters)
                )
                flat_target_parameters = content_condition_parameters[:, None, :].expand(
                    -1, source_count, -1
                ).reshape(batch_size * source_count, self.num_condition_parameters)
                transfer = self._parameter_context(
                    flat_target_parameters - flat_source_parameters
                )
                if self.residual_multi_source_style_attention:
                    target_qkv_conditioning = (
                        self.time_encoder(
                            sinusoidal_timestep_embedding(
                                timestep, self.fault_embedding.embedding_dim
                            )
                        )[:, None, :]
                        + self._parameter_context(content_condition_parameters)[:, None, :]
                        + transfer.reshape(batch_size, source_count, -1)
                    )
                source_qkv_conditioning = (
                    self._parameter_context(flat_source_parameters) + transfer
                ).reshape(batch_size, source_count, -1)
            else:
                source_conditioning = self._conditioning(
                    torch.zeros_like(timestep),
                    source_condition_parameters,
                    fault,
                )
                source_lifted = self.lifting(source_signal)
                source_encoder_0 = self.encoder_0(
                    self._source_encoder_signal(source_lifted),
                    source_conditioning,
                )
                source_encoder_1 = self.encoder_1(
                    self.down_0(source_encoder_0),
                    source_conditioning,
                )
                source_middle = self.middle_0(
                    self.down_1(source_encoder_1),
                    source_conditioning,
                )
                if self.parameterized_qkv:
                    transfer = self._parameter_context(
                        content_condition_parameters - source_condition_parameters
                    )
                    target_qkv_conditioning = (
                        self.time_encoder(
                            sinusoidal_timestep_embedding(
                                timestep,
                                self.fault_embedding.embedding_dim,
                            )
                        )
                        + self._parameter_context(content_condition_parameters)
                        + transfer
                    )
                    source_qkv_conditioning = (
                        self._parameter_context(source_condition_parameters)
                        + transfer
                    )
                    if self.relative_condition_attention:
                        relative_qkv_conditioning = self._relative_condition_features(
                            source_condition_parameters,
                            content_condition_parameters,
                        )
        source_style_speed = None
        if (
            self.factorized_multi_source_attention
            and self.frequency_resolved_style_attention
            and self.style_spectral_coordinate == "order"
        ):
            source_style_speed = self.physical_speed_hz(
                source_condition_parameters.reshape(-1, self.num_condition_parameters)
            ).reshape(batch_size, source_signal.shape[1])
        middle = self.middle_attention(
            middle,
            conditioning,
            source_middle,
            source_conditioning,
            target_qkv_conditioning,
            source_qkv_conditioning,
            relative_qkv_conditioning,
            source_condition_parameters,
            content_condition_parameters,
            (
                (timestep / self.transport_time_scale).clamp(0.0, 1.0)
                if self.factorized_multi_source_attention
                else None
            ),
            source_style_speed,
            source_value_middle,
            hull_condition_residual,
            style_conditioning,
            device_adapter,
        )
        middle = self.middle_1(middle, conditioning, device_adapter)

        decoder_1 = F.interpolate(middle, size=encoder_1.shape[-1], mode="nearest")
        decoder_1 = self.up_1(decoder_1)
        decoder_1 = self.decoder_1(
            torch.cat((decoder_1, encoder_1), dim=1),
            conditioning,
            device_adapter,
        )
        decoder_0 = F.interpolate(decoder_1, size=encoder_0.shape[-1], mode="nearest")
        decoder_0 = self.up_0(decoder_0)
        decoder_0 = self.decoder_0(
            torch.cat((decoder_0, encoder_0), dim=1),
            conditioning,
            device_adapter,
        )

        combined = decoder_0 + F.silu(self.fault_reweight(fault_component))
        output = self.projection(F.silu(self.output_norm(combined)))
        return output[..., :target_length]


@dataclass(frozen=True, slots=True)
class DDPMSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_previous: torch.Tensor
    posterior_variance: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.betas.shape[0])

    def to(self, device: torch.device) -> DDPMSchedule:
        return DDPMSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bars=self.alpha_bars.to(device),
            alpha_bars_previous=self.alpha_bars_previous.to(device),
            posterior_variance=self.posterior_variance.to(device),
        )


def make_linear_ddpm_schedule(
    steps: int,
    *,
    beta_start: float,
    beta_end: float,
) -> DDPMSchedule:
    if steps < 2:
        raise ValueError("DDPM requires at least two diffusion steps")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
    posterior_variance = (
        betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
    ).clamp_min(1e-20)
    return DDPMSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_previous=alpha_bars_previous,
        posterior_variance=posterior_variance,
    )


def q_sample(
    clean_signal: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    if clean_signal.shape != noise.shape:
        raise ValueError("clean signal and noise shapes differ")
    if timestep.shape != (len(clean_signal),):
        raise ValueError("timestep must have shape [batch]")
    alpha_bar = schedule.alpha_bars.index_select(0, timestep)[:, None, None]
    return alpha_bar.sqrt() * clean_signal + (1.0 - alpha_bar).sqrt() * noise


@torch.no_grad()
def ddpm_sample(
    model: PhysiUNet1D,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    state = initial_noise
    batch_size = len(state)
    for step in range(schedule.steps - 1, -1, -1):
        timestep = torch.full(
            (batch_size,),
            step,
            dtype=torch.long,
            device=state.device,
        )
        predicted_noise = model(state, timestep, condition_parameters, fault)
        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        mean = (
            state - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)
        ) / torch.sqrt(alpha)
        if step:
            noise = torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
            state = mean + torch.sqrt(schedule.posterior_variance[step]) * noise
        else:
            state = mean
    return state


def ucfilter_features(signals: np.ndarray, spectrum_bins: int) -> np.ndarray:
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError("signals must have shape [samples,1,length]")
    if spectrum_bins <= 0:
        raise ValueError("spectrum_bins must be positive")
    waveform = signals[:, 0].astype(np.float64, copy=False)
    centered = waveform - waveform.mean(axis=1, keepdims=True)
    spectrum = np.log1p(np.abs(np.fft.rfft(centered, axis=1)))
    spectrum = spectrum[:, : min(spectrum_bins, spectrum.shape[1])]
    standard_deviation = centered.std(axis=1)
    root_mean_square = np.sqrt(np.mean(np.square(waveform), axis=1))
    peak = np.max(np.abs(centered), axis=1)
    normalized = centered / np.maximum(standard_deviation[:, None], 1e-12)
    kurtosis = np.mean(np.power(normalized, 4), axis=1)
    crest_factor = peak / np.maximum(root_mean_square, 1e-12)
    statistics = np.stack(
        (
            waveform.mean(axis=1),
            standard_deviation,
            root_mean_square,
            peak,
            kurtosis,
            crest_factor,
        ),
        axis=1,
    )
    features = np.concatenate((spectrum, statistics), axis=1)
    if not np.isfinite(features).all():
        raise RuntimeError("UCFilter features contain non-finite values")
    return features.astype(np.float32)


def ucfilter_select(
    signals: np.ndarray,
    intended_fault: np.ndarray,
    *,
    retain_per_fault: int,
    num_faults: int,
    spectrum_bins: int,
    pca_components: int,
    kmeans_restarts: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select cluster-central samples without reading real target signals."""

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if intended_fault.shape != (len(signals),):
        raise ValueError("intended_fault must align with signals")
    if retain_per_fault <= 0 or num_faults <= 0:
        raise ValueError("retain_per_fault and num_faults must be positive")
    counts = np.bincount(intended_fault, minlength=num_faults)
    if len(counts) != num_faults or np.any(counts < retain_per_fault):
        raise ValueError("every fault needs enough candidates for balanced filtering")

    features = ucfilter_features(signals, spectrum_bins)
    standardized = StandardScaler().fit_transform(features)
    component_count = min(pca_components, len(standardized) - 1, standardized.shape[1])
    if component_count <= 0:
        raise ValueError("not enough samples for UCFilter PCA")
    reduced = PCA(
        n_components=component_count,
        whiten=True,
        svd_solver="full",
    ).fit_transform(standardized)
    clusterer = KMeans(
        n_clusters=num_faults,
        n_init=kmeans_restarts,
        random_state=seed,
    ).fit(reduced)
    distance = np.linalg.norm(
        reduced - clusterer.cluster_centers_[clusterer.labels_],
        axis=1,
    )

    selected_parts: list[np.ndarray] = []
    for fault_id in range(num_faults):
        candidates = np.flatnonzero(intended_fault == fault_id)
        ranked = candidates[np.argsort(distance[candidates], kind="stable")]
        selected_parts.append(ranked[:retain_per_fault])
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    selected_counts = np.bincount(intended_fault[selected], minlength=num_faults)
    if np.any(selected_counts != retain_per_fault):
        raise RuntimeError("UCFilter did not preserve the requested class balance")
    report: dict[str, object] = {
        "candidate_count": int(len(signals)),
        "selected_count": int(len(selected)),
        "candidate_per_fault": counts.astype(int).tolist(),
        "selected_per_fault": selected_counts.astype(int).tolist(),
        "actual_retention_fraction": float(len(selected) / len(signals)),
        "feature_dimension": int(features.shape[1]),
        "pca_components": int(component_count),
        "kmeans_clusters": int(num_faults),
        "kmeans_inertia": float(clusterer.inertia_),
        "selected_distance_mean": float(distance[selected].mean()),
        "rejected_distance_mean": float(
            distance[np.setdiff1d(np.arange(len(signals)), selected)].mean()
        ),
    }
    return selected, report
