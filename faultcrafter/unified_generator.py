from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from faultcrafter.fault_composition import embed_fault_condition, validate_fault_condition
from faultcrafter.identity_guidance import IDENTITY_GUIDANCE_MODES, cap_guidance_rms, project_unit_peak
from faultcrafter.time_guidance import CONSTANT_GUIDANCE, TIME_GUIDANCE_SCHEDULES, scheduled_guidance_weight
from faultcrafter.training_numerics import decoder_linear_resize
from faultcrafter.physical_reference import LEGACY_CYCLIC_REFERENCE, resolve_physical_reference, validate_reference_policy


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


def timestep_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    half = dimension // 2
    if half == 0:
        return time[:, None]
    frequencies = torch.exp(
        torch.linspace(
            0.0,
            -math.log(10_000.0),
            half,
            device=time.device,
            dtype=time.dtype,
        )
    )
    angles = time[:, None] * frequencies[None, :] * (2.0 * math.pi)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[1]))
    return embedding


class RoutedLoRALinear(nn.Module):
    """A shared linear map plus an independently stored LoRA for each route."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        num_devices: int,
        rank: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dimension, output_dimension, num_devices, rank) <= 0:
            raise ValueError("LoRA dimensions must be positive")
        self.shared = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.rank = rank
        self.num_devices = num_devices
        self.lora_down = nn.Parameter(
            torch.empty(num_devices, rank, input_dimension)
        )
        self.lora_up = nn.Parameter(
            torch.zeros(num_devices, output_dimension, rank)
        )
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5.0))

    def forward(self, signal: torch.Tensor, device_id: torch.Tensor) -> torch.Tensor:
        if signal.ndim < 2:
            raise ValueError("routed linear input needs batch and feature dimensions")
        if device_id.shape != (len(signal),) or device_id.dtype != torch.long:
            raise ValueError("device_id must be a long tensor with shape [batch]")
        if torch.any(device_id < 0) or torch.any(device_id >= self.num_devices):
            raise ValueError("device route is outside the configured range")
        shared = self.shared(signal)
        down = self.lora_down.index_select(0, device_id)
        up = self.lora_up.index_select(0, device_id)
        latent = torch.einsum("b...i,bri->b...r", signal, down)
        delta = torch.einsum("b...r,bor->b...o", latent, up)
        return shared + delta / float(self.rank)

    def adapter_parameters(self, device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= device_index < self.num_devices:
            raise ValueError("device route is outside the configured range")
        return self.lora_down[device_index], self.lora_up[device_index]


class AffineStyleAttention(nn.Module):
    """Attention-form affine empirical field without softmax saturation."""

    def __init__(
        self,
        condition_dimension: int,
        style_dimension: int,
        attention_rank: int,
        token_count: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__()
        if condition_dimension <= 0 or token_count <= 0:
            raise ValueError("style field dimensions must be positive")
        self.query = RoutedLoRALinear(
            condition_dimension + 1,
            attention_rank,
            num_devices,
            lora_rank,
        )
        self.keys = nn.Parameter(torch.empty(token_count, attention_rank))
        self.values = nn.Parameter(torch.empty(token_count, style_dimension))
        self.bias = nn.Parameter(torch.zeros(style_dimension))
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def forward(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if condition_parameters.ndim != 2:
            raise ValueError("condition parameters must have shape [batch,parameter]")
        affine = torch.cat(
            (
                torch.ones(
                    len(condition_parameters),
                    1,
                    device=condition_parameters.device,
                    dtype=condition_parameters.dtype,
                ),
                condition_parameters,
            ),
            dim=1,
        )
        query = self.query(affine, device_id)
        weights = query @ self.keys.t() / math.sqrt(float(self.keys.shape[1]))
        style = self.bias + weights @ self.values
        if return_weights:
            return style, weights
        return style


class SourceRankConditionCoordinates(nn.Module):
    """Project only explicitly rank-deficient device conditions to source axes."""

    def __init__(self, condition_dimension: int, num_devices: int) -> None:
        super().__init__()
        if condition_dimension <= 0 or num_devices <= 0:
            raise ValueError("condition geometry dimensions must be positive")
        self.condition_dimension = condition_dimension
        self.num_devices = num_devices
        self.register_buffer(
            "origin",
            torch.zeros(num_devices, condition_dimension),
        )
        self.register_buffer(
            "basis",
            torch.eye(condition_dimension).expand(num_devices, -1, -1).clone(),
        )
        self.register_buffer(
            "rank",
            torch.full((num_devices,), condition_dimension, dtype=torch.long),
        )
        self.register_buffer(
            "enabled",
            torch.zeros(num_devices, dtype=torch.bool),
        )

    @torch.no_grad()
    def set_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
        enabled: torch.Tensor,
    ) -> None:
        if origin.shape != self.origin.shape or basis.shape != self.basis.shape:
            raise ValueError("source-rank condition geometry shape differs")
        if rank.shape != self.rank.shape or rank.dtype != torch.long:
            raise ValueError("source-rank condition rank shape or dtype differs")
        if enabled.shape != self.enabled.shape or enabled.dtype != torch.bool:
            raise ValueError("source-rank condition enable mask shape or dtype differs")
        if torch.any(rank < 0) or torch.any(rank > self.condition_dimension):
            raise ValueError("source-rank condition rank is outside the supported range")
        self.origin.copy_(origin.to(self.origin))
        self.basis.copy_(basis.to(self.basis))
        self.rank.copy_(rank.to(self.rank))
        self.enabled.copy_(enabled.to(self.enabled))

    def principal_coordinates(
        self,
        parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if parameters.shape != (len(device_id), self.condition_dimension):
            raise ValueError("source-rank condition parameter shape differs")
        origin = self.origin.index_select(0, device_id)
        basis = self.basis.index_select(0, device_id)
        coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            parameters - origin,
        )
        active = torch.arange(
            self.condition_dimension,
            device=parameters.device,
        )[None, :] < self.rank.index_select(0, device_id)[:, None]
        return coordinates * active.to(coordinates.dtype)

    def forward(
        self,
        parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = self.principal_coordinates(parameters, device_id)
        enabled = self.enabled.index_select(0, device_id)[:, None]
        return torch.where(enabled, coordinates, parameters)


class SourceSupportConditionMap(SourceRankConditionCoordinates):
    """Bound unsupported directions and scale active-axis excess distance."""

    def __init__(self, condition_dimension: int, num_devices: int) -> None:
        super().__init__(condition_dimension, num_devices)
        self.register_buffer(
            "support_minimum",
            torch.zeros(num_devices, condition_dimension),
        )
        self.register_buffer(
            "support_maximum",
            torch.zeros(num_devices, condition_dimension),
        )
        self.register_buffer(
            "lower_extrapolation_gain",
            torch.ones(num_devices, condition_dimension),
        )
        self.register_buffer(
            "upper_extrapolation_gain",
            torch.ones(num_devices, condition_dimension),
        )

    @torch.no_grad()
    def set_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
        enabled: torch.Tensor,
        support_minimum: torch.Tensor,
        support_maximum: torch.Tensor,
        lower_extrapolation_gain: torch.Tensor | None = None,
        upper_extrapolation_gain: torch.Tensor | None = None,
    ) -> None:
        super().set_geometry(origin, basis, rank, enabled)
        if support_minimum.shape != self.support_minimum.shape:
            raise ValueError("source support minimum shape differs")
        if support_maximum.shape != self.support_maximum.shape:
            raise ValueError("source support maximum shape differs")
        if torch.any(support_minimum > support_maximum):
            raise ValueError("source support bounds are reversed")
        lower_gain = (
            torch.ones_like(self.lower_extrapolation_gain)
            if lower_extrapolation_gain is None
            else lower_extrapolation_gain
        )
        upper_gain = (
            torch.ones_like(self.upper_extrapolation_gain)
            if upper_extrapolation_gain is None
            else upper_extrapolation_gain
        )
        for name, value in (
            ("lower", lower_gain),
            ("upper", upper_gain),
        ):
            if value.shape != self.lower_extrapolation_gain.shape:
                raise ValueError(f"source support {name} gain shape differs")
            if not torch.isfinite(value).all() or torch.any(value < 0.0):
                raise ValueError(
                    f"source support {name} gains must be finite and nonnegative"
                )
        self.support_minimum.copy_(support_minimum.to(self.support_minimum))
        self.support_maximum.copy_(support_maximum.to(self.support_maximum))
        self.lower_extrapolation_gain.copy_(
            lower_gain.to(self.lower_extrapolation_gain)
        )
        self.upper_extrapolation_gain.copy_(
            upper_gain.to(self.upper_extrapolation_gain)
        )

    def forward(
        self,
        parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if parameters.shape != (len(device_id), self.condition_dimension):
            raise ValueError("source-support condition parameter shape differs")
        origin = self.origin.index_select(0, device_id)
        basis = self.basis.index_select(0, device_id)
        coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            parameters - origin,
        )
        rank = self.rank.index_select(0, device_id)
        active = torch.arange(
            self.condition_dimension,
            device=parameters.device,
        )[None, :] < rank[:, None]
        lower = self.support_minimum.index_select(0, device_id)
        upper = self.support_maximum.index_select(0, device_id)
        lower_gain = self.lower_extrapolation_gain.index_select(0, device_id)
        upper_gain = self.upper_extrapolation_gain.index_select(0, device_id)
        lower_changed = active & (coordinates < lower) & (lower_gain != 1.0)
        upper_changed = active & (coordinates > upper) & (upper_gain != 1.0)
        active_mapped = torch.where(
            lower_changed,
            lower + lower_gain * (coordinates - lower),
            coordinates,
        )
        active_mapped = torch.where(
            upper_changed,
            upper + upper_gain * (coordinates - upper),
            active_mapped,
        )
        bounded = torch.where(
            active,
            active_mapped,
            torch.minimum(torch.maximum(coordinates, lower), upper),
        )
        enabled = self.enabled.index_select(0, device_id)
        changed = enabled & torch.any(bounded != coordinates, dim=1)
        reconstructed = origin + torch.einsum("bri,br->bi", basis, bounded)
        return torch.where(changed[:, None], reconstructed, parameters)


class DeviceRhoSourceSupportConditionMap(SourceSupportConditionMap):
    """Blend raw and support-mapped condition styles with one rho per device."""

    def __init__(
        self,
        condition_dimension: int,
        num_devices: int,
        *,
        initial_rho: float,
    ) -> None:
        super().__init__(condition_dimension, num_devices)
        if not 0.0 < initial_rho < 1.0:
            raise ValueError("initial device rho must lie strictly inside (0,1)")
        initial_logit = math.log(initial_rho / (1.0 - initial_rho))
        self.rho_logits = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
                for _ in range(num_devices)
            ]
        )

    def rho_values(self) -> torch.Tensor:
        logits = torch.stack(tuple(self.rho_logits))
        learned = torch.sigmoid(logits)
        return torch.where(self.enabled, learned, torch.ones_like(learned))

    def rho_for(self, device_id: torch.Tensor) -> torch.Tensor:
        if device_id.ndim != 1 or device_id.dtype != torch.long:
            raise ValueError("device rho route must be a one-dimensional long tensor")
        return self.rho_values().index_select(0, device_id)

    def blended_parameters(
        self,
        parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        mapped = super().forward(parameters, device_id)
        rho = self.rho_for(device_id)[:, None]
        return parameters + (1.0 - rho) * (mapped - parameters)


class RankAwareStyleResidual(nn.Module):
    """Signed affine condition lookup with independently routed style atoms."""

    def __init__(
        self,
        condition_dimension: int,
        style_dimension: int,
        attention_rank: int,
        token_count: int,
        num_devices: int,
    ) -> None:
        super().__init__()
        if min(
            condition_dimension,
            style_dimension,
            attention_rank,
            token_count,
            num_devices,
        ) <= 0:
            raise ValueError("rank-aware style residual dimensions must be positive")
        self.condition_dimension = condition_dimension
        self.style_dimension = style_dimension
        self.attention_rank = attention_rank
        self.token_count = token_count
        self.num_devices = num_devices
        self.query = nn.Parameter(
            torch.empty(num_devices, condition_dimension, attention_rank)
        )
        self.keys = nn.Parameter(
            torch.empty(num_devices, token_count, attention_rank)
        )
        self.values = nn.Parameter(
            torch.zeros(num_devices, token_count, style_dimension)
        )
        nn.init.xavier_uniform_(
            self.query.reshape(num_devices * condition_dimension, attention_rank)
        )
        nn.init.xavier_uniform_(
            self.keys.reshape(num_devices * token_count, attention_rank)
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        device_id: torch.Tensor,
        enabled: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.shape != (len(device_id), self.condition_dimension):
            raise ValueError("rank-aware residual condition shape differs")
        if enabled.shape != (len(device_id),) or enabled.dtype != torch.bool:
            raise ValueError("rank-aware residual enable mask differs")
        query_projection = self.query.index_select(0, device_id)
        keys = self.keys.index_select(0, device_id)
        values = self.values.index_select(0, device_id)
        query = torch.einsum("bi,bir->br", coordinates, query_projection)
        weights = torch.einsum("br,btr->bt", query, keys) / math.sqrt(
            float(self.attention_rank)
        )
        residual = torch.einsum("bt,btd->bd", weights, values)
        return residual * enabled[:, None].to(residual.dtype)


class AdaptiveResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        context_dimension: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.input_convolution = nn.Conv1d(
            input_channels, output_channels, kernel_size=5, padding=2
        )
        self.output_norm = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.context = RoutedLoRALinear(
            context_dimension,
            2 * output_channels,
            num_devices,
            lora_rank,
        )
        self.output_convolution = nn.Conv1d(
            output_channels, output_channels, kernel_size=5, padding=2
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        nn.init.zeros_(self.output_convolution.weight)
        nn.init.zeros_(self.output_convolution.bias)

    def forward(
        self,
        signal: torch.Tensor,
        context: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_convolution(F.silu(self.input_norm(signal)))
        shift, scale = self.context(context, device_id).chunk(2, dim=1)
        hidden = self.output_norm(hidden)
        hidden = hidden * (1.0 + scale[:, :, None]) + shift[:, :, None]
        hidden = self.output_convolution(F.silu(hidden))
        return self.skip(signal) + hidden


class PhysicalStyleAttention1D(nn.Module):
    """Local attention jointly controlled by style and relative shaft angle."""

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("attention channels must be divisible by heads")
        if radius <= 0 or chunk_size <= 0 or downsample_factor <= 0:
            raise ValueError("attention geometry must be positive")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dimension = channels // num_heads
        self.radius = radius
        self.chunk_size = chunk_size
        self.downsample_factor = downsample_factor
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.style_statistics = RoutedLoRALinear(
            style_dimension, 2 * channels, num_devices, lora_rank
        )
        self.qkv_style = RoutedLoRALinear(
            style_dimension, 6 * channels, num_devices, lora_rank
        )
        self.query = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.key = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.value = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.output = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank
        )
        self.log_order = nn.Parameter(torch.zeros(num_heads))
        self.angle_bias_strength = nn.Parameter(torch.zeros(num_heads))
        self.log_time_scale = nn.Parameter(torch.zeros(num_heads))

    def _heads(self, signal: torch.Tensor) -> torch.Tensor:
        batch, length, channels = signal.shape
        if channels != self.channels:
            raise ValueError("attention input channel count differs")
        return signal.reshape(
            batch, length, self.num_heads, self.head_dimension
        ).permute(0, 2, 1, 3)

    def _additional_physical_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(query)

    def _physical_phase_rho(
        self,
        device_id: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return an optional device-routed scale for the shaft-angle bias."""
        return None

    def _additional_attention_logits(
        self,
        query: torch.Tensor,
        key_windows: torch.Tensor,
        style: torch.Tensor | None,
        device_id: torch.Tensor,
        delta_time: torch.Tensor,
        delta_angle: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return an optional local logit correction for derived mechanisms."""
        return None

    def _attention_features(
        self,
        signal: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, length = signal.shape
        query = self._heads(query)
        key = self._heads(key)
        value = self._heads(value)
        radius = self.radius
        window = 2 * radius + 1
        key_padded = F.pad(key, (0, 0, radius, radius))
        value_padded = F.pad(value, (0, 0, radius, radius))
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=signal.device,
            dtype=signal.dtype,
        )
        delta_time = (
            offsets[None, :]
            * float(self.downsample_factor)
            / sample_rate_hz[:, None]
        )
        delta_angle = 2.0 * math.pi * shaft_speed_hz[:, None] * delta_time
        order = F.softplus(self.log_order) + 0.1
        angle_bias = torch.cos(
            delta_angle[:, None, :] * order[None, :, None]
        ) * self.angle_bias_strength[None, :, None]
        physical_phase_rho = self._physical_phase_rho(device_id)
        if physical_phase_rho is not None:
            angle_bias = angle_bias * physical_phase_rho[:, None, None]
        time_scale = F.softplus(self.log_time_scale) + 1e-4
        time_bias = -torch.abs(delta_time[:, None, :]) / time_scale[None, :, None]
        relative_bias = angle_bias + time_bias

        positions = torch.arange(length, device=signal.device)
        valid = (
            positions[:, None]
            + torch.arange(-radius, radius + 1, device=signal.device)[None, :]
        )
        valid = (valid >= 0) & (valid < length)
        chunks: list[torch.Tensor] = []
        for begin in range(0, length, self.chunk_size):
            end = min(begin + self.chunk_size, length)
            chunk_length = end - begin
            key_windows = key_padded[:, :, begin : end + 2 * radius].unfold(
                2, window, 1
            ).permute(0, 1, 2, 4, 3)
            value_windows = value_padded[:, :, begin : end + 2 * radius].unfold(
                2, window, 1
            ).permute(0, 1, 2, 4, 3)
            logits = (
                query[:, :, begin:end, None, :] * key_windows
            ).sum(dim=-1) / math.sqrt(float(self.head_dimension))
            logits = logits + relative_bias[:, :, None, :]
            additional_logits = self._additional_attention_logits(
                query[:, :, begin:end],
                key_windows,
                style,
                device_id,
                delta_time,
                delta_angle,
            )
            if additional_logits is not None:
                if additional_logits.shape != logits.shape:
                    raise RuntimeError("additional attention logit shape differs")
                logits = logits + additional_logits
            mask = valid[begin:end][None, None, :, :]
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1)
            attended = (weights[..., None] * value_windows).sum(dim=-2)
            if attended.shape[2] != chunk_length:
                raise RuntimeError("local attention chunk length differs")
            chunks.append(attended)
        attended = torch.cat(chunks, dim=2)
        attended = attended + self._additional_physical_attention(
            query,
            key,
            value,
            shaft_speed_hz,
            sample_rate_hz,
        )
        return attended.permute(0, 2, 1, 3).reshape(batch, length, channels)

    def _physical_attention(
        self,
        signal: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self._attention_features(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
            style,
        )
        projected = self.output(attended, device_id).transpose(1, 2)
        return signal + projected

    def forward(
        self,
        signal: torch.Tensor,
        style: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        *,
        allow_nonpositive_shaft_reference: bool = False,
    ) -> torch.Tensor:
        batch, channels, length = signal.shape
        if style.shape[0] != batch:
            raise ValueError("style batch size differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if name == "shaft speed" and allow_nonpositive_shaft_reference:
                if value.shape != (batch,) or not torch.isfinite(value).all():
                    raise ValueError("shaft reference must be finite with shape [batch]")
                continue
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        normalized = self.norm(signal)
        mean = normalized.mean(dim=-1, keepdim=True)
        deviation = normalized.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        shift, scale = self.style_statistics(style, device_id).chunk(2, dim=1)
        styled = (normalized - mean) / deviation
        styled = styled * (1.0 + scale[:, :, None]) + shift[:, :, None]
        tokens = styled.transpose(1, 2)

        qkv_shift, qkv_scale = self.qkv_style(style, device_id).chunk(2, dim=1)
        q_shift, k_shift, v_shift = qkv_shift.chunk(3, dim=1)
        q_scale, k_scale, v_scale = qkv_scale.chunk(3, dim=1)
        query = self.query(tokens, device_id)
        key = self.key(tokens, device_id)
        value = self.value(tokens, device_id)
        query = query * (1.0 + q_scale[:, None]) + q_shift[:, None]
        key = key * (1.0 + k_scale[:, None]) + k_shift[:, None]
        value = value * (1.0 + v_scale[:, None]) + v_shift[:, None]
        return self._physical_attention(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
            style,
        )


class LearnedDeviceConstraintFieldAttention1D(PhysicalStyleAttention1D):
    """Learn a continuous device-routed compatibility correction in Q/K space."""

    CONSTRAINT_FEATURE_NAMES = (
        "semantic_mismatch",
        "shaft_angle_mismatch",
        "normalized_time_distance",
        "semantic_x_shaft_angle",
        "semantic_x_time",
        "shaft_angle_x_time",
    )

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__(
            channels,
            style_dimension,
            num_heads,
            radius,
            chunk_size,
            downsample_factor,
            num_devices,
            lora_rank,
        )
        feature_count = len(self.CONSTRAINT_FEATURE_NAMES)
        with torch.random.fork_rng(devices=[]):
            self.constraint_coefficient_field = RoutedLoRALinear(
                style_dimension,
                num_heads * feature_count,
                num_devices,
                lora_rank,
            )
        nn.init.zeros_(self.constraint_coefficient_field.shared.weight)
        if self.constraint_coefficient_field.shared.bias is not None:
            nn.init.zeros_(self.constraint_coefficient_field.shared.bias)

    def constraint_coefficients(
        self,
        style: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.constraint_coefficient_field(style, device_id)
        return coefficients.reshape(
            len(style),
            self.num_heads,
            len(self.CONSTRAINT_FEATURE_NAMES),
        )

    def _additional_attention_logits(
        self,
        query: torch.Tensor,
        key_windows: torch.Tensor,
        style: torch.Tensor | None,
        device_id: torch.Tensor,
        delta_time: torch.Tensor,
        delta_angle: torch.Tensor,
    ) -> torch.Tensor:
        if style is None:
            raise RuntimeError("learned constraint field requires continuous style")
        normalized_query = F.normalize(query, dim=-1)
        normalized_key = F.normalize(key_windows, dim=-1)
        semantic_mismatch = 1.0 - (
            normalized_query[..., None, :] * normalized_key
        ).sum(dim=-1)
        semantic_mismatch = semantic_mismatch.clamp(0.0, 2.0)

        order = F.softplus(self.log_order) + 0.1
        shaft_angle_mismatch = 1.0 - torch.cos(
            delta_angle[:, None, None, :] * order[None, :, None, None]
        )
        shaft_angle_mismatch = shaft_angle_mismatch.expand_as(semantic_mismatch)
        maximum_time = delta_time.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        normalized_time = (delta_time.abs() / maximum_time)[:, None, None, :]
        normalized_time = normalized_time.expand_as(semantic_mismatch)

        features = torch.stack(
            (
                semantic_mismatch,
                shaft_angle_mismatch,
                normalized_time,
                semantic_mismatch * shaft_angle_mismatch,
                semantic_mismatch * normalized_time,
                shaft_angle_mismatch * normalized_time,
            ),
            dim=-1,
        )
        coefficients = self.constraint_coefficients(style, device_id)
        pre_energy = (
            features * coefficients[:, :, None, None, :]
        ).sum(dim=-1) / math.sqrt(float(len(self.CONSTRAINT_FEATURE_NAMES)))
        baseline = F.softplus(torch.zeros_like(pre_energy))
        energy_increment = F.softplus(pre_energy) - baseline
        return -energy_increment


class DeviceConstraintFusionAttention1D(PhysicalStyleAttention1D):
    """Fuse content, continuous style, and shaft phase with device-learned gates."""

    CONSTRAINT_RHO_NAMES = (
        "content_style_rho",
        "qkv_style_rho",
        "physical_phase_rho",
    )

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
        *,
        initial_rho: tuple[float, float, float],
    ) -> None:
        super().__init__(
            channels,
            style_dimension,
            num_heads,
            radius,
            chunk_size,
            downsample_factor,
            num_devices,
            lora_rank,
        )
        if len(initial_rho) != len(self.CONSTRAINT_RHO_NAMES) or any(
            not 0.0 < value < 1.0 for value in initial_rho
        ):
            raise ValueError(
                "device constraint rho values must contain three values in (0,1)"
            )
        initial = torch.tensor(initial_rho, dtype=torch.float32)
        logits = torch.log(initial / (1.0 - initial))
        self.constraint_rho_logits = nn.Parameter(
            logits[None, :].expand(num_devices, -1).clone()
        )

    def constraint_rho_values(self) -> torch.Tensor:
        return torch.sigmoid(self.constraint_rho_logits)

    def constraint_rho_for(self, device_id: torch.Tensor) -> torch.Tensor:
        if device_id.ndim != 1 or device_id.dtype != torch.long:
            raise ValueError(
                "device constraint route must be a one-dimensional long tensor"
            )
        values = self.constraint_rho_values()
        if torch.any(device_id < 0) or torch.any(device_id >= len(values)):
            raise ValueError("device constraint route is outside the configured range")
        return values.index_select(0, device_id)

    def _physical_phase_rho(self, device_id: torch.Tensor) -> torch.Tensor:
        return self.constraint_rho_for(device_id)[:, 2]

    def forward(
        self,
        signal: torch.Tensor,
        style: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _ = signal.shape
        if style.shape[0] != batch:
            raise ValueError("style batch size differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        rho = self.constraint_rho_for(device_id).to(signal.dtype)
        normalized = self.norm(signal)
        mean = normalized.mean(dim=-1, keepdim=True)
        deviation = normalized.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        content_tokens = ((normalized - mean) / deviation).transpose(1, 2)

        shift, scale = self.style_statistics(style, device_id).chunk(2, dim=1)
        fully_styled = content_tokens * (1.0 + scale[:, None]) + shift[:, None]
        content_style_rho = rho[:, 0, None, None]
        tokens = content_tokens + content_style_rho * (
            fully_styled - content_tokens
        )

        base_query = self.query(tokens, device_id)
        base_key = self.key(tokens, device_id)
        base_value = self.value(tokens, device_id)
        qkv_shift, qkv_scale = self.qkv_style(style, device_id).chunk(2, dim=1)
        q_shift, k_shift, v_shift = qkv_shift.chunk(3, dim=1)
        q_scale, k_scale, v_scale = qkv_scale.chunk(3, dim=1)
        conditioned_query = base_query * (1.0 + q_scale[:, None]) + q_shift[:, None]
        conditioned_key = base_key * (1.0 + k_scale[:, None]) + k_shift[:, None]
        conditioned_value = base_value * (1.0 + v_scale[:, None]) + v_shift[:, None]
        qkv_style_rho = rho[:, 1, None, None]
        query = base_query + qkv_style_rho * (conditioned_query - base_query)
        key = base_key + qkv_style_rho * (conditioned_key - base_key)
        value = base_value + qkv_style_rho * (conditioned_value - base_value)
        return self._physical_attention(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
        )


class ContentQueryAffineStyleRepairAttention1D(PhysicalStyleAttention1D):
    """Retrieve content-local style whose values are affine in source coordinates."""

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        condition_dimension: int,
        token_count: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__(
            channels,
            style_dimension,
            num_heads,
            radius,
            chunk_size,
            downsample_factor,
            num_devices,
            lora_rank,
        )
        if condition_dimension <= 0 or token_count <= 0:
            raise ValueError("content-query style repair dimensions must be positive")
        self.condition_dimension = condition_dimension
        self.token_count = token_count
        self.num_devices = num_devices
        with torch.random.fork_rng(devices=[]):
            self.condition_style_keys = nn.ParameterList(
                [nn.Parameter(torch.empty(token_count, channels)) for _ in range(num_devices)]
            )
            self.condition_style_values = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            condition_dimension,
                            token_count,
                            8,
                            channels,
                        )
                    )
                    for _ in range(num_devices)
                ]
            )
            for keys in self.condition_style_keys:
                nn.init.xavier_uniform_(keys)
        self.register_buffer(
            "condition_origin",
            torch.zeros(num_devices, condition_dimension),
        )
        self.register_buffer(
            "condition_basis",
            torch.eye(condition_dimension).expand(num_devices, -1, -1).clone(),
        )
        self.register_buffer(
            "condition_rank",
            torch.full((num_devices,), condition_dimension, dtype=torch.long),
        )
        self.register_buffer(
            "condition_enabled",
            torch.zeros(num_devices, dtype=torch.bool),
        )

    @torch.no_grad()
    def set_condition_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
        enabled: torch.Tensor,
    ) -> None:
        expected_origin = (self.num_devices, self.condition_dimension)
        expected_basis = (
            self.num_devices,
            self.condition_dimension,
            self.condition_dimension,
        )
        if origin.shape != expected_origin or basis.shape != expected_basis:
            raise ValueError("content-query condition geometry shape differs")
        if rank.shape != (self.num_devices,) or rank.dtype != torch.long:
            raise ValueError("content-query condition rank shape or dtype differs")
        if enabled.shape != (self.num_devices,) or enabled.dtype != torch.bool:
            raise ValueError("content-query condition enable mask differs")
        if torch.any(rank < 0) or torch.any(rank > self.condition_dimension):
            raise ValueError("content-query condition rank is outside the supported range")
        self.condition_origin.copy_(origin.to(self.condition_origin))
        self.condition_basis.copy_(basis.to(self.condition_basis))
        self.condition_rank.copy_(rank.to(self.condition_rank))
        self.condition_enabled.copy_(enabled.to(self.condition_enabled))

    def relative_coordinates(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if condition_parameters.shape != (
            len(device_id),
            self.condition_dimension,
        ):
            raise ValueError("content-query condition parameter shape differs")
        origin = self.condition_origin.index_select(0, device_id)
        basis = self.condition_basis.index_select(0, device_id)
        coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            condition_parameters - origin,
        )
        active = torch.arange(
            self.condition_dimension,
            device=condition_parameters.device,
        )[None, :] < self.condition_rank.index_select(0, device_id)[:, None]
        return coordinates * active.to(coordinates.dtype)

    def condition_modulation(
        self,
        content_tokens: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, channels = content_tokens.shape
        if channels != self.channels:
            raise ValueError("content-query token channel count differs")
        coordinates = self.relative_coordinates(condition_parameters, device_id)
        query = self._heads(self.query(content_tokens, device_id))
        keys = torch.stack(tuple(self.condition_style_keys), dim=0).index_select(
            0, device_id
        )
        keys = keys.reshape(
            batch,
            self.token_count,
            self.num_heads,
            self.head_dimension,
        ).permute(0, 2, 1, 3)
        weights = torch.softmax(
            torch.einsum("bhld,bhsd->bhls", query, keys)
            / math.sqrt(float(self.head_dimension)),
            dim=-1,
        )
        values = torch.stack(tuple(self.condition_style_values), dim=0).index_select(
            0, device_id
        )
        conditional_values = torch.einsum(
            "br,brsvc->bsvc",
            coordinates,
            values,
        )
        conditional_values = conditional_values.reshape(
            batch,
            self.token_count,
            8,
            self.num_heads,
            self.head_dimension,
        ).permute(0, 3, 1, 2, 4)
        retrieved = torch.einsum(
            "bhls,bhsvd->bhlvd",
            weights,
            conditional_values,
        ).permute(0, 2, 3, 1, 4).reshape(batch, length, 8, channels)
        enabled = self.condition_enabled.index_select(0, device_id)
        return retrieved * enabled[:, None, None, None].to(retrieved.dtype)

    def forward(
        self,
        signal: torch.Tensor,
        style: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _ = signal.shape
        if style.shape[0] != batch:
            raise ValueError("style batch size differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        normalized = self.norm(signal)
        mean = normalized.mean(dim=-1, keepdim=True)
        deviation = normalized.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        content_tokens = ((normalized - mean) / deviation).transpose(1, 2)
        modulation = self.condition_modulation(
            content_tokens,
            condition_parameters,
            device_id,
        )
        (
            local_shift,
            local_scale,
            local_q_shift,
            local_k_shift,
            local_v_shift,
            local_q_scale,
            local_k_scale,
            local_v_scale,
        ) = modulation.unbind(dim=2)

        global_shift, global_scale = self.style_statistics(style, device_id).chunk(
            2, dim=1
        )
        styled = content_tokens * (1.0 + global_scale[:, None] + local_scale)
        styled = styled + global_shift[:, None] + local_shift
        qkv_shift, qkv_scale = self.qkv_style(style, device_id).chunk(2, dim=1)
        q_shift, k_shift, v_shift = qkv_shift.chunk(3, dim=1)
        q_scale, k_scale, v_scale = qkv_scale.chunk(3, dim=1)
        query = self.query(styled, device_id)
        key = self.key(styled, device_id)
        value = self.value(styled, device_id)
        query = query * (1.0 + q_scale[:, None] + local_q_scale)
        query = query + q_shift[:, None] + local_q_shift
        key = key * (1.0 + k_scale[:, None] + local_k_scale)
        key = key + k_shift[:, None] + local_k_shift
        value = value * (1.0 + v_scale[:, None] + local_v_scale)
        value = value + v_shift[:, None] + local_v_shift
        return self._physical_attention(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
        )


class RelativeStyleTransportAttention1D(PhysicalStyleAttention1D):
    """Base-preserving content lookup of signed condition-style directions."""

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        condition_dimension: int,
        token_count: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
    ) -> None:
        super().__init__(
            channels,
            style_dimension,
            num_heads,
            radius,
            chunk_size,
            downsample_factor,
            num_devices,
            lora_rank,
        )
        if condition_dimension <= 0 or token_count <= 0:
            raise ValueError("relative style dimensions must be positive")
        self.condition_dimension = condition_dimension
        self.token_count = token_count
        self.num_devices = num_devices

        # Keep construction of the inherited base independent of this branch.
        with torch.random.fork_rng(devices=[]):
            self.relative_style_keys = nn.Parameter(
                torch.empty(condition_dimension, token_count, channels)
            )
            self.relative_style_values = nn.Parameter(
                torch.empty(condition_dimension, token_count, 8, channels)
            )
            self.relative_output = RoutedLoRALinear(
                channels,
                channels,
                num_devices,
                lora_rank,
                bias=False,
            )
            nn.init.xavier_uniform_(
                self.relative_style_keys.reshape(condition_dimension * token_count, channels)
            )
            nn.init.xavier_uniform_(
                self.relative_style_values.reshape(
                    condition_dimension * token_count * 8,
                    channels,
                )
            )
            nn.init.zeros_(self.relative_output.shared.weight)

        self.register_buffer(
            "condition_origin",
            torch.zeros(num_devices, condition_dimension),
        )
        self.register_buffer(
            "condition_basis",
            torch.eye(condition_dimension).expand(num_devices, -1, -1).clone(),
        )
        self.register_buffer(
            "condition_rank",
            torch.full((num_devices,), condition_dimension, dtype=torch.long),
        )

    @torch.no_grad()
    def set_condition_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
    ) -> None:
        expected_origin = (self.num_devices, self.condition_dimension)
        expected_basis = (
            self.num_devices,
            self.condition_dimension,
            self.condition_dimension,
        )
        if origin.shape != expected_origin or basis.shape != expected_basis:
            raise ValueError("relative condition geometry shape differs")
        if rank.shape != (self.num_devices,) or rank.dtype != torch.long:
            raise ValueError("relative condition rank must be long with shape [device]")
        if torch.any(rank < 0) or torch.any(rank > self.condition_dimension):
            raise ValueError("relative condition rank is outside the supported range")
        self.condition_origin.copy_(origin.to(self.condition_origin))
        self.condition_basis.copy_(basis.to(self.condition_basis))
        self.condition_rank.copy_(rank.to(self.condition_rank))

    def relative_coordinates(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if condition_parameters.shape != (
            len(device_id),
            self.condition_dimension,
        ):
            raise ValueError("condition parameter shape differs in relative transport")
        origin = self.condition_origin.index_select(0, device_id)
        basis = self.condition_basis.index_select(0, device_id)
        coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            condition_parameters - origin,
        )
        active = torch.arange(
            self.condition_dimension,
            device=condition_parameters.device,
        )[None, :] < self.condition_rank.index_select(0, device_id)[:, None]
        return coordinates * active.to(coordinates.dtype)

    def relative_modulation(
        self,
        content_tokens: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = self.relative_coordinates(condition_parameters, device_id)
        content_query = self.query(content_tokens, device_id)
        logits = torch.einsum(
            "blc,rsc->brls",
            content_query,
            self.relative_style_keys,
        ) / math.sqrt(float(self.channels))
        weights = torch.softmax(logits, dim=-1)
        directions = torch.einsum(
            "brls,rsvd->brlvd",
            weights,
            self.relative_style_values,
        )
        return (directions * coordinates[:, :, None, None, None]).sum(dim=1)

    def forward(
        self,
        signal: torch.Tensor,
        style: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, _ = signal.shape
        if style.shape[0] != batch:
            raise ValueError("style batch size differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        normalized = self.norm(signal)
        mean = normalized.mean(dim=-1, keepdim=True)
        deviation = normalized.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        content_tokens = ((normalized - mean) / deviation).transpose(1, 2)
        shift, scale = self.style_statistics(style, device_id).chunk(2, dim=1)
        styled = content_tokens * (1.0 + scale[:, None]) + shift[:, None]

        qkv_shift, qkv_scale = self.qkv_style(style, device_id).chunk(2, dim=1)
        q_shift, k_shift, v_shift = qkv_shift.chunk(3, dim=1)
        q_scale, k_scale, v_scale = qkv_scale.chunk(3, dim=1)
        query = self.query(styled, device_id)
        key = self.key(styled, device_id)
        value = self.value(styled, device_id)
        query = query * (1.0 + q_scale[:, None]) + q_shift[:, None]
        key = key * (1.0 + k_scale[:, None]) + k_shift[:, None]
        value = value * (1.0 + v_scale[:, None]) + v_shift[:, None]
        base_features = self._attention_features(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
        )
        baseline = signal + self.output(base_features, device_id).transpose(1, 2)

        modulation = self.relative_modulation(
            content_tokens,
            condition_parameters,
            device_id,
        )
        (
            local_shift,
            local_scale,
            local_q_shift,
            local_k_shift,
            local_v_shift,
            local_q_scale,
            local_k_scale,
            local_v_scale,
        ) = modulation.unbind(dim=2)
        transported = styled * (1.0 + local_scale) + local_shift
        transported_query = self.query(transported, device_id)
        transported_key = self.key(transported, device_id)
        transported_value = self.value(transported, device_id)
        transported_query = transported_query * (
            1.0 + q_scale[:, None] + local_q_scale
        ) + q_shift[:, None] + local_q_shift
        transported_key = transported_key * (
            1.0 + k_scale[:, None] + local_k_scale
        ) + k_shift[:, None] + local_k_shift
        transported_value = transported_value * (
            1.0 + v_scale[:, None] + local_v_scale
        ) + v_shift[:, None] + local_v_shift
        transported_features = self._attention_features(
            signal,
            transported_query,
            transported_key,
            transported_value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
        )
        correction = self.relative_output(
            transported_features - base_features,
            device_id,
        ).transpose(1, 2)
        if correction.shape != (batch, channels, signal.shape[-1]):
            raise RuntimeError("relative style correction shape differs")
        return baseline + correction


class ContinuousQKVStyleTransferAttention1D(nn.Module):
    """Content-query style transfer with an affine continuous condition field.

    Operating conditions produce style keys and values through an affine path.
    Signal content produces queries. Softmax retrieves high-dimensional style
    atoms but never bounds the operating parameters themselves. Retrieved
    values jointly define AdaAttN-like local statistics and physical Q/K/V.
    """

    def __init__(
        self,
        channels: int,
        condition_dimension: int,
        token_count: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
        query_uses_condition: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("attention channels must be divisible by heads")
        if min(condition_dimension, token_count, radius, chunk_size) <= 0:
            raise ValueError("style-transfer attention geometry must be positive")
        self.channels = channels
        self.condition_dimension = condition_dimension
        self.token_count = token_count
        self.num_heads = num_heads
        self.head_dimension = channels // num_heads
        self.radius = radius
        self.chunk_size = chunk_size
        self.downsample_factor = downsample_factor
        self.query_uses_condition = bool(query_uses_condition)

        self.content_norm = nn.GroupNorm(_group_count(channels), channels)
        self.content_query = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        condition_input = condition_dimension + 1
        if self.query_uses_condition:
            self.condition_query: RoutedLoRALinear | None = RoutedLoRALinear(
                condition_input, channels, num_devices, lora_rank, bias=False
            )
        else:
            self.condition_query = None
        self.style_key_base = nn.Parameter(torch.empty(token_count, channels))
        self.style_value_base = nn.Parameter(torch.empty(token_count, 5 * channels))
        self.style_key_field = RoutedLoRALinear(
            condition_input,
            token_count * channels,
            num_devices,
            lora_rank,
            bias=False,
        )
        self.style_value_field = RoutedLoRALinear(
            condition_input,
            token_count * 5 * channels,
            num_devices,
            lora_rank,
            bias=False,
        )
        self.temporal_query = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.temporal_key = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.temporal_value = RoutedLoRALinear(
            channels, channels, num_devices, lora_rank, bias=False
        )
        self.output = RoutedLoRALinear(channels, channels, num_devices, lora_rank)
        self.log_order = nn.Parameter(torch.zeros(num_heads))
        self.angle_bias_strength = nn.Parameter(torch.zeros(num_heads))
        self.log_time_scale = nn.Parameter(torch.zeros(num_heads))
        nn.init.xavier_uniform_(self.style_key_base)
        nn.init.xavier_uniform_(self.style_value_base)
        nn.init.zeros_(self.style_key_field.shared.weight)
        nn.init.zeros_(self.style_value_field.shared.weight)
        nn.init.zeros_(self.output.shared.weight)
        if self.output.shared.bias is not None:
            nn.init.zeros_(self.output.shared.bias)

    def _heads(self, signal: torch.Tensor) -> torch.Tensor:
        batch, length, channels = signal.shape
        if channels != self.channels:
            raise ValueError("attention input channel count differs")
        return signal.reshape(
            batch, length, self.num_heads, self.head_dimension
        ).permute(0, 2, 1, 3)

    def _condition_coordinates(self, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.ndim != 2 or parameters.shape[1] != self.condition_dimension:
            raise ValueError("condition parameter shape differs in style transfer")
        return torch.cat(
            (
                torch.ones(
                    len(parameters),
                    1,
                    device=parameters.device,
                    dtype=parameters.dtype,
                ),
                parameters,
            ),
            dim=1,
        )

    def forward(
        self,
        signal: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, length = signal.shape
        if channels != self.channels:
            raise ValueError("style-transfer input channel count differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        coordinates = self._condition_coordinates(condition_parameters)
        normalized = self.content_norm(signal).transpose(1, 2)
        content_query = self.content_query(normalized, device_id)
        if self.condition_query is not None:
            content_query = content_query + self.condition_query(
                coordinates, device_id
            )[:, None, :]
        content_query = self._heads(content_query)

        style_keys = self.style_key_base[None] + self.style_key_field(
            coordinates, device_id
        ).reshape(batch, self.token_count, channels)
        style_values = self.style_value_base[None] + self.style_value_field(
            coordinates, device_id
        ).reshape(batch, self.token_count, 5 * channels)
        style_keys = self._heads(style_keys)
        style_values = style_values.reshape(
            batch, self.token_count, 5, self.num_heads, self.head_dimension
        ).permute(0, 3, 1, 2, 4)
        style_logits = torch.einsum(
            "bhld,bhsd->bhls", content_query, style_keys
        ) / math.sqrt(float(self.head_dimension))
        style_weights = torch.softmax(style_logits, dim=-1)
        retrieved = torch.einsum(
            "bhls,bhsvd->bhlvd", style_weights, style_values
        )
        retrieved = retrieved.permute(0, 2, 3, 1, 4).reshape(
            batch, length, 5, channels
        )
        local_mean, local_log_scale, q_shift, k_shift, v_shift = retrieved.unbind(2)
        styled = normalized * local_log_scale.clamp(-2.0, 2.0).exp() + local_mean

        query = self.temporal_query(styled, device_id) + q_shift
        key = self.temporal_key(styled, device_id) + k_shift
        value = self.temporal_value(styled, device_id) + v_shift
        query = self._heads(query)
        key = self._heads(key)
        value = self._heads(value)

        radius = self.radius
        window = 2 * radius + 1
        key_padded = F.pad(key, (0, 0, radius, radius))
        value_padded = F.pad(value, (0, 0, radius, radius))
        offsets = torch.arange(
            -radius, radius + 1, device=signal.device, dtype=signal.dtype
        )
        delta_time = (
            offsets[None, :] * float(self.downsample_factor) / sample_rate_hz[:, None]
        )
        delta_angle = 2.0 * math.pi * shaft_speed_hz[:, None] * delta_time
        order = F.softplus(self.log_order) + 0.1
        angle_bias = torch.cos(
            delta_angle[:, None, :] * order[None, :, None]
        ) * self.angle_bias_strength[None, :, None]
        time_scale = F.softplus(self.log_time_scale) + 1e-4
        time_bias = -torch.abs(delta_time[:, None, :]) / time_scale[None, :, None]
        relative_bias = angle_bias + time_bias

        positions = torch.arange(length, device=signal.device)
        valid = (
            positions[:, None]
            + torch.arange(-radius, radius + 1, device=signal.device)[None, :]
        )
        valid = (valid >= 0) & (valid < length)
        chunks: list[torch.Tensor] = []
        for begin in range(0, length, self.chunk_size):
            end = min(begin + self.chunk_size, length)
            key_windows = key_padded[:, :, begin : end + 2 * radius].unfold(
                2, window, 1
            ).permute(0, 1, 2, 4, 3)
            value_windows = value_padded[:, :, begin : end + 2 * radius].unfold(
                2, window, 1
            ).permute(0, 1, 2, 4, 3)
            logits = (
                query[:, :, begin:end, None, :] * key_windows
            ).sum(dim=-1) / math.sqrt(float(self.head_dimension))
            logits = logits + relative_bias[:, :, None, :]
            logits = logits.masked_fill(
                ~valid[begin:end][None, None], torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=-1)
            chunks.append((weights[..., None] * value_windows).sum(dim=-2))
        attended = torch.cat(chunks, dim=2)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, channels)
        projected = self.output(attended, device_id).transpose(1, 2)
        return signal + projected


class FusedContinuousStyleAttention1D(PhysicalStyleAttention1D):
    """Preserve content Q/K/V while retrieving continuous condition style."""

    def __init__(
        self,
        channels: int,
        style_dimension: int,
        condition_dimension: int,
        token_count: int,
        num_heads: int,
        radius: int,
        chunk_size: int,
        downsample_factor: int,
        num_devices: int,
        lora_rank: int,
        shared_reference_transport: bool = False,
    ) -> None:
        super().__init__(
            channels,
            style_dimension,
            num_heads,
            radius,
            chunk_size,
            downsample_factor,
            num_devices,
            lora_rank,
        )
        if condition_dimension <= 0 or token_count <= 0:
            raise ValueError("fused style dimensions must be positive")
        self.condition_dimension = condition_dimension
        self.token_count = token_count
        self.shared_reference_transport = bool(shared_reference_transport)
        condition_input = (
            condition_dimension
            if self.shared_reference_transport
            else condition_dimension + 1
        )
        # Extra branch initialization must not change the legacy trunk RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.lookup_query = RoutedLoRALinear(
                channels, channels, num_devices, lora_rank, bias=False
            )
            self.style_key_base = nn.Parameter(torch.empty(token_count, channels))
            if self.shared_reference_transport:
                self.register_parameter("style_value_base", None)
                self.style_key_field = nn.Linear(
                    condition_input, token_count * channels, bias=False
                )
                self.style_value_field = nn.Linear(
                    condition_input, token_count * 8 * channels, bias=False
                )
            else:
                self.style_value_base = nn.Parameter(
                    torch.empty(token_count, 8 * channels)
                )
                self.style_key_field = RoutedLoRALinear(
                    condition_input,
                    token_count * channels,
                    num_devices,
                    lora_rank,
                    bias=False,
                )
                self.style_value_field = RoutedLoRALinear(
                    condition_input,
                    token_count * 8 * channels,
                    num_devices,
                    lora_rank,
                    bias=False,
                )
            nn.init.xavier_uniform_(self.style_key_base)
            if self.style_value_base is not None:
                nn.init.xavier_uniform_(self.style_value_base)
        self.style_residual_strength = nn.Parameter(torch.zeros(()))

    def _condition_coordinates(self, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.ndim != 2 or parameters.shape[1] != self.condition_dimension:
            raise ValueError("condition parameter shape differs in fused style transfer")
        if self.shared_reference_transport:
            return parameters
        return torch.cat(
            (
                torch.ones(
                    len(parameters),
                    1,
                    device=parameters.device,
                    dtype=parameters.dtype,
                ),
                parameters,
            ),
            dim=1,
        )

    def _condition_field(
        self,
        field: nn.Linear | RoutedLoRALinear,
        coordinates: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(field, RoutedLoRALinear):
            return field(coordinates, device_id)
        return field(coordinates)

    def forward(
        self,
        signal: torch.Tensor,
        style: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, length = signal.shape
        if style.shape[0] != batch:
            raise ValueError("style batch size differs")
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
        ):
            if value.shape != (batch,) or torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        coordinates = self._condition_coordinates(condition_parameters)
        normalized = self.norm(signal)
        mean = normalized.mean(dim=-1, keepdim=True)
        deviation = normalized.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        content = (normalized - mean) / deviation
        content_tokens = content.transpose(1, 2)
        lookup_query = self._heads(self.lookup_query(content_tokens, device_id))
        style_keys = self.style_key_base[None] + self._condition_field(
            self.style_key_field, coordinates, device_id
        ).reshape(batch, self.token_count, channels)
        style_keys = self._heads(style_keys)
        style_values = self._condition_field(
            self.style_value_field, coordinates, device_id
        ).reshape(batch, self.token_count, 8 * channels)
        if self.style_value_base is not None:
            style_values = self.style_value_base[None] + style_values
        style_values = style_values.reshape(
            batch, self.token_count, 8, self.num_heads, self.head_dimension
        ).permute(0, 3, 1, 2, 4)
        style_weights = torch.softmax(
            torch.einsum("bhld,bhsd->bhls", lookup_query, style_keys)
            / math.sqrt(float(self.head_dimension)),
            dim=-1,
        )
        retrieved = torch.einsum(
            "bhls,bhsvd->bhlvd", style_weights, style_values
        ).permute(0, 2, 3, 1, 4).reshape(batch, length, 8, channels)
        (
            local_shift,
            local_scale,
            local_q_shift,
            local_k_shift,
            local_v_shift,
            local_q_scale,
            local_k_scale,
            local_v_scale,
        ) = retrieved.unbind(2)
        residual = self.style_residual_strength

        global_shift, global_scale = self.style_statistics(style, device_id).chunk(
            2, dim=1
        )
        styled = content_tokens * (
            1.0
            + global_scale[:, None]
            + residual * local_scale.clamp(-2.0, 2.0)
        )
        styled = styled + global_shift[:, None] + residual * local_shift

        qkv_shift, qkv_scale = self.qkv_style(style, device_id).chunk(2, dim=1)
        q_shift, k_shift, v_shift = qkv_shift.chunk(3, dim=1)
        q_scale, k_scale, v_scale = qkv_scale.chunk(3, dim=1)
        query = self.query(styled, device_id)
        key = self.key(styled, device_id)
        value = self.value(styled, device_id)
        query = query * (
            1.0 + q_scale[:, None] + residual * local_q_scale.clamp(-2.0, 2.0)
        ) + q_shift[:, None] + residual * local_q_shift
        key = key * (
            1.0 + k_scale[:, None] + residual * local_k_scale.clamp(-2.0, 2.0)
        ) + k_shift[:, None] + residual * local_k_shift
        value = value * (
            1.0 + v_scale[:, None] + residual * local_v_scale.clamp(-2.0, 2.0)
        ) + v_shift[:, None] + residual * local_v_shift
        return self._physical_attention(
            signal,
            query,
            key,
            value,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
        )


class _PhaseAlignedAttentionMixin:
    phase_cycle_count: int
    phase_log_order: nn.Parameter
    phase_cycle_bias: nn.Parameter
    phase_residual_strength: nn.Parameter

    def _initialize_phase_alignment(self, cycle_count: int = 2) -> None:
        if cycle_count <= 0:
            raise ValueError("phase cycle count must be positive")
        self.phase_cycle_count = cycle_count
        initial_order = torch.linspace(2.0, 8.0, self.num_heads)
        inverse_softplus = torch.log(torch.expm1(initial_order - 0.25))
        with torch.random.fork_rng(devices=[]):
            self.phase_log_order = nn.Parameter(inverse_softplus)
            self.phase_cycle_bias = nn.Parameter(
                torch.zeros(self.num_heads, 2 * cycle_count)
            )
        self.phase_residual_strength = nn.Parameter(torch.zeros(self.num_heads))

    @staticmethod
    def _sample_tokens_at_positions(
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, length, dimension = tokens.shape
        if positions.shape[:3] != (batch, heads, length):
            raise ValueError("phase sampling positions differ from token geometry")
        candidate_count = positions.shape[3]
        valid = (positions >= 0.0) & (positions <= float(length - 1))
        lower = positions.floor()
        fraction = positions - lower
        lower_index = lower.to(torch.long).clamp(0, length - 1)
        upper_index = (lower_index + 1).clamp(0, length - 1)
        flat = tokens.reshape(batch * heads, length, dimension)

        def gather(index: torch.Tensor) -> torch.Tensor:
            flattened_index = index.reshape(batch * heads, length * candidate_count)
            selected = torch.gather(
                flat,
                1,
                flattened_index[:, :, None].expand(-1, -1, dimension),
            )
            return selected.reshape(
                batch, heads, length, candidate_count, dimension
            )

        sampled = gather(lower_index) * (1.0 - fraction[..., None])
        sampled = sampled + gather(upper_index) * fraction[..., None]
        return sampled, valid

    def _phase_sampling_positions(
        self,
        length: int,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        order = F.softplus(self.phase_log_order) + 0.25
        period_tokens = sample_rate_hz[:, None] / (
            shaft_speed_hz[:, None]
            * float(self.downsample_factor)
            * order[None, :]
        )
        cycles = torch.cat(
            (
                -torch.arange(
                    self.phase_cycle_count,
                    0,
                    -1,
                    device=device,
                    dtype=dtype,
                ),
                torch.arange(
                    1,
                    self.phase_cycle_count + 1,
                    device=device,
                    dtype=dtype,
                ),
            )
        )
        offsets = period_tokens[:, :, None] * cycles[None, None, :]
        positions = torch.arange(length, device=device, dtype=dtype)
        return positions[None, None, :, None] + offsets[:, :, None, :]

    def _additional_physical_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        length = query.shape[2]
        positions = self._phase_sampling_positions(
            length,
            shaft_speed_hz,
            sample_rate_hz,
            dtype=query.dtype,
            device=query.device,
        )
        sampled_key, valid = self._sample_tokens_at_positions(key, positions)
        sampled_value, _ = self._sample_tokens_at_positions(value, positions)
        logits = (
            query[:, :, :, None, :] * sampled_key
        ).sum(dim=-1) / math.sqrt(float(self.head_dimension))
        logits = logits + self.phase_cycle_bias[None, :, None, :]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
        attended = (weights[..., None] * sampled_value).sum(dim=-2)
        return attended * self.phase_residual_strength[None, :, None, None]


class PhaseAlignedFusedContinuousStyleAttention1D(
    _PhaseAlignedAttentionMixin,
    FusedContinuousStyleAttention1D,
):
    """Continuous style transfer with device-preserving mechanical-phase lookup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_phase_alignment()


@dataclass(frozen=True, slots=True)
class UnifiedGeneratorOutput:
    normalized_signal: torch.Tensor
    log_amplitude_mean: torch.Tensor
    log_amplitude_log_scale: torch.Tensor
    style: torch.Tensor
    location_mean: torch.Tensor | None = None
    location_log_scale: torch.Tensor | None = None
    group_normalized_signal: torch.Tensor | None = None
    identity_guidance_multiplier: torch.Tensor | None = None
    group_guidance_reference_detached: torch.Tensor | None = None
    identity_time_guidance_weight: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class UnifiedGeneratedSignal:
    normalized_signal: torch.Tensor
    location_offset: torch.Tensor
    restoration_scale: torch.Tensor
    physical_signal: torch.Tensor


class UnifiedSignalGenerator(nn.Module):
    """Source-free variable-length generator with one integrated style path."""

    def __init__(
        self,
        *,
        num_faults: int,
        num_fault_groups: int,
        condition_dimension: int,
        num_devices: int,
        base_channels: int = 32,
        context_dimension: int = 128,
        attention_heads: int = 4,
        style_attention_rank: int = 32,
        style_token_count: int = 8,
        lora_rank: int = 8,
        attention_radius: int = 16,
        attention_chunk_size: int = 256,
        affine_restoration: bool = False,
        style_mechanism: str = "legacy_affine",
        fault_identity_guidance_scale: float = 1.0,
        fault_identity_guidance_mode: str = "linear",
        fault_identity_guidance_schedule: str = CONSTANT_GUIDANCE,
        physics_reference_policy: str = LEGACY_CYCLIC_REFERENCE,
        device_rho_initial: float = 0.95,
        device_constraint_rho_initial: tuple[float, float, float] = (
            0.9,
            0.9,
            0.5,
        ),
    ) -> None:
        super().__init__()
        if min(
            num_faults,
            num_fault_groups,
            condition_dimension,
            num_devices,
            base_channels,
            context_dimension,
        ) <= 0:
            raise ValueError("generator dimensions must be positive")
        self.num_faults = num_faults
        self.num_fault_groups = num_fault_groups
        self.condition_dimension = condition_dimension
        self.physics_reference_policy = validate_reference_policy(physics_reference_policy, condition_dimension)
        if self.physics_reference_policy != LEGACY_CYCLIC_REFERENCE and style_mechanism != "legacy_affine":
            raise ValueError("known mechanical reference currently requires legacy_affine attention")
        self.num_devices = num_devices
        self.context_dimension = context_dimension
        self.affine_restoration = bool(affine_restoration)
        if not math.isfinite(fault_identity_guidance_scale) or fault_identity_guidance_scale < 1.0:
            raise ValueError("fault identity guidance must be finite and at least one")
        self.fault_identity_guidance_scale = float(fault_identity_guidance_scale)
        if fault_identity_guidance_mode not in IDENTITY_GUIDANCE_MODES:
            raise ValueError("unsupported fault identity guidance mode")
        self.fault_identity_guidance_mode = fault_identity_guidance_mode
        if fault_identity_guidance_schedule not in TIME_GUIDANCE_SCHEDULES:
            raise ValueError("unsupported fault identity guidance schedule")
        if fault_identity_guidance_schedule != CONSTANT_GUIDANCE and fault_identity_guidance_mode != "linear":
            raise ValueError("time-varying identity guidance requires the declared linear mode")
        self.fault_identity_guidance_schedule = fault_identity_guidance_schedule
        if fault_identity_guidance_mode == "unit_peak_projection" and self.affine_restoration:
            raise ValueError("unit peak projection requires non-affine max_abs signals")
        if style_mechanism not in {
            "legacy_affine",
            "rank_aware_legacy",
            "rank_aware_style_residual",
            "source_support_map",
            "source_support_extrapolation_gain_map",
            "device_rho_source_support",
            "device_constraint_fusion",
            "learned_device_constraint_field",
            "content_query_affine_style_repair",
            "support_content_query_affine_style_repair",
            "continuous_qkv_adaattn",
            "disentangled_cross_attention",
            "fused_continuous_qkv_adaattn",
            "shared_reference_qkv_adaattn",
            "phase_aligned_fused_qkv_adaattn",
            "relative_style_transport",
        }:
            raise ValueError("unsupported style mechanism")
        self.style_mechanism = style_mechanism
        self.condition_coordinates = (
            SourceRankConditionCoordinates(condition_dimension, num_devices)
            if style_mechanism in {
                "rank_aware_legacy",
                "rank_aware_style_residual",
            }
            else None
        )
        if style_mechanism == "device_rho_source_support":
            self.condition_support: SourceSupportConditionMap | None = (
                DeviceRhoSourceSupportConditionMap(
                    condition_dimension,
                    num_devices,
                    initial_rho=device_rho_initial,
                )
            )
        elif style_mechanism in {
            "source_support_map",
            "source_support_extrapolation_gain_map",
            "device_constraint_fusion",
            "learned_device_constraint_field",
            "support_content_query_affine_style_repair",
        }:
            self.condition_support = SourceSupportConditionMap(
                condition_dimension,
                num_devices,
            )
        else:
            self.condition_support = None
        self.style_field = AffineStyleAttention(
            condition_dimension,
            context_dimension,
            style_attention_rank,
            style_token_count,
            num_devices,
            lora_rank,
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(context_dimension, context_dimension),
            nn.SiLU(),
            nn.Linear(context_dimension, context_dimension),
        )
        self.fault_group_embedding = nn.Embedding(
            num_fault_groups, context_dimension
        )
        self.fault_embedding = nn.Embedding(num_faults, context_dimension)

        self.lifting = nn.Conv1d(1, base_channels, kernel_size=5, padding=2)
        self.encoder_0 = AdaptiveResidualBlock(
            base_channels,
            base_channels,
            context_dimension,
            num_devices,
            lora_rank,
        )
        self.down_0 = nn.Conv1d(
            base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1
        )
        self.encoder_1 = AdaptiveResidualBlock(
            2 * base_channels,
            2 * base_channels,
            context_dimension,
            num_devices,
            lora_rank,
        )
        self.down_1 = nn.Conv1d(
            2 * base_channels,
            4 * base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.middle = AdaptiveResidualBlock(
            4 * base_channels,
            4 * base_channels,
            context_dimension,
            num_devices,
            lora_rank,
        )
        if style_mechanism in {
            "continuous_qkv_adaattn",
            "disentangled_cross_attention",
        }:
            self.attention = ContinuousQKVStyleTransferAttention1D(
                4 * base_channels,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
                query_uses_condition=(
                    style_mechanism == "continuous_qkv_adaattn"
                ),
            )
        elif style_mechanism in {
            "content_query_affine_style_repair",
            "support_content_query_affine_style_repair",
        }:
            self.attention = ContentQueryAffineStyleRepairAttention1D(
                4 * base_channels,
                context_dimension,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        elif style_mechanism == "relative_style_transport":
            self.attention = RelativeStyleTransportAttention1D(
                4 * base_channels,
                context_dimension,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        elif style_mechanism == "fused_continuous_qkv_adaattn":
            self.attention = FusedContinuousStyleAttention1D(
                4 * base_channels,
                context_dimension,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        elif style_mechanism == "shared_reference_qkv_adaattn":
            self.attention = FusedContinuousStyleAttention1D(
                4 * base_channels,
                context_dimension,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
                shared_reference_transport=True,
            )
        elif style_mechanism == "phase_aligned_fused_qkv_adaattn":
            self.attention = PhaseAlignedFusedContinuousStyleAttention1D(
                4 * base_channels,
                context_dimension,
                condition_dimension,
                style_token_count,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        elif style_mechanism == "device_constraint_fusion":
            self.attention = DeviceConstraintFusionAttention1D(
                4 * base_channels,
                context_dimension,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
                initial_rho=device_constraint_rho_initial,
            )
        elif style_mechanism == "learned_device_constraint_field":
            self.attention = LearnedDeviceConstraintFieldAttention1D(
                4 * base_channels,
                context_dimension,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        else:
            self.attention = PhysicalStyleAttention1D(
                4 * base_channels,
                context_dimension,
                attention_heads,
                attention_radius,
                attention_chunk_size,
                4,
                num_devices,
                lora_rank,
            )
        self.decoder_1 = AdaptiveResidualBlock(
            6 * base_channels,
            2 * base_channels,
            context_dimension,
            num_devices,
            lora_rank,
        )
        self.decoder_0 = AdaptiveResidualBlock(
            3 * base_channels,
            base_channels,
            context_dimension,
            num_devices,
            lora_rank,
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output = nn.Conv1d(base_channels, 1, kernel_size=5, padding=2)
        self.amplitude_head_0 = RoutedLoRALinear(
            2 * context_dimension + 1,
            context_dimension,
            num_devices,
            lora_rank,
        )
        self.amplitude_head_1 = RoutedLoRALinear(
            context_dimension,
            4 if self.affine_restoration else 2,
            num_devices,
            lora_rank,
        )
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        if style_mechanism == "rank_aware_style_residual":
            with torch.random.fork_rng(devices=[]):
                self.condition_adapter: RankAwareStyleResidual | None = (
                    RankAwareStyleResidual(
                        condition_dimension,
                        context_dimension,
                        style_attention_rank,
                        style_token_count,
                        num_devices,
                    )
                )
        else:
            self.condition_adapter = None

    @torch.no_grad()
    def set_condition_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
        enabled: torch.Tensor,
    ) -> None:
        if self.condition_coordinates is None:
            raise RuntimeError("generator has no source-rank condition coordinates")
        self.condition_coordinates.set_geometry(origin, basis, rank, enabled)

    @torch.no_grad()
    def set_condition_support_geometry(
        self,
        origin: torch.Tensor,
        basis: torch.Tensor,
        rank: torch.Tensor,
        enabled: torch.Tensor,
        support_minimum: torch.Tensor,
        support_maximum: torch.Tensor,
        lower_extrapolation_gain: torch.Tensor | None = None,
        upper_extrapolation_gain: torch.Tensor | None = None,
    ) -> None:
        if self.condition_support is None:
            raise RuntimeError("generator has no source-support condition map")
        self.condition_support.set_geometry(
            origin,
            basis,
            rank,
            enabled,
            support_minimum,
            support_maximum,
            lower_extrapolation_gain,
            upper_extrapolation_gain,
        )

    def style(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        style_parameters = condition_parameters
        mapped_parameters: torch.Tensor | None = None
        if self.condition_support is not None:
            mapped_parameters = self.condition_support(
                condition_parameters,
                device_id,
            )
            style_parameters = mapped_parameters
        if self.style_mechanism == "rank_aware_legacy":
            if self.condition_coordinates is None:
                raise RuntimeError("rank-aware condition coordinates are missing")
            style_parameters = self.condition_coordinates(
                condition_parameters,
                device_id,
            )
        if isinstance(
            self.condition_support,
            DeviceRhoSourceSupportConditionMap,
        ):
            if mapped_parameters is None:
                raise RuntimeError("device-rho style mapping is missing")
            raw_result = self.style_field(
                condition_parameters,
                device_id,
                return_weights=return_weights,
            )
            mapped_result = self.style_field(
                mapped_parameters,
                device_id,
                return_weights=return_weights,
            )
            rho = self.condition_support.rho_for(device_id)[:, None]
            if return_weights:
                raw_style, raw_weights = raw_result
                mapped_style, mapped_weights = mapped_result
                result = (
                    raw_style + (1.0 - rho) * (mapped_style - raw_style),
                    raw_weights + (1.0 - rho) * (mapped_weights - raw_weights),
                )
            else:
                result = raw_result + (1.0 - rho) * (
                    mapped_result - raw_result
                )
        else:
            result = self.style_field(
                style_parameters,
                device_id,
                return_weights=return_weights,
            )
        if return_weights:
            style, weights = result
        else:
            style = result
            weights = None
        if self.style_mechanism == "rank_aware_style_residual":
            if self.condition_coordinates is None or self.condition_adapter is None:
                raise RuntimeError("rank-aware style residual is incomplete")
            principal = self.condition_coordinates.principal_coordinates(
                condition_parameters,
                device_id,
            )
            enabled = self.condition_coordinates.enabled.index_select(0, device_id)
            style = style + self.condition_adapter(principal, device_id, enabled)
        return (style, weights) if return_weights else style

    def effective_condition_parameters(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return the condition request actually used by the continuous style path."""
        if self.condition_support is None:
            return condition_parameters
        if isinstance(
            self.condition_support,
            DeviceRhoSourceSupportConditionMap,
        ):
            return self.condition_support.blended_parameters(
                condition_parameters,
                device_id,
            )
        return self.condition_support(condition_parameters, device_id)

    def device_rho_values(self) -> torch.Tensor | None:
        if not isinstance(
            self.condition_support,
            DeviceRhoSourceSupportConditionMap,
        ):
            return None
        return self.condition_support.rho_values()

    @property
    def constraint_attention(self) -> nn.Module | None:
        return getattr(self, "attention", None)

    def device_constraint_rho_values(self) -> torch.Tensor | None:
        if not isinstance(self.constraint_attention, DeviceConstraintFusionAttention1D):
            return None
        return self.constraint_attention.constraint_rho_values()

    def device_constraint_rho_for(
        self,
        device_id: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(self.constraint_attention, DeviceConstraintFusionAttention1D):
            return None
        return self.constraint_attention.constraint_rho_for(device_id)

    def condition_request_diagnostics(
        self,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if self.condition_support is None:
            return None
        support_mapped = self.condition_support(condition_parameters, device_id)
        effective = self.effective_condition_parameters(
            condition_parameters,
            device_id,
        )
        support = self.condition_support
        origin = support.origin.index_select(0, device_id)
        basis = support.basis.index_select(0, device_id)
        requested_coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            condition_parameters - origin,
        )
        effective_coordinates = torch.einsum(
            "bri,bi->br",
            basis,
            effective - origin,
        )
        rank = support.rank.index_select(0, device_id)
        active = torch.arange(
            self.condition_dimension,
            device=condition_parameters.device,
        )[None, :] < rank[:, None]
        enabled = support.enabled.index_select(0, device_id)
        lower = support.support_minimum.index_select(0, device_id)
        upper = support.support_maximum.index_select(0, device_id)
        lower_gain = support.lower_extrapolation_gain.index_select(0, device_id)
        upper_gain = support.upper_extrapolation_gain.index_select(0, device_id)
        lower_requested = enabled[:, None] & active & (requested_coordinates < lower)
        upper_requested = enabled[:, None] & active & (requested_coordinates > upper)
        applied_gain = torch.ones_like(requested_coordinates)
        applied_gain = torch.where(lower_requested, lower_gain, applied_gain)
        applied_gain = torch.where(upper_requested, upper_gain, applied_gain)
        residual = torch.linalg.vector_norm(
            condition_parameters - effective,
            dim=1,
        )
        active_residual = torch.linalg.vector_norm(
            torch.where(
                active,
                requested_coordinates - effective_coordinates,
                torch.zeros_like(requested_coordinates),
            ),
            dim=1,
        )
        unsupported_coordinates = torch.where(
            active,
            requested_coordinates,
            torch.minimum(torch.maximum(requested_coordinates, lower), upper),
        )
        unsupported_residual = torch.linalg.vector_norm(
            requested_coordinates - unsupported_coordinates,
            dim=1,
        )
        diagnostics = {
            "requested_parameters": condition_parameters,
            "effective_parameters": effective,
            "support_mapped_parameters": support_mapped,
            "requested_principal_coordinates": requested_coordinates,
            "effective_principal_coordinates": effective_coordinates,
            "projection_residual": residual,
            "unsupported_projection_residual": unsupported_residual,
            "active_extrapolation_residual": active_residual,
            "lower_extrapolation_requested": lower_requested,
            "upper_extrapolation_requested": upper_requested,
            "applied_extrapolation_gain": applied_gain,
            "active_extrapolation_adjusted": active_residual > 1e-7,
            "off_manifold_unsupported": unsupported_residual > 1e-7,
        }
        rho = self.device_rho_values()
        if rho is not None:
            diagnostics["device_rho"] = rho.index_select(0, device_id)
        constraint_rho = self.device_constraint_rho_for(device_id)
        if constraint_rho is not None:
            diagnostics["device_constraint_rho"] = constraint_rho
        return diagnostics

    def _validate_labels(
        self,
        fault: torch.Tensor,
        fault_group: torch.Tensor,
        device_id: torch.Tensor,
        batch_size: int,
    ) -> None:
        validate_fault_condition(fault, batch_size, self.num_faults, "fault")
        validate_fault_condition(fault_group, batch_size, self.num_fault_groups, "fault group")
        if device_id.shape != (batch_size,) or device_id.dtype != torch.long:
            raise ValueError("device must be a long tensor with shape [batch]")
        if torch.any(device_id < 0) or torch.any(device_id >= self.num_devices):
            raise ValueError("device index is outside the configured range")
        if fault.ndim != fault_group.ndim:
            raise ValueError("fault and fault group conditioning representations differ")

    def physical_shaft_reference(self, parameters: torch.Tensor, cyclic_reference_hz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return resolve_physical_reference(parameters, cyclic_reference_hz, self.physics_reference_policy)

    def forward(
        self,
        signal: torch.Tensor,
        time: torch.Tensor,
        condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        fault_group: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        normalization_length: torch.Tensor,
        *, detach_group_guidance_reference: bool = False,
    ) -> UnifiedGeneratorOutput:
        if type(detach_group_guidance_reference) is not bool:
            raise ValueError("group guidance reference detachment must be a boolean")
        if detach_group_guidance_reference and self.fault_identity_guidance_scale == 1.0:
            raise ValueError("group reference detachment requires an active guided branch")
        peak_projection = self.fault_identity_guidance_mode == "unit_peak_projection"
        if peak_projection and (signal.ndim != 3 or torch.any(normalization_length != signal.shape[-1])):
            raise ValueError("unit peak projection requires a complete parent waveform, not cropped views")
        shaft_speed_hz, _ = self.physical_shaft_reference(condition_parameters, shaft_speed_hz)
        arguments = (
            signal, time, condition_parameters, fault, fault_group, device_id,
            shaft_speed_hz, sample_rate_hz, normalization_length,
        )
        conditional = self._predict(*arguments, include_fault_identity=True)
        time_weight = None
        if self.fault_identity_guidance_schedule != CONSTANT_GUIDANCE:
            time_weight = scheduled_guidance_weight(time, self.fault_identity_guidance_scale, self.fault_identity_guidance_schedule)
            if time_weight.shape != (len(signal),):
                raise ValueError("time guidance batch size differs from the waveform")
        if self.fault_identity_guidance_scale == 1.0:
            if peak_projection:
                projected, multiplier = project_unit_peak(conditional.normalized_signal)
                return replace(conditional, normalized_signal=projected, identity_guidance_multiplier=multiplier)
            return replace(conditional, identity_time_guidance_weight=time_weight) if time_weight is not None else conditional
        group = self._predict(*arguments, include_fault_identity=False)
        guidance_reference = group.normalized_signal.detach() if detach_group_guidance_reference else group.normalized_signal
        if time_weight is None:
            prediction = conditional.normalized_signal + (
                self.fault_identity_guidance_scale - 1.0
            ) * (conditional.normalized_signal - guidance_reference)
        else:
            prediction = conditional.normalized_signal + time_weight[:, None, None] * (conditional.normalized_signal - guidance_reference)
        multiplier = None
        group_prediction = group.normalized_signal
        if self.fault_identity_guidance_mode == "conditional_rms_cap":
            prediction, multiplier = cap_guidance_rms(
                prediction, conditional.normalized_signal
            )
        elif peak_projection:
            prediction, multiplier = project_unit_peak(prediction)
            group_prediction, _ = project_unit_peak(group_prediction)
        return replace(
            conditional,
            normalized_signal=prediction,
            group_normalized_signal=group_prediction,
            identity_guidance_multiplier=multiplier,
            group_guidance_reference_detached=prediction.new_ones(()) if detach_group_guidance_reference else None,
            identity_time_guidance_weight=time_weight,
        )

    def _predict(
        self,
        signal: torch.Tensor,
        time: torch.Tensor,
        condition_parameters: torch.Tensor,
        fault: torch.Tensor,
        fault_group: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
        normalization_length: torch.Tensor,
        *,
        include_fault_identity: bool,
    ) -> UnifiedGeneratorOutput:
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError("generator input must have shape [batch,1,length]")
        batch_size, _, original_length = signal.shape
        if condition_parameters.shape != (batch_size, self.condition_dimension):
            raise ValueError("condition parameter shape differs")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        self._validate_labels(fault, fault_group, device_id, batch_size)
        for name, value in (
            ("shaft speed", shaft_speed_hz),
            ("sample rate", sample_rate_hz),
            ("normalization length", normalization_length),
        ):
            if name == "shaft speed" and self.physics_reference_policy != LEGACY_CYCLIC_REFERENCE:
                if value.shape != (batch_size,) or not torch.isfinite(value).all():
                    raise ValueError("shaft reference must be finite with shape [batch]")
                continue
            if value.shape != (batch_size,) or torch.any(value <= 0):
                raise ValueError(f"{name} must be positive with shape [batch]")

        style = self.style(condition_parameters, device_id)
        content = embed_fault_condition(self.fault_group_embedding, fault_group)
        if include_fault_identity:
            content = content + embed_fault_condition(self.fault_embedding, fault)
        content_context = (
            self.time_encoder(timestep_embedding(time, self.context_dimension))
            + content
        )
        if self.style_mechanism == "disentangled_cross_attention":
            context = content_context
        elif self.style_mechanism == "device_constraint_fusion":
            constraint_rho = self.device_constraint_rho_for(device_id)
            if constraint_rho is None:
                raise RuntimeError("device constraint rho is unavailable")
            context = content_context + constraint_rho[:, :1] * style
        else:
            context = content_context + style
        prediction = self._waveform_prediction(
            signal, context, style, condition_parameters, device_id,
            shaft_speed_hz, sample_rate_hz,
        )
        prediction = self._clean_prediction(signal, prediction, time)

        relative_log_length = torch.log(
            normalization_length.to(signal.dtype) / 1024.0
        )[:, None]
        amplitude_input = torch.cat((style, content, relative_log_length), dim=1)
        amplitude_hidden = F.silu(
            self.amplitude_head_0(amplitude_input, device_id)
        )
        amplitude = self.amplitude_head_1(amplitude_hidden, device_id)
        log_mean = amplitude[:, 0]
        log_scale = amplitude[:, 1].clamp(-6.0, 3.0)
        location_mean = amplitude[:, 2] if self.affine_restoration else None
        location_log_scale = (
            amplitude[:, 3].clamp(-6.0, 3.0)
            if self.affine_restoration
            else None
        )
        return UnifiedGeneratorOutput(
            normalized_signal=prediction,
            log_amplitude_mean=log_mean,
            log_amplitude_log_scale=log_scale,
            style=style,
            location_mean=location_mean,
            location_log_scale=location_log_scale,
        )

    def _clean_prediction(
        self, signal: torch.Tensor, prediction: torch.Tensor, time: torch.Tensor,
    ) -> torch.Tensor:
        return prediction

    def _waveform_prediction(
        self,
        signal: torch.Tensor,
        context: torch.Tensor,
        style: torch.Tensor,
        condition_parameters: torch.Tensor,
        device_id: torch.Tensor,
        shaft_speed_hz: torch.Tensor,
        sample_rate_hz: torch.Tensor,
    ) -> torch.Tensor:
        original_length = signal.shape[-1]
        pad = (-original_length) % 4
        hidden_input = F.pad(signal, (0, pad)) if pad else signal
        lifted = self.lifting(hidden_input)
        encoded_0 = self.encoder_0(lifted, context, device_id)
        encoded_1 = self.encoder_1(self.down_0(encoded_0), context, device_id)
        middle = self.middle(self.down_1(encoded_1), context, device_id)
        if self.style_mechanism in {
            "continuous_qkv_adaattn",
            "disentangled_cross_attention",
        }:
            middle = self.attention(
                middle,
                condition_parameters,
                device_id,
                shaft_speed_hz,
                sample_rate_hz,
            )
        elif self.style_mechanism in {
            "fused_continuous_qkv_adaattn",
            "shared_reference_qkv_adaattn",
            "phase_aligned_fused_qkv_adaattn",
            "relative_style_transport",
            "content_query_affine_style_repair",
            "support_content_query_affine_style_repair",
        }:
            middle = self.attention(
                middle,
                style,
                self.effective_condition_parameters(
                    condition_parameters,
                    device_id,
                ),
                device_id,
                shaft_speed_hz,
                sample_rate_hz,
            )
        else:
            middle = self.attention(
                middle,
                style,
                device_id,
                shaft_speed_hz,
                sample_rate_hz,
                **({"allow_nonpositive_shaft_reference": True} if self.physics_reference_policy != LEGACY_CYCLIC_REFERENCE else {}),
            )
        up_1 = decoder_linear_resize(middle, encoded_1.shape[-1])
        decoded_1 = self.decoder_1(
            torch.cat((up_1, encoded_1), dim=1), context, device_id
        )
        up_0 = decoder_linear_resize(decoded_1, encoded_0.shape[-1])
        decoded_0 = self.decoder_0(
            torch.cat((up_0, encoded_0), dim=1), context, device_id
        )
        prediction = self.output(F.silu(self.output_norm(decoded_0)))
        return prediction[..., :original_length]

    def lora_modules(self) -> Iterator[tuple[str, RoutedLoRALinear]]:
        for name, module in self.named_modules():
            if isinstance(module, RoutedLoRALinear):
                yield name, module


def _unified_terminal_inputs(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
    checkpoint_forward: bool = False,
 ) -> tuple[torch.Tensor, ...]:
    if steps < 2:
        raise ValueError("Heun sampling requires at least two steps")
    if not 0.0 < terminal_time < 1.0:
        raise ValueError("terminal time must lie in (0,1)")
    batch_size = len(initial_noise)
    normalization_length = torch.full(
        (batch_size,),
        initial_noise.shape[-1],
        device=initial_noise.device,
        dtype=torch.long,
    )
    times = torch.linspace(
        0.0,
        terminal_time,
        steps + 1,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
    )

    def predict_signal(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return model(
            state,
            time,
            condition_parameters,
            fault,
            fault_group,
            device_id,
            shaft_speed_hz,
            sample_rate_hz,
            normalization_length,
        ).normalized_signal

    def predict(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if checkpoint_forward:
            return checkpoint(predict_signal, state, time, use_reentrant=False)
        return predict_signal(state, time)

    state = initial_noise
    for index in range(steps):
        time = times[index].expand(batch_size)
        next_time = times[index + 1].expand(batch_size)
        step_size = times[index + 1] - times[index]
        prediction = predict(state, time)
        velocity = (prediction - state) / (1.0 - time[:, None, None])
        proposal = state + step_size * velocity
        next_prediction = predict(proposal, next_time)
        next_velocity = (next_prediction - proposal) / (
            1.0 - next_time[:, None, None]
        )
        state = state + 0.5 * step_size * (velocity + next_velocity)

    return (state, times[-1].expand(batch_size), condition_parameters, fault, fault_group,
            device_id, shaft_speed_hz, sample_rate_hz, normalization_length)


@torch.no_grad()
def integrate_unified_terminal_inputs(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
) -> tuple[torch.Tensor, ...]:
    return _unified_terminal_inputs(
        model, initial_noise, condition_parameters, fault, fault_group, device_id,
        shaft_speed_hz, sample_rate_hz, steps=steps, terminal_time=terminal_time,
    )


def differentiable_unified_sample(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
    checkpoint_forward: bool = True,
) -> torch.Tensor:
    if model.affine_restoration:
        raise ValueError("diagnostic sampling requires max-absolute scale-only restoration")
    inputs = _unified_terminal_inputs(
        model, initial_noise, condition_parameters, fault, fault_group, device_id,
        shaft_speed_hz, sample_rate_hz, steps=steps, terminal_time=terminal_time,
        checkpoint_forward=checkpoint_forward,
    )

    def final_signal(*arguments):
        return model(*arguments).normalized_signal

    normalized = checkpoint(final_signal, *inputs, use_reentrant=False) if checkpoint_forward else final_signal(*inputs)
    return normalized / normalized.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def _integrate_unified_generator(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
) -> UnifiedGeneratorOutput:
    inputs = integrate_unified_terminal_inputs(
        model, initial_noise, condition_parameters, fault, fault_group,
        device_id, shaft_speed_hz, sample_rate_hz, steps=steps, terminal_time=terminal_time,
    )
    return model(*inputs)


@torch.no_grad()
def sample_unified_generator(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
    amplitude_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    final = _integrate_unified_generator(
        model,
        initial_noise,
        condition_parameters,
        fault,
        fault_group,
        device_id,
        shaft_speed_hz,
        sample_rate_hz,
        steps=steps,
        terminal_time=terminal_time,
    )
    batch_size = len(initial_noise)
    normalized = final.normalized_signal
    maximum = normalized.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    normalized = normalized / maximum
    if amplitude_noise is None:
        amplitude_noise = torch.zeros(
            batch_size, device=initial_noise.device, dtype=initial_noise.dtype
        )
    if amplitude_noise.shape != (batch_size,):
        raise ValueError("amplitude noise must have shape [batch]")
    log_amplitude = final.log_amplitude_mean + (
        final.log_amplitude_log_scale.exp() * amplitude_noise
    )
    amplitude = log_amplitude.exp()
    physical = normalized * amplitude[:, None, None]
    return normalized, amplitude, physical


@torch.no_grad()
def sample_unified_generator_affine(
    model: UnifiedSignalGenerator,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    fault_group: torch.Tensor,
    device_id: torch.Tensor,
    shaft_speed_hz: torch.Tensor,
    sample_rate_hz: torch.Tensor,
    *,
    normalization_method: str,
    steps: int,
    terminal_time: float,
    scale_noise: torch.Tensor | None = None,
    location_noise: torch.Tensor | None = None,
) -> UnifiedGeneratedSignal:
    if normalization_method not in {"max_abs", "rms", "mean_ac_rms"}:
        raise ValueError(f"unsupported normalization method {normalization_method}")
    if not model.affine_restoration:
        raise ValueError("affine sampling requires an affine-restoration generator")
    final = _integrate_unified_generator(
        model,
        initial_noise,
        condition_parameters,
        fault,
        fault_group,
        device_id,
        shaft_speed_hz,
        sample_rate_hz,
        steps=steps,
        terminal_time=terminal_time,
    )
    normalized = final.normalized_signal
    if normalization_method == "max_abs":
        statistic = normalized.abs().amax(dim=(1, 2), keepdim=True)
    else:
        if normalization_method == "mean_ac_rms":
            normalized = normalized - normalized.mean(dim=(1, 2), keepdim=True)
        statistic = normalized.square().mean(dim=(1, 2), keepdim=True).sqrt()
    normalized = normalized / statistic.clamp_min(1e-8)

    batch_size = len(initial_noise)
    if scale_noise is None:
        scale_noise = torch.zeros(
            batch_size, device=initial_noise.device, dtype=initial_noise.dtype
        )
    if scale_noise.shape != (batch_size,):
        raise ValueError("scale noise must have shape [batch]")
    log_restoration_scale = final.log_amplitude_mean + (
        final.log_amplitude_log_scale.exp() * scale_noise
    )
    restoration_scale = log_restoration_scale.exp()

    if final.location_mean is None or final.location_log_scale is None:
        raise RuntimeError("affine restoration outputs are missing")
    if location_noise is None:
        location_noise = torch.zeros(
            batch_size, device=initial_noise.device, dtype=initial_noise.dtype
        )
    if location_noise.shape != (batch_size,):
        raise ValueError("location noise must have shape [batch]")
    predicted_location = final.location_mean + (
        final.location_log_scale.exp() * location_noise
    )
    location = (
        predicted_location
        if normalization_method == "mean_ac_rms"
        else torch.zeros_like(predicted_location)
    )
    physical = (
        location[:, None, None]
        + normalized * restoration_scale[:, None, None]
    )
    return UnifiedGeneratedSignal(
        normalized_signal=normalized,
        location_offset=location,
        restoration_scale=restoration_scale,
        physical_signal=physical,
    )
