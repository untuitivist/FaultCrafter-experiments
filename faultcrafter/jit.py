from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from faultcrafter.training_numerics import detached_lower_median


def _modulate(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return hidden * (1.0 + scale[:, None, :]) + shift[:, None, :]


def sinusoidal_time_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError(f"time must have shape [batch], got {tuple(time.shape)}")
    if dimension < 2:
        raise ValueError("time embedding dimension must be at least two")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / half
    )
    angles = time.float()[:, None] * 1_000.0 * frequencies[None, :]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=1)
    return embedding.to(dtype=time.dtype)


@dataclass(frozen=True)
class PatchLayout:
    signal_length: int
    padded_length: int
    starts: torch.Tensor
    centers: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor


def _validated_lengths(
    signal: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, channels, signal_length = signal.shape
    if channels != 1 or signal_length < 1:
        raise ValueError("waveform must have shape [batch,1,length] with positive length")
    if lengths is None:
        return torch.full(
            (batch_size,),
            signal_length,
            dtype=torch.long,
            device=signal.device,
        )
    if lengths.shape != (batch_size,):
        raise ValueError("lengths must have shape [batch]")
    lengths = lengths.to(device=signal.device, dtype=torch.long)
    if torch.any(lengths < 1) or torch.any(lengths > signal_length):
        raise ValueError("each waveform length must be within the padded signal extent")
    return lengths


def _patchify(
    signal: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    patch_size: int,
    patch_stride: int,
) -> tuple[torch.Tensor, PatchLayout]:
    if signal.ndim != 3:
        raise ValueError("waveform must have shape [batch,1,length]")
    valid_lengths = _validated_lengths(signal, lengths)
    signal_length = signal.shape[-1]
    patch_count = max(
        1,
        math.ceil(max(signal_length - patch_size, 0) / patch_stride) + 1,
    )
    padded_length = (patch_count - 1) * patch_stride + patch_size
    sample_mask = (
        torch.arange(signal_length, device=signal.device)[None, :]
        < valid_lengths[:, None]
    )
    masked = signal[:, 0] * sample_mask.to(signal.dtype)
    padded = F.pad(masked, (0, padded_length - signal_length))
    patches = padded.unfold(1, patch_size, patch_stride)
    starts = torch.arange(patch_count, device=signal.device) * patch_stride
    valid = starts[None, :] < valid_lengths[:, None]
    available = (valid_lengths[:, None] - starts[None, :]).clamp(
        min=1,
        max=patch_size,
    )
    centers = starts[None, :].to(signal.dtype) + (
        available.to(signal.dtype) - 1.0
    ) / 2.0
    centers = torch.where(valid, centers, torch.zeros_like(centers))
    return patches, PatchLayout(
        signal_length=signal_length,
        padded_length=padded_length,
        starts=starts,
        centers=centers,
        valid=valid,
        lengths=valid_lengths,
    )


def _overlap_add(
    patches: torch.Tensor,
    layout: PatchLayout,
    *,
    patch_size: int,
    patch_stride: int,
) -> torch.Tensor:
    if patches.ndim != 3 or patches.shape[1] != layout.valid.shape[1]:
        raise ValueError("predicted patches do not match the target patch layout")
    valid = layout.valid[:, :, None].to(patches.dtype)
    patches = patches * valid
    fold_input = patches.transpose(1, 2)
    output = F.fold(
        fold_input,
        output_size=(1, layout.padded_length),
        kernel_size=(1, patch_size),
        stride=(1, patch_stride),
    ).squeeze(2)
    count = F.fold(
        valid.expand_as(patches).transpose(1, 2),
        output_size=(1, layout.padded_length),
        kernel_size=(1, patch_size),
        stride=(1, patch_stride),
    ).squeeze(2)
    output = output / count.clamp_min(1.0)
    output = output[:, :, : layout.signal_length]
    sample_mask = (
        torch.arange(layout.signal_length, device=output.device)[None, :]
        < layout.lengths[:, None]
    )
    return output * sample_mask[:, None, :].to(output.dtype)


class PhysicalConditionEncoder(nn.Module):
    def __init__(self, num_parameters: int, hidden_size: int) -> None:
        super().__init__()
        self.absolute_linear = nn.Linear(num_parameters, hidden_size)
        self.absolute_nonlinear = nn.Sequential(
            nn.LayerNorm(num_parameters),
            nn.Linear(num_parameters, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.transfer_linear = nn.Linear(num_parameters, hidden_size)
        self.transfer_nonlinear = nn.Sequential(
            nn.LayerNorm(num_parameters),
            nn.Linear(num_parameters, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.null_absolute = nn.Parameter(torch.empty(1, hidden_size))
        self.null_transfer = nn.Parameter(torch.empty(1, hidden_size))

    @staticmethod
    def _select(
        encoded: torch.Tensor,
        null: torch.Tensor,
        drop_parameters: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            drop_parameters[:, None],
            null.expand(len(encoded), -1),
            encoded,
        )

    def forward(
        self,
        source_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
        drop_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.absolute_linear(
            source_parameters
        ) + self.absolute_nonlinear(source_parameters)
        target = self.absolute_linear(
            target_parameters
        ) + self.absolute_nonlinear(target_parameters)
        difference = target_parameters - source_parameters
        transfer = self.transfer_linear(
            difference
        ) + self.transfer_nonlinear(difference)
        return (
            self._select(source, self.null_absolute, drop_parameters),
            self._select(target, self.null_absolute, drop_parameters),
            self._select(transfer, self.null_transfer, drop_parameters),
        )


class ConditionedLowRankLinear(nn.Module):
    """Linear projection with a context-dependent low-rank weight update."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_size: int,
        rank: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("conditional projection rank must be positive")
        self.base = nn.Linear(input_size, output_size, bias=bias)
        self.left = nn.Linear(input_size, rank, bias=False)
        self.context_norm = nn.LayerNorm(context_size)
        self.coefficients = nn.Linear(context_size, rank, bias=False)
        self.right = nn.Linear(rank, output_size, bias=False)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or context.shape != (len(values), self.context_norm.normalized_shape[0]):
            raise ValueError("projection values or context have incompatible shapes")
        update = self.left(values) * self.coefficients(
            self.context_norm(context)
        )[:, None, :]
        return self.base(values) + self.right(update)

    def reset_conditional_update(self) -> None:
        nn.init.zeros_(self.right.weight)


def _rotate_section(
    values: torch.Tensor,
    positions: torch.Tensor,
    *,
    start: int,
    width: int,
    base: float,
) -> torch.Tensor:
    if width == 0:
        return values
    section = values[..., start : start + width]
    even = section[..., 0::2]
    odd = section[..., 1::2]
    inverse_frequency = torch.exp(
        -math.log(base)
        * torch.arange(width // 2, device=values.device, dtype=torch.float32)
        / max(width // 2, 1)
    ).to(values.dtype)
    angles = positions[:, None, :, None].to(values.dtype) * inverse_frequency
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(start_dim=-2)
    return torch.cat(
        (
            values[..., :start],
            rotated,
            values[..., start + width :],
        ),
        dim=-1,
    )


def apply_time_angle_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_time_position: torch.Tensor,
    key_time_position: torch.Tensor,
    query_angle_position: torch.Tensor,
    key_angle_position: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch,heads,tokens,head_dim]")
    head_dimension = query.shape[-1]
    if key.shape[-1] != head_dimension:
        raise ValueError("query and key head dimensions differ")
    rotary_width = head_dimension - head_dimension % 4
    if rotary_width < 4:
        return query, key
    time_width = rotary_width // 2
    angle_width = rotary_width - time_width
    query = _rotate_section(
        query,
        query_time_position,
        start=0,
        width=time_width,
        base=base,
    )
    key = _rotate_section(
        key,
        key_time_position,
        start=0,
        width=time_width,
        base=base,
    )
    query = _rotate_section(
        query,
        query_angle_position,
        start=time_width,
        width=angle_width,
        base=base,
    )
    key = _rotate_section(
        key,
        key_angle_position,
        start=time_width,
        width=angle_width,
        base=base,
    )
    return query, key


class ConditionalTransportAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        conditioning_rank: int,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        if query_chunk_size < 1:
            raise ValueError("query chunk size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query_chunk_size = query_chunk_size
        self.query_projection = ConditionedLowRankLinear(
            hidden_size,
            hidden_size,
            hidden_size,
            conditioning_rank,
        )
        self.key_projection = ConditionedLowRankLinear(
            hidden_size,
            hidden_size,
            hidden_size,
            conditioning_rank,
        )
        self.value_projection = ConditionedLowRankLinear(
            hidden_size,
            hidden_size,
            hidden_size,
            conditioning_rank,
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.source_role = nn.Parameter(torch.empty(1, hidden_size))
        self.target_role = nn.Parameter(torch.empty(1, hidden_size))
        self.statistic_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = values.shape
        return values.reshape(
            batch_size,
            tokens,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def forward(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_physical_context: torch.Tensor,
        target_physical_context: torch.Tensor,
        source_content_context: torch.Tensor,
        target_content_context: torch.Tensor,
        target_time_position: torch.Tensor,
        source_time_position: torch.Tensor,
        target_angle_position: torch.Tensor,
        source_angle_position: torch.Tensor,
        target_valid: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        source_physical = source_physical_context + self.source_role
        target_physical = target_physical_context + self.target_role
        source_content = source_content_context + self.source_role
        target_content = target_content_context + self.target_role

        query = self._heads(
            self.query_projection(target_tokens, target_physical)
        )
        source_key = self._heads(
            self.key_projection(source_tokens, source_physical)
        )
        target_key = self._heads(
            self.key_projection(target_tokens, target_physical)
        )
        source_value = self._heads(
            self.value_projection(source_tokens, source_content)
        )
        target_value = self._heads(
            self.value_projection(target_tokens, target_content)
        )
        key = torch.cat((source_key, target_key), dim=2)
        value = torch.cat((source_value, target_value), dim=2)
        key_time = torch.cat((source_time_position, target_time_position), dim=1)
        key_angle = torch.cat((source_angle_position, target_angle_position), dim=1)
        query, key = apply_time_angle_rope(
            query,
            key,
            target_time_position,
            key_time,
            target_angle_position,
            key_angle,
        )

        memory_valid = torch.cat((source_valid, target_valid), dim=1)
        normalized_target_value = F.layer_norm(
            target_value,
            (self.head_size,),
        )
        statistic_gate = self.statistic_gate(target_physical_context).reshape(
            len(target_tokens),
            self.num_heads,
            1,
            self.head_size,
        )
        chunks: list[torch.Tensor] = []
        scale = self.head_size**-0.5
        minimum = -torch.finfo(query.dtype).max
        for begin in range(0, target_tokens.shape[1], self.query_chunk_size):
            end = min(begin + self.query_chunk_size, target_tokens.shape[1])
            chunk_query = query[:, :, begin:end]
            logits = torch.matmul(chunk_query, key.transpose(-2, -1)) * scale
            logits = logits.masked_fill(
                ~memory_valid[:, None, None, :],
                minimum,
            )
            weights = torch.softmax(logits, dim=-1)
            weighted_mean = torch.matmul(weights, value)
            weighted_second_moment = torch.matmul(weights, value.square())
            weighted_standard_deviation = (
                weighted_second_moment - weighted_mean.square()
            ).clamp_min(1e-6).sqrt()
            chunks.append(
                weighted_mean
                + statistic_gate
                * weighted_standard_deviation
                * normalized_target_value[:, :, begin:end]
            )
        attended = torch.cat(chunks, dim=2)
        attended = attended.transpose(1, 2).reshape(
            len(target_tokens),
            target_tokens.shape[1],
            self.hidden_size,
        )
        attended = self.output_projection(attended)
        return attended * target_valid[:, :, None].to(attended.dtype)

    def reset_conditioning(self) -> None:
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
        ):
            projection.reset_conditional_update()
        nn.init.zeros_(self.statistic_gate[-1].weight)
        nn.init.zeros_(self.statistic_gate[-1].bias)


class ConditionalTransportBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        conditioning_rank: int,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        mlp_size = int(round(hidden_size * mlp_ratio))
        self.target_attention_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.source_attention_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.attention_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 3),
        )
        self.attention = ConditionalTransportAttention(
            hidden_size,
            num_heads,
            conditioning_rank,
            query_chunk_size,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_size, hidden_size),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_physical_context: torch.Tensor,
        target_physical_context: torch.Tensor,
        source_content_context: torch.Tensor,
        target_content_context: torch.Tensor,
        target_time_position: torch.Tensor,
        source_time_position: torch.Tensor,
        target_angle_position: torch.Tensor,
        source_angle_position: torch.Tensor,
        target_valid: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> torch.Tensor:
        attention_shift, attention_scale, attention_gate = (
            self.attention_modulation(target_content_context).chunk(3, dim=-1)
        )
        attention_input = _modulate(
            self.target_attention_norm(target_tokens),
            attention_shift,
            attention_scale,
        )
        attention_output = self.attention(
            attention_input,
            self.source_attention_norm(source_tokens),
            source_physical_context,
            target_physical_context,
            source_content_context,
            target_content_context,
            target_time_position,
            source_time_position,
            target_angle_position,
            source_angle_position,
            target_valid,
            source_valid,
        )
        target_tokens = (
            target_tokens + attention_gate[:, None, :] * attention_output
        )
        mlp_shift, mlp_scale, mlp_gate = self.mlp_modulation(
            target_content_context
        ).chunk(3, dim=-1)
        mlp_input = _modulate(
            self.mlp_norm(target_tokens),
            mlp_shift,
            mlp_scale,
        )
        target_tokens = target_tokens + mlp_gate[:, None, :] * self.mlp(mlp_input)
        return target_tokens * target_valid[:, :, None].to(target_tokens.dtype)


class ConditionalTransportJiT(nn.Module):
    """Variable-length direct-x JiT conditioned on a source waveform and parameters."""

    def __init__(
        self,
        *,
        patch_size: int,
        patch_stride: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        num_faults: int,
        num_condition_parameters: int,
        sample_rate_hz: float,
        speed_parameter_index: int | None,
        speed_to_hz: float = 1.0,
        parameter_mean: Sequence[float] | torch.Tensor | None = None,
        parameter_scale: Sequence[float] | torch.Tensor | None = None,
        conditioning_rank: int = 8,
        query_chunk_size: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if patch_size < 2 or patch_stride < 1 or patch_stride > patch_size:
            raise ValueError("patch size and stride are invalid")
        if hidden_size < 4 or hidden_size % num_heads:
            raise ValueError("hidden size must be positive and divisible by num_heads")
        if depth < 1 or num_faults < 1 or num_condition_parameters < 1:
            raise ValueError("depth, faults and condition parameter count must be positive")
        if sample_rate_hz <= 0 or speed_to_hz <= 0:
            raise ValueError("sample rate and speed conversion must be positive")
        if speed_parameter_index is not None and not (
            0 <= speed_parameter_index < num_condition_parameters
        ):
            raise ValueError("speed parameter index is outside the condition vector")

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hidden_size = hidden_size
        self.num_faults = num_faults
        self.num_condition_parameters = num_condition_parameters
        self.sample_rate_hz = float(sample_rate_hz)
        self.speed_parameter_index = speed_parameter_index
        self.speed_to_hz = float(speed_to_hz)

        mean = (
            torch.zeros(num_condition_parameters, dtype=torch.float32)
            if parameter_mean is None
            else torch.as_tensor(parameter_mean, dtype=torch.float32)
        )
        scale = (
            torch.ones(num_condition_parameters, dtype=torch.float32)
            if parameter_scale is None
            else torch.as_tensor(parameter_scale, dtype=torch.float32)
        )
        if mean.shape != (num_condition_parameters,) or scale.shape != (
            num_condition_parameters,
        ):
            raise ValueError("parameter mean and scale must match the condition vector")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("parameter scaling must be finite")
        if torch.any(scale <= 0):
            raise ValueError("parameter scales must be positive")
        self.register_buffer("parameter_mean", mean)
        self.register_buffer("parameter_scale", scale)

        self.patch_embedding = nn.Linear(patch_size, hidden_size)
        self.position_convolution = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
        )
        self.source_type = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.target_type = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.condition_encoder = PhysicalConditionEncoder(
            num_condition_parameters,
            hidden_size,
        )
        self.fault_embedding = nn.Embedding(num_faults, hidden_size)
        self.null_fault = nn.Parameter(torch.empty(1, hidden_size))
        self.blocks = nn.ModuleList(
            ConditionalTransportBlock(
                hidden_size,
                num_heads,
                mlp_ratio,
                conditioning_rank,
                query_chunk_size,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.output_projection = nn.Linear(hidden_size, patch_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for parameter in (
            self.source_type,
            self.target_type,
            self.condition_encoder.null_absolute,
            self.condition_encoder.null_transfer,
            self.null_fault,
        ):
            nn.init.normal_(parameter, std=0.02)
        nn.init.normal_(self.fault_embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attention.source_role, std=0.02)
            nn.init.normal_(block.attention.target_role, std=0.02)
            nn.init.zeros_(block.attention_modulation[-1].weight)
            nn.init.zeros_(block.attention_modulation[-1].bias)
            nn.init.zeros_(block.mlp_modulation[-1].weight)
            nn.init.zeros_(block.mlp_modulation[-1].bias)
            block.attention.reset_conditioning()
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _normalized_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return (parameters - self.parameter_mean) / self.parameter_scale

    def _speed_hz(
        self,
        parameters: torch.Tensor,
        drop_parameters: torch.Tensor,
    ) -> torch.Tensor:
        if self.speed_parameter_index is None:
            return torch.zeros(
                len(parameters),
                device=parameters.device,
                dtype=parameters.dtype,
            )
        speed = parameters[:, self.speed_parameter_index] * self.speed_to_hz
        if torch.any((speed <= 0) & ~drop_parameters):
            raise ValueError("non-dropped rotational speed must be positive")
        return torch.where(drop_parameters, torch.zeros_like(speed), speed)

    def _tokens(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor | None,
        token_type: torch.Tensor,
    ) -> tuple[torch.Tensor, PatchLayout]:
        patches, layout = _patchify(
            waveform,
            lengths,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        tokens = self.patch_embedding(patches) + token_type
        tokens = tokens + self.position_convolution(
            tokens.transpose(1, 2)
        ).transpose(1, 2)
        tokens = tokens * layout.valid[:, :, None].to(tokens.dtype)
        return tokens, layout

    def forward(
        self,
        noised_target: torch.Tensor,
        source_waveform: torch.Tensor,
        time: torch.Tensor,
        source_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
        fault: torch.Tensor,
        *,
        target_lengths: torch.Tensor | None = None,
        source_lengths: torch.Tensor | None = None,
        drop_parameters: torch.Tensor | None = None,
        drop_fault: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noised_target.ndim != 3 or source_waveform.ndim != 3:
            raise ValueError("source and target waveforms must have shape [batch,1,length]")
        batch_size = len(noised_target)
        if len(source_waveform) != batch_size:
            raise ValueError("source and target batch sizes differ")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        expected_parameters = (batch_size, self.num_condition_parameters)
        if source_parameters.shape != expected_parameters:
            raise ValueError("source parameter shape does not match the model")
        if target_parameters.shape != expected_parameters:
            raise ValueError("target parameter shape does not match the model")
        if fault.shape != (batch_size,):
            raise ValueError("fault must have shape [batch]")
        if torch.any(fault < 0) or torch.any(fault >= self.num_faults):
            raise ValueError("fault index is outside the configured label space")
        if not torch.isfinite(source_parameters).all() or not torch.isfinite(
            target_parameters
        ).all():
            raise ValueError("condition parameters must be finite")

        if drop_parameters is None:
            drop_parameters = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=noised_target.device,
            )
        if drop_fault is None:
            drop_fault = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=noised_target.device,
            )
        if drop_parameters.shape != (batch_size,) or drop_fault.shape != (
            batch_size,
        ):
            raise ValueError("condition drop masks must have shape [batch]")
        drop_parameters = drop_parameters.to(
            device=noised_target.device,
            dtype=torch.bool,
        )
        drop_fault = drop_fault.to(device=noised_target.device, dtype=torch.bool)

        target_tokens, target_layout = self._tokens(
            noised_target,
            target_lengths,
            self.target_type,
        )
        source_tokens, source_layout = self._tokens(
            source_waveform,
            source_lengths,
            self.source_type,
        )
        normalized_source = self._normalized_parameters(source_parameters)
        normalized_target = self._normalized_parameters(target_parameters)
        time_context = self.time_encoder(
            sinusoidal_time_embedding(time, self.hidden_size)
        )
        source_condition, target_condition, transfer_condition = (
            self.condition_encoder(
                normalized_source,
                normalized_target,
                drop_parameters,
            )
        )
        fault_context = self.fault_embedding(fault)
        fault_context = torch.where(
            drop_fault[:, None],
            self.null_fault.expand(batch_size, -1),
            fault_context,
        )
        source_physical_context = source_condition + transfer_condition
        target_physical_context = (
            time_context + target_condition + transfer_condition
        )
        source_content_context = source_condition + fault_context
        target_content_context = target_physical_context + fault_context

        source_speed_hz = self._speed_hz(source_parameters, drop_parameters)
        target_speed_hz = self._speed_hz(target_parameters, drop_parameters)
        source_time_position = source_layout.centers / self.patch_stride
        target_time_position = target_layout.centers / self.patch_stride
        source_angle_position = (
            2.0
            * math.pi
            * source_layout.centers
            * source_speed_hz[:, None]
            / self.sample_rate_hz
        )
        target_angle_position = (
            2.0
            * math.pi
            * target_layout.centers
            * target_speed_hz[:, None]
            / self.sample_rate_hz
        )

        for block in self.blocks:
            target_tokens = block(
                target_tokens,
                source_tokens,
                source_physical_context,
                target_physical_context,
                source_content_context,
                target_content_context,
                target_time_position,
                source_time_position,
                target_angle_position,
                source_angle_position,
                target_layout.valid,
                source_layout.valid,
            )
        shift, scale = self.final_modulation(target_content_context).chunk(2, dim=-1)
        target_tokens = _modulate(
            self.final_norm(target_tokens),
            shift,
            scale,
        )
        predicted_patches = self.output_projection(target_tokens)
        return _overlap_add(
            predicted_patches,
            target_layout,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )


def direct_x_transport_loss(
    prediction: torch.Tensor,
    clean_target: torch.Tensor,
    time: torch.Tensor,
    *,
    minimum_noise_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != clean_target.shape:
        raise ValueError("prediction and clean target shapes differ")
    if time.shape != (len(prediction),):
        raise ValueError("time must have shape [batch]")
    if not 0.0 < minimum_noise_fraction <= 1.0:
        raise ValueError("minimum noise fraction must be in (0,1]")
    clean_error = (
        (prediction - clean_target).square().flatten(start_dim=1).mean(dim=1)
    )
    noise_fraction = (1.0 - time).clamp_min(minimum_noise_fraction)
    velocity_equivalent = clean_error / noise_fraction.square()
    return velocity_equivalent.mean(), clean_error.mean()


def normalized_log_power_spectrum(signal: torch.Tensor) -> torch.Tensor:
    if signal.ndim != 3 or signal.shape[1] != 1 or signal.shape[-1] < 4:
        raise ValueError("signal must have shape [batch,1,length>=4]")
    window = torch.hann_window(
        signal.shape[-1],
        device=signal.device,
        dtype=signal.dtype,
    )
    power = torch.fft.rfft(signal[:, 0] * window, dim=-1).abs().square()
    floor = detached_lower_median(power).clamp_min(1e-12)
    features = torch.log(power + floor * 1e-6)
    features = features - features.mean(dim=1, keepdim=True)
    return F.normalize(features, dim=1, eps=1e-8)


def _hilbert_envelope(signal: torch.Tensor) -> torch.Tensor:
    length = signal.shape[-1]
    spectrum = torch.fft.fft(signal, dim=-1)
    multiplier = signal.new_zeros(length)
    multiplier[0] = 1.0
    if length % 2 == 0:
        multiplier[length // 2] = 1.0
        multiplier[1 : length // 2] = 2.0
    else:
        multiplier[1 : (length + 1) // 2] = 2.0
    analytic = torch.fft.ifft(spectrum * multiplier, dim=-1)
    return analytic.abs()


def normalized_log_order_envelope(
    signal: torch.Tensor,
    speed_hz: torch.Tensor,
    *,
    sample_rate_hz: float,
    maximum_order: float,
    feature_count: int,
) -> torch.Tensor:
    if signal.ndim != 3 or signal.shape[1] != 1 or signal.shape[-1] < 4:
        raise ValueError("signal must have shape [batch,1,length>=4]")
    if speed_hz.shape != (len(signal),) or torch.any(speed_hz <= 0):
        raise ValueError("speed_hz must contain one positive speed per waveform")
    if sample_rate_hz <= 0 or maximum_order <= 0 or feature_count < 2:
        raise ValueError("order feature settings must be positive")
    envelope = _hilbert_envelope(signal[:, 0])
    envelope = envelope - envelope.mean(dim=1, keepdim=True)
    window = torch.hann_window(
        signal.shape[-1],
        device=signal.device,
        dtype=signal.dtype,
    )
    magnitude = torch.fft.rfft(envelope * window, dim=-1).abs()
    frequency_step = sample_rate_hz / signal.shape[-1]
    orders = torch.linspace(
        0.0,
        maximum_order,
        feature_count,
        device=signal.device,
        dtype=signal.dtype,
    )
    fractional_index = speed_hz[:, None] * orders[None, :] / frequency_step
    maximum_index = magnitude.shape[1] - 1
    lower = fractional_index.floor().long().clamp(0, maximum_index)
    upper = (lower + 1).clamp(0, maximum_index)
    weight = (fractional_index - lower.to(fractional_index.dtype)).clamp(0.0, 1.0)
    lower_value = magnitude.gather(1, lower)
    upper_value = magnitude.gather(1, upper)
    sampled = lower_value * (1.0 - weight) + upper_value * weight
    sampled = torch.where(
        fractional_index <= maximum_index,
        sampled,
        torch.zeros_like(sampled),
    )
    floor = detached_lower_median(sampled).clamp_min(1e-12)
    features = torch.log(sampled + floor * 1e-6)
    features = features - features.mean(dim=1, keepdim=True)
    return F.normalize(features, dim=1, eps=1e-8)


def source_visible_transport_objective(
    prediction: torch.Tensor,
    clean_target: torch.Tensor,
    source_waveform: torch.Tensor,
    time: torch.Tensor,
    source_speed_hz: torch.Tensor,
    target_speed_hz: torch.Tensor,
    *,
    sample_rate_hz: float,
    minimum_noise_fraction: float,
    spectral_weight: float,
    order_weight: float,
    maximum_order: float,
    order_feature_count: int,
) -> dict[str, torch.Tensor]:
    if spectral_weight < 0 or order_weight < 0:
        raise ValueError("physical consistency weights cannot be negative")
    velocity, clean = direct_x_transport_loss(
        prediction,
        clean_target,
        time,
        minimum_noise_fraction=minimum_noise_fraction,
    )
    spectral = (
        normalized_log_power_spectrum(prediction)
        - normalized_log_power_spectrum(clean_target)
    ).square().mean()
    generated_order = normalized_log_order_envelope(
        prediction,
        target_speed_hz,
        sample_rate_hz=sample_rate_hz,
        maximum_order=maximum_order,
        feature_count=order_feature_count,
    )
    source_order = normalized_log_order_envelope(
        source_waveform,
        source_speed_hz,
        sample_rate_hz=sample_rate_hz,
        maximum_order=maximum_order,
        feature_count=order_feature_count,
    )
    order = (1.0 - (generated_order * source_order).sum(dim=1)).mean()
    total = velocity + spectral_weight * spectral + order_weight * order
    return {
        "loss": total,
        "velocity_equivalent": velocity,
        "clean_mse": clean,
        "spectral": spectral,
        "order": order,
    }


def source_conditioned_base(
    source_waveform: torch.Tensor,
    noise: torch.Tensor,
    source_content_fraction: float,
) -> torch.Tensor:
    if source_waveform.shape != noise.shape:
        raise ValueError("source waveform and noise shapes differ")
    if not 0.0 <= source_content_fraction < 1.0:
        raise ValueError("source content fraction must be in [0,1)")
    return (
        math.sqrt(source_content_fraction) * source_waveform
        + math.sqrt(1.0 - source_content_fraction) * noise
    )


def _guided_prediction(
    model: ConditionalTransportJiT,
    state: torch.Tensor,
    source_waveform: torch.Tensor,
    time: torch.Tensor,
    source_parameters: torch.Tensor,
    target_parameters: torch.Tensor,
    fault: torch.Tensor,
    *,
    target_lengths: torch.Tensor | None,
    source_lengths: torch.Tensor | None,
    condition_guidance: float,
    fault_guidance: float,
) -> torch.Tensor:
    full = model(
        state,
        source_waveform,
        time,
        source_parameters,
        target_parameters,
        fault,
        target_lengths=target_lengths,
        source_lengths=source_lengths,
    )
    if condition_guidance == 1.0 and fault_guidance == 1.0:
        return full
    batch_size = len(state)
    drop = torch.ones(batch_size, dtype=torch.bool, device=state.device)
    keep = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
    physical_only = model(
        state,
        source_waveform,
        time,
        source_parameters,
        target_parameters,
        fault,
        target_lengths=target_lengths,
        source_lengths=source_lengths,
        drop_parameters=keep,
        drop_fault=drop,
    )
    source_only = model(
        state,
        source_waveform,
        time,
        source_parameters,
        target_parameters,
        fault,
        target_lengths=target_lengths,
        source_lengths=source_lengths,
        drop_parameters=drop,
        drop_fault=drop,
    )
    return (
        source_only
        + condition_guidance * (physical_only - source_only)
        + fault_guidance * (full - physical_only)
    )


@torch.no_grad()
def heun_transport_sample(
    model: ConditionalTransportJiT,
    initial_state: torch.Tensor,
    source_waveform: torch.Tensor,
    source_parameters: torch.Tensor,
    target_parameters: torch.Tensor,
    fault: torch.Tensor,
    *,
    steps: int,
    terminal_time: float = 0.99,
    target_lengths: torch.Tensor | None = None,
    source_lengths: torch.Tensor | None = None,
    condition_guidance: float = 1.0,
    fault_guidance: float = 1.0,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("Heun sampling requires at least two steps")
    if not 0.0 < terminal_time < 1.0:
        raise ValueError("terminal time must be in (0,1)")
    if condition_guidance < 0 or fault_guidance < 0:
        raise ValueError("guidance scales cannot be negative")
    state = initial_state
    batch_size = len(state)
    times = torch.linspace(
        0.0,
        terminal_time,
        steps + 1,
        device=state.device,
        dtype=state.dtype,
    )
    for index in range(steps):
        time = times[index].expand(batch_size)
        next_time = times[index + 1].expand(batch_size)
        step_size = times[index + 1] - times[index]
        clean = _guided_prediction(
            model,
            state,
            source_waveform,
            time,
            source_parameters,
            target_parameters,
            fault,
            target_lengths=target_lengths,
            source_lengths=source_lengths,
            condition_guidance=condition_guidance,
            fault_guidance=fault_guidance,
        )
        velocity = (clean - state) / (1.0 - time[:, None, None])
        proposal = state + step_size * velocity
        next_clean = _guided_prediction(
            model,
            proposal,
            source_waveform,
            next_time,
            source_parameters,
            target_parameters,
            fault,
            target_lengths=target_lengths,
            source_lengths=source_lengths,
            condition_guidance=condition_guidance,
            fault_guidance=fault_guidance,
        )
        next_velocity = (next_clean - proposal) / (
            1.0 - next_time[:, None, None]
        )
        state = state + 0.5 * step_size * (velocity + next_velocity)
    terminal = times[-1].expand(batch_size)
    return _guided_prediction(
        model,
        state,
        source_waveform,
        terminal,
        source_parameters,
        target_parameters,
        fault,
        target_lengths=target_lengths,
        source_lengths=source_lengths,
        condition_guidance=condition_guidance,
        fault_guidance=fault_guidance,
    )
