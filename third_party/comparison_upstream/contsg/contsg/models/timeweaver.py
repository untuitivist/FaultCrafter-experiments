"""
TimeWeaver: Heterogeneous Attribute-Conditioned Time Series Generation.

This module implements the TimeWeaver model for the ConTSG framework.
TimeWeaver is an attribute-conditioned diffusion model that handles both
continuous and discrete attributes through heterogeneous conditioning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torch import Tensor

from contsg.config.schema import ExperimentConfig
from contsg.models.base import BaseGeneratorModule, DiffusionMixin
from contsg.registry import Registry


def get_torch_trans(heads=8, layers=1, channels=64):
    """Create standard PyTorch transformer encoder."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    """Create linear attention transformer."""
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Conv1d with Kaiming initialization."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv1d_init(indim, outdim, k, s, p):
    """Conv1d with custom parameters and Kaiming initialization."""
    layer = nn.Conv1d(indim, outdim, kernel_size=k, stride=s, padding=p)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def TConv1d_init(indim, outdim, k, s, p, op):
    """Transposed Conv1d with Kaiming initialization."""
    layer = nn.ConvTranspose1d(indim, outdim, kernel_size=k, stride=s, padding=p, output_padding=op)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""

    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SideEncoder(nn.Module):
    """
    Side information encoder for variable and time embeddings.
    Combines variable-specific embeddings with time positional encodings.
    """

    def __init__(self, num_var, var_emb_dim=128, time_emb_dim=128, device="cuda"):
        super().__init__()
        self.num_var = num_var
        self.var_emb_dim = var_emb_dim
        self.time_emb_dim = time_emb_dim
        self.device = device

        self.var_emb = nn.Embedding(num_embeddings=self.num_var, embedding_dim=var_emb_dim)
        self.register_buffer("var_ids", torch.arange(self.num_var))
        self.total_emb_dim = var_emb_dim + time_emb_dim

    def time_embedding(self, pos, d_model=128):
        """Sinusoidal time positional encoding."""
        device = pos.device
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tp):
        """
        Args:
            tp: Time positions (B, L)

        Returns:
            side_emb: Combined embeddings (B, D_total, V, L)
        """
        B, L = tp.shape
        # Time embedding: (B, L, D_t)
        time_emb = self.time_embedding(tp, self.time_emb_dim)
        # Expand to (B, L, V, D_t)
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, self.num_var, -1)

        # Variable embedding: (V, D_v)
        var_emb = self.var_emb(self.var_ids)
        # Expand to (B, L, V, D_v)
        var_emb = var_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # Concatenate and permute to (B, D_total, V, L)
        side_emb = torch.cat([time_emb, var_emb], dim=-1)
        side_emb = side_emb.permute(0, 3, 2, 1)
        return side_emb


class AttributeEncoder(nn.Module):
    """
    Heterogeneous attribute encoder supporting multiple discrete attributes.
    Each attribute can have a different number of options.
    """

    def __init__(self, num_attr_ops, emb_dim, device="cuda"):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.n_attr = len(num_attr_ops)
        self.n_ops_list = num_attr_ops

        # Initialize embeddings with shifted indices
        self.attr_emb = self._init_embs(num_attr_ops)
        self.register_buffer(
            "unknown_index",
            torch.tensor(self.n_ops_list, dtype=torch.long) - 1,
        )

        # Empty token embedding
        self.empty_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.emb_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

    def _init_embs(self, n_ops):
        """Initialize embeddings with cumulative shift for heterogeneous attributes."""
        import numpy as np
        shift = np.cumsum(n_ops)
        shift = np.insert(shift, 0, 0)
        emb = nn.Embedding(num_embeddings=shift[-1], embedding_dim=self.emb_dim)
        shift_tensor = torch.from_numpy(shift[:-1]).unsqueeze(0)
        self.register_buffer("attr_shift", shift_tensor)
        return emb

    def forward(self, attrs, replace_with_empty=False):
        """
        Args:
            attrs: Attribute indices (B, K)
            replace_with_empty: Whether to use empty token embeddings

        Returns:
            emb: Attribute embeddings (B, K, D)
        """
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(1)
        attrs = attrs.long()
        if (attrs < 0).any():
            attrs = attrs.clone()
            unknown_mask = attrs < 0
            unknown_index = self.unknown_index.to(attrs.device).unsqueeze(0).expand_as(attrs)
            attrs[unknown_mask] = unknown_index[unknown_mask]
        if replace_with_empty:
            idx = torch.zeros(attrs.shape, dtype=torch.long, device=attrs.device)
            emb = self.empty_emb(idx)
        else:
            emb = self.attr_emb(attrs + self.attr_shift)
        emb = self.out_proj(emb)
        return emb


class AttrProjector(nn.Module):
    """
    Projects attribute embeddings to model space.
    Uses average pooling followed by projection.
    """

    def __init__(self, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj_out = nn.Linear(dim_in, dim_out)

    def forward(self, attr):
        """
        Args:
            attr: Attribute embeddings (B, K, D)

        Returns:
            out: Projected embeddings (B, 1, 1, D)
        """
        B = attr.shape[0]
        h = torch.mean(attr, dim=1, keepdim=True)  # (B, 1, D)
        h = h[:, None, :, :]  # (B, 1, 1, D)
        out = self.proj_out(h)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with time and feature attention.
    Processes side information (time + variable embeddings) and diffusion timestep.
    """

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """Apply temporal attention."""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """Apply feature/variable attention."""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_emb, diffusion_emb):
        """
        Args:
            x: Input tensor (B, C, K, L)
            side_emb: Side information (B, D_side, K, L)
            diffusion_emb: Diffusion timestep embedding (B, D_diff)

        Returns:
            residual: Output tensor (B, C, K, L)
            skip: Skip connection (B, C, K, L)
        """
        B, channel, K, L = x.shape
        base_shape = x.shape

        # Add diffusion embedding
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb

        # Apply temporal and feature attention
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        # Process with side information
        y = y.reshape(base_shape)
        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        # Gated activation
        gate, filter_val = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_val)
        y = self.output_projection(y)

        # Split into residual and skip
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class TimeWeaverCore(nn.Module):
    """Core TimeWeaver architecture."""

    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        # Diffusion timestep embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # Side encoder (time + variable information)
        self.side_encoder = SideEncoder(
            num_var=config["side"]["num_var"],
            var_emb_dim=config["side"]["var_emb"],
            time_emb_dim=config["side"]["time_emb"],
            device=config["device"],
        )
        side_dim = self.side_encoder.total_emb_dim

        # Input/output projections with patching
        self.input_projection = Conv1d_init(
            inputdim, self.channels,
            config["base_patch"], config["base_patch"], config["base_patch"] // 2
        )
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = TConv1d_init(
            self.channels, 1, config["base_patch"], config["base_patch"], 0, 0
        )
        nn.init.zeros_(self.output_projection2.weight)

        # Residual blocks
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, tp, attr_emb, diffusion_step):
        """
        Args:
            x: Noisy input (B, C, K, L)
            tp: Time positions (B, L)
            attr_emb: Attribute embeddings (B, 1, 1, D) or None
            diffusion_step: Timestep indices (B,)

        Returns:
            pred: Predicted noise (B, K, L)
            extra: Additional outputs (empty dict)
        """
        B, inputdim, K, L = x.shape

        # Patch embedding
        x = x.permute(0, 2, 1, 3).reshape(B * K, inputdim, L)
        x = self.input_projection(x)
        x = F.relu(x)
        Nl = x.shape[2]
        x = x.reshape(B, K, -1, Nl).permute(0, 2, 1, 3)

        # Side information encoding
        side_emb = self.side_encoder(tp)
        side_emb = side_emb[:, :, :, :Nl]

        # Handle attribute conditioning
        if attr_emb is None:
            attr_emb = torch.zeros_like(x)
        else:
            attr_emb = attr_emb.permute(0, 3, 1, 2)

        # Diffusion timestep embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # Process through residual blocks
        x_in = x
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x + attr_emb + x_in, side_emb, diffusion_emb)
            skip.append(skip_connection)

        # Aggregate skip connections
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # Output projection
        x = x.reshape(B, self.channels, K * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)

        x = x.reshape(B, self.channels, K, Nl).permute(0, 2, 1, 3).reshape(B * K, self.channels, Nl)
        x = self.output_projection2(x)
        x = x.reshape(B, K, -1)
        x = x[:, :, :L]

        return x, {}


@Registry.register_model("timeweaver", aliases=["tw"])
class TimeWeaverModule(BaseGeneratorModule, DiffusionMixin):
    """TimeWeaver: Heterogeneous attribute-conditioned time series generation.

    TimeWeaver is an attribute-conditioned diffusion model that handles both
    continuous and discrete attributes through a heterogeneous conditioning mechanism.
    It uses multi-patch processing with temporal and feature attention.

    Key Features:
    - Attribute-conditioned generation (discrete and continuous attributes)
    - Multi-patch processing for handling different scales
    - Temporal and feature attention mechanisms
    - Side information encoding (time + variable embeddings)

    Training:
    - Single-stage training (finetune only)
    - DDPM/DDIM sampling

    Reference:
        TimeWeaver: Heterogeneous Attribute-Conditioned Time Series Generation
    """

    SUPPORTED_STAGES = ["finetune"]

    def _build_model(self) -> None:
        """Build TimeWeaver model architecture."""
        cfg = self.config.model
        data_cfg = self.config.data
        cond_cfg = self.config.condition

        # Model parameters
        self.n_var = data_cfg.n_var
        self.seq_length = data_cfg.seq_length
        self.channels = getattr(cfg, "channels", 64)
        self.layers = getattr(cfg, "layers", 3)
        self.nheads = getattr(cfg, "nheads", 8)
        self.diffusion_steps = getattr(cfg, "diffusion_steps", None)
        if self.diffusion_steps is None:
            self.diffusion_steps = getattr(cfg, "num_steps", 1000)
        self.base_patch = getattr(cfg, "base_patch", 4)
        self.is_linear = getattr(cfg, "is_linear", False)

        # Attribute conditioning configuration
        if not cond_cfg.attribute.enabled:
            raise ValueError("TimeWeaver requires attribute conditioning to be enabled")

        self.attr_emb_dim = getattr(cfg, "attr_dim", cond_cfg.attribute.output_dim)
        self.num_attr_ops = [
            disc["num_classes"] for disc in cond_cfg.attribute.discrete_configs
        ]

        # Build attribute encoder and projector
        self.attr_encoder = AttributeEncoder(
            num_attr_ops=self.num_attr_ops,
            emb_dim=self.attr_emb_dim,
            device=self.config.device,
        )
        self.attr_projector = AttrProjector(
            dim_in=self.attr_emb_dim,
            dim_out=self.channels,
        )

        # Build core TimeWeaver model
        model_config = {
            "channels": self.channels,
            "num_steps": self.diffusion_steps,
            "diffusion_embedding_dim": getattr(cfg, "diffusion_embedding_dim", 128),
            "base_patch": self.base_patch,
            "layers": self.layers,
            "nheads": self.nheads,
            "is_linear": self.is_linear,
            "device": self.config.device,
            "side": {
                "num_var": self.n_var,
                "var_emb": getattr(cfg, "var_emb", 16),
                "time_emb": getattr(cfg, "time_emb", 112),
            },
        }
        self.model = TimeWeaverCore(model_config, inputdim=1)

        # Diffusion parameters
        self.num_steps = self.diffusion_steps
        noise_schedule = getattr(cfg, "noise_schedule", None)
        if noise_schedule is None:
            noise_schedule = getattr(cfg, "schedule", "cosine")
        beta_start = getattr(cfg, "beta_start", None)
        beta_end = getattr(cfg, "beta_end", None)

        if noise_schedule == "quad":
            beta_start = beta_start if beta_start is not None else 0.0001
            beta_end = beta_end if beta_end is not None else 0.5
            betas = self.quad_beta_schedule(self.num_steps, beta_start, beta_end)
        elif noise_schedule == "linear":
            if beta_start is None and beta_end is None:
                betas = self.linear_beta_schedule(self.num_steps)
            else:
                beta_start = beta_start if beta_start is not None else 0.0001
                beta_end = beta_end if beta_end is not None else 0.02
                betas = torch.linspace(beta_start, beta_end, self.num_steps)
        elif noise_schedule == "cosine":
            betas = self.cosine_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_sqrt = torch.sqrt(alpha_bar)
        one_minus_alpha_bar_sqrt = torch.sqrt(1.0 - alpha_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar_sqrt)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", one_minus_alpha_bar_sqrt)
        self.register_buffer("alpha_bar_sqrt", alpha_bar_sqrt)
        self.register_buffer("alpha_bar_sqrt_inverse", 1.0 / alpha_bar_sqrt)
        self.register_buffer("one_minus_alpha_bar_sqrt", one_minus_alpha_bar_sqrt)
        self.register_buffer("reverse_coef1", 1.0 / torch.sqrt(alphas))
        self.register_buffer("reverse_coef2", (1.0 - alphas) / one_minus_alpha_bar_sqrt)

        if self.num_steps > 1:
            numerator = 1.0 - alpha_bar[:-1]
            denominator = 1.0 - alpha_bar[1:]
            sigma_sq = numerator / denominator * betas[1:]
            sigma = torch.sqrt(sigma_sq)
            reverse_coef2_determin = torch.sqrt(1.0 - alpha_bar[:-1])
            reverse_coef2_ddim = torch.sqrt(1.0 - alpha_bar[:-1] - sigma_sq)
            alpha_bar_sqrt_prev = alpha_bar_sqrt[:-1]
        else:
            sigma = torch.zeros(1, device=betas.device)
            reverse_coef2_determin = torch.zeros(1, device=betas.device)
            reverse_coef2_ddim = torch.zeros(1, device=betas.device)
            alpha_bar_sqrt_prev = torch.zeros(1, device=betas.device)

        self.register_buffer("sigma", sigma)
        self.register_buffer("reverse_coef2_determin", reverse_coef2_determin)
        self.register_buffer("reverse_coef2_ddim", reverse_coef2_ddim)
        self.register_buffer("alpha_bar_sqrt_prev", alpha_bar_sqrt_prev)

    def _encode_attributes(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        """
        Encode attributes from batch.

        Args:
            batch: Batch dictionary containing "attrs" key

        Returns:
            Attribute embeddings (B, 1, 1, D) or None if not using condition
        """
        if not self.use_condition or "attrs" not in batch:
            return None

        attrs = batch["attrs"]  # (B, K)
        attr_emb = self.attr_encoder(attrs)  # (B, K, D)
        attr_emb = self.attr_projector(attr_emb)  # (B, 1, 1, D)
        return attr_emb

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for training.

        Args:
            batch: Dictionary containing:
                - "ts": Time series (B, L, C)
                - "tp": Time positions (B, L)
                - "attrs": Attribute indices (B, K)

        Returns:
            Dictionary with "loss" and optional metrics
        """
        ts = batch["ts"].float()  # (B, L, C)
        tp = batch.get("tp")  # (B, L)
        B, L, C = ts.shape

        # Generate time positions if not provided
        if tp is None:
            tp = torch.arange(L, device=ts.device).unsqueeze(0).repeat(B, 1).float()

        # Encode attributes
        attr_emb = self._encode_attributes(batch)

        # Sample timestep
        t = torch.randint(0, self.num_steps, (B,), device=ts.device)

        # Add noise (forward diffusion)
        ts = ts.permute(0, 2, 1)  # (B, V, L)
        noisy_ts, noise = self.q_sample(
            ts, t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        )

        # Reshape for model: (B, 1, V, L)
        noisy_ts = noisy_ts.unsqueeze(1)

        # Predict noise
        pred_noise, _ = self.model(noisy_ts, tp, attr_emb, t)

        # Compute loss
        loss = F.mse_loss(pred_noise, noise)

        return {"loss": loss, "mse_loss": loss}

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        n_samples: int = 1,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate time series samples given attribute conditions.

        Args:
            condition: Attribute tensor (B, K) - indices for discrete attributes
            n_samples: Number of samples to generate per condition
            **kwargs: Additional parameters:
                - sampler: "ddpm" or "ddim" (default: "ddim")
                - eta: DDIM stochasticity parameter (default: 0.0)

        Returns:
            Generated time series (B, n_samples, L, C)
        """
        sampler = kwargs.get("sampler", "ddim")
        eta = kwargs.get("eta", 0.0)

        B = condition.shape[0]
        device = condition.device

        # Encode attributes
        attr_emb = self.attr_encoder(condition)  # (B, K, D)
        attr_emb = self.attr_projector(attr_emb)  # (B, 1, 1, D)

        # Expand for n_samples
        attr_emb = attr_emb.repeat_interleave(n_samples, dim=0)  # (B*n_samples, 1, 1, D)

        # Initialize from noise
        x = torch.randn(B * n_samples, 1, self.n_var, self.seq_length, device=device)

        # Time positions
        tp = kwargs.get("tp")
        if tp is None:
            tp = torch.arange(self.seq_length, device=device).unsqueeze(0).repeat(B * n_samples, 1).float()
        else:
            if tp.shape[0] != B * n_samples:
                tp = tp.repeat_interleave(n_samples, dim=0)

        # DDPM/DDIM reverse process
        B_total = x.shape[0]
        for t_int in reversed(range(self.num_steps)):
            t = torch.full((B_total,), t_int, device=device, dtype=torch.long)
            pred_noise, _ = self.model(x, tp, attr_emb, t)
            pred_noise = pred_noise.unsqueeze(1)  # (B, 1, V, L)
            if sampler == "ddpm":
                x = self._ddpm_reverse(x, pred_noise, t)
            else:
                x = self._ddim_reverse(x, pred_noise, t, eta=eta)

        # Reshape output: (B*n_samples, 1, V, L) -> (B, n_samples, L, V)
        x = x.squeeze(1).permute(0, 2, 1)  # (B*n_samples, L, V)
        x = x.view(B, n_samples, self.seq_length, self.n_var)

        return x

    def _ddpm_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor) -> Tensor:
        """DDPM reverse step."""
        coef1 = self.reverse_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.reverse_coef2[t].view(-1, 1, 1, 1)
        x_prev = coef1 * (x_t - coef2 * pred_noise)

        noise = torch.randn_like(x_t)
        t_prev = (t - 1).clamp(min=0)
        sigma = self.sigma[t_prev].view(-1, 1, 1, 1)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = x_prev + mask * sigma * noise
        return x_prev

    def _ddim_reverse(self, x_t: Tensor, pred_noise: Tensor, t: Tensor, eta: float = 0.0) -> Tensor:
        """DDIM reverse step (deterministic when eta=0)."""
        coef1 = self.one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
        coef2 = self.alpha_bar_sqrt_inverse[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - coef1 * pred_noise) * coef2

        t_prev = (t - 1).clamp(min=0)
        coef_prev = self.alpha_bar_sqrt_prev[t_prev].view(-1, 1, 1, 1)

        if eta == 0.0:
            coef2_prev = self.reverse_coef2_determin[t_prev].view(-1, 1, 1, 1)
            coef3 = 0.0
        else:
            coef2_prev = self.reverse_coef2_ddim[t_prev].view(-1, 1, 1, 1)
            coef3 = self.sigma[t_prev].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t)
        x_prev = coef_prev * x0_pred + coef2_prev * pred_noise + coef3 * noise

        mask = (t == 0).view(-1, 1, 1, 1)
        x_prev = torch.where(mask, x0_pred, x_prev)
        return x_prev
