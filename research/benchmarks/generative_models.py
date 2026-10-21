from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils import spectral_norm

from .upstream import flow_transformer, matrix_edm, time_weaver_core


def time_embedding(time, width=128):
    frequencies = torch.exp(torch.arange(width // 2, device=time.device) * (-math.log(10000) / (width // 2 - 1)))
    angles = time.float().reshape(-1, 1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class Residual1D(nn.Module):
    def __init__(self, inputs, outputs, condition_width=128):
        super().__init__()
        self.first = nn.Sequential(nn.GroupNorm(8, inputs), nn.SiLU(), nn.Conv1d(inputs, outputs, 3, padding=1))
        self.second = nn.Sequential(nn.GroupNorm(8, outputs), nn.SiLU(), nn.Conv1d(outputs, outputs, 3, padding=1))
        self.condition = nn.Linear(condition_width, outputs * 2)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, inputs, embedding):
        hidden = self.first(inputs)
        scale, shift = self.condition(embedding).chunk(2, dim=1)
        hidden = hidden * (1 + scale[..., None]) + shift[..., None]
        return self.skip(inputs) + self.second(hidden)


class SignalAttention(nn.Module):
    def __init__(self, width, *, cross=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context = nn.Linear(128, width) if cross else None

    def forward(self, signal, context=None):
        hidden = self.norm(signal.transpose(1, 2))
        tokens = self.context(context) if self.context is not None else hidden
        return signal + self.attention(hidden, tokens, tokens, need_weights=False)[0].transpose(1, 2)


class ConditionUNet(nn.Module):
    def __init__(self, condition_dim, *, cge=False):
        super().__init__()
        self.self_condition = False
        self.cge = cge
        self.time = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.condition = nn.Sequential(nn.Linear(condition_dim + (1 if cge else 0), 128), nn.SiLU(),
                                       nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 512 if cge else 128))
        self.initial = nn.Conv1d(1, 32, 3, padding=1)
        self.down = nn.ModuleList([Residual1D(32, 64), Residual1D(64, 128), Residual1D(128, 256)])
        self.pools = nn.ModuleList([nn.Conv1d(width, width, 4, stride=2, padding=1) for width in (64, 128, 256)])
        self.encode_attention = nn.ModuleList([SignalAttention(width) for width in (64, 128, 256)])
        self.middle = Residual1D(256, 256)
        self.up = nn.ModuleList([Residual1D(512, 128), Residual1D(256, 64), Residual1D(128, 32)])
        self.decode_attention = nn.ModuleList([SignalAttention(width, cross=True) for width in (128, 64, 32)])
        self.final = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv1d(32, 1, 3, padding=1))

    def forward(self, signal, time, condition):
        time_vector = time.reshape(-1)
        embedding = self.time(time_embedding(time_vector))
        if self.cge:
            context = self.condition(torch.cat((condition, time_vector[:, None] / 1200), dim=1)).reshape(-1, 4, 128)
        else:
            context = self.condition(condition)[:, None]
            embedding = embedding + context[:, 0]
        hidden = self.initial(signal)
        skips = []
        for residual, pool, attention in zip(self.down, self.pools, self.encode_attention):
            hidden = residual(hidden, embedding)
            skips.append(hidden)
            hidden = attention(pool(hidden))
        hidden = self.middle(hidden, embedding)
        for residual, attention, skip in zip(self.up, self.decode_attention, reversed(skips)):
            hidden = functional.interpolate(hidden, size=skip.shape[-1], mode="nearest")
            hidden = residual(torch.cat((hidden, skip), dim=1), embedding)
            hidden = attention(hidden, context)
        return self.final(hidden)


class DiffusionModel(nn.Module):
    def __init__(self, condition_dim, method):
        super().__init__()
        self.method = method
        self.steps = 1200 if method == "diffucd" else 50
        if method == "diffucd":
            self.network = ConditionUNet(condition_dim, cge=True)
            beta = torch.linspace(0.001, 0.05, self.steps)
        else:
            self.network = time_weaver_core()({"channels": 64, "num_steps": self.steps,
                "diffusion_embedding_dim": 128, "side": {"num_var": 1, "var_emb": 16, "time_emb": 112},
                "device": "cuda", "base_patch": 4, "layers": 3, "nheads": 8, "is_linear": False}, inputdim=1)
            self.condition = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, 64))
            beta = torch.linspace(math.sqrt(0.0001), math.sqrt(0.5), self.steps).square()
        self.register_buffer("alpha", (1 - beta).cumprod(dim=0))

    def forward(self, signal, time, condition):
        if self.method == "diffucd":
            return self.network(signal, time, condition)
        positions = torch.arange(signal.shape[-1], device=signal.device).float()[None].expand(len(signal), -1)
        attributes = self.condition(condition)[:, None, None]
        return self.network(signal[:, None], positions, attributes, time.long())[0]

    def loss(self, signal, condition, weights=None):
        time = torch.randint(self.steps, (len(signal),), device=signal.device)
        noise = torch.randn_like(signal)
        alpha = self.alpha[time, None, None]
        prediction = self(alpha.sqrt() * signal + (1 - alpha).sqrt() * noise, time, condition)
        return (prediction - noise).square().mean()

    @torch.no_grad()
    def sample(self, condition, steps=3, length=1024):
        signal = torch.randn(len(condition), 1, length, device=condition.device)
        schedule = torch.linspace(self.steps - 1, 0, steps, device=condition.device).long()
        for index, time in enumerate(schedule):
            alpha = self.alpha[time]
            prediction = self(signal, time.expand(len(signal)), condition)
            estimate = ((signal - (1 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-1, 1)
            if index + 1 == len(schedule):
                signal = estimate
            else:
                next_alpha = self.alpha[schedule[index + 1]]
                signal = next_alpha.sqrt() * estimate + (1 - next_alpha).sqrt() * prediction
        return signal


class FlowModel(nn.Module):
    def __init__(self, condition_dim, length=1024):
        super().__init__()
        self.network = flow_transformer()(n_feat=1, n_channel=length, n_layer_enc=3, n_layer_dec=6,
                                         n_embd=64, n_heads=4, max_len=length, attn_pdrop=0.0, resid_pdrop=0.0)
        self.condition = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, signal, time, condition):
        network = self.network
        embedded = network.emb(signal.transpose(1, 2)) + self.condition(condition)[:, None]
        encoded = network.encoder(embedded, time)
        decoded, means, trend, season = network.decoder(embedded, time, encoded)
        residual = network.inverse(decoded)
        residual_mean = residual.mean(dim=1, keepdim=True)
        seasonal = network.combine_s(season.transpose(1, 2)).transpose(1, 2) + residual - residual_mean
        return (network.combine_m(means) + residual_mean + trend + seasonal).transpose(1, 2)

    def loss(self, signal, condition, weights=None):
        noise = torch.randn_like(signal)
        time = torch.sigmoid(torch.randn(len(signal), 1, 1, device=signal.device))
        prediction = self(time * signal + (1 - time) * noise, time[:, 0, 0] * 1000, condition)
        return functional.mse_loss(prediction, signal - noise)

    @torch.no_grad()
    def sample(self, condition, steps=3, length=1024):
        signal = torch.randn(len(condition), 1, length, device=condition.device)
        uniform = torch.linspace(0, 1, steps + 1, device=condition.device)
        schedule = (1 - 3 * uniform / (1 + 2 * uniform)).flip(0)
        for current, following in zip(schedule[:-1], schedule[1:]):
            signal = signal + (following - current) * self(signal, (current * 1000).expand(len(signal)), condition)
        return signal


class ConditionalNorm(nn.Module):
    def __init__(self, width, embedding_dim):
        super().__init__()
        self.normalization = nn.BatchNorm1d(width, affine=False)
        self.gamma = nn.Linear(embedding_dim, width, bias=False)
        self.beta = nn.Linear(embedding_dim, width, bias=False)

    def forward(self, hidden, embedding):
        normalized = self.normalization(hidden)
        return normalized * (1 + self.gamma(embedding)[..., None]) + self.beta(embedding)[..., None]


class GANUp(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.norm1, self.norm2 = ConditionalNorm(inputs, 128), ConditionalNorm(outputs, 128)
        self.conv1, self.conv2 = nn.Conv1d(inputs, outputs, 3, padding=1), nn.Conv1d(outputs, outputs, 3, padding=1)
        self.skip = nn.Conv1d(inputs, outputs, 1)

    def forward(self, hidden, embedding):
        enlarged = functional.interpolate(hidden, scale_factor=2, mode="nearest")
        result = functional.interpolate(functional.relu(self.norm1(hidden, embedding)), scale_factor=2, mode="nearest")
        result = self.conv1(result)
        return self.conv2(functional.relu(self.norm2(result, embedding))) + self.skip(enlarged)


class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(128, 512 * 64)
        self.blocks = nn.ModuleList([GANUp(512, 256), GANUp(256, 128), GANUp(128, 64), GANUp(64, 32)])
        self.final = nn.Sequential(nn.BatchNorm1d(32), nn.ReLU(), nn.Conv1d(32, 1, 3, padding=1), nn.Tanh())

    def forward(self, noise, embedding):
        hidden = self.dense(noise).reshape(-1, 512, 64)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.final(hidden)


class GANDown(nn.Module):
    def __init__(self, inputs, outputs, *, first=False):
        super().__init__()
        self.first = first
        self.conv1 = spectral_norm(nn.Conv1d(inputs, outputs, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(outputs, outputs, 3, padding=1))
        self.skip = spectral_norm(nn.Conv1d(inputs, outputs, 1))

    def forward(self, hidden):
        residual = self.conv1(hidden if self.first else functional.relu(hidden))
        residual = self.conv2(functional.relu(residual))
        return functional.avg_pool1d(residual + self.skip(hidden), 2)


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(GANDown(1, 32, first=True), GANDown(32, 64), GANDown(64, 128), GANDown(128, 256))
        self.unconditional = spectral_norm(nn.Linear(256, 1))
        self.projection = spectral_norm(nn.Linear(128, 256, bias=False))

    def forward(self, signal, embedding):
        features = functional.relu(self.blocks(signal)).sum(dim=-1)
        return self.unconditional(features)[:, 0] + (self.projection(embedding) * features).sum(dim=-1)


class SignalRegressor(nn.Module):
    def __init__(self, condition_dim, hidden=128):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                         GANDown(32, 64), GANDown(64, 128), GANDown(128, 256), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                         nn.Linear(256, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, condition_dim)

    def forward(self, signal):
        return self.head(self.features(signal))


class CcGAN(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(condition_dim, 128), nn.GroupNorm(8, 128), nn.ReLU(),
                                        nn.Linear(128, 128), nn.GroupNorm(8, 128), nn.ReLU(), nn.Linear(128, 128))
        self.regressor = SignalRegressor(condition_dim)
        self.generator = GANGenerator()
        self.discriminator = GANDiscriminator()

    def forward(self, noise, condition):
        return self.generator(noise, self.embedding(condition))

    @torch.no_grad()
    def sample(self, condition, steps=3, length=1024):
        if length != 1024:
            raise ValueError("The CcGAN comparator has a fixed native length of 1024")
        return self(torch.randn(len(condition), 128, device=condition.device), condition)


class EDMNetwork(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.self_condition = False
        self.network = ConditionUNet(condition_dim)

    def forward(self, x, time, labels_emb, x_self_cond=None, keep_mask=None):
        condition = labels_emb if keep_mask is None else labels_emb * keep_mask[:, None]
        return self.network(x.flatten(start_dim=2), time, condition).reshape_as(x)

    def forward_with_cond_scale(self, x, time, labels_emb, x_self_cond=None, cond_scale=1.5, rescaled_phi=0.7):
        conditioned = self.forward(x, time, labels_emb)
        unconditioned = self.forward(x, time, torch.zeros_like(labels_emb))
        guided = unconditioned + cond_scale * (conditioned - unconditioned)
        deviation = conditioned.std(dim=(1, 2, 3), keepdim=True)
        corrected = guided * deviation / guided.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        return rescaled_phi * corrected + (1 - rescaled_phi) * guided


class ImprovedCCDM(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        self.regressor = SignalRegressor(condition_dim)
        self.cov_regressor = SignalRegressor(condition_dim, hidden=1024)
        self.covariance = nn.Sequential(nn.Linear(condition_dim, 256), nn.GroupNorm(8, 256), nn.ReLU(),
                                        nn.Linear(256, 1024), nn.ReLU())
        self.core = matrix_edm()(EDMNetwork(128), self.sigma_data, {"use_aux_reg_loss": False},
                    image_size=32, channels=1, num_sample_steps=32, use_y2cov=True, fn_y2cov=self.covariance,
                    y2cov_hy_weight_train=0.001, y2cov_hy_weight_test=0.001)

    @staticmethod
    def sigma_data(labels):
        return torch.full((len(labels),), 0.5, device=labels.device)

    def forward(self, signal, time, condition):
        carrier = signal.reshape(-1, 1, 32, 32)
        return self.core.preconditioned_network_forward(carrier, time, condition, self.embedding(condition)).reshape_as(signal)

    def loss(self, signal, condition, weights=None):
        carrier = (signal.reshape(-1, 1, 32, 32) + 1) / 2
        loss, auxiliary, factor = self.core(carrier, condition, self.embedding(condition), vicinal_weights=weights)
        return loss + (auxiliary * factor).mean()

    @torch.no_grad()
    def sample(self, condition, steps=3, length=1024):
        if length != 1024:
            raise ValueError("The iCCDM comparator has a fixed native length of 1024")
        sample = self.core.sample_using_ode(condition, self.embedding(condition), num_sample_steps=steps)
        return sample.reshape(-1, 1, 1024) * 2 - 1


def build_model(method, condition_dim):
    if method in ("diffucd", "time_weaver"):
        return DiffusionModel(condition_dim, method)
    return {"ccgan": CcGAN, "iccdm": ImprovedCCDM, "flowts": FlowModel}[method](condition_dim)
