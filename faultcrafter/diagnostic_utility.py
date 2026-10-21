from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as functional


@dataclass(frozen=True)
class UtilitySettings:
    interval: int = 50
    start_after: int = 1000
    weight: float = 10.0
    margin: float = 0.01
    warmup_steps: int = 100
    warmup_learning_rate: float = 0.001
    inner_steps: int = 5
    inner_learning_rate: float = 0.05
    batch_size: int = 64
    window_length: int = 1024
    stride: int = 819
    parent_length: int = 51802
    architectures: tuple[str, ...] = ("temporal", "spectral")

    def __post_init__(self):
        for name in ("interval", "warmup_steps", "inner_steps", "batch_size", "window_length", "stride", "parent_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"utility {name} must be a positive integer")
        if self.start_after < 0 or self.weight <= 0 or self.margin < 0:
            raise ValueError("invalid utility schedule or loss weights")
        if self.warmup_learning_rate <= 0 or self.inner_learning_rate <= 0:
            raise ValueError("utility learning rates must be positive")
        if self.parent_length < self.window_length or self.architectures != ("temporal", "spectral"):
            raise ValueError("utility requires complete parents and its two declared architectures")


class UtilityClassifier(nn.Module):
    def __init__(self, num_classes: int, architecture: str):
        super().__init__()
        if num_classes < 2 or architecture not in ("temporal", "spectral"):
            raise ValueError("invalid auxiliary diagnostic classifier")
        self.architecture = architecture
        if architecture == "temporal":
            channels, kernels, strides = (16, 32, 64), (17, 9, 5), (8, 4, 4)
        else:
            channels, kernels, strides = (12, 24, 48), (9, 5, 3), (2, 4, 4)
        layers = []
        previous = 1
        for width, kernel, stride in zip(channels, kernels, strides, strict=True):
            layers.extend((nn.Conv1d(previous, width, kernel, stride=stride, padding=kernel // 2),
                           nn.GroupNorm(4, width), nn.SiLU()))
            previous = width
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(previous, num_classes)

    def forward(self, signal):
        if self.architecture == "spectral":
            spectrum = torch.fft.rfft(signal, dim=-1)
            signal = torch.log1p((spectrum.real.square() + spectrum.imag.square() + 1e-8).sqrt())
        return self.output(self.features(signal).mean(dim=-1))


def parent_windows(parents: torch.Tensor, window_length: int, stride: int) -> torch.Tensor:
    if parents.ndim != 3 or parents.shape[1] != 1 or parents.shape[-1] < window_length or stride < 1:
        raise ValueError("invalid diagnostic parent shape or window layout")
    return parents.unfold(-1, window_length, stride).transpose(1, 2).reshape(-1, 1, window_length)


def brier_risk(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = functional.one_hot(labels, logits.shape[1]).to(logits.dtype)
    return 0.5 * (logits.softmax(dim=-1) - targets).square().sum(dim=-1).mean()


def parameter_risk(model, parameters, signals, labels, batch_size):
    totals = []
    for begin in range(0, len(signals), batch_size):
        end = min(begin + batch_size, len(signals))
        logits = functional_call(model, parameters, (signals[begin:end],), strict=True)
        totals.append(brier_risk(logits, labels[begin:end]) * (end - begin) / len(signals))
    return torch.stack(totals).sum()


def paired_query_risks(model, support, support_labels, generated, generated_labels, query, query_labels,
                       settings: UtilitySettings, *, seed: int):
    if len(support) != len(support_labels) or len(generated) != len(generated_labels) or len(query) != len(query_labels):
        raise ValueError("diagnostic signals and labels differ")
    if min(len(support), len(generated), len(query)) == 0 or any(True for _ in model.buffers()):
        raise ValueError("diagnostic pairs require nonempty sets and a stateless classifier")
    initial = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    real_parameters = {name: value.detach().clone().requires_grad_() for name, value in initial.items()}
    augmented_parameters = initial
    random = torch.Generator(device=support.device).manual_seed(seed)
    real_fraction = len(support) / (len(support) + len(generated))
    for _ in range(settings.inner_steps):
        source_indices = torch.randint(len(support), (settings.batch_size,), generator=random, device=support.device)
        generated_indices = torch.randint(len(generated), (settings.batch_size,), generator=random, device=support.device)
        source_batch, source_labels = support[source_indices], support_labels[source_indices]
        real_loss = functional.cross_entropy(functional_call(model, real_parameters, (source_batch,), strict=True), source_labels)
        real_gradients = torch.autograd.grad(real_loss, tuple(real_parameters.values()))
        real_parameters = {name: (value - settings.inner_learning_rate * gradient).detach().requires_grad_()
                           for (name, value), gradient in zip(real_parameters.items(), real_gradients, strict=True)}
        source_loss = functional.cross_entropy(functional_call(model, augmented_parameters, (source_batch,), strict=True), source_labels)
        generated_loss = functional.cross_entropy(
            functional_call(model, augmented_parameters, (generated[generated_indices],), strict=True), generated_labels[generated_indices])
        augmented_loss = real_fraction * source_loss + (1.0 - real_fraction) * generated_loss
        gradients = torch.autograd.grad(augmented_loss, tuple(augmented_parameters.values()), create_graph=True)
        augmented_parameters = {name: value - settings.inner_learning_rate * gradient
                                for (name, value), gradient in zip(augmented_parameters.items(), gradients, strict=True)}
    with torch.no_grad():
        real_risk = parameter_risk(model, real_parameters, query, query_labels, settings.batch_size)
    augmented_risk = parameter_risk(model, augmented_parameters, query, query_labels, settings.batch_size)
    return real_risk, augmented_risk


def warm_start_classifier(model, support, labels, settings: UtilitySettings, *, seed: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.warmup_learning_rate)
    random = torch.Generator(device=support.device).manual_seed(seed)
    model.train()
    for _ in range(settings.warmup_steps):
        indices = torch.randint(len(support), (settings.batch_size,), generator=random, device=support.device)
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(model(support[indices]), labels[indices])
        loss.backward()
        optimizer.step()
    model.zero_grad(set_to_none=True)
    model.eval()
    return float(loss.detach())


def utility_signal_gradient(parents, support, support_labels, query, query_labels, num_classes,
                            settings: UtilitySettings, *, seed: int):
    windows_per_parent = 1 + (parents.shape[-1] - settings.window_length) // settings.stride
    if parents.shape != (num_classes, 1, settings.parent_length):
        raise ValueError("utility must include one complete parent for every fault class")
    labels = torch.arange(num_classes, device=parents.device).repeat_interleave(windows_per_parent)
    best_gradient = None
    best_value = -float("inf")
    best_architecture = None
    reports = []
    for index, architecture in enumerate(settings.architectures):
        classifier_seed = seed + 1009 * index
        cuda_devices = [parents.device.index] if parents.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(classifier_seed)
            classifier = UtilityClassifier(num_classes, architecture).to(device=parents.device, dtype=parents.dtype)
            warmup_loss = warm_start_classifier(classifier, support, support_labels, settings, seed=classifier_seed + 1)
            leaf = parents.detach().requires_grad_()
            generated = parent_windows(leaf, settings.window_length, settings.stride)
            real_risk, augmented_risk = paired_query_risks(
                classifier, support, support_labels, generated, labels, query, query_labels, settings, seed=classifier_seed + 2)
            difference = augmented_risk - real_risk
            penalty = settings.weight * functional.relu(difference + settings.margin)
            gradient, = torch.autograd.grad(penalty, leaf)
            if not torch.isfinite(penalty) or not torch.isfinite(gradient).all():
                raise FloatingPointError("diagnostic risk or signal adjoint is nonfinite")
            value = float(penalty.detach())
            reports.append({"architecture": architecture, "initialization_seed": classifier_seed,
                            "support_warmup_loss": warmup_loss, "real_query_brier": float(real_risk),
                            "augmented_query_brier": float(augmented_risk.detach()), "risk_difference": float(difference.detach()),
                            "weighted_penalty": value, "signal_gradient_norm": float(gradient.norm())})
            if value > best_value:
                best_value = value
                best_gradient = gradient.detach()
                best_architecture = architecture
    return best_gradient, {"architectures": reports, "selected_architecture": best_architecture,
                           "weighted_penalty": best_value, "generated_window_count": len(labels),
                           "support_window_count": len(support), "query_window_count": len(query),
                           "auxiliary_warmup_updates": len(settings.architectures) * settings.warmup_steps,
                           "auxiliary_inner_updates": 2 * len(settings.architectures) * settings.inner_steps}
