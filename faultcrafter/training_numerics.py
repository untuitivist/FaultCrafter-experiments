from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import os
from typing import Iterator

import torch
import torch.nn.functional as functional


STRICT_REPLAY = "strict_replay_v1"
CUBLAS_WORKSPACE = ":4096:8"


def numerical_training_mode(generator: dict) -> str:
    mode = generator.get("numerical_training_mode", "legacy")
    if mode not in ("legacy", STRICT_REPLAY):
        raise ValueError(f"unsupported numerical training mode: {mode!r}")
    return mode


def strict_training_entry(function):
    signature = inspect.signature(function)

    @wraps(function)
    def configured(*arguments, **keywords):
        bound = signature.bind(*arguments, **keywords)
        config = bound.arguments["config"]
        compute_device = bound.arguments["compute_device"]
        with training_numerics(config["generator"], compute_device) as numerical_metadata:
            if numerical_metadata is not None:
                metadata = bound.arguments.get("checkpoint_metadata") or {}
                if "training_numerics" in metadata:
                    raise ValueError("training numerical metadata cannot be supplied externally")
                bound.arguments["checkpoint_metadata"] = {**metadata, "training_numerics": numerical_metadata}
            return function(*bound.args, **bound.kwargs)

    return configured


@contextmanager
def training_numerics(generator: dict, compute_device: torch.device) -> Iterator[dict | None]:
    mode = numerical_training_mode(generator)
    if mode == "legacy":
        yield None
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if compute_device.type == "cuda" and workspace != CUBLAS_WORKSPACE:
        raise ValueError("strict CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield {
            "mode": mode,
            "strict_algorithms": True,
            "warn_only": False,
            "cublas_workspace_config": workspace,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "compute_device_type": compute_device.type,
            "detached_floor": "sorted_lower_median_values",
            "decoder_backward": "native_forward_fixed_double_adjoint",
            "extra_runtime_parameters_or_files": False,
        }
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def detached_lower_median(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("spectral floor requires nonempty two-dimensional values")
    detached = values.detach()
    if not torch.are_deterministic_algorithms_enabled():
        return detached.median(dim=1, keepdim=True).values
    ordered = torch.sort(detached, dim=1, stable=True).values
    middle = (values.shape[1] - 1) // 2
    result = ordered[:, middle:middle + 1]
    return torch.where(torch.isnan(detached).any(dim=1, keepdim=True), torch.full_like(result, float("nan")), result)


class LinearDoubleAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(context, signal: torch.Tensor, output_length: int) -> torch.Tensor:
        if signal.ndim != 3 or signal.dtype not in (torch.float32, torch.float64):
            raise ValueError("strict decoder resize requires three-dimensional float32/64 input")
        input_length = signal.shape[-1]
        if input_length < 1 or output_length not in (input_length, 2 * input_length):
            raise ValueError("strict decoder resize supports nonempty identity or exact doubling only")
        context.input_length = input_length
        context.output_length = output_length
        return functional.interpolate(signal, size=output_length, mode="linear", align_corners=False)

    @staticmethod
    def backward(context, gradient: torch.Tensor):
        if context.input_length == context.output_length:
            return gradient, None
        even = gradient[..., ::2]
        odd = gradient[..., 1::2]
        even_weight = gradient.new_full((context.input_length,), 0.75)
        odd_weight = gradient.new_full((context.input_length,), 0.75)
        even_weight[0] = 1.0
        odd_weight[-1] = 1.0
        from_left = torch.cat((torch.zeros_like(odd[..., :1]), 0.25 * odd[..., :-1]), dim=-1)
        from_right = torch.cat((0.25 * even[..., 1:], torch.zeros_like(even[..., :1])), dim=-1)
        return even_weight * even + odd_weight * odd + from_left + from_right, None


def decoder_linear_resize(signal: torch.Tensor, output_length: int) -> torch.Tensor:
    if torch.are_deterministic_algorithms_enabled() and torch.is_grad_enabled() and signal.requires_grad:
        return LinearDoubleAdjoint.apply(signal, output_length)
    return functional.interpolate(signal, size=output_length, mode="linear", align_corners=False)
