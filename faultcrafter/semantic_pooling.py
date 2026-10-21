from __future__ import annotations

import torch
import torch.nn.functional as F


NATIVE_SEMANTIC_POOLING = "native"
FIXED_ORDER_ADAPTIVE = "fixed_order_adaptive_backward_v1"


class _FixedOrderAdaptivePool(torch.autograd.Function):
    @staticmethod
    def forward(context, signal: torch.Tensor, output_size: int) -> torch.Tensor:
        if signal.ndim not in (2, 3) or signal.shape[-1] <= 0 or not signal.is_floating_point():
            raise ValueError("semantic adaptive pooling requires a nonempty floating 1d signal")
        if type(output_size) is not int or output_size <= 0:
            raise ValueError("semantic adaptive pooling output size must be a positive integer")
        context.input_shape = signal.shape
        context.output_size = output_size
        return F.adaptive_avg_pool1d(signal, output_size)

    @staticmethod
    def backward(context, output_gradient: torch.Tensor):
        length = context.input_shape[-1]
        gradient = output_gradient.new_zeros(context.input_shape)
        for index in range(context.output_size):
            start = index * length // context.output_size
            end = ((index + 1) * length + context.output_size - 1) // context.output_size
            gradient[..., start:end] += output_gradient[..., index:index + 1] / (end - start)
        return gradient, None


class FixedOrderAdaptiveAvgPool1d(torch.nn.AdaptiveAvgPool1d):
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return _FixedOrderAdaptivePool.apply(signal, self.output_size)


def configure_semantic_pooling(classifier: torch.nn.Module, policy: str) -> None:
    if policy == NATIVE_SEMANTIC_POOLING:
        return
    if policy != FIXED_ORDER_ADAPTIVE:
        raise ValueError("unknown source semantic pooling policy")
    pooling = classifier.layer6[-1]
    if not isinstance(pooling, torch.nn.AdaptiveAvgPool1d) or pooling.output_size != 4:
        raise ValueError("fixed source semantic teacher must have its original four-bin adaptive pooling")
    classifier.layer6[-1] = FixedOrderAdaptiveAvgPool1d(4).eval()
