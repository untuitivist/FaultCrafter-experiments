from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


def model_cost(model, inputs, *, keyword_inputs=None, batch_size=1, scope="single_forward", operation=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    modes = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cuda_devices = sorted({parameter.device.index for parameter in model.parameters() if parameter.is_cuda})
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        model.eval()
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                (operation or model)(*inputs, **(keyword_inputs or {}))
        operations = {str(operator): count for operator, count in counter.get_flop_counts().get("Global", {}).items()}
        total = counter.get_total_flops()
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    parameters = list(model.parameters())
    return {
        "scope": scope,
        "parameters_total": sum(parameter.numel() for parameter in parameters),
        "parameters_trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        "batch_size": batch_size,
        "tensor_input_shapes": [list(value.shape) for value in inputs if isinstance(value, torch.Tensor)],
        "flops_counted_batch": total,
        "flops_counted_per_sample": total / batch_size,
        "gflops_counted_per_sample": total / batch_size / 1e9,
        "operations": operations,
        "counter": "torch.utils.flop_counter.FlopCounterMode",
        "torch": torch.__version__,
        "convention": "Multiply-add counts as two FLOPs; registered convolution, matrix multiplication and attention operations only. Bias, normalization, activation, pooling and other unregistered arithmetic are excluded. This is not a hardware instruction count or elapsed-time estimate.",
        "sampling_accounting": "For generation, profile the entire sampler with actual NFE and guidance calls; a single denoiser forward is not a complete generated signal.",
    }
