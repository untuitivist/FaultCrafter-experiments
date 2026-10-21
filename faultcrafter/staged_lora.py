from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import torch

from faultcrafter.compact_inference import is_device_lora


class DeviceBalancedCellSampler:
    def __init__(self, cells: tuple[tuple[int, int], ...], *, seed: int) -> None:
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("device balancing requires unique nonempty cells")
        self.device_ids = tuple(sorted({cell[0] for cell in cells}))
        self.cells = {
            device_id: tuple(cell for cell in cells if cell[0] == device_id)
            for device_id in self.device_ids
        }
        self.seed = seed
        self.cursor = 0
        self.device_cursors = dict.fromkeys(self.device_ids, 0)
        self.exposure = dict.fromkeys(self.device_ids, 0)

    def take(self, count: int) -> tuple[tuple[int, int], ...]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        selected = []
        for _ in range(count):
            cycle, offset = divmod(self.cursor, len(self.device_ids))
            order = np.random.default_rng(self.seed + cycle).permutation(len(self.device_ids))
            device_id = self.device_ids[int(order[offset])]
            cells = self.cells[device_id]
            cell_cycle, cell_offset = divmod(self.device_cursors[device_id], len(cells))
            cell_order = np.random.default_rng(
                self.seed + 100_000 * (1 + device_id) + cell_cycle
            ).permutation(len(cells))
            selected.append(cells[int(cell_order[cell_offset])])
            self.cursor += 1
            self.device_cursors[device_id] += 1
            self.exposure[device_id] += 1
        return tuple(selected)


class DeviceLoRAAdamW(torch.optim.AdamW):
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        device_index: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.banks = tuple(named_parameters)
        if not self.banks or any(not is_device_lora(name) for name, _ in self.banks):
            raise ValueError("device optimizer may only own existing LoRA tensors")
        if any(not 0 <= device_index < bank.shape[0] for _, bank in self.banks):
            raise ValueError("active LoRA device is outside the parameter bank")
        self.device_index = device_index
        self.local_parameters = [
            torch.nn.Parameter(bank[device_index].detach().clone())
            for _, bank in self.banks
        ]
        super().__init__(self.local_parameters, lr=learning_rate, weight_decay=weight_decay)

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        for (_, bank), local in zip(self.banks, self.local_parameters, strict=True):
            local.copy_(bank[self.device_index])

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for _, bank in self.banks:
            if set_to_none:
                bank.grad = None
            elif bank.grad is not None:
                bank.grad.zero_()

    @torch.no_grad()
    def prepare_gradients(self) -> None:
        for (_, bank), local in zip(self.banks, self.local_parameters, strict=True):
            if bank.grad is None:
                local.grad = None
                continue
            active_gradient = bank.grad[self.device_index].clone()
            bank.grad.zero_()
            bank.grad[self.device_index].copy_(active_gradient)
            local.grad = bank.grad[self.device_index]

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if closure is not None:
            raise ValueError("device-local optimizer does not support closures")
        result = super().step()
        for (_, bank), local in zip(self.banks, self.local_parameters, strict=True):
            bank[self.device_index].copy_(local)
        return result

    @torch.no_grad()
    def update_ema(self, ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
        target = dict(ema_model.named_parameters())
        for name, parameter in model.named_parameters():
            if is_device_lora(name):
                target[name][:self.device_index].copy_(parameter[:self.device_index])
                target[name][self.device_index + 1:].copy_(parameter[self.device_index + 1:])
                target[name][self.device_index].mul_(decay).add_(
                    parameter[self.device_index], alpha=1.0 - decay
                )
            else:
                target[name].copy_(parameter)
        target_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            target_buffers[name].copy_(buffer)


def state_partitions(state: dict[str, torch.Tensor], num_devices: int) -> dict[str, Any]:
    shared = hashlib.sha256()
    adapters = [hashlib.sha256() for _ in range(num_devices)]
    for name, value in sorted(state.items()):
        destinations = (
            [(adapters[index], value[index]) for index in range(num_devices)]
            if is_device_lora(name)
            else [(shared, value)]
        )
        for digest, tensor in destinations:
            digest.update(name.encode("utf-8"))
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {"shared_sha256": shared.hexdigest(), "device_lora_sha256": [item.hexdigest() for item in adapters]}


def verify_phase_isolation(before: dict[str, Any], after: dict[str, Any], device_index: int | None) -> None:
    previous = before["device_lora_sha256"]
    current = after["device_lora_sha256"]
    if len(previous) != len(current):
        raise ValueError("staged training changed the device bank size")
    if device_index is None:
        if previous != current:
            raise ValueError("shared training changed a frozen device LoRA")
        if before["shared_sha256"] == after["shared_sha256"]:
            raise ValueError("shared training did not update the backbone")
    else:
        if before["shared_sha256"] != after["shared_sha256"]:
            raise ValueError("device adaptation changed the frozen backbone")
        if any(old != new for index, (old, new) in enumerate(zip(previous, current, strict=True)) if index != device_index):
            raise ValueError("device adaptation changed another device LoRA")
        if previous[device_index] == current[device_index]:
            raise ValueError("device adaptation did not update its LoRA")
