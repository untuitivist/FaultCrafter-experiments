from __future__ import annotations

import copy
from typing import Any

import torch

from faultcrafter.compact_inference import is_device_lora


DENSE_ADAMW = "dense_adamw"
ROUTE_LOCAL_ADAMW = "route_local_adamw_v1"
GLOBAL_SLOPE_BANKS = ("style_field.query.lora_down", "style_field.query.lora_up")
IDENTITY_BANKS = {"fault_embedding.weight": "fault", "fault_group_embedding.weight": "group"}


def route_optimizer_policy(generator: dict) -> str:
    policy = generator.get("route_optimizer_policy", DENSE_ADAMW)
    if policy not in (DENSE_ADAMW, ROUTE_LOCAL_ADAMW):
        raise ValueError("unknown routed optimizer policy")
    return policy


def route_optimizer_metadata(generator: dict) -> dict:
    global_slope = float(generator.get("condition_slope_regularization_weight", 0.0)) > 0
    return {"policy": ROUTE_LOCAL_ADAMW, "shared_parameters": "original_dense_adamw_every_global_step",
            "device_lora": "existing_device_axis_blocks_active_by_batch_device",
            "identity_embeddings": "existing_rows_active_by_batch_fine_and_group_identity",
            "global_condition_regularizer_banks": list(GLOBAL_SLOPE_BANKS) if global_slope else [],
            "inactive_blocks": "grad_none_no_parameter_momentum_decay_or_local_clock_update",
            "active_zero_gradient_blocks": "normal_adamw_update",
            "unexpected_inactive_nonzero_gradient": "error_never_discard",
            "gradient_clipping": "original_model_parameter_list_before_work_block_mapping",
            "learning_rate_and_ema": "unchanged_global_schedule_and_global_ema",
            "optimizer_work_copies": "existing_routed_coordinates_only_not_new_model_degrees_of_freedom",
            "private_update_counts_equal_global_steps_claimed": False,
            "extra_runtime_parameters_or_files": False}


class RouteLocalAdamW(torch.optim.AdamW):
    def __init__(self, model: torch.nn.Module, generator: dict) -> None:
        if route_optimizer_policy(generator) != ROUTE_LOCAL_ADAMW or model.style_mechanism != "legacy_affine":
            raise ValueError("route-local optimizer requires its declared legacy-affine full-model policy")
        if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None:
            raise ValueError("route-local full-model training cannot use a frozen staged device subset")
        self.original = tuple(model.named_parameters())
        if not self.original or any(not parameter.requires_grad for _, parameter in self.original):
            raise ValueError("route-local optimizer requires all original model parameters trainable")
        self.metadata = route_optimizer_metadata(generator)
        self.banks = {}
        self.shared_parameters = []
        parameters = []
        for name, parameter in self.original:
            kind = "device" if is_device_lora(name) else IDENTITY_BANKS.get(name)
            if kind is None:
                self.shared_parameters.append(parameter)
                parameters.append(parameter)
                continue
            if parameter.ndim < 2 or parameter.shape[0] <= 0:
                raise ValueError("route banks must have a nonempty leading route axis")
            work = [torch.nn.Parameter(parameter[index].detach().clone()) for index in range(parameter.shape[0])]
            self.banks[name] = {"parameter": parameter, "kind": kind, "work": work}
            parameters.extend(work)
        if any(name not in self.banks for name in IDENTITY_BANKS):
            raise ValueError("route-local optimizer requires both original identity banks")
        self.route_sizes = {"device": int(model.num_devices), "fault": len(self.banks["fault_embedding.weight"]["work"]),
                            "group": len(self.banks["fault_group_embedding.weight"]["work"])}
        if any(len(bank["work"]) != self.route_sizes[bank["kind"]] for bank in self.banks.values()):
            raise ValueError("route bank sizes disagree with existing model routes")
        if any(name not in self.banks for name in self.metadata["global_condition_regularizer_banks"]):
            raise ValueError("global condition regularizer banks are missing")
        if sum(parameter.numel() for parameter in parameters) != sum(parameter.numel() for _, parameter in self.original):
            raise ValueError("optimizer work blocks changed the model coordinate count")
        self.total_steps = 0
        self.route_steps = {kind: [0] * size for kind, size in self.route_sizes.items()}
        self.total_updates = 0
        self.total_skips = 0
        self.current_routes = {}
        self.current_updates = 0
        self.current_skips = 0
        self.prepared = False
        super().__init__(parameters, lr=float(generator["learning_rate"]), weight_decay=float(generator["weight_decay"]))

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for bank in self.banks.values():
            parameter = bank["parameter"]
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        self.prepared = False

    def _routes(self, values: Any, kind: str) -> tuple[int, ...]:
        tensor = torch.as_tensor(values)
        if tensor.ndim != 1 or not tensor.numel() or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("route-local optimizer requires nonempty integer batch routes")
        routes = tuple(sorted(set(tensor.detach().cpu().tolist())))
        if routes[0] < 0 or routes[-1] >= self.route_sizes[kind]:
            raise ValueError("batch route is outside the existing parameter bank")
        return routes

    @torch.no_grad()
    def prepare_gradients(self, device: Any, fault: Any, group: Any) -> None:
        if self.prepared:
            raise RuntimeError("route optimizer gradients were already prepared")
        if len(device) != len(fault) or len(device) != len(group):
            raise ValueError("device and identity routes need one entry per original sample")
        routes = {kind: self._routes(values, kind) for kind, values in (("device", device), ("fault", fault), ("group", group))}
        if any(parameter.grad is None for _, parameter in self.original):
            raise ValueError("full-model route training requires every original parameter gradient tensor")
        inactive_checks = []
        active_by_bank = {}
        for name, bank in self.banks.items():
            active = tuple(range(len(bank["work"]))) if name in self.metadata["global_condition_regularizer_banks"] else routes[bank["kind"]]
            active_by_bank[name] = active
            inactive = sorted(set(range(len(bank["work"]))) - set(active))
            if inactive:
                inactive_checks.append(torch.any(bank["parameter"].grad[inactive] != 0))
        if inactive_checks and bool(torch.stack(inactive_checks).any()):
            raise ValueError("an inactive route has a genuine nonzero gradient; refusing to discard it")
        updates = 0
        skips = 0
        for name, bank in self.banks.items():
            active = set(active_by_bank[name])
            for index, work in enumerate(bank["work"]):
                work.grad = bank["parameter"].grad[index].detach().clone() if index in active else None
                updates += int(index in active)
                skips += int(index not in active)
        self.current_routes = routes
        self.current_updates = updates
        self.current_skips = skips
        self.prepared = True

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if closure is not None or not self.prepared:
            raise ValueError("route-local step requires prepared original gradients and no closure")
        result = super().step()
        for bank in self.banks.values():
            for index, work in enumerate(bank["work"]):
                if work.grad is not None:
                    bank["parameter"][index].copy_(work)
        self.total_steps += 1
        for kind, indices in self.current_routes.items():
            for index in indices:
                self.route_steps[kind][index] += 1
        self.total_updates += self.current_updates
        self.total_skips += self.current_skips
        self.prepared = False
        return result

    def diagnostics(self, *, optimizer_clocks: bool = False) -> dict:
        if self.prepared:
            raise ValueError("route optimizer diagnostics require a complete update, not prepared gradients")
        report = {"route_optimizer_policy_executed": 1.0, "route_optimizer_total_steps": self.total_steps,
                  "route_optimizer_active_devices": list(self.current_routes.get("device", ())),
                  "route_optimizer_active_faults": list(self.current_routes.get("fault", ())),
                  "route_optimizer_active_groups": list(self.current_routes.get("group", ())),
                  "route_optimizer_device_data_steps": list(self.route_steps["device"]),
                  "route_optimizer_fault_steps": list(self.route_steps["fault"]),
                  "route_optimizer_group_steps": list(self.route_steps["group"]),
                  "route_optimizer_blocks_updated": self.current_updates, "route_optimizer_blocks_skipped": self.current_skips,
                  "route_optimizer_total_block_updates": self.total_updates, "route_optimizer_total_block_skips": self.total_skips,
                  "route_optimizer_global_regularizer_banks": list(self.metadata["global_condition_regularizer_banks"])}
        if optimizer_clocks:
            clocks = {}
            for name, bank in self.banks.items():
                actual = [int(self.state.get(work, {}).get("step", 0)) for work in bank["work"]]
                expected = [self.total_steps] * len(bank["work"]) if name in self.metadata["global_condition_regularizer_banks"] else self.route_steps[bank["kind"]]
                if actual != expected:
                    raise RuntimeError("actual AdamW block clocks disagree with recorded routing")
                clocks[name] = actual
            shared_clocks = [int(self.state.get(parameter, {}).get("step", 0)) for parameter in self.shared_parameters]
            if shared_clocks != [self.total_steps] * len(self.shared_parameters):
                raise RuntimeError("shared AdamW clocks did not advance on every global update")
            report["route_optimizer_actual_block_steps"] = clocks
            report["route_optimizer_shared_parameter_steps"] = shared_clocks
        return report

    def state_dict(self) -> dict:
        if self.prepared:
            raise ValueError("route optimizer snapshots must be taken between complete updates")
        result = super().state_dict()
        result["route_local_counters"] = {"metadata": self.metadata, "total_steps": self.total_steps,
                                          "route_steps": copy.deepcopy(self.route_steps), "total_updates": self.total_updates,
                                          "total_skips": self.total_skips}
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        payload = copy.deepcopy(state_dict)
        counters = payload.pop("route_local_counters", None)
        if counters is None or counters["metadata"] != self.metadata or set(counters["route_steps"]) != set(self.route_sizes):
            raise ValueError("route-local optimizer resume metadata differs")
        for kind, values in counters["route_steps"].items():
            if len(values) != self.route_sizes[kind] or any(type(value) is not int or not 0 <= value <= counters["total_steps"] for value in values):
                raise ValueError("route-local optimizer resume clocks are invalid")
        super().load_state_dict(payload)
        for bank in self.banks.values():
            for index, work in enumerate(bank["work"]):
                work.copy_(bank["parameter"][index])
        self.total_steps = counters["total_steps"]
        self.route_steps = counters["route_steps"]
        self.total_updates = counters["total_updates"]
        self.total_skips = counters["total_skips"]
        self.current_routes = {}
        self.current_updates = 0
        self.current_skips = 0
        self.prepared = False
        self.diagnostics(optimizer_clocks=True)
