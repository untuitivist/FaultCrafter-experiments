from __future__ import annotations

import copy
import math

import torch

from faultcrafter.shared_update_cone import value_digest
from faultcrafter.unified_generator import RoutedLoRALinear


MATRIX_EMA = "dense_lora_ema_fixed_rank_v1"


def matrix_ema_policy(generator):
    policy = generator.get("matrix_ema_policy", "none")
    if policy not in ("none", MATRIX_EMA):
        raise ValueError("unknown matrix EMA policy")
    return policy


def matrix_ema_metadata(generator):
    if matrix_ema_policy(generator) != MATRIX_EMA:
        raise ValueError("matrix EMA is not enabled")
    defaults = {"trajectory_average_policy": "ema", "relative_lora_update_policy": "none", "lookahead_update_policy": "none",
                "shared_update_policy": "native", "gradient_clipping_policy": "global", "route_optimizer_policy": "native",
                "role_learning_rate_policy": "uniform", "source_episode_policy": "none", "condition_contrast_policy": "none"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("matrix EMA cannot combine undeclared update mechanisms")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None or any(generator.get(name, False) for name in ("resumable_training", "preserve_frozen_ema")):
        raise ValueError("matrix EMA requires finite full-model training")
    decay, interval = generator.get("ema_decay"), generator.get("evaluation_interval")
    if not isinstance(decay, (int, float)) or isinstance(decay, bool) or not math.isfinite(decay) or not 0 <= decay < 1 or type(interval) is not int or interval <= 0:
        raise ValueError("invalid matrix EMA decay or materialization interval")
    return {"policy": MATRIX_EMA, "decay": float(decay), "materialization_interval": interval,
            "formula": "dense=decay*dense+(1-decay)*(student_up@student_down/rank)",
            "dense_accumulator_never_overwritten_by_compression": True,
            "projection": "cpu_float64_svd_original_rank_then_original_dtype_factors",
            "scope": "each_original_device_lora_effective_matrix_separately",
            "student_and_native_optimizer_unchanged_by_averaging": True,
            "non_lora_original_ema_preserved": True, "source_validation_and_deployment_both_compressed": True,
            "additional_persistent_dense_lora_matrices": 1, "extra_runtime_parameters_or_files": False,
            "equal_compute_claimed": False}


@torch.no_grad()
def fixed_rank_factors(matrix, rank):
    if matrix.ndim != 3 or type(rank) is not int or rank <= 0 or matrix.dtype not in (torch.float32, torch.float64) or not torch.isfinite(matrix).all():
        raise ValueError("fixed-rank factorization requires finite batched matrices and positive rank")
    source = matrix.detach().to(device="cpu", dtype=torch.float64)
    left, singular, right = torch.linalg.svd(source, full_matrices=False)
    pivot = left.abs().argmax(dim=-2, keepdim=True)
    signs = left.gather(-2, pivot).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    left, right = left * signs, right * signs.transpose(-2, -1)
    retained = min(rank, source.shape[-2], source.shape[-1])
    scale = (rank * singular[..., :retained]).sqrt()
    up = torch.zeros(source.shape[0], source.shape[-2], rank, dtype=torch.float64)
    down = torch.zeros(source.shape[0], rank, source.shape[-1], dtype=torch.float64)
    up[..., :retained] = left[..., :retained] * scale.unsqueeze(-2)
    down[..., :retained, :] = scale.unsqueeze(-1) * right[..., :retained, :]
    up, down = up.to(device=matrix.device, dtype=matrix.dtype), down.to(device=matrix.device, dtype=matrix.dtype)
    reconstructed = up.double().cpu() @ down.double().cpu() / rank
    rows = [{"singular_values": values.tolist(), "retained_directions": retained,
             "discarded_frobenius": float(torch.linalg.vector_norm(values[retained:])),
             "dense_frobenius": float(torch.linalg.vector_norm(original)),
             "reconstructed_error_frobenius": float(torch.linalg.vector_norm(original - actual))}
            for values, original, actual in zip(singular, source, reconstructed, strict=True)]
    return up, down, rows


class MatrixEMA:
    def __init__(self, average, model, generator):
        self.metadata = matrix_ema_metadata(generator)
        self.average, self.model = average, model
        parameters, targets = dict(model.named_parameters()), dict(average.named_parameters())
        if average is model or not parameters or parameters.keys() != targets.keys():
            raise ValueError("matrix EMA requires an independent matching average model")
        for name, value in parameters.items():
            target = targets[name]
            if not value.requires_grad or target.requires_grad or value.dtype not in (torch.float32, torch.float64) or value.shape != target.shape or value.dtype != target.dtype or value.device != target.device or value.data_ptr() == target.data_ptr():
                raise ValueError("matrix EMA parameter roles or storage differ")
        modules = dict(average.named_modules())
        self.routes = tuple((name, module, modules[name]) for name, module in model.named_modules() if isinstance(module, RoutedLoRALinear))
        if not self.routes or any(module.rank != target.rank or module.num_devices != model.num_devices for _, module, target in self.routes):
            raise ValueError("matrix EMA requires all original device LoRA routes")
        self.factor_names = {f"{name + '.' if name else ''}{factor}" for name, _, _ in self.routes for factor in ("lora_up", "lora_down")}
        with torch.no_grad():
            self.dense = {name: (module.lora_up @ module.lora_down / module.rank).detach() for name, module, _ in self.routes}
        if any(not torch.isfinite(value).all() for value in self.dense.values()):
            raise FloatingPointError("nonfinite matrix EMA initialization")
        self.schema = [{"name": name, "shape": list(self.dense[name].shape), "dtype": str(self.dense[name].dtype), "rank": module.rank}
                       for name, module, _ in self.routes]
        self.initial_sha256 = value_digest(self.dense)
        self.steps = []
        self.materializations = 0
        self.last_materialized_step = 0

    @torch.no_grad()
    def update(self, optimizer, step, routes, base_update):
        if type(optimizer) is not torch.optim.AdamW or type(step) is not int or step != len(self.steps) + 1:
            raise ValueError("matrix EMA requires consecutive native AdamW updates")
        parameters = tuple(self.model.parameters())
        actual = [value for group in optimizer.param_groups for value in group["params"]]
        if len(actual) != len(parameters) or {id(value) for value in actual} != {id(value) for value in parameters} or any(int(optimizer.state.get(value, {}).get("step", -1)) != step for value in parameters):
            raise ValueError("matrix EMA must follow the matching full-model native update")
        decay = self.metadata["decay"]
        row = {"step": step, "source_routes": routes.tolist(), "dense_before_sha256": value_digest(self.dense),
               "materialized": step % self.metadata["materialization_interval"] == 0}
        base_update(self.average, self.model, decay, preserve_frozen=False)
        for name, module, _ in self.routes:
            current = module.lora_up @ module.lora_down / module.rank
            self.dense[name].mul_(decay).add_(current, alpha=1 - decay)
            if not torch.isfinite(self.dense[name]).all():
                raise FloatingPointError("nonfinite matrix EMA update")
        row["dense_after_sha256"] = value_digest(self.dense)
        if row["materialized"]:
            projections = []
            for name, module, target in self.routes:
                up, down, evidence = fixed_rank_factors(self.dense[name], module.rank)
                target.lora_up.copy_(up)
                target.lora_down.copy_(down)
                projections.extend({"module": name, "device_index": index, **values} for index, values in enumerate(evidence))
            if value_digest(self.dense) != row["dense_after_sha256"]:
                raise RuntimeError("matrix EMA compression changed its dense accumulator")
            self.materializations += 1
            self.last_materialized_step = step
            row.update({"projections": projections, "materialized_model_sha256": value_digest(self.average.state_dict())})
        row["completed_materializations"] = self.materializations
        self.steps.append(row)
        return {"matrix_ema_completed_steps": step, "matrix_ema_materialized": row["materialized"],
                "matrix_ema_step_evidence": copy.deepcopy(row)}

    def audit(self):
        return {"metadata": self.metadata, "schema": copy.deepcopy(self.schema), "initial_dense_sha256": self.initial_sha256,
                "persistent_dense_bytes": sum(value.numel() * value.element_size() for value in self.dense.values()),
                "completed_steps": len(self.steps), "completed_materializations": self.materializations,
                "steps": copy.deepcopy(self.steps)}
