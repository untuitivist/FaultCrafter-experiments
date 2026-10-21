from __future__ import annotations

from collections import Counter
import copy
import hashlib
import itertools
import json
import math

import numpy as np
import torch

from faultcrafter.compact_inference import is_device_lora
from faultcrafter.shared_gradient import DeviceObjectivePartition


NATIVE_UPDATE = "native"
SHARED_UPDATE_CONE = "source_cone_adamw_v1"
PREFIX = "shared_update_"
SOLVER_TOLERANCE = 1e-10


def shared_update_policy(generator):
    policy = generator.get("shared_update_policy", NATIVE_UPDATE)
    if policy not in (NATIVE_UPDATE, SHARED_UPDATE_CONE):
        raise ValueError("unknown shared optimizer update policy")
    return policy


def shared_update_metadata(generator):
    if shared_update_policy(generator) != SHARED_UPDATE_CONE or generator.get("gradient_clip_norm") != 1.0:
        raise ValueError("shared update cone requires its declared policy and original global clip1")
    for name, default in (("shared_gradient_policy", "joint"), ("condition_gradient_policy", "joint"),
                          ("gradient_clipping_policy", "global"), ("route_optimizer_policy", "dense_adamw"),
                          ("training_learning_rate_policy", "uniform")):
        if generator.get(name, default) != default:
            raise ValueError("shared update cone cannot combine another gradient/optimizer/clipping policy")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None or generator.get("resumable_training", False):
        raise ValueError("shared update cone requires finite full-model non-resumable training")
    return {"policy": SHARED_UPDATE_CONE, "objective": "minimum_euclidean_change_to_actual_native_adamw_shared_displacement",
            "constraints": "current_device_data_gradients_and_original_joint_objective_shared_gradient",
            "shared_coordinates": "all_original_parameters_except_device_lora_and_fine_identity_including_coarse_identity",
            "device_contribution_denominators": "original_global_samples_eligible_orders_and_eligible_condition_contrasts",
            "original_joint_backward_then_task_autograd_then_global_clip_then_native_adamw": True,
            "private_parameters_optimizer_moments_and_clocks_unmodified_by_projection": True,
            "ema_observes_corrected_student": True, "solver": "float64_qr_compression_exhaustive_active_sets_svd_kkt",
            "unit_geometry_tolerance": SOLVER_TOLERANCE, "rank_relative_threshold": "64*float64_epsilon*maximum_matrix_dimension",
            "parameter_rounding": "original_dtype_measured_displacement_error_cauchy_bound",
            "strict_rounded_feasibility_or_source_loss_monotonicity_claimed": False,
            "extra_runtime_parameters_or_files": False}


class SharedUpdatePartition(DeviceObjectivePartition):
    def __init__(self):
        super().__init__()
        self.extra_device_losses = {}

    def capture_contrast(self, routes, losses, weight):
        if routes.shape != losses.shape or routes.dtype != torch.long or not len(routes) or self.extra_device_losses:
            raise ValueError("shared update contrast partition requires one eligible contribution per source row")
        self.extra_device_losses = {int(route): weight * losses[routes == route].sum() / len(losses)
                                    for route in torch.unique(routes, sorted=True).tolist()}


def shared_update_parameters(model):
    return tuple((name, parameter) for name, parameter in model.named_parameters()
                 if not is_device_lora(name) and name != "fault_embedding.weight")


def shared_update_schema(model):
    named = tuple(model.named_parameters())
    if not named or any(not parameter.requires_grad or parameter.dtype not in (torch.float32, torch.float64) for _, parameter in named):
        raise ValueError("shared update cone requires all original float32/64 parameters trainable")
    shared = shared_update_parameters(model)
    private = [(name, parameter) for name, parameter in named if is_device_lora(name) or name == "fault_embedding.weight"]
    if not private or not any(name == "fault_group_embedding.weight" for name, _ in shared):
        raise ValueError("shared update cone needs original device LoRA and both identity banks")
    if sum(parameter.numel() for _, parameter in (*shared, *private)) != sum(parameter.numel() for _, parameter in named):
        raise ValueError("shared update parameter coordinates are incomplete")
    return {"shared_parameters": [{"name": name, "shape": list(parameter.shape), "elements": parameter.numel()} for name, parameter in shared],
            "private_parameters": [{"name": name, "shape": list(parameter.shape), "elements": parameter.numel()} for name, parameter in private],
            "shared_elements": sum(parameter.numel() for _, parameter in shared), "all_original_parameter_tensors": len(named),
            "device_count": int(model.num_devices), "coarse_identity_is_shared": True}


def value_digest(value):
    digest = hashlib.sha256()
    def collect(item):
        if torch.is_tensor(item):
            body = item.detach().cpu().contiguous().numpy().tobytes()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode("ascii") + b"\0" + len(body).to_bytes(8, "little") + body)
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=str):
                collect(key)
                collect(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for nested in item:
                collect(nested)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8") + b"\0")
    collect(value)
    return digest.hexdigest()


def project_small_cone(normals, original):
    normals, original = np.asarray(normals, dtype=np.float64), np.asarray(original, dtype=np.float64)
    if normals.ndim != 2 or original.shape != (normals.shape[1],) or not 1 <= len(normals) <= 5 or not np.isfinite(normals).all() or not np.isfinite(original).all():
        raise ValueError("shared update solver requires at most five finite normalized constraints")
    if np.max(np.linalg.norm(normals, axis=1)) > 1 + SOLVER_TOLERANCE or np.linalg.norm(original) > 1 + SOLVER_TOLERANCE:
        raise ValueError("shared update solver coordinates are not normalized")
    if np.all(normals @ original >= 0):
        return {"coordinates": original.tolist(), "multipliers": [0.0] * len(normals), "active_set": [], "distance": 0.0}
    best = None
    for size in range(1, len(normals) + 1):
        for indices in itertools.combinations(range(len(normals)), size):
            active = normals[list(indices)]
            _, singular, right = np.linalg.svd(active, full_matrices=False)
            cutoff = 64 * np.finfo(np.float64).eps * max(active.shape) * singular[0]
            rank = int(np.count_nonzero(singular > cutoff))
            if rank != size:
                continue
            candidate = original - right[:rank].T @ (right[:rank] @ original)
            increment = candidate - original
            multipliers, _, _, _ = np.linalg.lstsq(active.T, increment, rcond=64 * np.finfo(np.float64).eps * max(active.shape))
            if np.min(multipliers) < -SOLVER_TOLERANCE or np.min(normals @ candidate) < -SOLVER_TOLERANCE:
                continue
            multipliers = np.maximum(multipliers, 0)
            if np.linalg.norm(increment - active.T @ multipliers) > SOLVER_TOLERANCE:
                continue
            if np.max(np.abs(multipliers * (active @ candidate))) > SOLVER_TOLERANCE:
                continue
            distance = float(np.linalg.norm(increment))
            if distance > 1 + SOLVER_TOLERANCE or np.linalg.norm(candidate) > np.linalg.norm(original) + SOLVER_TOLERANCE:
                continue
            if best is None or distance < best["distance"]:
                full = np.zeros(len(normals), dtype=np.float64)
                full[list(indices)] = multipliers
                best = {"coordinates": candidate.tolist(), "multipliers": full.tolist(), "active_set": list(indices), "distance": distance}
    if best is None:
        raise ValueError("shared update cone has no numerically certified active-set solution")
    return best


def validate_shared_update_row(row, schema):
    if type(row.get("step")) is not int or row.get("executed") != 1.0:
        raise ValueError("shared update audit requires an executed integer step")
    if row.get("dtype") not in ("torch.float32", "torch.float64"):
        raise ValueError("shared update audit has an unsupported original parameter dtype")
    routes = row.get("source_routes", [])
    if not routes or any(type(route) is not int or not 0 <= route < schema["device_count"] for route in routes):
        raise ValueError("shared update trace lacks original source routes")
    labels = ["joint"] + [f"device_{route}" for route in sorted(set(routes))]
    if row.get("constraint_labels") != labels or row.get("task_backward_calls") != len(labels) - 1:
        raise ValueError("shared update trace changes the actual batch device constraints")
    if row.get("samples_by_device") != {str(route): count for route, count in sorted(Counter(routes).items())}:
        raise ValueError("shared update trace changes original source device exposure")
    norms = np.asarray(row["gradient_norms"], dtype=np.float64)
    before = np.asarray(row["original_unit_dots"], dtype=np.float64)
    ideal = np.asarray(row["ideal_unit_dots"], dtype=np.float64)
    actual = np.asarray(row["actual_unit_dots"], dtype=np.float64)
    if any(values.shape != (len(labels),) or not np.isfinite(values).all() for values in (norms, before, ideal, actual)) or np.min(norms) < 0:
        raise ValueError("shared update constraint arrays are invalid")
    for name in ("native_update_norm", "ideal_update_norm", "actual_update_norm", "rounding_norm", "rounding_unit_bound", "correction_norm", "partition_scalar_error"):
        if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]) or row[name] < 0:
            raise ValueError("shared update geometry must be finite and nonnegative")
    tolerance = SOLVER_TOLERANCE
    scale = row["native_update_norm"]
    compression = row["compression"]
    if compression is None:
        if scale and np.min(before) < 0:
            raise ValueError("shared update skipped a conflicting native displacement")
        if row["native_shared_sha256"] != row["corrected_shared_sha256"] or row["correction_norm"] != 0 or row["rounding_norm"] != 0:
            raise ValueError("shared update changed an already feasible native displacement")
        if not np.array_equal(before, ideal) or not np.array_equal(before, actual):
            raise ValueError("shared update inactive trace differs from the native displacement")
    else:
        normals = np.asarray(compression["normals"], dtype=np.float64)
        coordinates = np.asarray(compression["native_coordinates"], dtype=np.float64)
        solution = compression["solution"]
        projected = np.asarray(solution["coordinates"], dtype=np.float64)
        multipliers = np.asarray(solution["multipliers"], dtype=np.float64)
        dimension = min(len(labels), schema["shared_elements"])
        if normals.shape != (len(labels), dimension) or coordinates.shape != (dimension,) or projected.shape != coordinates.shape or multipliers.shape != (len(labels),):
            raise ValueError("shared update compressed coordinates changed dimension")
        if any(not np.isfinite(value).all() for value in (normals, coordinates, projected, multipliers)):
            raise ValueError("shared update solver trace is not finite")
        if np.max(np.abs(normals @ coordinates - before)) > tolerance or np.max(np.abs(normals @ projected - ideal)) > tolerance:
            raise ValueError("shared update compressed constraint dots do not reconstruct")
        if np.min(multipliers) < 0 or np.min(ideal) < -tolerance or np.linalg.norm(projected - coordinates - normals.T @ multipliers) > tolerance:
            raise ValueError("shared update solver violates primal or dual stationarity conditions")
        if np.max(np.abs(multipliers * ideal)) > tolerance:
            raise ValueError("shared update solver violates complementarity")
        active = solution.get("active_set", [])
        if not active or len(set(active)) != len(active) or any(type(index) is not int or not 0 <= index < len(labels) for index in active):
            raise ValueError("shared update active-set certificate is invalid")
        if any(multipliers[index] != 0 for index in range(len(labels)) if index not in active):
            raise ValueError("shared update has a nonzero multiplier outside its active set")
        increment = float(np.linalg.norm(projected - coordinates))
        if abs(increment - solution["distance"]) > tolerance or increment > 1 + tolerance or np.linalg.norm(projected) > np.linalg.norm(coordinates) + tolerance:
            raise ValueError("shared update projection is not the minimum-change contraction")
        if abs(row["correction_norm"] - increment * scale) > tolerance * max(1.0, scale):
            raise ValueError("shared update correction norm differs from compressed solution")
    if row["ideal_update_norm"] > scale + tolerance * max(1.0, scale):
        raise ValueError("shared update ideal projection increased native update norm")
    if abs(row["actual_update_norm"] - row["ideal_update_norm"]) > row["rounding_norm"] + tolerance * max(1.0, scale):
        raise ValueError("shared update actual norm is inconsistent with measured parameter rounding")
    expected_bound = row["rounding_norm"] / scale if scale else 0.0
    if abs(expected_bound - row["rounding_unit_bound"]) > tolerance:
        raise ValueError("shared update rounding bound differs from actual error norm")
    if np.max(np.abs(actual - ideal)) > expected_bound + tolerance or np.min(actual) < -expected_bound - tolerance:
        raise ValueError("shared update actual parameter displacement violates its measured rounding bound")
    for before_name, after_name in (("private_native_sha256", "private_corrected_sha256"), ("optimizer_native_sha256", "optimizer_corrected_sha256"), ("clipped_gradients_native_sha256", "clipped_gradients_corrected_sha256")):
        if row[before_name] != row[after_name]:
            raise ValueError("shared update projection changed private parameters, optimizer state, or clipped gradients")
    for name, value in row.items():
        if name.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("shared update audit hash is invalid")


class SharedUpdateActivity:
    def __init__(self, generator, schema):
        self.metadata = shared_update_metadata(generator)
        self.schema = copy.deepcopy(schema)
        self.rows = []
        self.totals = {"task_backward_calls": 0, "triggered_steps": 0, "modified_steps": 0, "native_negative_constraints": 0, "rounded_negative_constraints": 0}

    def record(self, row):
        if row.get("step") != len(self.rows) + 1:
            raise ValueError("shared update audit requires every consecutive original optimizer step")
        validate_shared_update_row(row, self.schema)
        current = {"task_backward_calls": row["task_backward_calls"], "triggered_steps": int(row["compression"] is not None),
                   "modified_steps": int(row["native_shared_sha256"] != row["corrected_shared_sha256"]),
                   "native_negative_constraints": sum(value < 0 for value in row["original_unit_dots"]),
                   "rounded_negative_constraints": sum(value < 0 for value in row["actual_unit_dots"])}
        self.rows.append(copy.deepcopy(row))
        for name, value in current.items():
            self.totals[name] += value
        return {PREFIX + "executed": 1.0, PREFIX + "total_steps": len(self.rows),
                **{PREFIX + name: value for name, value in current.items()}, **{PREFIX + "total_" + name: value for name, value in self.totals.items()},
                **{PREFIX + name: row[name] for name in ("native_update_norm", "actual_update_norm", "correction_norm", "rounding_norm")}}

    def audit(self):
        return {"metadata": copy.deepcopy(self.metadata), "schema": copy.deepcopy(self.schema), "completed_steps": len(self.rows),
                "steps": copy.deepcopy(self.rows), "totals": dict(self.totals), "historical_full_gradient_and_optimizer_trajectory_reconstructed": False,
                "extra_runtime_parameters_or_files": False}


class SharedUpdateGuard:
    def __init__(self, model, generator):
        self.model = model
        self.schema = shared_update_schema(model)
        self.named = shared_update_parameters(model)
        self.private = tuple((name, parameter) for name, parameter in model.named_parameters() if is_device_lora(name) or name == "fault_embedding.weight")
        self.activity = SharedUpdateActivity(generator, self.schema)
        self.pending = None

    def prepare(self, objective, partition, step, source_routes):
        if self.pending is not None or type(step) is not int or step != len(self.activity.rows) + 1 or not isinstance(partition, SharedUpdatePartition):
            raise ValueError("shared update requires one captured source graph per consecutive optimizer step")
        routes = np.asarray(source_routes)
        if routes.ndim != 1 or not len(routes) or routes.dtype.kind not in "iu" or np.min(routes) < 0 or np.max(routes) >= self.schema["device_count"]:
            raise ValueError("shared update requires original integer source routes")
        present = sorted(set(routes.tolist()))
        if present != sorted(partition.device_losses) or partition.sample_count != len(routes):
            raise ValueError("shared update partition does not cover actual source data")
        objective.backward(retain_graph=True)
        named_all = tuple(self.model.named_parameters())
        gradient_hash = value_digest({name: parameter.grad for name, parameter in named_all})
        parameters = tuple(parameter for _, parameter in self.named)
        def flatten(gradients):
            return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).detach().reshape(-1)
                              for parameter, gradient in zip(parameters, gradients, strict=True)])
        gradients = [flatten([parameter.grad for parameter in parameters])]
        for index, route in enumerate(present):
            gradients.append(flatten(torch.autograd.grad(partition.device_losses[route], parameters,
                                                        retain_graph=index < len(present) - 1, allow_unused=True)))
        if value_digest({name: parameter.grad for name, parameter in named_all}) != gradient_hash:
            raise ValueError("shared update task autograd altered the original joint backward gradients")
        stacked = torch.stack(gradients).to(torch.float64)
        if not torch.isfinite(stacked).all():
            raise ValueError("shared update constraint gradients must be finite")
        norms = torch.linalg.vector_norm(stacked, dim=1)
        if not torch.isfinite(norms).all():
            raise ValueError("shared update constraint norms are not finite")
        normalized = stacked / torch.where(norms > 0, norms, torch.ones_like(norms))[:, None]
        self.pending = {"step": step, "source_routes": routes.tolist(), "constraint_labels": ["joint"] + [f"device_{route}" for route in present],
                        "task_backward_calls": len(present), "gradient_norms": norms.cpu().tolist(),
                        "joint_preclip_gradients_sha256": gradient_hash, "constraint_gradients_sha256": value_digest(stacked),
                        "partition_scalar_error": partition.reconstruction_error, "samples_by_device": {str(route): count for route, count in sorted(Counter(routes.tolist()).items())},
                        "before": torch.cat([parameter.detach().reshape(-1) for parameter in parameters]), "normalized": normalized}

    @torch.no_grad()
    def apply(self, optimizer):
        if self.pending is None or type(optimizer) is not torch.optim.AdamW:
            raise ValueError("shared update requires a prepared original native AdamW step")
        pending = self.pending
        before, normalized = pending.pop("before"), pending.pop("normalized")
        native = torch.cat([parameter.detach().reshape(-1) for _, parameter in self.named])
        if not torch.isfinite(native).all():
            raise ValueError("shared update native parameters are not finite")
        native_update = before.double() - native.double()
        scale = float(torch.linalg.vector_norm(native_update))
        private_hash = value_digest(dict(self.private))
        optimizer_hash = value_digest(optimizer.state_dict())
        gradients_hash = value_digest({name: parameter.grad for name, parameter in self.model.named_parameters()})
        original_dots = normalized @ (native_update / scale) if scale else normalized.new_zeros(len(normalized))
        ideal_update = native_update
        compression = None
        correction_norm = 0.0
        if scale and bool(torch.any(original_dots < 0)):
            basis, triangular = torch.linalg.qr(normalized.T, mode="reduced")
            coordinates = basis.T @ (native_update / scale)
            normals = triangular.T.cpu().numpy()
            solution = project_small_cone(normals, coordinates.cpu().numpy())
            projected = torch.tensor(solution["coordinates"], dtype=torch.float64, device=before.device)
            correction = scale * (basis @ (projected - coordinates))
            ideal_update = native_update + correction
            correction_norm = float(torch.linalg.vector_norm(correction))
            desired = (before.double() - ideal_update).to(before.dtype)
            offset = 0
            for _, parameter in self.named:
                parameter.copy_(desired[offset:offset + parameter.numel()].reshape_as(parameter))
                offset += parameter.numel()
            compression = {"normals": normals.tolist(), "native_coordinates": coordinates.cpu().tolist(), "solution": solution}
        corrected = torch.cat([parameter.detach().reshape(-1) for _, parameter in self.named])
        actual_update = before.double() - corrected.double()
        rounding = float(torch.linalg.vector_norm(actual_update - ideal_update))
        row = {**pending, "executed": 1.0, "dtype": str(before.dtype), "native_update_norm": scale,
               "ideal_update_norm": float(torch.linalg.vector_norm(ideal_update)), "actual_update_norm": float(torch.linalg.vector_norm(actual_update)),
               "correction_norm": correction_norm, "rounding_norm": rounding, "rounding_unit_bound": rounding / scale if scale else 0.0,
               "original_unit_dots": original_dots.cpu().tolist(),
               "ideal_unit_dots": (normalized @ (ideal_update / scale)).cpu().tolist() if scale else original_dots.cpu().tolist(),
               "actual_unit_dots": (normalized @ (actual_update / scale)).cpu().tolist() if scale else original_dots.cpu().tolist(),
               "compression": compression, "native_shared_sha256": value_digest(native), "corrected_shared_sha256": value_digest(corrected),
               "private_native_sha256": private_hash, "private_corrected_sha256": value_digest(dict(self.private)),
               "optimizer_native_sha256": optimizer_hash, "optimizer_corrected_sha256": value_digest(optimizer.state_dict()),
               "clipped_gradients_native_sha256": gradients_hash,
               "clipped_gradients_corrected_sha256": value_digest({name: parameter.grad for name, parameter in self.model.named_parameters()}),
               "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
        self.pending = None
        return self.activity.record(row)
