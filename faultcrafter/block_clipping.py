from __future__ import annotations

import copy
import hashlib
import math
from numbers import Integral
from typing import Any, Mapping

import torch

from faultcrafter.compact_inference import is_device_lora


GLOBAL_CLIPPING = "global"
SHARED_PRIVATE_CLIPPING = "shared_private_blocks_v1"
PREFIX = "block_clipping_"


def block_clipping_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("gradient_clipping_policy", GLOBAL_CLIPPING)
    if policy not in (GLOBAL_CLIPPING, SHARED_PRIVATE_CLIPPING):
        raise ValueError("unknown gradient clipping policy")
    return policy


def block_clipping_metadata(generator: Mapping[str, Any]) -> dict:
    if block_clipping_policy(generator) != SHARED_PRIVATE_CLIPPING or generator.get("gradient_clip_norm") != 1.0:
        raise ValueError("block clipping requires the original shared L2 threshold1")
    for field, default in (("shared_gradient_policy", "joint"), ("condition_gradient_policy", "joint"),
                           ("route_optimizer_policy", "dense_adamw"), ("training_learning_rate_policy", "uniform")):
        if field in generator and generator[field] != default:
            raise ValueError("block clipping cannot combine another gradient or optimizer policy")
    if any(generator.get(field, False) for field in ("resumable_training", "device_balanced_sampling", "relation_balanced_sampling", "relation_coupled_noise_time")):
        raise ValueError("block clipping requires the original finite unbalanced source protocol")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None:
        raise ValueError("block clipping requires the full trainable model")
    return {"policy": SHARED_PRIVATE_CLIPPING, "norm_type": 2.0, "max_norm": 1.0, "denominator_epsilon": 1e-6,
            "formula": "coefficient=min(1,max_norm/(block_l2_norm+1e-6));gradient*=coefficient",
            "shared_block": "all_parameters_except_device_lora_and_fine_identity_including_coarse_identity",
            "private_blocks": "existing_device_lora_axis_slices_and_existing_device_fine_identity_rows",
            "none_gradients_preserved": True, "zero_gradients_keep_original_dense_adamw_behavior": True,
            "global_model_post_norm_bounded_by_one": False, "source_sampling_loss_and_backward_unchanged": True,
            "optimizer_lr_ema_and_precision_controls_unchanged": True, "extra_runtime_parameters_or_files": False}


def fine_identity_ranges(devices) -> list[tuple[int, int, int]]:
    ranges, offset = [], 0
    for device in devices:
        if device.index != len(ranges) or device.fault_offset != offset or getattr(device, "model_fault_labels", None) is not None:
            raise ValueError("block clipping requires contiguous single-fault device identity ranges")
        count = len(device.corpus.fault_labels)
        if count <= 0:
            raise ValueError("block clipping cannot have empty device identities")
        ranges.append((device.index, offset, offset + count))
        offset += count
    return ranges


def block_gradient_schema(model, ranges) -> dict:
    named = tuple(model.named_parameters())
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("block clipping requires every original model parameter trainable")
    devices = int(model.num_devices)
    if len(ranges) != devices or not devices:
        raise ValueError("block clipping device identity map does not cover model routes")
    offset = 0
    for expected, triple in enumerate(ranges):
        if len(triple) != 3 or any(type(value) is not int for value in triple):
            raise ValueError("block clipping identity coordinates must be integer triples")
        device, start, end = triple
        if device != expected or start != offset or end <= start:
            raise ValueError("block clipping identity ranges overlap or have gaps")
        offset = end
    parameters = dict(named)
    if "fault_embedding.weight" not in parameters or "fault_group_embedding.weight" not in parameters:
        raise ValueError("block clipping requires both existing identity banks")
    if parameters["fault_embedding.weight"].shape[0] != offset:
        raise ValueError("block clipping fine identity rows are not completely assigned")
    blocks = [{"id": "shared", "device": None, "views": []}]
    blocks += [{"id": f"device_{device}", "device": device, "views": []} for device in range(devices)]
    banks = 0
    for name, parameter in named:
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("block clipping only supports original finite float32/64 gradients")
        if is_device_lora(name):
            if parameter.ndim < 2 or parameter.shape[0] != devices:
                raise ValueError("block clipping LoRA banks need the existing leading device axis")
            banks += 1
            for device in range(devices):
                blocks[device + 1]["views"].append({"parameter": name, "rows": [device, device + 1], "elements": parameter[device:device + 1].numel()})
        elif name == "fault_embedding.weight":
            for device, start, end in ranges:
                blocks[device + 1]["views"].append({"parameter": name, "rows": [start, end], "elements": parameter[start:end].numel()})
        else:
            blocks[0]["views"].append({"parameter": name, "rows": None, "elements": parameter.numel()})
    if not banks or any(not block["views"] for block in blocks):
        raise ValueError("block clipping requires nonempty shared and device-private blocks")
    for block in blocks:
        block["elements"] = sum(view["elements"] for view in block["views"])
    total = sum(parameter.numel() for _, parameter in named)
    if sum(block["elements"] for block in blocks) != total:
        raise ValueError("block clipping coordinates do not cover the original model")
    return {"blocks": blocks, "original_parameter_tensors": len(named), "original_parameter_elements": total,
            "device_count": devices, "lora_parameter_banks": banks, "fine_identity_ranges": [list(value) for value in ranges],
            "coarse_identity_in_shared_block": True}


def gradient_views(model, schema):
    parameters = dict(model.named_parameters())
    result = []
    for block in schema["blocks"]:
        views = []
        for descriptor in block["views"]:
            gradient = parameters[descriptor["parameter"]].grad
            if gradient is not None and descriptor["rows"] is not None:
                start, end = descriptor["rows"]
                gradient = gradient[start:end]
            if gradient is not None and (gradient.is_sparse or gradient.numel() != descriptor["elements"]):
                raise ValueError("block clipping gradient coordinates changed")
            views.append(gradient)
        result.append(views)
    return result


def view_digest(views):
    value = hashlib.sha256()
    for gradient in views:
        if gradient is None:
            value.update(b"none\0")
        else:
            payload = gradient.detach().cpu().contiguous().numpy().tobytes()
            value.update(len(payload).to_bytes(8, "little") + payload)
    return value.hexdigest()


def validate_block_clipping_row(row, schema):
    dtype = row.get("dtype")
    if dtype not in ("torch.float32", "torch.float64") or row.get("executed") != 1.0:
        raise ValueError("block clipping trace lacks its original numerical dtype or execution")
    epsilon = torch.finfo(torch.float32 if dtype == "torch.float32" else torch.float64).eps
    blocks = row.get("blocks", [])
    if len(blocks) != len(schema["blocks"]) or row.get("max_norm") != 1.0:
        raise ValueError("block clipping trace changes declared blocks or threshold")
    def close(actual, expected):
        return type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= 32 * epsilon * max(1.0, abs(expected))
    for block, descriptor in zip(blocks, schema["blocks"], strict=True):
        if block.get("id") != descriptor["id"]:
            raise ValueError("block clipping route order changed")
        before, after = block.get("tensor_norms_before", []), block.get("tensor_norms_after", [])
        if len(before) != len(descriptor["views"]) or len(after) != len(before):
            raise ValueError("block clipping requires all original gradient view norms")
        elements = 0
        for original, clipped, view in zip(before, after, descriptor["views"], strict=True):
            if original is None:
                if clipped is not None:
                    raise ValueError("block clipping activated a None gradient")
            else:
                if type(original) not in (int, float) or not math.isfinite(original) or original < 0 or not close(clipped, original * block["coefficient"]):
                    raise ValueError("block clipping tensor scaling differs from declared coefficient")
                elements += view["elements"]
        original_norm = math.sqrt(math.fsum(value ** 2 for value in before if value is not None))
        clipped_norm = math.sqrt(math.fsum(value ** 2 for value in after if value is not None))
        if not close(block.get("norm_before"), original_norm) or not close(block.get("norm_after"), clipped_norm):
            raise ValueError("block clipping aggregate norm differs from per-view evidence")
        expected = min(1.0, 1.0 / (float(block["norm_before"]) + 1e-6))
        if not close(block.get("coefficient"), expected) or not 0 < block["coefficient"] <= 1 or block["norm_after"] > 1.0 + 32 * epsilon:
            raise ValueError("block clipping violates its original per-block L2 rule")
        if block.get("gradient_elements") != elements or type(block["gradient_elements"]) is not int:
            raise ValueError("block clipping gradient coverage changed")
        for name in ("before_sha256", "after_sha256"):
            digest = block.get(name)
            if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError("block clipping gradient hash is missing")
        if block["coefficient"] == 1.0 and block["before_sha256"] != block["after_sha256"]:
            raise ValueError("block clipping changed an unscaled gradient block")
    for suffix in ("before", "after"):
        expected = math.sqrt(math.fsum(block["norm_" + suffix] ** 2 for block in blocks))
        if not close(row.get("global_norm_" + suffix), expected):
            raise ValueError("block clipping global norm does not reconstruct")
    if not close(row.get("original_global_coefficient"), min(1.0, 1.0 / (row["global_norm_before"] + 1e-6))):
        raise ValueError("block clipping original-global comparison differs")
    routes = row.get("source_routes")
    if not isinstance(routes, list) or not routes or any(type(route) is not int or not 0 <= route < schema["device_count"] for route in routes):
        raise ValueError("block clipping source routes are missing")


class BlockClippingActivity:
    def __init__(self, generator, schema):
        self.metadata = block_clipping_metadata(generator)
        self.schema = copy.deepcopy(schema)
        self.steps = []
        self.totals = {"clipped_blocks": 0, "private_blocks_preserved_when_global_would_clip": 0}

    def record(self, row):
        if type(row.get("step")) is not int or row["step"] != len(self.steps) + 1:
            raise ValueError("block clipping requires consecutive complete optimizer-step records")
        validate_block_clipping_row(row, self.schema)
        self.steps.append(copy.deepcopy(row))
        clipped = sum(block["coefficient"] < 1.0 for block in row["blocks"])
        preserved = sum(block["norm_before"] > 0 and block["coefficient"] == 1.0 for block in row["blocks"][1:]) if row["original_global_coefficient"] < 1.0 else 0
        self.totals["clipped_blocks"] += clipped
        self.totals["private_blocks_preserved_when_global_would_clip"] += preserved
        return {PREFIX + "executed": 1.0, PREFIX + "clipped_blocks": clipped,
                PREFIX + "private_blocks_preserved_when_global_would_clip": preserved,
                PREFIX + "global_norm_before": row["global_norm_before"], PREFIX + "global_norm_after": row["global_norm_after"],
                PREFIX + "total_steps": len(self.steps), **{PREFIX + "total_" + key: value for key, value in self.totals.items()}}

    def audit(self):
        return {"metadata": copy.deepcopy(self.metadata), "schema": copy.deepcopy(self.schema),
                "completed_steps": len(self.steps), "steps": copy.deepcopy(self.steps), "totals": dict(self.totals),
                "all_historical_student_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}


class SharedPrivateGradientClipper:
    def __init__(self, model, generator, ranges):
        self.model = model
        self.schema = block_gradient_schema(model, ranges)
        self.activity = BlockClippingActivity(generator, self.schema)

    @torch.no_grad()
    def apply(self, step, source_routes):
        if type(step) is not int or step != len(self.activity.steps) + 1:
            raise ValueError("block clipping step must be consecutive before modifying gradients")
        source_routes = list(source_routes)
        if any(not isinstance(route, Integral) or isinstance(route, bool) for route in source_routes):
            raise ValueError("block clipping source routes must be original integer device IDs")
        routes = list(map(int, source_routes))
        if not routes or min(routes) < 0 or max(routes) >= self.schema["device_count"]:
            raise ValueError("block clipping source route is outside existing devices")
        original = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        first = next(self.model.parameters())
        if any(gradient.device != first.device or gradient.dtype != first.dtype or gradient.is_sparse for gradient in original):
            raise ValueError("block clipping gradients must have one original dense numerical dtype/device")
        global_before = torch.nn.utils.get_total_norm(original, 2.0, error_if_nonfinite=True) if original else first.new_zeros(())
        grouped = gradient_views(self.model, self.schema)
        norms = [torch.nn.utils.get_total_norm([gradient for gradient in views if gradient is not None], 2.0, error_if_nonfinite=True)
                 if any(gradient is not None for gradient in views) else global_before.new_zeros(()) for views in grouped]
        rows = []
        for descriptor, views, norm in zip(self.schema["blocks"], grouped, norms, strict=True):
            active = [gradient for gradient in views if gradient is not None]
            before = [float(torch.linalg.vector_norm(gradient, 2)) if gradient is not None else None for gradient in views]
            before_hash = view_digest(views)
            coefficient = (1.0 / (norm + 1e-6)).clamp(max=1.0)
            for gradient in active:
                gradient.mul_(coefficient)
            after = [float(torch.linalg.vector_norm(gradient, 2)) if gradient is not None else None for gradient in views]
            post_norm = torch.nn.utils.get_total_norm(active, 2.0, error_if_nonfinite=True) if active else global_before.new_zeros(())
            rows.append({"id": descriptor["id"], "tensor_norms_before": before, "tensor_norms_after": after,
                         "norm_before": float(norm), "norm_after": float(post_norm), "coefficient": float(coefficient),
                         "gradient_elements": sum(gradient.numel() for gradient in active),
                         "before_sha256": before_hash, "after_sha256": view_digest(views)})
        global_after = torch.nn.utils.get_total_norm(original, 2.0, error_if_nonfinite=True) if original else global_before.new_zeros(())
        row = {"step": step, "executed": 1.0, "dtype": str(first.dtype), "max_norm": 1.0,
               "source_routes": routes, "global_norm_before": float(global_before), "global_norm_after": float(global_after),
               "original_global_coefficient": float((1.0 / (global_before + 1e-6)).clamp(max=1.0)), "blocks": rows,
               "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
        return global_before, self.activity.record(row)
