from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import json
import math

import numpy as np
import torch

from faultcrafter.condition_contrast import SourceConditionContrastRoutes
from faultcrafter.condition_gradient import ConditionObjectivePartition
from faultcrafter.sharpness import RandomSnapshot
from faultcrafter.shared_update_cone import value_digest
from faultcrafter.source_episode import SourceEpisodeCell, SourceEpisodeRegistry


SOURCE_EPISODE_ADAMW = "source_condition_adamw_episode_v1"
PREFIX = "source_episode_"


def episode_adaptation_policy(generator):
    policy = generator.get("source_episode_policy", "none")
    if policy not in ("none", SOURCE_EPISODE_ADAMW):
        raise ValueError("unknown source episode adaptation policy")
    return policy


def episode_adaptation_metadata(generator):
    if episode_adaptation_policy(generator) != SOURCE_EPISODE_ADAMW or generator.get("gradient_clip_norm") != 1.0:
        raise ValueError("source episode requires its declared policy and original global clip1")
    defaults = {"shared_gradient_policy": "joint", "condition_gradient_policy": "joint", "gradient_clipping_policy": "global",
                "shared_update_policy": "native", "route_optimizer_policy": "dense_adamw", "training_learning_rate_policy": "uniform",
                "sharpness_training_policy": "standard", "source_condition_contrast_policy": "none"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("source episode cannot combine another gradient/optimizer/clipping or condition contrast policy")
    if generator.get("active_lora_device_index") is not None or generator.get("training_device_ids") is not None:
        raise ValueError("source episode requires all original parameters trainable")
    if any(generator.get(name, False) for name in ("resumable_training", "device_balanced_sampling", "relation_balanced_sampling", "relation_coupled_noise_time")):
        raise ValueError("source episode requires the original finite source sampling protocol")
    return {"policy": SOURCE_EPISODE_ADAMW, "baseline": "original79_source_objective",
            "condition_split": "per_device_sorted_present_typed_coordinates_query_index_(step-1)%count_if_count>=2",
            "data_denominators": "original_full_batch_samples_and_original_full_batch_eligible_orders",
            "regularizer": "original_slope_once_in_outer_base_not_virtual_support",
            "virtual_optimizer": "current_native_adamw_state_lr_decay_original_global_clip1",
            "outer_gradient": "base-query_original+query_virtual_float64_arithmetic_then_original_dtype",
            "identical_query_gradients_preserve_original_gradient_bitwise": True,
            "second_order_meta_derivatives_included": False, "previous_source_exposure_in_model_and_moments": True,
            "temporary_parameters_optimizer_buffers_and_rng_restored": True, "original_batch_noise_time_replayed": True,
            "empty_query": "original_joint_backward_without_virtual_update_or_second_objective",
            "additional_objective_calls_per_active_step": 1, "additional_autograd_calls_per_active_step": 3,
            "additional_virtual_optimizer_calls_per_active_step": 1, "equal_compute_claimed": False,
            "source_validation": "unchanged_original79_uniform_time_objective", "extra_runtime_parameters_or_files": False}


def training_episode_registry(devices, cells):
    metadata = SourceConditionContrastRoutes.from_source_cells(devices, cells)
    entries = []
    for device, record in cells:
        row = metadata.entries[metadata.records[(device, record)]]
        entries.append(SourceEpisodeCell(device, record, row["fault"], tuple(row["parameters"])))
    return SourceEpisodeRegistry(entries)


def episode_registry_audit(registry):
    return {"source_training_metadata_only": True, "cells": [asdict(row) for _, row in sorted(registry.records.items())],
            "counterfactual_condition_forward_used": False, "extra_runtime_parameters_or_files": False}


def route_key(entry):
    return entry.device, json.dumps(list(entry.condition), separators=(",", ":"))


class EpisodeObjectivePartition(ConditionObjectivePartition):
    def __init__(self, registry, cells, step):
        self.cells = tuple(cells)
        self.episode = registry.split(self.cells, step)
        super().__init__(tuple(route_key(registry.records[cell]) for cell in self.cells))
        self.query_routes = frozenset(self.routes[index] for index in self.episode.query_indices)

    def data_loss(self, *, query):
        if not self.task_losses or self.regularizer is None:
            raise ValueError("episode data loss requires its complete original objective partition")
        losses = [loss for route, loss in sorted(self.task_losses.items()) if (route in self.query_routes) == query]
        return torch.stack(losses).sum() if losses else None


def gradient_tuple(model):
    return tuple(None if parameter.grad is None else parameter.grad.detach().clone() for parameter in model.parameters())


def gradient_norm(gradients):
    values = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
    result = float(torch.stack(values).sum().sqrt()) if values else 0.0
    if not math.isfinite(result):
        raise ValueError("episode gradients must be finite")
    return result


def parameter_gradients(loss, parameters, *, retain_graph):
    if loss is None or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("episode data contribution must be a finite scalar")
    gradients = tuple(None if value is None else value.detach() for value in
                      torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True))
    gradient_norm(gradients)
    return gradients


def replace_query_gradients(base, original_query, virtual_query):
    if not len(base) == len(original_query) == len(virtual_query):
        raise ValueError("episode gradient collections have different parameter counts")
    result = []
    for original, before, after in zip(base, original_query, virtual_query, strict=True):
        present = [value for value in (original, before, after) if value is not None]
        if not present:
            result.append(None)
            continue
        reference = present[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device for value in present):
            raise ValueError("episode gradient coordinates or dtypes differ")
        if original is None and before is not None:
            raise ValueError("original query gradient exists outside the original full objective")
        if (before is None and after is None) or (before is not None and after is not None and torch.equal(before, after)):
            result.append(None if original is None else original.clone())
        else:
            combined = torch.zeros_like(reference, dtype=torch.float64) if original is None else original.double()
            if before is not None:
                combined = combined - before.double()
            if after is not None:
                combined = combined + after.double()
            result.append(combined.to(reference.dtype))
    result = tuple(result)
    gradient_norm(result)
    return result


def assign_gradients(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()


class EpisodeAdaptationActivity:
    def __init__(self, generator):
        self.metadata = episode_adaptation_metadata(generator)
        self.rows = []
        self.totals = {name: 0 for name in ("active_steps", "empty_steps", "virtual_optimizer_calls", "extra_objective_calls",
                                          "extra_autograd_calls", "joint_backward_calls", "samples", "support_samples", "query_samples",
                                          "changed_virtual_parameter_tensors", "changed_outer_gradient_tensors")}

    def record(self, row):
        if row["step"] != len(self.rows) + 1 or row["samples"] != row["support_samples"] + row["query_samples"]:
            raise ValueError("source episode activity requires consecutive complete batches")
        active = row["query_samples"] > 0
        expected = {"active_steps": int(active), "empty_steps": int(not active), "virtual_optimizer_calls": int(active),
                    "extra_objective_calls": int(active), "extra_autograd_calls": 3 * int(active), "joint_backward_calls": 1}
        if any(row[name] != value for name, value in expected.items()):
            raise ValueError("source episode computation counters disagree with its query branch")
        for name in ("parameters", "optimizer", "buffers"):
            if row[name + "_before_sha256"] != row[name + "_restored_sha256"]:
                raise ValueError("source episode temporary state leaked into the persistent update")
        if not row["rng_restored_to_single_original_pass"] or not row["replayed_rng_matches_original_pass"]:
            raise ValueError("source episode altered the original random stream")
        for name in self.totals:
            self.totals[name] += int(row[name])
        self.rows.append(copy.deepcopy(row))
        return {PREFIX + name: float(row[name]) for name in self.totals} | {
            PREFIX + "total_" + name: float(value) for name, value in self.totals.items()} | {
            PREFIX + name: float(row[name]) for name in ("virtual_displacement_norm", "outer_replacement_norm", "base_gradient_norm", "outer_gradient_norm")}

    def audit(self):
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "totals": dict(self.totals), "steps": copy.deepcopy(self.rows)}


class SourceEpisodeAdaptation:
    def __init__(self, model, generator, registry):
        self.model = model
        self.parameters = tuple(model.parameters())
        if not self.parameters or any(not parameter.requires_grad for parameter in self.parameters):
            raise ValueError("source episode requires the complete trainable original model")
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) != 1 or any(not parameter.is_floating_point() or parameter.layout != torch.strided for parameter in self.parameters):
            raise ValueError("source episode requires dense floating parameters on one device")
        self.cuda_devices = sorted(device.index for device in devices if device.type == "cuda")
        self.registry = registry
        self.activity = EpisodeAdaptationActivity(generator)

    def backward(self, optimizer, evaluate, batch, random, step):
        if type(optimizer) is not torch.optim.AdamW or not self.model.training or not torch.is_grad_enabled():
            raise ValueError("source episode requires active full-model native AdamW training")
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(optimized) != len(self.parameters) or {id(parameter) for parameter in optimized} != {id(parameter) for parameter in self.parameters}:
            raise ValueError("source episode optimizer must own every original parameter exactly once")
        if any(parameter.grad is not None for parameter in self.parameters) or type(step) is not int or step != len(self.activity.rows) + 1:
            raise ValueError("source episode requires cleared gradients and consecutive persistent steps")
        if batch.source_record_index is None:
            raise ValueError("source episode requires actual registered source records")
        cells = tuple(zip(batch.device_id.tolist(), batch.source_record_index.tolist(), strict=True))
        partition = EpisodeObjectivePartition(self.registry, cells, step)
        for cell, fault, condition in zip(cells, batch.fault.tolist(), batch.condition_parameters, strict=True):
            entry = self.registry.records[cell]
            if entry.fault != fault or entry.condition != tuple(condition.tolist()):
                raise ValueError("source episode batch differs from registered source identities or typed conditions")
        original_parameters = tuple(parameter.detach().clone() for parameter in self.parameters)
        original_buffers = {name: buffer.detach().clone() for name, buffer in self.model.named_buffers()}
        original_optimizer = copy.deepcopy(optimizer.state_dict())
        before = RandomSnapshot.capture(random, self.cuda_devices)
        restore_random = before
        original_gradients = tuple(None for _ in self.parameters)
        active = bool(partition.episode.query_indices)
        virtual_parameter_hash = None
        virtual_optimizer_hash = None
        virtual_displacement = 0.0
        changed_parameters = 0
        support_norm = clipped_support_norm = 0.0
        query_old = query_virtual = tuple(None for _ in self.parameters)
        support_gradients = tuple(None for _ in self.parameters)
        successful = False
        try:
            objective, components = evaluate(partition)
            if objective.ndim != 0 or not torch.isfinite(objective):
                raise ValueError("source episode requires a finite scalar original objective")
            objective.backward(retain_graph=active)
            original_gradients = gradient_tuple(self.model)
            gradient_norm(original_gradients)
            restore_random = RandomSnapshot.capture(random, self.cuda_devices)
            if value_digest(tuple(self.parameters)) != value_digest(original_parameters) or value_digest(dict(self.model.named_buffers())) != value_digest(original_buffers):
                raise ValueError("source episode original forward/backward mutated parameters or buffers")
            if active:
                support_gradients = parameter_gradients(partition.data_loss(query=False), self.parameters, retain_graph=True)
                query_old = parameter_gradients(partition.data_loss(query=True), self.parameters, retain_graph=False)
                if value_digest(gradient_tuple(self.model)) != value_digest(original_gradients) or not restore_random.equivalent(RandomSnapshot.capture(random, self.cuda_devices)):
                    raise ValueError("source episode task autograd changed original gradients or RNG")
                assign_gradients(self.parameters, support_gradients)
                support_norm = gradient_norm(support_gradients)
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
                clipped_support_norm = gradient_norm(gradient_tuple(self.model))
                optimizer.step()
                virtual_parameter_hash = value_digest(tuple(self.parameters))
                virtual_optimizer_hash = value_digest(optimizer.state_dict())
                virtual_displacement = gradient_norm(tuple(parameter.detach().double() - saved.double() for parameter, saved in zip(self.parameters, original_parameters, strict=True)))
                changed_parameters = sum(not torch.equal(parameter, saved) for parameter, saved in zip(self.parameters, original_parameters, strict=True))
                optimizer.zero_grad(set_to_none=True)
                before.restore(random)
                virtual_partition = EpisodeObjectivePartition(self.registry, cells, step)
                virtual_objective, virtual_components = evaluate(virtual_partition)
                if virtual_partition.episode != partition.episode or virtual_partition.eligible_order_sample_count != partition.eligible_order_sample_count:
                    raise ValueError("source episode virtual forward changed its original partition or denominator")
                query_virtual = parameter_gradients(virtual_partition.data_loss(query=True), self.parameters, retain_graph=False)
                if not restore_random.equivalent(RandomSnapshot.capture(random, self.cuda_devices)):
                    raise ValueError("source episode replay did not consume the original random stream exactly once")
                if value_digest(dict(self.model.named_buffers())) != value_digest(original_buffers):
                    raise ValueError("source episode virtual forward mutated model buffers")
                del virtual_objective, virtual_components, virtual_partition
            successful = True
        finally:
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, original_parameters, strict=True):
                    if not torch.equal(parameter, saved):
                        parameter.copy_(saved)
                current_buffers = dict(self.model.named_buffers())
                for name, saved in original_buffers.items():
                    if not torch.equal(current_buffers[name], saved):
                        current_buffers[name].copy_(saved)
            if value_digest(optimizer.state_dict()) != value_digest(original_optimizer):
                optimizer.load_state_dict(original_optimizer)
            assign_gradients(self.parameters, original_gradients)
            restore_random.restore(random)
        if not successful:
            raise RuntimeError("source episode did not complete its protected temporary calculation")
        outer = replace_query_gradients(original_gradients, query_old, query_virtual) if active else original_gradients
        assign_gradients(self.parameters, outer)
        changed_gradients = sum((first is None) != (second is None) or (first is not None and not torch.equal(first, second))
                                for first, second in zip(original_gradients, outer, strict=True))
        differences = tuple((torch.zeros_like(second, dtype=torch.float64) if first is None else first.double()) -
                             (torch.zeros_like(first, dtype=torch.float64) if second is None else second.double())
                             for first, second in zip(original_gradients, outer, strict=True) if first is not None or second is not None)
        row = {"step": step, "source_cells": [list(cell) for cell in cells], "source_routes": [cell[0] for cell in cells],
               "partition": json.loads(json.dumps(asdict(partition.episode))), "samples": len(cells), "support_samples": len(partition.episode.support_indices),
               "query_samples": len(partition.episode.query_indices), "eligible_order_samples": partition.eligible_order_sample_count,
               "partition_scalar_error": partition.reconstruction_error, "active_steps": int(active), "empty_steps": int(not active),
               "virtual_optimizer_calls": int(active), "extra_objective_calls": int(active), "extra_autograd_calls": 3 * int(active),
               "joint_backward_calls": 1, "changed_virtual_parameter_tensors": changed_parameters,
               "changed_outer_gradient_tensors": changed_gradients, "virtual_displacement_norm": virtual_displacement,
               "outer_replacement_norm": gradient_norm(differences), "base_gradient_norm": gradient_norm(original_gradients),
               "outer_gradient_norm": gradient_norm(outer), "support_gradient_norm": support_norm, "clipped_support_gradient_norm": clipped_support_norm,
               "base_gradient_sha256": value_digest(original_gradients), "support_gradient_sha256": value_digest(support_gradients),
               "query_original_gradient_sha256": value_digest(query_old), "query_virtual_gradient_sha256": value_digest(query_virtual),
               "outer_gradient_sha256": value_digest(outer), "parameters_before_sha256": value_digest(original_parameters),
               "parameters_virtual_sha256": virtual_parameter_hash, "parameters_restored_sha256": value_digest(tuple(self.parameters)),
               "optimizer_before_sha256": value_digest(original_optimizer), "optimizer_virtual_sha256": virtual_optimizer_hash,
               "optimizer_restored_sha256": value_digest(optimizer.state_dict()), "buffers_before_sha256": value_digest(original_buffers),
               "buffers_restored_sha256": value_digest(dict(self.model.named_buffers())),
               "rng_before_sha256": value_digest(before.explicit), "rng_after_original_sha256": value_digest(restore_random.explicit),
               "rng_restored_to_single_original_pass": restore_random.equivalent(RandomSnapshot.capture(random, self.cuda_devices)),
               "replayed_rng_matches_original_pass": True, "samples_by_device": {str(key): value for key, value in sorted(Counter(cell[0] for cell in cells).items())},
               "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
        metrics = self.activity.record(row)
        return objective.detach(), {**components, **metrics}
