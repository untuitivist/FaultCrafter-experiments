from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import torch


PERMUTED_BATCH_STRATA = "permuted_batch_strata_v1"


def time_sampling_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("training_time_sampling_policy", "independent_uniform")
    if policy not in ("independent_uniform", PERMUTED_BATCH_STRATA):
        raise ValueError("unknown training time sampling policy")
    if policy == PERMUTED_BATCH_STRATA:
        lower, upper = float(generator["minimum_time"]), float(generator["maximum_time"])
        if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper < 1:
            raise ValueError("stratified time requires a finite nonempty interval inside [0,1)")
        if generator.get("relation_coupled_noise_time", False) or generator.get("relation_balanced_sampling", False):
            raise ValueError("batch stratification cannot replace relation-coupled times")
    return policy


def stratum_edges(batch_size: int, lower: float, upper: float, device: torch.device) -> torch.Tensor:
    return lower + (upper - lower) * torch.arange(batch_size + 1, dtype=torch.float64, device=device) / batch_size


def times_from_strata(jitter: torch.Tensor, permutation: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if jitter.ndim != 1 or not len(jitter) or jitter.dtype != torch.float32 or permutation.shape != jitter.shape or permutation.dtype != torch.long or permutation.device != jitter.device:
        raise ValueError("stratification requires aligned nonempty float32 jitter and integer permutation")
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper < 1:
        raise ValueError("invalid stratification interval")
    if not bool(((jitter >= 0) & (jitter < 1)).all()) or not torch.equal(permutation.sort().values, torch.arange(len(jitter), device=jitter.device)):
        raise ValueError("invalid uniform draws or stratum permutation")
    edges = stratum_edges(len(jitter), lower, upper, jitter.device)
    left = edges[permutation]
    right = edges[permutation + 1]
    first = left.float()
    last = right.float()
    first = torch.where(first.double() < left, torch.nextafter(first, torch.full_like(first, math.inf)), first)
    last = torch.where(last.double() >= right, torch.nextafter(last, torch.full_like(last, -math.inf)), last)
    if bool((first > last).any()):
        raise ValueError("a time stratum has no representable float32 interior")
    values = (lower + (upper - lower) * (permutation.double() + jitter.double()) / len(jitter)).float()
    return torch.maximum(first, torch.minimum(last, values))


def sample_training_times(batch_size: int, generator: Mapping[str, Any], *, compute_device: torch.device, random: torch.Generator) -> torch.Tensor:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("time sampling requires a positive integer batch size")
    lower, upper = float(generator["minimum_time"]), float(generator["maximum_time"])
    if time_sampling_policy(generator) == "independent_uniform":
        return torch.empty(batch_size, device=compute_device).uniform_(lower, upper, generator=random)
    jitter = torch.rand(batch_size, dtype=torch.float32, device=compute_device, generator=random)
    permutation = torch.randperm(batch_size, device=compute_device, generator=random) if batch_size > 1 else torch.zeros(1, dtype=torch.long, device=compute_device)
    return times_from_strata(jitter, permutation, lower, upper)


def observe_stratified_times(generator: Mapping[str, Any], time: torch.Tensor, device_id: torch.Tensor) -> dict:
    if time_sampling_policy(generator) != PERMUTED_BATCH_STRATA:
        return {}
    if time.ndim != 1 or not len(time) or time.dtype != torch.float32 or device_id.shape != time.shape or device_id.dtype != torch.long or device_id.device != time.device:
        raise ValueError("time observations require aligned float32 times and device routes")
    lower, upper = float(generator["minimum_time"]), float(generator["maximum_time"])
    edges = stratum_edges(len(time), lower, upper, time.device)
    strata = torch.searchsorted(edges[1:-1], time.detach().double(), right=True)
    if not bool(((time.double() >= lower) & (time.double() < upper)).all()) or not torch.equal(strata.sort().values, torch.arange(len(time), device=time.device)):
        raise ValueError("actual training times do not occupy each half-open stratum exactly once")
    values, routes = time.detach().cpu().tolist(), device_id.detach().cpu().tolist()
    if any(route < 0 for route in routes):
        raise ValueError("negative device route")
    return {"time_stratification_batch_times": values, "time_stratification_batch_routes": routes,
            "time_stratification_batch_strata": strata.cpu().tolist(), "time_stratification_batch_samples": len(values)}


class TimeStratificationActivity:
    def __init__(self, generator: Mapping[str, Any], device_ids: Sequence[str]):
        if time_sampling_policy(generator) != PERMUTED_BATCH_STRATA or not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("stratification activity requires explicit policy and unique devices")
        self.generator = dict(generator)
        self.device_ids = tuple(device_ids)
        self.steps = 0
        self.batch_sizes = Counter()
        self.counts = {device: {"samples": 0, "time_sum": 0.0, "inverse_noise_weight_sum": 0.0, "ten_band_counts": [0] * 10} for device in device_ids}
        self.minimum, self.maximum = math.inf, -math.inf

    def metadata(self) -> dict:
        return {"policy": PERMUTED_BATCH_STRATA, "minimum_time": self.generator["minimum_time"], "maximum_time": self.generator["maximum_time"],
                "formula": "lower_plus_range_times_permuted_stratum_plus_uniform_jitter_over_batch_size",
                "finite_precision": "float64_coordinates_float32_half_open_interior_clamp", "batch_one": "one_jitter_no_permutation_draw",
                "random_stream": "original_main_torch_generator_additional_randperm_changes_future_noise", "source_sampling_changed": False,
                "source_validation": "same_stratification_original_per_batch_seeds", "loss_weighting_changed": False,
                "device_ids": list(self.device_ids), "device_balancing": False, "observations_change_training": False,
                "extra_runtime_parameters_or_files": False}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != self.steps + 1:
            raise ValueError("stratification must observe each consecutive update once")
        values = components["time_stratification_batch_times"]
        routes = components["time_stratification_batch_routes"]
        strata = components["time_stratification_batch_strata"]
        lower, upper = float(self.generator["minimum_time"]), float(self.generator["maximum_time"])
        if not values or components["time_stratification_batch_samples"] != len(values) or len(routes) != len(values) or sorted(strata) != list(range(len(values))):
            raise ValueError("incomplete batch time observations")
        if any(not math.isfinite(value) or not lower <= value < upper for value in values) or any(type(route) is not int or not 0 <= route < len(self.device_ids) for route in routes):
            raise ValueError("invalid time observations or device routing")
        for value, route in zip(values, routes, strict=True):
            bucket = self.counts[self.device_ids[route]]
            bucket["samples"] += 1
            bucket["time_sum"] += value
            bucket["inverse_noise_weight_sum"] += 1 / max(1 - value, float(self.generator["minimum_noise_fraction"])) ** 2
            bucket["ten_band_counts"][min(9, int((value - lower) / (upper - lower) * 10))] += 1
        self.minimum, self.maximum = min(self.minimum, min(values)), max(self.maximum, max(values))
        self.steps = step
        self.batch_sizes[len(values)] += 1
        return self.diagnostics()

    def diagnostics(self) -> dict:
        return {"time_stratification_total_steps": self.steps, "time_stratification_total_samples": sum(bucket["samples"] for bucket in self.counts.values()),
                "time_stratification_batch_size_counts": {str(size): count for size, count in sorted(self.batch_sizes.items())},
                "time_stratification_observed_minimum": self.minimum, "time_stratification_observed_maximum": self.maximum,
                "time_stratification_device_counts": {device: {**bucket, "ten_band_counts": list(bucket["ten_band_counts"])} for device, bucket in self.counts.items()}}
