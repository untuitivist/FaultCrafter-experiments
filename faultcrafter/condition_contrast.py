from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from faultcrafter.fault_contrast import source_spectral_contrast
from faultcrafter.physical_reference import KNOWN_MECHANICAL_REFERENCE, resolve_physical_reference
from faultcrafter.typed_conditions import TYPED_CONDITION_PROTOCOL, encode_model_conditions


SOURCE_CONDITION_CONTRAST = "same_fault_source_condition_spectral_v1"
PREFIX = "source_condition_contrast_"


def condition_contrast_policy(generator: Mapping[str, Any]) -> str:
    policy = generator.get("source_condition_contrast_policy", "none")
    if policy not in ("none", SOURCE_CONDITION_CONTRAST):
        raise ValueError("unknown source condition contrast policy")
    return policy


def condition_contrast_metadata(generator: Mapping[str, Any]) -> dict:
    if condition_contrast_policy(generator) != SOURCE_CONDITION_CONTRAST:
        raise ValueError("condition contrast metadata requires its active policy")
    weight = generator.get("condition_slope_regularization_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("condition contrast requires the existing positive finite condition slope weight")
    if generator.get("physics_reference_policy") != KNOWN_MECHANICAL_REFERENCE:
        raise ValueError("condition contrast requires typed known mechanical references")
    if any(generator.get(name, False) for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight", "relation_balanced_sampling", "relation_coupled_noise_time", "resumable_training", "device_balanced_sampling", "waveform_distribution_policy")):
        raise ValueError("condition contrast cannot combine undeclared auxiliary mechanisms")
    return {"policy": SOURCE_CONDITION_CONTRAST, "weight": float(weight), "weight_source": "existing_condition_slope_regularization_weight",
            "negatives": "other_source_condition_same_device_same_fine_fault_same_mask",
            "ordering": "sorted_device_global_fault_typed_coordinates",
            "offset": "1+(step+batch_position)%(source_condition_count-1)",
            "formula": "mean(softplus((positive_error-negative_error)/detach(positive_error+negative_error).clamp_min(1e-8))*(1-time)^2)",
            "error": "normalized_log_power_spectrum_mse", "source_validation": "unchanged_original_objective",
            "additional_random_draws": 0, "additional_source_waveforms": 0,
            "negative_forward_activation_checkpoint": "non_reentrant", "extra_runtime_parameters_or_files": False}


class SourceConditionContrastRoutes:
    def __init__(self, entries: Sequence[dict], records: Mapping[tuple[int, int], int]):
        self.entries = copy.deepcopy(list(entries))
        self.records = dict(records)
        if not self.entries or not self.records:
            raise ValueError("condition contrast requires nonempty eligible source metadata")
        keys = [(row["device"], row["fault"], tuple(row["parameters"])) for row in self.entries]
        if keys != sorted(set(keys)):
            raise ValueError("condition contrast entries must be sorted and unique")
        self.groups = {}
        for index, row in enumerate(self.entries):
            parameters = np.asarray(row["parameters"], dtype=np.float32)
            if parameters.shape != (8,) or not np.isfinite(parameters).all() or not np.isin(parameters[4:], [0, 1]).all():
                raise ValueError("condition contrast requires finite typed coordinates and independent binary masks")
            if any(type(row[name]) is not int or row[name] < 0 for name in ("device", "fault")):
                raise ValueError("condition contrast source device and fault indices must be nonnegative integers")
            speed, _ = resolve_physical_reference(torch.from_numpy(parameters[None]), torch.tensor([row["shaft_hz"]], dtype=torch.float32), KNOWN_MECHANICAL_REFERENCE)
            if float(speed[0]) != row["shaft_hz"]:
                raise ValueError("condition contrast cannot use a drive frequency as shaft Hz")
            key = (row["device"], row["fault"])
            self.groups.setdefault(key, []).append(index)
        for members in self.groups.values():
            if len({tuple(self.entries[index]["parameters"][4:]) for index in members}) != 1:
                raise ValueError("condition contrast cannot change availability masks within a fault")
        for (device, record), index in self.records.items():
            if type(device) is not int or type(record) is not int or min(device, record) < 0 or type(index) is not int or not 0 <= index < len(self.entries) or self.entries[index]["device"] != device:
                raise ValueError("condition contrast source record map is invalid")

    @classmethod
    def from_source_cells(cls, devices: Sequence[Any], cells: Sequence[tuple[int, int]]):
        if [device.index for device in devices] != list(range(len(devices))) or len(set(cells)) != len(cells):
            raise ValueError("condition contrast requires contiguous devices and unique source cells")
        entries, records = {}, {}
        for device_index, record_index in cells:
            device = devices[device_index]
            if device.condition_protocol != TYPED_CONDITION_PROTOCOL or device.model_fault_labels is not None:
                raise ValueError("condition contrast requires source typed single-fault labels")
            record = device.corpus.records[record_index]
            parameters = encode_model_conditions(device, record.parameters[None], 8)[0]
            native_hz = np.asarray(record.parameters[device.speed_parameter_index] * device.speed_to_hz, dtype=np.float32)
            speed, _ = resolve_physical_reference(torch.from_numpy(parameters[None]), torch.from_numpy(native_hz.reshape(1)), KNOWN_MECHANICAL_REFERENCE)
            fault = int(record.fault_id + device.fault_offset)
            key = (int(device_index), fault, tuple(parameters.tolist()))
            row = {"device": int(device_index), "fault": fault, "parameters": parameters.tolist(), "shaft_hz": float(speed[0])}
            if key in entries and entries[key] != row:
                raise ValueError("duplicate source conditions disagree on mechanical speed")
            entries[key] = row
            records[(int(device_index), int(record_index))] = key
        ordered = sorted(entries)
        indices = {key: index for index, key in enumerate(ordered)}
        return cls([entries[key] for key in ordered], {key: indices[value] for key, value in records.items()})

    def select(self, batch: Any, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        count = len(batch.device_id)
        if type(step) is not int or step < 1 or count < 1 or batch.source_record_index is None or batch.fault.shape != (count,):
            raise ValueError("condition contrast requires real source record indices, single-fault labels and a positive step")
        positive_ids, negative_ids, eligible = [], [], []
        for position, (device, record, fault, coordinates) in enumerate(zip(batch.device_id, batch.source_record_index, batch.fault, batch.condition_parameters, strict=True)):
            key = (int(device), int(record))
            if key not in self.records:
                raise ValueError("condition contrast received a non-source training record")
            positive = self.records[key]
            row = self.entries[positive]
            if row["fault"] != int(fault) or not np.array_equal(coordinates, np.asarray(row["parameters"], dtype=np.float32)):
                raise ValueError("condition contrast source identity or coordinates disagree")
            members = self.groups[(int(device), int(fault))]
            active = len(members) > 1
            offset = 1 + (step + position) % (len(members) - 1) if active else 0
            negative = members[(members.index(positive) + offset) % len(members)]
            positive_ids.append(positive)
            negative_ids.append(negative)
            eligible.append(active)
        selected = [self.entries[index] for index in negative_ids]
        diagnostics = {PREFIX + "positive_ids": positive_ids, PREFIX + "negative_ids": negative_ids,
                       PREFIX + "eligible": eligible, PREFIX + "routes": batch.device_id.tolist(),
                       PREFIX + "faults": batch.fault.tolist(), PREFIX + "source_records": batch.source_record_index.tolist(),
                       PREFIX + "samples": count, PREFIX + "elements": int(batch.normalized_signal.size),
                       PREFIX + "length": int(batch.normalized_signal.shape[-1])}
        return np.asarray([row["parameters"] for row in selected], dtype=np.float32), np.asarray([row["shaft_hz"] for row in selected], dtype=np.float32), np.asarray(eligible, dtype=bool), diagnostics

    def audit(self) -> dict:
        return {"entries": copy.deepcopy(self.entries), "records": [{"device": device, "record": record, "entry": entry} for (device, record), entry in sorted(self.records.items())],
                "source_training_metadata_only": True, "target_waveforms_or_labels_used": False, "extra_runtime_parameters_or_files": False}


def condition_contrast_loss(positive: torch.Tensor, negative: torch.Tensor, clean: torch.Tensor, time: torch.Tensor, *, sample_capture: list | None = None) -> tuple[torch.Tensor, dict]:
    loss, positive_error, negative_error = source_spectral_contrast(positive, negative, clean.detach(), time)
    if not torch.isfinite(loss) or not torch.isfinite(positive_error).all() or not torch.isfinite(negative_error).all():
        raise ValueError("condition contrast requires finite spectral errors")
    if sample_capture is not None:
        scale = (positive_error.detach() + negative_error.detach()).clamp_min(1e-8)
        per_sample = torch.nn.functional.softplus((positive_error - negative_error) / scale) * (1.0 - time).square()
        if sample_capture or not torch.equal(per_sample.mean(), loss):
            raise ValueError("condition contrast capture changed the original source objective")
        sample_capture.append(per_sample)
    return loss, {PREFIX + "executed": 1.0, PREFIX + "loss": float(loss.detach()),
                  PREFIX + "positive_errors": positive_error.detach().cpu().tolist(),
                  PREFIX + "negative_errors": negative_error.detach().cpu().tolist(), PREFIX + "times": time.detach().cpu().tolist()}


def validate_condition_contrast_row(row: dict) -> None:
    count, length, elements = (row.get(PREFIX + name) for name in ("samples", "length", "elements"))
    if any(type(value) is not int or value <= 0 for value in (count, length, elements)) or elements != count * length:
        raise ValueError("condition contrast source exposure is invalid")
    for name in ("positive_ids", "negative_ids", "routes", "faults", "source_records"):
        values = row.get(PREFIX + name, [])
        if len(values) != count or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("condition contrast source routes are invalid")
    eligible = row.get(PREFIX + "eligible", [])
    if len(eligible) != count or any(type(value) is not bool for value in eligible):
        raise ValueError("condition contrast eligibility is invalid")
    if any((positive != negative) != active for positive, negative, active in zip(row[PREFIX + "positive_ids"], row[PREFIX + "negative_ids"], eligible, strict=True)):
        raise ValueError("condition contrast did not select a different eligible condition")
    arrays = [np.asarray(row.get(PREFIX + name, []), dtype=np.float64) for name in ("positive_errors", "negative_errors", "times")]
    if any(values.shape != (sum(eligible),) or not np.isfinite(values).all() for values in arrays):
        raise ValueError("condition contrast error trace is invalid")
    positive, negative, times = arrays
    if np.any(positive < 0) or np.any(negative < 0) or np.any((times < 0) | (times > 1)):
        raise ValueError("condition contrast errors or times are outside bounds")
    loss = row.get(PREFIX + "loss", float("nan"))
    executed = row.get(PREFIX + "executed")
    if executed != float(bool(sum(eligible))) or isinstance(loss, bool) or not math.isfinite(loss) or loss < 0:
        raise ValueError("condition contrast activity is inconsistent")
    expected = float(np.mean(np.logaddexp(0, (positive - negative) / np.maximum(positive + negative, 1e-8)) * (1 - times) ** 2)) if sum(eligible) else 0.0
    if abs(loss - expected) > 8 * np.finfo(np.float32).eps * max(1.0, abs(expected)):
        raise ValueError("condition contrast loss differs from independent error arithmetic")


class ConditionContrastActivity:
    def __init__(self, generator: Mapping[str, Any]):
        self.metadata = condition_contrast_metadata(generator)
        self.rows = []
        self.totals = {"samples": 0, "elements": 0, "eligible_samples": 0, "nonzero_steps": 0, "loss_sum": 0.0}

    def record(self, components: dict, step: int) -> dict:
        if type(step) is not int or step != len(self.rows) + 1:
            raise ValueError("condition contrast requires every consecutive completed update")
        row = {"step": step, **copy.deepcopy({name: value for name, value in components.items() if name.startswith(PREFIX)})}
        validate_condition_contrast_row(row)
        self.rows.append(row)
        for name in ("samples", "elements"):
            self.totals[name] += row[PREFIX + name]
        self.totals["eligible_samples"] += sum(row[PREFIX + "eligible"])
        self.totals["nonzero_steps"] += int(row[PREFIX + "loss"] > 0)
        self.totals["loss_sum"] += row[PREFIX + "loss"]
        return {PREFIX + "total_steps": step, **{PREFIX + "total_" + name: value for name, value in self.totals.items()}}

    def audit(self) -> dict:
        return {"metadata": copy.deepcopy(self.metadata), "completed_steps": len(self.rows), "steps": copy.deepcopy(self.rows),
                "historical_predictions_or_student_gradients_reconstructed": False, "extra_runtime_parameters_or_files": False}
