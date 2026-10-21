from __future__ import annotations

import copy
import math
from typing import Any

import torch

from faultcrafter.initial_anchor import parameter_fingerprint
from faultcrafter.source_semantics import SourceSemanticTeachers
from faultcrafter.unified_generator import sample_unified_generator


SOURCE_GENERATION_VALIDATION = "source_generation_semantic_v1"
GENERATION_VALIDATION_WEIGHT = 0.05
GENERATION_VALIDATION_SEED_OFFSET = 62_000_000


def generation_validation_policy(config: dict) -> str:
    generator = config["generator"]
    policy = generator.get("source_generation_validation_policy", "none")
    if not isinstance(policy, str) or policy not in {"none", SOURCE_GENERATION_VALIDATION}:
        raise ValueError("unknown source generation validation policy")
    if policy == "none":
        return policy
    defaults = {"initial_parameter_anchor_policy": "none", "initial_prediction_anchor_policy": "none", "trajectory_average_policy": "ema",
                "shared_gradient_policy": "joint", "sharpness_training_policy": "standard", "direct_x_time_weighting": "inverse_noise_squared",
                "normalized_energy_policy": "none", "source_semantic_input_mode": "denoising_prediction"}
    if any(generator.get(name, default) != default for name, default in defaults.items()):
        raise ValueError("source generation validation cannot combine undeclared training mechanisms")
    if any(generator.get(name, False) for name in ("resumable_training", "initial_validation_fallback", "device_balanced_sampling", "device_balanced_validation",
           "relation_balanced_sampling", "relation_coupled_noise_time", "detach_group_guidance_reference", "preserve_frozen_ema")):
        raise ValueError("source generation validation requires original full-model data and checkpoint rules")
    if generator.get("active_lora_device_index") is not None or any(float(generator.get(name, 0)) != 0 for name in ("source_semantic_loss_weight", "fault_contrast_loss_weight")):
        raise ValueError("source generation validation must not add training teachers or restricted updates")
    if config.get("normalization_method") != "max_abs" or generator.get("affine_restoration", False) or generator.get("fault_identity_guidance_mode") != "linear":
        raise ValueError("source generation validation requires the declared non-affine linear generator")
    generation = config["generation"]
    if generation["sampling_steps"] != 12 or generation["terminal_time"] != 0.95 or generation["target_length"] != 51802 or not generation["stochastic_amplitude"]:
        raise ValueError("source generation validation requires the original complete sampling protocol")
    return policy


class SourceGenerationValidator:
    def __init__(self, config: dict, teachers: SourceSemanticTeachers):
        if generation_validation_policy(config) != SOURCE_GENERATION_VALIDATION:
            raise ValueError("source generation validation requires its explicit policy")
        self.sampling = copy.deepcopy(config["generation"])
        self.teachers = teachers
        self.teacher_hashes = self._teacher_hashes()
        self.evaluations: list[dict[str, Any]] = []
        self.current_step: int | None = None
        self.batches: list[dict[str, Any]] = []

    def _teacher_hashes(self) -> list[str]:
        if any(classifier.training or any(parameter.requires_grad for parameter in classifier.parameters()) for classifier in self.teachers.classifiers):
            raise ValueError("source validation teachers must stay frozen in eval mode")
        return [parameter_fingerprint(classifier.state_dict()) for classifier in self.teachers.classifiers]

    def verify_teachers_unchanged(self) -> None:
        if self._teacher_hashes() != self.teacher_hashes:
            raise RuntimeError("source validation teacher weights changed")

    def metadata(self) -> dict:
        return {"policy": SOURCE_GENERATION_VALIDATION, "weight": GENERATION_VALIDATION_WEIGHT,
                "seed_offset": GENERATION_VALIDATION_SEED_OFFSET, "sampling_steps": self.sampling["sampling_steps"],
                "terminal_time": self.sampling["terminal_time"], "source_validation_lengths": "original_complete_parent_panel",
                "score": "original_data_objective_plus_weight_times_confidence_semantic_kl",
                "training_loss_changed": False, "teacher_use": "source_validation_only_no_gradient",
                "teacher_state_sha256": list(self.teacher_hashes), "canonical_sampler": "sample_unified_generator",
                "time_gate": "none_completed_rollout", "extra_runtime_parameters_or_files": False}

    def begin(self, step: int) -> None:
        if self.current_step is not None or type(step) is not int or step <= 0 or (self.evaluations and step <= self.evaluations[-1]["step"]):
            raise ValueError("source generation validation evaluations must be complete and ordered")
        self.current_step = step
        self.batches = []

    @torch.no_grad()
    def score_batch(self, model, batch, data_objective: float, *, seed: int, compute_device: torch.device) -> float:
        if self.current_step is None or model.training or batch.source_record_index is None:
            raise ValueError("source generation scoring requires active validation and source record routing")
        length = int(batch.normalized_signal.shape[-1])
        if any(int(value) != length for value in batch.normalization_length) or not math.isfinite(data_objective):
            raise ValueError("source generation validation requires complete parents and finite data objective")
        random = torch.Generator(device=compute_device).manual_seed(seed + GENERATION_VALIDATION_SEED_OFFSET)
        clean = torch.from_numpy(batch.normalized_signal).to(compute_device)
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=compute_device, generator=random)
        amplitude_noise = torch.randn(len(clean), dtype=clean.dtype, device=compute_device, generator=random)
        condition = torch.from_numpy(batch.condition_parameters).to(compute_device)
        fault = torch.from_numpy(batch.fault).to(compute_device)
        group = torch.from_numpy(batch.fault_group).to(compute_device)
        device_id = torch.from_numpy(batch.device_id).to(compute_device)
        speed = torch.from_numpy(batch.shaft_speed_hz).to(compute_device)
        sample_rate = torch.from_numpy(batch.sample_rate_hz).to(compute_device)
        prediction, _, _ = sample_unified_generator(model, noise, condition, fault, group, device_id, speed, sample_rate,
            steps=self.sampling["sampling_steps"], terminal_time=self.sampling["terminal_time"], amplitude_noise=amplitude_noise)
        semantic, diagnostics = self.teachers.objective(prediction, clean, fault, device_id,
            torch.from_numpy(batch.source_record_index).to(compute_device), torch.from_numpy(batch.amplitude_scale).to(compute_device),
            torch.zeros(len(clean), dtype=clean.dtype, device=compute_device), prediction_normalization="generated_record")
        semantic_value = float(semantic)
        if not math.isfinite(semantic_value) or semantic_value < 0:
            raise FloatingPointError("source generation semantic score must be finite and nonnegative")
        self.batches.append({"seed": seed + GENERATION_VALIDATION_SEED_OFFSET, "length": length, "batch_size": len(clean),
            "device_indices": batch.device_id.tolist(), "fault_indices": batch.fault.tolist(),
            "source_record_indices": batch.source_record_index.tolist(), "source_parent_ids": [str(value) for value in batch.parent_segment_id],
            "no_grad_forwards": 2 * self.sampling["sampling_steps"] + 1, "gradient_forwards": 0,
            "data_objective": data_objective, "semantic_loss": semantic_value,
            "source_confidence_mean": diagnostics["source_semantic_confidence_mean"], "source_correct_fraction": diagnostics["source_semantic_correct_fraction"]})
        return data_objective + GENERATION_VALIDATION_WEIGHT * semantic_value

    def finish(self, score: float) -> dict:
        if self.current_step is None or not self.batches:
            raise ValueError("source generation validation cannot finish an empty or inactive evaluation")
        count = sum(row["batch_size"] for row in self.batches)
        data = sum(row["data_objective"] * row["batch_size"] for row in self.batches) / count
        semantic = sum(row["semantic_loss"] * row["batch_size"] for row in self.batches) / count
        if not math.isfinite(score) or not math.isclose(score, data + GENERATION_VALIDATION_WEIGHT * semantic, rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError("source generation validation score differs from original sample-weighted aggregation")
        row = {"step": self.current_step, "data_objective": data, "semantic_loss": semantic, "score": score,
               "generated_source_samples": count, "rollout_batches": len(self.batches), "no_grad_forwards": sum(item["no_grad_forwards"] for item in self.batches),
               "gradient_forwards": 0, "batches": self.batches}
        self.evaluations.append(row)
        self.current_step = None
        return {"source_generation_validation_" + key: value for key, value in row.items() if key != "batches"}

    def audit(self) -> dict:
        return {**self.metadata(), "source_teachers": self.teachers.audit, "evaluations": self.evaluations,
                "target_waveforms_used": False, "teacher_weights_saved": False,
                "generated_source_samples": sum(row["generated_source_samples"] for row in self.evaluations),
                "no_grad_forwards": sum(row["no_grad_forwards"] for row in self.evaluations), "gradient_forwards": 0}
