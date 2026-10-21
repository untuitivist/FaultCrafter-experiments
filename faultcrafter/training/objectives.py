from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from faultcrafter.shared_gradient import DeviceObjectivePartition, validate_partition_recipe
from faultcrafter.condition_gradient import ConditionObjectivePartition
from faultcrafter.episode_adaptation import SOURCE_EPISODE_ADAMW, EpisodeObjectivePartition, episode_adaptation_policy
from faultcrafter.physical_reference import LEGACY_CYCLIC_REFERENCE
from faultcrafter.condition_guidance import condition_slope_penalty
from faultcrafter.direct_x import direct_x_loss, linear_noise_path
from faultcrafter.time_weighting import resolve_time_weighting
from faultcrafter.waveform_energy import normalized_energy_weight, normalized_log_rms_loss
from faultcrafter.waveform_distribution import SORTED_WAVEFORM_L1, sorted_waveform_l1, waveform_distribution_metadata, waveform_distribution_policy
from faultcrafter.multiresolution_spectrum import MULTIRESOLUTION_STFT, multiresolution_spectral_errors, multiresolution_spectral_metadata, spectral_reconstruction_policy
from faultcrafter.normalized_spectrum import NORMALIZED_SPECTRAL_DISTANCE, normalized_spectral_errors, normalized_spectral_metadata
from faultcrafter.order_resolution import RESOLUTION_AWARE_ORDER, OrderResolutionBatch, order_resolution_policy
from faultcrafter.low_signal_time import low_signal_training_times
from faultcrafter.time_guidance import CONSTANT_GUIDANCE, time_guidance_diagnostics, time_guidance_metadata, time_guidance_policy
from faultcrafter.time_coverage import observe_training_times
from faultcrafter.time_stratification import observe_stratified_times, sample_training_times, time_sampling_policy
from faultcrafter.robust_reconstruction import SCALED_HUBER_VELOCITY, observe_robust_reconstruction, robust_reconstruction_policy
from faultcrafter.reconstruction_priority import RECONSTRUCTION_PRIORITY, ReconstructionPriority, auxiliary_gradient_policy
from faultcrafter.antithetic_noise import ANTITHETIC_PAIR, AntitheticNoisePair, noise_pair_policy
from faultcrafter.paired_time import REFLECTED_RAW_TIME, PairedTime, paired_time_metadata, paired_time_policy
from faultcrafter.prediction_anchor import INITIAL_PREDICTION_ANCHOR, InitialPredictionAnchor, prediction_anchor_policy
from faultcrafter.fault_contrast import SourceFaultContrastRoutes, source_spectral_contrast
from faultcrafter.relative_energy import RELATIVE_ENERGY_POLICIES, relative_energy_losses, relative_energy_metadata, relative_energy_policy
from faultcrafter.solver_state import DETACHED_SOLVER_STATE, SourceSolverStep, solver_state_policy
from faultcrafter.condition_contrast import SOURCE_CONDITION_CONTRAST, PREFIX as CONDITION_CONTRAST_PREFIX, SourceConditionContrastRoutes, condition_contrast_loss, condition_contrast_metadata, condition_contrast_policy
from faultcrafter.source_semantics import SourceSemanticTeachers
from faultcrafter.source_rollout_semantics import SEMANTIC_INPUT_MODES, SourceRolloutSampling, source_rollout_prediction
from faultcrafter.jit import normalized_log_order_envelope, normalized_log_power_spectrum
from faultcrafter.shared_update_cone import SHARED_UPDATE_CONE, SharedUpdatePartition, shared_update_metadata, shared_update_policy
from faultcrafter.style_transport import four_cell_relation_transport_losses, paired_relative_transport_losses
from faultcrafter.unified_generator import UnifiedSignalGenerator


@dataclass(frozen=True, slots=True)
class UnifiedJointBatch:
    normalized_signal: np.ndarray
    amplitude_scale: np.ndarray
    location_offset: np.ndarray
    condition_parameters: np.ndarray
    fault: np.ndarray
    fault_group: np.ndarray
    device_id: np.ndarray
    shaft_speed_hz: np.ndarray
    sample_rate_hz: np.ndarray
    normalization_length: np.ndarray
    parent_segment_id: np.ndarray
    source_record_index: np.ndarray | None = None


def amplitude_nll(
    log_amplitude: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
) -> torch.Tensor:
    if log_amplitude.shape != mean.shape or mean.shape != log_scale.shape:
        raise ValueError("amplitude distribution tensors differ in shape")
    inverse_scale = torch.exp(-log_scale)
    standardized = (log_amplitude - mean) * inverse_scale
    return (0.5 * standardized.square() + log_scale).mean()


def multiscale_style_content_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-dimensional AdaAttN/AdaIN analogues on fixed signal features."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("style/content loss signals must have matching [B,C,T] shape")
    global_terms: list[torch.Tensor] = []
    local_terms: list[torch.Tensor] = []
    content_terms: list[torch.Tensor] = []
    length = prediction.shape[-1]
    for requested_kernel in (8, 32, 128):
        kernel = min(requested_kernel, length)
        predicted_mean = F.avg_pool1d(prediction, kernel, stride=kernel)
        target_mean = F.avg_pool1d(target, kernel, stride=kernel)
        predicted_energy = F.avg_pool1d(
            prediction.square(), kernel, stride=kernel
        ).add(1e-6).sqrt()
        target_energy = F.avg_pool1d(
            target.square(), kernel, stride=kernel
        ).add(1e-6).sqrt()
        predicted_features = torch.cat((predicted_mean, predicted_energy), dim=1)
        target_features = torch.cat((target_mean, target_energy), dim=1)

        predicted_global_mean = predicted_features.mean(dim=-1)
        target_global_mean = target_features.mean(dim=-1)
        predicted_global_std = predicted_features.var(
            dim=-1, unbiased=False
        ).add(1e-6).sqrt()
        target_global_std = target_features.var(
            dim=-1, unbiased=False
        ).add(1e-6).sqrt()
        global_terms.append(
            F.smooth_l1_loss(predicted_global_mean, target_global_mean)
            + F.smooth_l1_loss(predicted_global_std, target_global_std)
        )

        feature_length = predicted_features.shape[-1]
        local_kernel = min(8, feature_length)
        predicted_local_mean = F.avg_pool1d(
            predicted_features, local_kernel, stride=local_kernel
        )
        target_local_mean = F.avg_pool1d(
            target_features, local_kernel, stride=local_kernel
        )
        predicted_local_std = F.avg_pool1d(
            predicted_features.square(), local_kernel, stride=local_kernel
        ).sub(predicted_local_mean.square()).clamp_min(1e-6).sqrt()
        target_local_std = F.avg_pool1d(
            target_features.square(), local_kernel, stride=local_kernel
        ).sub(target_local_mean.square()).clamp_min(1e-6).sqrt()
        local_terms.append(
            F.smooth_l1_loss(predicted_local_mean, target_local_mean)
            + F.smooth_l1_loss(predicted_local_std, target_local_std)
        )

        predicted_content = (
            predicted_features - predicted_global_mean[:, :, None]
        ) / predicted_global_std[:, :, None]
        target_content = (
            target_features - target_global_mean[:, :, None]
        ) / target_global_std[:, :, None]
        content_terms.append(F.smooth_l1_loss(predicted_content, target_content))
    return (
        torch.stack(global_terms).mean(),
        torch.stack(local_terms).mean(),
        torch.stack(content_terms).mean(),
    )


def _fixed_style_analysis_features(signal: torch.Tensor) -> torch.Tensor:
    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError("fixed style analysis expects [B,1,T]")
    length = signal.shape[-1]
    features: list[torch.Tensor] = []
    for requested_kernel in (3, 9, 33, 129):
        kernel = min(requested_kernel, length if length % 2 else length - 1)
        kernel = max(kernel, 1)
        padding = kernel // 2
        smoothed = F.avg_pool1d(
            F.pad(signal, (padding, padding), mode="replicate"),
            kernel,
            stride=1,
        )
        detail = signal - smoothed
        envelope = F.avg_pool1d(
            F.pad(detail.square(), (padding, padding), mode="replicate"),
            kernel,
            stride=1,
        ).add(1e-6).sqrt()
        features.extend((detail, envelope))
    return torch.cat(features, dim=1)


def disentangled_style_content_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition_parameters: torch.Tensor,
    device_id: torch.Tensor,
    fault: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AdaIN statistics, local AdaAttN statistics, and content preservation."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("style/content loss signals must have matching [B,C,T] shape")
    batch = prediction.shape[0]
    if condition_parameters.ndim != 2 or condition_parameters.shape[0] != batch:
        raise ValueError("condition parameters differ from style/content batch")
    for name, value in (("device", device_id), ("fault", fault)):
        if value.shape != (batch,):
            raise ValueError(f"{name} labels differ from style/content batch")

    predicted_features = _fixed_style_analysis_features(prediction)
    target_features = _fixed_style_analysis_features(target)
    predicted_mean = predicted_features.mean(dim=-1)
    target_mean = target_features.mean(dim=-1)
    predicted_std = predicted_features.var(dim=-1, unbiased=False).add(1e-6).sqrt()
    target_std = target_features.var(dim=-1, unbiased=False).add(1e-6).sqrt()
    style_global = F.smooth_l1_loss(predicted_mean, target_mean) + F.smooth_l1_loss(
        predicted_std, target_std
    )

    segment_count = min(16, prediction.shape[-1])
    predicted_local_mean = F.adaptive_avg_pool1d(
        predicted_features, segment_count
    )
    target_local_mean = F.adaptive_avg_pool1d(target_features, segment_count)
    predicted_local_std = F.adaptive_avg_pool1d(
        predicted_features.square(), segment_count
    ).sub(predicted_local_mean.square()).clamp_min(1e-6).sqrt()
    target_local_std = F.adaptive_avg_pool1d(
        target_features.square(), segment_count
    ).sub(target_local_mean.square()).clamp_min(1e-6).sqrt()
    style_local = F.smooth_l1_loss(
        predicted_local_mean, target_local_mean
    ) + F.smooth_l1_loss(predicted_local_std, target_local_std)

    predicted_content = (predicted_features - predicted_mean[:, :, None]) / (
        predicted_std[:, :, None] + 1e-6
    )
    target_content = (target_features - target_mean[:, :, None]) / (
        target_std[:, :, None] + 1e-6
    )
    content = F.smooth_l1_loss(predicted_content, target_content)

    descriptor_prediction = torch.cat((predicted_mean, predicted_std), dim=1)
    descriptor_target = torch.cat((target_mean, target_std), dim=1)
    parameter_distance = torch.cdist(condition_parameters, condition_parameters)
    pair_mask = (
        (device_id[:, None] == device_id[None, :])
        & (fault[:, None] == fault[None, :])
        & (parameter_distance > 1e-6)
        & torch.triu(
            torch.ones(batch, batch, dtype=torch.bool, device=prediction.device),
            diagonal=1,
        )
    )
    if torch.any(pair_mask):
        predicted_delta = (
            descriptor_prediction[:, None, :] - descriptor_prediction[None, :, :]
        )[pair_mask]
        target_delta = (
            descriptor_target[:, None, :] - descriptor_target[None, :, :]
        )[pair_mask]
        style_relation = F.smooth_l1_loss(predicted_delta, target_delta)
    else:
        style_relation = prediction.sum() * 0.0
    return style_global, style_local, style_relation, content


def relation_coupled_noise_and_time(
    clean: torch.Tensor,
    generator: dict[str, Any],
    *,
    random: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give each ordered four-cell relation one nuisance realization."""
    batch = len(clean)
    if batch == 0 or batch % 4:
        raise ValueError("coupled relation batch must be a non-empty multiple of four")
    group_count = batch // 4
    group_noise = torch.randn(
        (group_count, *clean.shape[1:]),
        device=clean.device,
        dtype=clean.dtype,
        generator=random,
    )
    group_time = torch.empty(
        group_count,
        device=clean.device,
        dtype=clean.dtype,
    ).uniform_(
        float(generator["minimum_time"]),
        float(generator["maximum_time"]),
        generator=random,
    )
    return (
        torch.repeat_interleave(group_noise, 4, dim=0),
        torch.repeat_interleave(group_time, 4, dim=0),
    )


def batch_objective(
    model: UnifiedSignalGenerator,
    batch: UnifiedJointBatch,
    generator: dict[str, Any],
    *,
    compute_device: torch.device,
    random: torch.Generator,
    normalization_method: str = "max_abs",
    condition_projectors: torch.Tensor | None = None,
    fault_contrast_routes: SourceFaultContrastRoutes | None = None,
    training_step: int | None = None,
    source_semantic_teachers: SourceSemanticTeachers | None = None,
    rollout_sampling: SourceRolloutSampling | None = None,
    gradient_partition: DeviceObjectivePartition | ConditionObjectivePartition | None = None,
    prediction_reference: InitialPredictionAnchor | None = None,
    reconstruction_priority: ReconstructionPriority | None = None,
    antithetic_noise: AntitheticNoisePair | None = None,
    paired_time: PairedTime | None = None,
    condition_contrast_routes: SourceConditionContrastRoutes | None = None,
    solver_step: SourceSolverStep | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    guidance_schedule = time_guidance_policy(generator)
    needs_time_pair = paired_time_policy(generator) == REFLECTED_RAW_TIME and model.training and torch.is_grad_enabled()
    if needs_time_pair != (paired_time is not None):
        raise ValueError("paired time requires its recipe only during active student training")
    if paired_time is not None:
        paired_time_metadata(generator)
        if any(value is not None for value in (gradient_partition, prediction_reference, reconstruction_priority, antithetic_noise, solver_step)):
            raise ValueError("paired time cannot combine undeclared training mechanisms")
    if getattr(model, "fault_identity_guidance_schedule", CONSTANT_GUIDANCE) != guidance_schedule:
        raise ValueError("training and validation must use the model's declared identity guidance schedule")
    if guidance_schedule != CONSTANT_GUIDANCE:
        time_guidance_metadata(generator)
    needs_episode = episode_adaptation_policy(generator) == SOURCE_EPISODE_ADAMW and model.training and torch.is_grad_enabled()
    if needs_episode != isinstance(gradient_partition, EpisodeObjectivePartition):
        raise ValueError("source episode requires its partition only in active student training")
    needs_shared_update = shared_update_policy(generator) == SHARED_UPDATE_CONE and model.training and torch.is_grad_enabled()
    if needs_shared_update != isinstance(gradient_partition, SharedUpdatePartition):
        raise ValueError("shared update cone requires its device partition only during active student training")
    if needs_shared_update:
        shared_update_metadata(generator)
    needs_relative_energy = relative_energy_policy(generator) in RELATIVE_ENERGY_POLICIES and model.training and torch.is_grad_enabled()
    if needs_relative_energy:
        relative_energy_metadata(generator)
        if normalization_method != "max_abs" or any(value is not None for value in (gradient_partition, prediction_reference, reconstruction_priority, antithetic_noise, solver_step)):
            raise ValueError("relative source energy requires the declared max_abs reconstruction without undeclared gradient mechanisms")
    needs_solver_step = solver_state_policy(generator) == DETACHED_SOLVER_STATE and model.training and torch.is_grad_enabled()
    if needs_solver_step != (solver_step is not None):
        raise ValueError("source solver recipe is required only during active solver state training")
    if needs_solver_step and (gradient_partition is not None or prediction_reference is not None or reconstruction_priority is not None or antithetic_noise is not None):
        raise ValueError("solver state cannot combine undeclared gradient mechanisms")
    needs_condition_contrast = condition_contrast_policy(generator) == SOURCE_CONDITION_CONTRAST and model.training and torch.is_grad_enabled()
    if needs_condition_contrast != (condition_contrast_routes is not None):
        raise ValueError("source condition routes are required only during active contrast training")
    if needs_condition_contrast and ((gradient_partition is not None and not isinstance(gradient_partition, SharedUpdatePartition)) or prediction_reference is not None or reconstruction_priority is not None or antithetic_noise is not None):
        raise ValueError("condition contrast cannot combine undeclared gradient mechanisms")
    needs_noise_pair = noise_pair_policy(generator) == ANTITHETIC_PAIR and model.training and torch.is_grad_enabled()
    if needs_noise_pair != (antithetic_noise is not None):
        raise ValueError("antithetic noise requires its pair only during active student training")
    if antithetic_noise is not None and (gradient_partition is not None or prediction_reference is not None or reconstruction_priority is not None):
        raise ValueError("antithetic noise cannot combine undeclared gradient mechanisms")
    needs_priority = auxiliary_gradient_policy(generator) == RECONSTRUCTION_PRIORITY and model.training and torch.is_grad_enabled()
    if needs_priority != (reconstruction_priority is not None):
        raise ValueError("reconstruction priority requires its capture only during active student training")
    if reconstruction_priority is not None and (gradient_partition is not None or prediction_reference is not None):
        raise ValueError("reconstruction priority cannot combine a different gradient partition or prediction anchor")
    prediction_policy = prediction_anchor_policy(generator)
    needs_prediction_reference = prediction_policy == INITIAL_PREDICTION_ANCHOR and model.training and torch.is_grad_enabled()
    if needs_prediction_reference != (prediction_reference is not None):
        raise ValueError("prediction anchor requires a reference only in active student training, never source validation")
    if gradient_partition is not None:
        validate_partition_recipe(model, generator, normalization_method, within_device_conditions=isinstance(gradient_partition, ConditionObjectivePartition), for_shared_update=isinstance(gradient_partition, SharedUpdatePartition), for_episode=isinstance(gradient_partition, EpisodeObjectivePartition))
    time_weighting = resolve_time_weighting(generator)
    residual_policy = robust_reconstruction_policy(generator)
    if residual_policy == SCALED_HUBER_VELOCITY and gradient_partition is not None:
        raise ValueError("robust reconstruction cannot use a squared-loss gradient partition")
    energy_weight = normalized_energy_weight(generator)
    if energy_weight is not None and (normalization_method != "max_abs" or model.affine_restoration or model.fault_identity_guidance_mode != "unit_peak_projection"):
        raise ValueError("normalized energy requires the full-parent max_abs unit peak projection path")
    if getattr(model, "fault_identity_guidance_mode", "linear") == "unit_peak_projection" and normalization_method != "max_abs":
        raise ValueError("unit peak projection requires max_abs training targets")
    detach_reference = generator.get("detach_group_guidance_reference", False)
    if type(detach_reference) is not bool:
        raise ValueError("group guidance reference detachment must be a boolean")
    if detach_reference:
        declared_group_weight = float(generator.get("group_reconstruction_loss_weight", 0.0))
        if not np.isfinite(declared_group_weight) or declared_group_weight <= 0.0:
            raise ValueError("detached guidance requires the separate trainable group reconstruction objective")
    clean = torch.from_numpy(batch.normalized_signal).to(compute_device)
    parameters = torch.from_numpy(batch.condition_parameters).to(compute_device)
    fault = torch.from_numpy(batch.fault).to(compute_device)
    fault_group = torch.from_numpy(batch.fault_group).to(compute_device)
    device_id = torch.from_numpy(batch.device_id).to(compute_device)
    speed = torch.from_numpy(batch.shaft_speed_hz).to(compute_device)
    reference_policy = generator.get("physics_reference_policy", LEGACY_CYCLIC_REFERENCE)
    if reference_policy != model.physics_reference_policy:
        raise ValueError("training and model physical reference policies differ")
    native_speed = speed
    speed, mechanical_present = model.physical_shaft_reference(parameters, speed)
    effective_speeds = speed.detach().cpu().numpy() if mechanical_present is not None else batch.shaft_speed_hz
    sample_rate = torch.from_numpy(batch.sample_rate_hz).to(compute_device)
    normalization_length = torch.from_numpy(batch.normalization_length).to(
        compute_device
    )
    time_sampling_policy(generator)
    coupled_relations = bool(generator.get("relation_coupled_noise_time", False))
    if paired_time is not None and paired_time.stage != 0:
        noise, time = paired_time.replay(clean, random)
    elif antithetic_noise is not None and antithetic_noise.stage != 0:
        noise, time = antithetic_noise.replay(clean, random)
    elif coupled_relations:
        if not bool(generator.get("relation_balanced_sampling", False)):
            raise ValueError("coupled relation noise requires relation sampling")
        noise, time = relation_coupled_noise_and_time(
            clean,
            generator,
            random=random,
        )
    else:
        noise = torch.randn(clean.shape, device=compute_device, generator=random)
        time = sample_training_times(len(clean), generator, compute_device=compute_device, random=random)
    if antithetic_noise is not None and antithetic_noise.stage == 0:
        antithetic_noise.capture(clean, noise, time, random)
    if paired_time is not None and paired_time.stage == 0:
        paired_time.capture(clean, noise, time, random)
    time, low_signal_diagnostics = low_signal_training_times(generator, time, noise, device_id, random,
        active=bool(model.training and torch.is_grad_enabled()))
    state = linear_noise_path(clean, noise, time)
    output = model(
        state,
        time,
        parameters,
        fault,
        fault_group,
        device_id,
        speed,
        sample_rate,
        normalization_length,
        **({"detach_group_guidance_reference": True} if detach_reference else {}),
    )
    if detach_reference and (not isinstance(output.group_guidance_reference_detached, torch.Tensor)
                             or output.group_guidance_reference_detached.numel() != 1
                             or float(output.group_guidance_reference_detached) != 1.0):
        raise RuntimeError("training requested detached group guidance but forward did not execute it")
    prediction = output.normalized_signal
    condition_contrast_diagnostics = {}
    if needs_condition_contrast:
        contrast_metadata = condition_contrast_metadata(generator)
        negative_parameters, negative_speed, condition_eligible, condition_contrast_diagnostics = condition_contrast_routes.select(batch, training_step)
        condition_contrast_value = prediction.sum() * 0.0
        condition_contrast_diagnostics.update({CONDITION_CONTRAST_PREFIX + "executed": 0.0, CONDITION_CONTRAST_PREFIX + "loss": 0.0,
                                             CONDITION_CONTRAST_PREFIX + "positive_errors": [], CONDITION_CONTRAST_PREFIX + "negative_errors": [], CONDITION_CONTRAST_PREFIX + "times": []})
        if np.any(condition_eligible):
            eligible_conditions = torch.from_numpy(condition_eligible).to(compute_device)
            def predict_negative_condition(*arguments):
                return model(*arguments).normalized_signal
            negative_prediction = activation_checkpoint(predict_negative_condition,
                state[eligible_conditions], time[eligible_conditions], torch.from_numpy(negative_parameters[condition_eligible]).to(compute_device),
                fault[eligible_conditions], fault_group[eligible_conditions], device_id[eligible_conditions],
                torch.from_numpy(negative_speed[condition_eligible]).to(compute_device), sample_rate[eligible_conditions], normalization_length[eligible_conditions],
                use_reentrant=False)
            contrast_capture = [] if isinstance(gradient_partition, SharedUpdatePartition) else None
            condition_contrast_value, contrast_metrics = condition_contrast_loss(prediction[eligible_conditions], negative_prediction, clean[eligible_conditions], time[eligible_conditions],
                **({"sample_capture": contrast_capture} if contrast_capture is not None else {}))
            if contrast_capture is not None:
                gradient_partition.capture_contrast(device_id[eligible_conditions], contrast_capture[0], contrast_metadata["weight"])
            condition_contrast_diagnostics.update(contrast_metrics)
    solver_diagnostics = {}
    if solver_step is not None:
        proposal, next_time, solver_diagnostics = solver_step.propose(state, prediction, time, device_id)
        def predict_future_state(*arguments):
            return model(*arguments).normalized_signal
        future_prediction = activation_checkpoint(predict_future_state, proposal, next_time, parameters, fault, fault_group,
            device_id, speed, sample_rate, normalization_length, use_reentrant=False)
        future_loss, future_diagnostics = solver_step.loss(future_prediction, clean, next_time)
        solver_diagnostics.update(future_diagnostics)
    contrast_weight = float(generator.get("fault_contrast_loss_weight", 0.0))
    if not np.isfinite(contrast_weight) or contrast_weight < 0.0:
        raise ValueError("fault contrast weight must be finite and nonnegative")
    contrast_loss = prediction.sum() * 0.0
    contrast_diagnostics = {}
    if contrast_weight and model.training:
        if fault_contrast_routes is None or training_step is None:
            raise ValueError("fault contrast training requires source routes and the actual training step")
        negative_fault, negative_group, eligible = fault_contrast_routes.negatives(fault, device_id, training_step)
        if not torch.equal(fault_group, fault_contrast_routes.groups[fault]):
            raise ValueError("source fault and group routes disagree")
        contrast_diagnostics["fault_contrast_eligible_fraction"] = float(eligible.float().mean())
        contrast_diagnostics["fault_contrast_correct_fraction"] = 0.0
        if torch.any(eligible):
            def predict_negative(*arguments):
                return model(*arguments).normalized_signal
            negative_prediction = activation_checkpoint(
                predict_negative, state[eligible], time[eligible], parameters[eligible],
                negative_fault[eligible], negative_group[eligible], device_id[eligible],
                speed[eligible], sample_rate[eligible], normalization_length[eligible],
                use_reentrant=False,
            )
            contrast_loss, positive_error, negative_error = source_spectral_contrast(
                prediction[eligible], negative_prediction, clean[eligible], time[eligible]
            )
            contrast_diagnostics["fault_contrast_correct_fraction"] = float((positive_error.detach() < negative_error.detach()).float().mean())
        contrast_diagnostics["fault_contrast_loss"] = float(contrast_loss.detach())
    semantic_weight = float(generator.get("source_semantic_loss_weight", 0.0))
    if not np.isfinite(semantic_weight) or semantic_weight < 0.0:
        raise ValueError("source semantic weight must be finite and nonnegative")
    semantic_loss = prediction.sum() * 0.0
    semantic_diagnostics = {}
    semantic_mode = generator.get("source_semantic_input_mode", "denoising_prediction")
    if semantic_mode not in SEMANTIC_INPUT_MODES:
        raise ValueError("unknown source semantic input mode")
    if semantic_weight and model.training:
        semantic_prediction, semantic_time = prediction, time
        prediction_normalization = "source_parent"
        execute_semantic = True
        if semantic_mode == "source_rollout":
            if rollout_sampling is None or rollout_sampling.interval != generator.get("source_semantic_rollout_interval"):
                raise ValueError("source rollout requires the matching full generation recipe")
            execute_semantic = rollout_sampling.due(training_step)
            semantic_diagnostics["source_semantic_rollout_executed"] = 0.0
        if execute_semantic:
            if normalization_method != "max_abs" or source_semantic_teachers is None or batch.source_record_index is None:
                raise ValueError("source semantic training requires source record routing and max_abs teachers")
            if source_semantic_teachers.audit.get("pooling_policy", "native") != generator.get("source_semantic_pooling_policy", "native"):
                raise ValueError("source semantic teacher pooling differs from declared training recipe")
            if semantic_mode == "source_rollout":
                semantic_prediction, semantic_diagnostics = source_rollout_prediction(
                    model, parameters, fault, fault_group, device_id, speed, sample_rate,
                    length=clean.shape[-1], sampling=rollout_sampling,
                    base_seed=random.initial_seed(), training_step=training_step,
                )
                semantic_time = torch.zeros_like(time)
                prediction_normalization = "generated_record"
            semantic_loss, teacher_diagnostics = source_semantic_teachers.objective(
                semantic_prediction, clean, fault, device_id,
                torch.from_numpy(batch.source_record_index).to(compute_device),
                torch.from_numpy(batch.amplitude_scale).to(compute_device), semantic_time,
                prediction_normalization=prediction_normalization,
            )
            semantic_diagnostics.update(teacher_diagnostics)
    waveform, clean_mse = direct_x_loss(
        prediction,
        clean,
        time,
        minimum_noise_fraction=float(generator["minimum_noise_fraction"]),
        weighting=time_weighting,
        **({"residual_policy": residual_policy} if residual_policy == SCALED_HUBER_VELOCITY else {}),
    )
    group_weight = float(generator.get("group_reconstruction_loss_weight", 0.0))
    slope_weight = float(generator.get("condition_slope_regularization_weight", 0.0))
    if min(group_weight, slope_weight) < 0.0:
        raise ValueError("compact guidance loss weights must be non-negative")
    group_waveform = prediction.sum() * 0.0
    if group_weight:
        if output.group_normalized_signal is None:
            raise ValueError("group reconstruction requires trained identity guidance")
        if detach_reference and model.training and torch.is_grad_enabled() and not output.group_normalized_signal.requires_grad:
            raise RuntimeError("detached guidance must retain the group reconstruction gradient")
        group_waveform, _ = direct_x_loss(
            output.group_normalized_signal, clean, time,
            minimum_noise_fraction=float(generator["minimum_noise_fraction"]),
            weighting=time_weighting,
            **({"residual_policy": residual_policy} if residual_policy == SCALED_HUBER_VELOCITY else {}),
        )
    relative_energy_diagnostics = {}
    if needs_relative_energy:
        if output.group_normalized_signal is None:
            raise ValueError("relative source energy requires the original group prediction")
        waveform, group_waveform, relative_energy_diagnostics = relative_energy_losses(
            prediction, output.group_normalized_signal, clean, time, device_id, policy=relative_energy_policy(generator))
    if reconstruction_priority is not None:
        reconstruction_priority.capture(waveform, group_waveform, group_weight)
    condition_slope = prediction.sum() * 0.0
    if slope_weight:
        if condition_projectors is None:
            raise ValueError("condition slope regularization requires source projectors")
        condition_slope = condition_slope_penalty(
            model, condition_projectors,
            device_index=generator.get("active_lora_device_index"),
        )
    spectral_diagnostics = {}
    if spectral_reconstruction_policy(generator) == MULTIRESOLUTION_STFT:
        if normalization_method != "max_abs":
            raise ValueError("multiresolution source supervision requires complete-parent max_abs normalization")
        multiresolution_spectral_metadata(float(generator["spectral_loss_weight"]))
        spectral_errors, spectral_diagnostics = multiresolution_spectral_errors(prediction, clean)
    elif spectral_reconstruction_policy(generator) == NORMALIZED_SPECTRAL_DISTANCE:
        if normalization_method != "max_abs":
            raise ValueError("normalized spectral distance requires complete-parent max_abs normalization")
        normalized_spectral_metadata(float(generator["spectral_loss_weight"]))
        spectral_errors, spectral_diagnostics = normalized_spectral_errors(prediction, clean)
    else:
        spectral_errors = (
            normalized_log_power_spectrum(prediction)
            - normalized_log_power_spectrum(clean)
        ).square()
    spectral = spectral_errors.mean()
    order_terms: list[torch.Tensor] = []
    order_numerators: dict[int, torch.Tensor] = {}
    condition_order_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    eligible_count = 0
    unknown_order_count = 0
    minimum_rotations = float(generator["minimum_order_rotations"])
    order_resolution = None
    if order_resolution_policy(generator) == RESOLUTION_AWARE_ORDER:
        order_resolution = OrderResolutionBatch(generator, prediction.shape[-1], len(prediction))
    for device_index in np.unique(batch.device_id):
        indices = np.flatnonzero(batch.device_id == device_index)
        rotations = (
            prediction.shape[-1]
            * effective_speeds[indices]
            / batch.sample_rate_hz[indices]
        )
        eligible = rotations >= minimum_rotations
        if mechanical_present is not None:
            eligible &= mechanical_present.detach().cpu().numpy()[indices] & (effective_speeds[indices] > 0)
        indices = indices[eligible]
        if not len(indices):
            continue
        selected = torch.from_numpy(indices).to(compute_device)
        if mechanical_present is not None:
            unknown_order_count += int((~mechanical_present.index_select(0, selected)).sum())
        selected_speed = speed.index_select(0, selected)
        predicted_order = normalized_log_order_envelope(
            prediction.index_select(0, selected),
            selected_speed,
            sample_rate_hz=float(batch.sample_rate_hz[indices[0]]),
            maximum_order=float(generator["maximum_order"]),
            feature_count=int(generator["order_feature_count"]),
        )
        clean_order = normalized_log_order_envelope(
            clean.index_select(0, selected),
            selected_speed,
            sample_rate_hz=float(batch.sample_rate_hz[indices[0]]),
            maximum_order=float(generator["maximum_order"]),
            feature_count=int(generator["order_feature_count"]),
        )
        sample_order_errors = 1.0 - (predicted_order * clean_order).sum(dim=1)
        if order_resolution is not None:
            sample_order_errors = order_resolution.weight(sample_order_errors, rotations[eligible], indices, int(device_index))
        order_terms.append(sample_order_errors.sum())
        if gradient_partition is not None:
            order_numerators[int(device_index)] = order_terms[-1]
            if isinstance(gradient_partition, ConditionObjectivePartition):
                condition_order_samples.append((torch.as_tensor(indices, device=compute_device, dtype=torch.long), sample_order_errors))
        eligible_count += len(indices)
    order = (
        torch.stack(order_terms).sum() / eligible_count
        if order_terms
        else prediction.sum() * 0.0
    )
    log_amplitude = torch.log(
        torch.from_numpy(batch.amplitude_scale).to(compute_device)
    )
    amplitude = amplitude_nll(
        log_amplitude,
        output.log_amplitude_mean,
        output.log_amplitude_log_scale,
    )
    if model.affine_restoration and normalization_method == "mean_ac_rms":
        if output.location_mean is None or output.location_log_scale is None:
            raise RuntimeError("affine restoration outputs are missing")
        location_target = torch.from_numpy(batch.location_offset).to(compute_device)
        location = amplitude_nll(
            location_target,
            output.location_mean,
            output.location_log_scale,
        )
        max_abs = prediction.sum() * 0.0
    else:
        location = prediction.sum() * 0.0
        target_maximum = clean.abs().amax(dim=(1, 2))
        predicted_maximum = prediction.abs().amax(dim=(1, 2))
        max_abs = F.smooth_l1_loss(predicted_maximum, target_maximum)
    style_global_weight = float(generator.get("style_global_loss_weight", 0.0))
    style_local_weight = float(generator.get("style_local_loss_weight", 0.0))
    style_relation_weight = float(
        generator.get("style_relation_loss_weight", 0.0)
    )
    content_weight = float(generator.get("content_loss_weight", 0.0))
    if style_global_weight or style_local_weight or style_relation_weight or content_weight:
        loss_kind = str(generator.get("style_content_loss", "waveform_statistics"))
        if loss_kind == "fixed_filter_cross_attention":
            (
                style_global,
                style_local,
                style_relation,
                content_loss,
            ) = disentangled_style_content_losses(
                prediction,
                clean,
                parameters,
                device_id,
                fault,
            )
        elif loss_kind == "waveform_statistics":
            style_global, style_local, content_loss = multiscale_style_content_losses(
                prediction, clean
            )
            style_relation = prediction.sum() * 0.0
        else:
            raise ValueError(f"unsupported style/content loss: {loss_kind}")
    else:
        style_global = prediction.sum() * 0.0
        style_local = prediction.sum() * 0.0
        style_relation = prediction.sum() * 0.0
        content_loss = prediction.sum() * 0.0
    condition_style_weight = float(
        generator.get("condition_style_transport_loss_weight", 0.0)
    )
    condition_slope_weight = float(
        generator.get("condition_slope_transport_loss_weight", 0.0)
    )
    fault_transport_weight = float(
        generator.get("fault_content_transport_loss_weight", 0.0)
    )
    difference_weight = float(
        generator.get("difference_of_differences_loss_weight", 0.0)
    )
    fault_margin_weight = float(
        generator.get("fault_separation_margin_loss_weight", 0.0)
    )
    four_cell_objective = coupled_relations or difference_weight or fault_margin_weight
    if four_cell_objective:
        if not bool(generator.get("relation_balanced_sampling", False)):
            raise ValueError("four-cell objective requires relation sampling")
        paired = four_cell_relation_transport_losses(
            prediction,
            clean,
            parameters,
            device_id,
            fault,
            fault_separation_reference_ratio=float(
                generator.get("fault_separation_reference_ratio", 0.75)
            ),
        )
    elif condition_style_weight or condition_slope_weight or fault_transport_weight:
        paired = paired_relative_transport_losses(
            prediction,
            clean,
            parameters,
            device_id,
            fault,
        )
    else:
        zero = prediction.sum() * 0.0
        paired = {
            "condition_style": zero,
            "fault_content": zero,
            "transport_slope": zero,
            "difference_of_differences": zero,
            "fault_separation_margin": zero,
            "condition_count": zero,
            "identifiable_pair_count": zero,
            "relation_group_count": zero,
        }
    objective = (
        waveform
        + float(generator["spectral_loss_weight"]) * spectral
        + float(generator["order_loss_weight"]) * order
        + float(generator["amplitude_loss_weight"]) * amplitude
        + float(generator.get("location_loss_weight", 0.0)) * location
        + float(generator.get("max_abs_loss_weight", 0.0)) * max_abs
        + style_global_weight * style_global
        + style_local_weight * style_local
        + style_relation_weight * style_relation
        + content_weight * content_loss
        + condition_style_weight * paired["condition_style"]
        + condition_slope_weight * paired["transport_slope"]
        + fault_transport_weight * paired["fault_content"]
        + difference_weight * paired["difference_of_differences"]
        + fault_margin_weight * paired["fault_separation_margin"]
        + group_weight * group_waveform
        + slope_weight * condition_slope
    )
    if contrast_diagnostics:
        objective = objective + contrast_weight * contrast_loss
    if semantic_diagnostics:
        objective = objective + semantic_weight * semantic_loss
    if condition_contrast_diagnostics:
        objective = objective + contrast_metadata["weight"] * condition_contrast_value
    if solver_diagnostics:
        objective = objective + solver_step.weight * future_loss
    distribution_diagnostics = {}
    if waveform_distribution_policy(generator) == SORTED_WAVEFORM_L1 and model.training and torch.is_grad_enabled():
        distribution_metadata = waveform_distribution_metadata(generator)
        distribution_loss, distribution_diagnostics = sorted_waveform_l1(prediction, clean, device_id)
        objective = objective + distribution_metadata["weight"] * distribution_loss
    energy_diagnostics = {}
    if energy_weight is not None:
        energy_loss, energy_diagnostics = normalized_log_rms_loss(prediction, clean)
        objective = objective + energy_weight * energy_loss
        energy_diagnostics.update({"normalized_energy_policy_executed": 1.0, "normalized_energy_weight": energy_weight})
    if gradient_partition is not None:
        noise_fraction = (1.0 - time).clamp_min(float(generator["minimum_noise_fraction"]))
        sample_losses = (prediction - clean).square().flatten(1).mean(dim=1) / noise_fraction.square()
        if group_weight:
            sample_losses = sample_losses + group_weight * (output.group_normalized_signal - clean).square().flatten(1).mean(dim=1) / noise_fraction.square()
        sample_spectral = spectral_errors.flatten(1).mean(dim=1)
        sample_amplitude = 0.5 * ((log_amplitude - output.log_amplitude_mean) * torch.exp(-output.log_amplitude_log_scale)).square() + output.log_amplitude_log_scale
        sample_maximum = F.smooth_l1_loss(prediction.abs().amax(dim=(1, 2)), clean.abs().amax(dim=(1, 2)), reduction="none")
        sample_losses = (sample_losses + float(generator["spectral_loss_weight"]) * sample_spectral
                         + float(generator["amplitude_loss_weight"]) * sample_amplitude
                         + float(generator.get("max_abs_loss_weight", 0.0)) * sample_maximum)
        gradient_partition.collect(device_id, sample_losses, condition_order_samples if isinstance(gradient_partition, ConditionObjectivePartition) else order_numerators, eligible_order_samples=eligible_count,
                                   order_weight=float(generator["order_loss_weight"]), regularizer=slope_weight * condition_slope, objective=objective,
                                   **({"extra_device_losses": gradient_partition.extra_device_losses} if isinstance(gradient_partition, SharedUpdatePartition) else {}))
    time_diagnostics = {}
    if time_weighting is not None:
        weights = time_weighting.weights(time).detach()
        time_diagnostics = {"direct_x_time_weighting_executed": 1.0, "direct_x_time_weight_mean": float(weights.mean()),
                            "direct_x_time_weight_min": float(weights.min()), "direct_x_time_weight_max": float(weights.max()),
                            "direct_x_time_weight_normalizer": time_weighting.normalizer}
    components = {
        **time_diagnostics,
        **low_signal_diagnostics,
        **energy_diagnostics,
        **spectral_diagnostics,
        **(order_resolution.diagnostics(order) if order_resolution is not None else {}),
        **observe_training_times(generator, time, device_id),
        **observe_stratified_times(generator, time, device_id),
        **(observe_robust_reconstruction(generator, prediction, output.group_normalized_signal, clean, time, device_id) if residual_policy == SCALED_HUBER_VELOCITY else {}),
        "objective": float(objective.detach()),
        "waveform": float(waveform.detach()),
        "clean_mse": float(clean_mse.detach()),
        "group_waveform": float(group_waveform.detach()),
        "condition_slope_regularization": float(condition_slope.detach()),
        "spectral": float(spectral.detach()),
        "order": float(order.detach()),
        "order_eligible_fraction": eligible_count / len(clean),
        "amplitude_nll": float(amplitude.detach()),
        "location_nll": float(location.detach()),
        "max_abs": float(max_abs.detach()),
        "style_global": float(style_global.detach()),
        "style_local": float(style_local.detach()),
        "style_relation": float(style_relation.detach()),
        "content": float(content_loss.detach()),
        "condition_style_transport": float(paired["condition_style"].detach()),
        "condition_slope_transport": float(paired["transport_slope"].detach()),
        "fault_content_transport": float(paired["fault_content"].detach()),
        "difference_of_differences": float(
            paired["difference_of_differences"].detach()
        ),
        "fault_separation_margin": float(
            paired["fault_separation_margin"].detach()
        ),
        "relation_condition_count": int(paired["condition_count"].detach()),
        "relation_identifiable_pair_count": int(
            paired["identifiable_pair_count"].detach()
        ),
        "relation_group_count": int(paired["relation_group_count"].detach()),
    }
    if output.identity_guidance_multiplier is not None:
        multiplier = output.identity_guidance_multiplier.detach()
        if model.fault_identity_guidance_mode == "unit_peak_projection":
            components["identity_peak_projection_changed_fraction"] = float(((multiplier - 1.0).abs() > 1e-6).float().mean())
            components["identity_peak_projection_mean_multiplier"] = float(multiplier.mean())
            components["identity_peak_projection_max_error"] = float((prediction.detach().abs().flatten(1).amax(dim=1) - 1.0).abs().max())
        else:
            components["identity_rms_cap_fraction"] = float((multiplier < 1.0 - 1e-6).float().mean())
            components["identity_rms_cap_mean_multiplier"] = float(multiplier.mean())
    components.update(contrast_diagnostics)
    if guidance_schedule != CONSTANT_GUIDANCE:
        components.update(time_guidance_diagnostics(time, output.identity_time_guidance_weight, device_id,
                                                   active=model.training and torch.is_grad_enabled()))
    components.update(semantic_diagnostics)
    components.update(distribution_diagnostics)
    components.update(condition_contrast_diagnostics)
    components.update(solver_diagnostics)
    components.update(relative_energy_diagnostics)
    if mechanical_present is not None:
        components.update({"physics_known_mechanical_fraction": float(mechanical_present.float().mean()),
                           "physics_missing_reference_count": int((~mechanical_present).sum()),
                           "physics_proxy_suppressed_fraction": float(((~mechanical_present) & (native_speed != 0)).float().mean()),
                           "order_unknown_reference_selected_count": unknown_order_count})
    if detach_reference:
        components["group_guidance_reference_detached"] = float(output.group_guidance_reference_detached)
    if prediction_reference is not None:
        data_objective = objective
        penalty, diagnostics = prediction_reference.loss(prediction, time, (state, time, parameters, fault, fault_group, device_id, speed, sample_rate, normalization_length))
        objective = data_objective + penalty
        components.update(diagnostics)
        components["prediction_anchor_data_objective"] = float(data_objective.detach())
        components["objective"] = float(objective.detach())
    return objective, components
