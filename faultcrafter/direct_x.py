from __future__ import annotations

import torch
from torch import nn

from faultcrafter.time_weighting import DirectXTimeWeighting
from faultcrafter.robust_reconstruction import SCALED_HUBER_VELOCITY, SQUARED_RESIDUAL


def linear_noise_path(
    clean_signal: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Interpolate from noise at t=0 to data at t=1."""

    if clean_signal.shape != noise.shape:
        raise ValueError("clean signal and noise shapes differ")
    if time.shape != (len(clean_signal),):
        raise ValueError("time must have shape [batch]")
    if torch.any((time < 0.0) | (time > 1.0)):
        raise ValueError("time must lie in [0,1]")
    return time[:, None, None] * clean_signal + (1.0 - time[:, None, None]) * noise


def direct_x_loss(
    prediction: torch.Tensor,
    clean_signal: torch.Tensor,
    time: torch.Tensor,
    *,
    minimum_noise_fraction: float,
    weighting: DirectXTimeWeighting | None = None,
    residual_policy: str = SQUARED_RESIDUAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """JiT-style clean-signal loss and its unweighted reconstruction error."""

    if prediction.shape != clean_signal.shape:
        raise ValueError("prediction and clean signal shapes differ")
    if time.shape != (len(prediction),):
        raise ValueError("time must have shape [batch]")
    if not 0.0 < minimum_noise_fraction <= 1.0:
        raise ValueError("minimum_noise_fraction must be in (0,1]")
    if residual_policy not in (SQUARED_RESIDUAL, SCALED_HUBER_VELOCITY):
        raise ValueError("unknown direct-x residual policy")
    if residual_policy == SCALED_HUBER_VELOCITY and weighting is not None:
        raise ValueError("scaled Huber velocity loss cannot combine a different time weighting")
    clean_error = (
        (prediction - clean_signal).square().flatten(start_dim=1).mean(dim=1)
    )
    if weighting is None:
        noise_fraction = (1.0 - time).clamp_min(minimum_noise_fraction)
        if residual_policy == SCALED_HUBER_VELOCITY:
            error = prediction - clean_signal
            absolute_error = error.abs()
            threshold = noise_fraction.reshape((-1,) + (1,) * (prediction.ndim - 1))
            robust_error = torch.where(absolute_error <= threshold, error.square(), threshold * (2.0 * absolute_error - threshold))
            objective = robust_error.flatten(start_dim=1).mean(dim=1) / noise_fraction.square()
        else:
            objective = clean_error / noise_fraction.square()
    else:
        if weighting.minimum_noise_fraction != minimum_noise_fraction:
            raise ValueError("direct-x loss and time weighting noise floors differ")
        objective = clean_error * weighting.weights(time)
    return objective.mean(), clean_error.mean()


@torch.no_grad()
def heun_direct_x_sample(
    model: nn.Module,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
    time_embedding_scale: float,
    source_signal: torch.Tensor | None = None,
    source_condition_parameters: torch.Tensor | None = None,
    projected_condition_parameters: torch.Tensor | None = None,
    condition_residual_parameters: torch.Tensor | None = None,
    device_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the velocity implied by repeated clean-signal predictions."""

    if steps < 2:
        raise ValueError("Heun sampling requires at least two steps")
    if not 0.0 < terminal_time < 1.0:
        raise ValueError("terminal_time must be in (0,1)")
    if time_embedding_scale <= 0.0:
        raise ValueError("time_embedding_scale must be positive")
    state = initial_noise
    batch_size = len(state)
    times = torch.linspace(
        0.0,
        terminal_time,
        steps + 1,
        device=state.device,
        dtype=state.dtype,
    )
    def predict(current: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if source_signal is None:
            arguments = {}
            if device_id is not None:
                arguments["device_id"] = device_id
            return model(
                current,
                time * time_embedding_scale,
                condition_parameters,
                fault,
                **arguments,
            )
        if device_id is not None:
            raise ValueError("device_id is only supported for source-free sampling")
        return model(
            current,
            time * time_embedding_scale,
            condition_parameters,
            fault,
            source_signal=source_signal,
            source_condition_parameters=source_condition_parameters,
            projected_condition_parameters=projected_condition_parameters,
            condition_residual_parameters=condition_residual_parameters,
        )

    for index in range(steps):
        time = times[index].expand(batch_size)
        next_time = times[index + 1].expand(batch_size)
        step_size = times[index + 1] - times[index]
        prediction = predict(state, time)
        velocity = (prediction - state) / (1.0 - time[:, None, None])
        proposal = state + step_size * velocity
        next_prediction = predict(proposal, next_time)
        next_velocity = (next_prediction - proposal) / (
            1.0 - next_time[:, None, None]
        )
        state = state + 0.5 * step_size * (velocity + next_velocity)
    terminal = times[-1].expand(batch_size)
    return predict(state, terminal)


@torch.no_grad()
def heun_direct_x_inpaint_sample(
    model: nn.Module,
    initial_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    known_clean_prefix: torch.Tensor,
    known_noise_prefix: torch.Tensor,
    *,
    steps: int,
    terminal_time: float,
    time_embedding_scale: float,
) -> torch.Tensor:
    """Sample a window while preserving a known clean/noise prefix trajectory."""

    if steps < 2:
        raise ValueError("Heun sampling requires at least two steps")
    if not 0.0 < terminal_time < 1.0:
        raise ValueError("terminal_time must be in (0,1)")
    if time_embedding_scale <= 0.0:
        raise ValueError("time_embedding_scale must be positive")
    if known_clean_prefix.shape != known_noise_prefix.shape:
        raise ValueError("known clean and noise prefixes differ in shape")
    if known_clean_prefix.shape[:-1] != initial_noise.shape[:-1]:
        raise ValueError("known prefix batch or channel dimensions differ")
    prefix_length = known_clean_prefix.shape[-1]
    if prefix_length <= 0 or prefix_length >= initial_noise.shape[-1]:
        raise ValueError("known prefix length must lie inside the sampled window")
    if condition_parameters.shape[0] != initial_noise.shape[0]:
        raise ValueError("condition batch size differs from initial noise")
    if fault.shape != (initial_noise.shape[0],):
        raise ValueError("fault must have shape [batch]")

    batch_size = len(initial_noise)
    times = torch.linspace(
        0.0,
        terminal_time,
        steps + 1,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
    )

    def clamp_state(current: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        constrained = current.clone()
        constrained[..., :prefix_length] = (
            time[:, None, None] * known_clean_prefix
            + (1.0 - time[:, None, None]) * known_noise_prefix
        )
        return constrained

    def predict(current: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        prediction = model(
            current,
            time * time_embedding_scale,
            condition_parameters,
            fault,
        )
        prediction = prediction.clone()
        prediction[..., :prefix_length] = known_clean_prefix
        return prediction

    state = initial_noise.clone()
    state[..., :prefix_length] = known_noise_prefix
    for index in range(steps):
        time = times[index].expand(batch_size)
        next_time = times[index + 1].expand(batch_size)
        step_size = times[index + 1] - times[index]
        state = clamp_state(state, time)
        prediction = predict(state, time)
        velocity = (prediction - state) / (1.0 - time[:, None, None])
        proposal = clamp_state(state + step_size * velocity, next_time)
        next_prediction = predict(proposal, next_time)
        next_velocity = (next_prediction - proposal) / (
            1.0 - next_time[:, None, None]
        )
        state = clamp_state(
            state + 0.5 * step_size * (velocity + next_velocity),
            next_time,
        )
    terminal = times[-1].expand(batch_size)
    return predict(state, terminal)


@torch.no_grad()
def autoregressive_overlap_direct_x_sample(
    model: nn.Module,
    continuous_noise: torch.Tensor,
    condition_parameters: torch.Tensor,
    fault: torch.Tensor,
    *,
    window_length: int,
    stride: int,
    steps: int,
    terminal_time: float,
    time_embedding_scale: float,
) -> torch.Tensor:
    """Generate one continuous signal by overlap-constrained JiT continuation."""

    if window_length <= 0 or stride <= 0 or stride >= window_length:
        raise ValueError("stride must lie in [1, window_length)")
    total_length = continuous_noise.shape[-1]
    if total_length < window_length:
        raise ValueError("continuous noise is shorter than one window")
    if (total_length - window_length) % stride != 0:
        raise ValueError("requested length is not aligned to window and stride")
    overlap = window_length - stride
    first_noise = continuous_noise[..., :window_length]
    first = heun_direct_x_sample(
        model,
        first_noise,
        condition_parameters,
        fault,
        steps=steps,
        terminal_time=terminal_time,
        time_embedding_scale=time_embedding_scale,
    )
    pieces = [first]
    previous_window = first
    for offset in range(stride, total_length - window_length + 1, stride):
        window_noise = continuous_noise[..., offset : offset + window_length]
        current = heun_direct_x_inpaint_sample(
            model,
            window_noise,
            condition_parameters,
            fault,
            previous_window[..., -overlap:],
            window_noise[..., :overlap],
            steps=steps,
            terminal_time=terminal_time,
            time_embedding_scale=time_embedding_scale,
        )
        pieces.append(current[..., overlap:])
        previous_window = current
    generated = torch.cat(pieces, dim=-1)
    if generated.shape != continuous_noise.shape:
        raise RuntimeError("overlap continuation produced an unexpected shape")
    return generated
