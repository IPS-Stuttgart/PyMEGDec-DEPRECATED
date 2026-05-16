"""Sensor-level alpha movement trajectories for MEG trials."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pymegdec.alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_SENSOR_POSITION_UNIT,
    compute_alpha_analytic_window,
    count_trials,
    get_channel_names,
    get_channel_positions_mm,
    load_participant_data,
    project_sensor_positions,
    select_channels,
    write_alpha_metrics_csv,
)
from pymegdec.alpha_signal import get_data_field, get_time_vector, get_trial_signal

DEFAULT_SENSOR_PATTERN = r"^M"
DEFAULT_MOVEMENT_TIME_WINDOW = (-0.4, 0.8)
DEFAULT_TRAJECTORY_STEP_S = 0.02
POWER_EPSILON = 1e-12


@dataclass(frozen=True)
class AlphaMovementConfig:
    """Parameters for sensor-level alpha movement trajectory extraction."""

    location_pattern: str = DEFAULT_SENSOR_PATTERN
    time_window: tuple[float, float] = DEFAULT_MOVEMENT_TIME_WINDOW
    frequency_range: tuple[float, float] = DEFAULT_FREQUENCY_RANGE
    trajectory_step_s: float | None = DEFAULT_TRAJECTORY_STEP_S
    filter_order: int = 5
    sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT


@dataclass(frozen=True)
class _MovementContext:
    data: object
    trial_idx: int
    participant_id: object
    dataset: str
    config: AlphaMovementConfig


@dataclass(frozen=True)
class _MovementGeometry:
    channel_indices: np.ndarray
    channel_names: np.ndarray
    positions: np.ndarray
    projected_positions: np.ndarray


@dataclass
class _MovementState:
    first: dict | None = None
    previous: dict | None = None
    previous_time: float = np.nan


def _trial_label(data, trial_idx):
    try:
        trialinfo = np.asarray(get_data_field(data, "trialinfo")).ravel()
    except (KeyError, ValueError):
        return np.nan
    return trialinfo[trial_idx].item()


def _resolve_channel_indices(data, channel_indices, location_pattern):
    if channel_indices is None:
        channel_indices = select_channels(data, location_pattern)

    resolved = np.asarray(channel_indices, dtype=int)
    if resolved.size == 0:
        raise ValueError(f"No channels matched pattern: {location_pattern}")
    return resolved


def _sampling_rate(time_vector):
    diffs = np.diff(time_vector)
    if diffs.size == 0:
        raise ValueError("Time vector must contain at least two samples.")
    return float(1 / np.median(diffs))


def sample_time_indices(time_vector, time_window, trajectory_step_s):
    """Return time indices inside ``time_window`` sampled at ``trajectory_step_s``."""

    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")

    time_vector = np.asarray(time_vector, dtype=float).ravel()
    if time_vector.size < 2:
        raise ValueError("Time vector must contain at least two samples.")
    tolerance = max(abs(float(np.median(np.diff(time_vector)))) * 1e-6, 1e-12)
    window_indices = np.flatnonzero((time_vector >= start - tolerance) & (time_vector <= stop + tolerance))
    if window_indices.size == 0:
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    if trajectory_step_s is None:
        return window_indices
    if trajectory_step_s <= 0:
        raise ValueError("trajectory_step_s must be positive.")

    first_time = max(start, float(time_vector[window_indices[0]]))
    last_time = min(stop, float(time_vector[window_indices[-1]]))
    targets = np.arange(first_time, last_time + trajectory_step_s / 2, trajectory_step_s)
    sampled = [int(window_indices[np.argmin(np.abs(time_vector[window_indices] - target))]) for target in targets]
    return np.unique(sampled)


def _alpha_power(signal, time_vector, sample_indices, config):
    _sampling_rate(time_vector)
    alpha_window, window_indices = compute_alpha_analytic_window(signal, time_vector, config)
    relative_indices = np.array(
        [int(np.argmin(np.abs(window_indices - sample_index))) for sample_index in sample_indices],
        dtype=int,
    )
    return np.abs(np.take(alpha_window, relative_indices, axis=-1)) ** 2


def _spatial_concentration(weights):
    probabilities = weights / np.sum(weights)
    entropy = -float(np.sum(probabilities * np.log(probabilities + POWER_EPSILON)))
    max_entropy = np.log(weights.size)
    if max_entropy <= 0:
        return 1.0
    return float(1 - entropy / max_entropy)


def _movement_values(centroid, projected, first, previous, previous_time, time_s):
    if previous is None:
        return {
            "displacement_mm": 0.0,
            "projected_displacement_mm": 0.0,
            "speed_mm_per_s": np.nan,
            "projected_speed_mm_per_s": np.nan,
            "projected_direction_rad": np.nan,
        }

    dt = time_s - previous_time
    if dt <= 0:
        speed = np.nan
        projected_speed = np.nan
    else:
        speed = float(np.linalg.norm(centroid - previous["centroid"]) / dt)
        projected_speed = float(np.linalg.norm(projected - previous["projected"]) / dt)

    projected_step = projected - previous["projected"]
    return {
        "displacement_mm": float(np.linalg.norm(centroid - first["centroid"])),
        "projected_displacement_mm": float(np.linalg.norm(projected - first["projected"])),
        "speed_mm_per_s": speed,
        "projected_speed_mm_per_s": projected_speed,
        "projected_direction_rad": float(np.arctan2(projected_step[1], projected_step[0])),
    }


def _selected_geometry(data, trial_signal, channel_indices, sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT):
    positions = np.take(
        get_channel_positions_mm(data, trial_signal.shape[0], sensor_position_unit=sensor_position_unit),
        channel_indices,
        axis=0,
    )
    channel_names = np.asarray(get_channel_names(data, trial_signal.shape[0]), dtype=object)[channel_indices]
    return _MovementGeometry(
        channel_indices=channel_indices,
        channel_names=channel_names,
        positions=positions,
        projected_positions=project_sensor_positions(positions),
    )


def _trajectory_row(context, geometry, weights, time_s, state):
    centroid = np.average(geometry.positions, axis=0, weights=weights)
    projected = np.average(geometry.projected_positions, axis=0, weights=weights)
    peak_local_index = int(np.argmax(weights))
    current = {"centroid": centroid, "projected": projected}
    if state.first is None:
        state.first = current

    row = {
        "participant": (context.participant_id if context.participant_id is not None else ""),
        "dataset": context.dataset,
        "trial": context.trial_idx,
        "trial_label": _trial_label(context.data, context.trial_idx),
        "time_s": time_s,
        "low_freq": context.config.frequency_range[0],
        "high_freq": context.config.frequency_range[1],
        "n_channels": int(geometry.channel_indices.size),
        "mean_alpha_power": float(np.mean(weights)),
        "total_alpha_power": float(np.sum(weights)),
        "peak_alpha_power": float(weights[peak_local_index]),
        "peak_channel": int(geometry.channel_indices[peak_local_index]),
        "peak_channel_name": str(geometry.channel_names[peak_local_index]),
        "spatial_concentration": _spatial_concentration(weights),
        "centroid_x_mm": float(centroid[0]),
        "centroid_y_mm": float(centroid[1]),
        "centroid_z_mm": float(centroid[2]),
        "projected_x_mm": float(projected[0]),
        "projected_y_mm": float(projected[1]),
    }
    row.update(
        _movement_values(
            centroid,
            projected,
            state.first,
            state.previous,
            state.previous_time,
            time_s,
        )
    )
    state.previous = current
    state.previous_time = time_s
    return row


def compute_alpha_movement_trajectory(
    data,
    trial_idx,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Track the alpha-power centroid across the MEG sensor array for one trial."""

    config = config or AlphaMovementConfig()
    trial_signal = get_trial_signal(data, trial_idx)
    channel_indices = _resolve_channel_indices(data, channel_indices, config.location_pattern)
    time_vector = get_time_vector(data, trial_idx)
    sample_indices = sample_time_indices(time_vector, config.time_window, config.trajectory_step_s)
    powers = _alpha_power(
        np.take(trial_signal, channel_indices, axis=0),
        time_vector,
        sample_indices,
        config,
    )
    geometry = _selected_geometry(data, trial_signal, channel_indices, config.sensor_position_unit)
    context = _MovementContext(data, trial_idx, participant_id, dataset, config)
    state = _MovementState()
    return [
        _trajectory_row(
            context,
            geometry,
            powers[:, column] + POWER_EPSILON,
            float(time_vector[time_index]),
            state,
        )
        for column, time_index in enumerate(sample_indices)
    ]


def compute_alpha_movement(
    data,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Track alpha-power centroids for every trial in ``data``."""

    config = config or AlphaMovementConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config.location_pattern)
    rows = []
    for trial_idx in range(count_trials(data)):
        rows.extend(
            compute_alpha_movement_trajectory(
                data,
                trial_idx,
                participant_id=participant_id,
                dataset=dataset,
                channel_indices=channel_indices,
                config=config,
            )
        )
    return rows


def _finite_mean(values):
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def summarize_alpha_movement(rows):
    """Average sensor-level alpha movement by participant, condition, and time."""

    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row["participant"]),
            str(row["dataset"]),
            str(row["trial_label"]),
            round(float(row["time_s"]), 9),
        )
        grouped[key].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        participant, dataset, trial_label, time_s = key
        trials = {int(row["trial"]) for row in group_rows}
        summary_rows.append(
            {
                "participant": participant,
                "dataset": dataset,
                "trial_label": trial_label,
                "time_s": time_s,
                "n_trials": len(trials),
                "mean_alpha_power": _finite_mean(row["mean_alpha_power"] for row in group_rows),
                "spatial_concentration": _finite_mean(row["spatial_concentration"] for row in group_rows),
                "centroid_x_mm": _finite_mean(row["centroid_x_mm"] for row in group_rows),
                "centroid_y_mm": _finite_mean(row["centroid_y_mm"] for row in group_rows),
                "centroid_z_mm": _finite_mean(row["centroid_z_mm"] for row in group_rows),
                "projected_x_mm": _finite_mean(row["projected_x_mm"] for row in group_rows),
                "projected_y_mm": _finite_mean(row["projected_y_mm"] for row in group_rows),
                "displacement_mm": _finite_mean(row["displacement_mm"] for row in group_rows),
                "speed_mm_per_s": _finite_mean(row["speed_mm_per_s"] for row in group_rows),
                "projected_speed_mm_per_s": _finite_mean(row["projected_speed_mm_per_s"] for row in group_rows),
            }
        )
    return summary_rows


def write_alpha_movement_csv(rows, output_path):
    """Write alpha movement rows to ``output_path``."""

    write_alpha_metrics_csv(rows, output_path)


def export_alpha_movement(
    data_folder,
    participants,
    trajectory_output_path,
    *,
    summary_output_path=None,
    cue=False,
    config=None,
):
    """Export sensor-level alpha movement trajectories for participants."""

    config = config or AlphaMovementConfig()
    dataset = "cue" if cue else "main"
    rows = []
    for participant_id in participants:
        data = load_participant_data(data_folder, participant_id, cue=cue)
        rows.extend(
            compute_alpha_movement(
                data,
                participant_id=participant_id,
                dataset=dataset,
                config=config,
            )
        )

    write_alpha_movement_csv(rows, trajectory_output_path)
    summary_rows = summarize_alpha_movement(rows)
    if summary_output_path:
        write_alpha_movement_csv(summary_rows, summary_output_path)
    return rows, summary_rows
