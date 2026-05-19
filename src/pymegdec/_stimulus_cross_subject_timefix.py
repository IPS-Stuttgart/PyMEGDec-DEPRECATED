"""Per-trial time-axis fixes for cross-subject stimulus features."""

from __future__ import annotations

import numpy as np

from pymegdec import _stimulus_cross_subject_core as _core

_impl = _core._impl


def _extract_window_features(data, time_window, *, feature_mode, trial_indices=None):
    """Extract fixed-window features using each trial's own time vector."""

    feature_mode = _impl._normalize_feature_mode(feature_mode)
    features = []
    n_window_samples = None
    for trial_idx in _impl._iter_trial_indices(data, trial_indices):
        time_vector = _validated_trial_time_vector(data, trial_idx)
        signal = _validated_trial_signal(data, trial_idx, time_vector)
        mask = _time_mask_for_trial(time_vector, time_window, trial_idx)
        n_window_samples = _require_consistent_sample_count(
            int(np.sum(mask)),
            n_window_samples,
            trial_idx,
            "time_window",
        )
        window_signal = signal[:, mask]
        if feature_mode == "sensor_mean":
            feature = np.mean(window_signal, axis=1)
        elif feature_mode == "sensor_flat":
            feature = window_signal.reshape(-1, order="F")
        elif feature_mode == "sensor_mean_slope":
            feature = _impl._sensor_mean_slope_feature(window_signal, time_vector[mask])
        elif feature_mode == "sensor_mean_slope_std":
            feature = _impl._sensor_mean_slope_std_feature(window_signal, time_vector[mask])
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    if n_window_samples is None:
        raise ValueError("No trials were selected for window feature extraction.")
    return np.vstack(features), int(n_window_samples)


def _baseline_channel_statistics(data, baseline_window, trial_indices):
    """Compute baseline channel statistics with per-trial time masks."""

    n_channels = int(_impl._trial_signal(data, 0).shape[0])
    sum_values = np.zeros(n_channels, dtype=float)
    sum_squares = np.zeros(n_channels, dtype=float)
    n_values = 0
    n_baseline_samples = None
    for trial_idx in _impl._iter_trial_indices(data, trial_indices):
        time_vector = _validated_trial_time_vector(data, trial_idx)
        signal = _validated_trial_signal(data, trial_idx, time_vector)
        mask = _time_mask_for_trial(time_vector, baseline_window, trial_idx)
        n_baseline_samples = _require_consistent_sample_count(
            int(np.sum(mask)),
            n_baseline_samples,
            trial_idx,
            "baseline_window",
        )
        baseline_signal = signal[:, mask]
        sum_values += np.sum(baseline_signal, axis=1)
        sum_squares += np.sum(np.square(baseline_signal), axis=1)
        n_values += baseline_signal.shape[1]
    if n_baseline_samples is None or n_values == 0:
        raise ValueError("No trials were selected for baseline statistics.")
    mean = sum_values / n_values
    variance = np.maximum(sum_squares / n_values - np.square(mean), 0.0)
    return mean, np.sqrt(variance), int(n_baseline_samples)


def _validated_trial_time_vector(data, trial_idx):
    time_vector = _impl._time_vector(data, trial_idx)
    if time_vector.size == 0:
        raise ValueError(f"Time vector for trial {trial_idx} is empty.")
    if time_vector.size > 1 and np.any(np.diff(time_vector) <= 0):
        raise ValueError(f"Time vector for trial {trial_idx} must be strictly increasing.")
    return time_vector


def _validated_trial_signal(data, trial_idx, time_vector):
    signal = _impl._trial_signal(data, trial_idx)
    if signal.ndim != 2:
        raise ValueError(f"Trial {trial_idx} must be a 2D channels-by-time array.")
    if signal.shape[1] != time_vector.size:
        raise ValueError(
            f"Trial {trial_idx} has {signal.shape[1]} samples but its time vector has "
            f"{time_vector.size} entries."
        )
    return signal


def _time_mask_for_trial(time_vector, time_window, trial_idx):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    tolerance = _time_support_tolerance(time_vector)
    if start < time_vector[0] - tolerance or stop > time_vector[-1] + tolerance:
        raise ValueError(f"time_window {time_window} is outside trial {trial_idx}'s time support.")
    mask = _impl._time_mask(time_vector, time_window)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} contains no samples in trial {trial_idx}.")
    return mask


def _time_support_tolerance(time_vector):
    if time_vector.size < 2:
        return 1e-12
    return 0.5 * float(np.median(np.diff(time_vector))) + 1e-12


def _require_consistent_sample_count(sample_count, expected_count, trial_idx, window_name):
    if expected_count is None:
        return sample_count
    if sample_count != expected_count:
        raise ValueError(
            f"{window_name} for trial {trial_idx} contains {sample_count} samples; "
            f"expected {expected_count}. Check per-trial time vectors."
        )
    return expected_count


def _install_module_fixes():
    _impl._extract_window_features = _extract_window_features
    _impl._baseline_channel_statistics = _baseline_channel_statistics
    setattr(_core, "_extract_window_features", _extract_window_features)
    setattr(_core, "_baseline_channel_statistics", _baseline_channel_statistics)


_install_module_fixes()
