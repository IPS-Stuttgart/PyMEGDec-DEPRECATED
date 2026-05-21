"""Compatibility helpers for alpha-band signal extraction.

Generic filtering, Hilbert-phase extraction, sampling-rate validation, and
phase averaging are delegated to NeuRepTrace when its optional signal helpers
are available. Local fallbacks keep PyMEGDec's historical public API working in
environments with older NeuRepTrace installations.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

try:  # pragma: no cover - exercised only when NeuRepTrace exposes this module.
    from neureptrace.signal import band as _neureptrace_band
except ImportError:  # pragma: no cover - normal path for older NeuRepTrace versions.
    _neureptrace_band = None


def get_data_field(data, field_name):
    if isinstance(data, dict):
        return data[field_name]

    field = data[field_name]
    if isinstance(field, np.ndarray) and field.size == 1:
        return field.item()
    return field


def _unwrap_outer_cell_array(cell_array):
    values = np.asarray(cell_array, dtype=object)
    while values.dtype == object and values.size == 1:
        item = values.item()
        item_array = np.asarray(item)
        if not isinstance(item, np.ndarray) or item_array.dtype != object:
            break
        values = np.asarray(item, dtype=object)
    return values


def _cell_item(cell_array, index):
    values = _unwrap_outer_cell_array(cell_array)
    if values.ndim == 0:
        return values.item()
    if values.ndim == 2 and values.shape[0] == 1:
        return values[0, index]
    if values.ndim == 2 and values.shape[1] == 1:
        return values[index, 0]
    return values[index]


def get_time_vector(data, trial_idx=0):
    time_vector = _cell_item(get_data_field(data, "time"), trial_idx)
    return np.asarray(time_vector, dtype=float).ravel()


def get_trial_signal(data, trial_idx=0):
    trial_signal = _cell_item(get_data_field(data, "trial"), trial_idx)
    return np.asarray(trial_signal, dtype=float)


def uniform_sample_interval(time_vector):
    """Return the sample interval after validating a finite regular time axis."""

    time_vector = np.asarray(time_vector, dtype=float).ravel()
    if time_vector.size < 2:
        raise ValueError("time_vector must contain at least two samples.")
    if not np.all(np.isfinite(time_vector)):
        raise ValueError("time_vector must contain only finite values.")

    diffs = np.diff(time_vector)
    if np.any(diffs <= 0):
        raise ValueError("time_vector must be strictly increasing.")

    sample_interval = float(np.median(diffs))
    if not np.allclose(diffs, sample_interval, rtol=1e-6, atol=1e-12):
        raise ValueError("time_vector must be uniformly sampled.")
    return sample_interval


def sampling_rate_from_time_vector(time_vector):
    """Return sampling rate in Hz after validating ``time_vector``."""

    if _neureptrace_band is not None:
        sampling_rate = getattr(_neureptrace_band, "sampling_rate_from_time_vector", None)
        if sampling_rate is not None:
            return float(sampling_rate(time_vector))
    return float(1.0 / uniform_sample_interval(time_vector))


def _validated_sampling_rate(sampling_rate):
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_rate must be a positive finite value.") from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be a positive finite value.")
    return sampling_rate


def _validated_signal_values(signal_values):
    signal_values = np.asarray(signal_values, dtype=float)
    if signal_values.ndim == 0:
        raise ValueError("signal_values must have at least one sample dimension.")
    if signal_values.shape[-1] < 2:
        raise ValueError("signal_values must contain at least two samples along the last axis.")
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("signal_values must contain only finite values.")
    return signal_values


def _validated_trial_signal(data, trial_idx, time_vector):
    signal = get_trial_signal(data, trial_idx)
    if signal.ndim != 2:
        raise ValueError(f"Trial {trial_idx} must be a 2D channels-by-time array.")
    if signal.shape[1] != time_vector.size:
        raise ValueError(
            f"Trial {trial_idx} has {signal.shape[1]} samples but its time vector has "
            f"{time_vector.size} entries."
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Trial {trial_idx} signal must contain only finite values.")
    return signal


def _parse_channel_index(value):
    if not isinstance(value, (int, np.integer)):
        raise ValueError("channel_range must contain integer channel indices.")
    return int(value)


def _channel_indices_from_range(channel_range, n_channels):
    try:
        start, stop = channel_range
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_range must contain exactly two integer indices.") from exc

    start = _parse_channel_index(start)
    stop = _parse_channel_index(stop)
    if start > stop:
        raise ValueError("channel_range start must be less than or equal to stop.")
    if start < 0 or stop >= int(n_channels):
        raise ValueError(
            "channel_range is outside the available channels: "
            f"got ({start}, {stop}) for {n_channels} channels."
        )
    return range(start, stop + 1)


def bandpass_filter_signal(signal_values, sampling_rate, lowcut=8.0, highcut=12.0, order=5):
    signal_values = _validated_signal_values(signal_values)
    sampling_rate = _validated_sampling_rate(sampling_rate)
    nyquist = 0.5 * sampling_rate
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be lower than highcut.")
    if highcut >= nyquist:
        raise ValueError("highcut must be lower than the Nyquist frequency.")

    sos = scipy.signal.butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return scipy.signal.sosfiltfilt(sos, signal_values)


def extract_alpha_signal_and_phase(signal_values, sampling_rate, lowcut=8.0, highcut=12.0):
    """Return the alpha-band signal and Hilbert phase using PyMEGDec defaults."""

    filtered_signal = bandpass_filter_signal(signal_values, sampling_rate, lowcut, highcut)
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    return filtered_signal, np.angle(analytic_signal)


def extract_phase(signal_values, sampling_rate, lowcut=8.0, highcut=12.0):
    """
    Extracts the phase of the given signal using bandpass filtering and
    Hilbert transform.
    """

    if _neureptrace_band is not None:
        phase_extractor = getattr(_neureptrace_band, "extract_phase", None)
        if phase_extractor is not None:
            return phase_extractor(signal_values, sampling_rate, lowcut=lowcut, highcut=highcut)

    _, phase = extract_alpha_signal_and_phase(signal_values, sampling_rate, lowcut, highcut)
    return phase


def average_phases(phases):
    """
    Averages the phases across multiple channels.
    """

    if not phases:
        raise ValueError("At least one phase array is required.")

    if _neureptrace_band is not None:
        phase_averager = getattr(_neureptrace_band, "average_phases", None)
        if phase_averager is not None:
            return phase_averager(phases)

    phase_matrix = np.vstack(phases)
    mean_phase = np.angle(np.mean(np.exp(1j * phase_matrix), axis=0))
    return mean_phase


def extract_time_basis(data, trial_idx=0, channel_range=(187, 198)):
    """
    Extract a robust alpha-phase time basis across multiple channels.

    Generic alpha filtering and Hilbert phase extraction are delegated to
    :mod:`neureptrace.signal.band` when available; local fallbacks keep
    PyMEGDec's historical FieldTrip/MATLAB trial and channel-range conventions.
    """

    time_vector = get_time_vector(data, trial_idx)
    sampling_rate = sampling_rate_from_time_vector(time_vector)
    signal = _validated_trial_signal(data, trial_idx, time_vector)
    channel_indices = _channel_indices_from_range(channel_range, signal.shape[0])

    phases = []
    for channel_idx in channel_indices:
        signal_curr_chan = signal[channel_idx, :]
        phase = extract_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)

    mean_phase = average_phases(phases)
    return mean_phase


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio

    demo_data_folder = r"."
    demo_part = 2
    demo_data = sio.loadmat(f"{demo_data_folder}/Part{demo_part}Data.mat")["data"][0]

    demo_time_basis = extract_time_basis(demo_data, trial_idx=0, channel_range=(187, 198))

    print("Robust time basis (average phase):", demo_time_basis)

    demo_time_vector = get_time_vector(demo_data)
    plt.plot(demo_time_vector, demo_time_basis, label="Average Phase")
    plt.title("Average Alpha Phase Across Channels 187-198")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()
