"""Feature preprocessing helpers for MEG decoding data."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt


def preprocess_features(
    data,
    frequency_range,
    new_framerate,
    window_size,
    train_window_center,
    null_window_center,
):
    data = filter_features(data, frequency_range[0], frequency_range[1])
    if new_framerate != float("inf"):
        data = downsample_data(data, new_framerate)

    train_window = (
        train_window_center - window_size / 2,
        train_window_center + window_size / 2,
    )
    null_time_window = (null_window_center - window_size / 2, null_window_center + window_size / 2) if not np.isnan(null_window_center) else (np.nan, np.nan)
    if not np.isnan(null_time_window).all() and null_time_window[1] > train_window[0]:
        raise ValueError("Null window must be before train window")

    stimuli_features_cell, null_features_cell = extract_windows(data, train_window, null_time_window)
    return stimuli_features_cell, null_features_cell


def filter_features(data, low_freq, high_freq):
    if not data["time"][0][0][0].size:
        raise ValueError("Time vector is empty or not provided correctly.")

    sample_rate = float(1 / np.diff(data["time"][0][0][0][0, :2])[0])

    if low_freq < 0:
        raise ValueError("Low frequency must be greater than or equal to 0")
    if high_freq < 0:
        raise ValueError("High frequency must be greater than or equal to 0")
    if high_freq < low_freq:
        raise ValueError("High frequency must be greater than or equal to low frequency")

    if low_freq == 0 and high_freq == float("inf"):
        return data
    if low_freq == 0:
        b, a = butter(4, high_freq / (sample_rate / 2), "low")
    elif high_freq != float("inf"):
        cutoff = [low_freq / (sample_rate / 2), high_freq / (sample_rate / 2)]
        b, a = butter(4, cutoff, "bandpass")
    else:
        raise ValueError("Highpass filter not supported.")

    for i in range(len(data["trial"][0][0])):
        data["trial"][0][0][i] = filtfilt(b, a, data["trial"][0][0][i].T, axis=0).T
    return data


def downsample_data(data, new_framerate):
    raw_fs = round(float(1 / np.median(np.diff(data["time"][0][0][0][0]))))
    if new_framerate != raw_fs:
        first_time = data["time"][0][0][0][0]
        step = 1 / new_framerate
        new_t = np.arange(first_time[0], first_time[-1] + step / 2, step)
        for i in range(len(data["trial"][0][0])):
            data["trial"][0][0][i] = detrend(data["trial"][0][0][i], axis=1)
            interpolator = interp1d(
                data["time"][0][0][i][0, :],
                data["trial"][0][0][i],
                axis=1,
                fill_value="extrapolate",
            )
            data["trial"][0][0][i] = interpolator(new_t)
            data["time"][0][0][i] = new_t[None]
    return data


def extract_windows(data, train_window, null_time_window):
    time = data["time"][0][0][0]
    train_begin_index = np.argmin(np.abs(time - train_window[0]))
    train_end_index = np.argmin(np.abs(time - train_window[1]))
    stimuli_features_cell = [trial[:, train_begin_index : train_end_index + 1].reshape(-1, 1, order="F") for trial in data["trial"][0][0]]

    if np.isnan(null_time_window).all():
        null_features_cell = []
    elif null_time_window[1] > 0:
        raise ValueError("Null window should not contain positive time points")
    elif null_time_window[1] - null_time_window[0] >= 0:
        null_begin_index = np.argmin(np.abs(time - null_time_window[0]))
        null_end_index = null_begin_index + (train_end_index - train_begin_index)
        null_features_cell = [trial[:, null_begin_index : null_end_index + 1].reshape(-1, 1, order="F") for trial in data["trial"][0][0]]
    else:
        raise ValueError("Invalid null window")

    return stimuli_features_cell, null_features_cell
