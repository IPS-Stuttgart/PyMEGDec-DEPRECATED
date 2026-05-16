"""Exploratory alpha-band metrics for MEG trials."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal
from pymegdec.alpha_signal import get_data_field, get_time_vector, get_trial_signal
from pymegdec.data_config import resolve_data_folder
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

DEFAULT_OCCIPITAL_PATTERN = r"^M[LRZ]O"
DEFAULT_TIME_WINDOW = (-0.4, -0.05)
DEFAULT_FREQUENCY_RANGE = (8.0, 12.0)
DEFAULT_SENSOR_POSITION_UNIT = "auto"
_SENSOR_POSITION_UNIT_SCALE_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}
_PROJECTION_EPSILON = 1e-12


@dataclass(frozen=True)
class AlphaMetricConfig:
    """Parameters controlling alpha metric extraction."""

    location_pattern: str = DEFAULT_OCCIPITAL_PATTERN
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW
    frequency_range: tuple[float, float] = DEFAULT_FREQUENCY_RANGE
    filter_order: int = 5
    sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT


def _unwrap_singleton(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def _get_struct_field(value, field_name):
    if isinstance(value, dict):
        return value[field_name]
    if isinstance(value, np.void):
        return value[field_name]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return value[field_name]
    raise TypeError(f"Cannot read field {field_name!r} from {type(value).__name__}.")


def _value_to_string(value):
    value = _unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in {"S", "U"}:
            return "".join(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in array.ravel()
            )
        if array.dtype == object:
            items = [_unwrap_singleton(item) for item in array.ravel()]
            if all(isinstance(item, (bytes, str)) for item in items):
                return "".join(
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in items
                )
    return str(value)


def _label_to_string(label):
    return _value_to_string(label)


def _normalize_sensor_position_unit(unit):
    unit_text = _value_to_string(unit).strip().lower()
    canonical = {
        "m": "m",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "cm": "cm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "mm": "mm",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
    }
    if unit_text not in canonical:
        raise ValueError(
            "sensor_position_unit must be 'auto', 'm', 'cm', 'mm', or a common "
            f"spelled-out equivalent; got {unit!r}."
        )
    return canonical[unit_text]


def get_channel_names(data, n_channels=None):
    """Return channel names from a FieldTrip-like MATLAB structure."""

    labels = np.asarray(get_data_field(data, "label"), dtype=object).ravel()
    if n_channels is not None:
        labels = labels[:n_channels].ravel()
    return [_label_to_string(label) for label in labels]


def get_channel_positions(data, n_channels=None):
    """Return unscaled channel positions from ``data.grad.chanpos``."""

    grad = get_data_field(data, "grad")
    chanpos = _unwrap_singleton(_get_struct_field(grad, "chanpos"))
    positions: np.ndarray = np.asarray(chanpos, dtype=float)
    if n_channels is None:
        return positions
    return positions[:n_channels]


def get_channel_position_unit(data):
    """Return the unit stored in ``data.grad.unit``, or ``None`` if absent."""

    try:
        grad = get_data_field(data, "grad")
        unit = _get_struct_field(grad, "unit")
    except (KeyError, TypeError, ValueError):
        return None

    unit_text = _value_to_string(unit).strip()
    if not unit_text:
        return None
    return _normalize_sensor_position_unit(unit_text)


def resolve_sensor_position_unit(data, sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT):
    """Resolve the unit for ``data.grad.chanpos``.

    ``"auto"`` reads FieldTrip's ``data.grad.unit`` when present and otherwise
    falls back to millimetres to preserve the historical PyMEGDec convention.
    """

    if sensor_position_unit is None or _value_to_string(sensor_position_unit).strip().lower() == "auto":
        return get_channel_position_unit(data) or "mm"
    return _normalize_sensor_position_unit(sensor_position_unit)


def get_channel_positions_mm(data, n_channels=None, *, sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT):
    """Return channel positions converted to millimetres."""

    positions = get_channel_positions(data, n_channels)
    unit = resolve_sensor_position_unit(data, sensor_position_unit)
    return positions * _SENSOR_POSITION_UNIT_SCALE_TO_MM[unit]


def select_channels(data, location_pattern=DEFAULT_OCCIPITAL_PATTERN):
    """Select channels whose labels match ``location_pattern``."""

    n_channels = get_trial_signal(data, 0).shape[0]
    pattern = re.compile(location_pattern)
    channel_names = get_channel_names(data, n_channels)
    return [index for index, channel_name in enumerate(channel_names) if pattern.search(channel_name)]


def _check_positions_2d_array(positions):
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2:
        raise ValueError("positions must be a 2D array with shape (n_sensors, n_coordinates).")
    if array.shape[0] < 3:
        raise ValueError("At least three sensor positions are required for 2D projection.")
    if array.shape[1] < 2:
        raise ValueError("Sensor positions must have at least two coordinate dimensions.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Sensor positions must be finite.")
    return array


def _pca_plane_normal(centered):
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    if singular_values.size < 2 or singular_values[1] <= scale * _PROJECTION_EPSILON:
        raise ValueError("At least two non-collinear sensor positions are required for 2D projection.")
    if centered.shape[1] != 3 or vt.shape[0] < 3:
        return None
    return vt[2]


def _anchored_plane_axes(normal):
    """Return deterministic in-plane axes anchored to the sensor coordinates."""

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= _PROJECTION_EPSILON:
        raise ValueError("Could not determine a stable sensor projection plane normal.")
    normal = normal / normal_norm

    anchored_axes = []
    for reference in np.eye(3):
        # Project the next global coordinate axis into the PCA plane, then
        # orthogonalize it against already chosen in-plane axes.  This makes the
        # first projected axis follow global +x when possible, and the second
        # follow global +y when possible; if either is normal to the plane, the
        # next coordinate axis is used instead.
        candidate = reference - float(np.dot(reference, normal)) * normal
        for axis in anchored_axes:
            candidate = candidate - float(np.dot(candidate, axis)) * axis

        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= _PROJECTION_EPSILON:
            continue

        candidate = candidate / candidate_norm
        if float(np.dot(candidate, reference)) < 0.0:
            candidate = -candidate
        anchored_axes.append(candidate)
        if len(anchored_axes) == 2:
            return np.column_stack(anchored_axes)

    raise ValueError("Could not anchor two projected axes to the sensor coordinate frame.")


def project_sensor_positions(positions):
    """Project sensor positions to a deterministic coordinate-anchored 2D plane.

    The projection plane is the best-fitting PCA/SVD plane through the selected
    sensors.  Unlike raw SVD coordinates, the in-plane axes are anchored to the
    original sensor coordinate frame: projected x follows global +x when
    possible, projected y follows global +y when possible, and axes that are
    normal to the fitted plane are skipped.  This keeps projected directions and
    projected x/y trajectories comparable across participants and reruns.
    """

    positions = _check_positions_2d_array(positions)
    centered = positions - np.mean(positions, axis=0)
    if centered.shape[1] == 2:
        return centered

    normal = _pca_plane_normal(centered)
    if normal is None:
        # Keep non-3D inputs deterministic by falling back to signed PCA axes.
        # FieldTrip/CTF sensor positions are 3D, so the coordinate-anchored path
        # above is used for normal PyMEGDec alpha analyses.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes = vt[:2].T
        for column_index in range(axes.shape[1]):
            axis = axes[:, column_index]
            largest_component = int(np.argmax(np.abs(axis)))
            if axis[largest_component] < 0:
                axes[:, column_index] = -axis
        return centered @ axes

    return centered @ _anchored_plane_axes(normal)


def _delaunay_edges(coords2d):
    if len(coords2d) < 3:
        raise ValueError("At least three sensor positions are required.")

    triangulation = Delaunay(coords2d)
    edges = set()
    for simplex in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((int(simplex[first]), int(simplex[second])))))

    edge_indices = np.array(sorted(edges), dtype=int)
    edge_vectors = coords2d[edge_indices[:, 1]] - coords2d[edge_indices[:, 0]]
    return edge_indices, edge_vectors, np.linalg.pinv(edge_vectors)


def _trial_label(data, trial_idx):
    if isinstance(data, dict) and "trialinfo" not in data:
        return np.nan
    if not isinstance(data, dict) and "trialinfo" not in data.dtype.names:
        return np.nan
    trialinfo = np.asarray(get_data_field(data, "trialinfo")).ravel()
    return trialinfo[trial_idx].item()


def count_trials(data):
    """Return the number of trials in a FieldTrip-like data structure."""

    trial_field = np.asarray(get_data_field(data, "trial"), dtype=object)
    if trial_field.ndim == 2 and trial_field.shape[0] == 1:
        return trial_field.shape[1]
    return len(trial_field.ravel())


def _time_mask(time_vector, time_window):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    mask = (time_vector >= start) & (time_vector <= stop)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _phase_gradient_metrics(phase, edge_indices, edge_vectors, edge_pinv, center_frequency):
    phase_delta = np.angle(np.exp(1j * (phase[edge_indices[:, 1], :] - phase[edge_indices[:, 0], :])))
    gradients = edge_pinv @ phase_delta
    predicted_delta = edge_vectors @ gradients
    residual = np.angle(np.exp(1j * (phase_delta - predicted_delta)))
    fit = np.abs(np.mean(np.exp(1j * residual), axis=0))
    gradient_norm = np.linalg.norm(gradients, axis=0)
    weights = fit + 1e-12
    mean_gradient = np.average(gradients, axis=1, weights=weights)
    valid = (fit > 0.5) & (gradient_norm > 1e-4)

    speed_m_per_s = np.nan
    if np.any(valid):
        # alpha phase velocity = angular frequency / spatial angular frequency.
        speed_m_per_s = np.nanmedian((2 * np.pi * center_frequency / gradient_norm[valid]) / 1000.0)

    return {
        "phase_plane_fit": float(np.mean(fit)),
        "spatial_freq_rad_per_mm": float(np.average(gradient_norm, weights=weights)),
        "speed_m_per_s": float(speed_m_per_s),
        "gradient_x": float(mean_gradient[0]),
        "gradient_y": float(mean_gradient[1]),
        "direction_rad": float(np.arctan2(mean_gradient[1], mean_gradient[0])),
    }


def _resolve_channel_indices(data, channel_indices, config):
    if channel_indices is None:
        channel_indices = select_channels(data, config.location_pattern)
    channel_indices = np.asarray(channel_indices, dtype=int)
    if channel_indices.size == 0:
        raise ValueError(f"No channels matched pattern: {config.location_pattern}")
    return channel_indices


def compute_alpha_analytic_window(signal, time_vector, config):
    """Return alpha-band analytic signal samples in ``config.time_window``."""

    sampling_rate = 1 / np.diff(time_vector[:2])[0]
    time_indices = np.flatnonzero(_time_mask(time_vector, config.time_window))
    low_freq, high_freq = config.frequency_range

    sos = scipy.signal.butter(
        config.filter_order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    alpha_signal = scipy.signal.sosfiltfilt(sos, signal, axis=-1)
    analytic_signal = scipy.signal.hilbert(alpha_signal, axis=-1)
    alpha_window = np.take(analytic_signal, time_indices, axis=-1)
    return alpha_window, time_indices


def _alpha_window_and_phase(signal, time_vector, config):
    alpha_window, _ = compute_alpha_analytic_window(signal, time_vector, config)
    return alpha_window, np.angle(alpha_window)


def _phase_geometry(data, channel_indices, sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT):
    positions = np.take(
        get_channel_positions_mm(
            data,
            get_trial_signal(data, 0).shape[0],
            sensor_position_unit=sensor_position_unit,
        ),
        channel_indices,
        axis=0,
    )
    coords2d = project_sensor_positions(positions)
    return _delaunay_edges(coords2d)


def compute_alpha_trial_metrics(
    data,
    trial_idx,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Compute exploratory prestimulus alpha metrics for one trial."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)
    time_vector = get_time_vector(data, trial_idx)
    signal = np.take(get_trial_signal(data, trial_idx), channel_indices, axis=0)
    alpha_window, phase = _alpha_window_and_phase(signal, time_vector, config)
    edge_indices, edge_vectors, edge_pinv = _phase_geometry(data, channel_indices, config.sensor_position_unit)

    row = {
        "participant": participant_id if participant_id is not None else "",
        "dataset": dataset,
        "trial": trial_idx,
        "trial_label": _trial_label(data, trial_idx),
        "time_window_start": config.time_window[0],
        "time_window_stop": config.time_window[1],
        "low_freq": config.frequency_range[0],
        "high_freq": config.frequency_range[1],
        "n_channels": int(len(channel_indices)),
        "alpha_power": float(np.mean(np.abs(alpha_window) ** 2)),
        "log_alpha_power": float(np.mean(np.log(np.abs(alpha_window) ** 2 + 1e-12))),
        "phase_concentration": float(np.abs(np.mean(np.exp(1j * phase)))),
    }
    row.update(
        _phase_gradient_metrics(
            phase,
            edge_indices,
            edge_vectors,
            edge_pinv,
            center_frequency=sum(config.frequency_range) / 2,
        )
    )
    return row


def compute_alpha_metrics(
    data,
    *,
    participant_id=None,
    dataset="main",
    channel_indices=None,
    config=None,
):
    """Compute alpha metrics for every trial in ``data``."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)

    n_trials = count_trials(data)
    return [
        compute_alpha_trial_metrics(
            data,
            trial_idx,
            participant_id=participant_id,
            dataset=dataset,
            channel_indices=channel_indices,
            config=config,
        )
        for trial_idx in range(n_trials)
    ]


def write_alpha_metrics_csv(rows, output_path):
    """Write alpha metric rows to ``output_path``."""

    if not rows:
        raise ValueError("At least one row is required.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_participant_data(data_folder, participant_id, *, cue=False):
    """Load a participant's main or cue MATLAB data file."""

    data_folder = resolve_data_folder(data_folder)
    suffix = "CueData" if cue else "Data"
    data_path = Path(data_folder) / f"Part{participant_id}{suffix}.mat"
    return sio.loadmat(data_path)["data"][0]


def export_participant_alpha_metrics(
    data_folder,
    participant_id,
    output_path,
    *,
    cue=False,
    config=None,
):
    """Load participant data, compute alpha metrics, and write a CSV."""

    config = config or AlphaMetricConfig()
    data = load_participant_data(data_folder, participant_id, cue=cue)
    dataset = "cue" if cue else "main"
    rows = compute_alpha_metrics(
        data,
        participant_id=participant_id,
        dataset=dataset,
        config=config,
    )
    write_alpha_metrics_csv(rows, output_path)
    return rows
