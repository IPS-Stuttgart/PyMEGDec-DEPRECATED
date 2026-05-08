"""Synthetic MATLAB fixtures for PyMEGDec smoke tests and demos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


# This config intentionally keeps all generator knobs in one immutable value
# object so tests, docs, and CLI output can serialize the exact configuration.
@dataclass(frozen=True)
class SyntheticDataConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a private-data-free PyMEGDec demo dataset.

    The generated files mimic the FieldTrip-like MATLAB structs expected by the
    existing PyMEGDec loaders: ``trial``, ``time``, ``trialinfo``, ``label``, and
    ``grad.chanpos``. Labels are one-based stimulus ids so that the data can be
    used with the legacy cross-validation and transfer workflows.
    """

    participant_id: int = 2
    n_classes: int = 16
    main_repeats_per_class: int = 10
    cue_repeats_per_class: int = 5
    n_channels: int = 8
    n_times: int = 261
    tmin: float = -0.5
    tmax: float = 0.8
    stimulus_window: tuple[float, float] = (0.15, 0.25)
    signal_scale: float = 6.0
    noise_scale: float = 0.05
    alpha_scale: float = 0.02
    cue_shift_scale: float = 0.15
    random_seed: int = 13


@dataclass(frozen=True)
class SyntheticDataOutput:
    """Paths and dimensions written by :func:`write_synthetic_dataset`."""

    data_dir: Path
    participant_id: int
    main_path: Path
    cue_path: Path
    manifest_path: Path | None
    main_trials: int
    cue_trials: int
    n_classes: int
    n_channels: int
    n_times: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(config: SyntheticDataConfig) -> None:
    for field_name, minimum, message in (
        ("participant_id", 1, "participant_id must be positive."),
        ("n_classes", 2, "n_classes must be at least 2 for decoding demos."),
        ("main_repeats_per_class", 1, "main_repeats_per_class must be at least 1."),
        ("cue_repeats_per_class", 1, "cue_repeats_per_class must be at least 1."),
        ("n_channels", 3, "n_channels must be at least 3 so sensor geometry is usable."),
        ("n_times", 3, "n_times must be at least 3."),
    ):
        _require(getattr(config, field_name) >= minimum, message)

    window_start, window_stop = config.stimulus_window
    _require(config.tmin < config.tmax, "tmin must be smaller than tmax.")
    _require(window_start < window_stop, "stimulus_window start must be smaller than stop.")
    _require(
        window_stop >= config.tmin and window_start <= config.tmax,
        "stimulus_window must overlap the generated time vector.",
    )
    for field_name, message in (
        ("signal_scale", "signal_scale must be positive."),
        ("noise_scale", "noise_scale must be non-negative."),
        ("alpha_scale", "alpha_scale must be non-negative."),
        ("cue_shift_scale", "cue_shift_scale must be non-negative."),
    ):
        lower_bound = 0.0 if field_name != "signal_scale" else np.finfo(float).tiny
        _require(getattr(config, field_name) >= lower_bound, message)


def _balanced_labels(n_classes: int, repeats_per_class: int) -> np.ndarray:
    """Return cyclic one-based labels with contiguous-fold friendly ordering."""

    return np.tile(np.arange(1, n_classes + 1, dtype=int), repeats_per_class)


def _class_prototypes(rng: np.random.Generator, n_classes: int, n_channels: int) -> np.ndarray:
    prototypes = rng.normal(size=(n_classes, n_channels))
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norms, np.finfo(float).eps)


def _channel_names(n_channels: int) -> np.ndarray:
    prefixes = ("MLO", "MRO", "MZO", "MLT", "MRT", "MZT", "MLF", "MRF")
    return np.asarray(
        [[f"{prefixes[index % len(prefixes)]}{index + 1:03d}" for index in range(n_channels)]],
        dtype=object,
    )


def _channel_positions(n_channels: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
    radius_mm = 80.0
    return np.column_stack(
        [
            radius_mm * np.cos(angles),
            radius_mm * np.sin(angles),
            20.0 * np.sin(2.0 * angles),
        ]
    )


def _synthetic_part_data(
    labels: np.ndarray,
    *,
    time_vector: np.ndarray,
    prototypes: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticDataConfig,
    cue: bool = False,
) -> dict[str, object]:
    stimulus_mask = (time_vector >= config.stimulus_window[0]) & (time_vector <= config.stimulus_window[1])
    channel_phase = np.linspace(0.0, np.pi, config.n_channels, endpoint=False)[:, None]
    alpha_carrier = np.sin(2.0 * np.pi * 10.0 * time_vector[None, :] + channel_phase)
    cue_shift = rng.normal(scale=config.cue_shift_scale, size=(config.n_channels, 1)) if cue else 0.0

    trials = np.empty((1, labels.size), dtype=object)
    times = np.empty((1, labels.size), dtype=object)
    for trial_index, label in enumerate(labels):
        trial = rng.normal(scale=config.noise_scale, size=(config.n_channels, config.n_times))
        trial += config.alpha_scale * alpha_carrier
        class_pattern = prototypes[int(label) - 1][:, None]
        trial[:, stimulus_mask] += config.signal_scale * class_pattern + cue_shift
        # Add a tiny deterministic trial offset so tests can detect accidental
        # trial reordering while keeping class structure dominant.
        trial += 1e-4 * (trial_index + 1)
        trials[0, trial_index] = trial.astype(float, copy=False)
        times[0, trial_index] = time_vector[None, :]

    return {
        "trial": trials,
        "time": times,
        "trialinfo": labels[None, :],
        "label": _channel_names(config.n_channels),
        "grad": {"chanpos": _channel_positions(config.n_channels)},
    }


def _manifest(output: SyntheticDataOutput, config: SyntheticDataConfig) -> dict[str, object]:
    return {
        "participant_id": output.participant_id,
        "main_file": output.main_path.name,
        "cue_file": output.cue_path.name,
        "main_trials": output.main_trials,
        "cue_trials": output.cue_trials,
        "n_classes": output.n_classes,
        "n_channels": output.n_channels,
        "n_times": output.n_times,
        "config": asdict(config),
    }


def write_synthetic_dataset(
    data_dir: str | Path,
    config: SyntheticDataConfig | None = None,
    *,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> SyntheticDataOutput:
    """Write balanced ``Part*Data.mat`` and ``Part*CueData.mat`` demo files.

    Parameters
    ----------
    data_dir:
        Directory that receives the generated MAT files.
    config:
        Synthetic dataset parameters. Defaults are chosen to support the
        package's cross-validation and transfer commands without private data.
    overwrite:
        Replace existing output files when true.
    write_manifest:
        Write a small JSON manifest next to the MAT files when true.
    """

    config = config or SyntheticDataConfig()
    _validate_config(config)

    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / f"Part{config.participant_id}Data.mat"
    cue_path = output_dir / f"Part{config.participant_id}CueData.mat"
    manifest_path = output_dir / "synthetic_data_manifest.json" if write_manifest else None

    existing = [path for path in (main_path, cue_path, manifest_path) if path is not None and path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Synthetic data output already exists: {names}. Pass overwrite=True to replace it.")

    rng = np.random.default_rng(config.random_seed)
    prototypes = _class_prototypes(rng, config.n_classes, config.n_channels)
    time_vector = np.linspace(config.tmin, config.tmax, config.n_times)
    main_labels = _balanced_labels(config.n_classes, config.main_repeats_per_class)
    cue_labels = _balanced_labels(config.n_classes, config.cue_repeats_per_class)

    sio.savemat(
        main_path,
        {"data": _synthetic_part_data(main_labels, time_vector=time_vector, prototypes=prototypes, rng=rng, config=config)},
    )
    sio.savemat(
        cue_path,
        {"data": _synthetic_part_data(cue_labels, time_vector=time_vector, prototypes=prototypes, rng=rng, config=config, cue=True)},
    )

    output = SyntheticDataOutput(
        data_dir=output_dir,
        participant_id=config.participant_id,
        main_path=main_path,
        cue_path=cue_path,
        manifest_path=manifest_path,
        main_trials=int(main_labels.size),
        cue_trials=int(cue_labels.size),
        n_classes=config.n_classes,
        n_channels=config.n_channels,
        n_times=config.n_times,
    )
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(_manifest(output, config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return output
