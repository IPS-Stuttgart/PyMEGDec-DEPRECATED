"""Public time-resolved stimulus decoding API."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
import typing as _typing

import numpy as _np

import pymegdec._stimulus_decoding_core as _core
from pymegdec._stimulus_summary import (
    summarize_stimulus_decoding,
    summarize_stimulus_temporal_generalization,
)

DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER = _core.DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER
DEFAULT_ONSET_SCAN_TIME_WINDOW = _core.DEFAULT_ONSET_SCAN_TIME_WINDOW
DEFAULT_ONSET_SCAN_STEP_S = _core.DEFAULT_ONSET_SCAN_STEP_S
DEFAULT_ONSET_THRESHOLD_WINDOW = _core.DEFAULT_ONSET_THRESHOLD_WINDOW
DEFAULT_ONSET_THRESHOLD_QUANTILE = _core.DEFAULT_ONSET_THRESHOLD_QUANTILE
DEFAULT_ONSET_THRESHOLD_METHOD = _core.DEFAULT_ONSET_THRESHOLD_METHOD
DEFAULT_ONSET_MIN_CONSECUTIVE = _core.DEFAULT_ONSET_MIN_CONSECUTIVE
DEFAULT_ONSET_MIN_DURATION = _core.DEFAULT_ONSET_MIN_DURATION
DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION = _core.DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION
ONSET_SCORE_TYPE_PREDICTED_CLASS = "predicted_class_score"
window_centers_from_range = _core.window_centers_from_range
summarize_stimulus_decoding_peaks = _core.summarize_stimulus_decoding_peaks
summarize_stimulus_prediction_diagnostics = _core.summarize_stimulus_prediction_diagnostics
write_stimulus_decoding_plots = _core.write_stimulus_decoding_plots


@_dataclass(frozen=True)
class StimulusDecodingConfig(_core.StimulusDecodingConfig):
    """Stimulus-decoding config with an inferred chance level by default.

    By default, the reported chance level uses the number of stimulus classes
    actually present in the evaluated validation labels.  The legacy default
    value ``DEFAULT_CHANCE_CLASSES`` is also treated as automatic while
    ``infer_chance_classes`` is true, so existing CLI and workflow defaults do
    not silently keep reporting 1/16 for subset analyses.

    Set ``infer_chance_classes=False`` to force ``chance_classes`` exactly,
    including an explicit 16-class chance level.
    """

    chance_classes: int | None = None  # type: ignore[assignment]
    infer_chance_classes: bool = True


def evaluate_time_resolved_stimulus_transfer(
    data_folder,
    participants,
    *,
    config=None,
    progress=None,
):
    """Evaluate train-main/validate-cue stimulus decoding across time windows."""

    core_config, auto_chance = _config_for_core(config)
    rows = _core.evaluate_time_resolved_stimulus_transfer(
        data_folder,
        participants,
        config=core_config,
        progress=progress,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


def evaluate_participant_time_resolved_stimulus_transfer(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate one participant's stimulus transfer accuracy across window centers."""

    core_config, auto_chance = _config_for_core(config)
    rows = _core.evaluate_participant_time_resolved_stimulus_transfer(
        data_folder,
        participant,
        config=core_config,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


def evaluate_participant_stimulus_decoding_diagnostics(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    """Evaluate one participant and return accuracy rows plus prediction diagnostics."""

    core_config, auto_chance = _config_for_core(config)
    rows, prediction_rows = _core.evaluate_participant_stimulus_decoding_diagnostics(
        data_folder,
        participant,
        config=core_config,
        diagnostic_window_centers=diagnostic_window_centers,
    )
    return (_patch_auto_chance(rows) if auto_chance else rows), prediction_rows


# jscpd:ignore-start
def evaluate_participant_stimulus_temporal_generalization(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate train-time/test-time stimulus decoding for one participant."""

    core_config, auto_chance = _config_for_core(config)
    rows = _core.evaluate_participant_stimulus_temporal_generalization(
        data_folder,
        participant,
        config=core_config,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


# jscpd:ignore-end
# jscpd:ignore-start
def evaluate_participant_stimulus_onset_scan(
    data_folder,
    participant,
    *,
    config=None,
    train_window_center=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,  # noqa: F405
    threshold_window=DEFAULT_ONSET_THRESHOLD_WINDOW,  # noqa: F405
    threshold_quantile=DEFAULT_ONSET_THRESHOLD_QUANTILE,  # noqa: F405
    threshold_method=DEFAULT_ONSET_THRESHOLD_METHOD,  # noqa: F405
    min_consecutive=DEFAULT_ONSET_MIN_CONSECUTIVE,  # noqa: F405
    min_duration=DEFAULT_ONSET_MIN_DURATION,  # noqa: F405
    require_stable_prediction=DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,  # noqa: F405
    detection_start_s=None,
):
    """Scan validation trials for stimulus identity without using onset at test time."""

    config = _onset_scan_config(config)
    core_config, auto_chance = _config_for_core(config)
    scan_rows, event_rows = _core.evaluate_participant_stimulus_onset_scan(
        data_folder,
        participant,
        config=core_config,
        train_window_center=train_window_center,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
        detection_start_s=detection_start_s,
    )
    scan_rows = _patch_onset_score_columns(scan_rows)
    event_rows = _patch_onset_event_score_columns(event_rows)
    return (_patch_auto_chance(scan_rows) if auto_chance else scan_rows), event_rows


# jscpd:ignore-end
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def export_time_resolved_stimulus_decoding(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    participant_peaks_output_path=None,
    diagnostic_window_centers=None,
    plots_dir=None,
    config=None,
    progress=None,
):
    """Run time-resolved stimulus decoding and write CSV/plot artifacts."""

    config = config or StimulusDecodingConfig()
    data_folder = _core.resolve_data_folder(data_folder)
    rows = []
    prediction_rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        participant_rows, participant_prediction_rows = evaluate_participant_stimulus_decoding_diagnostics(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=diagnostic_window_centers,
        )
        rows.extend(participant_rows)
        prediction_rows.extend(participant_prediction_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")
    _core.write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_decoding(rows)
    if summary_output_path:
        _core.write_alpha_metrics_csv(summary_rows, summary_output_path)
    if participant_peaks_output_path:
        _core.write_alpha_metrics_csv(summarize_stimulus_decoding_peaks(rows), participant_peaks_output_path)  # noqa: F405
    if predictions_output_path and prediction_rows:
        _core.write_alpha_metrics_csv(prediction_rows, predictions_output_path)
    if (confusion_output_path or per_stimulus_output_path) and prediction_rows:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)  # noqa: F405
        if confusion_output_path:
            _core.write_alpha_metrics_csv(confusion_rows, confusion_output_path)
        if per_stimulus_output_path:
            _core.write_alpha_metrics_csv(per_stimulus_rows, per_stimulus_output_path)
    if plots_dir:
        write_stimulus_decoding_plots(summary_rows, plots_dir)  # noqa: F405
    return rows, summary_rows


# jscpd:ignore-start
def export_stimulus_temporal_generalization(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    config=None,
    progress=None,
):
    """Run stimulus temporal generalization and write CSV artifacts."""

    config = config or StimulusDecodingConfig()
    data_folder = _core.resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_stimulus_temporal_generalization(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    _core.write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_temporal_generalization(rows)
    if summary_output_path:
        _core.write_alpha_metrics_csv(summary_rows, summary_output_path)
    return rows, summary_rows


# jscpd:ignore-end
# jscpd:ignore-start
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def export_stimulus_onset_scan(
    data_folder,
    participants,
    output_path,
    events_output_path,
    *,
    summary_output_path=None,
    event_summary_output_path=None,
    config=None,
    train_window_center=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,  # noqa: F405
    threshold_window=DEFAULT_ONSET_THRESHOLD_WINDOW,  # noqa: F405
    threshold_quantile=DEFAULT_ONSET_THRESHOLD_QUANTILE,  # noqa: F405
    threshold_method=DEFAULT_ONSET_THRESHOLD_METHOD,  # noqa: F405
    min_consecutive=DEFAULT_ONSET_MIN_CONSECUTIVE,  # noqa: F405
    min_duration=DEFAULT_ONSET_MIN_DURATION,  # noqa: F405
    require_stable_prediction=DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,  # noqa: F405
    detection_start_s=None,
    progress=None,
):
    """Run onset-blind stimulus scanning and write trial/window and event CSVs."""

    config = _onset_scan_config(config)
    data_folder = _core.resolve_data_folder(data_folder)
    scan_rows = []
    event_rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        participant_scan_rows, participant_event_rows = evaluate_participant_stimulus_onset_scan(
            data_folder,
            participant,
            config=config,
            train_window_center=train_window_center,
            threshold_window=threshold_window,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=require_stable_prediction,
            detection_start_s=detection_start_s,
        )
        scan_rows.extend(participant_scan_rows)
        event_rows.extend(participant_event_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")

    _core.write_alpha_metrics_csv(scan_rows, output_path)
    _core.write_alpha_metrics_csv(event_rows, events_output_path)
    summary_rows = summarize_stimulus_onset_scan(scan_rows)  # noqa: F405
    if summary_output_path:
        _core.write_alpha_metrics_csv(summary_rows, summary_output_path)
    event_summary_rows = summarize_stimulus_onset_events(event_rows)  # noqa: F405
    if event_summary_output_path:
        _core.write_alpha_metrics_csv(event_summary_rows, event_summary_output_path)
    return scan_rows, event_rows, summary_rows, event_summary_rows


# jscpd:ignore-end
def _onset_scan_config(config):
    if config is not None:
        return config
    return StimulusDecodingConfig(
        window_centers=window_centers_from_range(  # noqa: F405
            DEFAULT_ONSET_SCAN_TIME_WINDOW,  # noqa: F405
            DEFAULT_ONSET_SCAN_STEP_S,  # noqa: F405
        ),
    )


def _config_for_core(config):
    config = config or StimulusDecodingConfig()
    auto_chance = _uses_auto_chance(config)
    core_chance_classes = 1 if auto_chance else _positive_int(getattr(config, "chance_classes", None))
    if core_chance_classes is None:
        raise ValueError("chance_classes must be a positive integer unless chance inference is enabled.")
    return _replace(config, chance_classes=core_chance_classes), auto_chance


def _uses_auto_chance(config):
    if not bool(getattr(config, "infer_chance_classes", True)):
        return False
    chance_classes = getattr(config, "chance_classes", None)
    if chance_classes is None:
        return True
    if isinstance(chance_classes, str):
        return chance_classes.strip().lower() in {"auto", "actual", "infer", "inferred"}
    try:
        return int(chance_classes) == int(_core.DEFAULT_CHANCE_CLASSES)
    except (TypeError, ValueError, OverflowError):
        return False


def _patch_auto_chance(rows):
    patched_rows = [dict(row) for row in rows]
    true_label_class_counts = _true_label_class_counts(patched_rows)
    for row in patched_rows:
        class_count = _row_validation_class_count(row, true_label_class_counts)
        if class_count is None:
            continue
        chance_accuracy = 1.0 / class_count
        row["chance_accuracy"] = chance_accuracy
        row["chance_percent"] = 100.0 * chance_accuracy
        if "accuracy" in row:
            accuracy = _to_float(row.get("accuracy"))
            row["above_chance"] = bool(_np.isfinite(accuracy) and accuracy > chance_accuracy)
    return patched_rows


def _patch_onset_score_columns(rows):
    """Add explicit onset-score semantics while preserving legacy columns."""

    patched_rows = []
    for row in rows:
        patched = dict(row)
        legacy_score = _to_float(patched.get("stimulus_score"))
        predicted_score = _to_float(patched.get("predicted_class_score", legacy_score))
        correct = bool(patched.get("correct", False))
        true_score = _to_float(patched.get("true_class_score", predicted_score if correct else _np.nan))
        score_margin = _to_float(patched.get("score_margin", _np.nan))
        onset_score = _to_float(patched.get("onset_score", predicted_score))
        patched["predicted_class_score"] = predicted_score
        patched["true_class_score"] = true_score
        patched["score_margin"] = score_margin
        patched["onset_score"] = onset_score
        patched["onset_score_type"] = patched.get("onset_score_type") or ONSET_SCORE_TYPE_PREDICTED_CLASS
        patched["stimulus_score"] = legacy_score if _np.isfinite(legacy_score) else onset_score
        patched_rows.append(patched)
    return patched_rows


def _patch_onset_event_score_columns(rows):
    """Add explicit score semantics to first-threshold-crossing event rows."""

    patched_rows = []
    for row in rows:
        patched = dict(row)
        detection_score = _to_float(patched.get("stimulus_score_at_detection"))
        predicted_score = _to_float(patched.get("predicted_class_score_at_detection", detection_score))
        correct = bool(patched.get("correct_detected_stimulus", False))
        true_score = _to_float(patched.get("true_class_score_at_detection", predicted_score if correct else _np.nan))
        patched["onset_score_at_detection"] = _to_float(patched.get("onset_score_at_detection", predicted_score))
        patched["onset_score_type"] = patched.get("onset_score_type") or ONSET_SCORE_TYPE_PREDICTED_CLASS
        patched["predicted_class_score_at_detection"] = predicted_score
        patched["true_class_score_at_detection"] = true_score
        patched["score_margin_at_detection"] = _to_float(patched.get("score_margin_at_detection", _np.nan))
        patched_rows.append(patched)
    return patched_rows


def summarize_stimulus_onset_scan(rows):
    """Summarize onset-blind scan rows with explicit onset-score semantics."""

    patched_rows = _patch_onset_score_columns(rows)
    summary_rows = _core.summarize_stimulus_onset_scan(patched_rows)
    grouped = _group_rows_for_onset_scan_summary(patched_rows)
    for summary_row in summary_rows:
        group_rows = grouped.get(_onset_scan_summary_key(summary_row), [])
        if not group_rows:
            continue
        onset_scores = _finite_values(row.get("onset_score") for row in group_rows)
        predicted_scores = _finite_values(row.get("predicted_class_score") for row in group_rows)
        true_scores = _finite_values(row.get("true_class_score") for row in group_rows)
        margins = _finite_values(row.get("score_margin") for row in group_rows)
        summary_row.update(
            {
                "onset_score_type": group_rows[0].get("onset_score_type", ONSET_SCORE_TYPE_PREDICTED_CLASS),
                "mean_onset_score": _mean_or_nan(onset_scores),
                "median_onset_score": _median_or_nan(onset_scores),
                "mean_predicted_class_score": _mean_or_nan(predicted_scores),
                "median_predicted_class_score": _median_or_nan(predicted_scores),
                "mean_true_class_score": _mean_or_nan(true_scores),
                "median_true_class_score": _median_or_nan(true_scores),
                "mean_score_margin": _mean_or_nan(margins),
                "median_score_margin": _median_or_nan(margins),
            }
        )
    return summary_rows


def summarize_stimulus_onset_events(rows):
    """Summarize first-detection event rows with explicit onset-score semantics."""

    patched_rows = _patch_onset_event_score_columns(rows)
    summary_rows = _core.summarize_stimulus_onset_events(patched_rows)
    grouped = _group_rows_for_onset_event_summary(patched_rows)
    for summary_row in summary_rows:
        group_rows = grouped.get(_onset_event_summary_key(summary_row), [])
        if not group_rows:
            continue
        detection_scores = _finite_values(row.get("onset_score_at_detection") for row in group_rows)
        predicted_scores = _finite_values(row.get("predicted_class_score_at_detection") for row in group_rows)
        true_scores = _finite_values(row.get("true_class_score_at_detection") for row in group_rows)
        summary_row.update(
            {
                "onset_score_type": group_rows[0].get("onset_score_type", ONSET_SCORE_TYPE_PREDICTED_CLASS),
                "onset_score_at_detection_mean": _mean_or_nan(detection_scores),
                "predicted_class_score_at_detection_mean": _mean_or_nan(predicted_scores),
                "true_class_score_at_detection_mean": _mean_or_nan(true_scores),
            }
        )
    return summary_rows


def _group_rows_for_onset_scan_summary(rows):
    return _group_rows_by_summary_key(rows, _onset_scan_summary_key)


def _group_rows_for_onset_event_summary(rows):
    return _group_rows_by_summary_key(rows, _onset_event_summary_key)


def _group_rows_by_summary_key(rows, key_factory):
    grouped: dict[tuple[object, ...], list[dict[str, _typing.Any]]] = {}
    for row in rows:
        grouped.setdefault(key_factory(row), []).append(row)
    return grouped


_ONSET_BASE_KEY_FIELDS = (
    "participant",
    "variant",
    "transfer_direction",
    "train_window_center_s",
    "threshold_method",
    "min_consecutive",
    "min_duration_s",
    "require_stable_prediction",
)
_ONSET_MODEL_KEY_FIELDS = ("classifier", "components_pca", "frequency_low_hz", "frequency_high_hz")


def _summary_key_from_fields(row, fields):
    return tuple(row.get(field) for field in fields)


def _onset_scan_summary_key(row):
    return _summary_key_from_fields(row, (*_ONSET_BASE_KEY_FIELDS, "scan_window_center_s", *_ONSET_MODEL_KEY_FIELDS))


def _onset_event_summary_key(row):
    return _summary_key_from_fields(row, (*_ONSET_BASE_KEY_FIELDS, *_ONSET_MODEL_KEY_FIELDS))


def _finite_values(values):
    finite = []
    for value in values:
        parsed = _to_float(value)
        if _np.isfinite(parsed):
            finite.append(parsed)
    return finite


def _mean_or_nan(values):
    return float(_np.mean(values)) if values else _np.nan


def _median_or_nan(values):
    return float(_np.median(values)) if values else _np.nan


def _row_validation_class_count(row, true_label_class_counts):
    class_count = _positive_int(row.get("n_validation_classes"))
    if class_count is not None:
        return class_count
    return true_label_class_counts.get(_chance_group_key(row))


def _true_label_class_counts(rows):
    labels_by_group: dict[tuple[object, object, object], set[object]] = {}
    for row in rows:
        if "true_label" not in row:
            continue
        labels_by_group.setdefault(_chance_group_key(row), set()).add(row["true_label"])
    return {
        group_key: len(labels)
        for group_key, labels in labels_by_group.items()
        if labels
    }


def _chance_group_key(row):
    return (
        row.get("participant"),
        row.get("variant"),
        row.get("transfer_direction"),
    )


def _positive_int(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return _np.nan


_existing_core_originals = getattr(_core, "_PYMEGDEC_AUTO_CHANCE_ORIGINALS", None)
_CORE_AUTO_CHANCE_ORIGINALS: dict[str, _typing.Any]
if isinstance(_existing_core_originals, dict):
    _CORE_AUTO_CHANCE_ORIGINALS = _existing_core_originals
else:
    _CORE_AUTO_CHANCE_ORIGINALS = {
        "evaluate_time_resolved_stimulus_transfer": _core.evaluate_time_resolved_stimulus_transfer,
        "evaluate_participant_time_resolved_stimulus_transfer": _core.evaluate_participant_time_resolved_stimulus_transfer,
        "evaluate_participant_stimulus_decoding_diagnostics": _core.evaluate_participant_stimulus_decoding_diagnostics,
        "evaluate_participant_stimulus_temporal_generalization": _core.evaluate_participant_stimulus_temporal_generalization,
        "evaluate_participant_stimulus_onset_scan": _core.evaluate_participant_stimulus_onset_scan,
    }
    setattr(_core, "_PYMEGDEC_AUTO_CHANCE_ORIGINALS", _CORE_AUTO_CHANCE_ORIGINALS)


def _core_evaluate_time_resolved_stimulus_transfer(data_folder, participants, *, config=None, progress=None):
    core_config, auto_chance = _config_for_core(config)
    rows = _CORE_AUTO_CHANCE_ORIGINALS["evaluate_time_resolved_stimulus_transfer"](
        data_folder,
        participants,
        config=core_config,
        progress=progress,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


def _core_evaluate_participant_time_resolved_stimulus_transfer(data_folder, participant, *, config=None):
    core_config, auto_chance = _config_for_core(config)
    rows = _CORE_AUTO_CHANCE_ORIGINALS["evaluate_participant_time_resolved_stimulus_transfer"](
        data_folder,
        participant,
        config=core_config,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


def _core_evaluate_participant_stimulus_decoding_diagnostics(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    core_config, auto_chance = _config_for_core(config)
    rows, prediction_rows = _CORE_AUTO_CHANCE_ORIGINALS["evaluate_participant_stimulus_decoding_diagnostics"](
        data_folder,
        participant,
        config=core_config,
        diagnostic_window_centers=diagnostic_window_centers,
    )
    return (_patch_auto_chance(rows) if auto_chance else rows), prediction_rows


def _core_evaluate_participant_stimulus_temporal_generalization(data_folder, participant, *, config=None):
    core_config, auto_chance = _config_for_core(config)
    rows = _CORE_AUTO_CHANCE_ORIGINALS["evaluate_participant_stimulus_temporal_generalization"](
        data_folder,
        participant,
        config=core_config,
    )
    return _patch_auto_chance(rows) if auto_chance else rows


def _core_evaluate_participant_stimulus_onset_scan(
    data_folder,
    participant,
    *,
    config=None,
    train_window_center=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,  # noqa: F405
    threshold_window=DEFAULT_ONSET_THRESHOLD_WINDOW,  # noqa: F405
    threshold_quantile=DEFAULT_ONSET_THRESHOLD_QUANTILE,  # noqa: F405
    threshold_method=DEFAULT_ONSET_THRESHOLD_METHOD,  # noqa: F405
    min_consecutive=DEFAULT_ONSET_MIN_CONSECUTIVE,  # noqa: F405
    min_duration=DEFAULT_ONSET_MIN_DURATION,  # noqa: F405
    require_stable_prediction=DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,  # noqa: F405
    detection_start_s=None,
):
    config = _onset_scan_config(config)
    core_config, auto_chance = _config_for_core(config)
    scan_rows, event_rows = _CORE_AUTO_CHANCE_ORIGINALS["evaluate_participant_stimulus_onset_scan"](
        data_folder,
        participant,
        config=core_config,
        train_window_center=train_window_center,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        threshold_method=threshold_method,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
        detection_start_s=detection_start_s,
    )
    scan_rows = _patch_onset_score_columns(scan_rows)
    event_rows = _patch_onset_event_score_columns(event_rows)
    return (_patch_auto_chance(scan_rows) if auto_chance else scan_rows), event_rows


def _install_core_private_import_fixes():
    """Make direct private-core imports use the public chance-level semantics."""

    _core.evaluate_time_resolved_stimulus_transfer = _core_evaluate_time_resolved_stimulus_transfer
    _core.evaluate_participant_time_resolved_stimulus_transfer = _core_evaluate_participant_time_resolved_stimulus_transfer
    _core.evaluate_participant_stimulus_decoding_diagnostics = _core_evaluate_participant_stimulus_decoding_diagnostics
    _core.evaluate_participant_stimulus_temporal_generalization = _core_evaluate_participant_stimulus_temporal_generalization
    _core.evaluate_participant_stimulus_onset_scan = _core_evaluate_participant_stimulus_onset_scan
    _core.export_time_resolved_stimulus_decoding = export_time_resolved_stimulus_decoding
    _core.export_stimulus_temporal_generalization = export_stimulus_temporal_generalization
    _core.export_stimulus_onset_scan = export_stimulus_onset_scan


_install_core_private_import_fixes()


def __getattr__(name):
    """Delegate private legacy helpers for compatibility with existing tests/scripts."""

    return getattr(_core, name)


for _name in dir(_core):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_core, _name)


__all__ = [name for name in globals() if not name.startswith("_")]
