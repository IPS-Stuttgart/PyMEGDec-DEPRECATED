"""Time-resolved stimulus decoding analyses."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.preprocessing import (
    downsample_data,
    extract_windows,
    filter_features,
    reduce_features_pca,
)
from reptrace.decoding.temporal_generalization import TemporalFeatureWindow, compute_temporal_generalization_matrix  # pylint: disable=no-name-in-module
from reptrace.metrics.confusion import confusion_counts, per_class_accuracy  # pylint: disable=no-name-in-module
from reptrace.onset_detection import annotate_threshold_crossings, detect_onsets
from reptrace.results.tables import peak_metric_rows, summarize_metric_table  # pylint: disable=no-name-in-module

DEFAULT_DECODING_TIME_WINDOW = (-0.2, 0.6)
DEFAULT_DECODING_STEP_S = 0.05
DEFAULT_STIMULUS_WINDOW_SIZE = 0.1
DEFAULT_CHANCE_CLASSES = 16
DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER = 0.175
DEFAULT_ONSET_SCAN_TIME_WINDOW = (-0.4, 0.8)
DEFAULT_ONSET_SCAN_STEP_S = 0.025
DEFAULT_ONSET_THRESHOLD_WINDOW = (-0.35, -0.05)
DEFAULT_ONSET_THRESHOLD_QUANTILE = 0.95
TRANSFER_DIRECTIONS = ("main-to-cue", "cue-to-main")
SUMMARY_GROUP_FIELDS = (
    "control",
    "control_label",
    "transfer_direction",
    "variant",
    "window_center_s",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)
TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS = (
    "transfer_direction",
    "variant",
    "train_window_center_s",
    "test_window_center_s",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)
TEMPORAL_GENERALIZATION_ROW_COLUMNS = (
    "participant",
    "variant",
    "transfer_direction",
    "train_window_center_s",
    "train_window_start_s",
    "train_window_stop_s",
    "test_window_center_s",
    "test_window_start_s",
    "test_window_stop_s",
    "is_diagonal",
    "accuracy",
    "percent",
    "chance_accuracy",
    "chance_percent",
    "above_chance",
    "n_train_trials",
    "n_validation_trials",
    "n_train_classes",
    "n_validation_classes",
    "classifier",
    "classifier_param",
    "components_pca",
    "actual_components_pca",
    "pca_explained_variance_percent",
    "frequency_low_hz",
    "frequency_high_hz",
)
DEFAULT_WINDOW_CENTERS = tuple(
    float(value)
    for value in np.round(
        np.arange(
            DEFAULT_DECODING_TIME_WINDOW[0],
            DEFAULT_DECODING_TIME_WINDOW[1] + DEFAULT_DECODING_STEP_S / 2,
            DEFAULT_DECODING_STEP_S,
        ),
        10,
    )
)


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class StimulusDecodingConfig:
    """Parameters for time-resolved stimulus decoding."""

    window_centers: tuple[float, ...] = DEFAULT_WINDOW_CENTERS
    window_size: float = DEFAULT_STIMULUS_WINDOW_SIZE
    null_window_center: float = float("nan")
    new_framerate: float = float("inf")
    classifier: str = "multiclass-svm"
    classifier_param: object = float("nan")
    components_pca: int | float = 100
    frequency_range: tuple[float, float] = (0.0, float("inf"))
    chance_classes: int = DEFAULT_CHANCE_CLASSES
    random_state: int | None = None
    permutations: int = 0
    permutation_seed: int | None = None
    transfer_direction: str = "main-to-cue"


def window_centers_from_range(time_window: tuple[float, float], step_s: float) -> tuple[float, ...]:
    """Build evenly spaced window centers from a start/stop range."""

    start, stop = time_window
    if step_s <= 0:
        raise ValueError("Window step must be positive.")
    if start > stop:
        raise ValueError("Time window start must be before stop.")
    return tuple(float(value) for value in np.round(np.arange(start, stop + step_s / 2, step_s), 10))


def evaluate_time_resolved_stimulus_transfer(
    data_folder,
    participants,
    *,
    config=None,
    progress=None,
):
    """Evaluate train-main/validate-cue stimulus decoding across time windows."""

    config = config or StimulusDecodingConfig()
    data_folder = resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_time_resolved_stimulus_transfer(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    return rows


def evaluate_participant_time_resolved_stimulus_transfer(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate one participant's stimulus transfer accuracy across window centers."""

    rows, _ = _evaluate_participant_time_resolved_stimulus_transfer(
        data_folder,
        participant,
        config=config,
        diagnostic_window_centers=(),
    )
    return rows


def evaluate_participant_stimulus_decoding_diagnostics(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    """Evaluate one participant and return accuracy rows plus prediction diagnostics."""

    return _evaluate_participant_time_resolved_stimulus_transfer(
        data_folder,
        participant,
        config=config,
        diagnostic_window_centers=diagnostic_window_centers,
    )


# jscpd:ignore-start
def evaluate_participant_stimulus_temporal_generalization(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate train-time/test-time stimulus decoding for one participant."""

    config = config or StimulusDecodingConfig()
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    train_cue, validation_cue = _transfer_direction_cue_flags(config.transfer_direction)
    train_data = _load_participant_data(data_folder, participant, cue=train_cue)
    validation_data = _load_participant_data(data_folder, participant, cue=validation_cue)
    _check_matching_sample_rate(train_data, validation_data)

    labels_train = np.asarray(train_data["trialinfo"][0][0], dtype=int).ravel()
    labels_validation = np.asarray(validation_data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(config.null_window_center):
        labels_train = labels_train - 1
        labels_validation = labels_validation - 1

    if not np.array_equal(np.unique(labels_train), np.unique(labels_validation)):
        warnings.warn("There are labels in the training or validation experiment that are not in the other experiment.")

    train_data = _prepare_data(train_data, config)
    validation_data = _prepare_data(validation_data, config)
    train_windows = [
        _temporal_feature_window(float(window_center), labels_train, config)
        for window_center in config.window_centers
    ]
    test_windows = [
        _temporal_feature_window(
            float(window_center),
            labels_validation,
            config,
            features=_validation_features_for_window(validation_data, float(window_center), config),
        )
        for window_center in config.window_centers
    ]

    variant = "without_null" if np.isnan(config.null_window_center) else "with_null"
    matrix = compute_temporal_generalization_matrix(
        train_windows,
        test_windows,
        fit_model=lambda window: _train_window_model(
            train_data,
            labels_train,
            float(window.center),
            classifier_param,
            config,
        ),
        predict_labels=lambda model_bundle, window: _predict_window_model(model_bundle, window.features)[0],
        chance_accuracy=1.0 / config.chance_classes,
        metadata={
            "participant": participant,
            "variant": variant,
            "transfer_direction": config.transfer_direction,
            "classifier": config.classifier,
            "classifier_param": classifier_param,
            "components_pca": config.components_pca,
            "frequency_low_hz": config.frequency_range[0],
            "frequency_high_hz": config.frequency_range[1],
        },
        model_metadata=lambda model_bundle: {
            "actual_components_pca": model_bundle.actual_components_pca,
            "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        },
    )
    return matrix[list(TEMPORAL_GENERALIZATION_ROW_COLUMNS)].to_dict(orient="records")


# jscpd:ignore-end
def _evaluate_participant_time_resolved_stimulus_transfer(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    config = config or StimulusDecodingConfig()
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    train_cue, validation_cue = _transfer_direction_cue_flags(config.transfer_direction)
    train_data = _load_participant_data(data_folder, participant, cue=train_cue)
    validation_data = _load_participant_data(data_folder, participant, cue=validation_cue)
    _check_matching_sample_rate(train_data, validation_data)

    labels_train = np.asarray(train_data["trialinfo"][0][0], dtype=int).ravel()
    labels_validation = np.asarray(validation_data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(config.null_window_center):
        labels_train = labels_train - 1
        labels_validation = labels_validation - 1

    if not np.array_equal(np.unique(labels_train), np.unique(labels_validation)):
        warnings.warn("There are labels in the training or validation experiment " "that are not in the other experiment.")

    train_data = _prepare_data(train_data, config)
    validation_data = _prepare_data(validation_data, config)
    permutation_rng = np.random.default_rng(config.permutation_seed)
    diagnostic_centers = _window_center_set(diagnostic_window_centers or ())

    rows = []
    prediction_rows = []
    for window_center in config.window_centers:
        include_predictions = _window_center_key(window_center) in diagnostic_centers
        result = _evaluate_window(
            train_data,
            validation_data,
            labels_train,
            labels_validation,
            participant,
            float(window_center),
            classifier_param,
            config,
            permutation_rng=permutation_rng,
            include_predictions=include_predictions,
        )
        if include_predictions:
            row, window_prediction_rows = result
            rows.append(row)
            prediction_rows.extend(window_prediction_rows)
        else:
            rows.append(result)
    return rows, prediction_rows


def summarize_stimulus_decoding(rows):
    """Summarize decoding rows across participants for each window center."""

    if not rows:
        return []

    group_fields = _present_group_fields(rows, SUMMARY_GROUP_FIELDS)
    frame = _rows_frame(rows)
    metric_summary = summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_to_float(row["accuracy"]) for row in group_rows]
        std = _legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _legacy_sem(base_summary["accuracy_sem"], accuracies)
        permutation_p = [_to_float(row.get("permutation_p_value")) for row in group_rows]
        n_with_permutation = sum(np.isfinite(permutation_p))
        significant_05 = sum(value < 0.05 for value in permutation_p if np.isfinite(value))
        significant_01 = sum(value < 0.01 for value in permutation_p if np.isfinite(value))
        chance_accuracy = _to_float(group_rows[0]["chance_accuracy"])
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": len(group_rows),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": int(base_summary["accuracy_above_chance_count"]),
                "n_with_permutation": int(n_with_permutation),
                "n_significant_p_0.05": int(significant_05),
                "n_significant_p_0.01": int(significant_01),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_stimulus_decoding_peaks(rows):
    """Return the best decoding window per participant and variant."""

    if not rows:
        return []

    group_fields = _present_group_fields(rows, ("control", "control_label", "transfer_direction", "variant", "participant"))
    peaks = peak_metric_rows(_rows_frame(rows), "accuracy", group_fields, time_column="window_center_s", prefer_time=0.0)
    peak_rows = []
    for peak in peaks.to_dict("records"):
        peak_row = {field: peak.get(field, "") for field in group_fields}
        peak_row.update(
            {
                "peak_window_center_s": peak["window_center_s"],
                "peak_window_start_s": peak["window_start_s"],
                "peak_window_stop_s": peak["window_stop_s"],
                "peak_accuracy": peak["accuracy"],
                "peak_percent": peak["percent"],
                "chance_accuracy": peak["chance_accuracy"],
                "chance_percent": peak["chance_percent"],
            }
        )
        peak_rows.append(peak_row)
    return peak_rows


def summarize_stimulus_prediction_diagnostics(prediction_rows):
    """Summarize trial-level prediction diagnostics."""

    if not prediction_rows:
        return [], []

    group_fields = _present_group_fields(prediction_rows, ("control", "control_label", "transfer_direction", "variant", "window_center_s"))
    frame = _rows_frame(prediction_rows)
    participant_column = "participant" if "participant" in frame.columns else None

    confusion_rows = []
    confusion_summary = confusion_counts(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=group_fields,
    )
    for summary in confusion_summary.to_dict("records"):
        row = {field: summary.get(field, "") for field in group_fields}
        row.update(
            {
                "true_stimulus": summary["true_label"],
                "predicted_stimulus": summary["predicted_label"],
                "count": summary["count"],
            }
        )
        confusion_rows.append(row)

    per_stimulus_rows = []
    per_stimulus_summary = per_class_accuracy(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column=participant_column,
        group_columns=group_fields,
    )
    for summary in per_stimulus_summary.to_dict("records"):
        accuracy = summary["accuracy"]
        row = {field: summary.get(field, "") for field in group_fields}
        row.update(
            {
                "true_stimulus": summary["true_label"],
                "n_participants": summary.get("n_participants", np.nan),
                "n_trials": summary["n_trials"],
                "n_correct": summary["n_correct"],
                "accuracy": accuracy,
                "percent": 100.0 * accuracy,
            }
        )
        per_stimulus_rows.append(row)
    return confusion_rows, per_stimulus_rows


# jscpd:ignore-start
def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows across participants."""

    if not rows:
        return []

    group_fields = _present_group_fields(rows, TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    frame = _rows_frame(rows)
    metric_summary = summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_to_float(row["accuracy"]) for row in group_rows]
        std = _legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _legacy_sem(base_summary["accuracy_sem"], accuracies)
        chance_accuracy = _to_float(group_rows[0]["chance_accuracy"])
        diagonal_values = {_window_center_key(row["train_window_center_s"]) == _window_center_key(row["test_window_center_s"]) for row in group_rows}
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": len(group_rows),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": int(base_summary["accuracy_above_chance_count"]),
                "is_diagonal": bool(diagonal_values == {True}),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


# jscpd:ignore-end
def write_stimulus_decoding_plots(summary_rows, output_dir):
    """Write group-level stimulus decoding plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_group_accuracy(summary_rows, output_dir / "stimulus_decoding_accuracy.png")


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
    data_folder = resolve_data_folder(data_folder)
    rows = []
    prediction_rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        participant_rows, participant_prediction_rows = _evaluate_participant_time_resolved_stimulus_transfer(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=diagnostic_window_centers,
        )
        rows.extend(participant_rows)
        prediction_rows.extend(participant_prediction_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")
    write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_decoding(rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    if participant_peaks_output_path:
        write_alpha_metrics_csv(summarize_stimulus_decoding_peaks(rows), participant_peaks_output_path)
    if predictions_output_path and prediction_rows:
        write_alpha_metrics_csv(prediction_rows, predictions_output_path)
    if (confusion_output_path or per_stimulus_output_path) and prediction_rows:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)
        if confusion_output_path:
            write_alpha_metrics_csv(confusion_rows, confusion_output_path)
        if per_stimulus_output_path:
            write_alpha_metrics_csv(per_stimulus_rows, per_stimulus_output_path)
    if plots_dir:
        write_stimulus_decoding_plots(summary_rows, plots_dir)
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
    data_folder = resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_stimulus_temporal_generalization(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_temporal_generalization(rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    return rows, summary_rows


# jscpd:ignore-end
# jscpd:ignore-start
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def evaluate_participant_stimulus_onset_scan(
    data_folder,
    participant,
    *,
    config=None,
    train_window_center=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,
    threshold_window=DEFAULT_ONSET_THRESHOLD_WINDOW,
    threshold_quantile=DEFAULT_ONSET_THRESHOLD_QUANTILE,
    detection_start_s=None,
):
    """Scan validation trials for stimulus identity without using onset at test time."""

    config = config or StimulusDecodingConfig(
        window_centers=window_centers_from_range(DEFAULT_ONSET_SCAN_TIME_WINDOW, DEFAULT_ONSET_SCAN_STEP_S),
    )
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    train_cue, validation_cue = _transfer_direction_cue_flags(config.transfer_direction)
    train_data = _load_participant_data(data_folder, participant, cue=train_cue)
    validation_data = _load_participant_data(data_folder, participant, cue=validation_cue)
    _check_matching_sample_rate(train_data, validation_data)

    labels_train = np.asarray(train_data["trialinfo"][0][0], dtype=int).ravel()
    labels_validation = np.asarray(validation_data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(config.null_window_center):
        labels_train = labels_train - 1
        labels_validation = labels_validation - 1

    train_data = _prepare_data(train_data, config)
    validation_data = _prepare_data(validation_data, config)
    model_bundle = _train_window_model(
        train_data,
        labels_train,
        float(train_window_center),
        classifier_param,
        config,
    )

    variant = "without_null" if np.isnan(config.null_window_center) else "with_null"
    scan_rows = []
    for scan_window_center in config.window_centers:
        scan_window_center = float(scan_window_center)
        validation_features = _validation_features_for_window(validation_data, scan_window_center, config)
        predictions, scores = _predict_window_model(model_bundle, validation_features)
        scan_rows.extend(
            _stimulus_onset_scan_rows(
                participant,
                variant,
                float(train_window_center),
                scan_window_center,
                labels_validation,
                predictions,
                scores,
                classifier_param,
                model_bundle,
                config,
                threshold_window,
                threshold_quantile,
            )
        )

    scan_rows = _annotate_stimulus_onset_scan_with_reptrace(
        scan_rows,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
    )
    event_rows = _stimulus_onset_event_rows_from_reptrace(
        scan_rows,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        detection_start_s=detection_start_s,
    )
    return scan_rows, event_rows


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
    train_window_center=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,
    threshold_window=DEFAULT_ONSET_THRESHOLD_WINDOW,
    threshold_quantile=DEFAULT_ONSET_THRESHOLD_QUANTILE,
    detection_start_s=None,
    progress=None,
):
    """Run onset-blind stimulus scanning and write trial/window and event CSVs."""

    config = config or StimulusDecodingConfig(
        window_centers=window_centers_from_range(DEFAULT_ONSET_SCAN_TIME_WINDOW, DEFAULT_ONSET_SCAN_STEP_S),
    )
    data_folder = resolve_data_folder(data_folder)
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
            detection_start_s=detection_start_s,
        )
        scan_rows.extend(participant_scan_rows)
        event_rows.extend(participant_event_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")

    write_alpha_metrics_csv(scan_rows, output_path)
    write_alpha_metrics_csv(event_rows, events_output_path)
    summary_rows = summarize_stimulus_onset_scan(scan_rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    event_summary_rows = summarize_stimulus_onset_events(event_rows)
    if event_summary_output_path:
        write_alpha_metrics_csv(event_summary_rows, event_summary_output_path)
    return scan_rows, event_rows, summary_rows, event_summary_rows


def summarize_stimulus_onset_scan(rows):
    """Summarize onset-blind scan rows by participant and scan window."""

    group_fields = _present_group_fields(
        rows,
        (
            "participant",
            "variant",
            "transfer_direction",
            "train_window_center_s",
            "scan_window_center_s",
            "classifier",
            "components_pca",
            "frequency_low_hz",
            "frequency_high_hz",
        ),
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        correct = [bool(row["correct"]) for row in group_rows]
        scores = [_to_float(row.get("stimulus_score")) for row in group_rows]
        finite_scores = [score for score in scores if np.isfinite(score)]
        above_threshold = [bool(row.get("above_threshold", False)) for row in group_rows]
        accuracy = float(np.mean(correct)) if correct else np.nan
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_trials": len(group_rows),
                "accuracy": accuracy,
                "percent": 100.0 * accuracy,
                "mean_stimulus_score": float(np.mean(finite_scores)) if finite_scores else np.nan,
                "median_stimulus_score": float(np.median(finite_scores)) if finite_scores else np.nan,
                "above_threshold_count": sum(above_threshold),
                "above_threshold_rate": float(np.mean(above_threshold)) if above_threshold else np.nan,
                "score_threshold": group_rows[0].get("score_threshold", np.nan),
                "threshold_quantile": group_rows[0].get("threshold_quantile", np.nan),
                "threshold_window_start_s": group_rows[0].get("threshold_window_start_s", np.nan),
                "threshold_window_stop_s": group_rows[0].get("threshold_window_stop_s", np.nan),
                "chance_accuracy": group_rows[0].get("chance_accuracy", np.nan),
                "chance_percent": group_rows[0].get("chance_percent", np.nan),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_stimulus_onset_events(rows):
    """Summarize first-detection event rows by participant."""

    group_fields = _present_group_fields(
        rows,
        (
            "participant",
            "variant",
            "transfer_direction",
            "train_window_center_s",
            "classifier",
            "components_pca",
            "frequency_low_hz",
            "frequency_high_hz",
        ),
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        detected = [bool(row["detected"]) for row in group_rows]
        correct_detection = [bool(row["detected"]) and bool(row["correct_detected_stimulus"]) for row in group_rows]
        false_alarm = [bool(row["detected_before_stimulus"]) for row in group_rows]
        post_detection_rows = [row for row in group_rows if bool(row["detected"]) and not bool(row["detected_before_stimulus"])]
        post_latencies = [_to_float(row["detection_latency_s"]) for row in post_detection_rows]
        post_latencies = [value for value in post_latencies if np.isfinite(value)]
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_trials": len(group_rows),
                "detected_count": sum(detected),
                "detected_rate": float(np.mean(detected)) if detected else np.nan,
                "false_alarm_count": sum(false_alarm),
                "false_alarm_rate": float(np.mean(false_alarm)) if false_alarm else np.nan,
                "post_stimulus_detected_count": len(post_detection_rows),
                "post_stimulus_detected_rate": len(post_detection_rows) / len(group_rows) if group_rows else np.nan,
                "correct_detection_count": sum(correct_detection),
                "correct_detection_rate": float(np.mean(correct_detection)) if correct_detection else np.nan,
                "post_detection_latency_mean_s": float(np.mean(post_latencies)) if post_latencies else np.nan,
                "post_detection_latency_median_s": float(np.median(post_latencies)) if post_latencies else np.nan,
                "score_threshold": group_rows[0].get("score_threshold", np.nan),
                "threshold_quantile": group_rows[0].get("threshold_quantile", np.nan),
                "threshold_window_start_s": group_rows[0].get("threshold_window_start_s", np.nan),
                "threshold_window_stop_s": group_rows[0].get("threshold_window_stop_s", np.nan),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


# jscpd:ignore-end
def _load_participant_data(data_folder, participant, *, cue):
    suffix = "CueData" if cue else "Data"
    path = Path(data_folder) / f"Part{participant}{suffix}.mat"
    return sio.loadmat(path)["data"][0]


def _transfer_direction_cue_flags(transfer_direction):
    if transfer_direction == "main-to-cue":
        return False, True
    if transfer_direction == "cue-to-main":
        return True, False
    supported = ", ".join(TRANSFER_DIRECTIONS)
    raise ValueError(f"Unsupported transfer direction: {transfer_direction}. Supported directions: {supported}")


def _check_matching_sample_rate(train_data, validation_data):
    train_sample_interval = np.diff(train_data["time"][0][0][0][0, :2])
    validation_sample_interval = np.diff(validation_data["time"][0][0][0][0, :2])
    if not np.allclose(train_sample_interval, validation_sample_interval):
        raise ValueError("Sampling rate of the two experiments must match.")


def _prepare_data(data, config):
    data = filter_features(data, config.frequency_range[0], config.frequency_range[1])
    if config.new_framerate != float("inf"):
        data = downsample_data(data, config.new_framerate)
    return data


# jscpd:ignore-start
@dataclass(frozen=True)
class _WindowModelBundle:
    model: object
    train_window: tuple[float, float]
    train_labels: np.ndarray
    pca_coeff: np.ndarray | None
    train_features_mean: np.ndarray | None
    explained_variance_percent: float
    actual_components_pca: int


def _train_window_model(train_data, labels_train, window_center, classifier_param, config):
    train_window = _centered_window(window_center, config.window_size)
    null_window = _null_window(config)
    train_stimuli_features, train_null_features = extract_windows(train_data, train_window, null_window)
    train_features = np.hstack(train_stimuli_features + train_null_features).T
    train_labels = labels_train
    if train_null_features:
        train_labels = np.concatenate((labels_train, np.zeros(len(train_null_features), dtype=int)))

    pca_components = _actual_pca_components(config.components_pca, train_features)
    pca_coeff = None
    train_features_mean = None
    explained_variance = np.nan
    if config.components_pca != float("inf"):
        train_features, pca_coeff, train_features_mean, explained_variance = reduce_features_pca(train_features, int(config.components_pca))

    model = train_multiclass_classifier(
        train_features,
        train_labels,
        config.classifier,
        classifier_param,
        random_state=config.random_state,
    )
    return _WindowModelBundle(
        model=model,
        train_window=train_window,
        train_labels=train_labels,
        pca_coeff=pca_coeff,
        train_features_mean=train_features_mean,
        explained_variance_percent=explained_variance,
        actual_components_pca=pca_components,
    )


def _validation_features_for_window(validation_data, window_center, config):
    test_window = _centered_window(window_center, config.window_size)
    validation_stimuli_features, _ = extract_windows(validation_data, test_window, (np.nan, np.nan))
    return np.hstack(validation_stimuli_features).T


def _temporal_feature_window(window_center, labels, config, *, features=None):
    window = _centered_window(window_center, config.window_size)
    return TemporalFeatureWindow(
        center=window_center,
        features=features,
        labels=np.asarray(labels),
        start=window[0],
        stop=window[1],
    )


def _predict_window_model(model_bundle, features):
    if model_bundle.pca_coeff is not None:
        features = (features - model_bundle.train_features_mean) @ model_bundle.pca_coeff[:, : model_bundle.actual_components_pca]
    predictions = model_bundle.model.predict(features)
    scores = _prediction_scores(model_bundle.model, features)
    return predictions, scores


def _prediction_scores(model, features):
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            return np.abs(scores)
        return np.max(scores, axis=1)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(features), dtype=float)
        return np.max(scores, axis=1)
    return np.full(features.shape[0], np.nan, dtype=float)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _stimulus_onset_scan_rows(
    participant,
    variant,
    train_window_center,
    scan_window_center,
    labels_validation,
    predictions,
    scores,
    classifier_param,
    model_bundle,
    config,
    threshold_window,
    threshold_quantile,
):
    scan_window = _centered_window(scan_window_center, config.window_size)
    chance_accuracy = 1.0 / config.chance_classes
    rows = []
    for trial_idx, (true_label, predicted_label, score) in enumerate(zip(labels_validation, predictions, scores)):
        true_stimulus = _display_stimulus_label(true_label, variant)
        predicted_stimulus = _display_stimulus_label(predicted_label, variant)
        rows.append(
            {
                "participant": participant,
                "variant": variant,
                "transfer_direction": config.transfer_direction,
                "train_window_center_s": train_window_center,
                "train_window_start_s": model_bundle.train_window[0],
                "train_window_stop_s": model_bundle.train_window[1],
                "scan_window_center_s": scan_window_center,
                "scan_window_start_s": scan_window[0],
                "scan_window_stop_s": scan_window[1],
                "trial": trial_idx,
                "validation_trial_index": trial_idx,
                "validation_trial_number": trial_idx + 1,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "true_stimulus_id": true_stimulus,
                "predicted_stimulus_id": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
                "stimulus_score": float(score),
                "score_threshold": np.nan,
                "above_threshold": False,
                "threshold_quantile": threshold_quantile,
                "threshold_window_start_s": threshold_window[0],
                "threshold_window_stop_s": threshold_window[1],
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "classifier": config.classifier,
                "classifier_param": classifier_param,
                "components_pca": config.components_pca,
                "actual_components_pca": model_bundle.actual_components_pca,
                "pca_explained_variance_percent": model_bundle.explained_variance_percent,
                "frequency_low_hz": config.frequency_range[0],
                "frequency_high_hz": config.frequency_range[1],
            }
        )
    return rows


def _stimulus_score_observation_frame(scan_rows):
    frame = pd.DataFrame(scan_rows)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["sequence_id"] = frame["validation_trial_index"]
    frame["time"] = frame["scan_window_center_s"]
    frame["window_start"] = frame["scan_window_start_s"]
    frame["window_stop"] = frame["scan_window_stop_s"]
    frame["true_class"] = frame["true_stimulus_id"]
    frame["predicted_class"] = frame["predicted_stimulus_id"]
    frame["is_correct"] = frame["correct"].astype(bool)
    return frame


def _annotate_stimulus_onset_scan_with_reptrace(scan_rows, *, threshold_window, threshold_quantile):
    if not scan_rows:
        return []

    original_columns = list(pd.DataFrame(scan_rows).columns)
    observations = _stimulus_score_observation_frame(scan_rows)
    thresholded = annotate_threshold_crossings(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column="stimulus_score",
    )
    thresholded["threshold_window_start_s"] = thresholded["threshold_window_start"]
    thresholded["threshold_window_stop_s"] = thresholded["threshold_window_stop"]
    return thresholded[original_columns].to_dict(orient="records")


def _stimulus_onset_event_rows_from_reptrace(scan_rows, *, threshold_window, threshold_quantile, detection_start_s=None):
    if not scan_rows:
        return []

    observations = _stimulus_score_observation_frame(scan_rows)
    events = detect_onsets(
        observations,
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column="stimulus_score",
        detection_start=detection_start_s,
    )
    reference_rows = (
        observations.sort_values(["sequence_id", "time"])
        .groupby("sequence_id", sort=True)
        .first()
        .to_dict(orient="index")
    )
    return [
        _stimulus_onset_event_row_from_reptrace(reference_rows[event["sequence_id"]], event, detection_start_s)
        for event in events.to_dict(orient="records")
    ]


def _stimulus_onset_event_row_from_reptrace(reference_row, event, detection_start_s):
    detected = bool(event["detected"])
    return {
        "participant": reference_row["participant"],
        "variant": reference_row["variant"],
        "transfer_direction": reference_row["transfer_direction"],
        "train_window_center_s": reference_row["train_window_center_s"],
        "train_window_start_s": reference_row["train_window_start_s"],
        "train_window_stop_s": reference_row["train_window_stop_s"],
        "validation_trial_index": reference_row["validation_trial_index"],
        "validation_trial_number": reference_row["validation_trial_number"],
        "true_label": reference_row["true_label"],
        "true_stimulus": reference_row["true_stimulus"],
        "true_stimulus_id": reference_row["true_stimulus_id"],
        "detected": detected,
        "detection_window_center_s": event["detection_time"],
        "detection_window_start_s": event["detection_window_start"],
        "detection_window_stop_s": event["detection_window_stop"],
        "detection_latency_s": event["detection_latency"],
        "detected_before_stimulus": bool(event["detected_before_zero"]),
        "predicted_label_at_detection": event["predicted_label_at_detection"] if detected else np.nan,
        "predicted_stimulus_id_at_detection": event["predicted_class_at_detection"] if detected else np.nan,
        "correct_detected_stimulus": bool(event["is_correct_at_detection"]) if detected else False,
        "stimulus_score_at_detection": event["score_at_detection"],
        "score_threshold": event["score_threshold"],
        "threshold_quantile": event["threshold_quantile"],
        "threshold_window_start_s": event["threshold_window_start"],
        "threshold_window_stop_s": event["threshold_window_stop"],
        "detection_start_s": detection_start_s if detection_start_s is not None else np.nan,
        "n_scanned_windows": event["n_time_points"],
        "classifier": reference_row["classifier"],
        "components_pca": reference_row["components_pca"],
        "actual_components_pca": reference_row["actual_components_pca"],
        "frequency_low_hz": reference_row["frequency_low_hz"],
        "frequency_high_hz": reference_row["frequency_high_hz"],
    }


# jscpd:ignore-end
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _evaluate_window(
    train_data,
    validation_data,
    labels_train,
    labels_validation,
    participant,
    window_center,
    classifier_param,
    config,
    permutation_rng=None,
    include_predictions=False,
):
    train_window = _centered_window(window_center, config.window_size)
    null_window = _null_window(config)
    train_stimuli_features, train_null_features = extract_windows(train_data, train_window, null_window)
    validation_stimuli_features, _ = extract_windows(validation_data, train_window, (np.nan, np.nan))
    train_features = np.hstack(train_stimuli_features + train_null_features).T
    train_labels = labels_train
    if train_null_features:
        train_labels = np.concatenate((labels_train, np.zeros(len(train_null_features), dtype=int)))
    validation_features = np.hstack(validation_stimuli_features).T

    pca_components = _actual_pca_components(config.components_pca, train_features)
    explained_variance = np.nan
    if config.components_pca != float("inf"):
        train_features, coeff, train_features_mean, explained_variance = reduce_features_pca(train_features, int(config.components_pca))
        validation_features = (validation_features - train_features_mean) @ coeff[:, :pca_components]

    model = train_multiclass_classifier(
        train_features,
        train_labels,
        config.classifier,
        classifier_param,
        random_state=config.random_state,
    )
    predictions = model.predict(validation_features)
    accuracy = float(np.mean(predictions == labels_validation))
    permutation_accuracy = np.array([], dtype=float)
    permutation_p = np.nan
    if config.permutations > 0:
        permutation_accuracy = _permutation_accuracy_curve(
            train_features,
            validation_features,
            labels_validation,
            train_labels,
            config.classifier,
            classifier_param,
            config.random_state,
            config.permutations,
            permutation_rng,
        )
        permutation_p = float(np.mean(permutation_accuracy >= accuracy))
        if np.isfinite(permutation_p):
            permutation_p = (permutation_p * config.permutations + 1.0) / (config.permutations + 1.0)
    chance_accuracy = 1.0 / config.chance_classes
    variant = "without_null" if np.isnan(config.null_window_center) else "with_null"
    null_prediction_rate = float(np.mean(predictions == 0)) if variant == "with_null" else np.nan

    row = {
        "participant": participant,
        "variant": variant,
        "transfer_direction": config.transfer_direction,
        "window_center_s": window_center,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "above_chance": accuracy > chance_accuracy,
        "n_train_trials": len(labels_train),
        "n_validation_trials": len(labels_validation),
        "n_train_classes": len(np.unique(labels_train)),
        "n_validation_classes": len(np.unique(labels_validation)),
        "n_permutations": int(config.permutations),
        "permutation_seed": config.permutation_seed,
        "permutation_p_value": permutation_p,
        "permutation_accuracy_mean": (float(np.mean(permutation_accuracy)) if permutation_accuracy.size else np.nan),
        "permutation_accuracy_std": (float(np.std(permutation_accuracy, ddof=1)) if permutation_accuracy.size > 1 else np.nan),
        "null_window_center_s": config.null_window_center,
        "null_prediction_rate": null_prediction_rate,
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": pca_components,
        "pca_explained_variance_percent": explained_variance,
        "frequency_low_hz": config.frequency_range[0],
        "frequency_high_hz": config.frequency_range[1],
    }

    if include_predictions:
        return row, _stimulus_prediction_rows(
            participant,
            variant,
            window_center,
            train_window[0],
            train_window[1],
            labels_validation,
            predictions,
            config,
            pca_components,
        )
    return row


def _stimulus_prediction_rows(
    participant,
    variant,
    window_center,
    window_start,
    window_stop,
    labels_validation,
    predictions,
    config,
    actual_components_pca,
):
    rows = []
    for trial_idx, (true_label, predicted_label) in enumerate(zip(labels_validation, predictions)):
        true_stimulus = _display_stimulus_label(true_label, variant)
        predicted_stimulus = _display_stimulus_label(predicted_label, variant)
        rows.append(
            {
                "participant": participant,
                "variant": variant,
                "transfer_direction": config.transfer_direction,
                "window_center_s": window_center,
                "window_start_s": window_start,
                "window_stop_s": window_stop,
                "trial": trial_idx,
                "validation_trial_index": trial_idx,
                "validation_trial_number": trial_idx + 1,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "true_stimulus_id": true_stimulus,
                "predicted_stimulus_id": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "actual_components_pca": actual_components_pca,
            }
        )
    return rows


def _display_stimulus_label(label, variant):
    label = int(label)
    if variant == "without_null":
        return label + 1
    return label


def _centered_window(center, size):
    return center - size / 2, center + size / 2


def _null_window(config):
    if np.isnan(config.null_window_center):
        return np.nan, np.nan
    return _centered_window(config.null_window_center, config.window_size)


def _actual_pca_components(components_pca, features):
    if components_pca == float("inf"):
        return features.shape[1]
    return min(int(components_pca), features.shape[0], features.shape[1])


def _window_center_key(value):
    return float(np.round(float(value), 10))


def _window_center_set(values):
    return {_window_center_key(value) for value in values}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _permutation_accuracy_curve(
    train_features,
    validation_features,
    labels_validation,
    train_labels,
    classifier,
    classifier_param,
    random_state,
    n_permutations,
    permutation_rng,
):
    if permutation_rng is None:
        permutation_rng = np.random.default_rng()

    permuted_scores = []
    for _ in range(int(n_permutations)):
        permuted_train_labels = np.array(train_labels, copy=True)
        permutation_rng.shuffle(permuted_train_labels)
        model = train_multiclass_classifier(
            train_features,
            permuted_train_labels,
            classifier,
            classifier_param,
            random_state=random_state,
        )
        predictions = model.predict(validation_features)
        permuted_scores.append(float(np.mean(predictions == labels_validation)))
    return np.asarray(permuted_scores, dtype=float)


def _summary_stats(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), std, float(std / np.sqrt(values.size))


def _rows_frame(rows):
    return pd.DataFrame(list(rows))


def _legacy_std(summary_value, values):
    finite_values = np.asarray(list(values), dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 1:
        return 0.0
    return _to_float(summary_value)


def _legacy_sem(summary_value, values):
    finite_values = np.asarray(list(values), dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 1:
        return 0.0
    return _to_float(summary_value)


def _present_group_fields(rows, fields):
    return tuple(field for field in fields if any(field in row for row in rows))


def _plot_group_accuracy(summary_rows, output_path):
    figure, axes = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["variant"]].append(row)

    chance_percent = None
    for variant, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: _to_float(row["window_center_s"]))
        x = np.asarray([_to_float(row["window_center_s"]) for row in rows], dtype=float)
        y = np.asarray([_to_float(row["percent_mean"]) for row in rows], dtype=float)
        sem = np.asarray([_to_float(row["percent_sem"]) for row in rows], dtype=float)
        chance_percent = _to_float(rows[0]["chance_percent"])
        axes.plot(x, y, marker="o", label=variant.replace("_", " "))
        axes.fill_between(x, y - sem, y + sem, alpha=0.2)

    if chance_percent is not None:
        axes.axhline(chance_percent, color="black", linewidth=1, linestyle="--")
    axes.axvline(0, color="black", linewidth=1, linestyle=":")
    axes.set_xlabel("window center from stimulus (s)")
    axes.set_ylabel("stimulus decoding accuracy (%)")
    axes.grid(True, alpha=0.25)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
