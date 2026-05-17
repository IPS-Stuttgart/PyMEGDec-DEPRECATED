"""Participant-aware stimulus decoding summaries."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

import pymegdec._stimulus_decoding_core as _core


# jscpd:ignore-start
# pylint: disable=protected-access,too-many-locals

def summarize_stimulus_decoding(rows):
    """Summarize decoding rows while counting unique participants."""

    if not rows:
        return []

    group_fields = _core._present_group_fields(rows, _core.SUMMARY_GROUP_FIELDS)
    frame = _core._rows_frame(rows)
    participant_column = _participant_summary_column(rows)
    metric_summary = _core.summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        participant_column=participant_column,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_core._to_float(row["accuracy"]) for row in group_rows]
        std = _core._legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _core._legacy_sem(base_summary["accuracy_sem"], accuracies)
        permutation_p = [_core._to_float(row.get("permutation_p_value")) for row in group_rows]
        n_with_permutation = sum(np.isfinite(permutation_p))
        significant_05 = sum(value < 0.05 for value in permutation_p if np.isfinite(value))
        significant_01 = sum(value < 0.01 for value in permutation_p if np.isfinite(value))
        chance_fields = _chance_summary_fields(group_rows)
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": int(base_summary.get("n_participants", len(group_rows))),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "above_chance_count": _above_chance_count(group_rows),
                "n_with_permutation": int(n_with_permutation),
                "n_significant_p_0.05": int(significant_05),
                "n_significant_p_0.01": int(significant_01),
            }
        )
        summary_row.update(chance_fields)
        summary_rows.append(summary_row)
    return summary_rows


def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows while counting unique participants."""

    if not rows:
        return []

    group_fields = _core._present_group_fields(rows, _core.TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    frame = _core._rows_frame(rows)
    participant_column = _participant_summary_column(rows)
    metric_summary = _core.summarize_metric_table(
        frame,
        "accuracy",
        group_fields,
        participant_column=participant_column,
        chance_column="chance_accuracy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for base_summary in metric_summary.to_dict("records"):
        key = tuple(base_summary.get(field, "") for field in group_fields)
        group_rows = grouped[key]
        accuracies = [_core._to_float(row["accuracy"]) for row in group_rows]
        std = _core._legacy_std(base_summary["accuracy_std"], accuracies)
        sem = _core._legacy_sem(base_summary["accuracy_sem"], accuracies)
        diagonal_values = {
            _core._window_center_key(row["train_window_center_s"])
            == _core._window_center_key(row["test_window_center_s"])
            for row in group_rows
        }
        chance_fields = _chance_summary_fields(group_rows)
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": int(base_summary.get("n_participants", len(group_rows))),
                "accuracy_mean": base_summary["accuracy_mean"],
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * base_summary["accuracy_mean"],
                "percent_median": 100.0 * base_summary["accuracy_median"],
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "above_chance_count": _above_chance_count(group_rows),
                "is_diagonal": bool(diagonal_values == {True}),
            }
        )
        summary_row.update(chance_fields)
        summary_rows.append(summary_row)
    return summary_rows


def _chance_summary_fields(rows):
    chance_values = np.asarray([_row_chance_accuracy(row) for row in rows], dtype=float)
    chance_classes = np.asarray([_row_chance_classes(row) for row in rows], dtype=float)
    chance_mean = _nanmean(chance_values)
    return {
        "chance_accuracy": chance_mean,
        "chance_percent": _percent(chance_mean),
        "chance_accuracy_min": _nanmin(chance_values),
        "chance_accuracy_max": _nanmax(chance_values),
        "chance_classes_mean": _nanmean(chance_classes),
        "chance_classes_counts": _chance_classes_counts(chance_classes),
    }


def _above_chance_count(rows):
    count = 0
    for row in rows:
        accuracy = _core._to_float(row.get("accuracy"))
        chance = _row_chance_accuracy(row)
        if np.isfinite(accuracy) and np.isfinite(chance) and accuracy > chance:
            count += 1
    return int(count)


def _row_chance_accuracy(row):
    chance = _positive_float(row.get("chance_accuracy"))
    if chance is not None:
        return chance
    class_count = _row_chance_classes(row)
    if class_count is None:
        return np.nan
    return 1.0 / class_count


def _row_chance_classes(row):
    for key in ("chance_classes", "n_chance_classes", "n_validation_classes"):
        class_count = _positive_int(row.get(key))
        if class_count is not None:
            return float(class_count)
    chance = _positive_float(row.get("chance_accuracy"))
    if chance is None:
        return np.nan
    return float(round(1.0 / chance))


def _chance_classes_counts(chance_classes):
    counter: Counter[int] = Counter()
    for class_count in np.asarray(chance_classes, dtype=float):
        if np.isfinite(class_count) and class_count > 0.0:
            counter[int(round(class_count))] += 1
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _positive_int(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value):
    parsed = _core._to_float(value)
    return float(parsed) if np.isfinite(parsed) and parsed > 0.0 else None


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _nanmin(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def _nanmax(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def _percent(value):
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _participant_summary_column(rows):
    if rows and all("participant" in row for row in rows):
        return "participant"
    return None


# jscpd:ignore-end
