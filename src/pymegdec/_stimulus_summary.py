"""Participant-aware stimulus decoding summaries."""

from __future__ import annotations

import pymegdec._stimulus_decoding_core as _core
from reptrace.results.tables import summarize_metric_table

_CHANCE_CLASS_COLUMNS = ("chance_classes", "n_chance_classes", "n_validation_classes")
_CHANCE_SUMMARY_COLUMNS = (
    "chance_accuracy",
    "chance_percent",
    "chance_accuracy_min",
    "chance_accuracy_max",
    "chance_classes_mean",
    "chance_classes_counts",
)


# pylint: disable=protected-access

def summarize_stimulus_decoding(rows):
    """Summarize decoding rows while counting unique participants."""

    if not rows:
        return []
    summary, group_fields = _stimulus_summary_frame(
        rows,
        _core.SUMMARY_GROUP_FIELDS,
        permutation_p_column="permutation_p_value",
    )
    return _summary_records(summary, _summary_columns(group_fields, include_permutation=True))


def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows while counting unique participants."""

    if not rows:
        return []
    summary, group_fields = _stimulus_summary_frame(rows, _core.TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    summary["is_diagonal"] = summary.apply(_is_diagonal_summary_row, axis=1)
    return _summary_records(summary, _summary_columns(group_fields, include_diagonal=True))


def _stimulus_summary_frame(rows, group_field_candidates, *, permutation_p_column=None):
    group_fields = _core._present_group_fields(rows, group_field_candidates)
    summary = summarize_metric_table(
        _core._rows_frame(rows),
        "accuracy",
        group_fields,
        participant_column=_participant_summary_column(rows),
        chance_column="chance_accuracy",
        percent_scale=100.0,
        chance_percent_column="chance_percent",
        chance_class_columns=_CHANCE_CLASS_COLUMNS,
        permutation_p_column=permutation_p_column,
        zero_singleton_dispersion=True,
    )
    if "n_participants" not in summary.columns:
        summary["n_participants"] = summary["n_rows"].astype(int)
    summary = summary.rename(
        columns={
            "accuracy_above_chance_count": "above_chance_count",
            "chance_accuracy_mean": "chance_accuracy",
        }
    )
    return summary, group_fields


def _summary_columns(group_fields, *, include_permutation=False, include_diagonal=False):
    columns = [
        *group_fields,
        "n_participants",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_sem",
        "percent_mean",
        "percent_median",
        "percent_std",
        "percent_sem",
        "above_chance_count",
    ]
    if include_permutation:
        columns.extend(("n_with_permutation", "n_significant_p_0.05", "n_significant_p_0.01"))
    if include_diagonal:
        columns.append("is_diagonal")
    columns.extend(_CHANCE_SUMMARY_COLUMNS)
    return columns


def _summary_records(summary, columns):
    return summary[[column for column in columns if column in summary.columns]].to_dict("records")


def _is_diagonal_summary_row(row):
    return bool(
        _core._window_center_key(row["train_window_center_s"])
        == _core._window_center_key(row["test_window_center_s"])
    )


def _participant_summary_column(rows):
    if rows and all("participant" in row for row in rows):
        return "participant"
    return None
