"""NeuRepTrace-backed stimulus onset-scan compatibility helpers.

PyMEGDec no longer owns the reusable onset detector.  This module keeps the
historical PyMEGDec output slots while delegating thresholding and event
extraction to :mod:`neureptrace.onset_detection` on probability-observation CSVs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.onset_detection import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    detect_onsets_from_csvs,
)

DEFAULT_ONSET_SCORE_COLUMN = "confidence"


def _paths(paths: Sequence[str | Path]) -> list[Path]:
    if not paths:
        raise ValueError("At least one NeuRepTrace probability-observation CSV is required.")
    return [Path(path) for path in paths]


def _read_records(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _copy_column(frame: pd.DataFrame, target: str, source: str) -> None:
    if target not in frame.columns and source in frame.columns:
        frame[target] = frame[source]


def _compat_thresholded_observations(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    _copy_column(frame, "scan_window_center_s", "time")
    _copy_column(frame, "scan_window_start_s", "window_start")
    _copy_column(frame, "scan_window_stop_s", "window_stop")
    _copy_column(frame, "threshold_window_start_s", "threshold_window_start")
    _copy_column(frame, "threshold_window_stop_s", "threshold_window_stop")
    _copy_column(frame, "stimulus_score", "onset_score")
    _copy_column(frame, "stimulus_score", "confidence")
    _copy_column(frame, "correct", "is_correct")
    frame.to_csv(path, index=False)
    return frame.to_dict(orient="records")


def _compat_events(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    _copy_column(frame, "detection_window_center_s", "detection_time")
    _copy_column(frame, "detection_latency_s", "detection_latency")
    _copy_column(frame, "detected_before_stimulus", "detected_before_zero")
    _copy_column(frame, "stimulus_score_at_detection", "score_at_detection")
    _copy_column(frame, "predicted_stimulus_id_at_detection", "predicted_class_at_detection")
    _copy_column(frame, "correct_detected_stimulus", "is_correct_at_detection")
    _copy_column(frame, "threshold_window_start_s", "threshold_window_start")
    _copy_column(frame, "threshold_window_stop_s", "threshold_window_stop")
    _copy_column(frame, "min_duration_s", "min_duration")
    _copy_column(frame, "detection_run_duration_s", "detection_run_duration")
    _copy_column(frame, "detection_run_stop_time_s", "detection_run_stop_time")
    _copy_column(frame, "stimulus_score_peak_in_run", "score_peak_in_run")
    frame.to_csv(path, index=False)
    return frame.to_dict(orient="records")


def run_neureptrace_onset_scan(
    observation_csvs: Sequence[str | Path],
    *,
    output_path: str | Path,
    events_output_path: str | Path,
    summary_output_path: str | Path | None = None,
    event_summary_output_path: str | Path | None = None,
    threshold_window: tuple[float, float] = DEFAULT_THRESHOLD_WINDOW,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    score_column: str = DEFAULT_ONSET_SCORE_COLUMN,
    threshold_method: str = "point",
    detection_start_s: float | None = None,
    detection_window: tuple[float, float] = DEFAULT_DETECTION_WINDOW,
    min_consecutive: int = 1,
    min_duration: float | None = None,
    require_stable_prediction: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run NeuRepTrace onset detection and return PyMEGDec-style row lists."""

    events, event_summary = detect_onsets_from_csvs(
        _paths(observation_csvs),
        threshold_window=threshold_window,
        threshold_quantile=threshold_quantile,
        score_column=score_column,
        threshold_method=threshold_method,
        detection_start=detection_start_s,
        detection_window=detection_window,
        min_consecutive=min_consecutive,
        min_duration=min_duration,
        require_stable_prediction=require_stable_prediction,
        out_events=Path(events_output_path),
        out_summary=Path(event_summary_output_path) if event_summary_output_path else None,
        out_thresholded_observations=Path(output_path),
        out_threshold_summary=Path(summary_output_path) if summary_output_path else None,
    )
    scan_rows = _compat_thresholded_observations(output_path)
    event_rows = _compat_events(events_output_path)
    summary_rows = _read_records(summary_output_path)
    event_summary_rows = event_summary.to_dict(orient="records")
    if event_summary_output_path and not Path(event_summary_output_path).exists():
        Path(event_summary_output_path).parent.mkdir(parents=True, exist_ok=True)
        event_summary.to_csv(event_summary_output_path, index=False)
    if not event_rows:
        event_rows = events.to_dict(orient="records")
    return scan_rows, event_rows, summary_rows, event_summary_rows


__all__ = [
    "DEFAULT_ONSET_SCORE_COLUMN",
    "THRESHOLD_METHODS",
    "run_neureptrace_onset_scan",
]
