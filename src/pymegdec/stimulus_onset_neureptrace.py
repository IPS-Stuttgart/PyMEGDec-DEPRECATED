"""NeuRepTrace-backed stimulus onset-scan compatibility helpers.

PyMEGDec no longer owns the reusable onset detector.  This module keeps the
historical PyMEGDec output slots while delegating thresholding and event
extraction to :mod:`neureptrace.onset_detection` on probability-observation CSVs.
"""

from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from neureptrace.onset_detection import (
    DEFAULT_DETECTION_WINDOW,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_THRESHOLD_WINDOW,
    THRESHOLD_METHODS,
    detect_onsets_from_csvs,
)

from pymegdec.cli import normalize_argv, parse_float_or_inf

DEFAULT_ONSET_SCORE_COLUMN = "confidence"
DEFAULT_ONSET_MIN_CONSECUTIVE = 1
DEFAULT_ONSET_MIN_DURATION = None
DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION = False


def _paths(paths: Sequence[str | Path]) -> list[Path]:
    if not paths:
        raise ValueError("At least one NeuRepTrace probability-observation CSV is required.")
    resolved: list[Path] = []
    for item in paths:
        text = str(item)
        matches = sorted(glob.glob(text))
        if matches:
            resolved.extend(Path(match) for match in matches)
        elif Path(text).exists():
            resolved.append(Path(text))
        else:
            raise FileNotFoundError(f"Observation CSV input does not exist or match any file: {text}")
    if not resolved:
        raise ValueError("No NeuRepTrace probability-observation CSVs were resolved.")
    return resolved


def _read_records(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    return cast(list[dict[str, Any]], pd.read_csv(path).to_dict(orient="records"))


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
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


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
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _time_window(value: str | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, str):
        normalized = value.replace(":", ",")
        parts = tuple(float(token.strip()) for token in normalized.split(",", maxsplit=1))
    else:
        parts = tuple(float(token) for token in value)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window must have the form start,stop or start:stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Window start must be before stop.")
    return parts


def _optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return None
    parsed = float(value)
    return None if pd.isna(parsed) else parsed


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
    min_consecutive: int = DEFAULT_ONSET_MIN_CONSECUTIVE,
    min_duration: float | None = DEFAULT_ONSET_MIN_DURATION,
    require_stable_prediction: bool = DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,
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


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Detect stimulus onsets from NeuRepTrace probability-observation CSVs.",
    )
    parser.add_argument("--observation-csv", "--observations", nargs="+", dest="observation_csvs", help="NeuRepTrace probability-observation CSVs or glob patterns.")
    parser.add_argument("--threshold-window", type=_time_window, default=DEFAULT_THRESHOLD_WINDOW, help="Baseline threshold window as start,stop or start:stop in seconds.")
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_THRESHOLD_QUANTILE, help="Detection threshold quantile.")
    parser.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default="point")
    parser.add_argument("--score-column", default=DEFAULT_ONSET_SCORE_COLUMN, help="Score column to threshold; confidence and probability_true_class can be inferred.")
    parser.add_argument("--detection-start-s", type=parse_float_or_inf, default=None, help="Optional earliest scan center considered for first detection.")
    parser.add_argument("--detection-window", type=_time_window, default=DEFAULT_DETECTION_WINDOW, help="Detection scan window as start,stop or start:stop.")
    parser.add_argument("--min-consecutive", type=int, default=DEFAULT_ONSET_MIN_CONSECUTIVE)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_ONSET_MIN_DURATION)
    parser.add_argument("--require-stable-prediction", action="store_true", default=DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION)
    parser.add_argument("--output", default="outputs/stimulus_onset_scan.csv", help="Thresholded observation output CSV.")
    parser.add_argument("--events-output", default="outputs/stimulus_onset_events.csv", help="Onset-event output CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_onset_scan_summary.csv", help="Threshold-crossing summary CSV.")
    parser.add_argument("--event-summary-output", default="outputs/stimulus_onset_event_summary.csv", help="Onset-event summary CSV.")

    # Legacy raw BUSH-MEG scan options are accepted only to produce one clear
    # migration error when old scripts are run unchanged.
    parser.add_argument("--data-dir", dest="data_folder", help=argparse.SUPPRESS)
    parser.add_argument("--participants", help=argparse.SUPPRESS)
    parser.add_argument("--train-window-center", help=argparse.SUPPRESS)
    parser.add_argument("--scan-time-window", help=argparse.SUPPRESS)
    parser.add_argument("--window-step-s", help=argparse.SUPPRESS)
    parser.add_argument("--window-size", help=argparse.SUPPRESS)
    parser.add_argument("--null-window-center", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-direction", help=argparse.SUPPRESS)
    parser.add_argument("--new-framerate", help=argparse.SUPPRESS)
    parser.add_argument("--classifier", help=argparse.SUPPRESS)
    parser.add_argument("--classifier-param", help=argparse.SUPPRESS)
    parser.add_argument("--components-pca", help=argparse.SUPPRESS)
    parser.add_argument("--frequency-range", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--chance-classes", help=argparse.SUPPRESS)
    return parser


def stimulus_onset_scan(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Compatibility CLI for NeuRepTrace onset detection."""

    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    if not args.observation_csvs:
        parser.error(
            "PyMEGDec no longer performs raw BUSH-MEG onset scans from Part*Data.mat files. "
            "Generate NeuRepTrace probability observations first, then pass them with --observation-csv. "
            "You can also call `neureptrace onset-detect` directly."
        )
    if not 0.0 <= args.threshold_quantile <= 1.0:
        parser.error("--threshold-quantile must be between 0 and 1.")
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be at least 1.")
    if args.min_duration is not None and args.min_duration < 0:
        parser.error("--min-duration must be non-negative.")

    scan_rows, event_rows, summary_rows, event_summary_rows = run_neureptrace_onset_scan(
        args.observation_csvs,
        output_path=args.output,
        events_output_path=args.events_output,
        summary_output_path=args.summary_output,
        event_summary_output_path=args.event_summary_output,
        threshold_window=args.threshold_window,
        threshold_quantile=args.threshold_quantile,
        score_column=args.score_column,
        threshold_method=args.threshold_method,
        detection_start_s=_optional_float(args.detection_start_s),
        detection_window=args.detection_window,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
    )
    print(f"Wrote {len(scan_rows)} thresholded observation rows to {args.output}")
    print(f"Wrote {len(event_rows)} onset event rows to {args.events_output}")
    print(f"Wrote {len(summary_rows)} threshold summary rows to {args.summary_output}")
    print(f"Wrote {len(event_summary_rows)} event summary rows to {args.event_summary_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return stimulus_onset_scan(argv)


__all__ = [
    "DEFAULT_ONSET_MIN_CONSECUTIVE",
    "DEFAULT_ONSET_MIN_DURATION",
    "DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION",
    "DEFAULT_ONSET_SCORE_COLUMN",
    "THRESHOLD_METHODS",
    "run_neureptrace_onset_scan",
    "stimulus_onset_scan",
]
