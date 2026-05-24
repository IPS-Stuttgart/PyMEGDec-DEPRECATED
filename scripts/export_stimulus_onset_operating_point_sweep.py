"""Sweep NeuRepTrace onset detector settings from probability observations."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from neureptrace.onset_detection import detect_onsets, summarize_onset_events  # noqa: E402
from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402


def _float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _window(text: str) -> tuple[float, float]:
    start, stop = _float_list(text)
    if start > stop:
        raise argparse.ArgumentTypeError("window start must be before stop")
    return start, stop


def _optional_float(text: str | None) -> float | None:
    if text is None or text.strip() == "" or text.lower() in {"none", "nan", "null"}:
        return None
    return float(text)


def _bool_list(text: str) -> tuple[bool, ...]:
    values = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"true", "1", "yes", "y"}:
            values.append(True)
        elif token in {"false", "0", "no", "n"}:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid boolean value: {item!r}")
    return tuple(values)


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean()) if column in frame and not frame.empty else np.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").median()) if column in frame and not frame.empty else np.nan


def _aggregate(summary: pd.DataFrame, *, method: str, quantile: float, min_consecutive: int, min_duration: float, stable: bool, max_false_alarm_rate: float) -> dict:
    false_alarm = _mean(summary, "false_alarm_rate")
    post_rate = _mean(summary, "post_zero_detected_rate")
    correct_rate = _mean(summary, "correct_detection_rate")
    return {
        "setting": f"{method}_sweep",
        "threshold_method": method,
        "threshold_quantile": quantile,
        "min_consecutive": min_consecutive,
        "min_duration_s": min_duration,
        "require_stable_prediction": bool(stable),
        "summary_rows": len(summary),
        "participants": summary["subject"].nunique() if "subject" in summary else len(summary),
        "false_alarm_rate_mean": false_alarm,
        "post_stimulus_detected_rate_mean": post_rate,
        "correct_detection_rate_mean": correct_rate,
        "conditional_correct_detection_rate": correct_rate / post_rate if np.isfinite(correct_rate) and np.isfinite(post_rate) and post_rate else np.nan,
        "median_latency_s": _median(summary, "post_detection_latency_median"),
        "mean_latency_s": _mean(summary, "post_detection_latency_mean"),
        "meets_false_alarm_constraint": bool(np.isfinite(false_alarm) and false_alarm <= max_false_alarm_rate),
        "selection_max_false_alarm_rate": max_false_alarm_rate,
    }


def _ranking(row: dict, feasible: bool) -> tuple:
    false_alarm = row.get("false_alarm_rate_mean", np.inf)
    post = row.get("post_stimulus_detected_rate_mean", -np.inf)
    correct = row.get("correct_detection_rate_mean", -np.inf)
    latency = row.get("median_latency_s", np.inf)
    false_alarm = false_alarm if np.isfinite(false_alarm) else np.inf
    post = post if np.isfinite(post) else -np.inf
    correct = correct if np.isfinite(correct) else -np.inf
    latency = latency if np.isfinite(latency) else np.inf
    tail = (row["threshold_quantile"], row["min_consecutive"], row["min_duration_s"], int(bool(row.get("require_stable_prediction", False))))
    return (-post, -correct, latency, *tail) if feasible else (false_alarm, -post, -correct, latency, *tail)


def _select(rows: list[dict], *, method: str) -> dict:
    feasible = [row for row in rows if row.get("meets_false_alarm_constraint")]
    selected = min(feasible, key=lambda row: _ranking(row, True)) if feasible else min(rows, key=lambda row: _ranking(row, False))
    selected = dict(selected)
    selected["setting"] = f"{method}_sweep_selected"
    selected["selected_feasible"] = bool(feasible)
    selected["selection_rule"] = "constrain false alarms, maximize post-stimulus detections and correct detections, then minimize latency"
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-input", required=True, help="Thresholded or raw NeuRepTrace probability-observation CSV.")
    parser.add_argument("--threshold-method", choices=("point", "max_run"), default="point")
    parser.add_argument("--threshold-window", type=_window, default=(-0.35, -0.05))
    parser.add_argument("--threshold-quantiles", type=_float_list, default=(0.95, 0.975, 0.99))
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--min-consecutives", type=_int_list, default=(1, 2, 3))
    parser.add_argument("--min-durations", type=_float_list, default=(0.025, 0.05, 0.075))
    parser.add_argument("--require-stable-prediction-values", type=_bool_list, default=(False,))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    parser.add_argument("--detection-start-s", default=None)
    parser.add_argument("--events-output", default="outputs/stimulus_onset_events_point_sweep.csv")
    parser.add_argument("--event-summary-output", default="outputs/stimulus_onset_event_summary_point_sweep.csv")
    parser.add_argument("--summary-output", default="outputs/stimulus_onset_operating_point_sweep.csv")
    parser.add_argument("--selected-output", default="outputs/stimulus_onset_operating_point_selected.csv")
    parser.add_argument("--selected-events-output", default="outputs/stimulus_onset_events_point_selected.csv")
    parser.add_argument("--selected-event-summary-output", default="outputs/stimulus_onset_event_summary_point_selected.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frame = pd.read_csv(args.scan_input)
    detection_start = _optional_float(args.detection_start_s)
    all_events: list[dict] = []
    all_summaries: list[dict] = []
    aggregate_rows: list[dict] = []
    by_key = {}
    for quantile, min_consecutive, min_duration, stable in itertools.product(args.threshold_quantiles, args.min_consecutives, args.min_durations, args.require_stable_prediction_values):
        events = detect_onsets(
            frame,
            threshold_window=args.threshold_window,
            threshold_quantile=quantile,
            score_column=args.score_column,
            threshold_method=args.threshold_method,
            min_consecutive=min_consecutive,
            min_duration=min_duration,
            require_stable_prediction=stable,
            detection_start=detection_start,
        )
        summary = summarize_onset_events(events)
        aggregate_rows.append(_aggregate(summary, method=args.threshold_method, quantile=quantile, min_consecutive=min_consecutive, min_duration=min_duration, stable=stable, max_false_alarm_rate=args.max_false_alarm_rate))
        by_key[(quantile, min_consecutive, min_duration, bool(stable))] = (events, summary)
        all_events.extend(events.to_dict(orient="records"))
        all_summaries.extend(summary.to_dict(orient="records"))
    selected = _select(aggregate_rows, method=args.threshold_method)
    selected_key = (selected["threshold_quantile"], selected["min_consecutive"], selected["min_duration_s"], bool(selected.get("require_stable_prediction", False)))
    selected_events, selected_summary = by_key[selected_key]
    write_alpha_metrics_csv(all_events, args.events_output)
    write_alpha_metrics_csv(all_summaries, args.event_summary_output)
    write_alpha_metrics_csv(aggregate_rows, args.summary_output)
    write_alpha_metrics_csv([selected], args.selected_output)
    write_alpha_metrics_csv(selected_events.to_dict(orient="records"), args.selected_events_output)
    write_alpha_metrics_csv(selected_summary.to_dict(orient="records"), args.selected_event_summary_output)
    print(f"Wrote {len(aggregate_rows)} operating-point rows to {args.summary_output}")
    print(f"Wrote selected operating point to {args.selected_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
