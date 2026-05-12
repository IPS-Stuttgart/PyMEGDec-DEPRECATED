"""Sweep onset detector settings from an exported stimulus-onset scan."""

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

from pymegdec.alpha_metrics import write_alpha_metrics_csv  # noqa: E402
from pymegdec.stimulus_decoding import (  # noqa: E402
    _stimulus_onset_event_rows_from_reptrace,
    summarize_stimulus_onset_events,
)

GROUP_FIELDS = (
    "participant",
    "variant",
    "transfer_direction",
    "train_window_center_s",
    "threshold_method",
    "min_consecutive",
    "min_duration_s",
    "require_stable_prediction",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)

SCAN_GROUP_FIELDS = (
    "participant",
    "variant",
    "transfer_direction",
    "train_window_center_s",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)


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


def _stable(value):
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return value
    return value


def _stable_rows(rows: list[dict], fields: tuple[str, ...] = GROUP_FIELDS) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        for field in fields:
            item[field] = _stable(item.get(field, ""))
        out.append(item)
    return out


def _grouped_event_rows(
    base_frame: pd.DataFrame,
    *,
    threshold_window: tuple[float, float],
    threshold_quantile: float,
    threshold_method: str,
    min_consecutive: int,
    min_duration: float,
    require_stable_prediction: bool,
    detection_start: float | None,
) -> list[dict]:
    grouped_events: list[dict] = []
    for _, group in base_frame.groupby(list(SCAN_GROUP_FIELDS), sort=False, dropna=False):
        grouped_events.extend(
            _stimulus_onset_event_rows_from_reptrace(
                group.to_dict(orient="records"),
                threshold_window=threshold_window,
                threshold_quantile=threshold_quantile,
                threshold_method=threshold_method,
                min_consecutive=min_consecutive,
                min_duration=min_duration,
                require_stable_prediction=require_stable_prediction,
                detection_start_s=detection_start,
            )
        )
    return _stable_rows(grouped_events)


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean()) if column in frame else np.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").median()) if column in frame else np.nan


def _aggregate(
    rows: list[dict],
    *,
    threshold_method: str,
    threshold_quantile: float,
    min_consecutive: int,
    min_duration: float,
    require_stable_prediction: bool,
    max_false_alarm_rate: float,
) -> dict:
    frame = pd.DataFrame(rows)
    fa = _mean(frame, "false_alarm_rate")
    post = _mean(frame, "post_stimulus_detected_rate")
    correct = _mean(frame, "correct_detection_rate")
    return {
        "setting": f"{threshold_method}_sweep",
        "threshold_method": threshold_method,
        "threshold_quantile": threshold_quantile,
        "min_consecutive": min_consecutive,
        "min_duration_s": min_duration,
        "require_stable_prediction": bool(require_stable_prediction),
        "summary_rows": len(frame),
        "participants": frame["participant"].nunique() if "participant" in frame else len(frame),
        "false_alarm_rate_mean": fa,
        "post_stimulus_detected_rate_mean": post,
        "correct_detection_rate_mean": correct,
        "conditional_correct_detection_rate": correct / post if np.isfinite(correct) and np.isfinite(post) and post else np.nan,
        "median_latency_s": _median(frame, "post_detection_latency_median_s"),
        "mean_latency_s": _mean(frame, "post_detection_latency_mean_s"),
        "meets_false_alarm_constraint": bool(np.isfinite(fa) and fa <= max_false_alarm_rate),
        "selection_max_false_alarm_rate": max_false_alarm_rate,
    }


def _ranking(row: dict, feasible: bool) -> tuple:
    fa = row.get("false_alarm_rate_mean", np.inf)
    post = row.get("post_stimulus_detected_rate_mean", -np.inf)
    correct = row.get("correct_detection_rate_mean", -np.inf)
    latency = row.get("median_latency_s", np.inf)
    fa = fa if np.isfinite(fa) else np.inf
    post = post if np.isfinite(post) else -np.inf
    correct = correct if np.isfinite(correct) else -np.inf
    latency = latency if np.isfinite(latency) else np.inf
    tail = (
        row["threshold_quantile"],
        row["min_consecutive"],
        row["min_duration_s"],
        int(bool(row.get("require_stable_prediction", False))),
    )
    return (-post, -correct, latency, *tail) if feasible else (fa, -post, -correct, latency, *tail)


def _select(rows: list[dict], *, threshold_method: str) -> dict:
    feasible = [row for row in rows if row.get("meets_false_alarm_constraint")]
    if feasible:
        selected = min(feasible, key=lambda row: _ranking(row, True))
        selected_feasible = True
    else:
        selected = min(rows, key=lambda row: _ranking(row, False))
        selected_feasible = False
    selected = dict(selected)
    selected["setting"] = f"{threshold_method}_sweep_selected"
    selected["selected_feasible"] = selected_feasible
    selected["selection_rule"] = (
        "false_alarm_rate_mean <= max_false_alarm_rate; then maximize post_stimulus_detected_rate_mean; "
        "then maximize correct_detection_rate_mean; then minimize median_latency_s; then prefer lower settings"
    )
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-input", required=True)
    parser.add_argument("--threshold-method", choices=("point", "max_run"), default="point")
    parser.add_argument("--threshold-window", type=_window, default=(-0.35, -0.05))
    parser.add_argument("--threshold-quantiles", type=_float_list, default=(0.95, 0.975, 0.99))
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
    detection_start = _optional_float(args.detection_start_s)
    base_frame = pd.read_csv(args.scan_input)
    all_events, all_summaries, aggregate_rows = [], [], []
    by_key = {}
    for q, n, duration, require_stable_prediction in itertools.product(
        args.threshold_quantiles,
        args.min_consecutives,
        args.min_durations,
        args.require_stable_prediction_values,
    ):
        events = _grouped_event_rows(
            base_frame,
            threshold_window=args.threshold_window,
            threshold_quantile=q,
            threshold_method=args.threshold_method,
            min_consecutive=n,
            min_duration=duration,
            require_stable_prediction=require_stable_prediction,
            detection_start=detection_start,
        )
        summaries = _stable_rows(summarize_stimulus_onset_events(events))
        row = _aggregate(
            summaries,
            threshold_method=args.threshold_method,
            threshold_quantile=q,
            min_consecutive=n,
            min_duration=duration,
            require_stable_prediction=require_stable_prediction,
            max_false_alarm_rate=args.max_false_alarm_rate,
        )
        by_key[(q, n, duration, require_stable_prediction)] = (events, summaries)
        all_events.extend(events)
        all_summaries.extend(summaries)
        aggregate_rows.append(row)
    selected = _select(aggregate_rows, threshold_method=args.threshold_method)
    selected_key = (
        selected["threshold_quantile"],
        selected["min_consecutive"],
        selected["min_duration_s"],
        bool(selected.get("require_stable_prediction", False)),
    )
    selected_events, selected_summaries = by_key[selected_key]
    write_alpha_metrics_csv(all_events, args.events_output)
    write_alpha_metrics_csv(all_summaries, args.event_summary_output)
    write_alpha_metrics_csv(aggregate_rows, args.summary_output)
    write_alpha_metrics_csv([selected], args.selected_output)
    write_alpha_metrics_csv(selected_events, args.selected_events_output)
    write_alpha_metrics_csv(selected_summaries, args.selected_event_summary_output)
    print(f"Wrote {len(aggregate_rows)} operating-point rows to {args.summary_output}")
    print(f"Wrote selected operating point to {args.selected_output}")
    print(
        f"Selected {args.threshold_method} operating point: "
        f"threshold_quantile={selected['threshold_quantile']}, "
        f"min_consecutive={selected['min_consecutive']}, "
        f"min_duration_s={selected['min_duration_s']}, "
        f"require_stable_prediction={selected['require_stable_prediction']}, "
        f"false_alarm_rate_mean={selected['false_alarm_rate_mean']:.4f}, "
        f"post_stimulus_detected_rate_mean={selected['post_stimulus_detected_rate_mean']:.4f}, "
        f"correct_detection_rate_mean={selected['correct_detection_rate_mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
