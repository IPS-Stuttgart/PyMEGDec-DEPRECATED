"""Command-line entry points and shared CLI helpers for PyMEGDec."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Sequence

import numpy as np

from .alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_TIME_WINDOW,
    AlphaMetricConfig,
)
from .alpha_movement_analysis import (
    DEFAULT_POST_WINDOW,
    DEFAULT_PRE_WINDOW,
    AlphaMovementAnalysisConfig,
    export_alpha_movement_analysis,
)
from .cross_validation import cross_validate_single_dataset
from .model_transfer import evaluate_model_transfer
from .reaction_time_analysis import available_participants, parse_participant_spec
from .stimulus_decoding import (
    DEFAULT_DECODING_STEP_S,
    DEFAULT_DECODING_TIME_WINDOW,
    DEFAULT_STIMULUS_WINDOW_SIZE,
    TRANSFER_DIRECTIONS,
    StimulusDecodingConfig,
    export_time_resolved_stimulus_decoding,
    window_centers_from_range,
)


def _float_or_inf(value: str) -> float:
    normalized = value.lower()
    if normalized in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if normalized in {"nan", "+nan", "-nan"}:
        return float("nan")
    return float(value)


def _int_or_inf(value: str) -> int | float:
    parsed = _float_or_inf(value)
    if np.isinf(parsed):
        return parsed
    return int(parsed)


def _parse_classifier_param(value: str | None):
    if value is None:
        return np.nan

    normalized = value.strip()
    if normalized.lower() == "nan":
        return np.nan

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(normalized)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            pass

    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("classifier parameters must be numeric, JSON, or a Python literal") from exc


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


_VALUE_OPTIONS_THAT_CAN_START_WITH_DASH = {
    "--window-centers",
    "--time-window",
    "--scan-time-window",
    "--threshold-window",
}


def normalize_argv(argv: Sequence[str] | None) -> list[str]:
    """Normalize selected option-value pairs for negative comma-separated ranges."""

    if argv is None:
        argv = sys.argv[1:]
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _VALUE_OPTIONS_THAT_CAN_START_WITH_DASH and index + 1 < len(argv):
            normalized.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def parse_float_or_inf(value: str) -> float:
    """Parse a float, ``nan``, or infinity token for CLI arguments."""

    return _float_or_inf(value)


def parse_int_or_inf(value: str) -> int | float:
    """Parse an integer or infinity token for CLI arguments."""

    return _int_or_inf(value)


def parse_classifier_param(value: str | None):
    """Parse a classifier parameter value from a CLI token."""

    return _parse_classifier_param(value)


parse_float_list = _parse_float_list


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        dest="data_folder",
        default=None,
        help="Directory containing Part*Data.mat files.",
    )
    parser.add_argument("--participant", type=int, default=2, help="Participant id to evaluate.")
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument(
        "--train-window-center",
        type=float,
        default=0.2,
        help="Center of the stimulus training window.",
    )
    parser.add_argument(
        "--null-window-center",
        type=_float_or_inf,
        default=-0.2,
        help="Center of the null window, or nan.",
    )
    parser.add_argument(
        "--new-framerate",
        type=_float_or_inf,
        default=float("inf"),
        help="Target frame rate, or inf.",
    )
    parser.add_argument("--classifier", default="multiclass-svm", help="Classifier name.")
    parser.add_argument(
        "--classifier-param",
        default=None,
        help="Classifier parameter value, JSON, or Python literal.",
    )
    parser.add_argument(
        "--components-pca",
        type=_int_or_inf,
        default=100,
        help="Number of PCA components, or inf.",
    )
    parser.add_argument(
        "--frequency-range",
        type=_float_or_inf,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0, float("inf")),
        help="Frequency range in Hz.",
    )


def _common_kwargs(args: argparse.Namespace) -> dict:
    return {
        "window_size": args.window_size,
        "train_window_center": args.train_window_center,
        "null_window_center": args.null_window_center,
        "new_framerate": args.new_framerate,
        "classifier": args.classifier,
        "classifier_param": _parse_classifier_param(args.classifier_param),
        "components_pca": args.components_pca,
        "frequency_range": tuple(args.frequency_range),
    }


def _build_cross_validate_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Cross-validate one participant dataset.")
    _add_common_args(parser)
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds.")
    return parser


def _build_transfer_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Evaluate model transfer for one participant.")
    _add_common_args(parser)
    return parser


def _build_stimulus_decoding_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run time-resolved train-main/validate-cue stimulus decoding.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_folder",
        default=None,
        help="Directory containing Part*Data.mat and Part*CueData.mat files.",
    )
    parser.add_argument(
        "--participants",
        default=None,
        help=("Participant ids such as 1-4,6,8. Defaults to all participants " "with main and cue files."),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV for participant/window decoding accuracies.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output CSV summarized across participants by time window.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional trial-level prediction CSV for selected diagnostic windows.",
    )
    parser.add_argument(
        "--confusion-output",
        default=None,
        help="Optional confusion-count CSV for selected diagnostic windows.",
    )
    parser.add_argument(
        "--per-stimulus-output",
        default=None,
        help="Optional per-stimulus accuracy CSV for selected diagnostic windows.",
    )
    parser.add_argument(
        "--participant-peaks-output",
        default=None,
        help="Optional CSV with each participant's peak decoding window.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional directory for group-level stimulus decoding plots.",
    )
    parser.add_argument(
        "--time-window",
        type=parse_range,
        default=DEFAULT_DECODING_TIME_WINDOW,
        help="Window-center range as start,stop in seconds.",
    )
    parser.add_argument(
        "--window-centers",
        type=_parse_float_list,
        default=None,
        help=("Explicit comma-separated window centers in seconds. Overrides " "--time-window."),
    )
    parser.add_argument(
        "--diagnostic-window-centers",
        type=_parse_float_list,
        default=None,
        help="Comma-separated window centers for trial prediction diagnostics.",
    )
    parser.add_argument(
        "--window-step-s",
        type=float,
        default=DEFAULT_DECODING_STEP_S,
        help="Step between time-window centers in seconds.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=DEFAULT_STIMULUS_WINDOW_SIZE,
        help="Window size in seconds.",
    )
    parser.add_argument(
        "--null-window-center",
        type=_float_or_inf,
        default=float("nan"),
        help="Center of an optional pre-stimulus null window, or nan.",
    )
    parser.add_argument(
        "--transfer-direction",
        choices=TRANSFER_DIRECTIONS,
        default="main-to-cue",
        help="Train/validation dataset direction for stimulus transfer.",
    )
    parser.add_argument(
        "--new-framerate",
        type=_float_or_inf,
        default=float("inf"),
        help="Target frame rate, or inf.",
    )
    parser.add_argument("--classifier", default="multiclass-svm", help="Classifier name.")
    parser.add_argument(
        "--classifier-param",
        default=None,
        help="Classifier parameter value, JSON, or Python literal.",
    )
    parser.add_argument(
        "--components-pca",
        type=_int_or_inf,
        default=100,
        help="Number of PCA components, or inf.",
    )
    parser.add_argument(
        "--frequency-range",
        type=_float_or_inf,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, float("inf")),
        help="Frequency range in Hz.",
    )
    parser.add_argument(
        "--chance-classes",
        type=int,
        default=16,
        help="Number of stimulus classes used for the chance line.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help=("Number of label shuffles per participant window for " "permutation p-values."),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Seed for permutation label shuffles. Keep fixed for reproducibility.",
    )
    return parser


def cross_validate(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_cross_validate_parser(prog=prog)
    args = parser.parse_args(argv)
    accuracy = cross_validate_single_dataset(
        args.data_folder,
        args.participant,
        n_folds=args.folds,
        **_common_kwargs(args),
    )
    print(accuracy)
    return 0


def transfer(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_transfer_parser(prog=prog)
    args = parser.parse_args(argv)
    accuracy = evaluate_model_transfer(
        args.data_folder,
        args.participant,
        **_common_kwargs(args),
    )
    print(accuracy)
    return 0


def _transfer_participants(participant_spec, data_folder):
    if participant_spec:
        return parse_participant_spec(participant_spec)
    main_participants = set(available_participants(data_folder, cue=False))
    cue_participants = set(available_participants(data_folder, cue=True))
    return sorted(main_participants & cue_participants)


def stimulus_decoding(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_stimulus_decoding_parser(prog=prog)
    args = parser.parse_args(argv)
    participants = _transfer_participants(args.participants, args.data_folder)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data " "directory with matching main and cue MAT files.")
    window_centers = args.window_centers
    if window_centers is None:
        window_centers = window_centers_from_range(args.time_window, args.window_step_s)
    diagnostic_requested = any(
        (
            args.predictions_output,
            args.confusion_output,
            args.per_stimulus_output,
        )
    )
    if diagnostic_requested and not args.diagnostic_window_centers:
        parser.error("--diagnostic-window-centers is required for prediction, confusion, or per-stimulus outputs.")

    config = StimulusDecodingConfig(
        window_centers=window_centers,
        window_size=args.window_size,
        null_window_center=args.null_window_center,
        new_framerate=args.new_framerate,
        classifier=args.classifier,
        classifier_param=_parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        frequency_range=tuple(args.frequency_range),
        chance_classes=args.chance_classes,
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
        transfer_direction=args.transfer_direction,
    )

    rows, summary_rows = export_time_resolved_stimulus_decoding(
        args.data_folder,
        participants,
        args.output,
        summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        participant_peaks_output_path=args.participant_peaks_output,
        diagnostic_window_centers=args.diagnostic_window_centers,
        plots_dir=args.plots_dir,
        config=config,
        progress=print,
    )
    print(f"Wrote {len(rows)} participant/window rows to {args.output}")
    if args.summary_output:
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    if args.participant_peaks_output:
        print(f"Wrote participant peaks to {args.participant_peaks_output}")
    if args.diagnostic_window_centers:
        print(f"Wrote diagnostics for windows {args.diagnostic_window_centers}")
    if args.plots_dir:
        print(f"Wrote plots to {args.plots_dir}")
    return 0


def _build_alpha_movement_results_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Analyze exported alpha movement summaries.")
    parser.add_argument(
        "--movement-summary",
        required=True,
        help="Input CSV from analyze_alpha_movement.py --summary-output.",
    )
    parser.add_argument(
        "--effect-output",
        required=True,
        help="Output CSV for participant/condition pre-post effects.",
    )
    parser.add_argument(
        "--condition-summary-output",
        required=True,
        help="Output CSV for condition-level effect summaries.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional output directory for condition-level PNG plots.",
    )
    parser.add_argument(
        "--pre-window",
        type=parse_range,
        default=DEFAULT_PRE_WINDOW,
        help="Pre-stimulus window as start,stop in seconds.",
    )
    parser.add_argument(
        "--post-window",
        type=parse_range,
        default=DEFAULT_POST_WINDOW,
        help="Post-stimulus window as start,stop in seconds.",
    )
    parser.add_argument(
        "--plot-labels",
        nargs="*",
        default=None,
        help="Optional condition labels to include in plots.",
    )
    return parser


def alpha_movement_results(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_movement_results_parser(prog=prog)
    args = parser.parse_args(argv)
    config = AlphaMovementAnalysisConfig(
        pre_window=args.pre_window,
        post_window=args.post_window,
        plot_labels=(None if args.plot_labels is None else tuple(str(label) for label in args.plot_labels)),
    )
    effect_rows, summary_rows = export_alpha_movement_analysis(
        args.movement_summary,
        args.effect_output,
        args.condition_summary_output,
        plots_dir=args.plots_dir,
        config=config,
    )
    print(f"Wrote {len(effect_rows)} participant-condition rows to {args.effect_output}")
    print(f"Wrote {len(summary_rows)} condition summary rows to " f"{args.condition_summary_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument(
        "command",
        choices=[
            "cross-validate",
            "transfer",
            "stimulus-decoding",
            "alpha-movement-results",
        ],
        help="Workflow to run.",
    )

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command, *remaining = argv
    if command == "cross-validate":
        return cross_validate(remaining, prog="pymegdec cross-validate")
    if command == "transfer":
        return transfer(remaining, prog="pymegdec transfer")
    if command == "stimulus-decoding":
        return stimulus_decoding(remaining, prog="pymegdec stimulus-decoding")
    if command == "alpha-movement-results":
        return alpha_movement_results(remaining, prog="pymegdec alpha-movement-results")
    parser.error(f"Unsupported command: {command}")
    return 2


def parse_range(value: str) -> tuple[float, float]:
    """Parse a comma-separated numeric range."""

    lower, upper = value.split(",", maxsplit=1)
    return float(lower), float(upper)


def add_alpha_metric_arguments(parser: argparse.ArgumentParser) -> None:
    """Add alpha metric extraction options to an argument parser."""

    parser.add_argument(
        "--location-pattern",
        default=DEFAULT_OCCIPITAL_PATTERN,
        help="Regex for selecting channels by label.",
    )
    parser.add_argument(
        "--time-window",
        type=parse_range,
        default=DEFAULT_TIME_WINDOW,
        help="Time window as start,stop in seconds.",
    )
    parser.add_argument(
        "--frequency-range",
        type=parse_range,
        default=DEFAULT_FREQUENCY_RANGE,
        help="Frequency range as low,high in Hz.",
    )


def alpha_metric_config_from_args(args: argparse.Namespace) -> AlphaMetricConfig:
    """Build alpha metric config from parsed command-line arguments."""

    return AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
    )


if __name__ == "__main__":
    raise SystemExit(main())
