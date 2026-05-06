"""Export two-window stimulus-decoding robustness controls."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Sequence

from export_stimulus_predictions import (
    _float_or_inf,
    _int_or_inf,
    _normalize_argv,
    _parse_classifier_param,
    _parse_float_list,
    _transfer_participants,
)
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.data_config import resolve_data_folder
from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    summarize_stimulus_decoding,
)
from reptrace.decoding.robustness import RobustnessCondition, run_participant_robustness_conditions

DEFAULT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_WINDOW_CENTERS = (-0.175, 0.175)


# jscpd:ignore-start
ROBUSTNESS_CONTROLS = (
    RobustnessCondition("default", "Main-to-cue SVM, PCA 100, broadband"),
    RobustnessCondition("reverse_transfer", "Cue-to-main SVM, PCA 100, broadband", {"transfer_direction": "cue-to-main"}),
    RobustnessCondition("weighted_svm", "Main-to-cue balanced SVM, PCA 100, broadband", {"classifier": "multiclass-svm-weighted"}),
    RobustnessCondition("pca_50", "Main-to-cue SVM, PCA 50, broadband", {"components_pca": 50}),
    RobustnessCondition("pca_200", "Main-to-cue SVM, PCA 200, broadband", {"components_pca": 200}),
    RobustnessCondition("low_frequency", "Main-to-cue SVM, PCA 100, 0-30 Hz", {"frequency_range": (0.0, 30.0)}),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export two-window robustness controls for stimulus decoding.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS, help="Participant ids such as 1-4,6,8. Defaults to the full current analysis set.")
    parser.add_argument("--window-centers", type=_parse_float_list, default=DEFAULT_WINDOW_CENTERS, help="Comma-separated window centers in seconds.")
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument("--null-window-center", type=_float_or_inf, default=float("nan"), help="Center of an optional pre-stimulus null window, or nan.")
    parser.add_argument("--new-framerate", type=_float_or_inf, default=float("inf"), help="Target frame rate, or inf.")
    parser.add_argument("--classifier", default="multiclass-svm", help="Base classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Base classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=_int_or_inf, default=100, help="Base number of PCA components, or inf.")
    parser.add_argument(
        "--frequency-range",
        type=_float_or_inf,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, float("inf")),
        help="Base frequency range in Hz.",
    )
    parser.add_argument("--chance-classes", type=int, default=16, help="Number of stimulus classes used for the chance line.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_robustness_predictions.csv", help="Output CSV with one row per validation trial/window/control.")
    parser.add_argument("--accuracy-output", default="outputs/stimulus_robustness_accuracy.csv", help="Output CSV with one row per participant/window/control.")
    parser.add_argument("--summary-output", default="outputs/stimulus_robustness_summary.csv", help="Output CSV summarized across participants by window/control.")
    return parser


def _control_config(base_config: StimulusDecodingConfig, control: RobustnessCondition) -> StimulusDecodingConfig:
    return replace(base_config, **dict(control.parameters))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _transfer_participants(args.participants, data_folder)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data directory with matching main and cue MAT files.")

    base_config = StimulusDecodingConfig(
        window_centers=args.window_centers,
        window_size=args.window_size,
        null_window_center=args.null_window_center,
        new_framerate=args.new_framerate,
        classifier=args.classifier,
        classifier_param=_parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        frequency_range=tuple(args.frequency_range),
        chance_classes=args.chance_classes,
        permutations=0,
    )

    def run_participant(control, participant):
        config = _control_config(base_config, control)
        participant_accuracy, participant_predictions = evaluate_participant_stimulus_decoding_diagnostics(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=args.window_centers,
        )
        return {
            "accuracy": participant_accuracy,
            "predictions": participant_predictions,
        }

    artifacts = run_participant_robustness_conditions(
        ROBUSTNESS_CONTROLS,
        participants,
        run_participant,
        progress=lambda message: print(message, flush=True),
    )
    accuracy_rows = artifacts.get("accuracy", [])
    prediction_rows = artifacts.get("predictions", [])

    write_alpha_metrics_csv(prediction_rows, args.predictions_output)
    print(f"Wrote {len(prediction_rows)} trial prediction rows to {args.predictions_output}")
    write_alpha_metrics_csv(accuracy_rows, args.accuracy_output)
    print(f"Wrote {len(accuracy_rows)} participant/window/control rows to {args.accuracy_output}")
    summary_rows = summarize_stimulus_decoding(accuracy_rows)
    write_alpha_metrics_csv(summary_rows, args.summary_output)
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
# jscpd:ignore-end
