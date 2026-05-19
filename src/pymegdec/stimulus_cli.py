"""Stimulus-analysis command handlers for the grouped PyMEGDec CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import (
    normalize_argv,
    parse_classifier_param,
    parse_float_list,
    parse_float_or_inf,
    parse_int_or_inf,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import (
    available_participants,
    parse_participant_spec,
)
from pymegdec.stimulus_cue_calibration import (
    CUE_CALIBRATION_DATASETS,
    CUE_CALIBRATION_TEMPLATE_POLICIES,
    DECODE_REFERENCE_TOKEN,
    CueCalibrationConfig,
    export_cross_subject_cue_calibrated_stimulus,
)
from pymegdec.stimulus_cross_subject import (
    ALIGNMENT_MODES,
    DEFAULT_CROSS_SUBJECT_ALIGNMENT,
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    DEFAULT_CROSS_SUBJECT_CLASSIFIER,
    DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
    DEFAULT_CROSS_SUBJECT_FEATURE_MODE,
    DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    DEFAULT_CROSS_SUBJECT_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    FEATURE_MODES,
    DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    DEFAULT_CROSS_SUBJECT_WINDOW_CENTER,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    ENSEMBLE_SCORE_NORMALIZATION_MODES,
    NORMALIZATION_MODES,
    SELECTION_ENSEMBLE_DIVERSITY_MODES,
    SELECTION_ENSEMBLE_WEIGHTING_MODES,
    AUTO_CLASSIFIER_PARAM_GRID_TOKEN,
    AUTO_COMPONENTS_PCA_GRID_TOKEN,
    CrossSubjectStimulusConfig,
    TRIAL_SELECTION_MODES,
    export_cross_subject_stimulus_smoke,
    export_nested_cross_subject_stimulus,
    make_cross_subject_candidate_configs,
)
from pymegdec.stimulus_decoding import (
    DEFAULT_ONSET_MIN_CONSECUTIVE,
    DEFAULT_ONSET_MIN_DURATION,
    DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,
    DEFAULT_ONSET_SCAN_STEP_S,
    DEFAULT_ONSET_SCAN_TIME_WINDOW,
    DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER,
    DEFAULT_ONSET_THRESHOLD_METHOD,
    DEFAULT_ONSET_THRESHOLD_QUANTILE,
    DEFAULT_ONSET_THRESHOLD_WINDOW,
    ONSET_THRESHOLD_METHODS,
    TRANSFER_DIRECTIONS,
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    export_stimulus_onset_scan,
    export_stimulus_temporal_generalization,
    summarize_stimulus_decoding,
    summarize_stimulus_prediction_diagnostics,
    window_centers_from_range,
)
from reptrace.decoding.robustness import (
    RobustnessCondition,
    run_participant_robustness_conditions,
)

DEFAULT_PREDICTION_WINDOW_CENTERS = (-0.175, 0.175)
DEFAULT_ROBUSTNESS_PARTICIPANTS = "1-4,6,8,9,10,13-27"
ROBUSTNESS_CONTROLS = (
    RobustnessCondition("default", "Main-to-cue SVM, PCA 100, broadband"),
    RobustnessCondition("reverse_transfer", "Cue-to-main SVM, PCA 100, broadband", {"transfer_direction": "cue-to-main"}),
    RobustnessCondition("weighted_svm", "Main-to-cue balanced SVM, PCA 100, broadband", {"classifier": "multiclass-svm-weighted"}),
    RobustnessCondition("pca_50", "Main-to-cue SVM, PCA 50, broadband", {"components_pca": 50}),
    RobustnessCondition("pca_200", "Main-to-cue SVM, PCA 200, broadband", {"components_pca": 200}),
    RobustnessCondition("low_frequency", "Main-to-cue SVM, PCA 100, 0-30 Hz", {"frequency_range": (0.0, 30.0)}),
)


def _transfer_participants(participant_spec: str | None, data_folder) -> list[int]:
    if participant_spec:
        return parse_participant_spec(participant_spec)
    main_participants = set(available_participants(data_folder, cue=False))
    cue_participants = set(available_participants(data_folder, cue=True))
    return sorted(main_participants & cue_participants)


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window start must be before stop.")
    return parts


def _normalization_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _feature_mode_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _alignment_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_feature_mode_list(value: str) -> tuple[str, ...]:
    return tuple(_feature_mode_token(token) for token in _parse_token_list(value))


def _parse_normalization_list(value: str) -> tuple[str, ...]:
    return tuple(_normalization_token(token) for token in _parse_token_list(value))


def _parse_alignment_list(value: str) -> tuple[str, ...]:
    return tuple(_alignment_token(token) for token in _parse_token_list(value))


def _parse_int_or_inf_list(value: str) -> tuple[int | float | str, ...]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower().replace("_", "-") == AUTO_COMPONENTS_PCA_GRID_TOKEN:
            values.append(AUTO_COMPONENTS_PCA_GRID_TOKEN)
        else:
            values.append(parse_int_or_inf(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return tuple(values)


def _parse_classifier_param_grid(value: str) -> tuple[object, ...]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"default", "defaults"}:
            values.append(float("nan"))
        elif token.lower().replace("_", "-") == AUTO_CLASSIFIER_PARAM_GRID_TOKEN:
            values.append(AUTO_CLASSIFIER_PARAM_GRID_TOKEN)
        else:
            values.append(parse_classifier_param(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one classifier parameter value is required.")
    return tuple(values)


def _add_model_args(parser: argparse.ArgumentParser, *, include_transfer_direction: bool = True) -> None:
    parser.add_argument("--window-size", type=float, default=0.1, help="Window size in seconds.")
    parser.add_argument("--null-window-center", type=parse_float_or_inf, default=float("nan"), help="Center of an optional pre-stimulus null window, or nan.")
    if include_transfer_direction:
        parser.add_argument("--transfer-direction", choices=TRANSFER_DIRECTIONS, default="main-to-cue", help="Train/validation dataset direction.")
    parser.add_argument("--new-framerate", type=parse_float_or_inf, default=float("inf"), help="Target frame rate, or inf.")
    parser.add_argument("--classifier", default="multiclass-svm", help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=100, help="Number of PCA components, or inf.")
    parser.add_argument("--frequency-range", type=parse_float_or_inf, nargs=2, metavar=("LOW", "HIGH"), default=(0.0, float("inf")), help="Frequency range in Hz.")
    parser.add_argument("--chance-classes", type=int, default=16, help="Number of stimulus classes used for chance level.")


def _base_config(args: argparse.Namespace, *, window_centers: tuple[float, ...], transfer_direction: str | None = None) -> StimulusDecodingConfig:
    return StimulusDecodingConfig(
        window_centers=window_centers,
        window_size=args.window_size,
        null_window_center=args.null_window_center,
        new_framerate=args.new_framerate,
        classifier=args.classifier,
        classifier_param=parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        frequency_range=tuple(args.frequency_range),
        chance_classes=args.chance_classes,
        permutations=0,
        transfer_direction=transfer_direction or args.transfer_direction,
    )


def _participants_or_error(parser: argparse.ArgumentParser, spec: str | None, data_folder) -> list[int]:
    participants = _transfer_participants(spec, data_folder)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data directory with matching main and cue MAT files.")
    return participants


def _build_predictions_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export trial-level stimulus predictions for selected windows.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all participants with main and cue files.")
    parser.add_argument("--window-centers", type=parse_float_list, default=DEFAULT_PREDICTION_WINDOW_CENTERS, help="Comma-separated window centers in seconds.")
    _add_model_args(parser, include_transfer_direction=True)
    parser.add_argument("--output", default="outputs/stimulus_predictions.csv", help="Output CSV with one row per validation trial and window.")
    parser.add_argument("--summary-output", default="outputs/stimulus_prediction_summary.csv", help="Optional participant/window accuracy summary CSV.")
    parser.add_argument("--accuracy-output", default=None, help="Optional participant/window accuracy CSV.")
    parser.add_argument("--confusion-output", default=None, help="Optional confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default=None, help="Optional per-stimulus recall CSV.")
    return parser


def stimulus_predictions(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_predictions_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _participants_or_error(parser, args.participants, data_folder)
    config = _base_config(args, window_centers=args.window_centers)

    accuracy_rows = []
    prediction_rows = []
    for participant in participants:
        print(f"START participant={participant}", flush=True)
        participant_accuracy, participant_predictions = evaluate_participant_stimulus_decoding_diagnostics(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=args.window_centers,
        )
        accuracy_rows.extend(participant_accuracy)
        prediction_rows.extend(participant_predictions)
        print(f"DONE participant={participant}", flush=True)

    write_alpha_metrics_csv(prediction_rows, args.output)
    print(f"Wrote {len(prediction_rows)} trial prediction rows to {args.output}")
    summary_rows = summarize_stimulus_decoding(accuracy_rows)
    if args.summary_output:
        write_alpha_metrics_csv(summary_rows, args.summary_output)
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    if args.accuracy_output:
        write_alpha_metrics_csv(accuracy_rows, args.accuracy_output)
        print(f"Wrote {len(accuracy_rows)} participant/window rows to {args.accuracy_output}")
    if args.confusion_output or args.per_stimulus_output:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)
        if args.confusion_output:
            write_alpha_metrics_csv(confusion_rows, args.confusion_output)
            print(f"Wrote {len(confusion_rows)} confusion rows to {args.confusion_output}")
        if args.per_stimulus_output:
            write_alpha_metrics_csv(per_stimulus_rows, args.per_stimulus_output)
            print(f"Wrote {len(per_stimulus_rows)} per-stimulus rows to {args.per_stimulus_output}")
    return 0


def _build_cross_subject_smoke_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run a fixed-pipeline leave-one-subject-out stimulus decoding smoke test using Part*Data.mat files only.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--window-center", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER, help="Stimulus decoding window center in seconds.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Stimulus decoding window size in seconds.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument(
        "--feature-mode", type=_feature_mode_token, default=DEFAULT_CROSS_SUBJECT_FEATURE_MODE, choices=FEATURE_MODES, help="Feature extraction mode."
    )
    parser.add_argument(
        "--normalization",
        type=_normalization_token,
        default=DEFAULT_CROSS_SUBJECT_NORMALIZATION,
        choices=NORMALIZATION_MODES,
        help="Subject-level normalization mode.",
    )
    parser.add_argument(
        "--alignment",
        type=_alignment_token,
        default=DEFAULT_CROSS_SUBJECT_ALIGNMENT,
        choices=ALIGNMENT_MODES,
        help="Cross-subject training alignment mode.",
    )
    parser.add_argument("--classifier", default=DEFAULT_CROSS_SUBJECT_CLASSIFIER, help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA, help="Number of PCA components, or inf.")
    parser.add_argument(
        "--max-trials-per-class-per-participant",
        type=int,
        default=None,
        help="Optional deterministic cap on trials per stimulus class and participant for quick screening.",
    )
    parser.add_argument(
        "--trial-selection",
        choices=TRIAL_SELECTION_MODES,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
        help="Trial subset policy used when --max-trials-per-class-per-participant is set. 'random' samples a seeded subset; 'first' keeps legacy file-order trials.",
    )
    parser.add_argument(
        "--trial-selection-seed",
        type=int,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
        help="Seed for random trial selection; ignored with --trial-selection first.",
    )
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to the classifier.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_outer.csv", help="Held-out participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_smoke(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_cross_subject_smoke_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    config = CrossSubjectStimulusConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        alignment=args.alignment,
        classifier=args.classifier,
        classifier_param=parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        trial_selection=args.trial_selection,
        trial_selection_seed=args.trial_selection_seed,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    artifacts = export_cross_subject_stimulus_smoke(
        data_folder,
        participants,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(artifacts['outer'])} held-out participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['confusion'])} confusion rows to {args.confusion_output}")
    print(f"Wrote {len(artifacts['per_stimulus'])} per-stimulus rows to {args.per_stimulus_output}")
    print(f"Wrote {len(artifacts['confusion_pairs'])} confusion-pair rows to {args.confusion_pairs_output}")
    return 0


def _build_cross_subject_cue_calibrated_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run leave-one-subject-out main-task stimulus decoding after cue/localizer Procrustes calibration.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument(
        "--outer-participants",
        default=None,
        help="Optional held-out participant ids to evaluate in this run. Defaults to all participants.",
    )
    parser.add_argument("--window-center", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER, help="Main-task decoding window center in seconds.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Main-task decoding window size in seconds.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Main-task baseline window as start,stop in seconds.")
    parser.add_argument(
        "--feature-mode", type=_feature_mode_token, default=DEFAULT_CROSS_SUBJECT_FEATURE_MODE, choices=FEATURE_MODES, help="Main-task feature extraction mode."
    )
    parser.add_argument(
        "--normalization",
        type=_normalization_token,
        default=DEFAULT_CROSS_SUBJECT_NORMALIZATION,
        choices=NORMALIZATION_MODES,
        help="Main-task subject-level normalization mode.",
    )
    parser.add_argument("--classifier", default=DEFAULT_CROSS_SUBJECT_CLASSIFIER, help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan.")
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA, help="Number of PCA components, or inf.")
    parser.add_argument(
        "--max-trials-per-class-per-participant",
        type=int,
        default=None,
        help="Optional deterministic cap on main-task trials per stimulus class and participant.",
    )
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to the classifier.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--calibration-data", default="cue", choices=CUE_CALIBRATION_DATASETS, help="Auxiliary dataset used to fit calibration transforms.")
    parser.add_argument(
        "--calibration-template-policy",
        default="source_only",
        choices=CUE_CALIBRATION_TEMPLATE_POLICIES,
        help="Template policy for cue calibration. source_only avoids putting the held-out participant into the group template.",
    )
    parser.add_argument("--calibration-window-center", type=float, default=None, help="Cue calibration window center in seconds. Defaults to --window-center.")
    parser.add_argument("--calibration-window-size", type=float, default=None, help="Cue calibration window size in seconds. Defaults to --window-size.")
    parser.add_argument(
        "--calibration-baseline-window",
        type=_parse_time_window,
        default=None,
        help="Cue calibration baseline window as start,stop in seconds. Defaults to --baseline-window.",
    )
    parser.add_argument(
        "--calibration-feature-mode",
        type=_feature_mode_token,
        default=DECODE_REFERENCE_TOKEN,
        choices=(DECODE_REFERENCE_TOKEN, *FEATURE_MODES),
        help="Cue calibration feature mode. decode reuses --feature-mode.",
    )
    parser.add_argument(
        "--calibration-normalization",
        type=_normalization_token,
        default=DECODE_REFERENCE_TOKEN,
        choices=(DECODE_REFERENCE_TOKEN, *NORMALIZATION_MODES),
        help="Cue calibration normalization. decode reuses --normalization.",
    )
    parser.add_argument(
        "--calibration-max-trials-per-class-per-participant",
        type=int,
        default=None,
        help="Optional deterministic cap on cue calibration trials per stimulus class and participant.",
    )
    parser.add_argument(
        "--label-shuffle-control",
        action="store_true",
        help="Shuffle main-task training labels within each source participant for a null-control benchmark.",
    )
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the main-task label-shuffle control.")
    parser.add_argument(
        "--target-calibration-label-shuffle-control",
        action="store_true",
        help="Shuffle held-out participant cue labels before fitting that participant's calibration transform.",
    )
    parser.add_argument("--target-calibration-label-shuffle-seed", type=int, default=0, help="Seed for the target cue-label shuffle control.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_cue_calibrated_outer.csv", help="Held-out participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_cue_calibrated_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_cue_calibrated_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_cue_calibrated_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_cue_calibrated_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_cue_calibrated_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_cue_calibrated(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_cross_subject_cue_calibrated_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    decode_config = CrossSubjectStimulusConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        alignment="none",
        classifier=args.classifier,
        classifier_param=parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    calibration_config = CueCalibrationConfig(
        calibration_data=args.calibration_data,
        template_policy=args.calibration_template_policy,
        window_center=args.calibration_window_center if args.calibration_window_center is not None else args.window_center,
        window_size=args.calibration_window_size if args.calibration_window_size is not None else args.window_size,
        baseline_window=args.calibration_baseline_window if args.calibration_baseline_window is not None else args.baseline_window,
        feature_mode=args.calibration_feature_mode,
        normalization=args.calibration_normalization,
        max_trials_per_class_per_participant=args.calibration_max_trials_per_class_per_participant,
    )
    artifacts = export_cross_subject_cue_calibrated_stimulus(
        data_folder,
        participants,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        decode_config=decode_config,
        calibration_config=calibration_config,
        outer_participants=outer_participants,
        progress=lambda message: print(message, flush=True),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
        target_calibration_label_shuffle_control=args.target_calibration_label_shuffle_control,
        target_calibration_label_shuffle_seed=args.target_calibration_label_shuffle_seed,
    )
    print(f"Wrote {len(artifacts['outer'])} cue-calibrated held-out participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['confusion'])} confusion rows to {args.confusion_output}")
    print(f"Wrote {len(artifacts['per_stimulus'])} per-stimulus rows to {args.per_stimulus_output}")
    print(f"Wrote {len(artifacts['confusion_pairs'])} confusion-pair rows to {args.confusion_pairs_output}")
    return 0


def _build_cross_subject_nested_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run nested leave-one-subject-out stimulus decoding with inner LOSO model selection.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument(
        "--outer-participants",
        default=None,
        help="Optional held-out participant ids to evaluate in this run. Defaults to all participants.",
    )
    parser.add_argument(
        "--window-centers",
        type=parse_float_list,
        default=DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
        help="Comma-separated candidate window centers in seconds.",
    )
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Candidate window size in seconds.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument(
        "--feature-modes",
        type=_parse_feature_mode_list,
        default=(DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
        help="Comma-separated feature modes, e.g. sensor_mean,sensor_flat.",
    )
    parser.add_argument(
        "--normalizations",
        type=_parse_normalization_list,
        default=(DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
        help="Comma-separated subject normalization modes.",
    )
    parser.add_argument(
        "--alignments",
        type=_parse_alignment_list,
        default=(DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
        help="Comma-separated cross-subject training alignment modes.",
    )
    parser.add_argument(
        "--classifiers",
        type=_parse_token_list,
        default=(DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
        help="Comma-separated classifier names.",
    )
    parser.add_argument(
        "--classifier-params",
        type=_parse_classifier_param_grid,
        default=(float("nan"),),
        help="Comma-separated classifier parameters. Use default to use each classifier default.",
    )
    parser.add_argument(
        "--components-pca-values",
        type=_parse_int_or_inf_list,
        default=(DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
        help="Comma-separated PCA component counts, or inf.",
    )
    parser.add_argument(
        "--max-trials-per-class-per-participant",
        type=int,
        default=None,
        help="Optional deterministic cap on trials per stimulus class and participant for quick nested screening.",
    )
    parser.add_argument(
        "--trial-selection",
        choices=TRIAL_SELECTION_MODES,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
        help="Trial subset policy used when --max-trials-per-class-per-participant is set.",
    )
    parser.add_argument(
        "--trial-selection-seed",
        type=int,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
        help="Seed for random trial selection; ignored with --trial-selection first.",
    )
    parser.add_argument(
        "--selection-ensemble-size",
        type=int,
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
        help="Evaluate a row-z-softmax score ensemble over the top K inner-LOSO candidates instead of only the single winner.",
    )
    parser.add_argument(
        "--selection-ensemble-diversity",
        choices=SELECTION_ENSEMBLE_DIVERSITY_MODES,
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
        help="Prefer diverse candidates before filling the nested top-K ensemble.",
    )
    parser.add_argument(
        "--selection-ensemble-score-normalization",
        choices=ENSEMBLE_SCORE_NORMALIZATION_MODES,
        default=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
        help="Transform per-class candidate scores before top-K ensemble averaging.",
    )
    parser.add_argument(
        "--selection-ensemble-weighting",
        choices=SELECTION_ENSEMBLE_WEIGHTING_MODES,
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
        help="Weighting scheme for --selection-ensemble-size values greater than one.",
    )
    parser.add_argument(
        "--selection-ensemble-temperature",
        type=float,
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
        help="Softmax temperature for --selection-ensemble-weighting inner_softmax.",
    )
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to classifiers.")
    parser.add_argument(
        "--label-shuffle-control",
        action="store_true",
        help="Shuffle training labels within each participant for a nested null-control benchmark.",
    )
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the nested label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_nested_outer.csv", help="Untouched outer participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_nested_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_nested_inner_validation.csv", help="Inner validation score CSV.")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_nested_selected.csv", help="Selected hyperparameter CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_nested_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_nested_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_nested_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_nested_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    parser.add_argument("--resume", action="store_true", help="Skip outer participants already present in the existing nested output CSVs.")
    parser.add_argument("--write-incremental", action="store_true", help="Rewrite nested output CSVs after each completed outer participant.")
    return parser


def stimulus_cross_subject_nested(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_cross_subject_nested_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    candidate_configs = make_cross_subject_candidate_configs(
        window_centers=args.window_centers,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_modes=args.feature_modes,
        normalizations=args.normalizations,
        alignments=args.alignments,
        classifiers=args.classifiers,
        classifier_params=args.classifier_params,
        components_pca_values=args.components_pca_values,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        trial_selection=args.trial_selection,
        trial_selection_seed=args.trial_selection_seed,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    artifacts = export_nested_cross_subject_stimulus(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        inner_validation_output_path=args.inner_validation_output,
        selected_output_path=args.selected_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        resume=args.resume,
        write_incremental=args.write_incremental,
        outer_participants=outer_participants,
        selection_ensemble_size=args.selection_ensemble_size,
        selection_ensemble_diversity=args.selection_ensemble_diversity,
        selection_ensemble_score_normalization=args.selection_ensemble_score_normalization,
        selection_ensemble_weighting=args.selection_ensemble_weighting,
        selection_ensemble_temperature=args.selection_ensemble_temperature,
        progress=lambda message: print(message, flush=True),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    print(f"Wrote {len(artifacts['outer'])} untouched outer participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['inner_validation'])} inner validation rows to {args.inner_validation_output}")
    print(f"Wrote {len(artifacts['selected'])} selected hyperparameter rows to {args.selected_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['confusion'])} confusion rows to {args.confusion_output}")
    print(f"Wrote {len(artifacts['per_stimulus'])} per-stimulus rows to {args.per_stimulus_output}")
    print(f"Wrote {len(artifacts['confusion_pairs'])} confusion-pair rows to {args.confusion_pairs_output}")
    return 0


def _build_robustness_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export two-window robustness controls for stimulus decoding.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=DEFAULT_ROBUSTNESS_PARTICIPANTS, help="Participant ids such as 1-4,6,8. Defaults to the full current analysis set.")
    parser.add_argument("--window-centers", type=parse_float_list, default=DEFAULT_PREDICTION_WINDOW_CENTERS, help="Comma-separated window centers in seconds.")
    _add_model_args(parser, include_transfer_direction=False)
    parser.add_argument("--predictions-output", default="outputs/stimulus_robustness_predictions.csv", help="Output CSV with one row per validation trial/window/control.")
    parser.add_argument("--accuracy-output", default="outputs/stimulus_robustness_accuracy.csv", help="Output CSV with one row per participant/window/control.")
    parser.add_argument("--summary-output", default="outputs/stimulus_robustness_summary.csv", help="Output CSV summarized across participants by window/control.")
    return parser


def stimulus_robustness(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_robustness_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _participants_or_error(parser, args.participants, data_folder)
    base_config = _base_config(args, window_centers=args.window_centers, transfer_direction="main-to-cue")

    def run_participant(control: RobustnessCondition, participant: int):
        config = replace(base_config, **dict(control.parameters))
        accuracy, predictions = evaluate_participant_stimulus_decoding_diagnostics(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=args.window_centers,
        )
        return {"accuracy": accuracy, "predictions": predictions}

    artifacts = run_participant_robustness_conditions(
        ROBUSTNESS_CONTROLS,
        participants,
        run_participant,
        progress=lambda message: print(message, flush=True),
    )
    accuracy_rows = artifacts.get("accuracy", [])
    prediction_rows = artifacts.get("predictions", [])
    write_alpha_metrics_csv(prediction_rows, args.predictions_output)
    write_alpha_metrics_csv(accuracy_rows, args.accuracy_output)
    summary_rows = summarize_stimulus_decoding(accuracy_rows)
    write_alpha_metrics_csv(summary_rows, args.summary_output)
    print(f"Wrote {len(prediction_rows)} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(accuracy_rows)} participant/window/control rows to {args.accuracy_output}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    return 0


def _build_temporal_generalization_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export stimulus temporal generalization across train/test windows.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all participants with main and cue files.")
    parser.add_argument("--time-window", type=_parse_time_window, default=(-0.4, 0.8), help="Window-center range as start,stop in seconds.")
    parser.add_argument("--window-step-s", type=float, default=0.025, help="Step between train/test window centers in seconds.")
    _add_model_args(parser, include_transfer_direction=True)
    parser.add_argument("--output", default="outputs/stimulus_temporal_generalization.csv", help="Output CSV with one row per participant/train-window/test-window.")
    parser.add_argument("--summary-output", default="outputs/stimulus_temporal_generalization_summary.csv", help="Output CSV summarized across participants by train/test window.")
    return parser


def stimulus_temporal_generalization(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_temporal_generalization_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _participants_or_error(parser, args.participants, data_folder)
    config = _base_config(args, window_centers=window_centers_from_range(args.time_window, args.window_step_s))
    rows, summary_rows = export_stimulus_temporal_generalization(
        data_folder,
        participants,
        args.output,
        summary_output_path=args.summary_output,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(rows)} participant/train/test rows to {args.output}")
    print(f"Wrote {len(summary_rows)} train/test summary rows to {args.summary_output}")
    return 0


def _build_onset_scan_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export onset-blind stimulus identity scans across validation windows.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all participants with main and cue files.")
    parser.add_argument("--train-window-center", type=float, default=DEFAULT_ONSET_SCAN_TRAIN_WINDOW_CENTER, help="Known-onset training window center in seconds.")
    parser.add_argument("--scan-time-window", type=_parse_time_window, default=DEFAULT_ONSET_SCAN_TIME_WINDOW, help="Validation scan center range as start,stop in seconds.")
    parser.add_argument("--window-step-s", type=float, default=DEFAULT_ONSET_SCAN_STEP_S, help="Step between scan window centers in seconds.")
    parser.add_argument(
        "--threshold-window", type=_parse_time_window, default=DEFAULT_ONSET_THRESHOLD_WINDOW, help="Window-center range used to estimate the confidence threshold."
    )
    parser.add_argument("--threshold-quantile", type=float, default=DEFAULT_ONSET_THRESHOLD_QUANTILE, help="Quantile of threshold-window scores used as detection threshold.")
    parser.add_argument(
        "--threshold-method",
        choices=ONSET_THRESHOLD_METHODS,
        default=DEFAULT_ONSET_THRESHOLD_METHOD,
        help="Threshold estimator: point uses pointwise baseline scores; max_run uses sequence-level baseline maxima under the run criteria.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=DEFAULT_ONSET_MIN_CONSECUTIVE,
        help="Minimum number of adjacent above-threshold scan windows required for a detection.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_ONSET_MIN_DURATION,
        help="Optional minimum above-threshold run duration in seconds.",
    )
    parser.add_argument(
        "--require-stable-prediction",
        action="store_true",
        default=DEFAULT_ONSET_REQUIRE_STABLE_PREDICTION,
        help="Break onset runs when the predicted stimulus changes across adjacent above-threshold windows.",
    )
    parser.add_argument("--detection-start-s", type=parse_float_or_inf, default=None, help="Optional earliest scan center considered for first detection.")
    _add_model_args(parser, include_transfer_direction=True)
    parser.add_argument("--output", default="outputs/stimulus_onset_scan.csv", help="Output CSV with one row per validation trial and scan window.")
    parser.add_argument("--events-output", default="outputs/stimulus_onset_events.csv", help="Output CSV with one first-detection row per validation trial.")
    parser.add_argument("--summary-output", default="outputs/stimulus_onset_scan_summary.csv", help="Optional participant/scan-window summary CSV.")
    parser.add_argument("--event-summary-output", default="outputs/stimulus_onset_event_summary.csv", help="Optional participant first-detection summary CSV.")
    return parser


def stimulus_onset_scan(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_onset_scan_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    if not 0.0 <= args.threshold_quantile <= 1.0:
        parser.error("--threshold-quantile must be between 0 and 1.")
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be at least 1.")
    if args.min_duration is not None and args.min_duration < 0:
        parser.error("--min-duration must be non-negative.")
    data_folder = resolve_data_folder(args.data_folder)
    participants = _participants_or_error(parser, args.participants, data_folder)
    config = _base_config(args, window_centers=window_centers_from_range(args.scan_time_window, args.window_step_s))
    scan_rows, event_rows, summary_rows, event_summary_rows = export_stimulus_onset_scan(
        data_folder,
        participants,
        args.output,
        args.events_output,
        summary_output_path=args.summary_output,
        event_summary_output_path=args.event_summary_output,
        config=config,
        train_window_center=args.train_window_center,
        threshold_window=args.threshold_window,
        threshold_quantile=args.threshold_quantile,
        threshold_method=args.threshold_method,
        min_consecutive=args.min_consecutive,
        min_duration=args.min_duration,
        require_stable_prediction=args.require_stable_prediction,
        detection_start_s=args.detection_start_s,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(scan_rows)} trial/window scan rows to {args.output}")
    print(f"Wrote {len(event_rows)} first-detection rows to {args.events_output}")
    print(f"Wrote {len(summary_rows)} scan summary rows to {args.summary_output}")
    print(f"Wrote {len(event_summary_rows)} event summary rows to {args.event_summary_output}")
    return 0
