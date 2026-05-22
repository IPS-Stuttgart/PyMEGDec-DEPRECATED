"""Low-capacity cue/localizer calibration for cross-subject stimulus decoding."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_float_or_inf, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cue_calibration import load_participant_cue_calibration_features
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig
from pymegdec.classifiers import get_default_classifier_param, should_use_default_classifier_param

CUE_LOW_CAPACITY_MODES = ("latency_shift", "expert_mixture")
DEFAULT_CUE_LATENCY_PEAK_WINDOW = (-0.05, 0.35)
DEFAULT_MAX_LATENCY_SHIFT_S = 0.05
DEFAULT_EXPERT_TOP_K = 8
DEFAULT_EXPERT_TEMPERATURE = 0.25


def evaluate_cross_subject_cue_latency_stimulus(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    decode_config=None,
    outer_participants=None,
    cue_peak_window=DEFAULT_CUE_LATENCY_PEAK_WINDOW,
    max_latency_shift_s=DEFAULT_MAX_LATENCY_SHIFT_S,
    progress=None,
):
    """Run LOSO decoding after cue-derived window-center shifts only.

    This uses cue data only to estimate each subject's global cue response peak.
    It does not fit a spatial transform, and it never uses main-task target labels.
    """

    decode_config = cross_subject._normalized_config(decode_config or CrossSubjectStimulusConfig())  # pylint: disable=protected-access
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    classifier_param = _resolved_classifier_param(decode_config)
    cue_peaks = {participant: estimate_cue_peak_latency(data_folder, participant, peak_window=cue_peak_window) for participant in participants}

    outer_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        train_participants = tuple(participant for participant in participants if participant != test_participant)
        reference_peak = float(np.median([cue_peaks[participant] for participant in train_participants]))
        shifted_sets = {}
        for participant in participants:
            shift = float(np.clip(reference_peak - cue_peaks[participant], -max_latency_shift_s, max_latency_shift_s))
            participant_config = replace(decode_config, window_center=decode_config.window_center + shift)
            shifted_sets[participant] = cross_subject.load_participant_stimulus_features(data_folder, participant, config=participant_config)
        train_sets = [shifted_sets[participant] for participant in train_participants]
        test_set = shifted_sets[test_participant]
        outer_row, participant_predictions = cross_subject._evaluate_outer_fold(  # pylint: disable=protected-access
            train_sets,
            test_set,
            config=decode_config,
            classifier_param=classifier_param,
        )
        extra = _latency_fields(test_participant, cue_peaks[test_participant], reference_peak, reference_peak - cue_peaks[test_participant], max_latency_shift_s)
        outer_row.update(extra)
        for row in participant_predictions:
            row.update(extra)
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE cue_latency outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")
    return _assemble_artifacts(outer_rows, prediction_rows, decode_config, mode="latency_shift")


def evaluate_cross_subject_cue_expert_mixture_stimulus(  # pylint: disable=too-many-arguments,too-many-locals
    data_folder,
    participants,
    *,
    decode_config=None,
    cue_config=None,
    outer_participants=None,
    top_k=DEFAULT_EXPERT_TOP_K,
    temperature=DEFAULT_EXPERT_TEMPERATURE,
    progress=None,
):
    """Train source-subject experts and weight them by cue similarity to target."""

    decode_config = cross_subject._normalized_config(decode_config or CrossSubjectStimulusConfig())  # pylint: disable=protected-access
    cue_config = cross_subject._normalized_config(cue_config or replace(decode_config, alignment="none"))  # pylint: disable=protected-access
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    classifier_param = _resolved_classifier_param(decode_config)
    main_sets = {participant: cross_subject.load_participant_stimulus_features(data_folder, participant, config=decode_config) for participant in participants}
    cue_sets = {participant: load_participant_cue_calibration_features(data_folder, participant, config=cue_config) for participant in participants}

    outer_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        source_participants = tuple(participant for participant in participants if participant != test_participant)
        similarities = np.asarray([_cue_pattern_similarity(cue_sets[participant], cue_sets[test_participant]) for participant in source_participants], dtype=float)
        selected_positions = _top_k_positions(similarities, min(int(top_k), len(source_participants)))
        selected_participants = tuple(source_participants[index] for index in selected_positions)
        weights = _softmax_weights(similarities[selected_positions], temperature=float(temperature))
        fitted_models = [
            cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
                [main_sets[participant]],
                decode_config,
                classifier_param,
                fit_score_calibration=False,
            )
            for participant in selected_participants
        ]
        selected_rows = [
            {
                "selected_candidate_index": int(participant),
                "selected_inner_balanced_accuracy_mean": float(similarity),
                "selected_inner_balanced_accuracy_sem": 0.0,
            }
            for participant, similarity in zip(selected_participants, similarities[selected_positions], strict=True)
        ]
        outer_row, participant_predictions = cross_subject._score_outer_fold_ensemble_models(  # pylint: disable=protected-access
            fitted_models,
            [main_sets[test_participant]] * len(fitted_models),
            [decode_config] * len(fitted_models),
            selected_rows,
            ensemble_weights=weights,
            ensemble_weighting="uniform",
            ensemble_temperature=temperature,
            ensemble_score_normalization="row_z_softmax",
            include_predictions=True,
        )
        extra = _expert_fields(test_participant, selected_participants, weights, similarities[selected_positions], top_k, temperature)
        outer_row.update(extra)
        for row in participant_predictions:
            row.update(extra)
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE cue_expert_mixture outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")
    return _assemble_artifacts(outer_rows, prediction_rows, decode_config, mode="expert_mixture")


def export_cross_subject_cue_low_capacity_stimulus(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    mode,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    decode_config=None,
    cue_config=None,
    outer_participants=None,
    cue_peak_window=DEFAULT_CUE_LATENCY_PEAK_WINDOW,
    max_latency_shift_s=DEFAULT_MAX_LATENCY_SHIFT_S,
    expert_top_k=DEFAULT_EXPERT_TOP_K,
    expert_temperature=DEFAULT_EXPERT_TEMPERATURE,
    progress=None,
):
    mode = _normalize_mode(mode)
    if mode == "latency_shift":
        artifacts = evaluate_cross_subject_cue_latency_stimulus(
            data_folder,
            participants,
            decode_config=decode_config,
            outer_participants=outer_participants,
            cue_peak_window=cue_peak_window,
            max_latency_shift_s=max_latency_shift_s,
            progress=progress,
        )
    else:
        artifacts = evaluate_cross_subject_cue_expert_mixture_stimulus(
            data_folder,
            participants,
            decode_config=decode_config,
            cue_config=cue_config,
            outer_participants=outer_participants,
            top_k=expert_top_k,
            temperature=expert_temperature,
            progress=progress,
        )
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    if confusion_pairs_output_path and artifacts["confusion_pairs"]:
        write_alpha_metrics_csv(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


def estimate_cue_peak_latency(data_folder, participant, *, peak_window=DEFAULT_CUE_LATENCY_PEAK_WINDOW):
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}CueData.mat"
    data = sio.loadmat(data_path)["data"][0]
    time = cross_subject._time_vector(data, 0)  # pylint: disable=protected-access
    mask = cross_subject._time_mask(time, peak_window)  # pylint: disable=protected-access
    accumulator = np.zeros(int(np.sum(mask)), dtype=float)
    n_trials = cross_subject._count_trials(data)  # pylint: disable=protected-access
    for trial_idx in range(n_trials):
        signal = cross_subject._trial_signal(data, trial_idx)[:, mask]  # pylint: disable=protected-access
        accumulator += np.sqrt(np.mean(np.square(signal), axis=0))
    mean_rms = accumulator / max(n_trials, 1)
    return float(time[mask][int(np.argmax(mean_rms))])


def _cue_pattern_similarity(source_set, target_set):
    source_labels = np.asarray(source_set.labels, dtype=int)
    target_labels = np.asarray(target_set.labels, dtype=int)
    common = tuple(sorted(set(source_labels.tolist()) & set(target_labels.tolist())))
    if len(common) < 2:
        return 0.0
    source = np.concatenate([np.mean(source_set.features[source_labels == label], axis=0) for label in common])
    target = np.concatenate([np.mean(target_set.features[target_labels == label], axis=0) for label in common])
    source = source - np.mean(source)
    target = target - np.mean(target)
    denom = float(np.linalg.norm(source) * np.linalg.norm(target))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(source, target) / denom)


def _top_k_positions(values, k):
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    return order[: int(k)]


def _softmax_weights(values, temperature):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return np.full(values.size, 1.0 / max(values.size, 1), dtype=float)
    temperature = max(float(temperature), 1e-6)
    logits = (values - np.max(values)) / temperature
    weights = np.exp(np.clip(logits, -50.0, 50.0))
    return weights / np.sum(weights)


def _latency_fields(participant, peak, reference_peak, shift, max_shift):
    clipped_shift = float(np.clip(shift, -max_shift, max_shift))
    return {
        "cue_low_capacity_mode": "latency_shift",
        "calibration_data": "cue",
        "calibration_alignment": "latency_shift",
        "target_calibration_participant": int(participant),
        "cue_latency_peak_s": float(peak),
        "cue_latency_reference_peak_s": float(reference_peak),
        "cue_latency_shift_s": clipped_shift,
        "cue_latency_max_shift_s": float(max_shift),
    }


def _expert_fields(participant, selected_participants, weights, similarities, top_k, temperature):
    return {
        "cue_low_capacity_mode": "expert_mixture",
        "calibration_data": "cue",
        "calibration_alignment": "source_expert_weighting",
        "target_calibration_participant": int(participant),
        "cue_expert_top_k": int(top_k),
        "cue_expert_temperature": float(temperature),
        "cue_expert_participants": ";".join(str(int(value)) for value in selected_participants),
        "cue_expert_weights": ";".join(f"{int(participant)}:{float(weight):.6g}" for participant, weight in zip(selected_participants, weights, strict=True)),
        "cue_expert_similarities": ";".join(f"{int(participant)}:{float(value):.6g}" for participant, value in zip(selected_participants, similarities, strict=True)),
    }


def _assemble_artifacts(outer_rows, prediction_rows, decode_config, *, mode):
    group_summary_rows = cross_subject.summarize_cross_subject_stimulus_smoke(outer_rows, config=decode_config)
    for row in group_summary_rows:
        row["cue_low_capacity_mode"] = mode
        row["calibration_data"] = "cue"
    confusion_rows, per_stimulus_rows = cross_subject.summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = cross_subject.summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def _normalize_outer_participants(participants, outer_participants):
    if outer_participants is None:
        return tuple(participants)
    outer_participants = tuple(int(participant) for participant in outer_participants)
    unknown = sorted(set(outer_participants) - set(participants))
    if unknown:
        raise ValueError(f"Outer participants must be part of participants: {unknown}")
    return outer_participants


def _normalize_mode(mode):
    token = str(mode).strip().lower().replace("-", "_")
    if token not in CUE_LOW_CAPACITY_MODES:
        raise ValueError(f"mode must be one of {CUE_LOW_CAPACITY_MODES}.")
    return token


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        return get_default_classifier_param(config.classifier)
    return classifier_param


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2 or parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop with start <= stop.")
    return parts


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run low-capacity cue-calibrated LOSO stimulus decoding.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat and Part*CueData.mat files.")
    parser.add_argument("--participants", default=cross_subject.DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate in this run.")
    parser.add_argument("--mode", choices=CUE_LOW_CAPACITY_MODES, default="latency_shift")
    parser.add_argument("--window-center", type=float, default=cross_subject.DEFAULT_CROSS_SUBJECT_WINDOW_CENTER)
    parser.add_argument("--window-size", type=float, default=cross_subject.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=_parse_time_window, default=cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW)
    parser.add_argument("--feature-mode", default=cross_subject.DEFAULT_CROSS_SUBJECT_FEATURE_MODE, choices=cross_subject.FEATURE_MODES)
    parser.add_argument("--normalization", default=cross_subject.DEFAULT_CROSS_SUBJECT_NORMALIZATION, choices=cross_subject.NORMALIZATION_MODES)
    parser.add_argument("--classifier", default=cross_subject.DEFAULT_CROSS_SUBJECT_CLASSIFIER)
    parser.add_argument("--classifier-param", default=None)
    parser.add_argument("--components-pca", type=parse_int_or_inf, default=cross_subject.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA)
    parser.add_argument("--sample-weighting", default=cross_subject.DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING, choices=cross_subject.SAMPLE_WEIGHTING_MODES)
    parser.add_argument("--score-calibration", default=cross_subject.DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION, choices=cross_subject.SCORE_CALIBRATION_MODES)
    parser.add_argument("--alignment-alpha", type=float, default=cross_subject.DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA)
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None)
    parser.add_argument("--chance-classes", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--signflip-permutations", type=int, default=10000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    parser.add_argument("--cue-window-center", type=float, default=None)
    parser.add_argument("--cue-window-size", type=float, default=None)
    parser.add_argument("--cue-baseline-window", type=_parse_time_window, default=None)
    parser.add_argument("--cue-feature-mode", default=None, choices=cross_subject.FEATURE_MODES)
    parser.add_argument("--cue-normalization", default=None, choices=cross_subject.NORMALIZATION_MODES)
    parser.add_argument("--latency-peak-window", type=_parse_time_window, default=DEFAULT_CUE_LATENCY_PEAK_WINDOW)
    parser.add_argument("--max-latency-shift-s", type=float, default=DEFAULT_MAX_LATENCY_SHIFT_S)
    parser.add_argument("--expert-top-k", type=int, default=DEFAULT_EXPERT_TOP_K)
    parser.add_argument("--expert-temperature", type=float, default=DEFAULT_EXPERT_TEMPERATURE)
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_cue_low_capacity_outer.csv")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_cue_low_capacity_group_summary.csv")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_cue_low_capacity_predictions.csv")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_cue_low_capacity_confusion.csv")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_cue_low_capacity_per_stimulus.csv")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_cue_low_capacity_confusion_pairs.csv")
    return parser


def stimulus_cross_subject_cue_low_capacity(argv=None, prog=None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
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
        sample_weighting=args.sample_weighting,
        score_calibration=args.score_calibration,
        alignment_alpha=args.alignment_alpha,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    cue_config = replace(
        decode_config,
        window_center=args.cue_window_center if args.cue_window_center is not None else args.window_center,
        window_size=args.cue_window_size if args.cue_window_size is not None else args.window_size,
        baseline_window=args.cue_baseline_window if args.cue_baseline_window is not None else args.baseline_window,
        feature_mode=args.cue_feature_mode if args.cue_feature_mode is not None else args.feature_mode,
        normalization=args.cue_normalization if args.cue_normalization is not None else args.normalization,
        score_calibration="none",
    )
    participants = parse_participant_spec(args.participants)
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    artifacts = export_cross_subject_cue_low_capacity_stimulus(
        resolve_data_folder(args.data_folder),
        participants,
        mode=args.mode,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        decode_config=decode_config,
        cue_config=cue_config,
        outer_participants=outer_participants,
        cue_peak_window=args.latency_peak_window,
        max_latency_shift_s=args.max_latency_shift_s,
        expert_top_k=args.expert_top_k,
        expert_temperature=args.expert_temperature,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(artifacts['outer'])} cue-low-capacity held-out participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    return 0


def main(argv=None) -> int:
    return stimulus_cross_subject_cue_low_capacity(argv)


if __name__ == "__main__":
    raise SystemExit(main())
