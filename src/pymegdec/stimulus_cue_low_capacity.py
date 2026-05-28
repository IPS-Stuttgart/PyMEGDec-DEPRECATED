"""Low-capacity cue/localizer calibration for cross-subject stimulus decoding."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_int_or_inf
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
DEFAULT_EXPERT_TOP_K_GRID = (4, 8, 12)
DEFAULT_EXPERT_TEMPERATURE_GRID = (0.10, 0.25, 0.50)
DEFAULT_EXPERT_RELIABILITY = "none"
EXPERT_RELIABILITY_MODES = ("none", "source_oof_balanced")


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
    expert_reliability=DEFAULT_EXPERT_RELIABILITY,
    tune_expert_hyperparameters=False,
    expert_top_k_grid=DEFAULT_EXPERT_TOP_K_GRID,
    expert_temperature_grid=DEFAULT_EXPERT_TEMPERATURE_GRID,
    progress=None,
):
    """Train source-subject experts and weight them by cue similarity to target.

    ``source_oof_balanced`` reliability estimates each source expert's transfer
    quality on the remaining source subjects inside the outer fold. It never
    scores the held-out target subject while choosing expert weights.
    """

    decode_config = cross_subject._normalized_config(decode_config or CrossSubjectStimulusConfig())  # pylint: disable=protected-access
    cue_config = cross_subject._normalized_config(cue_config or replace(decode_config, alignment="none"))  # pylint: disable=protected-access
    expert_reliability = _normalize_expert_reliability(expert_reliability)
    top_k_grid = _normalize_expert_top_k_grid(expert_top_k_grid)
    temperature_grid = _normalize_expert_temperature_grid(expert_temperature_grid)
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    classifier_param = _resolved_classifier_param(decode_config)
    main_sets = {participant: cross_subject.load_participant_stimulus_features(data_folder, participant, config=decode_config) for participant in participants}
    cue_sets = {participant: load_participant_cue_calibration_features(data_folder, participant, config=cue_config) for participant in participants}

    requested_top_k = int(top_k)
    requested_temperature = float(temperature)
    outer_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        source_participants = tuple(participant for participant in participants if participant != test_participant)
        top_k = requested_top_k
        temperature = requested_temperature
        fitted_model_cache = {}
        transfer_matrix = None
        if tune_expert_hyperparameters or expert_reliability != "none":
            transfer_matrix, fitted_model_cache = _source_expert_transfer_matrix(
                main_sets,
                source_participants,
                decode_config,
                classifier_param,
            )
        tuned_metadata = _expert_tuning_fields(
            enabled=False,
            top_k=top_k,
            temperature=temperature,
            top_k_grid=top_k_grid,
            temperature_grid=temperature_grid,
        )
        if tune_expert_hyperparameters:
            tuned = _tune_expert_hyperparameters(
                main_sets,
                cue_sets,
                source_participants,
                decode_config,
                classifier_param,
                fitted_model_cache,
                transfer_matrix,
                top_k_grid=top_k_grid,
                temperature_grid=temperature_grid,
                expert_reliability=expert_reliability,
            )
            top_k = tuned["top_k"]
            temperature = tuned["temperature"]
            tuned_metadata = _expert_tuning_fields(
                enabled=True,
                top_k=top_k,
                temperature=temperature,
                top_k_grid=top_k_grid,
                temperature_grid=temperature_grid,
                inner_balanced_accuracy=tuned["inner_balanced_accuracy"],
                n_inner_folds=tuned["n_inner_folds"],
            )
        similarities = np.asarray([_cue_pattern_similarity(cue_sets[participant], cue_sets[test_participant]) for participant in source_participants], dtype=float)
        reliabilities = _source_expert_reliabilities(
            source_participants,
            transfer_matrix,
            reliability_participants=source_participants,
            mode=expert_reliability,
        )
        selection_scores = _expert_selection_scores(
            similarities,
            reliabilities,
            mode=expert_reliability,
            chance_accuracy=1.0 / decode_config.chance_classes,
        )
        selected_positions = _top_k_positions(selection_scores, min(int(top_k), len(source_participants)))
        selected_participants = tuple(source_participants[index] for index in selected_positions)
        weights = _expert_weights(
            similarities[selected_positions],
            reliabilities[selected_positions],
            temperature=float(temperature),
            reliability_mode=expert_reliability,
            chance_accuracy=1.0 / decode_config.chance_classes,
        )
        fitted_models = [
            fitted_model_cache.get(participant)
            or cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
                [main_sets[participant]], decode_config, classifier_param, fit_score_calibration=False
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
        extra = _expert_fields(
            test_participant,
            selected_participants,
            weights,
            similarities[selected_positions],
            reliabilities[selected_positions],
            selection_scores[selected_positions],
            top_k,
            temperature,
            expert_reliability,
            tuned_metadata,
        )
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
    expert_reliability=DEFAULT_EXPERT_RELIABILITY,
    tune_expert_hyperparameters=False,
    expert_top_k_grid=DEFAULT_EXPERT_TOP_K_GRID,
    expert_temperature_grid=DEFAULT_EXPERT_TEMPERATURE_GRID,
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
            expert_reliability=expert_reliability,
            tune_expert_hyperparameters=tune_expert_hyperparameters,
            expert_top_k_grid=expert_top_k_grid,
            expert_temperature_grid=expert_temperature_grid,
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


def _source_expert_transfer_matrix(main_sets, source_participants, decode_config, classifier_param):
    source_participants = tuple(int(participant) for participant in source_participants)
    transfer = np.full((len(source_participants), len(source_participants)), np.nan, dtype=float)
    fitted_models = {}
    for source_index, participant in enumerate(source_participants):
        fitted_model = cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
            [main_sets[participant]],
            decode_config,
            classifier_param,
            fit_score_calibration=False,
        )
        fitted_models[int(participant)] = fitted_model
        for validation_index, validation_participant in enumerate(source_participants):
            if int(validation_participant) == int(participant):
                continue
            validation_row, _predictions = cross_subject._score_outer_fold_model(  # pylint: disable=protected-access
                fitted_model,
                main_sets[validation_participant],
                decode_config,
                include_predictions=False,
            )
            transfer[source_index, validation_index] = float(validation_row["balanced_accuracy"])
    return transfer, fitted_models


def _source_expert_reliabilities(source_participants, transfer_matrix, *, reliability_participants, mode):
    mode = _normalize_expert_reliability(mode)
    reliabilities = np.full(len(source_participants), np.nan, dtype=float)
    if mode == "none" or transfer_matrix is None:
        return reliabilities
    all_participants = tuple(int(participant) for participant in reliability_participants)
    participant_to_index = {participant: index for index, participant in enumerate(all_participants)}
    transfer = np.asarray(transfer_matrix, dtype=float)
    for output_index, participant in enumerate(source_participants):
        source_index = participant_to_index[int(participant)]
        validation_indices = [
            participant_to_index[int(validation_participant)]
            for validation_participant in source_participants
            if int(validation_participant) != int(participant)
        ]
        scores = transfer[source_index, validation_indices]
        finite = scores[np.isfinite(scores)]
        if finite.size:
            reliabilities[output_index] = float(np.mean(finite))
    return reliabilities


def _tune_expert_hyperparameters(  # pylint: disable=too-many-arguments,too-many-locals
    main_sets,
    cue_sets,
    source_participants,
    decode_config,
    classifier_param,
    fitted_model_cache,
    transfer_matrix,
    *,
    top_k_grid,
    temperature_grid,
    expert_reliability,
):
    rows = []
    source_participants = tuple(int(participant) for participant in source_participants)
    for validation_participant in source_participants:
        candidate_participants = tuple(participant for participant in source_participants if int(participant) != int(validation_participant))
        if not candidate_participants:
            continue
        similarities = np.asarray([_cue_pattern_similarity(cue_sets[participant], cue_sets[validation_participant]) for participant in candidate_participants], dtype=float)
        reliabilities = _source_expert_reliabilities(
            candidate_participants,
            transfer_matrix,
            reliability_participants=source_participants,
            mode=expert_reliability,
        )
        selection_scores = _expert_selection_scores(
            similarities,
            reliabilities,
            mode=expert_reliability,
            chance_accuracy=1.0 / decode_config.chance_classes,
        )
        for candidate_top_k in top_k_grid:
            for candidate_temperature in temperature_grid:
                selected_positions = _top_k_positions(selection_scores, min(candidate_top_k, len(candidate_participants)))
                selected_participants = tuple(candidate_participants[index] for index in selected_positions)
                weights = _expert_weights(
                    similarities[selected_positions],
                    reliabilities[selected_positions],
                    temperature=float(candidate_temperature),
                    reliability_mode=expert_reliability,
                    chance_accuracy=1.0 / decode_config.chance_classes,
                )
                outer_row, _predictions = _score_source_expert_ensemble(
                    main_sets,
                    validation_participant,
                    selected_participants,
                    decode_config,
                    classifier_param,
                    fitted_model_cache,
                    weights,
                    similarities[selected_positions],
                    include_predictions=False,
                )
                rows.append(
                    {
                        "top_k": int(candidate_top_k),
                        "temperature": float(candidate_temperature),
                        "validation_participant": int(validation_participant),
                        "balanced_accuracy": float(outer_row["balanced_accuracy"]),
                    }
                )
    return _select_best_expert_hyperparameters(rows)


def _select_best_expert_hyperparameters(rows):
    if not rows:
        raise ValueError("Expert hyperparameter tuning requires at least one validation row.")
    summaries = []
    settings = sorted({(int(row["top_k"]), float(row["temperature"])) for row in rows})
    for top_k, temperature in settings:
        scores = np.asarray(
            [float(row["balanced_accuracy"]) for row in rows if int(row["top_k"]) == top_k and float(row["temperature"]) == temperature],
            dtype=float,
        )
        summaries.append(
            {
                "top_k": int(top_k),
                "temperature": float(temperature),
                "inner_balanced_accuracy": float(np.mean(scores)),
                "n_inner_folds": int(scores.size),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            float(row["inner_balanced_accuracy"]),
            -abs(int(row["top_k"]) - DEFAULT_EXPERT_TOP_K),
            -abs(float(row["temperature"]) - DEFAULT_EXPERT_TEMPERATURE),
            -int(row["top_k"]),
        ),
        reverse=True,
    )[0]


def _score_source_expert_ensemble(
    main_sets,
    test_participant,
    selected_participants,
    decode_config,
    classifier_param,
    fitted_model_cache,
    weights,
    similarities,
    *,
    include_predictions,
):
    fitted_models = [
        fitted_model_cache.get(participant)
        or cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
            [main_sets[participant]], decode_config, classifier_param, fit_score_calibration=False
        )
        for participant in selected_participants
    ]
    selected_rows = [
        {
            "selected_candidate_index": int(participant),
            "selected_inner_balanced_accuracy_mean": float(similarity),
            "selected_inner_balanced_accuracy_sem": 0.0,
        }
        for participant, similarity in zip(selected_participants, similarities, strict=True)
    ]
    return cross_subject._score_outer_fold_ensemble_models(  # pylint: disable=protected-access
        fitted_models,
        [main_sets[test_participant]] * len(fitted_models),
        [decode_config] * len(fitted_models),
        selected_rows,
        ensemble_weights=weights,
        ensemble_weighting="uniform",
        ensemble_temperature=0.0,
        ensemble_score_normalization="row_z_softmax",
        include_predictions=include_predictions,
    )


def _expert_selection_scores(similarities, reliabilities, *, mode, chance_accuracy):
    similarities = np.asarray(similarities, dtype=float)
    mode = _normalize_expert_reliability(mode)
    if mode == "none":
        return similarities
    reliability_bonus = _expert_reliability_multiplier(reliabilities, chance_accuracy=chance_accuracy) - float(chance_accuracy)
    return similarities + reliability_bonus


def _expert_weights(similarities, reliabilities, *, temperature, reliability_mode, chance_accuracy):
    weights = _softmax_weights(similarities, temperature)
    reliability_mode = _normalize_expert_reliability(reliability_mode)
    if reliability_mode == "none":
        return weights
    multipliers = _expert_reliability_multiplier(reliabilities, chance_accuracy=chance_accuracy)
    weighted = weights * multipliers
    weight_sum = float(np.sum(weighted))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return weights
    return weighted / weight_sum


def _expert_reliability_multiplier(reliabilities, *, chance_accuracy):
    reliabilities = np.asarray(reliabilities, dtype=float)
    chance_accuracy = max(float(chance_accuracy), 1e-12)
    sanitized = np.where(np.isfinite(reliabilities), reliabilities, chance_accuracy)
    return np.maximum(sanitized, chance_accuracy)


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


def _expert_fields(participant, selected_participants, weights, similarities, reliabilities, selection_scores, top_k, temperature, expert_reliability, tuning_metadata):
    fields = {
        "cue_low_capacity_mode": "expert_mixture",
        "calibration_data": "cue",
        "calibration_alignment": "source_expert_weighting",
        "target_calibration_participant": int(participant),
        "cue_expert_top_k": int(top_k),
        "cue_expert_temperature": float(temperature),
        "cue_expert_reliability": _normalize_expert_reliability(expert_reliability),
        "cue_expert_participants": ";".join(str(int(value)) for value in selected_participants),
        "cue_expert_weights": ";".join(f"{int(participant)}:{float(weight):.6g}" for participant, weight in zip(selected_participants, weights, strict=True)),
        "cue_expert_similarities": ";".join(f"{int(participant)}:{float(value):.6g}" for participant, value in zip(selected_participants, similarities, strict=True)),
        "cue_expert_reliabilities": ";".join(
            f"{int(participant)}:{float(value):.6g}" for participant, value in zip(selected_participants, reliabilities, strict=True)
        ),
        "cue_expert_selection_scores": ";".join(
            f"{int(participant)}:{float(value):.6g}" for participant, value in zip(selected_participants, selection_scores, strict=True)
        ),
    }
    fields.update(tuning_metadata)
    return fields


def _expert_tuning_fields(
    *,
    enabled,
    top_k,
    temperature,
    top_k_grid,
    temperature_grid,
    inner_balanced_accuracy="",
    n_inner_folds="",
):
    return {
        "cue_expert_tuned": bool(enabled),
        "cue_expert_tuned_top_k": int(top_k),
        "cue_expert_tuned_temperature": float(temperature),
        "cue_expert_top_k_grid": ",".join(str(int(value)) for value in top_k_grid),
        "cue_expert_temperature_grid": ",".join(f"{float(value):.6g}" for value in temperature_grid),
        "cue_expert_tuning_inner_balanced_accuracy": inner_balanced_accuracy,
        "cue_expert_tuning_n_inner_folds": n_inner_folds,
    }


def _assemble_artifacts(outer_rows, prediction_rows, decode_config, *, mode):
    group_summary_rows = cross_subject.summarize_cross_subject_stimulus_smoke(outer_rows, config=decode_config)
    for row in group_summary_rows:
        row["cue_low_capacity_mode"] = mode
        row["calibration_data"] = "cue"
        if mode == "expert_mixture":
            _add_expert_summary_fields(row, outer_rows)
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


def _add_expert_summary_fields(row, outer_rows):
    for column in (
        "cue_expert_reliability",
        "cue_expert_tuned",
        "cue_expert_tuned_top_k",
        "cue_expert_tuned_temperature",
        "cue_expert_top_k_grid",
        "cue_expert_temperature_grid",
    ):
        row[f"{column}_counts"] = _format_counter(Counter(str(outer_row.get(column, "")) for outer_row in outer_rows))


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


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


def _normalize_expert_reliability(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in EXPERT_RELIABILITY_MODES:
        raise ValueError(f"expert_reliability must be one of {EXPERT_RELIABILITY_MODES}.")
    return token


def _normalize_expert_top_k_grid(values):
    output = tuple(int(value) for value in values)
    if not output or any(value <= 0 for value in output):
        raise ValueError("expert_top_k_grid must contain positive integers.")
    return output


def _normalize_expert_temperature_grid(values):
    output = tuple(float(value) for value in values)
    if not output or any(value <= 0.0 or not np.isfinite(value) for value in output):
        raise ValueError("expert_temperature_grid must contain positive finite values.")
    return output


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


def _parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        return _normalize_expert_top_k_grid(token.strip() for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        return _normalize_expert_temperature_grid(token.strip() for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
    parser.add_argument("--expert-reliability", choices=EXPERT_RELIABILITY_MODES, default=DEFAULT_EXPERT_RELIABILITY)
    parser.add_argument("--tune-expert-hyperparameters", action="store_true")
    parser.add_argument("--expert-top-k-grid", type=_parse_int_grid, default=DEFAULT_EXPERT_TOP_K_GRID)
    parser.add_argument("--expert-temperature-grid", type=_parse_float_grid, default=DEFAULT_EXPERT_TEMPERATURE_GRID)
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
        expert_reliability=args.expert_reliability,
        tune_expert_hyperparameters=args.tune_expert_hyperparameters,
        expert_top_k_grid=args.expert_top_k_grid,
        expert_temperature_grid=args.expert_temperature_grid,
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
