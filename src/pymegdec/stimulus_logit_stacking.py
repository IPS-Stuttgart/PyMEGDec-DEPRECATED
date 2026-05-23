"""Source-only cross-subject logit stacking for BUSH-MEG stimulus decoding.

The stacker is fitted only from source-subject out-of-fold predictions inside
an outer LOSO split.  The held-out participant contributes no labels to model
selection, weight fitting, or score calibration.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_float_list, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec

LOGIT_STACK_CLASSIFIER = "source_oof_logit_stack"
DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION = "row_z"
LOGIT_STACK_SCORE_NORMALIZATION_MODES = ("row_z", "rank", "none")
DEFAULT_LOGIT_STACK_WEIGHTING = "greedy_balanced"
LOGIT_STACK_WEIGHTING_MODES = ("uniform", "inner_softmax", "greedy_balanced")
DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE = 0.02
DEFAULT_LOGIT_STACK_CLASS_BIAS = True
DEFAULT_LOGIT_STACK_CLASS_BIAS_L2 = 1e-3
DEFAULT_LOGIT_STACK_MAX_BASE_MODELS = 12


@dataclass(frozen=True)
class LogitStackingFit:
    """Fitted source-only stacking parameters for one outer fold."""

    weights: np.ndarray
    class_bias: np.ndarray
    class_order: np.ndarray
    candidate_indices: tuple[int, ...]
    inner_candidate_balanced: np.ndarray
    inner_stacked_balanced_accuracy: float
    inner_unbiased_balanced_accuracy: float
    score_normalization: str
    weighting: str
    weighting_temperature: float
    class_bias_enabled: bool
    class_bias_l2: float
    max_base_models: int | None


def evaluate_cross_subject_logit_stacking(  # pylint: disable=too-many-arguments,too-many-locals
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    score_normalization=DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION,
    weighting=DEFAULT_LOGIT_STACK_WEIGHTING,
    weighting_temperature=DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE,
    max_base_models=DEFAULT_LOGIT_STACK_MAX_BASE_MODELS,
    fit_class_bias=DEFAULT_LOGIT_STACK_CLASS_BIAS,
    class_bias_l2=DEFAULT_LOGIT_STACK_CLASS_BIAS_L2,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run LOSO source-OOF logit stacking on ``Part*Data.mat`` files only."""

    candidate_configs = tuple(cross_subject._normalized_config(config) for config in candidate_configs)  # pylint: disable=protected-access
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 4:
        raise ValueError("At least four participants are required: one outer test subject and at least three source subjects for OOF stacking.")
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    data_folder = resolve_data_folder(data_folder)
    score_normalization = _normalize_score_normalization(score_normalization)
    weighting = _normalize_weighting(weighting)
    weighting_temperature = _normalize_temperature(weighting_temperature)
    max_base_models = _normalize_max_base_models(max_base_models)
    class_bias_l2 = _normalize_nonnegative_float(class_bias_l2, "class_bias_l2")

    feature_cache = _load_feature_cache(data_folder, participants, candidate_configs, progress=progress)
    outer_rows = []
    inner_rows = []
    selected_rows = []
    prediction_rows = []

    for test_participant in outer_participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        outer_train_participants = tuple(participant for participant in participants if participant != test_participant)
        stack_fit, fold_inner_rows = _fit_source_oof_stacker(
            test_participant,
            outer_train_participants,
            candidate_configs,
            feature_cache,
            score_normalization=score_normalization,
            weighting=weighting,
            weighting_temperature=weighting_temperature,
            max_base_models=max_base_models,
            fit_class_bias=fit_class_bias,
            class_bias_l2=class_bias_l2,
            progress=progress,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        selected_row = _selected_row_for_stacker(test_participant, stack_fit, candidate_configs, fold_inner_rows)
        outer_row, fold_predictions = _score_outer_with_stacker(
            test_participant,
            outer_train_participants,
            candidate_configs,
            feature_cache,
            stack_fit,
            selected_row,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(test_participant),),
        )
        _add_selected_candidate_fields(outer_row, selected_row)
        for prediction_row in fold_predictions:
            _add_selected_candidate_fields(prediction_row, selected_row)
        inner_rows.extend(fold_inner_rows)
        outer_rows.append(outer_row)
        selected_rows.append(selected_row)
        prediction_rows.extend(fold_predictions)
        if progress is not None:
            progress(
                "DONE outer_test_participant="
                f"{test_participant} stacker_inner={stack_fit.inner_stacked_balanced_accuracy:.4f} "
                f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f} "
                f"n_base_models={len(stack_fit.candidate_indices)}"
            )

    group_summary_rows = cross_subject.summarize_nested_cross_subject_stimulus(
        outer_rows,
        signflip_permutations=candidate_configs[0].signflip_permutations,
        signflip_seed=candidate_configs[0].signflip_seed,
    )
    _add_logit_stack_group_summary_fields(group_summary_rows, outer_rows)
    confusion_rows, per_stimulus_rows = cross_subject.summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = cross_subject.summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "inner_validation": inner_rows,
        "selected": selected_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def export_cross_subject_logit_stacking(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_output_path,
    group_summary_output_path=None,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    outer_participants=None,
    score_normalization=DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION,
    weighting=DEFAULT_LOGIT_STACK_WEIGHTING,
    weighting_temperature=DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE,
    max_base_models=DEFAULT_LOGIT_STACK_MAX_BASE_MODELS,
    fit_class_bias=DEFAULT_LOGIT_STACK_CLASS_BIAS,
    class_bias_l2=DEFAULT_LOGIT_STACK_CLASS_BIAS_L2,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run source-OOF logit stacking and write compact CSV artifacts."""

    artifacts = evaluate_cross_subject_logit_stacking(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_participants=outer_participants,
        score_normalization=score_normalization,
        weighting=weighting,
        weighting_temperature=weighting_temperature,
        max_base_models=max_base_models,
        fit_class_bias=fit_class_bias,
        class_bias_l2=class_bias_l2,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    _write_rows_if_present(artifacts["outer"], outer_output_path)
    _write_rows_if_present(artifacts["group_summary"], group_summary_output_path)
    _write_rows_if_present(artifacts["inner_validation"], inner_validation_output_path)
    _write_rows_if_present(artifacts["selected"], selected_output_path)
    _write_rows_if_present(artifacts["predictions"], predictions_output_path)
    _write_rows_if_present(artifacts["confusion"], confusion_output_path)
    _write_rows_if_present(artifacts["per_stimulus"], per_stimulus_output_path)
    _write_rows_if_present(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


def make_logit_stack_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=cross_subject.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=cross_subject.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(cross_subject.DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(cross_subject.DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(cross_subject.DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(cross_subject.DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(cross_subject.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    sample_weightings=(cross_subject.DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING,),
    score_calibrations=("none",),
    alignment_alphas=(cross_subject.DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA,),
    max_trials_per_class_per_participant=None,
    trial_selection=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    chance_classes=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build base-model candidates for source-OOF stacking."""

    return cross_subject.make_cross_subject_candidate_configs(
        window_centers=window_centers,
        window_size=window_size,
        baseline_window=baseline_window,
        feature_modes=feature_modes,
        normalizations=normalizations,
        alignments=alignments,
        classifiers=classifiers,
        classifier_params=classifier_params,
        components_pca_values=components_pca_values,
        sample_weightings=sample_weightings,
        score_calibrations=score_calibrations,
        alignment_alphas=alignment_alphas,
        max_trials_per_class_per_participant=max_trials_per_class_per_participant,
        trial_selection=trial_selection,
        trial_selection_seed=trial_selection_seed,
        chance_classes=chance_classes,
        random_state=random_state,
        signflip_permutations=signflip_permutations,
        signflip_seed=signflip_seed,
    )


def _fit_source_oof_stacker(  # pylint: disable=too-many-arguments,too-many-locals
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    *,
    score_normalization,
    weighting,
    weighting_temperature,
    max_base_models,
    fit_class_bias,
    class_bias_l2,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    class_order = np.arange(int(candidate_configs[0].chance_classes), dtype=int)
    reference_sets = feature_cache[_feature_cache_key(candidate_configs[0])]
    oof_labels = np.concatenate([np.asarray(reference_sets[participant].labels, dtype=int) - 1 for participant in outer_train_participants])
    oof_scores = np.zeros((len(candidate_configs), oof_labels.shape[0], class_order.shape[0]), dtype=float)
    inner_rows = []

    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        candidate_feature_sets = feature_cache[_feature_cache_key(candidate_config)]
        start = 0
        for validation_index, validation_participant in enumerate(outer_train_participants):
            validation_set = candidate_feature_sets[validation_participant]
            inner_train_sets = [candidate_feature_sets[participant] for participant in outer_train_participants if participant != validation_participant]
            fitted_model = _fit_candidate_model(
                inner_train_sets,
                candidate_config,
                label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
                label_shuffle_context=(int(test_participant), int(candidate_index), int(validation_participant), int(validation_index)),
            )
            normalized_scores, score_classes = _candidate_scores_for_stacking(fitted_model, validation_set, candidate_config, class_order, score_normalization)
            stop = start + normalized_scores.shape[0]
            expected_labels = oof_labels[start:stop]
            actual_labels = np.asarray(validation_set.labels, dtype=int) - 1
            if not np.array_equal(expected_labels, actual_labels):
                raise ValueError("All logit-stack candidate configurations must keep identical source OOF trial order and labels.")
            oof_scores[candidate_index - 1, start:stop] = normalized_scores
            inner_rows.append(
                _inner_validation_row(
                    fitted_model,
                    validation_set,
                    candidate_config,
                    candidate_index,
                    test_participant,
                    validation_participant,
                    normalized_scores,
                    score_classes,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
            )
            start = stop
            if progress is not None:
                progress(
                    "DONE logit_stack_inner "
                    f"outer_test_participant={test_participant} "
                    f"candidate={candidate_index}/{len(candidate_configs)} "
                    f"validation_participant={validation_participant}"
                )
        if start != oof_labels.shape[0]:
            raise ValueError("Internal OOF score assembly error: source rows were not filled exactly once.")

    inner_candidate_balanced = np.asarray([_balanced_accuracy_for_scores(oof_scores[index], oof_labels, class_order) for index in range(len(candidate_configs))], dtype=float)
    weights, selected_candidate_positions = _fit_stacker_weights(
        oof_scores,
        oof_labels,
        class_order,
        inner_candidate_balanced,
        weighting=weighting,
        temperature=weighting_temperature,
        max_base_models=max_base_models,
    )
    weighted_scores = _weighted_score_average(oof_scores, weights)
    inner_unbiased = _balanced_accuracy_for_scores(weighted_scores, oof_labels, class_order)
    if fit_class_bias:
        class_bias, inner_stacked = _optimize_class_bias(weighted_scores, oof_labels, class_order, l2=class_bias_l2)
    else:
        class_bias = np.zeros(class_order.shape[0], dtype=float)
        inner_stacked = inner_unbiased
    return (
        LogitStackingFit(
            weights=weights,
            class_bias=class_bias,
            class_order=class_order,
            candidate_indices=tuple(int(index + 1) for index in selected_candidate_positions),
            inner_candidate_balanced=inner_candidate_balanced,
            inner_stacked_balanced_accuracy=float(inner_stacked),
            inner_unbiased_balanced_accuracy=float(inner_unbiased),
            score_normalization=score_normalization,
            weighting=weighting,
            weighting_temperature=float(weighting_temperature),
            class_bias_enabled=bool(fit_class_bias),
            class_bias_l2=float(class_bias_l2),
            max_base_models=max_base_models,
        ),
        inner_rows,
    )


def _score_outer_with_stacker(  # pylint: disable=too-many-arguments,too-many-locals
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    stack_fit,
    selected_row,
    *,
    label_shuffle_seed=None,
    label_shuffle_context=(),
):
    outer_scores = []
    fitted_models = []
    test_sets = []
    actual_components = []
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        candidate_feature_sets = feature_cache[_feature_cache_key(candidate_config)]
        train_sets = [candidate_feature_sets[participant] for participant in outer_train_participants]
        test_set = candidate_feature_sets[test_participant]
        fitted_model = _fit_candidate_model(
            train_sets,
            candidate_config,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=(*tuple(label_shuffle_context), int(candidate_index)),
        )
        normalized_scores, _score_classes = _candidate_scores_for_stacking(
            fitted_model,
            test_set,
            candidate_config,
            stack_fit.class_order,
            stack_fit.score_normalization,
        )
        outer_scores.append(normalized_scores)
        fitted_models.append(fitted_model)
        test_sets.append(test_set)
        actual_components.append(fitted_model["model_bundle"].actual_components_pca)

    reference_test_set, test_labels = _validate_test_sets(test_sets)
    score_cube = np.stack(outer_scores, axis=0)
    stacked_scores = _weighted_score_average(score_cube, stack_fit.weights) + stack_fit.class_bias[None, :]
    predictions = stack_fit.class_order[np.argmax(stacked_scores, axis=1)]
    rank_metrics = cross_subject._ranked_label_metrics(test_labels, stacked_scores, stack_fit.class_order)  # pylint: disable=protected-access
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))

    outer_row, _template_predictions = cross_subject._score_outer_fold_model(  # pylint: disable=protected-access
        fitted_models[0],
        test_sets[0],
        candidate_configs[0],
        include_predictions=False,
    )
    outer_row.update(
        {
            "classifier": LOGIT_STACK_CLASSIFIER,
            "classifier_param": "",
            "accuracy": accuracy,
            "percent": 100.0 * accuracy,
            "balanced_accuracy": balanced_accuracy,
            "balanced_percent": 100.0 * balanced_accuracy,
            "top2_accuracy": rank_metrics["top2_accuracy"],
            "top2_percent": 100.0 * rank_metrics["top2_accuracy"],
            "top3_accuracy": rank_metrics["top3_accuracy"],
            "top3_percent": 100.0 * rank_metrics["top3_accuracy"],
            "mean_true_label_rank": rank_metrics["mean_true_label_rank"],
            "median_true_label_rank": rank_metrics["median_true_label_rank"],
            "above_chance": bool(balanced_accuracy > outer_row["chance_accuracy"]),
            "actual_components_pca": _format_sequence(actual_components),
            "pca_explained_variance_percent": "",
        }
    )
    _add_stack_fit_fields(outer_row, stack_fit, selected_row, candidate_configs)

    prediction_rows = cross_subject._prediction_rows(  # pylint: disable=protected-access
        reference_test_set,
        test_labels,
        predictions,
        rank_metrics["true_label_ranks"],
        config=candidate_configs[0],
        actual_components_pca=_format_sequence(actual_components),
    )
    for row in prediction_rows:
        row["classifier"] = LOGIT_STACK_CLASSIFIER
        row["classifier_param"] = ""
        _add_stack_fit_fields(row, stack_fit, selected_row, candidate_configs)
    return outer_row, prediction_rows


def _fit_candidate_model(train_sets, config, *, label_shuffle_seed=None, label_shuffle_context=()):
    return cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
        train_sets,
        config,
        cross_subject._resolved_classifier_param(config),  # pylint: disable=protected-access
        label_shuffle_seed=label_shuffle_seed,
        label_shuffle_context=label_shuffle_context,
    )


def _candidate_scores_for_stacking(fitted_model, feature_set, config, class_order, score_normalization):
    scores, classes = cross_subject._candidate_model_scores(fitted_model, feature_set, config)  # pylint: disable=protected-access
    aligned = _align_score_columns(scores, classes, class_order)
    return _normalize_scores(aligned, mode=score_normalization), np.asarray(class_order, dtype=int)


def _inner_validation_row(
    fitted_model,
    validation_set,
    config,
    candidate_index,
    outer_test_participant,
    validation_participant,
    normalized_scores,
    score_classes,
    *,
    label_shuffle_control,
    label_shuffle_seed,
):
    labels = np.asarray(validation_set.labels, dtype=int) - 1
    predictions = np.asarray(score_classes, dtype=int)[np.argmax(normalized_scores, axis=1)]
    rank_metrics = cross_subject._ranked_label_metrics(labels, normalized_scores, score_classes)  # pylint: disable=protected-access
    train_window = cross_subject._centered_window(config.window_center, config.window_size)  # pylint: disable=protected-access
    train_labels = fitted_model["train_labels"]
    train_class_counts = fitted_model["train_class_counts"]
    test_class_counts = Counter(np.asarray(validation_set.labels, dtype=int).tolist())
    return {
        "selection_mode": "source_oof_logit_stack",
        "selection_metric": "source_oof_balanced_accuracy",
        "outer_fold": int(outer_test_participant),
        "test_participant": int(outer_test_participant),
        "outer_test_participant": int(outer_test_participant),
        "inner_fold": int(validation_participant),
        "inner_validation_participant": int(validation_participant),
        "inner_train_participants": ",".join(str(participant) for participant in fitted_model["train_participants"]),
        "n_inner_train_participants": fitted_model["n_train_participants"],
        "candidate_index": int(candidate_index),
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": config.alignment,
        "classifier": config.classifier,
        "classifier_param": fitted_model["classifier_param"],
        "components_pca": config.components_pca,
        "sample_weighting": getattr(config, "sample_weighting", "none"),
        "score_calibration": getattr(config, "score_calibration", "none"),
        "alignment_alpha": getattr(config, "alignment_alpha", 1.0),
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        "trial_selection": getattr(config, "trial_selection", ""),
        "trial_selection_seed": cross_subject._seed_field(getattr(config, "trial_selection_seed", None)),  # pylint: disable=protected-access
        "accuracy": float(accuracy_score(labels, predictions)),
        "percent": 100.0 * float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "balanced_percent": 100.0 * float(balanced_accuracy_score(labels, predictions)),
        "top2_accuracy": rank_metrics["top2_accuracy"],
        "top2_percent": 100.0 * rank_metrics["top2_accuracy"],
        "top3_accuracy": rank_metrics["top3_accuracy"],
        "top3_percent": 100.0 * rank_metrics["top3_accuracy"],
        "mean_true_label_rank": rank_metrics["mean_true_label_rank"],
        "median_true_label_rank": rank_metrics["median_true_label_rank"],
        "chance_accuracy": 1.0 / int(config.chance_classes),
        "chance_percent": 100.0 / int(config.chance_classes),
        "above_chance": bool(float(balanced_accuracy_score(labels, predictions)) > 1.0 / int(config.chance_classes)),
        "n_train_trials": int(train_labels.shape[0]),
        "n_test_trials": int(labels.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())) if train_class_counts else 0,
        "min_test_trials_per_class": int(min(test_class_counts.values())) if test_class_counts else 0,
        "actual_components_pca": fitted_model["model_bundle"].actual_components_pca,
        "pca_explained_variance_percent": fitted_model["model_bundle"].explained_variance_percent,
        "label_shuffle_control": bool(label_shuffle_control),
        "label_shuffle_seed": int(label_shuffle_seed) if label_shuffle_control else "",
    }


def _selected_row_for_stacker(test_participant, stack_fit, candidate_configs, inner_rows):
    best_position = int(np.nanargmax(stack_fit.inner_candidate_balanced))
    best_index = best_position + 1
    best_config = candidate_configs[best_position]
    train_window = cross_subject._centered_window(best_config.window_center, best_config.window_size)  # pylint: disable=protected-access
    candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == best_index]
    best_raw = np.asarray([float(row["balanced_accuracy"]) for row in candidate_rows], dtype=float)
    row = {
        "selection_mode": "source_oof_logit_stack",
        "selection_metric": "source_oof_balanced_accuracy",
        "outer_fold": int(test_participant),
        "test_participant": int(test_participant),
        "selected_candidate_index": int(best_index),
        "n_candidates": int(len(candidate_configs)),
        "n_inner_folds": int(len(candidate_rows)),
        "selected_inner_balanced_accuracy_mean": float(np.mean(best_raw)) if best_raw.size else float(stack_fit.inner_candidate_balanced[best_position]),
        "selected_inner_balanced_accuracy_median": float(np.median(best_raw)) if best_raw.size else float(stack_fit.inner_candidate_balanced[best_position]),
        "selected_inner_balanced_accuracy_sem": cross_subject._sem(best_raw) if best_raw.size else 0.0,  # pylint: disable=protected-access
        "selected_window_center_s": best_config.window_center,
        "selected_window_size_s": best_config.window_size,
        "selected_window_start_s": train_window[0],
        "selected_window_stop_s": train_window[1],
        "selected_feature_mode": best_config.feature_mode,
        "selected_normalization": best_config.normalization,
        "selected_alignment": best_config.alignment,
        "selected_classifier": best_config.classifier,
        "selected_classifier_param": best_config.classifier_param,
        "selected_components_pca": best_config.components_pca,
        "selected_max_trials_per_class_per_participant": best_config.max_trials_per_class_per_participant,
        "selected_sample_weighting": getattr(best_config, "sample_weighting", "none"),
        "selected_score_calibration": getattr(best_config, "score_calibration", "none"),
        "selected_alignment_alpha": getattr(best_config, "alignment_alpha", 1.0),
    }
    _add_stack_fit_fields(row, stack_fit, row, candidate_configs)
    return row


def _add_selected_candidate_fields(row, selected_row):
    for key, value in selected_row.items():
        row[key] = value


def _add_stack_fit_fields(row, stack_fit, _selected_row, candidate_configs):
    nonzero = [(index + 1, float(weight)) for index, weight in enumerate(stack_fit.weights) if float(weight) > 1e-12]
    row["outer_evaluation_mode"] = LOGIT_STACK_CLASSIFIER
    row["stacker_score_normalization"] = stack_fit.score_normalization
    row["stacker_weighting"] = stack_fit.weighting
    row["stacker_weighting_temperature"] = stack_fit.weighting_temperature
    row["stacker_max_base_models"] = "" if stack_fit.max_base_models is None else int(stack_fit.max_base_models)
    row["stacker_class_bias"] = bool(stack_fit.class_bias_enabled)
    row["stacker_class_bias_l2"] = stack_fit.class_bias_l2
    row["stacker_candidate_indices"] = _format_sequence(index for index, _weight in nonzero)
    row["stacker_weights"] = _format_float_mapping(nonzero)
    row["stacker_n_base_models"] = int(len(nonzero))
    row["stacker_inner_balanced_accuracy"] = stack_fit.inner_stacked_balanced_accuracy
    row["stacker_inner_unbiased_balanced_accuracy"] = stack_fit.inner_unbiased_balanced_accuracy
    row["stacker_candidate_inner_balanced_accuracy"] = _format_float_mapping(
        (index + 1, value) for index, value in enumerate(stack_fit.inner_candidate_balanced)
    )
    row["stacker_class_bias_min"] = float(np.min(stack_fit.class_bias)) if stack_fit.class_bias.size else 0.0
    row["stacker_class_bias_max"] = float(np.max(stack_fit.class_bias)) if stack_fit.class_bias.size else 0.0
    row["stacker_class_bias_l2_norm"] = float(np.sqrt(np.sum(np.square(stack_fit.class_bias))))
    row["stacker_classifiers"] = _format_sequence(candidate_configs[index - 1].classifier for index, _weight in nonzero)
    row["stacker_window_centers_s"] = _format_sequence(candidate_configs[index - 1].window_center for index, _weight in nonzero)
    row["stacker_feature_modes"] = _format_sequence(candidate_configs[index - 1].feature_mode for index, _weight in nonzero)
    row["stacker_normalizations"] = _format_sequence(candidate_configs[index - 1].normalization for index, _weight in nonzero)
    row["stacker_components_pca"] = _format_sequence(candidate_configs[index - 1].components_pca for index, _weight in nonzero)
    row["selected_candidate_indices"] = row["stacker_candidate_indices"]
    row["selected_ensemble_weights"] = row["stacker_weights"]


def _add_logit_stack_group_summary_fields(summary_rows, outer_rows):
    if not outer_rows:
        return
    for row in summary_rows:
        row["outer_evaluation_mode"] = LOGIT_STACK_CLASSIFIER
        row["stacker_score_normalization_counts"] = _format_counter(Counter(str(value.get("stacker_score_normalization", "")) for value in outer_rows))
        row["stacker_weighting_counts"] = _format_counter(Counter(str(value.get("stacker_weighting", "")) for value in outer_rows))
        row["stacker_n_base_models_mean"] = _nanmean([value.get("stacker_n_base_models", np.nan) for value in outer_rows])
        row["stacker_inner_balanced_accuracy_mean"] = _nanmean([value.get("stacker_inner_balanced_accuracy", np.nan) for value in outer_rows])
        row["stacker_inner_unbiased_balanced_accuracy_mean"] = _nanmean([value.get("stacker_inner_unbiased_balanced_accuracy", np.nan) for value in outer_rows])


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    representative_configs = {}
    for candidate_config in candidate_configs:
        representative_configs.setdefault(_feature_cache_key(candidate_config), candidate_config)
    feature_cache = {}
    for key, candidate_config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD logit_stack_feature_set "
                f"window_center={candidate_config.window_center} "
                f"feature_mode={candidate_config.feature_mode} "
                f"normalization={candidate_config.normalization}"
            )
        feature_cache[key] = {
            participant: cross_subject.load_participant_stimulus_features(data_folder, participant, config=candidate_config) for participant in participants
        }
    return feature_cache


def _feature_cache_key(config):
    return cross_subject._feature_cache_key(config)  # pylint: disable=protected-access


def _fit_stacker_weights(score_cube, labels, class_order, inner_candidate_balanced, *, weighting, temperature, max_base_models):
    score_cube = np.asarray(score_cube, dtype=float)
    if score_cube.ndim != 3:
        raise ValueError("score_cube must have shape n_models x n_trials x n_classes.")
    n_models = int(score_cube.shape[0])
    positions = np.arange(n_models, dtype=int)
    if max_base_models is not None and max_base_models < n_models:
        order = np.argsort(-np.asarray(inner_candidate_balanced, dtype=float), kind="mergesort")[:max_base_models]
        positions = np.sort(order)
        working_scores = score_cube[positions]
        working_inner = np.asarray(inner_candidate_balanced, dtype=float)[positions]
    else:
        working_scores = score_cube
        working_inner = np.asarray(inner_candidate_balanced, dtype=float)

    if weighting == "uniform":
        working_weights = np.full(working_scores.shape[0], 1.0 / working_scores.shape[0], dtype=float)
    else:
        working_weights = _softmax_weights(working_inner, temperature=temperature)
        if weighting == "greedy_balanced" and working_weights.shape[0] > 1:
            working_weights = _coordinate_simplex_search(working_scores, labels, class_order, working_weights)

    weights = np.zeros(n_models, dtype=float)
    weights[positions] = working_weights
    weights = _normalize_weight_vector(weights)
    return weights, tuple(int(position) for position in positions if weights[position] > 1e-12)


def _coordinate_simplex_search(score_cube, labels, class_order, initial_weights):
    weights = _normalize_weight_vector(initial_weights)
    best = _stack_objective(score_cube, labels, class_order, weights)
    for step in (0.25, 0.10, 0.05, 0.02, 0.01):
        improved = True
        passes = 0
        while improved and passes < 100:
            improved = False
            passes += 1
            for source, target in product(range(weights.shape[0]), repeat=2):
                if source == target or weights[source] <= 1e-12:
                    continue
                delta = min(float(step), float(weights[source]))
                candidate = weights.copy()
                candidate[source] -= delta
                candidate[target] += delta
                value = _stack_objective(score_cube, labels, class_order, candidate)
                if value > best + 1e-12:
                    weights = candidate
                    best = value
                    improved = True
    return _normalize_weight_vector(weights)


def _stack_objective(score_cube, labels, class_order, weights):
    return _balanced_accuracy_for_scores(_weighted_score_average(score_cube, weights), labels, class_order)


def _weighted_score_average(score_cube, weights):
    return np.tensordot(_normalize_weight_vector(weights), np.asarray(score_cube, dtype=float), axes=(0, 0))


def _softmax_weights(scores, *, temperature):
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        return np.full(scores.shape[0], 1.0 / scores.shape[0], dtype=float)
    centered = scores - np.max(scores)
    exp_scores = np.exp(np.clip(centered / float(temperature), -50.0, 50.0))
    return _normalize_weight_vector(exp_scores)


def _normalize_weight_vector(weights):
    weights = np.asarray(weights, dtype=float).ravel()
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape[0], 1.0 / weights.shape[0], dtype=float)
    return weights / total


def _optimize_class_bias(scores, labels, class_order, *, l2):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    bias = np.zeros(class_order.shape[0], dtype=float)
    best = _bias_objective(scores, labels, class_order, bias, l2=l2)
    for step in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        improved = True
        passes = 0
        while improved and passes < 100:
            improved = False
            passes += 1
            for column in range(bias.shape[0]):
                for direction in (1.0, -1.0):
                    candidate = bias.copy()
                    candidate[column] += direction * step
                    candidate -= np.mean(candidate)
                    value = _bias_objective(scores, labels, class_order, candidate, l2=l2)
                    if value > best + 1e-12:
                        bias = candidate
                        best = value
                        improved = True
    return bias, _balanced_accuracy_for_scores(scores + bias[None, :], labels, class_order)


def _bias_objective(scores, labels, class_order, bias, *, l2):
    balanced = _balanced_accuracy_for_scores(scores + bias[None, :], labels, class_order)
    return balanced - float(l2) * float(np.mean(np.square(bias)))


def _balanced_accuracy_for_scores(scores, labels, class_order):
    predictions = np.asarray(class_order, dtype=int)[np.argmax(scores, axis=1)]
    return float(balanced_accuracy_score(labels, predictions))


def _align_score_columns(scores, score_classes, class_order):
    scores = np.asarray(scores, dtype=float)
    score_classes = np.asarray(score_classes, dtype=int).ravel()
    class_order = np.asarray(class_order, dtype=int).ravel()
    if scores.ndim != 2:
        return np.zeros((0, class_order.shape[0]), dtype=float)
    if scores.shape[1] == 0 or score_classes.size == 0:
        return np.zeros((scores.shape[0], class_order.shape[0]), dtype=float)
    finite_scores = np.where(np.isfinite(scores), scores, np.nan)
    finite_min = np.nanmin(finite_scores, axis=1)
    finite_min = np.where(np.isfinite(finite_min), finite_min - 1.0, -1.0)
    aligned = np.repeat(finite_min[:, None], class_order.shape[0], axis=1)
    class_to_column = {int(label): column for column, label in enumerate(class_order.tolist())}
    for source_column, label in enumerate(score_classes.tolist()):
        target_column = class_to_column.get(int(label))
        if target_column is not None and source_column < scores.shape[1]:
            aligned[:, target_column] = scores[:, source_column]
    return aligned


def _normalize_scores(scores, *, mode):
    mode = _normalize_score_normalization(mode)
    scores = np.asarray(scores, dtype=float)
    if mode == "none":
        return np.where(np.isfinite(scores), scores, 0.0)
    if mode == "row_z":
        sanitized = np.where(np.isfinite(scores), scores, np.nan)
        row_mean = np.nanmean(sanitized, axis=1, keepdims=True)
        row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
        centered = np.where(np.isfinite(scores), scores, row_mean) - row_mean
        row_std = np.std(centered, axis=1, keepdims=True)
        row_std = np.where(row_std > 1e-12, row_std, 1.0)
        return centered / row_std
    if mode == "rank":
        output = np.empty_like(scores, dtype=float)
        for row_index, row in enumerate(scores):
            finite = np.isfinite(row)
            if not np.any(finite):
                output[row_index] = 0.0
                continue
            ranking_scores = np.where(finite, row, -np.inf)
            descending = np.argsort(-ranking_scores, kind="mergesort")
            ranks = np.empty(row.shape[0], dtype=float)
            ranks[descending] = np.arange(row.shape[0], dtype=float)
            output[row_index] = -ranks
        return output
    raise ValueError(f"Unsupported score normalization: {mode}")


def _validate_test_sets(test_sets):
    reference = test_sets[0]
    reference_labels = np.asarray(reference.labels, dtype=int) - 1
    reference_trials = cross_subject._feature_set_trial_indices(reference)  # pylint: disable=protected-access
    for test_set in test_sets[1:]:
        labels = np.asarray(test_set.labels, dtype=int) - 1
        trials = cross_subject._feature_set_trial_indices(test_set)  # pylint: disable=protected-access
        if int(test_set.participant) != int(reference.participant):
            raise ValueError("Logit stacking requires all candidates to score the same held-out participant.")
        if not np.array_equal(labels, reference_labels) or not np.array_equal(trials, reference_trials):
            raise ValueError("Logit stacking requires identical held-out trial labels and trial order across candidate configurations.")
    return reference, reference_labels


def _normalize_outer_participants(participants, outer_participants):
    if outer_participants is None:
        return tuple(participants)
    outer_participants = tuple(int(participant) for participant in outer_participants)
    if not outer_participants:
        raise ValueError("At least one outer participant is required.")
    unknown = sorted(set(outer_participants) - set(participants))
    if unknown:
        raise ValueError(f"Outer participants must be part of participants: {unknown}")
    return outer_participants


def _normalize_score_normalization(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in LOGIT_STACK_SCORE_NORMALIZATION_MODES:
        raise ValueError(f"score_normalization must be one of {LOGIT_STACK_SCORE_NORMALIZATION_MODES}.")
    return token


def _normalize_weighting(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in LOGIT_STACK_WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {LOGIT_STACK_WEIGHTING_MODES}.")
    return token


def _normalize_temperature(value):
    temperature = float(value)
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("weighting_temperature must be finite and positive.")
    return temperature


def _normalize_max_base_models(value):
    if value in (None, "", "none", "all"):
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("max_base_models must be positive, all, or none.")
    return value


def _normalize_nonnegative_float(value, name):
    value = float(value)
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _format_sequence(values):
    return ";".join(str(value) for value in values)


def _format_float_mapping(items):
    return ";".join(f"{key}:{float(value):.10g}" for key, value in items)


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _nanmean(values):
    values = np.asarray([float(value) for value in values if _is_float_like(value)], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _is_float_like(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _write_rows_if_present(rows, path):
    if path and rows:
        write_alpha_metrics_csv(_rows_with_consistent_fields(rows), path)


def _rows_with_consistent_fields(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window start must be before stop.")
    return parts


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_feature_mode_list(value: str) -> tuple[str, ...]:
    return tuple(token.strip().lower().replace("-", "_") for token in _parse_token_list(value))


def _parse_normalization_list(value: str) -> tuple[str, ...]:
    return tuple(token.strip().lower().replace("-", "_") for token in _parse_token_list(value))


def _parse_alignment_list(value: str) -> tuple[str, ...]:
    return tuple(token.strip().lower().replace("-", "_") for token in _parse_token_list(value))


def _parse_sample_weighting_list(value: str) -> tuple[str, ...]:
    return tuple(token.strip().lower().replace("-", "_") for token in _parse_token_list(value))


def _parse_score_calibration_list(value: str) -> tuple[str, ...]:
    return tuple(token.strip().lower().replace("-", "_") for token in _parse_token_list(value))


def _parse_int_or_inf_list(value: str) -> tuple[int | float | str, ...]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower().replace("_", "-") == cross_subject.AUTO_COMPONENTS_PCA_GRID_TOKEN:
            values.append(cross_subject.AUTO_COMPONENTS_PCA_GRID_TOKEN)
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
        elif token.lower().replace("_", "-") == cross_subject.AUTO_CLASSIFIER_PARAM_GRID_TOKEN:
            values.append(cross_subject.AUTO_CLASSIFIER_PARAM_GRID_TOKEN)
        else:
            values.append(parse_classifier_param(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one classifier parameter value is required.")
    return tuple(values)


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run source-only LOSO logit stacking with source-subject OOF calibration on Part*Data.mat files only.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=cross_subject.DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate. Defaults to all participants.")
    parser.add_argument(
        "--window-centers",
        type=parse_float_list,
        default=cross_subject.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
        help="Comma-separated candidate window centers in seconds.",
    )
    parser.add_argument("--window-size", type=float, default=cross_subject.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Candidate window size in seconds.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument("--feature-modes", type=_parse_feature_mode_list, default=("sensor_flat",), help="Comma-separated feature modes.")
    parser.add_argument("--normalizations", type=_parse_normalization_list, default=("subject_baseline_whiten",), help="Comma-separated subject normalization modes.")
    parser.add_argument("--alignments", type=_parse_alignment_list, default=("none",), help="Comma-separated training alignment modes. Use none for strict source-only decoding.")
    parser.add_argument("--classifiers", type=_parse_token_list, default=("multinomial-logistic", "multiclass-svm"), help="Comma-separated classifier names.")
    parser.add_argument(
        "--classifier-params",
        type=_parse_classifier_param_grid,
        default=(float("nan"),),
        help="Comma-separated classifier params; default or auto-grid are supported.",
    )
    parser.add_argument("--components-pca-values", type=_parse_int_or_inf_list, default=(64, 128), help="Comma-separated PCA component counts, inf, or auto-grid.")
    parser.add_argument("--sample-weightings", type=_parse_sample_weighting_list, default=("none", "subject_class_balanced"), help="Comma-separated sample weighting modes.")
    parser.add_argument(
        "--score-calibrations",
        type=_parse_score_calibration_list,
        default=("none",),
        help="Base-model score calibration modes. Keep none to let the stacker calibrate scores.",
    )
    parser.add_argument("--alignment-alphas", type=parse_float_list, default=(1.0,), help="Comma-separated alignment blend factors in [0,1].")
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Optional deterministic cap on trials per class and participant.")
    parser.add_argument(
        "--trial-selection",
        choices=cross_subject.TRIAL_SELECTION_MODES,
        default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
        help="Trial subset policy when a trial cap is set.",
    )
    parser.add_argument("--trial-selection-seed", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED, help="Seed for random trial selection.")
    parser.add_argument(
        "--stacker-score-normalization",
        choices=LOGIT_STACK_SCORE_NORMALIZATION_MODES,
        default=DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION,
        help="Per-base-model score transform before stacking.",
    )
    parser.add_argument("--stacker-weighting", choices=LOGIT_STACK_WEIGHTING_MODES, default=DEFAULT_LOGIT_STACK_WEIGHTING, help="Source-OOF base-model weighting policy.")
    parser.add_argument(
        "--stacker-weighting-temperature",
        type=float,
        default=DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE,
        help="Softmax temperature for inner_softmax and greedy initialization.",
    )
    parser.add_argument(
        "--stacker-max-base-models",
        default=str(DEFAULT_LOGIT_STACK_MAX_BASE_MODELS),
        help="Preselect at most this many base models by source OOF BA; use all/none for no cap.",
    )
    parser.add_argument("--no-stacker-class-bias", action="store_true", help="Disable source-OOF class-bias calibration.")
    parser.add_argument("--stacker-class-bias-l2", type=float, default=DEFAULT_LOGIT_STACK_CLASS_BIAS_L2, help="L2 penalty for source-OOF class-bias calibration.")
    parser.add_argument("--chance-classes", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to classifiers.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Shuffle training labels within each source participant for a null-control benchmark.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_logit_stack_outer.csv", help="Held-out participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_logit_stack_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_logit_stack_inner_validation.csv", help="Source OOF inner validation score CSV.")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_logit_stack_selected.csv", help="Stacker weights and calibration CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_logit_stack_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_logit_stack_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_logit_stack_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_logit_stack_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_logit_stack(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    candidate_configs = make_logit_stack_candidate_configs(
        window_centers=args.window_centers,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_modes=args.feature_modes,
        normalizations=args.normalizations,
        alignments=args.alignments,
        classifiers=args.classifiers,
        classifier_params=args.classifier_params,
        components_pca_values=args.components_pca_values,
        sample_weightings=args.sample_weightings,
        score_calibrations=args.score_calibrations,
        alignment_alphas=args.alignment_alphas,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        trial_selection=args.trial_selection,
        trial_selection_seed=args.trial_selection_seed,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    artifacts = export_cross_subject_logit_stacking(
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
        outer_participants=outer_participants,
        score_normalization=args.stacker_score_normalization,
        weighting=args.stacker_weighting,
        weighting_temperature=args.stacker_weighting_temperature,
        max_base_models=_normalize_max_base_models(args.stacker_max_base_models),
        fit_class_bias=not args.no_stacker_class_bias,
        class_bias_l2=args.stacker_class_bias_l2,
        progress=lambda message: print(message, flush=True),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    print(f"Wrote {len(artifacts['outer'])} logit-stack outer rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['inner_validation'])} source-OOF validation rows to {args.inner_validation_output}")
    print(f"Wrote {len(artifacts['selected'])} stacker selected rows to {args.selected_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['confusion'])} confusion rows to {args.confusion_output}")
    print(f"Wrote {len(artifacts['per_stimulus'])} per-stimulus rows to {args.per_stimulus_output}")
    print(f"Wrote {len(artifacts['confusion_pairs'])} confusion-pair rows to {args.confusion_pairs_output}")
    return 0


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_cross_subject_logit_stack(argv, prog)


if __name__ == "__main__":
    raise SystemExit(main())
