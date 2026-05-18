"""Cross-subject stimulus decoding with Procrustes hyperalignment."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from reptrace.decoding.hyperalignment import (
    CLASS_ALIGNMENT_SAMPLE_MODES,
    class_alignment_matrices,
    fit_class_hyperalignment,
    fit_projection_to_hyperalignment,
)
from reptrace.decoding.windowed import fit_window_model, predict_window_model, transform_window_features
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec.alignment_window import (
    resolved_alignment_window,
    transform_with_alignment_projection,
    uses_separate_alignment_window,
    validate_paired_feature_sets,
)
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import get_default_classifier_param, should_use_default_classifier_param, train_multiclass_classifier
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cue_calibration import load_participant_cue_calibration_features
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    DEFAULT_CROSS_SUBJECT_CLASSIFIER,
    DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
    DEFAULT_CROSS_SUBJECT_NORMALIZATION,
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    DEFAULT_CROSS_SUBJECT_WINDOW_CENTER,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    FEATURE_MODES,
    NORMALIZATION_MODES,
    CrossSubjectStimulusConfig,
    load_participant_stimulus_features,
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_predictions,
)

HYPERALIGNMENT_TARGET_CENTERING_MODES = ("group_mean", "target_unsupervised")
DEFAULT_HYPERALIGNMENT_COMPONENTS = 64
DEFAULT_HYPERALIGNMENT_SAMPLE_MODE = "class_repetition"
DEFAULT_HYPERALIGNMENT_TARGET_CENTERING = "target_unsupervised"
ALIGNMENT_DATASETS = ("main", "cue")


@dataclass(frozen=True)
class CrossSubjectHyperalignmentConfig:  # pylint: disable=too-many-instance-attributes
    """Parameters for LOSO cross-subject stimulus decoding with hyperalignment."""

    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    alignment_window_center: float | None = None
    alignment_window_size: float | None = None
    alignment_data: str = "main"
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = "sensor_flat"
    normalization: str = DEFAULT_CROSS_SUBJECT_NORMALIZATION
    classifier: str = DEFAULT_CROSS_SUBJECT_CLASSIFIER
    classifier_param: object = float("nan")
    components_pca: int | float = DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA
    max_trials_per_class_per_participant: int | None = None
    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0
    hyperalignment_components: int | float = DEFAULT_HYPERALIGNMENT_COMPONENTS
    hyperalignment_iterations: int = 10
    hyperalignment_initialization: str = "pca"
    hyperalignment_sample_mode: str = DEFAULT_HYPERALIGNMENT_SAMPLE_MODE
    hyperalignment_repetitions_per_class: int | None = None
    target_calibration_trials_per_class: int = 0
    target_centering: str = DEFAULT_HYPERALIGNMENT_TARGET_CENTERING


def evaluate_cross_subject_hyperalignment(data_folder, participants, *, config=None, outer_participants=None, progress=None, label_shuffle_control=False, label_shuffle_seed=0):
    """Run fixed-pipeline LOSO stimulus decoding with Procrustes hyperalignment.

    With the default ``target_calibration_trials_per_class=0``, the held-out
    participant is transformed by the average training-subject hyperalignment projection.
    This is calibration-free with respect to target labels. Setting a positive
    target-calibration count estimates a target-specific Procrustes projection from
    those labeled target trials and excludes them from scoring.
    """

    config = _normalized_hyperalignment_config(config or CrossSubjectHyperalignmentConfig())
    if label_shuffle_control and config.target_calibration_trials_per_class > 0:
        raise ValueError("label_shuffle_control is only supported with target_calibration_trials_per_class=0.")
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for cross-subject hyperalignment decoding.")
    outer_participants = tuple(participants if outer_participants is None else [int(participant) for participant in outer_participants])
    unknown = sorted(set(outer_participants) - set(participants))
    if unknown:
        raise ValueError(f"Outer participants must be part of participants: {unknown}")

    feature_config = _feature_extraction_config(config, window_center=config.window_center, window_size=config.window_size)
    alignment_window = resolved_alignment_window(config)
    alignment_feature_config = _feature_extraction_config(config, window_center=alignment_window.center, window_size=alignment_window.size)
    feature_sets = []
    alignment_feature_sets = []
    for participant in participants:
        if progress is not None:
            progress(f"LOAD participant={participant}")
        feature_set = load_participant_stimulus_features(data_folder, participant, config=feature_config)
        if _alignment_data(config) == "cue":
            if progress is not None:
                progress(f"LOAD cue_alignment participant={participant}")
            alignment_feature_set = load_participant_cue_calibration_features(data_folder, participant, config=alignment_feature_config)
        elif uses_separate_alignment_window(config):
            alignment_feature_set = load_participant_stimulus_features(data_folder, participant, config=alignment_feature_config)
            validate_paired_feature_sets(feature_set, alignment_feature_set, participant=participant)
        else:
            alignment_feature_set = feature_set
        feature_sets.append(feature_set)
        alignment_feature_sets.append(alignment_feature_set)
    alignment_sets_by_participant = {feature_set.participant: feature_set for feature_set in alignment_feature_sets}

    classifier_param = _resolved_classifier_param(config)
    outer_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        train_sets = [feature_set for feature_set in feature_sets if int(feature_set.participant) != int(test_participant)]
        test_set = next(feature_set for feature_set in feature_sets if int(feature_set.participant) == int(test_participant))
        train_alignment_sets = [alignment_sets_by_participant[feature_set.participant] for feature_set in train_sets]
        test_alignment_set = alignment_sets_by_participant[test_set.participant]
        outer_row, participant_predictions = _evaluate_hyperalignment_outer_fold(
            train_sets,
            test_set,
            train_alignment_sets,
            test_alignment_set,
            config,
            classifier_param,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
        )
        outer_row["label_shuffle_control"] = bool(label_shuffle_control)
        outer_row["label_shuffle_seed"] = int(label_shuffle_seed) if label_shuffle_control else ""
        for row in participant_predictions:
            row["label_shuffle_control"] = bool(label_shuffle_control)
            row["label_shuffle_seed"] = int(label_shuffle_seed) if label_shuffle_control else ""
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    group_summary_rows = summarize_cross_subject_hyperalignment(outer_rows, config=config)
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def export_cross_subject_hyperalignment(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    config=None,
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run cross-subject Procrustes hyperalignment decoding and write CSV artifacts."""

    artifacts = evaluate_cross_subject_hyperalignment(
        data_folder,
        participants,
        config=config,
        outer_participants=outer_participants,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
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


def summarize_cross_subject_hyperalignment(outer_rows, *, config=None):
    """Summarize held-out participant scores for one hyperalignment configuration."""

    if not outer_rows:
        return []
    config = _normalized_hyperalignment_config(config or CrossSubjectHyperalignmentConfig())
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = _finite_metric_values(outer_rows, "top2_accuracy")
    top3 = _finite_metric_values(outer_rows, "top3_accuracy")
    ranks = _finite_metric_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    selected_actual_components = Counter(int(row["hyperalignment_actual_components"]) for row in outer_rows)
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": _centered_window(config.window_center, config.window_size)[0],
            "window_stop_s": _centered_window(config.window_center, config.window_size)[1],
            "alignment_window_center_s": resolved_alignment_window(config).center,
            "alignment_window_size_s": resolved_alignment_window(config).size,
            "alignment_window_start_s": resolved_alignment_window(config).start,
            "alignment_window_stop_s": resolved_alignment_window(config).stop,
            "alignment_data": _alignment_data(config),
            "baseline_window_start_s": config.baseline_window[0],
            "baseline_window_stop_s": config.baseline_window[1],
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "alignment": _alignment_label(config),
            "hyperalignment_sample_mode": config.hyperalignment_sample_mode,
            "hyperalignment_requested_components": config.hyperalignment_components,
            "hyperalignment_actual_component_counts": _format_counter(selected_actual_components),
            "hyperalignment_iterations": config.hyperalignment_iterations,
            "hyperalignment_initialization": config.hyperalignment_initialization,
            "hyperalignment_repetitions_per_class": config.hyperalignment_repetitions_per_class,
            "target_calibration_trials_per_class": config.target_calibration_trials_per_class,
            "target_centering": config.target_centering,
            "classifier": config.classifier,
            "components_pca": config.components_pca,
            "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "top2_accuracy_mean": _nanmean_or_nan(top2),
            "top2_percent_mean": _percent_nanmean_or_nan(top2),
            "top2_chance_accuracy": min(2.0 * chance, 1.0),
            "top2_chance_percent": min(200.0 * chance, 100.0),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "top3_chance_accuracy": min(3.0 * chance, 1.0),
            "top3_chance_percent": min(300.0 * chance, 100.0),
            "mean_true_label_rank_mean": _nanmean_or_nan(ranks),
            "mean_true_label_rank_sem": _sem_or_nan(ranks),
            "chance_mean_rank": 0.5 * ((1.0 / chance) + 1.0),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": int(np.sum(differences > 0)),
            "participants_total": int(len(differences)),
            "participants_at_or_below_chance": int(np.sum(differences <= 0)),
            "one_sided_exact_sign_p_value": _one_sided_exact_sign_p_value(differences),
            "one_sided_signflip_p_value": _one_sided_signflip_p_value(differences, n_permutations=config.signflip_permutations, seed=config.signflip_seed),
        }
    ]


def _evaluate_hyperalignment_outer_fold(  # pylint: disable=too-many-locals
    train_sets,
    test_set,
    train_alignment_sets,
    test_alignment_set,
    config,
    classifier_param,
    *,
    label_shuffle_seed=None,
):
    train_features_by_subject = {
        feature_set.participant: alignment_feature_set.features
        for feature_set, alignment_feature_set in zip(train_sets, train_alignment_sets, strict=True)
    }
    train_decode_labels_by_subject = {
        feature_set.participant: _training_labels(feature_set, label_shuffle_seed=label_shuffle_seed, context=(test_set.participant,)) for feature_set in train_sets
    }
    train_alignment_labels_by_subject = {
        feature_set.participant: train_decode_labels_by_subject[feature_set.participant] if _alignment_data(config) == "main" else np.asarray(alignment_feature_set.labels, dtype=int)
        for feature_set, alignment_feature_set in zip(train_sets, train_alignment_sets, strict=True)
    }
    if _alignment_data(config) == "cue":
        alignment_classes = _common_alignment_classes([*train_alignment_labels_by_subject.values(), np.asarray(test_alignment_set.labels, dtype=int)])
        train_features_by_subject, train_alignment_labels_by_subject = _restrict_alignment_to_classes(
            train_features_by_subject,
            train_alignment_labels_by_subject,
            alignment_classes,
        )
    score_mask = np.ones(test_set.labels.shape[0], dtype=bool)
    calibration_mask = np.zeros(test_set.labels.shape[0], dtype=bool)
    if config.target_calibration_trials_per_class > 0:
        calibration_mask = _target_calibration_mask(test_set.labels, config.target_calibration_trials_per_class)
        score_mask = ~calibration_mask

    alignment_repetitions = config.hyperalignment_repetitions_per_class
    if alignment_repetitions is None and config.target_calibration_trials_per_class > 0 and config.hyperalignment_sample_mode == "class_repetition":
        alignment_repetitions = config.target_calibration_trials_per_class
    hyperalignment_model, class_alignment = fit_class_hyperalignment(
        train_features_by_subject,
        train_alignment_labels_by_subject,
        sample_mode=config.hyperalignment_sample_mode,
        n_repetitions_per_class=alignment_repetitions,
        n_components=config.hyperalignment_components,
        n_iterations=config.hyperalignment_iterations,
    )

    transformed_train = []
    transformed_labels = []
    for feature_set, alignment_feature_set in zip(train_sets, train_alignment_sets, strict=True):
        transformed_train.append(_transform_fitted_subject(hyperalignment_model, feature_set, alignment_feature_set))
        transformed_labels.append(train_decode_labels_by_subject[feature_set.participant])
    train_matrix = np.vstack(transformed_train)
    train_labels = np.concatenate(transformed_labels)
    test_labels = np.asarray(test_set.labels, dtype=int)[score_mask]
    if _alignment_data(config) == "cue":
        target_aligned = _target_alignment_matrix(
            test_alignment_set.features,
            np.asarray(test_alignment_set.labels, dtype=int),
            classes=class_alignment.classes,
            sample_mode=config.hyperalignment_sample_mode,
            n_repetitions_per_class=class_alignment.n_repetitions_per_class,
        )
        target_projection = fit_projection_to_hyperalignment(
            target_aligned,
            template=hyperalignment_model.template,
        )
        test_matrix = transform_with_alignment_projection(
            test_set.features[score_mask],
            decode_feature_set=test_set,
            projection=target_projection.projection,
            projection_feature_mean=target_projection.feature_mean,
            projection_feature_set=test_alignment_set,
        )
        target_transform = "cue_target_calibrated"
        n_target_calibration_trials = _count_labels_in_classes(test_alignment_set.labels, class_alignment.classes)
    elif config.target_calibration_trials_per_class > 0:
        target_alignment = class_alignment_matrices(
            {test_set.participant: test_alignment_set.features[calibration_mask]},
            {test_set.participant: np.asarray(test_set.labels, dtype=int)[calibration_mask]},
            sample_mode=config.hyperalignment_sample_mode,
            n_repetitions_per_class=class_alignment.n_repetitions_per_class,
        )
        target_projection = fit_projection_to_hyperalignment(
            target_alignment.aligned_by_subject[test_set.participant],
            template=hyperalignment_model.template,
        )
        test_matrix = transform_with_alignment_projection(
            test_set.features[score_mask],
            decode_feature_set=test_set,
            projection=target_projection.projection,
            projection_feature_mean=target_projection.feature_mean,
            projection_feature_set=test_alignment_set,
        )
        target_transform = "target_calibrated"
        n_target_calibration_trials = int(np.sum(calibration_mask))
    else:
        test_matrix = _transform_group_subject(
            hyperalignment_model,
            test_set,
            test_alignment_set,
            config,
            score_mask=score_mask,
        )
        target_transform = "group_average"
        n_target_calibration_trials = 0

    model_bundle = fit_window_model(
        train_matrix,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
        ),
        components_pca=config.components_pca,
        train_window=_centered_window(config.window_center, config.window_size),
    )
    predictions, confidence_scores = predict_window_model(model_bundle, test_matrix)
    class_scores, score_classes = _class_score_matrix(model_bundle, test_matrix)
    top_metrics = _topk_and_rank_metrics(test_labels, class_scores, score_classes)
    outer_row = _outer_row(
        train_sets,
        test_set,
        config,
        classifier_param,
        model_bundle,
        hyperalignment_model,
        class_alignment,
        test_labels,
        predictions,
        target_transform=target_transform,
        n_calibration_trials=n_target_calibration_trials,
        n_scored_trials=int(np.sum(score_mask)),
        top_metrics=top_metrics,
    )
    prediction_rows = _prediction_rows(
        test_set,
        config,
        test_labels,
        predictions,
        confidence_scores,
        class_scores,
        score_classes,
        score_mask=score_mask,
        target_transform=target_transform,
        hyperalignment_actual_components=hyperalignment_model.n_components,
        model_bundle=model_bundle,
    )
    return outer_row, prediction_rows


def _outer_row(  # pylint: disable=too-many-arguments
    train_sets,
    test_set,
    config,
    classifier_param,
    model_bundle,
    hyperalignment_model,
    class_alignment,
    test_labels,
    predictions,
    *,
    target_transform,
    n_calibration_trials,
    n_scored_trials,
    top_metrics,
):
    accuracy = float(accuracy_score(test_labels, predictions)) if len(test_labels) else np.nan
    balanced = float(balanced_accuracy_score(test_labels, predictions)) if len(test_labels) else np.nan
    chance = 1.0 / config.chance_classes
    window = _centered_window(config.window_center, config.window_size)
    alignment_window = resolved_alignment_window(config)
    return {
        "test_participant": int(test_set.participant),
        "train_participants": ",".join(str(int(feature_set.participant)) for feature_set in train_sets),
        "n_train_participants": len(train_sets),
        "n_train_trials": int(sum(feature_set.features.shape[0] for feature_set in train_sets)),
        "n_test_trials": int(n_scored_trials),
        "n_target_calibration_trials": int(n_calibration_trials),
        "n_scored_trials": int(n_scored_trials),
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": window[0],
        "window_stop_s": window[1],
        "alignment_window_center_s": alignment_window.center,
        "alignment_window_size_s": alignment_window.size,
        "alignment_window_start_s": alignment_window.start,
        "alignment_window_stop_s": alignment_window.stop,
        "alignment_data": _alignment_data(config),
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": _alignment_label(config),
        "hyperalignment_sample_mode": config.hyperalignment_sample_mode,
        "hyperalignment_requested_components": config.hyperalignment_components,
        "hyperalignment_actual_components": int(hyperalignment_model.n_components),
        "hyperalignment_iterations": config.hyperalignment_iterations,
        "hyperalignment_initialization": config.hyperalignment_initialization,
        "hyperalignment_repetitions_per_class": class_alignment.n_repetitions_per_class,
        "hyperalignment_alignment_classes": ",".join(str(int(value)) for value in class_alignment.classes),
        "target_transform": target_transform,
        "target_calibration_trials_per_class": config.target_calibration_trials_per_class,
        "target_centering": config.target_centering,
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        "chance_accuracy": chance,
        "chance_percent": 100.0 * chance,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced,
        "balanced_percent": 100.0 * balanced,
        "above_chance": bool(balanced > chance),
        "top2_accuracy": top_metrics["top2_accuracy"],
        "top2_percent": 100.0 * top_metrics["top2_accuracy"] if np.isfinite(top_metrics["top2_accuracy"]) else np.nan,
        "top3_accuracy": top_metrics["top3_accuracy"],
        "top3_percent": 100.0 * top_metrics["top3_accuracy"] if np.isfinite(top_metrics["top3_accuracy"]) else np.nan,
        "mean_true_label_rank": top_metrics["mean_true_label_rank"],
    }


def _prediction_rows(  # pylint: disable=too-many-arguments
    test_set,
    config,
    test_labels,
    predictions,
    confidence_scores,
    class_scores,
    score_classes,
    *,
    score_mask,
    target_transform,
    hyperalignment_actual_components,
    model_bundle,
):
    trial_indices = np.flatnonzero(score_mask)
    window = _centered_window(config.window_center, config.window_size)
    alignment_window = resolved_alignment_window(config)
    ranks = _true_label_ranks(test_labels, class_scores, score_classes)
    rows = []
    for output_index, (trial_index, true_label, predicted_label, confidence_score) in enumerate(zip(trial_indices, test_labels, predictions, confidence_scores)):
        row = {
            "test_participant": int(test_set.participant),
            "trial_index": int(trial_index),
            "true_stimulus": int(true_label),
            "predicted_stimulus": int(predicted_label),
            "correct": bool(int(true_label) == int(predicted_label)),
            "stimulus_score": float(confidence_score),
            "true_label_rank": ranks[output_index],
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": window[0],
            "window_stop_s": window[1],
            "alignment_window_center_s": alignment_window.center,
            "alignment_window_size_s": alignment_window.size,
            "alignment_window_start_s": alignment_window.start,
            "alignment_window_stop_s": alignment_window.stop,
            "alignment_data": _alignment_data(config),
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "alignment": _alignment_label(config),
            "hyperalignment_sample_mode": config.hyperalignment_sample_mode,
            "hyperalignment_actual_components": int(hyperalignment_actual_components),
            "target_transform": target_transform,
            "target_calibration_trials_per_class": config.target_calibration_trials_per_class,
            "target_centering": config.target_centering,
            "classifier": config.classifier,
            "components_pca": config.components_pca,
            "actual_components_pca": model_bundle.actual_components_pca,
            "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        }
        for class_index, class_label in enumerate(score_classes):
            row[f"score_class_{int(class_label)}"] = float(class_scores[output_index, class_index])
        rows.append(row)
    return rows


def _class_score_matrix(model_bundle, features):
    transformed = transform_window_features(model_bundle, features)
    model = model_bundle.model
    classes = np.asarray(getattr(model, "classes_", np.unique(model_bundle.train_labels)))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed), dtype=float)
    else:
        predictions = np.asarray(model.predict(transformed))
        scores = np.zeros((len(predictions), len(classes)), dtype=float)
        for row_index, predicted in enumerate(predictions):
            matches = np.flatnonzero(classes == predicted)
            if matches.size:
                scores[row_index, matches[0]] = 1.0
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    if scores.shape[1] != len(classes) and hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed), dtype=float)
    return scores, classes


def _topk_and_rank_metrics(true_labels, class_scores, score_classes):
    if class_scores.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    ranks = np.asarray(_true_label_ranks(true_labels, class_scores, score_classes), dtype=float)
    finite_ranks = ranks[np.isfinite(ranks)]
    if ranks.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    return {
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _true_label_ranks(true_labels, class_scores, score_classes):
    descending = np.argsort(-class_scores, axis=1)
    ranks = []
    for row_index, true_label in enumerate(true_labels):
        ranked_classes = score_classes[descending[row_index]]
        matches = np.flatnonzero(ranked_classes == true_label)
        ranks.append(float(matches[0] + 1) if matches.size else np.nan)
    return ranks


def _target_calibration_mask(labels, trials_per_class):
    labels = np.asarray(labels, dtype=int)
    if trials_per_class < 1:
        return np.zeros(labels.shape[0], dtype=bool)
    mask = np.zeros(labels.shape[0], dtype=bool)
    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size <= trials_per_class:
            raise ValueError(
                f"Target class {class_label} has {class_indices.size} trials, which is not enough for " f"{trials_per_class} calibration trials plus at least one scored trial."
            )
        mask[class_indices[:trials_per_class]] = True
    return mask


def _common_alignment_classes(label_arrays):
    label_sets = [set(np.asarray(labels, dtype=int).ravel().tolist()) for labels in label_arrays]
    common = sorted(set.intersection(*label_sets)) if label_sets else []
    if len(common) < 2:
        raise ValueError("Cue alignment requires at least two stimulus classes shared by source and target participants.")
    return np.asarray(common, dtype=int)


def _restrict_alignment_to_classes(features_by_subject, labels_by_subject, classes):
    classes = np.asarray(classes, dtype=int)
    filtered_features = {}
    filtered_labels = {}
    for subject_id, labels in labels_by_subject.items():
        label_array = np.asarray(labels, dtype=int).ravel()
        mask = np.isin(label_array, classes)
        filtered_features[subject_id] = np.asarray(features_by_subject[subject_id], dtype=float)[mask]
        filtered_labels[subject_id] = label_array[mask]
    return filtered_features, filtered_labels


def _count_labels_in_classes(labels, classes):
    return int(np.sum(np.isin(np.asarray(labels, dtype=int), np.asarray(classes, dtype=int))))


def _target_alignment_matrix(features, labels, *, classes, sample_mode, n_repetitions_per_class):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int).ravel()
    classes = np.asarray(classes, dtype=int).ravel()
    if features.ndim != 2:
        raise ValueError("target calibration features must be a two-dimensional matrix.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("target calibration feature and label counts must match.")
    if sample_mode == "class_mean":
        return np.vstack([np.mean(features[labels == label], axis=0) for label in classes])
    if sample_mode == "class_repetition":
        if n_repetitions_per_class is None:
            raise ValueError("class_repetition target calibration requires n_repetitions_per_class.")
        rows = []
        for label in classes:
            class_features = features[labels == label]
            if class_features.shape[0] < n_repetitions_per_class:
                raise ValueError(f"Target class {label} has {class_features.shape[0]} calibration trials, need {n_repetitions_per_class}.")
            rows.extend(class_features[:n_repetitions_per_class])
        return np.vstack(rows)
    raise ValueError(f"Unsupported hyperalignment sample mode: {sample_mode}.")


def _training_labels(feature_set, *, label_shuffle_seed=None, context=()):
    labels = np.asarray(feature_set.labels, dtype=int)
    if label_shuffle_seed is None:
        return labels
    seed_values = [int(label_shuffle_seed), *[int(value) for value in context], int(feature_set.participant)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _feature_extraction_config(config, *, window_center, window_size):
    return CrossSubjectStimulusConfig(
        window_center=window_center,
        window_size=window_size,
        baseline_window=config.baseline_window,
        feature_mode=config.feature_mode,
        normalization=config.normalization,
        alignment="none",
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        return get_default_classifier_param(config.classifier)
    return classifier_param


def _transform_fitted_subject(model, feature_set, alignment_feature_set):
    projection = model.projections[feature_set.participant]
    return transform_with_alignment_projection(
        feature_set.features,
        decode_feature_set=feature_set,
        projection=projection.projection,
        projection_feature_mean=projection.feature_mean,
        projection_feature_set=alignment_feature_set,
    )


def _transform_group_subject(model, feature_set, alignment_feature_set, config, *, score_mask):
    if model.group_projection is None or model.group_feature_mean is None:
        raise ValueError("A group hyperalignment projection is unavailable for the held-out participant.")
    target_mean = np.mean(feature_set.features, axis=0) if config.target_centering == "target_unsupervised" else None
    return transform_with_alignment_projection(
        feature_set.features[score_mask],
        decode_feature_set=feature_set,
        projection=model.group_projection,
        projection_feature_mean=model.group_feature_mean,
        projection_feature_set=alignment_feature_set,
        feature_mean=target_mean,
        feature_mean_set=feature_set if target_mean is not None else None,
    )


def _normalized_hyperalignment_config(config):
    feature_mode = str(config.feature_mode).strip().lower().replace("-", "_")
    normalization = str(config.normalization).strip().lower().replace("-", "_")
    sample_mode = str(config.hyperalignment_sample_mode).strip().lower().replace("-", "_")
    target_centering = str(config.target_centering).strip().lower().replace("-", "_")
    alignment_data = _alignment_data(config)
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {config.feature_mode}. Supported modes: {', '.join(FEATURE_MODES)}.")
    if normalization not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization: {config.normalization}. Supported modes: {', '.join(NORMALIZATION_MODES)}.")
    if sample_mode not in CLASS_ALIGNMENT_SAMPLE_MODES:
        raise ValueError(f"Unsupported hyperalignment sample mode: {config.hyperalignment_sample_mode}. Supported modes: {', '.join(CLASS_ALIGNMENT_SAMPLE_MODES)}.")
    if target_centering not in HYPERALIGNMENT_TARGET_CENTERING_MODES:
        raise ValueError(f"Unsupported target centering: {config.target_centering}. Supported modes: {', '.join(HYPERALIGNMENT_TARGET_CENTERING_MODES)}.")
    if alignment_data not in ALIGNMENT_DATASETS:
        raise ValueError(f"Unsupported alignment data: {config.alignment_data}. Supported modes: {', '.join(ALIGNMENT_DATASETS)}.")
    if config.target_calibration_trials_per_class < 0:
        raise ValueError("target_calibration_trials_per_class must be non-negative.")
    if alignment_data == "cue" and config.target_calibration_trials_per_class > 0:
        raise ValueError("alignment_data='cue' uses independent cue target calibration; target_calibration_trials_per_class must be 0.")
    if config.hyperalignment_iterations < 0:
        raise ValueError("hyperalignment_iterations must be non-negative.")
    if float(config.window_size) <= 0:
        raise ValueError("window_size must be positive.")
    if resolved_alignment_window(config).size <= 0:
        raise ValueError("alignment_window_size must be positive.")
    baseline_window = tuple(float(value) for value in config.baseline_window)
    if len(baseline_window) != 2:
        raise ValueError("baseline_window must contain exactly two values.")
    return CrossSubjectHyperalignmentConfig(
        window_center=float(config.window_center),
        window_size=float(config.window_size),
        alignment_window_center=None if config.alignment_window_center is None else float(config.alignment_window_center),
        alignment_window_size=None if config.alignment_window_size is None else float(config.alignment_window_size),
        alignment_data=alignment_data,
        baseline_window=(baseline_window[0], baseline_window[1]),
        feature_mode=feature_mode,
        normalization=normalization,
        classifier=str(config.classifier),
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        chance_classes=int(config.chance_classes),
        random_state=config.random_state,
        signflip_permutations=int(config.signflip_permutations),
        signflip_seed=config.signflip_seed,
        hyperalignment_components=config.hyperalignment_components,
        hyperalignment_iterations=int(config.hyperalignment_iterations),
        hyperalignment_initialization=str(config.hyperalignment_initialization).strip().lower().replace("-", "_"),
        hyperalignment_sample_mode=sample_mode,
        hyperalignment_repetitions_per_class=config.hyperalignment_repetitions_per_class,
        target_calibration_trials_per_class=int(config.target_calibration_trials_per_class),
        target_centering=target_centering,
    )


def _alignment_label(config):
    if _alignment_data(config) == "cue":
        return "class_hyperalignment_cue_calibrated"
    if config.target_calibration_trials_per_class > 0:
        return "class_hyperalignment_target_calibrated"
    return "class_hyperalignment_group_average"


def _alignment_data(config):
    return str(config.alignment_data).strip().lower().replace("-", "_")


def _centered_window(center, size):
    return (float(center) - float(size) / 2.0, float(center) + float(size) / 2.0)


def _finite_metric_values(rows, key):
    values = [float(row.get(key, np.nan)) for row in rows]
    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else np.nan


def _percent_nanmean_or_nan(values):
    mean = _nanmean_or_nan(values)
    return float(100.0 * mean) if np.isfinite(mean) else np.nan


def _sem(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0 if values.size == 1 else np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _sem_or_nan(values):
    values = np.asarray(values, dtype=float)
    return _sem(values) if values.size else np.nan


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str))


def _one_sided_exact_sign_p_value(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    n = int(nonzero.size)
    if n == 0:
        return np.nan
    positives = int(np.sum(nonzero > 0))
    tail = sum(comb(n, k) for k in range(positives, n + 1))
    return float(tail / (2**n))


def _one_sided_signflip_p_value(differences, *, n_permutations, seed):
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0 or n_permutations <= 0:
        return np.nan
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    null_values: list[float] = []
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size)
        null_values.append(float(np.mean(differences * signs)))
    null_array = np.asarray(null_values, dtype=float)
    return float((np.sum(null_array >= observed) + 1.0) / (null_array.size + 1.0))


def _parse_time_window(value: str) -> tuple[float, float]:
    parts = tuple(float(token.strip()) for token in value.split(",", maxsplit=1))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have the form start,stop.")
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Time window start must be before stop.")
    return parts


def _parse_optional_int_or_inf(value: str):
    normalized = value.strip().lower()
    if normalized in {"none", "auto", "null"}:
        return None
    return parse_int_or_inf(value)


def _build_cross_subject_hyperalignment_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run leave-one-subject-out stimulus decoding with Procrustes hyperalignment.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate. Defaults to all participants.")
    parser.add_argument("--window-center", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_CENTER, help="Stimulus decoding window center in seconds.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE, help="Stimulus decoding window size in seconds.")
    parser.add_argument("--alignment-window-center", type=float, default=None, help="Optional alignment/calibration window center in seconds. Defaults to --window-center.")
    parser.add_argument("--alignment-window-size", type=float, default=None, help="Optional alignment/calibration window size in seconds. Defaults to --window-size.")
    parser.add_argument("--alignment-data", choices=ALIGNMENT_DATASETS, default="main", help="Use main or cue files to fit hyperalignment projections.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="sensor_flat", help="Feature extraction mode.")
    parser.add_argument("--normalization", choices=NORMALIZATION_MODES, default=DEFAULT_CROSS_SUBJECT_NORMALIZATION, help="Subject-level normalization mode.")
    parser.add_argument("--classifier", default=DEFAULT_CROSS_SUBJECT_CLASSIFIER, help="Classifier name.")
    parser.add_argument("--classifier-param", default=None, help="Classifier parameter value, JSON, Python literal, numeric value, or nan/default.")
    parser.add_argument(
        "--components-pca", type=parse_int_or_inf, default=DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA, help="Post-hyperalignment PCA components for the classifier, or inf."
    )
    parser.add_argument("--hyperalignment-components", type=parse_int_or_inf, default=DEFAULT_HYPERALIGNMENT_COMPONENTS, help="Number of hyperalignment components, or inf.")
    parser.add_argument("--hyperalignment-iterations", type=int, default=10, help="Number of Procrustes template-refinement iterations.")
    parser.add_argument("--hyperalignment-initialization", choices=("pca", "mean"), default="pca", help="Template initialization mode.")
    parser.add_argument(
        "--hyperalignment-sample-mode",
        choices=CLASS_ALIGNMENT_SAMPLE_MODES,
        default=DEFAULT_HYPERALIGNMENT_SAMPLE_MODE,
        help="How to build aligned hyperalignment rows from stimulus labels.",
    )
    parser.add_argument("--hyperalignment-repetitions-per-class", type=int, default=None, help="Optional cap for class_repetition alignment rows per class.")
    parser.add_argument(
        "--target-calibration-trials-per-class",
        type=int,
        default=0,
        help="Labeled held-out trials per class used only to fit the target hyperalignment projection; excluded from scoring.",
    )
    parser.add_argument(
        "--target-centering",
        choices=HYPERALIGNMENT_TARGET_CENTERING_MODES,
        default=DEFAULT_HYPERALIGNMENT_TARGET_CENTERING,
        help="Centering used with the calibration-free group projection.",
    )
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Optional deterministic cap on trials per stimulus class and participant.")
    parser.add_argument("--chance-classes", type=int, default=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to classifiers.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Shuffle training labels within each participant for a null-control benchmark.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_hyperalignment_outer.csv", help="Held-out participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_hyperalignment_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_hyperalignment_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_hyperalignment_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_hyperalignment_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_hyperalignment_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_hyperalignment(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """CLI entry point for fixed-pipeline cross-subject Procrustes hyperalignment decoding."""

    parser = _build_cross_subject_hyperalignment_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    config = CrossSubjectHyperalignmentConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        alignment_window_center=args.alignment_window_center,
        alignment_window_size=args.alignment_window_size,
        alignment_data=args.alignment_data,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        classifier=args.classifier,
        classifier_param=parse_classifier_param(args.classifier_param),
        components_pca=args.components_pca,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
        hyperalignment_components=args.hyperalignment_components,
        hyperalignment_iterations=args.hyperalignment_iterations,
        hyperalignment_initialization=args.hyperalignment_initialization,
        hyperalignment_sample_mode=args.hyperalignment_sample_mode,
        hyperalignment_repetitions_per_class=args.hyperalignment_repetitions_per_class,
        target_calibration_trials_per_class=args.target_calibration_trials_per_class,
        target_centering=args.target_centering,
    )
    artifacts = export_cross_subject_hyperalignment(
        data_folder,
        participants,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        config=config,
        outer_participants=outer_participants,
        progress=lambda message: print(message, flush=True),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    print(f"Wrote {len(artifacts['outer'])} held-out participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['confusion'])} confusion rows to {args.confusion_output}")
    print(f"Wrote {len(artifacts['per_stimulus'])} per-stimulus rows to {args.per_stimulus_output}")
    print(f"Wrote {len(artifacts['confusion_pairs'])} confusion-pair rows to {args.confusion_pairs_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return stimulus_cross_subject_hyperalignment(argv, prog="pymegdec stimulus-cross-subject-hyperalignment")


if __name__ == "__main__":
    raise SystemExit(main())
