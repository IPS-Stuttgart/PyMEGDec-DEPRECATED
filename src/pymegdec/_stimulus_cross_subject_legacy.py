"""Cross-subject stimulus decoding smoke benchmarks."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.alpha_signal import get_data_field
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from reptrace.decoding.windowed import fit_window_model as fit_reptrace_window_model
from reptrace.decoding.windowed import (
    predict_window_model as predict_reptrace_window_model,
)
from reptrace.decoding.windowed import (
    transform_window_features as transform_reptrace_window_features,
)
from reptrace.metrics.confusion import (
    confusion_category_enrichment,
    confusion_category_matrix,
    confusion_counts,
    confusion_pair_summary,
    per_class_accuracy,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DEFAULT_CROSS_SUBJECT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_CROSS_SUBJECT_WINDOW_CENTER = 0.175
DEFAULT_CROSS_SUBJECT_WINDOW_SIZE = 0.1
DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW = (-0.5, 0.0)
DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES = 16
DEFAULT_CROSS_SUBJECT_FEATURE_MODE = "sensor_mean"
DEFAULT_CROSS_SUBJECT_NORMALIZATION = "subject_baseline_z"
DEFAULT_CROSS_SUBJECT_ALIGNMENT = "none"
DEFAULT_CROSS_SUBJECT_CLASSIFIER = "multiclass-svm"
DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA = 64
DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS = (0.150, 0.175, 0.200)
DEFAULT_CROSS_SUBJECT_SELECTION_METRIC = "balanced_accuracy"
CROSS_SUBJECT_SELECTION_METRIC_CHOICES = (
    "balanced_accuracy",
    "accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "mean_true_label_rank",
    "balanced_top2",
    "balanced_top2_top3",
    "balanced_top2_top3_rank",
    "balanced_top2_top3_rank_lcb",
    "balanced_rank",
)
DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE = 1
DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING = "uniform"
DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE = 0.02
DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION = "row_z_softmax"
DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY = "none"
SELECTION_ENSEMBLE_WEIGHTING_MODES = (
    "uniform",
    "inner_softmax",
    "inner_lcb_softmax",
    "inner_selection_softmax",
    "inner_selection_lcb_softmax",
)
ENSEMBLE_SCORE_NORMALIZATION_MODES = (
    "row_z_softmax",
    "rank_softmax",
    "rank_softmax_t2",
    "rank_softmax_t3",
    "rank_reciprocal",
    "rank_top2_vote",
    "rank_top3_vote",
    "rank_z_blend",
    "rank_softmax_balanced_quota",
    "rank_softmax_t2_balanced_quota",
    "rank_softmax_t3_balanced_quota",
    "rank_z_blend_balanced_quota",
    "rank_softmax_inner_balanced_balanced_quota",
    "rank_reciprocal_inner_balanced_balanced_quota",
    "rank_z_blend_inner_balanced_balanced_quota",
    "rank_softmax_inner_confusion_balanced_quota",
    "rank_reciprocal_inner_confusion_balanced_quota",
    "rank_z_blend_inner_confusion_balanced_quota",
    "rank_softmax_inner_balanced",
    "rank_softmax_t2_inner_balanced",
    "rank_softmax_t3_inner_balanced",
    "rank_reciprocal_inner_balanced",
    "rank_top2_vote_inner_balanced",
    "rank_top3_vote_inner_balanced",
    "rank_z_blend_inner_balanced",
    "rank_softmax_inner_confusion",
    "rank_softmax_t2_inner_confusion",
    "rank_softmax_t3_inner_confusion",
    "rank_reciprocal_inner_confusion",
    "rank_top2_vote_inner_confusion",
    "rank_top3_vote_inner_confusion",
    "rank_z_blend_inner_confusion",
)
INNER_BALANCED_ENSEMBLE_SCORE_NORMALIZATION_BASES = {
    "rank_softmax_inner_balanced": "rank_softmax",
    "rank_softmax_t2_inner_balanced": "rank_softmax_t2",
    "rank_softmax_t3_inner_balanced": "rank_softmax_t3",
    "rank_reciprocal_inner_balanced": "rank_reciprocal",
    "rank_top2_vote_inner_balanced": "rank_top2_vote",
    "rank_top3_vote_inner_balanced": "rank_top3_vote",
    "rank_z_blend_inner_balanced": "rank_z_blend",
}
INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES = {
    "rank_softmax_inner_confusion": "rank_softmax",
    "rank_softmax_t2_inner_confusion": "rank_softmax_t2",
    "rank_softmax_t3_inner_confusion": "rank_softmax_t3",
    "rank_reciprocal_inner_confusion": "rank_reciprocal",
    "rank_top2_vote_inner_confusion": "rank_top2_vote",
    "rank_top3_vote_inner_confusion": "rank_top3_vote",
    "rank_z_blend_inner_confusion": "rank_z_blend",
}
INNER_CONFUSION_CORRECTION_SMOOTHING = 1.0
INNER_CONFUSION_CORRECTION_IDENTITY_BLEND = 0.20
RANK_SOFTMAX_TEMPERATURES = {
    "rank_softmax": 1.0,
    "rank_softmax_t2": 2.0,
    "rank_softmax_t3": 3.0,
}
SELECTION_ENSEMBLE_DIVERSITY_MODES = (
    "none",
    "window",
    "classifier",
    "window_classifier",
    "window_feature_classifier",
    "window_feature_classifier_score_calibration",
    "window_feature_classifier_sample_weighting_score_calibration",
    "window_feature_classifier_sample_weighting_score_calibration_pca",
    "window_feature_classifier_param_sample_weighting_score_calibration_pca",
    "full_config",
)
NESTED_SCORE_ENSEMBLE_CLASSIFIER = "nested_topk_score_ensemble"
NESTED_SCORE_ENSEMBLE_NORMALIZATION = DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION
FEATURE_MODES = (
    "sensor_mean",
    "sensor_flat",
    "sensor_mean_slope",
    "sensor_mean_slope_std",
    "sensor_temporal_pyramid",
)
TEMPORAL_PYRAMID_LEVELS = (1, 2, 4)
NORMALIZATION_MODES = ("none", "subject_z", "subject_trial_z", "subject_baseline_z", "subject_baseline_whiten")
ALIGNMENT_MODES = ("none", "train_class_procrustes")
BASELINE_WHITENING_SHRINKAGE = 0.1
BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = (
    "window_center_s",
    "feature_mode",
    "normalization",
    "alignment",
    "calibration_data",
    "calibration_alignment",
    "calibration_template_policy",
    "calibration_window_center_s",
    "calibration_window_size_s",
    "calibration_feature_mode",
    "calibration_normalization",
    "target_calibration_label_shuffle_control",
    "target_calibration_label_shuffle_seed",
    "classifier",
    "components_pca",
    "max_trials_per_class_per_participant",
    "label_shuffle_control",
    "label_shuffle_seed",
)
STIMULUS_METADATA_ID_COLUMNS = ("stimulus", "stimulus_id", "true_stimulus", "label", "image_id")


@dataclass(frozen=True)
class CrossSubjectStimulusConfig:  # pylint: disable=too-many-instance-attributes
    """Parameters for the fixed-pipeline cross-subject stimulus smoke test."""

    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = DEFAULT_CROSS_SUBJECT_FEATURE_MODE
    normalization: str = DEFAULT_CROSS_SUBJECT_NORMALIZATION
    alignment: str = DEFAULT_CROSS_SUBJECT_ALIGNMENT
    classifier: str = DEFAULT_CROSS_SUBJECT_CLASSIFIER
    classifier_param: object = float("nan")
    components_pca: int | float = DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA
    max_trials_per_class_per_participant: int | None = None
    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class ParticipantFeatureSet:
    """Windowed features for one participant."""

    participant: int
    labels: np.ndarray
    features: np.ndarray
    normalization: str
    baseline_features: np.ndarray | None
    baseline_feature_mean: np.ndarray | None
    baseline_feature_std: np.ndarray | None
    baseline_whitening_matrix: np.ndarray | None
    n_channels: int
    n_window_samples: int
    n_baseline_samples: int
    max_trials_per_class_per_participant: int | None


def evaluate_cross_subject_stimulus_smoke(data_folder, participants, *, config=None, progress=None):
    """Run fixed-pipeline leave-one-subject-out stimulus decoding on ``Part*Data.mat`` files only."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for a cross-subject smoke benchmark.")

    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    feature_sets = []
    for participant in participants:
        if progress is not None:
            progress(f"LOAD participant={participant}")
        feature_sets.append(load_participant_stimulus_features(data_folder, participant, config=config))

    outer_rows = []
    prediction_rows = []
    for test_participant in participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        train_sets = [feature_set for feature_set in feature_sets if feature_set.participant != test_participant]
        test_set = next(feature_set for feature_set in feature_sets if feature_set.participant == test_participant)
        outer_row, participant_predictions = _evaluate_outer_fold(
            train_sets,
            test_set,
            config=config,
            classifier_param=classifier_param,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    group_summary_rows = summarize_cross_subject_stimulus_smoke(outer_rows, config=config)
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


def evaluate_nested_cross_subject_stimulus(
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    selection_ensemble_size=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
    progress=None,
    existing_artifacts=None,
    after_outer_fold=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO model selection and evaluate each untouched outer participant once."""

    candidate_configs = _normalized_candidate_configs(candidate_configs)
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for nested cross-subject decoding.")
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    selection_ensemble_size = _normalize_selection_ensemble_size(selection_ensemble_size)
    selection_ensemble_weighting = _normalize_selection_ensemble_weighting(selection_ensemble_weighting)
    selection_ensemble_temperature = _normalize_selection_ensemble_temperature(selection_ensemble_temperature)
    selection_ensemble_score_normalization = _normalize_ensemble_score_normalization(selection_ensemble_score_normalization)
    selection_ensemble_diversity = _normalize_selection_ensemble_diversity(selection_ensemble_diversity)
    selection_metric = _normalize_selection_metric(selection_metric)

    resumed = _existing_nested_artifact_rows(existing_artifacts)
    inner_rows = resumed["inner_validation"]
    outer_rows = resumed["outer"]
    selected_rows = resumed["selected"]
    prediction_rows = resumed["predictions"]
    completed_outer_folds = {int(row["test_participant"]) for row in outer_rows}
    missing_participants = tuple(participant for participant in outer_participants if participant not in completed_outer_folds)
    feature_cache = _load_feature_cache(data_folder, participants, candidate_configs, progress=progress) if missing_participants else {}
    inner_pair_cache: dict[tuple[int, tuple[int, int]], dict] = {}
    for test_participant in outer_participants:
        if int(test_participant) in completed_outer_folds:
            if progress is not None:
                progress(f"SKIP outer_test_participant={test_participant} resume=complete")
            continue
        outer_row, outer_inner_rows, selected_row, participant_predictions = _evaluate_nested_outer_fold(
            test_participant,
            participants,
            candidate_configs,
            feature_cache,
            inner_pair_cache,
            selection_ensemble_size=selection_ensemble_size,
            selection_ensemble_weighting=selection_ensemble_weighting,
            selection_ensemble_temperature=selection_ensemble_temperature,
            selection_ensemble_score_normalization=selection_ensemble_score_normalization,
            selection_ensemble_diversity=selection_ensemble_diversity,
            selection_metric=selection_metric,
            progress=progress,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        inner_rows.extend(outer_inner_rows)
        outer_rows.append(outer_row)
        selected_rows.append(selected_row)
        prediction_rows.extend(participant_predictions)
        if after_outer_fold is not None:
            after_outer_fold(_assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs))

    return _assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    max_trials_per_class_per_participant=None,
    chance_classes=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build a candidate grid for nested cross-subject model selection."""

    return tuple(
        CrossSubjectStimulusConfig(
            window_center=window_center,
            window_size=window_size,
            baseline_window=baseline_window,
            feature_mode=feature_mode,
            normalization=normalization,
            alignment=alignment,
            classifier=classifier,
            classifier_param=classifier_param,
            components_pca=components_pca,
            max_trials_per_class_per_participant=max_trials_per_class_per_participant,
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, classifier_param, components_pca in product(
            window_centers,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            classifier_params,
            components_pca_values,
        )
    )


def load_participant_stimulus_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract fixed-window features."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = sio.loadmat(data_path)["data"][0]
    all_labels = _trialinfo_labels(data)
    trial_indices = _selected_trial_indices(all_labels, config.max_trials_per_class_per_participant)
    labels = all_labels[trial_indices]
    features, n_window_samples = _extract_window_features(
        data,
        _centered_window(config.window_center, config.window_size),
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_features = None
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = _baseline_feature_statistics(data, config, n_window_samples, trial_indices)
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = _baseline_channel_whitening_matrix(data, config.baseline_window, trial_indices)
    normalized_features = _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix)
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=normalized_features,
        normalization=config.normalization,
        baseline_features=baseline_features,
        baseline_feature_mean=baseline_feature_mean,
        baseline_feature_std=baseline_feature_std,
        baseline_whitening_matrix=baseline_whitening_matrix,
        n_channels=int(_trial_signal(data, 0).shape[0]),
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
    )


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize held-out participant scores with a one-sided subject-level sign-flip test."""

    if not outer_rows:
        return []

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = _finite_metric_values(outer_rows, "top2_accuracy")
    top3 = _finite_metric_values(outer_rows, "top3_accuracy")
    mean_ranks = _finite_metric_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    participants_above_chance = _participants_above_chance(differences)
    participants_total = _participants_total(differences)
    exact_sign_p_value = _one_sided_exact_sign_p_value(differences)
    signflip_p_value = _one_sided_signflip_p_value(
        differences,
        n_permutations=config.signflip_permutations,
        seed=config.signflip_seed,
    )
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": _centered_window(config.window_center, config.window_size)[0],
            "window_stop_s": _centered_window(config.window_center, config.window_size)[1],
            "baseline_window_start_s": config.baseline_window[0],
            "baseline_window_stop_s": config.baseline_window[1],
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "alignment": config.alignment,
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
            "top2_percent_sem": _percent_sem_or_nan(top2),
            "top2_chance_accuracy": min(2.0 / config.chance_classes, 1.0),
            "top2_chance_percent": min(200.0 / config.chance_classes, 100.0),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "top3_percent_sem": _percent_sem_or_nan(top3),
            "top3_chance_accuracy": min(3.0 / config.chance_classes, 1.0),
            "top3_chance_percent": min(300.0 / config.chance_classes, 100.0),
            "mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "chance_mean_rank": 0.5 * (config.chance_classes + 1),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": participants_above_chance,
            "participants_total": participants_total,
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_exact_sign_p_value": exact_sign_p_value,
            "one_sided_signflip_p_value": signflip_p_value,
        }
    ]


def summarize_nested_cross_subject_stimulus(outer_rows, *, signflip_permutations=10_000, signflip_seed=0):
    """Summarize nested cross-subject held-out scores without assuming one fixed configuration."""

    if not outer_rows:
        return []

    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = _finite_metric_values(outer_rows, "top2_accuracy")
    top3 = _finite_metric_values(outer_rows, "top3_accuracy")
    mean_ranks = _finite_metric_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    participants_above_chance = _participants_above_chance(differences)
    participants_total = _participants_total(differences)
    exact_sign_p_value = _one_sided_exact_sign_p_value(differences)
    signflip_p_value = _one_sided_signflip_p_value(differences, n_permutations=signflip_permutations, seed=signflip_seed)
    selected_counts = Counter(int(row["selected_candidate_index"]) for row in outer_rows)
    classifier_counts = _row_value_counts(outer_rows, "selected_classifier", fallback_key="classifier")
    window_counts = _row_value_counts(outer_rows, "selected_window_center_s", fallback_key="window_center_s", transform=float)
    feature_mode_counts = _row_value_counts(outer_rows, "selected_feature_mode", fallback_key="feature_mode")
    normalization_counts = _row_value_counts(outer_rows, "selected_normalization", fallback_key="normalization")
    alignment_counts = _row_value_counts(outer_rows, "selected_alignment", fallback_key="alignment")
    components_pca_counts = _row_value_counts(outer_rows, "selected_components_pca", fallback_key="components_pca")
    trial_cap_counts = Counter(str(row["max_trials_per_class_per_participant"]) for row in outer_rows)
    winner_margins = _finite_metric_values(outer_rows, "selected_inner_winner_margin")
    label_shuffle_control = _single_row_value(outer_rows, "label_shuffle_control", default=False)
    label_shuffle_seed = _single_row_value(outer_rows, "label_shuffle_seed", default="")
    selection_metric = _single_row_value(outer_rows, "selection_metric", default=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC)
    outer_evaluation_mode = _single_row_value(outer_rows, "outer_evaluation_mode", default="single_best")
    selection_ensemble_size = _single_row_value(outer_rows, "selection_ensemble_size", default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE)
    selection_ensemble_diversity = _single_row_value(
        outer_rows,
        "selection_ensemble_diversity",
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    )
    selection_ensemble_score_normalization = _single_row_value(
        outer_rows,
        "selection_ensemble_score_normalization",
        default=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    )
    selection_ensemble_weighting = _single_row_value(
        outer_rows,
        "selection_ensemble_weighting",
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    )
    selection_ensemble_temperature = _single_row_value(
        outer_rows,
        "selection_ensemble_temperature",
        default=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    )
    ensemble_candidate_counts = _row_semicolon_value_counts(outer_rows, "selected_candidate_indices")
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "selection_mode": "nested_loso",
            "selection_metric": selection_metric,
            "outer_evaluation_mode": outer_evaluation_mode,
            "selection_ensemble_size": selection_ensemble_size,
            "selection_ensemble_diversity": selection_ensemble_diversity,
            "selection_ensemble_score_normalization": selection_ensemble_score_normalization,
            "selection_ensemble_weighting": selection_ensemble_weighting,
            "selection_ensemble_temperature": selection_ensemble_temperature,
            "label_shuffle_control": label_shuffle_control,
            "label_shuffle_seed": label_shuffle_seed,
            "n_candidates": int(max(int(row["n_candidates"]) for row in outer_rows)),
            "selected_candidate_counts": _format_counter(selected_counts),
            "selected_ensemble_candidate_counts": _format_counter(ensemble_candidate_counts),
            "selected_classifier_counts": _format_counter(classifier_counts),
            "selected_window_center_counts": _format_counter(window_counts),
            "selected_feature_mode_counts": _format_counter(feature_mode_counts),
            "selected_normalization_counts": _format_counter(normalization_counts),
            "selected_alignment_counts": _format_counter(alignment_counts),
            "selected_components_pca_counts": _format_counter(components_pca_counts),
            "max_trials_per_class_per_participant_counts": _format_counter(trial_cap_counts),
            "inner_winner_margin_mean": _nanmean_or_nan(winner_margins),
            "inner_winner_margin_median": _nanmedian_or_nan(winner_margins),
            "inner_winner_margin_min": _nanmin_or_nan(winner_margins),
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "top2_accuracy_mean": _nanmean_or_nan(top2),
            "top2_percent_mean": _percent_nanmean_or_nan(top2),
            "top2_percent_sem": _percent_sem_or_nan(top2),
            "top2_chance_accuracy": min(2.0 * chance, 1.0),
            "top2_chance_percent": min(200.0 * chance, 100.0),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "top3_percent_sem": _percent_sem_or_nan(top3),
            "top3_chance_accuracy": min(3.0 * chance, 1.0),
            "top3_chance_percent": min(300.0 * chance, 100.0),
            "mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "chance_mean_rank": 0.5 * ((1.0 / chance) + 1.0),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": participants_above_chance,
            "participants_total": participants_total,
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_exact_sign_p_value": exact_sign_p_value,
            "one_sided_signflip_p_value": signflip_p_value,
        }
    ]


def summarize_cross_subject_predictions(prediction_rows):
    """Return confusion-count and per-stimulus recall summaries for cross-subject predictions."""

    if not prediction_rows:
        return [], []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    group_columns = _present_group_columns(frame)
    confusion = confusion_counts(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=group_columns,
    )
    per_stimulus = per_class_accuracy(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant",
        group_columns=group_columns,
    )
    return confusion.to_dict(orient="records"), per_stimulus.to_dict(orient="records")


def summarize_cross_subject_confusion_pairs(prediction_rows, *, stimulus_metadata_rows=None):
    """Summarize off-diagonal errors as unordered, bidirectional stimulus pairs."""

    if not prediction_rows:
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    pairs = confusion_pair_summary(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=group_columns,
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        metadata_frame=_stimulus_metadata_frame(stimulus_metadata_rows),
        metadata_label_columns=STIMULUS_METADATA_ID_COLUMNS,
        label_prefix="stimulus",
    )
    return pairs.to_dict(orient="records")


def summarize_cross_subject_confusion_category_enrichment(
    prediction_rows,
    *,
    stimulus_metadata_rows,
    category_columns=None,
    n_permutations=10_000,
    seed=0,
):
    """Test whether off-diagonal errors stay within stimulus metadata categories."""

    if not prediction_rows or not stimulus_metadata_rows:
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    enrichment = confusion_category_enrichment(
        frame,
        metadata_frame=_stimulus_metadata_frame(stimulus_metadata_rows),
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        category_columns=category_columns,
        group_columns=group_columns,
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        metadata_label_columns=STIMULUS_METADATA_ID_COLUMNS,
        n_permutations=n_permutations,
        seed=seed,
    )
    return enrichment.to_dict(orient="records")


def summarize_cross_subject_confusion_category_matrix(
    prediction_rows,
    *,
    stimulus_metadata_rows,
    category_columns=None,
):
    """Summarize directional category-to-category error counts and lifts."""

    if not prediction_rows or not stimulus_metadata_rows:
        return []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    required = {"true_stimulus", "predicted_stimulus"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")

    group_columns = _present_group_columns(frame)
    matrix = confusion_category_matrix(
        frame,
        metadata_frame=_stimulus_metadata_frame(stimulus_metadata_rows),
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        category_columns=category_columns,
        group_columns=group_columns,
        participant_column="test_participant" if "test_participant" in frame.columns else None,
        metadata_label_columns=STIMULUS_METADATA_ID_COLUMNS,
    )
    return matrix.to_dict(orient="records")


def _assemble_nested_artifacts(outer_rows, inner_rows, selected_rows, prediction_rows, candidate_configs):
    group_summary_rows = summarize_nested_cross_subject_stimulus(
        outer_rows,
        signflip_permutations=candidate_configs[0].signflip_permutations,
        signflip_seed=candidate_configs[0].signflip_seed,
    )
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows)
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


def _existing_nested_artifact_rows(existing_artifacts):
    empty_artifacts: dict[str, list] = {
        "outer": [],
        "inner_validation": [],
        "selected": [],
        "predictions": [],
    }
    if existing_artifacts is None:
        return empty_artifacts
    return {key: list(existing_artifacts.get(key, [])) for key in empty_artifacts}


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


def _read_csv_rows(path):
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_nested_output_rows(
    *,
    outer_output_path,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
):
    return {
        "outer": _read_csv_rows(outer_output_path),
        "inner_validation": _read_csv_rows(inner_validation_output_path),
        "selected": _read_csv_rows(selected_output_path),
        "predictions": _read_csv_rows(predictions_output_path),
    }


def _write_nested_output_rows(
    artifacts,
    *,
    outer_output_path,
    group_summary_output_path=None,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
):
    _write_rows_if_present(artifacts["outer"], outer_output_path)
    _write_rows_if_present(artifacts["group_summary"], group_summary_output_path)
    _write_rows_if_present(artifacts["inner_validation"], inner_validation_output_path)
    _write_rows_if_present(artifacts["selected"], selected_output_path)
    _write_rows_if_present(artifacts["predictions"], predictions_output_path)
    _write_rows_if_present(artifacts["confusion"], confusion_output_path)
    _write_rows_if_present(artifacts["per_stimulus"], per_stimulus_output_path)
    _write_rows_if_present(artifacts["confusion_pairs"], confusion_pairs_output_path)


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


def _present_group_columns(frame):
    return tuple(column for column in CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS if column in frame.columns)


def _stimulus_metadata_frame(stimulus_metadata_rows):
    if not stimulus_metadata_rows:
        return None

    import pandas as pd

    return pd.DataFrame(stimulus_metadata_rows)


def export_cross_subject_stimulus_smoke(  # pylint: disable=too-many-arguments
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
    progress=None,
):
    """Run the cross-subject smoke benchmark and write compact CSV artifacts."""

    artifacts = evaluate_cross_subject_stimulus_smoke(data_folder, participants, config=config, progress=progress)
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


def export_nested_cross_subject_stimulus(  # pylint: disable=too-many-arguments
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
    resume=False,
    write_incremental=False,
    outer_participants=None,
    selection_ensemble_size=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO cross-subject decoding and write compact CSV artifacts."""

    existing_artifacts = (
        _read_nested_output_rows(
            outer_output_path=outer_output_path,
            inner_validation_output_path=inner_validation_output_path,
            selected_output_path=selected_output_path,
            predictions_output_path=predictions_output_path,
        )
        if resume
        else None
    )

    def write_outputs(current_artifacts):
        _write_nested_output_rows(
            current_artifacts,
            outer_output_path=outer_output_path,
            group_summary_output_path=group_summary_output_path,
            inner_validation_output_path=inner_validation_output_path,
            selected_output_path=selected_output_path,
            predictions_output_path=predictions_output_path,
            confusion_output_path=confusion_output_path,
            per_stimulus_output_path=per_stimulus_output_path,
            confusion_pairs_output_path=confusion_pairs_output_path,
        )

    artifacts = evaluate_nested_cross_subject_stimulus(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_participants=outer_participants,
        progress=progress,
        existing_artifacts=existing_artifacts,
        after_outer_fold=write_outputs if write_incremental else None,
        selection_ensemble_size=selection_ensemble_size,
        selection_ensemble_weighting=selection_ensemble_weighting,
        selection_ensemble_temperature=selection_ensemble_temperature,
        selection_ensemble_score_normalization=selection_ensemble_score_normalization,
        selection_ensemble_diversity=selection_ensemble_diversity,
        selection_metric=selection_metric,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    write_outputs(artifacts)
    return artifacts


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    representative_configs: dict[tuple[float, float, float, float, str, str, int | None], CrossSubjectStimulusConfig] = {}
    for candidate_config in candidate_configs:
        representative_configs.setdefault(_feature_cache_key(candidate_config), candidate_config)

    feature_cache = {}
    for key, candidate_config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD feature_set "
                f"window_center={candidate_config.window_center} "
                f"feature_mode={candidate_config.feature_mode} "
                f"normalization={candidate_config.normalization} "
                f"alignment={candidate_config.alignment}"
            )
        feature_cache[key] = {participant: load_participant_stimulus_features(data_folder, participant, config=candidate_config) for participant in participants}
    return feature_cache


def _evaluate_nested_outer_fold(
    test_participant,
    participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    selection_ensemble_size=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE,
    selection_ensemble_weighting=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    selection_ensemble_temperature=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    selection_ensemble_score_normalization=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    selection_ensemble_diversity=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    if progress is not None:
        progress(f"START outer_test_participant={test_participant}")
    outer_train_participants = tuple(participant for participant in participants if participant != test_participant)
    outer_inner_rows = _evaluate_nested_inner_rows(
        test_participant,
        outer_train_participants,
        candidate_configs,
        feature_cache,
        inner_pair_cache,
        selection_metric=selection_metric,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    selected_row, selected_candidate_rows = _select_nested_candidate_ensemble(
        outer_inner_rows,
        selection_ensemble_size=selection_ensemble_size,
        selection_ensemble_weighting=selection_ensemble_weighting,
        selection_ensemble_temperature=selection_ensemble_temperature,
        selection_ensemble_score_normalization=selection_ensemble_score_normalization,
        selection_ensemble_diversity=selection_ensemble_diversity,
        selection_metric=selection_metric,
        candidate_configs=candidate_configs,
    )
    if int(selected_row["selection_ensemble_size"]) == 1:
        selected_config = candidate_configs[int(selected_row["selected_candidate_index"]) - 1]
        selected_feature_sets = feature_cache[_feature_cache_key(selected_config)]
        train_sets = [selected_feature_sets[participant] for participant in outer_train_participants]
        test_set = selected_feature_sets[test_participant]
        outer_row, participant_predictions = _evaluate_outer_fold(
            train_sets,
            test_set,
            config=selected_config,
            classifier_param=_resolved_classifier_param(selected_config),
            include_predictions=True,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(test_participant), int(selected_row["selected_candidate_index"]), 0),
        )
    else:
        fitted_models = []
        test_sets = []
        selected_configs = []
        for ensemble_rank, candidate_row in enumerate(selected_candidate_rows):
            candidate_index = int(candidate_row["selected_candidate_index"])
            selected_config = candidate_configs[candidate_index - 1]
            selected_feature_sets = feature_cache[_feature_cache_key(selected_config)]
            train_sets = [selected_feature_sets[participant] for participant in outer_train_participants]
            fitted_models.append(
                _fit_outer_fold_model(
                    train_sets,
                    selected_config,
                    _resolved_classifier_param(selected_config),
                    label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
                    label_shuffle_context=(int(test_participant), candidate_index, ensemble_rank),
                )
            )
            test_sets.append(selected_feature_sets[test_participant])
            selected_configs.append(selected_config)
        outer_row, participant_predictions = _score_outer_fold_ensemble_models(
            fitted_models,
            test_sets,
            selected_configs,
            selected_candidate_rows,
            ensemble_weights=_nested_ensemble_weights(
                selected_candidate_rows,
                weighting=selected_row["selection_ensemble_weighting"],
                temperature=selected_row["selection_ensemble_temperature"],
            ),
            ensemble_weighting=selected_row["selection_ensemble_weighting"],
            ensemble_temperature=selected_row["selection_ensemble_temperature"],
            ensemble_score_normalization=selected_row["selection_ensemble_score_normalization"],
            include_predictions=True,
        )
    _add_selected_candidate_fields(outer_row, selected_row)
    for prediction_row in participant_predictions:
        _add_selected_candidate_fields(prediction_row, selected_row)
    if progress is not None:
        progress(
            "DONE outer_test_participant="
            f"{test_participant} selected_candidate={selected_row['selected_candidate_index']} "
            f"selection_ensemble_size={selected_row['selection_ensemble_size']} "
            f"selection_ensemble_diversity={selected_row['selection_ensemble_diversity']} "
            f"score_normalization={selected_row['selection_ensemble_score_normalization']} "
            f"selection_ensemble_weighting={selected_row['selection_ensemble_weighting']} "
            f"inner_mean={selected_row['selected_inner_balanced_accuracy_mean']:.4f} "
            f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
        )
    return outer_row, outer_inner_rows, selected_row, participant_predictions


def _evaluate_nested_inner_rows(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    inner_pair_cache,
    *,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    inner_rows = []
    completed = 0
    total = len(candidate_configs) * len(outer_train_participants)
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        feature_sets = feature_cache[_feature_cache_key(candidate_config)]
        for validation_participant in outer_train_participants:
            excluded_pair = tuple(sorted((int(test_participant), int(validation_participant))))
            pair_rows = _cached_nested_pair_rows(
                candidate_index,
                candidate_config,
                excluded_pair,
                feature_sets,
                inner_pair_cache,
                selection_metric=selection_metric,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
            )
            inner_rows.append(pair_rows[(int(test_participant), int(validation_participant))])
            completed += 1
            if progress is not None:
                progress(
                    "DONE inner_validation "
                    f"outer_test_participant={test_participant} "
                    f"candidate={candidate_index}/{len(candidate_configs)} "
                    f"validation_participant={validation_participant} "
                    f"progress={completed}/{total}"
                )
    return inner_rows


def _cached_nested_pair_rows(
    candidate_index,
    candidate_config,
    excluded_pair,
    feature_sets,
    inner_pair_cache,
    *,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    cache_key = (
        int(candidate_index),
        tuple(excluded_pair),
        _normalize_selection_metric(selection_metric),
        bool(label_shuffle_control),
        int(label_shuffle_seed),
    )
    if cache_key not in inner_pair_cache:
        train_sets = [feature_set for participant, feature_set in feature_sets.items() if int(participant) not in excluded_pair]
        fitted_model = _fit_outer_fold_model(
            train_sets,
            candidate_config,
            _resolved_classifier_param(candidate_config),
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(candidate_index), *tuple(int(participant) for participant in excluded_pair)),
        )
        first_participant, second_participant = excluded_pair
        pair_rows = {}
        for outer_test_participant, validation_participant in (
            (first_participant, second_participant),
            (second_participant, first_participant),
        ):
            inner_row, _predictions = _score_outer_fold_model(
                fitted_model,
                feature_sets[validation_participant],
                candidate_config,
                include_predictions=False,
            )
            pair_rows[(outer_test_participant, validation_participant)] = _nested_inner_row(
                inner_row,
                outer_test_participant,
                validation_participant,
                candidate_index,
                selection_metric=selection_metric,
            )
        inner_pair_cache[cache_key] = pair_rows
    return inner_pair_cache[cache_key]


def _training_labels(feature_set, *, label_shuffle_seed=None, label_shuffle_context=()):
    labels = np.asarray(feature_set.labels, dtype=int)
    if label_shuffle_seed is None:
        return labels
    seed_values = [int(label_shuffle_seed), *[int(value) for value in label_shuffle_context], int(feature_set.participant)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    if config.alignment == "none":
        return features_by_subject, _alignment_metadata(config.alignment, common_classes=(), aligned_participants=())
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    common_classes = _common_label_values(labels_by_subject)
    if len(common_classes) < 2:
        return features_by_subject, _alignment_metadata(config.alignment, common_classes=common_classes, aligned_participants=())

    class_patterns = [
        _participant_class_channel_patterns(features, labels, feature_set, common_classes)
        for feature_set, features, labels in zip(feature_sets, features_by_subject, labels_by_subject)
    ]
    transforms = _fit_channel_procrustes_transforms(class_patterns)
    aligned_features = [
        _apply_channel_procrustes_transform(features, feature_set, transform) for feature_set, features, transform in zip(feature_sets, features_by_subject, transforms)
    ]
    return aligned_features, _alignment_metadata(
        config.alignment,
        common_classes=common_classes,
        aligned_participants=(feature_set.participant for feature_set in feature_sets),
    )


def _alignment_metadata(alignment, *, common_classes, aligned_participants):
    return {
        "alignment": alignment,
        "common_classes": ",".join(str(int(label)) for label in common_classes),
        "aligned_participants": ",".join(str(int(participant)) for participant in aligned_participants),
    }


def _common_label_values(labels_by_subject):
    label_sets = [set(np.asarray(labels, dtype=int).tolist()) for labels in labels_by_subject]
    if not label_sets:
        return tuple()
    return tuple(sorted(set.intersection(*label_sets)))


def _participant_class_channel_patterns(features, labels, feature_set, common_classes):
    channel_features = _features_as_trial_channel_matrix(features, feature_set)
    labels = np.asarray(labels, dtype=int)
    patterns = []
    for label in common_classes:
        class_features = channel_features[labels == int(label)]
        if class_features.size == 0:
            raise ValueError(f"Missing class {label} while fitting Procrustes alignment.")
        patterns.append(np.mean(class_features, axis=(0, 1)))
    return np.vstack(patterns)


def _features_as_trial_channel_matrix(features, feature_set):
    features = np.asarray(features, dtype=float)
    n_channels = int(feature_set.n_channels)
    if features.shape[1] == n_channels:
        return features[:, None, :]
    if features.shape[1] % n_channels:
        raise ValueError("Feature width is incompatible with n_channels.")
    n_feature_blocks = int(features.shape[1] // n_channels)
    return features.reshape(features.shape[0], n_feature_blocks, n_channels)


def _fit_channel_procrustes_transforms(class_patterns):
    template = np.mean(np.stack(class_patterns, axis=0), axis=0)
    for _ in range(3):
        transforms = [_channel_procrustes_transform(patterns, template) for patterns in class_patterns]
        aligned_patterns = [_apply_channel_pattern_transform(patterns, transform) for patterns, transform in zip(class_patterns, transforms)]
        template = np.mean(np.stack(aligned_patterns, axis=0), axis=0)
    return [_channel_procrustes_transform(patterns, template) for patterns in class_patterns]


def _channel_procrustes_transform(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    cross_covariance = source_centered.T @ target_centered
    left, _singular_values, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_t
    return {
        "source_center": source_center,
        "target_center": target_center,
        "rotation": rotation,
    }


def _apply_channel_pattern_transform(patterns, transform):
    return (np.asarray(patterns, dtype=float) - transform["source_center"]) @ transform["rotation"] + transform["target_center"]


def _apply_channel_procrustes_transform(features, feature_set, transform):
    channel_features = _features_as_trial_channel_matrix(features, feature_set)
    aligned = (channel_features - transform["source_center"]) @ transform["rotation"] + transform["target_center"]
    if aligned.shape[1] == 1:
        return aligned[:, 0, :]
    return aligned.reshape(features.shape[0], -1)


def _feature_cache_key(config):
    return (
        float(config.window_center),
        float(config.window_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.feature_mode),
        str(config.normalization),
        config.max_trials_per_class_per_participant,
    )


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    return classifier_param


def _nested_inner_row(row, outer_test_participant, validation_participant, candidate_index, *, selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC):
    inner_row = dict(row)
    inner_row.update(
        {
            "selection_mode": "nested_loso",
            "selection_metric": _normalize_selection_metric(selection_metric),
            "outer_test_participant": int(outer_test_participant),
            "inner_fold": int(validation_participant),
            "inner_validation_participant": int(validation_participant),
            "inner_train_participants": row["train_participants"],
            "n_inner_train_participants": row["n_train_participants"],
            "candidate_index": int(candidate_index),
        }
    )
    return inner_row


def _select_nested_candidate(inner_rows, *, selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC):
    return _rank_nested_candidates(inner_rows, selection_metric=selection_metric)[0]


def _select_nested_candidate_ensemble(
    inner_rows,
    *,
    selection_ensemble_size,
    selection_ensemble_weighting,
    selection_ensemble_temperature,
    selection_ensemble_score_normalization,
    selection_ensemble_diversity,
    candidate_configs,
    selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
):
    ranked = _rank_nested_candidates(inner_rows, selection_metric=selection_metric)
    requested_size = _normalize_selection_ensemble_size(selection_ensemble_size)
    weighting = _normalize_selection_ensemble_weighting(selection_ensemble_weighting)
    temperature = _normalize_selection_ensemble_temperature(selection_ensemble_temperature)
    score_normalization = _normalize_ensemble_score_normalization(selection_ensemble_score_normalization)
    diversity = _normalize_selection_ensemble_diversity(selection_ensemble_diversity)
    selected_rows = _select_diverse_nested_rows(
        ranked,
        requested_size=requested_size,
        candidate_configs=candidate_configs,
        diversity=diversity,
    )
    weights = _nested_ensemble_weights(selected_rows, weighting=weighting, temperature=temperature)
    selected = dict(selected_rows[0])
    selected["selection_ensemble_requested_size"] = int(requested_size)
    selected["selection_ensemble_size"] = int(len(selected_rows))
    selected["selection_ensemble_diversity"] = diversity
    selected["selection_ensemble_score_normalization"] = score_normalization
    selected["selection_ensemble_weighting"] = weighting
    selected["selection_ensemble_temperature"] = float(temperature)
    selected["selected_candidate_indices"] = _format_sequence(row["selected_candidate_index"] for row in selected_rows)
    selected["selected_ensemble_inner_balanced_accuracy_means"] = _format_float_mapping(
        (row["selected_candidate_index"], row["selected_inner_balanced_accuracy_mean"]) for row in selected_rows
    )
    selected["selected_ensemble_inner_selection_score_means"] = _format_float_mapping(
        (row["selected_candidate_index"], row["selected_inner_selection_score_mean"]) for row in selected_rows
    )
    selected["selected_ensemble_weights"] = _format_float_mapping((row["selected_candidate_index"], weight) for row, weight in zip(selected_rows, weights))
    selected_configs = tuple(candidate_configs[int(row["selected_candidate_index"]) - 1] for row in selected_rows)
    selected["selected_ensemble_classifier_counts"] = _format_counter(Counter(config.classifier for config in selected_configs))
    selected["selected_ensemble_classifier_param_counts"] = _format_counter(Counter(str(_resolved_classifier_param(config)) for config in selected_configs))
    selected["selected_ensemble_window_center_counts"] = _format_counter(Counter(float(config.window_center) for config in selected_configs))
    selected["selected_ensemble_feature_mode_counts"] = _format_counter(Counter(config.feature_mode for config in selected_configs))
    selected["selected_ensemble_normalization_counts"] = _format_counter(Counter(config.normalization for config in selected_configs))
    selected["selected_ensemble_alignment_counts"] = _format_counter(Counter(config.alignment for config in selected_configs))
    selected["selected_ensemble_sample_weighting_counts"] = _format_counter(Counter(str(getattr(config, "sample_weighting", "none")) for config in selected_configs))
    selected["selected_ensemble_score_calibration_counts"] = _format_counter(Counter(str(getattr(config, "score_calibration", "none")) for config in selected_configs))
    selected["selected_ensemble_components_pca_counts"] = _format_counter(Counter(str(config.components_pca) for config in selected_configs))
    selected["selected_ensemble_diversity_keys"] = _format_sequence(
        _ensemble_diversity_key(candidate_configs[int(row["selected_candidate_index"]) - 1], diversity)
        for row in selected_rows
    )
    return selected, selected_rows


def _select_diverse_nested_rows(ranked_rows, *, requested_size, candidate_configs, diversity):
    ranked_rows = tuple(ranked_rows)
    requested_size = min(_normalize_selection_ensemble_size(requested_size), len(ranked_rows))
    diversity = _normalize_selection_ensemble_diversity(diversity)
    if diversity == "none":
        return tuple(ranked_rows[:requested_size])

    selected = []
    selected_indices = set()
    seen_keys = set()
    for row in ranked_rows:
        key = _ensemble_diversity_key(candidate_configs[int(row["selected_candidate_index"]) - 1], diversity)
        if key in seen_keys:
            continue
        selected.append(row)
        selected_indices.add(int(row["selected_candidate_index"]))
        seen_keys.add(key)
        if len(selected) == requested_size:
            return tuple(selected)

    for row in ranked_rows:
        candidate_index = int(row["selected_candidate_index"])
        if candidate_index in selected_indices:
            continue
        selected.append(row)
        if len(selected) == requested_size:
            break
    return tuple(selected)


def _ensemble_diversity_key(config, diversity):
    diversity = _normalize_selection_ensemble_diversity(diversity)
    if diversity == "none":
        return "all"
    if diversity == "window":
        return f"window={float(config.window_center):.6g}/{float(config.window_size):.6g}"
    if diversity == "classifier":
        return f"classifier={config.classifier}"
    if diversity == "window_classifier":
        return f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},classifier={config.classifier}"
    if diversity == "window_feature_classifier":
        return (
            f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
            f"feature={config.feature_mode},classifier={config.classifier}"
        )
    if diversity == "window_feature_classifier_score_calibration":
        return (
            f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
            f"feature={config.feature_mode},classifier={config.classifier},"
            f"score_calibration={getattr(config, 'score_calibration', 'none')}"
        )
    if diversity == "window_feature_classifier_sample_weighting_score_calibration":
        return (
            f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
            f"feature={config.feature_mode},classifier={config.classifier},"
            f"sample_weighting={getattr(config, 'sample_weighting', 'none')},"
            f"score_calibration={getattr(config, 'score_calibration', 'none')}"
        )
    if diversity == "window_feature_classifier_sample_weighting_score_calibration_pca":
        return (
            f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
            f"feature={config.feature_mode},classifier={config.classifier},"
            f"sample_weighting={getattr(config, 'sample_weighting', 'none')},"
            f"score_calibration={getattr(config, 'score_calibration', 'none')},"
            f"pca={config.components_pca}"
        )
    if diversity == "window_feature_classifier_param_sample_weighting_score_calibration_pca":
        return (
            f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
            f"feature={config.feature_mode},classifier={config.classifier},"
            f"param={_resolved_classifier_param(config)},"
            f"sample_weighting={getattr(config, 'sample_weighting', 'none')},"
            f"score_calibration={getattr(config, 'score_calibration', 'none')},"
            f"pca={config.components_pca}"
        )
    return (
        f"window={float(config.window_center):.6g}/{float(config.window_size):.6g},"
        f"feature={config.feature_mode},norm={config.normalization},alignment={config.alignment},"
        f"classifier={config.classifier},param={config.classifier_param},pca={config.components_pca},"
        f"trial_cap={config.max_trials_per_class_per_participant}"
    )


def _nested_ensemble_weights(
    selected_rows,
    *,
    weighting=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    temperature=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
):
    selected_rows = tuple(selected_rows)
    if not selected_rows:
        raise ValueError("At least one selected candidate row is required for ensemble weighting.")
    weighting = _normalize_selection_ensemble_weighting(weighting)
    temperature = _normalize_selection_ensemble_temperature(temperature)
    if weighting == "uniform" or len(selected_rows) == 1:
        return np.full(len(selected_rows), 1.0 / len(selected_rows), dtype=float)
    scores = _nested_ensemble_weight_scores(selected_rows, weighting=weighting)
    if not np.all(np.isfinite(scores)):
        return np.full(len(selected_rows), 1.0 / len(selected_rows), dtype=float)
    logits = (scores - np.max(scores)) / float(temperature)
    weights = np.exp(np.clip(logits, -50.0, 50.0))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.full(len(selected_rows), 1.0 / len(selected_rows), dtype=float)
    return weights / weight_sum


def _nested_ensemble_weight_scores(selected_rows, *, weighting):
    weighting = _normalize_selection_ensemble_weighting(weighting)
    if weighting == "inner_softmax":
        means = np.asarray([float(row["selected_inner_balanced_accuracy_mean"]) for row in selected_rows], dtype=float)
        return means
    if weighting == "inner_lcb_softmax":
        means = np.asarray([float(row["selected_inner_balanced_accuracy_mean"]) for row in selected_rows], dtype=float)
        sems = np.asarray([float(row.get("selected_inner_balanced_accuracy_sem", 0.0)) for row in selected_rows], dtype=float)
        sems = np.where(np.isfinite(sems), np.maximum(sems, 0.0), 0.0)
        return means - sems
    if weighting == "inner_selection_softmax":
        return np.asarray([float(row["selected_inner_selection_score_mean"]) for row in selected_rows], dtype=float)
    if weighting == "inner_selection_lcb_softmax":
        means = np.asarray([float(row["selected_inner_selection_score_mean"]) for row in selected_rows], dtype=float)
        sems = np.asarray([float(row.get("selected_inner_selection_score_sem", 0.0)) for row in selected_rows], dtype=float)
        sems = np.where(np.isfinite(sems), np.maximum(sems, 0.0), 0.0)
        return means - sems
    raise ValueError(f"Unsupported selection_ensemble_weighting: {weighting}")


def _rank_nested_candidates(inner_rows, *, selection_metric=DEFAULT_CROSS_SUBJECT_SELECTION_METRIC):
    if not inner_rows:
        raise ValueError("At least one inner-validation row is required for nested selection.")

    selection_metric = _normalize_selection_metric(selection_metric)
    summaries = []
    candidate_indices = sorted({int(row["candidate_index"]) for row in inner_rows})
    for candidate_index in candidate_indices:
        candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == candidate_index]
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in candidate_rows], dtype=float)
        raw = np.asarray([float(row["accuracy"]) for row in candidate_rows], dtype=float)
        top2 = _finite_metric_values(candidate_rows, "top2_accuracy")
        top3 = _finite_metric_values(candidate_rows, "top3_accuracy")
        mean_ranks = _finite_metric_values(candidate_rows, "mean_true_label_rank")
        selection_scores = np.asarray([_nested_row_selection_score(row, selection_metric) for row in candidate_rows], dtype=float)
        example = candidate_rows[0]
        inner_test_label_counts = _sum_formatted_counters(candidate_rows, "test_label_counts")
        inner_predicted_label_counts = _sum_formatted_counters(candidate_rows, "predicted_label_counts")
        inner_true_predicted_label_pair_counts = _sum_formatted_counters(
            candidate_rows, "true_predicted_label_pair_counts"
        )
        inner_confusion_counts = _sum_formatted_confusion_counters(
            candidate_rows, "confusion_counts"
        )
        chance_mean_rank = float(example.get("chance_mean_rank", 0.5 * (example.get("chance_classes", DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES) + 1)))
        summary = {
            "selection_mode": "nested_loso",
            "selection_metric": selection_metric,
            "outer_fold": int(example["outer_test_participant"]),
            "test_participant": int(example["outer_test_participant"]),
            "selected_candidate_index": int(candidate_index),
            "n_candidates": len(candidate_indices),
            "n_inner_folds": len(candidate_rows),
            "selected_inner_balanced_accuracy_mean": float(np.mean(balanced)),
            "selected_inner_balanced_accuracy_median": float(np.median(balanced)),
            "selected_inner_balanced_accuracy_sem": _sem(balanced),
            "selected_inner_accuracy_mean": float(np.mean(raw)),
            "selected_inner_accuracy_median": float(np.median(raw)),
            "selected_inner_accuracy_sem": _sem(raw),
            "selected_inner_top2_accuracy_mean": _nanmean_or_nan(top2),
            "selected_inner_top2_accuracy_median": _nanmedian_or_nan(top2),
            "selected_inner_top2_accuracy_sem": _sem_or_nan(top2),
            "selected_inner_top3_accuracy_mean": _nanmean_or_nan(top3),
            "selected_inner_top3_accuracy_median": _nanmedian_or_nan(top3),
            "selected_inner_top3_accuracy_sem": _sem_or_nan(top3),
            "selected_inner_mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "selected_inner_mean_true_label_rank_median": _nanmedian_or_nan(mean_ranks),
            "selected_inner_mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "selected_inner_chance_mean_rank": chance_mean_rank,
            "selected_inner_selection_score_mean": _nanmean_or_nan(selection_scores),
            "selected_inner_selection_score_median": _nanmedian_or_nan(selection_scores),
            "selected_inner_selection_score_sem": _sem_or_nan(selection_scores),
            "selected_inner_test_label_counts": _format_counter(inner_test_label_counts),
            "selected_inner_predicted_label_counts": _format_counter(inner_predicted_label_counts),
            "selected_inner_true_predicted_label_pair_counts": _format_counter(
                inner_true_predicted_label_pair_counts
            ),
            "selected_inner_confusion_counts": _format_confusion_counter(
                inner_confusion_counts
            ),
            "selected_window_center_s": example["window_center_s"],
            "selected_window_size_s": example["window_size_s"],
            "selected_window_start_s": example["window_start_s"],
            "selected_window_stop_s": example["window_stop_s"],
            "selected_feature_mode": example["feature_mode"],
            "selected_normalization": example["normalization"],
            "selected_alignment": example["alignment"],
            "selected_classifier": example["classifier"],
            "selected_classifier_param": example["classifier_param"],
            "selected_components_pca": example["components_pca"],
            "selected_max_trials_per_class_per_participant": example["max_trials_per_class_per_participant"],
            "label_shuffle_control": example.get("label_shuffle_control", False),
            "label_shuffle_seed": example.get("label_shuffle_seed", ""),
        }
        summary["selected_inner_rank_score_mean"] = _inner_rank_score(
            summary["selected_inner_mean_true_label_rank_mean"],
            chance_mean_rank=chance_mean_rank,
        )
        summary["selected_inner_selection_ranking_score"] = _nested_selection_score(summary, selection_metric)
        summaries.append(
            summary
        )
    ranked = sorted(
        summaries,
        key=_nested_selection_sort_key,
        reverse=True,
    )
    selected = ranked[0]
    selected_score = float(selected.get("selected_inner_selection_ranking_score", selected["selected_inner_selection_score_mean"]))
    if len(ranked) > 1:
        second_best_balanced_mean = float(ranked[1]["selected_inner_balanced_accuracy_mean"])
        second_best_score = float(ranked[1].get("selected_inner_selection_ranking_score", ranked[1]["selected_inner_selection_score_mean"]))
        winner_margin = selected_score - second_best_score
    else:
        second_best_balanced_mean = np.nan
        second_best_score = np.nan
        winner_margin = np.nan
    for rank, row in enumerate(ranked, start=1):
        row["selected_inner_rank"] = int(rank)
        row["selected_inner_second_best_balanced_accuracy_mean"] = second_best_balanced_mean
        row["selected_inner_second_best_selection_score_mean"] = second_best_score
        row_score = float(row.get("selected_inner_selection_ranking_score", row["selected_inner_selection_score_mean"]))
        row["selected_inner_winner_margin"] = winner_margin if rank == 1 else selected_score - row_score
        row["selection_ensemble_requested_size"] = DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE
        row["selection_ensemble_size"] = DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_SIZE
        row["selection_ensemble_diversity"] = DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_DIVERSITY
        row["selection_ensemble_score_normalization"] = DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION
        row["selection_ensemble_weighting"] = DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING
        row["selection_ensemble_temperature"] = DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE
        row["selected_candidate_indices"] = str(row["selected_candidate_index"])
        row["selected_ensemble_inner_balanced_accuracy_means"] = _format_float_mapping(
            ((row["selected_candidate_index"], row["selected_inner_balanced_accuracy_mean"]),)
        )
        row["selected_ensemble_inner_selection_score_means"] = _format_float_mapping(
            ((row["selected_candidate_index"], row["selected_inner_selection_score_mean"]),)
        )
        row["selected_ensemble_weights"] = _format_float_mapping(((row["selected_candidate_index"], 1.0),))
    return ranked


def _add_selected_candidate_fields(row, selected_row):
    for key, value in selected_row.items():
        row[key] = value


def _evaluate_outer_fold(
    train_sets,
    test_set,
    *,
    config,
    classifier_param,
    include_predictions=True,
    label_shuffle_seed=None,
    label_shuffle_context=(),
):
    fitted_model = _fit_outer_fold_model(
        train_sets,
        config,
        classifier_param,
        label_shuffle_seed=label_shuffle_seed,
        label_shuffle_context=label_shuffle_context,
    )
    return _score_outer_fold_model(fitted_model, test_set, config, include_predictions=include_predictions)


def _fit_outer_fold_model(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=()):
    train_features_by_subject = [_normalized_subject_features(feature_set, config) for feature_set in train_sets]
    train_label_arrays = [
        _training_labels(
            feature_set,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
        for feature_set in train_sets
    ]
    train_features_by_subject, alignment_metadata = _align_training_features_by_subject(
        train_sets,
        train_features_by_subject,
        train_label_arrays,
        config,
    )
    train_features = np.vstack(train_features_by_subject)
    train_labels_one_based = np.concatenate(train_label_arrays)
    train_labels = train_labels_one_based - 1

    train_window = _centered_window(config.window_center, config.window_size)
    model_bundle = fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
        ),
        components_pca=config.components_pca,
        train_window=train_window,
    )
    return {
        "classifier_param": classifier_param,
        "model_bundle": model_bundle,
        "n_train_participants": len(train_sets),
        "train_class_counts": Counter(train_labels_one_based.tolist()),
        "train_labels": train_labels,
        "train_participants": tuple(feature_set.participant for feature_set in train_sets),
        "train_window": train_window,
        "label_shuffle_control": label_shuffle_seed is not None,
        "label_shuffle_seed": "" if label_shuffle_seed is None else int(label_shuffle_seed),
        "alignment_metadata": alignment_metadata,
    }


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    model_bundle = fitted_model["model_bundle"]
    test_features = _normalized_subject_features(test_set, config)
    test_labels_one_based = test_set.labels
    test_labels = test_labels_one_based - 1
    predictions, _scores = predict_reptrace_window_model(model_bundle, test_features)
    class_scores, score_classes = _model_class_scores(model_bundle, test_features)
    rank_metrics = _ranked_label_metrics(test_labels, class_scores, score_classes)
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))
    chance_accuracy = 1.0 / config.chance_classes
    train_class_counts = fitted_model["train_class_counts"]
    test_class_counts = Counter(test_labels_one_based.tolist())
    train_participants = fitted_model["train_participants"]
    train_labels = fitted_model["train_labels"]
    train_window = fitted_model["train_window"]
    alignment_metadata = fitted_model["alignment_metadata"]
    predicted_label_counts = Counter((np.asarray(predictions, dtype=int) + 1).tolist())
    true_predicted_label_pair_counts = _true_predicted_label_pair_counts(
        test_labels_one_based, predictions
    )
    fold_confusion_counts = _confusion_counter(
        test_labels_one_based, np.asarray(predictions, dtype=int) + 1
    )

    outer_row = {
        "outer_fold": int(test_set.participant),
        "test_participant": int(test_set.participant),
        "train_participants": ",".join(str(participant) for participant in train_participants),
        "n_train_participants": fitted_model["n_train_participants"],
        "n_test_participants": 1,
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": config.alignment,
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
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "top2_chance_accuracy": min(2.0 * chance_accuracy, 1.0),
        "top2_chance_percent": min(200.0 * chance_accuracy, 100.0),
        "top3_chance_accuracy": min(3.0 * chance_accuracy, 1.0),
        "top3_chance_percent": min(300.0 * chance_accuracy, 100.0),
        "chance_mean_rank": 0.5 * (config.chance_classes + 1),
        "above_chance": bool(balanced_accuracy > chance_accuracy),
        "n_train_trials": int(train_labels.shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())),
        "min_test_trials_per_class": int(min(test_class_counts.values())),
        "test_label_counts": _format_counter(test_class_counts),
        "predicted_label_counts": _format_counter(predicted_label_counts),
        "true_predicted_label_pair_counts": _format_counter(
            true_predicted_label_pair_counts
        ),
        "confusion_counts": _format_confusion_counter(fold_confusion_counts),
        "classifier": config.classifier,
        "classifier_param": fitted_model["classifier_param"],
        "components_pca": config.components_pca,
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        "actual_components_pca": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "n_channels": test_set.n_channels,
        "n_window_samples": test_set.n_window_samples,
        "n_baseline_samples": test_set.n_baseline_samples,
        "label_shuffle_control": bool(fitted_model["label_shuffle_control"]),
        "label_shuffle_seed": fitted_model["label_shuffle_seed"],
        "alignment_common_classes": alignment_metadata["common_classes"],
        "alignment_aligned_participants": alignment_metadata["aligned_participants"],
    }
    prediction_rows = []
    if include_predictions:
        prediction_rows = _prediction_rows(
            test_set,
            test_labels,
            predictions,
            rank_metrics["true_label_ranks"],
            config=config,
            actual_components_pca=model_bundle.actual_components_pca,
        )
    return outer_row, prediction_rows


def _score_outer_fold_ensemble_models(
    fitted_models,
    test_sets,
    configs,
    selected_rows,
    *,
    ensemble_weights=None,
    ensemble_weighting=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING,
    ensemble_temperature=DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE,
    ensemble_score_normalization=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION,
    include_predictions=True,
):
    fitted_models = tuple(fitted_models)
    test_sets = tuple(test_sets)
    configs = tuple(configs)
    selected_rows = tuple(selected_rows)
    if not fitted_models:
        raise ValueError("At least one fitted model is required for nested score ensembling.")
    if not (len(fitted_models) == len(test_sets) == len(configs) == len(selected_rows)):
        raise ValueError("Ensemble fitted models, test sets, configs, and selected rows must have the same length.")

    reference_test_set, test_labels, class_order = _validate_ensemble_test_sets(test_sets, configs)
    weights = _normalized_ensemble_weights(ensemble_weights, len(fitted_models))
    ensemble_score_normalization = _normalize_ensemble_score_normalization(ensemble_score_normalization)
    base_score_normalization = _base_ensemble_score_normalization(ensemble_score_normalization)
    probability_matrices = []
    actual_components = []
    for fitted_model, test_set, config in zip(fitted_models, test_sets, configs):
        class_scores, score_classes = _candidate_model_scores(fitted_model, test_set, config)
        probabilities = _class_score_probabilities(class_scores, score_normalization=base_score_normalization)
        probability_matrices.append(_align_score_columns(probabilities, score_classes, class_order))
        actual_components.append(fitted_model["model_bundle"].actual_components_pca)

    ensemble_probabilities = np.tensordot(weights, np.stack(probability_matrices, axis=0), axes=(0, 0))
    prior_balance_metadata = _inner_class_prior_balance_metadata(
        selected_rows, class_order, weights, ensemble_score_normalization
    )
    ensemble_probabilities = _apply_inner_class_prior_balance(
        ensemble_probabilities, prior_balance_metadata
    )
    confusion_correction_metadata = _inner_confusion_correction_metadata(
        selected_rows, class_order, weights, ensemble_score_normalization
    )
    ensemble_probabilities = _apply_inner_confusion_correction(
        ensemble_probabilities, confusion_correction_metadata
    )
    balanced_quota_metadata = _balanced_quota_metadata(
        ensemble_probabilities, class_order, ensemble_score_normalization
    )
    predictions = np.asarray(balanced_quota_metadata.get("predictions", ()), dtype=int)
    if predictions.shape[0] != ensemble_probabilities.shape[0]:
        predictions = class_order[np.argmax(ensemble_probabilities, axis=1)]
    true_predicted_label_pair_counts = _true_predicted_label_pair_counts(
        test_labels + 1, predictions
    )
    fold_confusion_counts = _confusion_counter(
        reference_test_set.labels, np.asarray(predictions, dtype=int) + 1
    )
    rank_metrics = _ranked_label_metrics(test_labels, ensemble_probabilities, class_order)
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))

    outer_row, _template_predictions = _score_outer_fold_model(fitted_models[0], test_sets[0], configs[0], include_predictions=False)
    outer_row.update(
        {
            "classifier": NESTED_SCORE_ENSEMBLE_CLASSIFIER,
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
            "predicted_label_counts": _format_counter(
                Counter((np.asarray(predictions, dtype=int) + 1).tolist())
            ),
            "true_predicted_label_pair_counts": _format_counter(
                true_predicted_label_pair_counts
            ),
            "confusion_counts": _format_confusion_counter(fold_confusion_counts),
        }
    )
    _add_ensemble_output_fields(
        outer_row,
        selected_rows,
        configs,
        weights=weights,
        actual_components=actual_components,
        ensemble_weighting=ensemble_weighting,
        ensemble_temperature=ensemble_temperature,
        ensemble_score_normalization=ensemble_score_normalization,
    )
    _add_inner_class_prior_balance_fields(outer_row, prior_balance_metadata)
    _add_inner_confusion_correction_fields(outer_row, confusion_correction_metadata)
    _add_balanced_quota_fields(outer_row, balanced_quota_metadata)

    prediction_rows = []
    if include_predictions:
        prediction_rows = _prediction_rows(
            reference_test_set,
            test_labels,
            predictions,
            rank_metrics["true_label_ranks"],
            config=configs[0],
            actual_components_pca=fitted_models[0]["model_bundle"].actual_components_pca,
        )
        for row in prediction_rows:
            row["classifier"] = NESTED_SCORE_ENSEMBLE_CLASSIFIER
            _add_ensemble_output_fields(
                row,
                selected_rows,
                configs,
                weights=weights,
                actual_components=actual_components,
                ensemble_weighting=ensemble_weighting,
                ensemble_temperature=ensemble_temperature,
                ensemble_score_normalization=ensemble_score_normalization,
            )
            _add_inner_class_prior_balance_fields(row, prior_balance_metadata)
            _add_inner_confusion_correction_fields(row, confusion_correction_metadata)
            _add_balanced_quota_fields(row, balanced_quota_metadata)
    return outer_row, prediction_rows


def _candidate_model_scores(fitted_model, test_set, config):
    model_bundle = fitted_model["model_bundle"]
    test_features = _normalized_subject_features(test_set, config)
    return _model_class_scores(model_bundle, test_features)


def _validate_ensemble_test_sets(test_sets, configs):
    reference_set = test_sets[0]
    reference_labels = np.asarray(reference_set.labels, dtype=int) - 1
    reference_trials = _feature_set_trial_indices(reference_set)
    chance_classes = int(configs[0].chance_classes)
    class_order = np.arange(chance_classes, dtype=int)
    for test_set, config in zip(test_sets, configs):
        labels = np.asarray(test_set.labels, dtype=int) - 1
        trials = _feature_set_trial_indices(test_set)
        if int(test_set.participant) != int(reference_set.participant):
            raise ValueError("Nested score ensembling requires all models to score the same held-out participant.")
        if int(config.chance_classes) != chance_classes:
            raise ValueError("Nested score ensembling requires candidate configurations with the same chance_classes value.")
        if not np.array_equal(labels, reference_labels) or not np.array_equal(trials, reference_trials):
            raise ValueError("Nested score ensembling requires identical held-out trial labels and trial order across selected candidates.")
    return reference_set, reference_labels, class_order


def _row_softmax_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Nested score ensembling requires a non-empty two-dimensional class-score matrix.")
    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=float)
            continue
        sanitized = np.asarray(row, dtype=float).copy()
        sanitized[~finite] = np.min(sanitized[finite])
        centered = sanitized - np.mean(sanitized)
        scale = float(np.std(centered))
        if scale > 1e-12:
            centered = centered / scale
        logits = centered - np.max(centered)
        exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
        probabilities[row_index] = exp_logits / np.sum(exp_logits)
    return probabilities


def _class_score_probabilities(scores, *, score_normalization=DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION):
    score_normalization = _base_ensemble_score_normalization(score_normalization)
    if score_normalization == "row_z_softmax":
        return _row_softmax_probabilities(scores)
    if score_normalization in RANK_SOFTMAX_TEMPERATURES:
        return _rank_softmax_probabilities(scores, temperature=RANK_SOFTMAX_TEMPERATURES[score_normalization])
    if score_normalization == "rank_reciprocal":
        return _rank_reciprocal_probabilities(scores)
    if score_normalization == "rank_top2_vote":
        return _rank_topk_vote_probabilities(scores, top_k=2)
    if score_normalization == "rank_top3_vote":
        return _rank_topk_vote_probabilities(scores, top_k=3)
    if score_normalization == "rank_z_blend":
        return _rank_z_blend_probabilities(scores)
    raise ValueError(f"Unsupported ensemble score normalization: {score_normalization}")


def _rank_softmax_probabilities(scores, *, temperature=1.0):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Nested score ensembling requires a non-empty two-dimensional class-score matrix.")
    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=float)
            continue
        temperature = float(temperature)
        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("rank_softmax temperature must be a positive finite value.")
        rank_scores = np.where(finite, row, -np.inf)
        descending_columns = np.argsort(-rank_scores, kind="mergesort")
        ranks = np.empty(row.shape[0], dtype=float)
        ranks[descending_columns] = np.arange(row.shape[0], dtype=float)
        logits = -ranks / temperature
        logits[~finite] = -50.0
        exp_logits = np.exp(logits - np.max(logits))
        probabilities[row_index] = exp_logits / np.sum(exp_logits)
    return probabilities


def _rank_reciprocal_probabilities(scores):
    """Convert class scores to a soft rank distribution using reciprocal ranks.

    ``rank_softmax`` is intentionally top-heavy: with 16 classes, the highest
    ranked class receives roughly 63% of a candidate's mass before averaging.
    The BUSH-MEG source-only runs show strong top-2/top-3 signal, so a softer
    reciprocal-rank distribution can let several models agree on a near-top
    class instead of letting each model's argmax dominate the ensemble.
    """

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Nested score ensembling requires a non-empty two-dimensional class-score matrix.")
    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=float)
            continue
        rank_scores = np.where(finite, row, -np.inf)
        descending_columns = np.argsort(-rank_scores, kind="mergesort")
        ranks = np.empty(row.shape[0], dtype=float)
        ranks[descending_columns] = np.arange(row.shape[0], dtype=float)
        weights = np.zeros(row.shape[0], dtype=float)
        weights[finite] = 1.0 / (ranks[finite] + 1.0)
        probabilities[row_index] = weights / np.sum(weights)
    return probabilities


def _rank_topk_vote_probabilities(scores, *, top_k):
    """Convert scores to a hard top-k vote distribution."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Nested score ensembling requires a non-empty two-dimensional class-score matrix.")
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=float)
            continue
        k = min(top_k, int(np.sum(finite)))
        rank_scores = np.where(finite, row, -np.inf)
        top_columns = np.argsort(-rank_scores, kind="mergesort")[:k]
        weights = np.zeros(row.shape[0], dtype=float)
        weights[top_columns] = 1.0
        probabilities[row_index] = weights / np.sum(weights)
    return probabilities


def _rank_z_blend_probabilities(scores):
    rank_probabilities = _rank_softmax_probabilities(scores)
    row_z_probabilities = _row_softmax_probabilities(scores)
    blended = 0.5 * rank_probabilities + 0.5 * row_z_probabilities
    row_sums = np.sum(blended, axis=1, keepdims=True)
    return np.divide(
        blended,
        row_sums,
        out=np.full_like(blended, 1.0 / blended.shape[1]),
        where=row_sums > 0.0,
    )


def _align_score_columns(probabilities, score_classes, class_order):
    probabilities = np.asarray(probabilities, dtype=float)
    score_classes = np.asarray(score_classes, dtype=int).ravel()
    class_order = np.asarray(class_order, dtype=int).ravel()
    aligned = np.zeros((probabilities.shape[0], class_order.shape[0]), dtype=float)
    class_to_column = {int(class_label): column for column, class_label in enumerate(class_order.tolist())}
    for source_column, class_label in enumerate(score_classes.tolist()):
        target_column = class_to_column.get(int(class_label))
        if target_column is not None:
            aligned[:, target_column] = probabilities[:, source_column]
    row_sums = np.sum(aligned, axis=1, keepdims=True)
    valid = row_sums[:, 0] > 1e-12
    aligned[valid] = aligned[valid] / row_sums[valid]
    aligned[~valid] = 1.0 / class_order.shape[0]
    return aligned


def _without_balanced_quota_suffix(score_normalization):
    score_normalization = str(score_normalization).strip()
    if score_normalization.endswith("_balanced_quota"):
        return score_normalization.removesuffix("_balanced_quota")
    return score_normalization


def _inner_class_prior_balance_mode(score_normalization):
    inner_mode = _without_balanced_quota_suffix(score_normalization)
    return (
        inner_mode
        if inner_mode in INNER_BALANCED_ENSEMBLE_SCORE_NORMALIZATION_BASES
        else None
    )


def _inner_confusion_correction_mode(score_normalization):
    inner_mode = _without_balanced_quota_suffix(score_normalization)
    return (
        inner_mode
        if inner_mode in INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES
        else None
    )


def _base_ensemble_score_normalization(score_normalization):
    score_normalization = _normalize_ensemble_score_normalization(score_normalization)
    inner_mode = _without_balanced_quota_suffix(score_normalization)
    return INNER_BALANCED_ENSEMBLE_SCORE_NORMALIZATION_BASES.get(
        inner_mode,
        INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.get(
            inner_mode, inner_mode
        ),
    )


def _inner_class_prior_balance_metadata(selected_rows, class_order, weights, score_normalization):
    score_normalization = _normalize_ensemble_score_normalization(score_normalization)
    inner_mode = _inner_class_prior_balance_mode(score_normalization)
    if inner_mode is None:
        return {"mode": "none"}

    selected_rows = tuple(selected_rows)
    class_order = np.asarray(class_order, dtype=int).ravel()
    labels_one_based = class_order + 1
    weights = _normalized_ensemble_weights(weights, len(selected_rows))
    target_counts = np.zeros(class_order.shape[0], dtype=float)
    predicted_counts = np.zeros(class_order.shape[0], dtype=float)
    for row, weight in zip(selected_rows, weights):
        target_counts += float(weight) * _counter_vector(row.get("selected_inner_test_label_counts", ""), labels_one_based)
        predicted_counts += float(weight) * _counter_vector(row.get("selected_inner_predicted_label_counts", ""), labels_one_based)

    if not np.any(target_counts > 0.0) or not np.any(predicted_counts > 0.0):
        return {"mode": "skipped_missing_inner_counts"}

    smoothing = 1.0
    target_distribution = target_counts + smoothing
    target_distribution /= np.sum(target_distribution)
    predicted_distribution = predicted_counts + smoothing
    predicted_distribution /= np.sum(predicted_distribution)
    log_adjustment = np.log(target_distribution) - np.log(predicted_distribution)
    log_adjustment -= np.mean(log_adjustment)
    log_adjustment = np.clip(log_adjustment, -1.5, 1.5)
    return {
        "mode": score_normalization,
        "inner_mode": inner_mode,
        "smoothing": smoothing,
        "class_order": class_order,
        "target_counts": target_counts,
        "predicted_counts": predicted_counts,
        "log_adjustment": log_adjustment,
    }


def _apply_inner_class_prior_balance(probabilities, metadata):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        _inner_class_prior_balance_mode(metadata.get("mode")) is None
        or "log_adjustment" not in metadata
    ):
        return probabilities
    adjustment = np.exp(np.asarray(metadata["log_adjustment"], dtype=float))
    adjusted = probabilities * adjustment[None, :]
    row_sums = np.sum(adjusted, axis=1, keepdims=True)
    return np.divide(
        adjusted,
        row_sums,
        out=np.full_like(adjusted, 1.0 / adjusted.shape[1]),
        where=row_sums > 0.0,
    )


def _balanced_quota_metadata(probabilities, class_order, score_normalization):
    """Return optional test-batch balanced assignment metadata."""

    mode = _normalize_ensemble_score_normalization(score_normalization)
    if not mode.endswith("_balanced_quota"):
        return {"mode": "none"}
    predictions, quota_counts, status = _balanced_quota_predictions(
        probabilities, class_order
    )
    class_order = np.asarray(class_order, dtype=int).ravel()
    return {
        "mode": mode,
        "status": status,
        "class_order": class_order,
        "quota_counts": quota_counts,
        "predictions": predictions,
    }


def _balanced_quota_predictions(probabilities, class_order):
    probabilities = np.asarray(probabilities, dtype=float)
    class_order = np.asarray(class_order, dtype=int).ravel()
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or probabilities.shape[1] == 0
    ):
        return (
            np.asarray([], dtype=int),
            np.zeros(class_order.shape[0], dtype=int),
            "skipped_empty_scores",
        )
    if class_order.shape[0] != probabilities.shape[1]:
        return (
            np.asarray([], dtype=int),
            np.zeros(class_order.shape[0], dtype=int),
            "skipped_class_order_mismatch",
        )

    sanitized = _finite_assignment_scores(probabilities)
    quotas = _balanced_quota_counts(sanitized)
    expanded_columns = np.repeat(np.arange(sanitized.shape[1], dtype=int), quotas)
    if expanded_columns.shape[0] != sanitized.shape[0]:
        return class_order[np.argmax(sanitized, axis=1)], quotas, "skipped_invalid_quota"

    row_indices, slot_indices = linear_sum_assignment(-sanitized[:, expanded_columns])
    prediction_columns = np.empty(sanitized.shape[0], dtype=int)
    prediction_columns[row_indices] = expanded_columns[slot_indices]
    return class_order[prediction_columns], quotas, "applied"


def _finite_assignment_scores(probabilities):
    scores = np.asarray(probabilities, dtype=float).copy()
    if scores.ndim != 2:
        raise ValueError(
            "Balanced quota assignment requires a two-dimensional score matrix."
        )
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if np.any(finite):
            row[~finite] = float(np.min(row[finite]) - 1.0)
        else:
            row[:] = 0.0
        scores[row_index] = row
    return scores


def _balanced_quota_counts(scores):
    scores = np.asarray(scores, dtype=float)
    n_trials, n_classes = scores.shape
    quotas = np.full(n_classes, n_trials // n_classes, dtype=int)
    remainder = int(n_trials - np.sum(quotas))
    if remainder > 0:
        class_totals = np.sum(scores, axis=0)
        extra_columns = np.argsort(-class_totals, kind="mergesort")[:remainder]
        quotas[extra_columns] += 1
    return quotas


def _add_inner_class_prior_balance_fields(row, metadata):
    row["ensemble_inner_class_prior_balancing"] = metadata.get("mode", "none")
    row["ensemble_inner_class_prior_smoothing"] = metadata.get("smoothing", "")
    class_order = np.asarray(metadata.get("class_order", ()), dtype=int)
    labels_one_based = class_order + 1
    row["ensemble_inner_class_prior_target_counts"] = _format_float_mapping(zip(labels_one_based, metadata.get("target_counts", ())))
    row["ensemble_inner_class_prior_predicted_counts"] = _format_float_mapping(zip(labels_one_based, metadata.get("predicted_counts", ())))
    row["ensemble_inner_class_prior_log_adjustment"] = _format_float_mapping(zip(labels_one_based, metadata.get("log_adjustment", ())))


def _inner_confusion_correction_metadata(
    selected_rows, class_order, weights, score_normalization
):
    """Return source-inner confusion metadata for leakage-safe re-ranking."""

    score_normalization = _normalize_ensemble_score_normalization(score_normalization)
    inner_mode = _inner_confusion_correction_mode(score_normalization)
    if inner_mode is None:
        return {"mode": "none"}

    selected_rows = tuple(selected_rows)
    class_order = np.asarray(class_order, dtype=int).ravel()
    labels_one_based = class_order + 1
    weights = _normalized_ensemble_weights(weights, len(selected_rows))
    counts = np.zeros((class_order.shape[0], class_order.shape[0]), dtype=float)
    for row, weight in zip(selected_rows, weights):
        row_counts = _true_predicted_pair_count_matrix(
            row.get("selected_inner_true_predicted_label_pair_counts", ""),
            labels_one_based,
        )
        if not np.any(row_counts > 0.0):
            row_counts = _confusion_counter_matrix(
                _parse_confusion_counter(row.get("selected_inner_confusion_counts", "")),
                labels_one_based,
            )
        counts += float(weight) * row_counts

    if not np.any(counts > 0.0):
        return {
            "mode": score_normalization,
            "inner_mode": inner_mode,
            "status": "skipped_missing_inner_confusion_counts",
        }

    smoothing = float(INNER_CONFUSION_CORRECTION_SMOOTHING)
    identity_blend = float(INNER_CONFUSION_CORRECTION_IDENTITY_BLEND)
    blend = 1.0
    smoothed = counts + smoothing
    predicted_totals = np.sum(smoothed, axis=0, keepdims=True)
    true_given_predicted = np.divide(
        smoothed,
        predicted_totals,
        out=np.full_like(smoothed, 1.0 / smoothed.shape[0]),
        where=predicted_totals > 0.0,
    )
    true_given_predicted = (
        (1.0 - identity_blend) * true_given_predicted
        + identity_blend * np.eye(class_order.shape[0], dtype=float)
    )
    posterior_totals = np.sum(true_given_predicted, axis=0, keepdims=True)
    true_given_predicted = np.divide(
        true_given_predicted,
        posterior_totals,
        out=np.full_like(true_given_predicted, 1.0 / true_given_predicted.shape[0]),
        where=posterior_totals > 0.0,
    )
    return {
        "mode": score_normalization,
        "inner_mode": inner_mode,
        "status": "applied",
        "smoothing": smoothing,
        "blend": blend,
        "identity_blend": identity_blend,
        "class_order": class_order,
        "true_predicted_counts": counts,
        "true_given_predicted": true_given_predicted,
    }


def _apply_inner_confusion_correction(probabilities, metadata):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        _inner_confusion_correction_mode(metadata.get("mode")) is None
        or "true_given_predicted" not in metadata
    ):
        return probabilities
    true_given_predicted = np.asarray(metadata["true_given_predicted"], dtype=float)
    corrected = probabilities @ true_given_predicted.T
    blend = float(metadata.get("blend", 0.35))
    adjusted = (1.0 - blend) * probabilities + blend * corrected
    row_sums = np.sum(adjusted, axis=1, keepdims=True)
    return np.divide(
        adjusted,
        row_sums,
        out=np.full_like(adjusted, 1.0 / adjusted.shape[1]),
        where=row_sums > 0.0,
    )


def _add_inner_confusion_correction_fields(row, metadata):
    row["ensemble_inner_confusion_correction"] = metadata.get("mode", "none")
    row["ensemble_inner_confusion_status"] = metadata.get("status", "")
    row["ensemble_inner_confusion_smoothing"] = metadata.get("smoothing", "")
    row["ensemble_inner_confusion_blend"] = metadata.get("blend", "")
    row["ensemble_inner_confusion_identity_blend"] = metadata.get(
        "identity_blend", ""
    )
    row["ensemble_inner_confusion_true_predicted_counts"] = _format_matrix_mapping(
        metadata.get("class_order", ()),
        metadata.get("true_predicted_counts", ()),
    )
    confusion = np.asarray(metadata.get("true_predicted_counts", ()), dtype=float)
    total = float(np.sum(confusion)) if confusion.ndim == 2 and confusion.size else np.nan
    if np.isfinite(total) and total > 0.0:
        row["ensemble_inner_confusion_total"] = total
        row["ensemble_inner_confusion_diagonal_fraction"] = float(
            np.trace(confusion) / total
        )
    else:
        row["ensemble_inner_confusion_total"] = ""
        row["ensemble_inner_confusion_diagonal_fraction"] = ""


def _add_balanced_quota_fields(row, metadata):
    row["ensemble_balanced_quota"] = metadata.get("mode", "none")
    row["ensemble_balanced_quota_status"] = metadata.get("status", "")
    class_order = np.asarray(metadata.get("class_order", ()), dtype=int)
    labels_one_based = class_order + 1
    row["ensemble_balanced_quota_counts"] = _format_float_mapping(
        zip(labels_one_based, metadata.get("quota_counts", ()))
    )


def _normalized_ensemble_weights(weights, expected_size):
    expected_size = int(expected_size)
    if weights is None:
        return np.full(expected_size, 1.0 / expected_size, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.shape[0] != expected_size:
        raise ValueError("Ensemble weights must match the number of fitted models.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Ensemble weights must be finite and non-negative.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("At least one ensemble weight must be positive.")
    return weights / weight_sum


def _feature_set_trial_indices(feature_set):
    trial_indices = getattr(feature_set, "trial_indices", None)
    if trial_indices is None:
        return np.arange(np.asarray(feature_set.labels).shape[0], dtype=int)
    return np.asarray(trial_indices, dtype=int).ravel()


def _add_ensemble_output_fields(
    row,
    selected_rows,
    configs,
    *,
    weights,
    actual_components,
    ensemble_weighting,
    ensemble_temperature,
    ensemble_score_normalization,
):
    candidate_indices = tuple(int(selected_row["selected_candidate_index"]) for selected_row in selected_rows)
    row["outer_evaluation_mode"] = "topk_score_ensemble"
    row["selection_ensemble_size"] = int(len(candidate_indices))
    row["selection_ensemble_score_normalization"] = _normalize_ensemble_score_normalization(ensemble_score_normalization)
    row["selection_ensemble_weighting"] = _normalize_selection_ensemble_weighting(ensemble_weighting)
    row["selection_ensemble_temperature"] = float(_normalize_selection_ensemble_temperature(ensemble_temperature))
    row["ensemble_score_normalization"] = _normalize_ensemble_score_normalization(ensemble_score_normalization)
    row["ensemble_candidate_indices"] = _format_sequence(candidate_indices)
    row["ensemble_weights"] = _format_float_mapping(zip(candidate_indices, weights))
    row["ensemble_classifiers"] = _format_sequence(config.classifier for config in configs)
    row["ensemble_window_centers_s"] = _format_sequence(config.window_center for config in configs)
    row["ensemble_feature_modes"] = _format_sequence(config.feature_mode for config in configs)
    row["ensemble_normalizations"] = _format_sequence(config.normalization for config in configs)
    row["ensemble_alignments"] = _format_sequence(config.alignment for config in configs)
    row["ensemble_components_pca"] = _format_sequence(config.components_pca for config in configs)
    row["ensemble_actual_components_pca"] = _format_sequence(actual_components)


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    train_window = _centered_window(config.window_center, config.window_size)
    rows = []
    for trial_idx, (true_label, predicted_label, true_label_rank) in enumerate(zip(test_labels, predictions, true_label_ranks)):
        true_stimulus = int(true_label) + 1
        predicted_stimulus = int(predicted_label) + 1
        rows.append(
            {
                "outer_fold": int(test_set.participant),
                "test_participant": int(test_set.participant),
                "window_center_s": config.window_center,
                "window_start_s": train_window[0],
                "window_stop_s": train_window[1],
                "feature_mode": config.feature_mode,
                "normalization": config.normalization,
                "alignment": config.alignment,
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
                "actual_components_pca": actual_components_pca,
                "trial": int(trial_idx),
                "test_trial_index": int(trial_idx),
                "test_trial_number": int(trial_idx + 1),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
                "true_label_rank": float(true_label_rank) if np.isfinite(true_label_rank) else np.nan,
                "top2_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 2),
                "top3_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 3),
            }
        )
    return rows


def _model_class_scores(model_bundle, features):
    transformed_features = transform_reptrace_window_features(model_bundle, features)
    model = model_bundle.model
    classes = np.asarray(getattr(model, "classes_", np.arange(len(np.unique(model_bundle.train_labels)))))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed_features), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed_features), dtype=float)
    else:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)

    if scores.ndim == 1:
        if classes.size != 2:
            return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != classes.size:
        return np.full((transformed_features.shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)
    return scores, classes


def _ranked_label_metrics(true_labels, class_scores, score_classes):
    true_label_ranks = _true_label_ranks(true_labels, class_scores, score_classes)
    finite_ranks = true_label_ranks[np.isfinite(true_label_ranks)]
    if finite_ranks.size == 0:
        return {
            "true_label_ranks": true_label_ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    return {
        "true_label_ranks": true_label_ranks,
        "top2_accuracy": float(np.mean(finite_ranks <= 2)),
        "top3_accuracy": float(np.mean(finite_ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)),
        "median_true_label_rank": float(np.median(finite_ranks)),
    }


def _true_label_ranks(true_labels, class_scores, score_classes):
    true_labels = np.asarray(true_labels)
    if class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return np.full(true_labels.shape[0], np.nan, dtype=float)

    label_to_column = {label: column for column, label in enumerate(np.asarray(score_classes).tolist())}
    ranks = []
    for true_label, trial_scores in zip(true_labels, class_scores):
        true_column = label_to_column.get(int(true_label))
        if true_column is None:
            ranks.append(np.nan)
            continue
        descending_columns = np.argsort(-trial_scores, kind="mergesort")
        rank_locations = np.flatnonzero(descending_columns == true_column)
        ranks.append(float(rank_locations[0] + 1) if rank_locations.size else np.nan)
    return np.asarray(ranks, dtype=float)


def _extract_window_features(data, time_window, *, feature_mode, trial_indices=None):
    feature_mode = _normalize_feature_mode(feature_mode)
    time_vector = _time_vector(data, 0)
    mask = _time_mask(time_vector, time_window)
    features = []
    for trial_idx in _iter_trial_indices(data, trial_indices):
        signal = _trial_signal(data, trial_idx)
        window_signal = signal[:, mask]
        if feature_mode == "sensor_mean":
            feature = np.mean(window_signal, axis=1)
        elif feature_mode == "sensor_flat":
            feature = window_signal.reshape(-1, order="F")
        elif feature_mode == "sensor_mean_slope":
            feature = _sensor_mean_slope_feature(window_signal, time_vector[mask])
        elif feature_mode == "sensor_mean_slope_std":
            feature = _sensor_mean_slope_std_feature(window_signal, time_vector[mask])
        elif feature_mode == "sensor_temporal_pyramid":
            feature = _sensor_temporal_pyramid_feature(window_signal)
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    return np.vstack(features), int(np.sum(mask))


def _sensor_mean_slope_feature(window_signal, window_time):
    window_signal = np.asarray(window_signal, dtype=float)
    window_time = np.asarray(window_time, dtype=float).ravel()
    means = np.mean(window_signal, axis=1)
    slopes = _sensor_window_slopes(window_signal, window_time, means)
    return np.concatenate((means, slopes))


def _sensor_mean_slope_std_feature(window_signal, window_time):
    window_signal = np.asarray(window_signal, dtype=float)
    window_time = np.asarray(window_time, dtype=float).ravel()
    means = np.mean(window_signal, axis=1)
    slopes = _sensor_window_slopes(window_signal, window_time, means)
    stds = np.std(window_signal, axis=1)
    return np.concatenate((means, slopes, stds))


def _sensor_temporal_pyramid_feature(window_signal):
    """Return channel means over fixed 1/2/4-bin temporal pyramid blocks.

    This keeps coarse within-window timing while avoiding the raw-sample
    dimensionality and latency sensitivity of ``sensor_flat``.  The output is
    block-major: all channels for the full window, then all channels for each
    half, then all channels for each quarter.  That layout is compatible with
    the existing blockwise channel whitening and Procrustes alignment helpers.
    """

    window_signal = np.asarray(window_signal, dtype=float)
    if window_signal.ndim != 2:
        raise ValueError("window_signal must be a channels-by-time array.")
    blocks = [np.mean(window_signal[:, start:stop], axis=1) for start, stop in _temporal_pyramid_slices(window_signal.shape[1])]
    return np.concatenate(blocks)


def _temporal_pyramid_slices(n_samples, *, levels=TEMPORAL_PYRAMID_LEVELS):
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("Temporal pyramid features require at least one time sample.")

    slices = []
    for n_bins in levels:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError("Temporal pyramid levels must contain positive bin counts.")
        for bin_index in range(n_bins):
            # Use floor/ceil boundaries rather than array_split so every bin is
            # non-empty even in very short windows.  If there are fewer samples
            # than bins, adjacent bins intentionally share the nearest sample;
            # this keeps feature dimensionality fixed across decode and
            # baseline windows.
            start = int(np.floor(bin_index * n_samples / n_bins))
            stop = int(np.ceil((bin_index + 1) * n_samples / n_bins))
            start = min(max(start, 0), n_samples - 1)
            stop = min(max(stop, start + 1), n_samples)
            slices.append((start, stop))
    return tuple(slices)


def _sensor_window_slopes(window_signal, window_time, means):
    if window_signal.shape[1] < 2 or np.ptp(window_time) <= 1e-12:
        return np.zeros(window_signal.shape[0], dtype=float)
    scaled_time = (window_time - np.mean(window_time)) / np.ptp(window_time)
    denominator = float(np.sum(np.square(scaled_time)))
    return (window_signal - means[:, None]) @ scaled_time / denominator


def _baseline_feature_statistics(data, config, n_window_samples, trial_indices):
    if config.feature_mode in {"sensor_mean", "sensor_mean_slope", "sensor_mean_slope_std", "sensor_temporal_pyramid"}:
        baseline_features, n_baseline_samples = _extract_window_features(data, config.baseline_window, feature_mode=config.feature_mode, trial_indices=trial_indices)
        mean = np.mean(baseline_features, axis=0, keepdims=True)
        std = np.std(baseline_features, axis=0, keepdims=True)
        return mean, _nonzero_std(std), n_baseline_samples

    if config.feature_mode == "sensor_flat":
        channel_mean, channel_std, n_baseline_samples = _baseline_channel_statistics(data, config.baseline_window, trial_indices)
        mean = np.tile(channel_mean, int(n_window_samples))[None, :]
        std = np.tile(channel_std, int(n_window_samples))[None, :]
        return mean, _nonzero_std(std), n_baseline_samples

    raise ValueError(f"Unsupported feature_mode: {config.feature_mode}")


def _baseline_channel_statistics(data, baseline_window, trial_indices):
    time_vector = _time_vector(data, 0)
    mask = _time_mask(time_vector, baseline_window)
    n_channels = int(_trial_signal(data, 0).shape[0])
    sum_values = np.zeros(n_channels, dtype=float)
    sum_squares = np.zeros(n_channels, dtype=float)
    n_values = 0
    for trial_idx in _iter_trial_indices(data, trial_indices):
        baseline_signal = _trial_signal(data, trial_idx)[:, mask]
        sum_values += np.sum(baseline_signal, axis=1)
        sum_squares += np.sum(np.square(baseline_signal), axis=1)
        n_values += baseline_signal.shape[1]
    mean = sum_values / n_values
    variance = np.maximum(sum_squares / n_values - np.square(mean), 0.0)
    return mean, np.sqrt(variance), int(np.sum(mask))


def _baseline_channel_whitening_matrix(data, baseline_window, trial_indices):
    baseline_features, n_baseline_samples = _extract_window_features(data, baseline_window, feature_mode="sensor_mean", trial_indices=trial_indices)
    covariance = _covariance_matrix(baseline_features)
    covariance = _shrink_covariance(covariance, shrinkage=BASELINE_WHITENING_SHRINKAGE)
    return _whitening_matrix(covariance), n_baseline_samples


def _covariance_matrix(features):
    features = np.asarray(features, dtype=float)
    n_features = int(features.shape[1])
    if features.shape[0] < 2:
        return np.eye(n_features, dtype=float)
    covariance = np.cov(features, rowvar=False)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return 0.5 * (covariance + covariance.T)


def _shrink_covariance(covariance, *, shrinkage):
    covariance = np.asarray(covariance, dtype=float)
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal


def _whitening_matrix(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_floor = max(float(np.max(eigenvalues)) * BASELINE_WHITENING_EIGENVALUE_FLOOR, 1e-12)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, eigen_floor))
    whitening = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    return 0.5 * (whitening + whitening.T)


def _selected_trial_indices(labels, max_trials_per_class):
    labels = np.asarray(labels).ravel()
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    max_trials_per_class = int(max_trials_per_class)
    if max_trials_per_class <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")

    selected = []
    counts: Counter[int] = Counter()
    for index, label in enumerate(labels):
        if counts[int(label)] < max_trials_per_class:
            selected.append(index)
            counts[int(label)] += 1
    return np.asarray(selected, dtype=int)


def _iter_trial_indices(data, trial_indices):
    if trial_indices is None:
        return range(_count_trials(data))
    return (int(index) for index in np.asarray(trial_indices, dtype=int).ravel())


def _trialinfo_labels(data):
    trialinfo = _unwrap_singleton(get_data_field(data, "trialinfo"))
    return np.asarray(trialinfo, dtype=int).ravel()


def _count_trials(data):
    trial_field = _unwrap_outer_cell(get_data_field(data, "trial"))
    values = np.asarray(trial_field, dtype=object)
    if values.ndim == 2 and values.shape[0] == 1:
        return int(values.shape[1])
    if values.ndim == 2 and values.shape[1] == 1:
        return int(values.shape[0])
    return int(values.size)


def _time_vector(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "time"), trial_idx), dtype=float).ravel()


def _trial_signal(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "trial"), trial_idx), dtype=float)


def _cell_item(cell, index):
    values = np.asarray(_unwrap_outer_cell(cell), dtype=object)
    if values.ndim == 0:
        return _unwrap_singleton(values.item())
    if values.ndim == 2 and values.shape[0] == 1:
        return _unwrap_singleton(values[0, index])
    if values.ndim == 2 and values.shape[1] == 1:
        return _unwrap_singleton(values[index, 0])
    return _unwrap_singleton(values[index])


def _unwrap_outer_cell(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _unwrap_singleton(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _time_mask(time_vector, time_window):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    tolerance = 1e-12
    mask = (time_vector >= start - tolerance) & (time_vector <= stop + tolerance)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _normalized_subject_features(feature_set, config):
    if feature_set.normalization == config.normalization:
        return feature_set.features
    if config.normalization == "none":
        return feature_set.features
    if config.normalization == "subject_z":
        reference = feature_set.features
        mean = np.mean(reference, axis=0, keepdims=True)
        std = np.std(reference, axis=0, keepdims=True)
        return (feature_set.features - mean) / _nonzero_std(std)
    if config.normalization == "subject_trial_z":
        return _trial_zscore_features(feature_set.features)
    if config.normalization == "subject_baseline_z":
        if feature_set.baseline_feature_mean is None or feature_set.baseline_feature_std is None:
            raise ValueError("subject_baseline_z requires baseline feature statistics.")
        mean = feature_set.baseline_feature_mean
        std = feature_set.baseline_feature_std
        return (feature_set.features - mean) / std
    if config.normalization == "subject_baseline_whiten":
        if feature_set.baseline_feature_mean is None or feature_set.baseline_whitening_matrix is None:
            raise ValueError("subject_baseline_whiten requires baseline feature statistics and a whitening matrix.")
        return _baseline_whiten_features(feature_set.features, config, feature_set.baseline_feature_mean, feature_set.baseline_whitening_matrix)
    raise ValueError(f"Unsupported normalization: {config.normalization}")


def _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix):
    features = np.asarray(features, dtype=float)
    if config.normalization == "none":
        return features
    if config.normalization == "subject_z":
        mean = np.mean(features, axis=0, keepdims=True)
        std = _nonzero_std(np.std(features, axis=0, keepdims=True))
        features -= mean
        features /= std
        return features
    if config.normalization == "subject_trial_z":
        return _trial_zscore_features(features)
    if config.normalization == "subject_baseline_z":
        if baseline_feature_mean is None or baseline_feature_std is None:
            raise ValueError("subject_baseline_z requires baseline feature statistics.")
        features -= baseline_feature_mean
        features /= baseline_feature_std
        return features
    if config.normalization == "subject_baseline_whiten":
        if baseline_feature_mean is None or baseline_whitening_matrix is None:
            raise ValueError("subject_baseline_whiten requires baseline feature statistics and a whitening matrix.")
        return _baseline_whiten_features(features, config, baseline_feature_mean, baseline_whitening_matrix)
    raise ValueError(f"Unsupported normalization: {config.normalization}")


def _trial_zscore_features(features):
    features = np.asarray(features, dtype=float)
    mean = np.mean(features, axis=1, keepdims=True)
    std = _nonzero_std(np.std(features, axis=1, keepdims=True))
    return (features - mean) / std


def _baseline_whiten_features(features, config, baseline_feature_mean, baseline_whitening_matrix):
    centered = np.asarray(features, dtype=float) - baseline_feature_mean
    whitening_matrix = np.asarray(baseline_whitening_matrix, dtype=float)
    if config.feature_mode == "sensor_mean":
        return centered @ whitening_matrix.T
    if config.feature_mode in {"sensor_flat", "sensor_mean_slope", "sensor_mean_slope_std", "sensor_temporal_pyramid"}:
        return _baseline_whiten_sensor_flat_features(centered, whitening_matrix)
    raise ValueError(f"Unsupported feature_mode: {config.feature_mode}")


def _baseline_whiten_sensor_flat_features(features, whitening_matrix):
    n_channels = int(whitening_matrix.shape[0])
    if features.shape[1] % n_channels:
        raise ValueError("Feature width must be a multiple of the number of whitening channels.")
    n_window_samples = int(features.shape[1] // n_channels)
    matrices = features.reshape(features.shape[0], n_window_samples, n_channels)
    whitened = matrices @ whitening_matrix.T
    return whitened.reshape(features.shape[0], -1)


def _nonzero_std(std):
    return np.where(std < 1e-12, 1.0, std)


def _centered_window(center, size):
    return float(np.round(center - size / 2, 10)), float(np.round(center + size / 2, 10))


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _finite_metric_values(rows, key):
    values = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _nanmedian_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def _nanmin_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.min(values))


def _sem_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return _sem(values)


def _percent_nanmean_or_nan(values):
    value = _nanmean_or_nan(values)
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _percent_sem_or_nan(values):
    value = _sem_or_nan(values)
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _confusion_counter(true_labels_one_based, predicted_labels_one_based):
    true_labels = np.asarray(true_labels_one_based, dtype=int).ravel()
    predicted_labels = np.asarray(predicted_labels_one_based, dtype=int).ravel()
    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError(
            "true and predicted labels must have the same length for confusion counting."
        )
    counter: Counter[tuple[int, int]] = Counter()
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        counter[(int(true_label), int(predicted_label))] += 1
    return counter


def _format_confusion_counter(counter):
    return ";".join(
        f"{int(true_label)}>{int(predicted_label)}:{_format_compact_count(count)}"
        for (true_label, predicted_label), count in sorted(counter.items())
    )


def _format_compact_count(value):
    value = float(value)
    rounded = int(round(value))
    if abs(value - rounded) <= 1e-9:
        return str(rounded)
    return f"{value:.6g}"


def _parse_confusion_counter(value):
    counter: dict[tuple[int, int], float] = {}
    if value in (None, ""):
        return counter
    for token in str(value).split(";"):
        token = token.strip()
        if not token or ":" not in token or ">" not in token:
            continue
        pair, count = token.rsplit(":", 1)
        true_label, predicted_label = pair.split(">", 1)
        try:
            key = (int(true_label), int(predicted_label))
            counter[key] = counter.get(key, 0.0) + float(count)
        except ValueError:
            continue
    return counter


def _sum_formatted_confusion_counters(rows, key):
    counter: dict[tuple[int, int], float] = {}
    for row in rows:
        for pair, count in _parse_confusion_counter(row.get(key, "")).items():
            counter[pair] = counter.get(pair, 0.0) + count
    return counter


def _confusion_counter_matrix(counter, labels_one_based):
    labels_one_based = np.asarray(labels_one_based, dtype=int).ravel()
    label_to_index = {
        int(label): index for index, label in enumerate(labels_one_based.tolist())
    }
    matrix = np.zeros(
        (labels_one_based.shape[0], labels_one_based.shape[0]), dtype=float
    )
    for (true_label, predicted_label), count in counter.items():
        true_index = label_to_index.get(int(true_label))
        predicted_index = label_to_index.get(int(predicted_label))
        if true_index is not None and predicted_index is not None:
            matrix[true_index, predicted_index] += float(count)
    return matrix


def _parse_formatted_counter(value):
    counter: Counter[int] = Counter()
    if value in (None, ""):
        return counter
    for token in str(value).split(";"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, count = token.rsplit(":", 1)
        try:
            counter[int(key)] += int(round(float(count)))
        except ValueError:
            continue
    return counter


def _sum_formatted_counters(rows, key):
    counter: Counter[int] = Counter()
    for row in rows:
        counter.update(_parse_formatted_counter(row.get(key, "")))
    return counter


def _counter_vector(value, labels):
    counter = _parse_formatted_counter(value)
    return np.asarray([float(counter.get(int(label), 0.0)) for label in labels], dtype=float)


def _true_predicted_label_pair_counts(true_labels_one_based, predictions_zero_based):
    return Counter(
        _true_predicted_pair_key(int(true_label), int(predicted_label) + 1)
        for true_label, predicted_label in zip(
            true_labels_one_based, predictions_zero_based
        )
    )


def _true_predicted_pair_key(true_label_one_based, predicted_label_one_based):
    return 1000 * int(true_label_one_based) + int(predicted_label_one_based)


def _true_predicted_pair_count_matrix(value, labels_one_based):
    labels_one_based = np.asarray(labels_one_based, dtype=int).ravel()
    label_to_index = {
        int(label): index for index, label in enumerate(labels_one_based.tolist())
    }
    matrix = np.zeros(
        (labels_one_based.shape[0], labels_one_based.shape[0]), dtype=float
    )
    for key, count in _parse_formatted_counter(value).items():
        true_label = int(key) // 1000
        predicted_label = int(key) % 1000
        true_index = label_to_index.get(true_label)
        predicted_index = label_to_index.get(predicted_label)
        if true_index is not None and predicted_index is not None:
            matrix[true_index, predicted_index] += float(count)
    return matrix


def _format_sequence(values):
    return ";".join(str(value) for value in values)


def _format_float_mapping(items):
    return ";".join(f"{key}:{float(value):.6g}" for key, value in items)


def _format_matrix_mapping(class_order, matrix):
    class_order = np.asarray(class_order, dtype=int).ravel()
    matrix = np.asarray(matrix, dtype=float)
    if class_order.size == 0 or matrix.shape != (class_order.size, class_order.size):
        return ""
    labels_one_based = class_order + 1
    return _format_float_mapping(
        (
            _true_predicted_pair_key(true_label, predicted_label),
            matrix[true_index, predicted_index],
        )
        for true_index, true_label in enumerate(labels_one_based)
        for predicted_index, predicted_label in enumerate(labels_one_based)
        if matrix[true_index, predicted_index] > 0.0
    )


def _row_semicolon_value_counts(rows, key):
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        for token in str(value).split(";"):
            token = token.strip()
            if token:
                counter[token] += 1
    return counter


def _row_value_counts(rows, key, *, fallback_key=None, transform=str):
    values = []
    for row in rows:
        value = row.get(key)
        if (value is None or value == "") and fallback_key is not None:
            value = row.get(fallback_key)
        if value is None or value == "":
            continue
        try:
            values.append(transform(value))
        except (TypeError, ValueError):
            continue
    return Counter(values)


def _single_row_value(rows, key, *, default=""):
    values = []
    for row in rows:
        value = row.get(key, default)
        if value in (None, ""):
            continue
        if value not in values:
            values.append(value)
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return ";".join(str(value) for value in values)


def _participants_total(differences):
    differences = np.asarray(differences, dtype=float)
    return int(np.sum(np.isfinite(differences)))


def _participants_above_chance(differences):
    differences = np.asarray(differences, dtype=float)
    finite = differences[np.isfinite(differences)]
    return int(np.sum(finite > 0.0))


def _one_sided_exact_sign_p_value(differences):
    differences = np.asarray(differences, dtype=float)
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return np.nan
    participants_above = int(np.sum(finite > 0.0))
    participants_total = int(finite.size)
    tail_count = sum(comb(participants_total, k) for k in range(participants_above, participants_total + 1))
    return float(tail_count / (2**participants_total))


def _one_sided_signflip_p_value(differences, *, n_permutations, seed):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan
    observed = float(np.mean(differences))
    if observed <= 0:
        return 1.0
    if differences.size <= 16:
        exact_signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * differences.size)).T.reshape(-1, differences.size)
        null_means = exact_signs @ differences / differences.size
        return float(np.mean(null_means >= observed))
    rng = np.random.default_rng(seed)
    random_signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), differences.size))
    null_means = random_signs @ differences / differences.size
    return float((np.sum(null_means >= observed) + 1) / (int(n_permutations) + 1))


def _normalized_config(config):
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=_normalize_feature_mode(config.feature_mode),
        normalization=_normalize_normalization(config.normalization),
        alignment=_normalize_alignment(config.alignment),
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=_normalize_trial_cap(config.max_trials_per_class_per_participant),
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalized_candidate_configs(candidate_configs):
    normalized_configs = tuple(_normalized_config(config) for config in candidate_configs)
    if not normalized_configs:
        raise ValueError("At least one candidate configuration is required.")
    chance_classes = {config.chance_classes for config in normalized_configs}
    if len(chance_classes) != 1:
        raise ValueError("All nested candidate configurations must use the same chance_classes value.")
    return normalized_configs


def _normalize_selection_metric(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in CROSS_SUBJECT_SELECTION_METRIC_CHOICES:
        raise ValueError(f"selection_metric must be one of {CROSS_SUBJECT_SELECTION_METRIC_CHOICES}.")
    return normalized


def _inner_rank_score(mean_rank, *, chance_mean_rank):
    mean_rank = float(mean_rank)
    chance_mean_rank = float(chance_mean_rank)
    if not np.isfinite(mean_rank) or not np.isfinite(chance_mean_rank) or chance_mean_rank <= 1.0:
        return np.nan
    return float(np.clip((chance_mean_rank - mean_rank) / (chance_mean_rank - 1.0), -1.0, 1.0))


def _row_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _nested_row_selection_score(row, selection_metric):
    selection_metric = _normalize_selection_metric(selection_metric)
    if selection_metric == "balanced_top2_top3_rank_lcb":
        selection_metric = "balanced_top2_top3_rank"
    if selection_metric == "balanced_accuracy":
        return _row_float(row, "balanced_accuracy")
    if selection_metric == "accuracy":
        return _row_float(row, "accuracy")
    if selection_metric == "top2_accuracy":
        return _row_float(row, "top2_accuracy")
    if selection_metric == "top3_accuracy":
        return _row_float(row, "top3_accuracy")
    if selection_metric == "mean_true_label_rank":
        return -_row_float(row, "mean_true_label_rank")
    if selection_metric == "balanced_top2":
        return 0.5 * _row_float(row, "balanced_accuracy") + 0.5 * _row_float(row, "top2_accuracy")
    if selection_metric == "balanced_top2_top3":
        return (
            _row_float(row, "balanced_accuracy")
            + _row_float(row, "top2_accuracy")
            + _row_float(row, "top3_accuracy")
        ) / 3.0
    if selection_metric == "balanced_top2_top3_rank":
        chance_mean_rank = float(row.get("chance_mean_rank", 0.5 * (row.get("chance_classes", DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES) + 1)))
        rank_score = _inner_rank_score(_row_float(row, "mean_true_label_rank"), chance_mean_rank=chance_mean_rank)
        return (
            _row_float(row, "balanced_accuracy")
            + _row_float(row, "top2_accuracy")
            + _row_float(row, "top3_accuracy")
            + rank_score
        ) / 4.0
    if selection_metric == "balanced_rank":
        chance_mean_rank = float(row.get("chance_mean_rank", 0.5 * (row.get("chance_classes", DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES) + 1)))
        rank_score = _inner_rank_score(_row_float(row, "mean_true_label_rank"), chance_mean_rank=chance_mean_rank)
        return 0.5 * _row_float(row, "balanced_accuracy") + 0.5 * rank_score
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def _nested_selection_score(summary, selection_metric):
    selection_metric = _normalize_selection_metric(selection_metric)
    if selection_metric == "balanced_top2_top3_rank_lcb":
        mean = float(summary["selected_inner_selection_score_mean"])
        sem = float(summary.get("selected_inner_selection_score_sem", 0.0))
        return mean - (sem if np.isfinite(sem) else 0.0)
    if selection_metric == "balanced_accuracy":
        return float(summary["selected_inner_balanced_accuracy_mean"])
    if selection_metric == "accuracy":
        return float(summary["selected_inner_accuracy_mean"])
    if selection_metric == "top2_accuracy":
        return float(summary["selected_inner_top2_accuracy_mean"])
    if selection_metric == "top3_accuracy":
        return float(summary["selected_inner_top3_accuracy_mean"])
    if selection_metric == "mean_true_label_rank":
        return -float(summary["selected_inner_mean_true_label_rank_mean"])
    if selection_metric == "balanced_top2":
        balanced = float(summary["selected_inner_balanced_accuracy_mean"])
        top2 = float(summary["selected_inner_top2_accuracy_mean"])
        return 0.5 * balanced + 0.5 * top2
    if selection_metric == "balanced_top2_top3":
        balanced = float(summary["selected_inner_balanced_accuracy_mean"])
        top2 = float(summary["selected_inner_top2_accuracy_mean"])
        top3 = float(summary["selected_inner_top3_accuracy_mean"])
        return (balanced + top2 + top3) / 3.0
    if selection_metric == "balanced_top2_top3_rank":
        balanced = float(summary["selected_inner_balanced_accuracy_mean"])
        top2 = float(summary["selected_inner_top2_accuracy_mean"])
        top3 = float(summary["selected_inner_top3_accuracy_mean"])
        rank_score = float(summary["selected_inner_rank_score_mean"])
        return (balanced + top2 + top3 + rank_score) / 4.0
    if selection_metric == "balanced_rank":
        balanced = float(summary["selected_inner_balanced_accuracy_mean"])
        rank_score = float(summary["selected_inner_rank_score_mean"])
        return 0.5 * balanced + 0.5 * rank_score
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def _finite_sort_value(value, *, missing=-np.inf):
    value = float(value)
    return value if np.isfinite(value) else float(missing)


def _nested_selection_sort_key(row):
    return (
        _finite_sort_value(row.get("selected_inner_selection_ranking_score", row["selected_inner_selection_score_mean"])),
        _finite_sort_value(row["selected_inner_balanced_accuracy_mean"]),
        _finite_sort_value(row["selected_inner_top2_accuracy_mean"]),
        _finite_sort_value(row["selected_inner_top3_accuracy_mean"]),
        _finite_sort_value(-float(row["selected_inner_mean_true_label_rank_mean"])),
        _finite_sort_value(row["selected_inner_accuracy_mean"]),
        -int(row["selected_candidate_index"]),
    )


def _normalize_selection_ensemble_size(value):
    value = int(value)
    if value <= 0:
        raise ValueError("selection_ensemble_size must be positive.")
    return value


def _normalize_selection_ensemble_diversity(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in SELECTION_ENSEMBLE_DIVERSITY_MODES:
        raise ValueError(f"selection_ensemble_diversity must be one of {SELECTION_ENSEMBLE_DIVERSITY_MODES}.")
    return normalized


def _normalize_ensemble_score_normalization(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in ENSEMBLE_SCORE_NORMALIZATION_MODES:
        raise ValueError(f"selection_ensemble_score_normalization must be one of {ENSEMBLE_SCORE_NORMALIZATION_MODES}.")
    return normalized


def _normalize_selection_ensemble_weighting(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in SELECTION_ENSEMBLE_WEIGHTING_MODES:
        raise ValueError(f"selection_ensemble_weighting must be one of {SELECTION_ENSEMBLE_WEIGHTING_MODES}.")
    return normalized


def _normalize_selection_ensemble_temperature(value):
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("selection_ensemble_temperature must be a positive finite value.")
    return value


def _normalize_trial_cap(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    return value


def _normalize_feature_mode(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}.")
    return normalized


def _normalize_normalization(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}.")
    return normalized


def _normalize_alignment(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in ALIGNMENT_MODES:
        raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}.")
    return normalized
