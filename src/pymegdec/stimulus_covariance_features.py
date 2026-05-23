"""Covariance-feature BUSH-MEG stimulus decoding.

This module adds a source-only LOSO benchmark for trial covariance features.  It
uses only ``Part*Data.mat`` main-task files; cue/localizer files remain reserved
for calibration-only workflows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import scipy.io as sio

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec import stimulus_full_epoch_lowrank as lowrank
from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec

DEFAULT_COVARIANCE_TIME_WINDOWS = ((0.05, 0.30),)
DEFAULT_COVARIANCE_BASELINE_WINDOW = cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
DEFAULT_COVARIANCE_NORMALIZATION = "subject_baseline_whiten"
DEFAULT_COVARIANCE_FEATURE_MODE = "logeuclidean_covariance"
COVARIANCE_FEATURE_MODES = (
    "logeuclidean_covariance",
    "covariance_upper",
    "correlation_upper",
    "variance",
)
DEFAULT_COVARIANCE_SHRINKAGE = 0.1
DEFAULT_COVARIANCE_EPSILON = 1e-6
DEFAULT_COVARIANCE_PROJECTION = "pca"
DEFAULT_COVARIANCE_COMPONENTS = (32, 64, 128)
DEFAULT_COVARIANCE_CLASSIFIER = "multinomial-logistic"
DEFAULT_COVARIANCE_CLASSIFIER_PARAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
COVARIANCE_FEATURE_FAMILY = "covariance"


@dataclass(frozen=True)
class CovarianceStimulusConfig:
    """One covariance-feature cross-subject decoding candidate."""

    time_window: tuple[float, float] = DEFAULT_COVARIANCE_TIME_WINDOWS[0]
    baseline_window: tuple[float, float] = DEFAULT_COVARIANCE_BASELINE_WINDOW
    normalization: str = DEFAULT_COVARIANCE_NORMALIZATION
    covariance_feature_mode: str = DEFAULT_COVARIANCE_FEATURE_MODE
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE
    covariance_epsilon: float = DEFAULT_COVARIANCE_EPSILON
    projection: str = DEFAULT_COVARIANCE_PROJECTION
    n_components: int | float = 64
    classifier: str = DEFAULT_COVARIANCE_CLASSIFIER
    classifier_param: object = float("nan")
    max_trials_per_class_per_participant: int | None = None
    trial_selection: str = cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED
    chance_classes: int = cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class CovarianceFeatureSet:
    """Covariance features for one participant."""

    participant: int
    labels: np.ndarray
    features: np.ndarray
    normalization: str
    covariance_feature_mode: str
    covariance_shrinkage: float
    covariance_epsilon: float
    n_channels: int
    n_time_bins: int
    n_baseline_samples: int
    trial_indices: np.ndarray
    max_trials_per_class_per_participant: int | None
    trial_selection: str
    trial_selection_seed: int | None


def make_covariance_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    time_windows=DEFAULT_COVARIANCE_TIME_WINDOWS,
    baseline_window=DEFAULT_COVARIANCE_BASELINE_WINDOW,
    normalizations=(DEFAULT_COVARIANCE_NORMALIZATION,),
    covariance_feature_modes=(DEFAULT_COVARIANCE_FEATURE_MODE,),
    covariance_shrinkages=(DEFAULT_COVARIANCE_SHRINKAGE,),
    covariance_epsilons=(DEFAULT_COVARIANCE_EPSILON,),
    projections=(DEFAULT_COVARIANCE_PROJECTION,),
    classifiers=(DEFAULT_COVARIANCE_CLASSIFIER,),
    classifier_params=DEFAULT_COVARIANCE_CLASSIFIER_PARAMS,
    components_values=DEFAULT_COVARIANCE_COMPONENTS,
    max_trials_per_class_per_participant=None,
    trial_selection=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    chance_classes=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build a nested-LOSO candidate grid for covariance-feature decoding."""

    normalized_time_windows = tuple(lowrank._normalize_time_window(time_window) for time_window in time_windows)  # pylint: disable=protected-access
    return tuple(
        CovarianceStimulusConfig(
            time_window=time_window,
            baseline_window=lowrank._normalize_time_window(baseline_window),  # pylint: disable=protected-access
            normalization=lowrank._normalize_normalization(normalization),  # pylint: disable=protected-access
            covariance_feature_mode=_normalize_covariance_feature_mode(covariance_feature_mode),
            covariance_shrinkage=_normalize_covariance_shrinkage(covariance_shrinkage),
            covariance_epsilon=_normalize_covariance_epsilon(covariance_epsilon),
            projection=lowrank._normalize_projection(projection),  # pylint: disable=protected-access
            n_components=components,
            classifier=classifier,
            classifier_param=classifier_param,
            max_trials_per_class_per_participant=lowrank._normalize_trial_cap(max_trials_per_class_per_participant),  # pylint: disable=protected-access
            trial_selection=lowrank._normalize_trial_selection(trial_selection),  # pylint: disable=protected-access
            trial_selection_seed=lowrank._normalize_trial_selection_seed(trial_selection_seed),  # pylint: disable=protected-access
            chance_classes=int(chance_classes),
            random_state=random_state,
            signflip_permutations=int(signflip_permutations),
            signflip_seed=signflip_seed,
        )
        for time_window, normalization, covariance_feature_mode, covariance_shrinkage, covariance_epsilon, projection, classifier, classifier_param, components in product(
            normalized_time_windows,
            tuple(normalizations),
            tuple(covariance_feature_modes),
            tuple(covariance_shrinkages),
            tuple(covariance_epsilons),
            tuple(projections),
            tuple(classifiers),
            tuple(classifier_params),
            tuple(components_values),
        )
    )


def evaluate_nested_covariance_stimulus(
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO model selection for covariance-feature decoding."""

    candidate_configs = tuple(_normalized_config(config) for config in candidate_configs)
    if not candidate_configs:
        raise ValueError("At least one candidate configuration is required.")
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for nested cross-subject decoding.")
    outer_participants = _normalize_outer_participants(participants, outer_participants)
    data_folder = resolve_data_folder(data_folder)

    feature_cache = _load_feature_cache(data_folder, participants, candidate_configs, progress=progress)
    outer_rows = []
    inner_rows = []
    selected_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        outer_train_participants = tuple(participant for participant in participants if participant != test_participant)
        fold_inner_rows = _evaluate_inner_rows(
            test_participant,
            outer_train_participants,
            candidate_configs,
            feature_cache,
            progress=progress,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        selected_row = _select_candidate(fold_inner_rows, candidate_configs)
        selected_config = candidate_configs[int(selected_row["selected_candidate_index"]) - 1]
        selected_features = feature_cache[_feature_cache_key(selected_config)]
        train_sets = [selected_features[participant] for participant in outer_train_participants]
        test_set = selected_features[test_participant]
        fitted_model = _fit_fold_model(
            train_sets,
            selected_config,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(test_participant), int(selected_row["selected_candidate_index"]), 0),
        )
        outer_row, fold_predictions = _score_fold_model(fitted_model, test_set, selected_config, include_predictions=True)
        lowrank._add_selected_candidate_fields(outer_row, selected_row)  # pylint: disable=protected-access
        for prediction_row in fold_predictions:
            lowrank._add_selected_candidate_fields(prediction_row, selected_row)  # pylint: disable=protected-access
        inner_rows.extend(fold_inner_rows)
        outer_rows.append(outer_row)
        selected_rows.append(selected_row)
        prediction_rows.extend(fold_predictions)
        if progress is not None:
            progress(
                "DONE covariance_outer "
                f"outer_test_participant={test_participant} selected_candidate={selected_row['selected_candidate_index']} "
                f"inner_mean={selected_row['selected_inner_balanced_accuracy_mean']:.4f} "
                f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
            )

    group_summary_rows = cross_subject.summarize_nested_cross_subject_stimulus(
        outer_rows,
        signflip_permutations=candidate_configs[0].signflip_permutations,
        signflip_seed=candidate_configs[0].signflip_seed,
    )
    _add_covariance_group_summary_fields(group_summary_rows, outer_rows)
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


def export_nested_covariance_stimulus(  # pylint: disable=too-many-arguments
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
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run the covariance-feature benchmark and write CSV artifacts."""

    artifacts = evaluate_nested_covariance_stimulus(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_participants=outer_participants,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )
    lowrank._write_rows_if_present(artifacts["outer"], outer_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["group_summary"], group_summary_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["inner_validation"], inner_validation_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["selected"], selected_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["predictions"], predictions_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["confusion"], confusion_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["per_stimulus"], per_stimulus_output_path)  # pylint: disable=protected-access
    lowrank._write_rows_if_present(artifacts["confusion_pairs"], confusion_pairs_output_path)  # pylint: disable=protected-access
    return artifacts


def load_participant_covariance_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract covariance features."""

    config = _normalized_config(config or CovarianceStimulusConfig())
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = sio.loadmat(data_path)["data"][0]
    all_labels = cross_subject._trialinfo_labels(data)  # pylint: disable=protected-access
    trial_indices = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
        all_labels,
        config.max_trials_per_class_per_participant,
        selection=config.trial_selection,
        seed=config.trial_selection_seed,
        participant=participant,
    )
    labels = np.asarray(all_labels[trial_indices], dtype=int)
    features, n_window_samples, n_baseline_samples = _extract_covariance_features(data, config, trial_indices=trial_indices)
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return CovarianceFeatureSet(
        participant=int(participant),
        labels=labels,
        features=np.asarray(features, dtype=float),
        normalization=config.normalization,
        covariance_feature_mode=config.covariance_feature_mode,
        covariance_shrinkage=config.covariance_shrinkage,
        covariance_epsilon=config.covariance_epsilon,
        n_channels=int(cross_subject._trial_signal(data, 0).shape[0]),  # pylint: disable=protected-access
        n_time_bins=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        trial_indices=np.asarray(trial_indices, dtype=int),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        trial_selection=config.trial_selection,
        trial_selection_seed=config.trial_selection_seed,
    )


def _extract_covariance_features(data, config, *, trial_indices=None):
    time_vector = cross_subject._time_vector(data, 0)  # pylint: disable=protected-access
    window_mask = cross_subject._time_mask(time_vector, config.time_window)  # pylint: disable=protected-access
    n_window_samples = int(np.sum(window_mask))
    if n_window_samples < 1:
        raise ValueError(f"Covariance time window {config.time_window} contains no samples.")

    channel_mean = None
    channel_std = None
    whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in {"subject_baseline_z", "subject_baseline_whiten"}:
        channel_mean, channel_std, n_baseline_samples = cross_subject._baseline_channel_statistics(  # pylint: disable=protected-access
            data,
            config.baseline_window,
            trial_indices,
        )
        if config.normalization == "subject_baseline_whiten":
            whitening_matrix, n_baseline_samples = cross_subject._baseline_channel_whitening_matrix(  # pylint: disable=protected-access
                data,
                config.baseline_window,
                trial_indices,
            )

    feature_rows = []
    for trial_idx in cross_subject._iter_trial_indices(data, trial_indices):  # pylint: disable=protected-access
        signal = np.asarray(cross_subject._trial_signal(data, trial_idx)[:, window_mask], dtype=float)  # pylint: disable=protected-access
        signal = _normalize_trial_signal(signal, config.normalization, channel_mean, channel_std, whitening_matrix)
        feature_rows.append(
            _covariance_feature_vector(
                signal,
                config.covariance_feature_mode,
                shrinkage=config.covariance_shrinkage,
                epsilon=config.covariance_epsilon,
            )
        )
    features = np.vstack(feature_rows)
    if config.normalization == "subject_z":
        features = (features - np.mean(features, axis=0, keepdims=True)) / _nonzero_std(np.std(features, axis=0, keepdims=True))
    elif config.normalization == "subject_trial_z":
        features = lowrank._trial_zscore_features(features)  # pylint: disable=protected-access
    elif config.normalization not in {"none", "subject_baseline_z", "subject_baseline_whiten"}:
        raise ValueError(f"Unsupported normalization: {config.normalization}")
    return features, n_window_samples, n_baseline_samples


def _normalize_trial_signal(signal, normalization, channel_mean, channel_std, whitening_matrix):
    signal = np.asarray(signal, dtype=float)
    if normalization == "subject_baseline_z":
        mean = np.asarray(channel_mean, dtype=float).reshape(-1, 1)
        std = _nonzero_std(np.asarray(channel_std, dtype=float).reshape(-1, 1))
        return (signal - mean) / std
    if normalization == "subject_baseline_whiten":
        mean = np.asarray(channel_mean, dtype=float).reshape(-1, 1)
        whitening_matrix = np.asarray(whitening_matrix, dtype=float)
        return whitening_matrix @ (signal - mean)
    return signal


def _covariance_feature_vector(signal, mode, *, shrinkage, epsilon):
    mode = _normalize_covariance_feature_mode(mode)
    covariance = _trial_covariance(signal, shrinkage=shrinkage, epsilon=epsilon)
    if mode == "variance":
        return np.log(np.maximum(np.diag(covariance), _eigen_floor(covariance, epsilon)))
    if mode == "correlation_upper":
        covariance = _covariance_to_correlation(covariance)
        return _vectorize_symmetric(covariance, scale_off_diagonal=True)
    if mode == "covariance_upper":
        return _vectorize_symmetric(covariance, scale_off_diagonal=True)
    if mode == "logeuclidean_covariance":
        return _vectorize_symmetric(_matrix_log_spd(covariance, epsilon), scale_off_diagonal=True)
    raise ValueError(f"Unsupported covariance feature mode: {mode}")


def _trial_covariance(signal, *, shrinkage, epsilon):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("Trial signal must be channels x time.")
    centered = signal - np.mean(signal, axis=1, keepdims=True)
    denominator = max(1, int(centered.shape[1]) - 1)
    covariance = (centered @ centered.T) / float(denominator)
    covariance = 0.5 * (covariance + covariance.T)
    n_channels = covariance.shape[0]
    trace_mean = float(np.trace(covariance) / max(1, n_channels))
    if not np.isfinite(trace_mean) or trace_mean <= 0.0:
        trace_mean = 1.0
    shrinkage = _normalize_covariance_shrinkage(shrinkage)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * np.eye(n_channels)
    covariance = covariance + _normalize_covariance_epsilon(epsilon) * trace_mean * np.eye(n_channels)
    return 0.5 * (covariance + covariance.T)


def _covariance_to_correlation(covariance):
    covariance = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    correlation = covariance / np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


def _matrix_log_spd(matrix, epsilon):
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = _eigen_floor(matrix, epsilon)
    log_values = np.log(np.maximum(eigenvalues, floor))
    return (eigenvectors * log_values[None, :]) @ eigenvectors.T


def _eigen_floor(matrix, epsilon):
    scale = float(np.trace(matrix) / max(1, matrix.shape[0]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return _normalize_covariance_epsilon(epsilon) * scale


def _vectorize_symmetric(matrix, *, scale_off_diagonal):
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = np.triu_indices(matrix.shape[0])
    values = np.asarray(matrix[rows, cols], dtype=float)
    if scale_off_diagonal:
        values = values.copy()
        values[rows != cols] *= np.sqrt(2.0)
    return values


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    representative_configs: dict[tuple, CovarianceStimulusConfig] = {}
    for config in candidate_configs:
        representative_configs.setdefault(_feature_cache_key(config), config)

    feature_cache = {}
    for key, config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD covariance_features "
                f"time_window={lowrank._time_window_string(config.time_window)} "  # pylint: disable=protected-access
                f"mode={config.covariance_feature_mode} "
                f"normalization={config.normalization} "
                f"shrinkage={config.covariance_shrinkage:g}"
            )
        feature_cache[key] = {participant: load_participant_covariance_features(data_folder, participant, config=config) for participant in participants}
    return feature_cache


def _feature_cache_key(config):
    config = _normalized_config(config)
    return (
        float(config.time_window[0]),
        float(config.time_window[1]),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.normalization),
        str(config.covariance_feature_mode),
        float(config.covariance_shrinkage),
        float(config.covariance_epsilon),
        config.max_trials_per_class_per_participant,
        str(config.trial_selection),
        lowrank._seed_field(config.trial_selection_seed),  # pylint: disable=protected-access
    )


def _evaluate_inner_rows(
    test_participant,
    outer_train_participants,
    candidate_configs,
    feature_cache,
    *,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    rows = []
    total = len(candidate_configs) * len(outer_train_participants)
    completed = 0
    for candidate_index, config in enumerate(candidate_configs, start=1):
        features = feature_cache[_feature_cache_key(config)]
        for validation_participant in outer_train_participants:
            inner_train_participants = tuple(participant for participant in outer_train_participants if participant != validation_participant)
            train_sets = [features[participant] for participant in inner_train_participants]
            fitted_model = _fit_fold_model(
                train_sets,
                config,
                label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
                label_shuffle_context=(int(test_participant), int(validation_participant), int(candidate_index)),
            )
            inner_row, _predictions = _score_fold_model(fitted_model, features[validation_participant], config, include_predictions=False)
            rows.append(lowrank._nested_inner_row(inner_row, test_participant, validation_participant, candidate_index))  # pylint: disable=protected-access
            completed += 1
            if progress is not None:
                progress(
                    "DONE covariance_inner_validation "
                    f"outer_test_participant={test_participant} "
                    f"candidate={candidate_index}/{len(candidate_configs)} "
                    f"validation_participant={validation_participant} "
                    f"progress={completed}/{total}"
                )
    return rows


def _fit_fold_model(train_sets, config, *, label_shuffle_seed=None, label_shuffle_context=()):
    return lowrank._fit_outer_fold_model(  # pylint: disable=protected-access
        train_sets,
        _lowrank_adapter_config(config),
        label_shuffle_seed=label_shuffle_seed,
        label_shuffle_context=label_shuffle_context,
    )


def _score_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    row, predictions = lowrank._score_outer_fold_model(  # pylint: disable=protected-access
        fitted_model,
        test_set,
        _lowrank_adapter_config(config),
        include_predictions=include_predictions,
    )
    _add_covariance_row_fields(row, config)
    for prediction in predictions:
        _add_covariance_row_fields(prediction, config)
    return row, predictions


def _lowrank_adapter_config(config):
    config = _normalized_config(config)
    # The reused scorer expects a positive bin size.  Covariance features are not
    # binned, so downstream rows are patched to leave time_bin_size_s empty.
    return lowrank.FullEpochLowRankConfig(
        time_window=config.time_window,
        time_bin_size=max(1e-6, float(config.time_window[1] - config.time_window[0])),
        baseline_window=config.baseline_window,
        normalization=config.normalization,
        projection=config.projection,
        n_components=config.n_components,
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        trial_selection=config.trial_selection,
        trial_selection_seed=config.trial_selection_seed,
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _add_covariance_row_fields(row, config):
    row["feature_mode"] = config.covariance_feature_mode
    row["feature_family"] = COVARIANCE_FEATURE_FAMILY
    row["covariance_feature_mode"] = config.covariance_feature_mode
    row["covariance_shrinkage"] = config.covariance_shrinkage
    row["covariance_epsilon"] = config.covariance_epsilon
    row["time_bin_size_s"] = ""
    return row


def _select_candidate(inner_rows, candidate_configs):
    selected_row = lowrank._select_candidate(inner_rows, candidate_configs)  # pylint: disable=protected-access
    selected_config = candidate_configs[int(selected_row["selected_candidate_index"]) - 1]
    selected_row["selected_feature_mode"] = selected_config.covariance_feature_mode
    selected_row["selected_feature_family"] = COVARIANCE_FEATURE_FAMILY
    selected_row["selected_covariance_feature_mode"] = selected_config.covariance_feature_mode
    selected_row["selected_covariance_shrinkage"] = selected_config.covariance_shrinkage
    selected_row["selected_covariance_epsilon"] = selected_config.covariance_epsilon
    selected_row["selected_time_bin_size_s"] = ""
    selected_row["selected_ensemble_feature_mode_counts"] = lowrank._format_counter(Counter((selected_config.covariance_feature_mode,)))  # pylint: disable=protected-access
    selected_row["selected_ensemble_covariance_feature_mode_counts"] = lowrank._format_counter(Counter((selected_config.covariance_feature_mode,)))  # pylint: disable=protected-access
    selected_row["selected_ensemble_covariance_shrinkage_counts"] = lowrank._format_counter(Counter((f"{selected_config.covariance_shrinkage:g}",)))  # pylint: disable=protected-access
    return selected_row


def _add_covariance_group_summary_fields(group_summary_rows, outer_rows):
    if not group_summary_rows or not outer_rows:
        return
    summary = group_summary_rows[0]
    summary["feature_family"] = COVARIANCE_FEATURE_FAMILY
    summary["selected_covariance_feature_mode_counts"] = lowrank._format_counter(  # pylint: disable=protected-access
        Counter(str(row.get("selected_covariance_feature_mode", row.get("covariance_feature_mode", ""))) for row in outer_rows)
    )
    summary["selected_covariance_shrinkage_counts"] = lowrank._format_counter(  # pylint: disable=protected-access
        Counter(str(row.get("selected_covariance_shrinkage", row.get("covariance_shrinkage", ""))) for row in outer_rows)
    )
    summary["selected_projection_counts"] = lowrank._format_counter(  # pylint: disable=protected-access
        Counter(str(row.get("selected_projection", row.get("projection", ""))) for row in outer_rows)
    )
    summary["selected_time_window_counts"] = lowrank._format_counter(  # pylint: disable=protected-access
        Counter(str(row.get("selected_time_window_s", row.get("time_window_s", ""))) for row in outer_rows)
    )


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


def _normalized_config(config):
    return CovarianceStimulusConfig(
        time_window=lowrank._normalize_time_window(config.time_window),  # pylint: disable=protected-access
        baseline_window=lowrank._normalize_time_window(config.baseline_window),  # pylint: disable=protected-access
        normalization=lowrank._normalize_normalization(config.normalization),  # pylint: disable=protected-access
        covariance_feature_mode=_normalize_covariance_feature_mode(config.covariance_feature_mode),
        covariance_shrinkage=_normalize_covariance_shrinkage(config.covariance_shrinkage),
        covariance_epsilon=_normalize_covariance_epsilon(config.covariance_epsilon),
        projection=lowrank._normalize_projection(config.projection),  # pylint: disable=protected-access
        n_components=config.n_components,
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        max_trials_per_class_per_participant=lowrank._normalize_trial_cap(config.max_trials_per_class_per_participant),  # pylint: disable=protected-access
        trial_selection=lowrank._normalize_trial_selection(config.trial_selection),  # pylint: disable=protected-access
        trial_selection_seed=lowrank._normalize_trial_selection_seed(config.trial_selection_seed),  # pylint: disable=protected-access
        chance_classes=int(config.chance_classes),
        random_state=config.random_state,
        signflip_permutations=int(config.signflip_permutations),
        signflip_seed=config.signflip_seed,
    )


def _normalize_covariance_feature_mode(value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "logeig_covariance": "logeuclidean_covariance",
        "log_covariance": "logeuclidean_covariance",
        "covariance_logeuclidean": "logeuclidean_covariance",
        "covariance": "covariance_upper",
        "correlation": "correlation_upper",
        "diag_variance": "variance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in COVARIANCE_FEATURE_MODES:
        raise ValueError(f"covariance_feature_mode must be one of {COVARIANCE_FEATURE_MODES}.")
    return normalized


def _normalize_covariance_shrinkage(value):
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("covariance_shrinkage must be a finite value in [0, 1].")
    return value


def _normalize_covariance_epsilon(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("covariance_epsilon must be a positive finite value.")
    return value


def _nonzero_std(std):
    return np.where(std < 1e-12, 1.0, std)


def _parse_covariance_feature_mode_list(value: str) -> tuple[str, ...]:
    return tuple(_normalize_covariance_feature_mode(token) for token in lowrank._parse_token_list(value))  # pylint: disable=protected-access


def _parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run nested LOSO covariance-feature stimulus decoding using Part*Data.mat files only.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=cross_subject.DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate in this run. Defaults to all participants.")
    parser.add_argument(
        "--time-windows",
        type=lowrank._parse_time_window_grid,  # pylint: disable=protected-access
        default=DEFAULT_COVARIANCE_TIME_WINDOWS,
        help="Comma-separated covariance crop windows as start:stop pairs, e.g. 0.05:0.30,0.08:0.35.",
    )
    parser.add_argument(
        "--baseline-window",
        type=lowrank._parse_time_window,  # pylint: disable=protected-access
        default=DEFAULT_COVARIANCE_BASELINE_WINDOW,
        help="Baseline window as start:stop in seconds.",
    )
    parser.add_argument(
        "--normalizations",
        type=lowrank._parse_normalization_list,  # pylint: disable=protected-access
        default=(DEFAULT_COVARIANCE_NORMALIZATION,),
        help="Comma-separated subject normalization modes.",
    )
    parser.add_argument(
        "--feature-modes",
        type=_parse_covariance_feature_mode_list,
        default=(DEFAULT_COVARIANCE_FEATURE_MODE,),
        help="Comma-separated covariance feature modes: logeuclidean_covariance,covariance_upper,correlation_upper,variance.",
    )
    parser.add_argument(
        "--covariance-shrinkages",
        type=_parse_float_grid,
        default=(DEFAULT_COVARIANCE_SHRINKAGE,),
        help="Comma-separated shrinkage values in [0,1] applied before covariance vectorization.",
    )
    parser.add_argument(
        "--covariance-epsilons",
        type=_parse_float_grid,
        default=(DEFAULT_COVARIANCE_EPSILON,),
        help="Comma-separated positive eigenvalue floors, relative to mean variance.",
    )
    parser.add_argument("--projections", type=lowrank._parse_projection_list, default=(DEFAULT_COVARIANCE_PROJECTION,), help="Comma-separated projection modes: pca,pls,none.")  # pylint: disable=protected-access
    parser.add_argument("--classifiers", type=lowrank._parse_token_list, default=(DEFAULT_COVARIANCE_CLASSIFIER,), help="Comma-separated classifier names.")  # pylint: disable=protected-access
    parser.add_argument(
        "--classifier-params",
        type=lowrank._parse_classifier_param_grid,  # pylint: disable=protected-access
        default=DEFAULT_COVARIANCE_CLASSIFIER_PARAMS,
        help="Comma-separated classifier parameters. Use default for each classifier default.",
    )
    parser.add_argument("--components-values", type=lowrank._parse_int_or_inf_list, default=DEFAULT_COVARIANCE_COMPONENTS, help="Comma-separated low-rank dimensions, or inf.")  # pylint: disable=protected-access
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Optional deterministic cap on trials per stimulus class and participant.")
    parser.add_argument("--trial-selection", choices=cross_subject.TRIAL_SELECTION_MODES, default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION, help="Trial subset policy used when a trial cap is set.")
    parser.add_argument("--trial-selection-seed", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED, help="Seed for random trial selection.")
    parser.add_argument("--chance-classes", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to projections and classifiers.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Shuffle training labels within each participant for a nested null-control benchmark.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the nested label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_covariance_outer.csv", help="Untouched outer participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_covariance_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_covariance_inner_validation.csv", help="Inner validation score CSV.")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_covariance_selected.csv", help="Selected hyperparameter CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_covariance_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_covariance_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_covariance_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_covariance_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_covariance(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    candidate_configs = make_covariance_candidate_configs(
        time_windows=args.time_windows,
        baseline_window=args.baseline_window,
        normalizations=args.normalizations,
        covariance_feature_modes=args.feature_modes,
        covariance_shrinkages=args.covariance_shrinkages,
        covariance_epsilons=args.covariance_epsilons,
        projections=args.projections,
        classifiers=args.classifiers,
        classifier_params=args.classifier_params,
        components_values=args.components_values,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        trial_selection=args.trial_selection,
        trial_selection_seed=args.trial_selection_seed,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    artifacts = export_nested_covariance_stimulus(
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


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_cross_subject_covariance(argv, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
