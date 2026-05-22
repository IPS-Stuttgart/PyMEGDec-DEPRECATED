"""Full-epoch supervised low-rank BUSH-MEG stimulus decoding."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.cli import normalize_argv, parse_classifier_param, parse_int_or_inf
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec

DEFAULT_FULL_EPOCH_TIME_WINDOWS = ((0.0, 0.45),)
DEFAULT_FULL_EPOCH_TIME_BIN_SIZE = 0.01
DEFAULT_FULL_EPOCH_COMPONENTS = (32, 64, 128)
DEFAULT_FULL_EPOCH_CLASSIFIER_PARAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_FULL_EPOCH_CLASSIFIER = "multinomial-logistic"
DEFAULT_FULL_EPOCH_PROJECTION = "pls"
FULL_EPOCH_FEATURE_MODE = "sensor_time_binned"
PROJECTION_MODES = ("pls", "pca", "none")
FULL_EPOCH_NORMALIZATION_MODES = cross_subject.NORMALIZATION_MODES


@dataclass(frozen=True)
class FullEpochLowRankConfig:
    """One full-epoch low-rank cross-subject decoding candidate."""

    time_window: tuple[float, float] = DEFAULT_FULL_EPOCH_TIME_WINDOWS[0]
    time_bin_size: float = DEFAULT_FULL_EPOCH_TIME_BIN_SIZE
    baseline_window: tuple[float, float] = cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    normalization: str = "subject_baseline_whiten"
    projection: str = DEFAULT_FULL_EPOCH_PROJECTION
    n_components: int | float = 64
    classifier: str = DEFAULT_FULL_EPOCH_CLASSIFIER
    classifier_param: object = float("nan")
    max_trials_per_class_per_participant: int | None = None
    trial_selection: str = cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED
    chance_classes: int = cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class FullEpochFeatureSet:
    """Binned full-epoch features for one participant."""

    participant: int
    labels: np.ndarray
    features: np.ndarray
    normalization: str
    n_channels: int
    n_time_bins: int
    n_baseline_samples: int
    trial_indices: np.ndarray
    max_trials_per_class_per_participant: int | None
    trial_selection: str
    trial_selection_seed: int | None


@dataclass
class LowRankModelBundle:
    """Classifier plus the fitted full-epoch projection."""

    model: object
    projection: str
    transformer: object | None
    train_labels: np.ndarray
    actual_components_pca: int | float
    explained_variance_percent: float


def make_full_epoch_lowrank_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    time_windows=DEFAULT_FULL_EPOCH_TIME_WINDOWS,
    time_bin_size=DEFAULT_FULL_EPOCH_TIME_BIN_SIZE,
    baseline_window=cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    normalizations=("subject_baseline_whiten",),
    projections=(DEFAULT_FULL_EPOCH_PROJECTION,),
    classifiers=(DEFAULT_FULL_EPOCH_CLASSIFIER,),
    classifier_params=DEFAULT_FULL_EPOCH_CLASSIFIER_PARAMS,
    components_values=DEFAULT_FULL_EPOCH_COMPONENTS,
    max_trials_per_class_per_participant=None,
    trial_selection=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    chance_classes=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    """Build a nested-LOSO candidate grid for full-epoch supervised low-rank decoding."""

    normalized_time_windows = tuple(_normalize_time_window(time_window) for time_window in time_windows)
    return tuple(
        FullEpochLowRankConfig(
            time_window=time_window,
            time_bin_size=_normalize_time_bin_size(time_bin_size),
            baseline_window=_normalize_time_window(baseline_window),
            normalization=_normalize_normalization(normalization),
            projection=_normalize_projection(projection),
            n_components=components,
            classifier=classifier,
            classifier_param=classifier_param,
            max_trials_per_class_per_participant=_normalize_trial_cap(max_trials_per_class_per_participant),
            trial_selection=_normalize_trial_selection(trial_selection),
            trial_selection_seed=_normalize_trial_selection_seed(trial_selection_seed),
            chance_classes=int(chance_classes),
            random_state=random_state,
            signflip_permutations=int(signflip_permutations),
            signflip_seed=signflip_seed,
        )
        for time_window, normalization, projection, classifier, classifier_param, components in product(
            normalized_time_windows,
            tuple(normalizations),
            tuple(projections),
            tuple(classifiers),
            tuple(classifier_params),
            tuple(components_values),
        )
    )


def evaluate_nested_full_epoch_lowrank_stimulus(
    data_folder,
    participants,
    *,
    candidate_configs,
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
):
    """Run nested LOSO model selection for the full-epoch low-rank decoder."""

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
        fitted_model = _fit_outer_fold_model(
            train_sets,
            selected_config,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(test_participant), int(selected_row["selected_candidate_index"]), 0),
        )
        outer_row, fold_predictions = _score_outer_fold_model(fitted_model, test_set, selected_config, include_predictions=True)
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
                f"{test_participant} selected_candidate={selected_row['selected_candidate_index']} "
                f"inner_mean={selected_row['selected_inner_balanced_accuracy_mean']:.4f} "
                f"outer_balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
            )

    group_summary_rows = cross_subject.summarize_nested_cross_subject_stimulus(
        outer_rows,
        signflip_permutations=candidate_configs[0].signflip_permutations,
        signflip_seed=candidate_configs[0].signflip_seed,
    )
    _add_full_epoch_group_summary_fields(group_summary_rows, outer_rows)
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


def export_nested_full_epoch_lowrank_stimulus(  # pylint: disable=too-many-arguments
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
    """Run the full-epoch low-rank benchmark and write compact CSV artifacts."""

    artifacts = evaluate_nested_full_epoch_lowrank_stimulus(
        data_folder,
        participants,
        candidate_configs=candidate_configs,
        outer_participants=outer_participants,
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


def load_participant_full_epoch_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract binned full-epoch features."""

    config = _normalized_config(config or FullEpochLowRankConfig())
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
    features, n_time_bins = _extract_binned_time_features(data, config.time_window, config.time_bin_size, trial_indices=trial_indices)
    n_baseline_samples = 0
    if config.normalization == "subject_z":
        mean = np.mean(features, axis=0, keepdims=True)
        std = _nonzero_std(np.std(features, axis=0, keepdims=True))
        features = (features - mean) / std
    elif config.normalization == "subject_trial_z":
        features = _trial_zscore_features(features)
    elif config.normalization in {"subject_baseline_z", "subject_baseline_whiten"}:
        channel_mean, channel_std, n_baseline_samples = cross_subject._baseline_channel_statistics(  # pylint: disable=protected-access
            data,
            config.baseline_window,
            trial_indices,
        )
        tiled_mean = np.tile(channel_mean, int(n_time_bins))[None, :]
        if config.normalization == "subject_baseline_z":
            tiled_std = _nonzero_std(np.tile(channel_std, int(n_time_bins))[None, :])
            features = (features - tiled_mean) / tiled_std
        else:
            whitening_matrix, n_baseline_samples = cross_subject._baseline_channel_whitening_matrix(  # pylint: disable=protected-access
                data,
                config.baseline_window,
                trial_indices,
            )
            features = _baseline_whiten_binned_features(features - tiled_mean, whitening_matrix)
    elif config.normalization != "none":
        raise ValueError(f"Unsupported normalization: {config.normalization}")

    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return FullEpochFeatureSet(
        participant=int(participant),
        labels=labels,
        features=np.asarray(features, dtype=float),
        normalization=config.normalization,
        n_channels=int(cross_subject._trial_signal(data, 0).shape[0]),  # pylint: disable=protected-access
        n_time_bins=int(n_time_bins),
        n_baseline_samples=int(n_baseline_samples),
        trial_indices=np.asarray(trial_indices, dtype=int),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        trial_selection=config.trial_selection,
        trial_selection_seed=config.trial_selection_seed,
    )


def _extract_binned_time_features(data, time_window, time_bin_size, *, trial_indices=None):
    time_window = _normalize_time_window(time_window)
    time_bin_size = _normalize_time_bin_size(time_bin_size)
    time_vector = cross_subject._time_vector(data, 0)  # pylint: disable=protected-access
    window_mask = cross_subject._time_mask(time_vector, time_window)  # pylint: disable=protected-access
    window_time = time_vector[window_mask]
    bin_edges = _bin_edges(time_window, time_bin_size)
    features = []
    for trial_idx in cross_subject._iter_trial_indices(data, trial_indices):  # pylint: disable=protected-access
        signal = cross_subject._trial_signal(data, trial_idx)  # pylint: disable=protected-access
        window_signal = signal[:, window_mask]
        features.append(_bin_trial_signal(window_signal, window_time, bin_edges))
    return np.vstack(features), int(len(bin_edges) - 1)


def _bin_edges(time_window, time_bin_size):
    start, stop = _normalize_time_window(time_window)
    n_bins = max(1, int(ceil((stop - start) / float(time_bin_size) - 1e-12)))
    edges = start + np.arange(n_bins + 1, dtype=float) * float(time_bin_size)
    edges[-1] = stop
    return edges


def _bin_trial_signal(window_signal, window_time, bin_edges):
    window_signal = np.asarray(window_signal, dtype=float)
    window_time = np.asarray(window_time, dtype=float).ravel()
    binned_columns = []
    tolerance = 1e-12
    for bin_index, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=True)):
        if bin_index == len(bin_edges) - 2:
            mask = (window_time >= left - tolerance) & (window_time <= right + tolerance)
        else:
            mask = (window_time >= left - tolerance) & (window_time < right - tolerance)
        if np.any(mask):
            binned_columns.append(np.mean(window_signal[:, mask], axis=1))
        else:
            midpoint = 0.5 * (left + right)
            nearest = int(np.argmin(np.abs(window_time - midpoint)))
            binned_columns.append(window_signal[:, nearest])
    binned = np.column_stack(binned_columns)
    return binned.reshape(-1, order="F")


def _baseline_whiten_binned_features(features, whitening_matrix):
    features = np.asarray(features, dtype=float)
    whitening_matrix = np.asarray(whitening_matrix, dtype=float)
    n_channels = int(whitening_matrix.shape[0])
    if features.shape[1] % n_channels:
        raise ValueError("Binned feature width must be a multiple of the number of whitening channels.")
    n_time_bins = int(features.shape[1] // n_channels)
    matrices = features.reshape(features.shape[0], n_time_bins, n_channels)
    whitened = matrices @ whitening_matrix.T
    return whitened.reshape(features.shape[0], -1)


def _load_feature_cache(data_folder, participants, candidate_configs, *, progress=None):
    representative_configs: dict[tuple, FullEpochLowRankConfig] = {}
    for config in candidate_configs:
        representative_configs.setdefault(_feature_cache_key(config), config)

    feature_cache = {}
    for key, config in representative_configs.items():
        if progress is not None:
            progress(
                "LOAD full_epoch_features "
                f"time_window={_time_window_string(config.time_window)} "
                f"time_bin_size={config.time_bin_size:g} "
                f"normalization={config.normalization}"
            )
        feature_cache[key] = {participant: load_participant_full_epoch_features(data_folder, participant, config=config) for participant in participants}
    return feature_cache


def _feature_cache_key(config):
    config = _normalized_config(config)
    return (
        float(config.time_window[0]),
        float(config.time_window[1]),
        float(config.time_bin_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.normalization),
        config.max_trials_per_class_per_participant,
        str(config.trial_selection),
        _seed_field(config.trial_selection_seed),
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
            fitted_model = _fit_outer_fold_model(
                train_sets,
                config,
                label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
                label_shuffle_context=(int(test_participant), int(validation_participant), int(candidate_index)),
            )
            inner_row, _predictions = _score_outer_fold_model(fitted_model, features[validation_participant], config, include_predictions=False)
            rows.append(_nested_inner_row(inner_row, test_participant, validation_participant, candidate_index))
            completed += 1
            if progress is not None:
                progress(
                    "DONE full_epoch_inner_validation "
                    f"outer_test_participant={test_participant} "
                    f"candidate={candidate_index}/{len(candidate_configs)} "
                    f"validation_participant={validation_participant} "
                    f"progress={completed}/{total}"
                )
    return rows


def _fit_outer_fold_model(train_sets, config, *, label_shuffle_seed=None, label_shuffle_context=()):
    config = _normalized_config(config)
    train_features = np.vstack([feature_set.features for feature_set in train_sets])
    train_labels_one_based = np.concatenate(
        [
            _training_labels(
                feature_set,
                label_shuffle_seed=label_shuffle_seed,
                label_shuffle_context=label_shuffle_context,
            )
            for feature_set in train_sets
        ]
    )
    train_labels = train_labels_one_based - 1
    classifier_param = _resolved_classifier_param(config)
    model_bundle = _fit_lowrank_model(train_features, train_labels, config, classifier_param)
    return {
        "classifier_param": classifier_param,
        "model_bundle": model_bundle,
        "n_train_participants": len(train_sets),
        "train_class_counts": Counter(train_labels_one_based.tolist()),
        "train_labels": train_labels,
        "train_participants": tuple(feature_set.participant for feature_set in train_sets),
        "train_window": config.time_window,
        "label_shuffle_control": label_shuffle_seed is not None,
        "label_shuffle_seed": _seed_field(label_shuffle_seed),
    }


def _fit_lowrank_model(train_features, train_labels, config, classifier_param):
    train_features = np.asarray(train_features, dtype=float)
    train_labels = np.asarray(train_labels, dtype=int).ravel()
    projected_features, transformer, actual_components, explained_variance_percent = _fit_projection(train_features, train_labels, config)
    model = train_multiclass_classifier(projected_features, train_labels, config.classifier, classifier_param, random_state=config.random_state)
    return LowRankModelBundle(
        model=model,
        projection=config.projection,
        transformer=transformer,
        train_labels=train_labels,
        actual_components_pca=actual_components,
        explained_variance_percent=explained_variance_percent,
    )


def _fit_projection(train_features, train_labels, config):
    projection = _normalize_projection(config.projection)
    if projection == "none":
        return train_features, None, int(train_features.shape[1]), np.nan

    requested = _requested_components(config.n_components)
    max_components = max(1, min(int(train_features.shape[1]), max(1, int(train_features.shape[0]) - 1)))
    actual_components = min(requested, max_components)
    if projection == "pca":
        transformer = PCA(n_components=actual_components, random_state=config.random_state)
        projected = transformer.fit_transform(train_features)
        explained_variance_percent = float(100.0 * np.sum(transformer.explained_variance_ratio_))
        return projected, transformer, int(actual_components), explained_variance_percent

    if projection == "pls":
        targets = _one_hot_labels(train_labels)
        transformer = PLSRegression(n_components=actual_components, scale=False)
        transformer.fit(train_features, targets)
        projected = transformer.transform(train_features)
        return projected, transformer, int(actual_components), np.nan

    raise ValueError(f"Unsupported projection: {projection}")


def _transform_features(model_bundle, features):
    features = np.asarray(features, dtype=float)
    if model_bundle.transformer is None:
        return features
    return model_bundle.transformer.transform(features)


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    config = _normalized_config(config)
    model_bundle = fitted_model["model_bundle"]
    test_features = np.asarray(test_set.features, dtype=float)
    test_labels_one_based = np.asarray(test_set.labels, dtype=int)
    test_labels = test_labels_one_based - 1
    transformed_features = _transform_features(model_bundle, test_features)
    predictions = np.asarray(model_bundle.model.predict(transformed_features), dtype=int)
    class_scores, score_classes = _model_class_scores(model_bundle, test_features)
    rank_metrics = _ranked_label_metrics(test_labels, class_scores, score_classes)
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))
    chance_accuracy = 1.0 / config.chance_classes
    train_class_counts = fitted_model["train_class_counts"]
    test_class_counts = Counter(test_labels_one_based.tolist())
    window_start, window_stop = config.time_window
    window_center = 0.5 * (window_start + window_stop)
    window_size = window_stop - window_start

    outer_row = {
        "outer_fold": int(test_set.participant),
        "test_participant": int(test_set.participant),
        "train_participants": ",".join(str(participant) for participant in fitted_model["train_participants"]),
        "n_train_participants": fitted_model["n_train_participants"],
        "n_test_participants": 1,
        "window_center_s": window_center,
        "window_size_s": window_size,
        "window_start_s": window_start,
        "window_stop_s": window_stop,
        "time_window_s": _time_window_string(config.time_window),
        "time_bin_size_s": config.time_bin_size,
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": FULL_EPOCH_FEATURE_MODE,
        "normalization": config.normalization,
        "alignment": "none",
        "projection": config.projection,
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
        "n_train_trials": int(np.asarray(fitted_model["train_labels"]).shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())),
        "min_test_trials_per_class": int(min(test_class_counts.values())),
        "classifier": config.classifier,
        "classifier_param": fitted_model["classifier_param"],
        "components_pca": config.n_components,
        "n_components": config.n_components,
        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
        "trial_selection": config.trial_selection,
        "trial_selection_seed": _seed_field(config.trial_selection_seed),
        "actual_components_pca": model_bundle.actual_components_pca,
        "projection_actual_components": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "n_channels": test_set.n_channels,
        "n_window_samples": test_set.n_time_bins,
        "n_time_bins": test_set.n_time_bins,
        "n_baseline_samples": test_set.n_baseline_samples,
        "label_shuffle_control": bool(fitted_model["label_shuffle_control"]),
        "label_shuffle_seed": fitted_model["label_shuffle_seed"],
        "alignment_common_classes": "",
        "alignment_aligned_participants": "",
    }
    prediction_rows = []
    if include_predictions:
        prediction_rows = _prediction_rows(test_set, test_labels, predictions, rank_metrics["true_label_ranks"], config=config, actual_components=model_bundle.actual_components_pca)
    return outer_row, prediction_rows


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components):
    window_start, window_stop = config.time_window
    rows = []
    trial_indices = np.asarray(test_set.trial_indices, dtype=int).ravel()
    for trial_idx, true_label, predicted_label, true_label_rank in zip(trial_indices, test_labels, predictions, true_label_ranks, strict=True):
        true_stimulus = int(true_label) + 1
        predicted_stimulus = int(predicted_label) + 1
        rows.append(
            {
                "outer_fold": int(test_set.participant),
                "test_participant": int(test_set.participant),
                "window_center_s": 0.5 * (window_start + window_stop),
                "window_start_s": window_start,
                "window_stop_s": window_stop,
                "time_window_s": _time_window_string(config.time_window),
                "time_bin_size_s": config.time_bin_size,
                "feature_mode": FULL_EPOCH_FEATURE_MODE,
                "normalization": config.normalization,
                "alignment": "none",
                "projection": config.projection,
                "classifier": config.classifier,
                "components_pca": config.n_components,
                "n_components": config.n_components,
                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,
                "trial_selection": config.trial_selection,
                "trial_selection_seed": _seed_field(config.trial_selection_seed),
                "actual_components_pca": actual_components,
                "projection_actual_components": actual_components,
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
    transformed_features = _transform_features(model_bundle, features)
    model = model_bundle.model
    classes = np.asarray(getattr(model, "classes_", np.arange(len(np.unique(model_bundle.train_labels)))), dtype=int)
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
    if true_label_ranks.size == 0 or class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return {
            "true_label_ranks": true_label_ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    return {
        "true_label_ranks": true_label_ranks,
        "top2_accuracy": float(np.mean(true_label_ranks <= 2)),
        "top3_accuracy": float(np.mean(true_label_ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _true_label_ranks(true_labels, class_scores, score_classes):
    true_labels = np.asarray(true_labels)
    if class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return np.full(true_labels.shape[0], np.nan, dtype=float)
    label_to_column = {int(label): column for column, label in enumerate(np.asarray(score_classes).tolist())}
    ranks = []
    for true_label, trial_scores in zip(true_labels, class_scores, strict=True):
        true_column = label_to_column.get(int(true_label))
        if true_column is None:
            ranks.append(np.nan)
            continue
        descending_columns = np.argsort(-trial_scores, kind="mergesort")
        rank_locations = np.flatnonzero(descending_columns == true_column)
        ranks.append(float(rank_locations[0] + 1) if rank_locations.size else np.nan)
    return np.asarray(ranks, dtype=float)


def _nested_inner_row(row, outer_test_participant, validation_participant, candidate_index):
    inner_row = dict(row)
    inner_row.update(
        {
            "selection_mode": "nested_loso",
            "selection_metric": cross_subject.DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
            "outer_test_participant": int(outer_test_participant),
            "inner_fold": int(validation_participant),
            "inner_validation_participant": int(validation_participant),
            "inner_train_participants": row["train_participants"],
            "n_inner_train_participants": row["n_train_participants"],
            "candidate_index": int(candidate_index),
        }
    )
    return inner_row


def _select_candidate(inner_rows, candidate_configs):
    summaries = []
    candidate_indices = sorted({int(row["candidate_index"]) for row in inner_rows})
    for candidate_index in candidate_indices:
        candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == candidate_index]
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in candidate_rows], dtype=float)
        raw = np.asarray([float(row["accuracy"]) for row in candidate_rows], dtype=float)
        example = candidate_rows[0]
        summaries.append(
            {
                "selection_mode": "nested_loso",
                "selection_metric": cross_subject.DEFAULT_CROSS_SUBJECT_SELECTION_METRIC,
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
                "selected_window_center_s": example["window_center_s"],
                "selected_window_size_s": example["window_size_s"],
                "selected_window_start_s": example["window_start_s"],
                "selected_window_stop_s": example["window_stop_s"],
                "selected_time_window_s": example["time_window_s"],
                "selected_time_bin_size_s": example["time_bin_size_s"],
                "selected_feature_mode": example["feature_mode"],
                "selected_normalization": example["normalization"],
                "selected_alignment": example["alignment"],
                "selected_projection": example["projection"],
                "selected_classifier": example["classifier"],
                "selected_classifier_param": example["classifier_param"],
                "selected_components_pca": example["components_pca"],
                "selected_n_components": example["n_components"],
                "selected_max_trials_per_class_per_participant": example["max_trials_per_class_per_participant"],
                "label_shuffle_control": example.get("label_shuffle_control", False),
                "label_shuffle_seed": example.get("label_shuffle_seed", ""),
            }
        )

    ranked = sorted(
        summaries,
        key=lambda row: (
            float(row["selected_inner_balanced_accuracy_mean"]),
            float(row["selected_inner_balanced_accuracy_median"]),
            -int(row["selected_candidate_index"]),
        ),
        reverse=True,
    )
    selected_mean = float(ranked[0]["selected_inner_balanced_accuracy_mean"])
    second_best_mean = float(ranked[1]["selected_inner_balanced_accuracy_mean"]) if len(ranked) > 1 else np.nan
    for rank, row in enumerate(ranked, start=1):
        row["selected_inner_rank"] = int(rank)
        row["selected_inner_second_best_balanced_accuracy_mean"] = second_best_mean
        row["selected_inner_winner_margin"] = selected_mean - second_best_mean if rank == 1 and np.isfinite(second_best_mean) else selected_mean - float(row["selected_inner_balanced_accuracy_mean"])
        row["selection_ensemble_requested_size"] = 1
        row["selection_ensemble_size"] = 1
        row["selection_ensemble_diversity"] = "none"
        row["selection_ensemble_score_normalization"] = cross_subject.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION
        row["selection_ensemble_weighting"] = cross_subject.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_WEIGHTING
        row["selection_ensemble_temperature"] = cross_subject.DEFAULT_CROSS_SUBJECT_SELECTION_ENSEMBLE_TEMPERATURE
        row["selected_candidate_indices"] = str(row["selected_candidate_index"])
        row["selected_ensemble_inner_balanced_accuracy_means"] = f"{row['selected_candidate_index']}:{float(row['selected_inner_balanced_accuracy_mean']):.6g}"
        row["selected_ensemble_weights"] = f"{row['selected_candidate_index']}:1"
        config = candidate_configs[int(row["selected_candidate_index"]) - 1]
        row["selected_ensemble_classifier_counts"] = _format_counter(Counter((config.classifier,)))
        row["selected_ensemble_window_center_counts"] = _format_counter(Counter((float(row["selected_window_center_s"]),)))
        row["selected_ensemble_feature_mode_counts"] = _format_counter(Counter((FULL_EPOCH_FEATURE_MODE,)))
        row["selected_ensemble_normalization_counts"] = _format_counter(Counter((config.normalization,)))
        row["selected_ensemble_alignment_counts"] = _format_counter(Counter(("none",)))
        row["selected_ensemble_projection_counts"] = _format_counter(Counter((config.projection,)))
        row["selected_ensemble_components_pca_counts"] = _format_counter(Counter((str(config.n_components),)))
    return ranked[0]


def _add_selected_candidate_fields(row, selected_row):
    for key, value in selected_row.items():
        row[key] = value


def _add_full_epoch_group_summary_fields(group_summary_rows, outer_rows):
    if not group_summary_rows or not outer_rows:
        return
    summary = group_summary_rows[0]
    summary["selected_projection_counts"] = _format_counter(Counter(str(row.get("selected_projection", row.get("projection", ""))) for row in outer_rows))
    summary["selected_time_bin_size_counts"] = _format_counter(Counter(str(row.get("selected_time_bin_size_s", row.get("time_bin_size_s", ""))) for row in outer_rows))
    summary["selected_time_window_counts"] = _format_counter(Counter(str(row.get("selected_time_window_s", row.get("time_window_s", ""))) for row in outer_rows))


def _one_hot_labels(labels):
    labels = np.asarray(labels, dtype=int).ravel()
    classes = np.unique(labels)
    class_to_column = {int(label): column for column, label in enumerate(classes.tolist())}
    targets = np.zeros((labels.shape[0], classes.shape[0]), dtype=float)
    for row, label in enumerate(labels.tolist()):
        targets[row, class_to_column[int(label)]] = 1.0
    return targets


def _training_labels(feature_set, *, label_shuffle_seed=None, label_shuffle_context=()):
    labels = np.asarray(feature_set.labels, dtype=int)
    if label_shuffle_seed is None:
        return labels
    seed_values = [int(label_shuffle_seed), *[int(value) for value in label_shuffle_context], int(feature_set.participant)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _requested_components(value):
    if value is None:
        return 64
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(value)
    if np.isinf(numeric):
        return np.iinfo(np.int32).max
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("n_components must be positive or inf.")
    return int(numeric)


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    return classifier_param


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
    return FullEpochLowRankConfig(
        time_window=_normalize_time_window(config.time_window),
        time_bin_size=_normalize_time_bin_size(config.time_bin_size),
        baseline_window=_normalize_time_window(config.baseline_window),
        normalization=_normalize_normalization(config.normalization),
        projection=_normalize_projection(config.projection),
        n_components=config.n_components,
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        max_trials_per_class_per_participant=_normalize_trial_cap(config.max_trials_per_class_per_participant),
        trial_selection=_normalize_trial_selection(config.trial_selection),
        trial_selection_seed=_normalize_trial_selection_seed(config.trial_selection_seed),
        chance_classes=int(config.chance_classes),
        random_state=config.random_state,
        signflip_permutations=int(config.signflip_permutations),
        signflip_seed=config.signflip_seed,
    )


def _normalize_time_window(value):
    if isinstance(value, str):
        if ":" in value:
            start_text, stop_text = value.split(":", maxsplit=1)
        elif ".." in value:
            start_text, stop_text = value.split("..", maxsplit=1)
        elif "," in value:
            start_text, stop_text = value.split(",", maxsplit=1)
        else:
            raise ValueError("Time window must have the form start:stop, start..stop, or start,stop.")
        value = (float(start_text), float(stop_text))
    if len(value) != 2:
        raise ValueError("Time window must contain exactly two values.")
    start, stop = float(value[0]), float(value[1])
    if start >= stop:
        raise ValueError("Time window start must be before stop.")
    return (start, stop)


def _normalize_time_bin_size(value):
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("time_bin_size must be a positive finite value.")
    return value


def _normalize_normalization(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in FULL_EPOCH_NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {FULL_EPOCH_NORMALIZATION_MODES}.")
    return normalized


def _normalize_projection(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in PROJECTION_MODES:
        raise ValueError(f"projection must be one of {PROJECTION_MODES}.")
    return normalized


def _normalize_trial_selection(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in cross_subject.TRIAL_SELECTION_MODES:
        raise ValueError(f"trial_selection must be one of {cross_subject.TRIAL_SELECTION_MODES}.")
    return normalized


def _normalize_trial_selection_seed(value):
    if value is None or value == "":
        return None
    value = int(value)
    if value < 0:
        raise ValueError("trial_selection_seed must be non-negative or None.")
    return value


def _normalize_trial_cap(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    return value


def _trial_zscore_features(features):
    features = np.asarray(features, dtype=float)
    mean = np.mean(features, axis=1, keepdims=True)
    std = _nonzero_std(np.std(features, axis=1, keepdims=True))
    return (features - mean) / std


def _nonzero_std(std):
    return np.where(std < 1e-12, 1.0, std)


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _format_counter(counter):
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _time_window_string(time_window):
    start, stop = _normalize_time_window(time_window)
    return f"{start:g}:{stop:g}"


def _seed_field(seed):
    return "" if seed is None else int(seed)


def _rows_with_consistent_fields(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _write_rows_if_present(rows, path):
    if path and rows:
        write_alpha_metrics_csv(_rows_with_consistent_fields(rows), path)


def _parse_time_window(value: str) -> tuple[float, float]:
    try:
        return _normalize_time_window(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_time_window_grid(value: str) -> tuple[tuple[float, float], ...]:
    values = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token and ".." not in token:
            raise argparse.ArgumentTypeError("Use start:stop pairs for --time-windows, e.g. 0.00:0.45,-0.05:0.60.")
        values.append(_parse_time_window(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one time window is required.")
    return tuple(values)


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_normalization_list(value: str) -> tuple[str, ...]:
    return tuple(_normalize_normalization(token) for token in _parse_token_list(value))


def _parse_projection_list(value: str) -> tuple[str, ...]:
    return tuple(_normalize_projection(token) for token in _parse_token_list(value))


def _parse_int_or_inf_list(value: str) -> tuple[int | float, ...]:
    values = tuple(parse_int_or_inf(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one component value is required.")
    return values


def _parse_classifier_param_grid(value: str) -> tuple[object, ...]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"default", "defaults"}:
            values.append(float("nan"))
        else:
            values.append(parse_classifier_param(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one classifier parameter value is required.")
    return tuple(values)


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run nested LOSO full-epoch supervised low-rank stimulus decoding using Part*Data.mat files only.",
    )
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=cross_subject.DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participant ids to evaluate in this run. Defaults to all participants.")
    parser.add_argument(
        "--time-windows",
        type=_parse_time_window_grid,
        default=DEFAULT_FULL_EPOCH_TIME_WINDOWS,
        help="Comma-separated full-epoch crop windows as start:stop pairs, e.g. 0.05:0.35,0.00:0.45,-0.05:0.60.",
    )
    parser.add_argument("--time-bin-size", type=float, default=DEFAULT_FULL_EPOCH_TIME_BIN_SIZE, help="Temporal bin width in seconds before flattening channels x time.")
    parser.add_argument("--baseline-window", type=_parse_time_window, default=cross_subject.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW, help="Baseline window as start,stop in seconds.")
    parser.add_argument("--normalizations", type=_parse_normalization_list, default=("subject_baseline_whiten",), help="Comma-separated subject normalization modes.")
    parser.add_argument("--projections", type=_parse_projection_list, default=(DEFAULT_FULL_EPOCH_PROJECTION,), help="Comma-separated projection modes: pls,pca,none.")
    parser.add_argument("--classifiers", type=_parse_token_list, default=(DEFAULT_FULL_EPOCH_CLASSIFIER,), help="Comma-separated classifier names.")
    parser.add_argument(
        "--classifier-params",
        type=_parse_classifier_param_grid,
        default=DEFAULT_FULL_EPOCH_CLASSIFIER_PARAMS,
        help="Comma-separated classifier parameters. Use default for each classifier default.",
    )
    parser.add_argument("--components-values", type=_parse_int_or_inf_list, default=DEFAULT_FULL_EPOCH_COMPONENTS, help="Comma-separated low-rank dimensions, or inf.")
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Optional deterministic cap on trials per stimulus class and participant.")
    parser.add_argument("--trial-selection", choices=cross_subject.TRIAL_SELECTION_MODES, default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION, help="Trial subset policy used when a trial cap is set.")
    parser.add_argument("--trial-selection-seed", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED, help="Seed for random trial selection.")
    parser.add_argument("--chance-classes", type=int, default=cross_subject.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES, help="Number of stimulus classes used for chance level.")
    parser.add_argument("--random-state", type=int, default=0, help="Random state passed to low-rank projections and classifiers.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Shuffle training labels within each participant for a nested null-control benchmark.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the nested label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Monte Carlo sign-flip permutations for the group summary.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Random seed for sign-flip permutations.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_outer.csv", help="Untouched outer participant score CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_group_summary.csv", help="Group summary CSV.")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_inner_validation.csv", help="Inner validation score CSV.")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_selected.csv", help="Selected hyperparameter CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_predictions.csv", help="Trial prediction CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_confusion.csv", help="Confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_per_stimulus.csv", help="Per-stimulus recall CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_confusion_pairs.csv", help="Bidirectional stimulus-pair confusion CSV.")
    return parser


def stimulus_cross_subject_full_epoch_lowrank(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if not participants:
        parser.error("At least one participant is required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    candidate_configs = make_full_epoch_lowrank_candidate_configs(
        time_windows=args.time_windows,
        time_bin_size=args.time_bin_size,
        baseline_window=args.baseline_window,
        normalizations=args.normalizations,
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
    artifacts = export_nested_full_epoch_lowrank_stimulus(
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
    return stimulus_cross_subject_full_epoch_lowrank(argv, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
