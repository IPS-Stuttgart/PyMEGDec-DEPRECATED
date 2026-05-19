"""Composed cross-subject stimulus decoding implementation.

The historical implementation remains in ``_stimulus_cross_subject_legacy``.
This module installs the result-changing scoring and target-alignment behavior
inside that implementation module, so the public API no longer depends on
package ``__init__`` side effects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from pymegdec import _stimulus_cross_subject_legacy as _impl

DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"
DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0
TRIAL_SELECTION_MODES = ("random", "first")
AUTO_CLASSIFIER_PARAM_GRID_TOKEN = "auto-grid"
AUTO_COMPONENTS_PCA_GRID_TOKEN = "auto-grid"
COMPONENTS_PCA_AUTO_GRID = (32, 64, 128)
CLASSIFIER_AUTO_PARAM_GRIDS = {
    "gaussian-naive-bayes": (1e-12, 1e-9, 1e-6),
    "multiclass-svm": (0.1, 1.0, 10.0),
    "multiclass-svm-weighted": (0.1, 1.0, 10.0),
    "multinomial-logistic": (0.1, 1.0, 10.0),
    "multinomial-logistic-weighted": (0.1, 1.0, 10.0),
    "regularized-qda": (0.25, 0.5, 0.75),
    "shrinkage-lda": ("auto", 0.1, 0.5, 0.9),
    "shrinkage-prototype": (0.0, 0.25, 0.5, 0.75),
}

_BASE_CROSS_SUBJECT_CONFIG = _impl.CrossSubjectStimulusConfig
_BASE_PARTICIPANT_FEATURE_SET = _impl.ParticipantFeatureSet
_ORIGINAL_SCORE_OUTER_FOLD_MODEL = _impl._score_outer_fold_model
_ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE = _impl.summarize_cross_subject_stimulus_smoke


@dataclass(frozen=True)
class CrossSubjectStimulusConfig(_BASE_CROSS_SUBJECT_CONFIG):
    """Cross-subject stimulus config with reproducible trial-cap sampling.

    ``train_class_procrustes`` applies only train-derived alignment parameters
    to held-out subjects; scored target trials are not used for centering.
    """

    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED


@dataclass(frozen=True)
class ParticipantFeatureSet(_BASE_PARTICIPANT_FEATURE_SET):
    """Windowed features with original trial-index bookkeeping."""

    trial_indices: np.ndarray | None = None
    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION
    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED


def _ranked_label_metrics(true_labels, class_scores, score_classes):
    """Return rank metrics without dropping unscoreable true-label trials."""

    true_label_ranks = _impl._true_label_ranks(true_labels, class_scores, score_classes)
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


def _alignment_model(alignment, *, common_classes, aligned_participants, transforms=(), target_transform=None):
    return {
        "metadata": _impl._alignment_metadata(
            alignment,
            common_classes=common_classes,
            aligned_participants=aligned_participants,
        ),
        "transforms": tuple(transforms),
        "target_transform": target_transform,
    }


def _group_average_channel_procrustes_transform(transforms):
    transforms = tuple(transforms)
    if not transforms:
        return None

    rotations = np.stack([np.asarray(transform["rotation"], dtype=float) for transform in transforms], axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    left, _singular_values, right_t = np.linalg.svd(mean_rotation, full_matrices=False)
    rotation = left @ right_t
    return {
        "source_center": np.mean(
            np.stack([np.asarray(transform["source_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "target_center": np.mean(
            np.stack([np.asarray(transform["target_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "rotation": rotation,
    }


def _fitted_alignment_model(fitted_model):
    alignment_metadata = fitted_model.get("alignment_metadata", {})
    if isinstance(alignment_metadata, dict) and "metadata" in alignment_metadata:
        return alignment_metadata
    return {
        "metadata": alignment_metadata,
        "transforms": tuple(),
        "target_transform": None,
    }


def _train_only_channel_procrustes_transform(target_transform):
    return {
        "source_center": np.asarray(target_transform["source_center"], dtype=float),
        "target_center": np.asarray(target_transform["target_center"], dtype=float),
        "rotation": np.asarray(target_transform["rotation"], dtype=float),
    }


def _test_alignment_metadata(test_transform, target_centering):
    return {"test_transform": test_transform, "target_centering": target_centering}


def _align_test_features_by_subject(test_features, test_set, config, alignment_model):
    if config.alignment == "none":
        return test_features, _test_alignment_metadata("none", "none")
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    target_transform = alignment_model.get("target_transform")
    if target_transform is None:
        return test_features, _test_alignment_metadata("none", "none")

    # Use only train-derived parameters. Re-centering with ``test_features``
    # would make the held-out subject's scored feature distribution influence
    # evaluation, which is a transductive target-alignment step rather than a
    # strict LOSO test.
    test_transform = _train_only_channel_procrustes_transform(target_transform)
    return (
        _impl._apply_channel_procrustes_transform(test_features, test_set, test_transform),
        _test_alignment_metadata("group_average_train_transform", "train_only_group_average"),
    )


def _prediction_group_columns_with_alignment():
    columns = tuple(_impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    additions = ("alignment_test_transform", "alignment_target_centering")
    if all(column in columns for column in additions):
        return columns
    output = []
    for column in columns:
        output.append(column)
        if column == "alignment":
            output.extend(addition for addition in additions if addition not in output)
    return tuple(output)


def _prediction_group_columns_with_trial_selection(columns):
    output = list(columns)
    for column in ("trial_selection", "trial_selection_seed"):
        if column not in output:
            output.append(column)
    return tuple(output)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=_impl.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
    window_size=_impl.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    baseline_window=_impl.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    feature_modes=(_impl.DEFAULT_CROSS_SUBJECT_FEATURE_MODE,),
    normalizations=(_impl.DEFAULT_CROSS_SUBJECT_NORMALIZATION,),
    alignments=(_impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT,),
    classifiers=(_impl.DEFAULT_CROSS_SUBJECT_CLASSIFIER,),
    classifier_params=(float("nan"),),
    components_pca_values=(_impl.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,),
    max_trials_per_class_per_participant=None,
    trial_selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    trial_selection_seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    chance_classes=_impl.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
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
            trial_selection=trial_selection,
            trial_selection_seed=trial_selection_seed,
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, components_pca in _impl.product(
            window_centers,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            _components_pca_values_for_grid(components_pca_values),
        )
        for classifier_param in _classifier_params_for_classifier(classifier, classifier_params)
    )


def _components_pca_values_for_grid(components_pca_values):
    values: list[object] = []
    for components_pca in components_pca_values:
        if _is_auto_components_pca_grid(components_pca):
            values.extend(COMPONENTS_PCA_AUTO_GRID)
        else:
            values.append(components_pca)
    return tuple(_dedupe_classifier_params(values))


def _is_auto_components_pca_grid(value):
    return isinstance(value, str) and value.strip().lower().replace("_", "-") == AUTO_COMPONENTS_PCA_GRID_TOKEN


def _classifier_params_for_classifier(classifier, classifier_params):
    """Expand classifier-specific parameter grids while preserving explicit values."""

    params: list[object] = []
    for classifier_param in classifier_params:
        if _is_auto_classifier_param_grid(classifier_param):
            params.extend(CLASSIFIER_AUTO_PARAM_GRIDS.get(str(classifier), (float("nan"),)))
        else:
            params.append(classifier_param)
    return tuple(_dedupe_classifier_params(params))


def _is_auto_classifier_param_grid(value):
    return isinstance(value, str) and value.strip().lower().replace("_", "-") == AUTO_CLASSIFIER_PARAM_GRID_TOKEN


def _dedupe_classifier_params(params):
    seen = set()
    for param in params:
        key = _classifier_param_dedupe_key(param)
        if key in seen:
            continue
        seen.add(key)
        yield param


def _classifier_param_dedupe_key(param):
    if isinstance(param, float) and np.isnan(param):
        return ("nan",)
    if isinstance(param, np.generic):
        param = param.item()
    try:
        hash(param)
    except TypeError:
        return ("repr", repr(param))
    return (type(param).__name__, param)


def load_participant_stimulus_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract fixed-window features."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_path = _impl.Path(_impl.resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = _impl.sio.loadmat(data_path)["data"][0]
    all_labels = _impl._trialinfo_labels(data)
    trial_indices = _selected_trial_indices(
        all_labels,
        config.max_trials_per_class_per_participant,
        selection=config.trial_selection,
        seed=config.trial_selection_seed,
        participant=participant,
    )
    labels = all_labels[trial_indices]
    features, n_window_samples = _impl._extract_window_features(
        data,
        _impl._centered_window(config.window_center, config.window_size),
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = _impl._baseline_feature_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = _impl._baseline_channel_whitening_matrix(
            data,
            config.baseline_window,
            trial_indices,
        )
    normalized_features = _impl._normalize_features(
        features,
        config,
        baseline_feature_mean,
        baseline_feature_std,
        baseline_whitening_matrix,
    )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=normalized_features,
        normalization=config.normalization,
        baseline_features=None,
        baseline_feature_mean=baseline_feature_mean,
        baseline_feature_std=baseline_feature_std,
        baseline_whitening_matrix=baseline_whitening_matrix,
        n_channels=int(_impl._trial_signal(data, 0).shape[0]),
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
        trial_indices=np.asarray(trial_indices, dtype=int),
        trial_selection=config.trial_selection,
        trial_selection_seed=config.trial_selection_seed,
    )


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize held-out participant scores and include trial-selection metadata."""

    rows = _ORIGINAL_SUMMARIZE_CROSS_SUBJECT_STIMULUS_SMOKE(outer_rows, config=config)
    config = _normalized_config(config or CrossSubjectStimulusConfig())
    for row in rows:
        row["trial_selection"] = config.trial_selection
        row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    return rows


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    if config.alignment == "none":
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=(),
            aligned_participants=(),
        )
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    common_classes = _impl._common_label_values(labels_by_subject)
    if len(common_classes) < 2:
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=common_classes,
            aligned_participants=(),
        )

    class_patterns = [
        _impl._participant_class_channel_patterns(features, labels, feature_set, common_classes)
        for feature_set, features, labels in zip(feature_sets, features_by_subject, labels_by_subject, strict=True)
    ]
    transforms = _impl._fit_channel_procrustes_transforms(class_patterns)
    aligned_features = [
        _impl._apply_channel_procrustes_transform(features, feature_set, transform)
        for feature_set, features, transform in zip(feature_sets, features_by_subject, transforms, strict=True)
    ]
    return aligned_features, _alignment_model(
        config.alignment,
        common_classes=common_classes,
        aligned_participants=(feature_set.participant for feature_set in feature_sets),
        transforms=transforms,
        target_transform=_group_average_channel_procrustes_transform(transforms),
    )


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    alignment_model = _fitted_alignment_model(fitted_model)
    test_features = _impl._normalized_subject_features(test_set, config)
    test_features, test_alignment_metadata = _align_test_features_by_subject(
        test_features,
        test_set,
        config,
        alignment_model,
    )
    scoring_set = replace(test_set, features=test_features, normalization=config.normalization)
    scoring_model = dict(fitted_model)
    scoring_model["alignment_metadata"] = alignment_model["metadata"]
    outer_row, prediction_rows = _ORIGINAL_SCORE_OUTER_FOLD_MODEL(
        scoring_model,
        scoring_set,
        config,
        include_predictions=include_predictions,
    )
    outer_row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
    outer_row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
    outer_row["trial_selection"] = config.trial_selection
    outer_row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    for row in prediction_rows:
        row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
        row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
        row["trial_selection"] = config.trial_selection
        row["trial_selection_seed"] = _seed_field(config.trial_selection_seed)
    return outer_row, prediction_rows


def _candidate_model_scores(fitted_model, test_set, config):
    alignment_model = _fitted_alignment_model(fitted_model)
    test_features = _impl._normalized_subject_features(test_set, config)
    test_features, _test_alignment_metadata = _align_test_features_by_subject(
        test_features,
        test_set,
        config,
        alignment_model,
    )
    return _impl._model_class_scores(fitted_model["model_bundle"], test_features)


def _feature_cache_key(config):
    return (
        float(config.window_center),
        float(config.window_size),
        float(config.baseline_window[0]),
        float(config.baseline_window[1]),
        str(config.feature_mode),
        str(config.normalization),
        config.max_trials_per_class_per_participant,
        str(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        _seed_field(getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)),
    )


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    train_window = _impl._centered_window(config.window_center, config.window_size)
    trial_indices = _feature_set_trial_indices(test_set)
    rows = []
    for trial_idx, true_label, predicted_label, true_label_rank in zip(trial_indices, test_labels, predictions, true_label_ranks, strict=True):
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
                "trial_selection": config.trial_selection,
                "trial_selection_seed": _seed_field(config.trial_selection_seed),
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


def _selected_trial_indices(
    labels,
    max_trials_per_class,
    *,
    selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    participant=None,
):
    labels = np.asarray(labels).ravel()
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    max_trials_per_class = int(max_trials_per_class)
    if max_trials_per_class <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    selection = _normalize_trial_selection(selection)

    if selection == "first":
        selected = []
        counts: Counter[int] = Counter()
        for index, label in enumerate(labels):
            if counts[int(label)] < max_trials_per_class:
                selected.append(index)
                counts[int(label)] += 1
        return np.asarray(selected, dtype=int)

    rng = _trial_selection_rng(seed, participant)
    selected = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size > max_trials_per_class:
            class_indices = rng.choice(class_indices, size=max_trials_per_class, replace=False)
        selected.extend(int(index) for index in class_indices)
    return np.asarray(sorted(selected), dtype=int)


def _trial_selection_rng(seed, participant):
    if seed is None:
        return np.random.default_rng()
    seed_values = [int(seed)]
    if participant is not None:
        seed_values.append(int(participant))
    return np.random.default_rng(np.random.SeedSequence(seed_values))


def _feature_set_trial_indices(feature_set):
    trial_indices = getattr(feature_set, "trial_indices", None)
    if trial_indices is None:
        return np.arange(np.asarray(feature_set.labels).shape[0], dtype=int)
    return np.asarray(trial_indices, dtype=int).ravel()


def _seed_field(seed):
    return "" if seed is None else int(seed)


def _normalized_config(config):
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=_impl._normalize_feature_mode(config.feature_mode),
        normalization=_impl._normalize_normalization(config.normalization),
        alignment=_impl._normalize_alignment(config.alignment),
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=_impl._normalize_trial_cap(config.max_trials_per_class_per_participant),
        trial_selection=_normalize_trial_selection(getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)),
        trial_selection_seed=_normalize_trial_selection_seed(getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)),
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalize_trial_selection(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TRIAL_SELECTION_MODES:
        raise ValueError(f"trial_selection must be one of {TRIAL_SELECTION_MODES}.")
    return normalized


def _normalize_trial_selection_seed(value):
    if value is None or value == "":
        return None
    value = int(value)
    if value < 0:
        raise ValueError("trial_selection_seed must be non-negative or None.")
    return value


def _install_module_fixes():
    _impl.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION  # type: ignore[attr-defined]
    _impl.DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED  # type: ignore[attr-defined]
    _impl.TRIAL_SELECTION_MODES = TRIAL_SELECTION_MODES  # type: ignore[attr-defined]
    _impl.AUTO_CLASSIFIER_PARAM_GRID_TOKEN = AUTO_CLASSIFIER_PARAM_GRID_TOKEN  # type: ignore[attr-defined]
    _impl.AUTO_COMPONENTS_PCA_GRID_TOKEN = AUTO_COMPONENTS_PCA_GRID_TOKEN  # type: ignore[attr-defined]
    _impl.CLASSIFIER_AUTO_PARAM_GRIDS = CLASSIFIER_AUTO_PARAM_GRIDS  # type: ignore[attr-defined]
    _impl.COMPONENTS_PCA_AUTO_GRID = COMPONENTS_PCA_AUTO_GRID  # type: ignore[attr-defined]
    _impl.CrossSubjectStimulusConfig = CrossSubjectStimulusConfig  # type: ignore[misc]
    _impl.ParticipantFeatureSet = ParticipantFeatureSet  # type: ignore[misc]
    _impl.make_cross_subject_candidate_configs = make_cross_subject_candidate_configs
    _impl._classifier_params_for_classifier = _classifier_params_for_classifier  # type: ignore[attr-defined]
    _impl._is_auto_classifier_param_grid = _is_auto_classifier_param_grid  # type: ignore[attr-defined]
    _impl._components_pca_values_for_grid = _components_pca_values_for_grid  # type: ignore[attr-defined]
    _impl._is_auto_components_pca_grid = _is_auto_components_pca_grid  # type: ignore[attr-defined]
    _impl.load_participant_stimulus_features = load_participant_stimulus_features
    _impl.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    _impl._ranked_label_metrics = _ranked_label_metrics
    _impl._align_training_features_by_subject = _align_training_features_by_subject
    _impl._align_test_features_by_subject = _align_test_features_by_subject  # type: ignore[attr-defined]
    _impl._score_outer_fold_model = _score_outer_fold_model
    _impl._candidate_model_scores = _candidate_model_scores
    _impl._feature_cache_key = _feature_cache_key
    _impl._prediction_rows = _prediction_rows
    _impl._selected_trial_indices = _selected_trial_indices
    _impl._feature_set_trial_indices = _feature_set_trial_indices  # type: ignore[attr-defined]
    _impl._seed_field = _seed_field  # type: ignore[attr-defined]
    _impl._normalized_config = _normalized_config
    _impl._normalize_trial_selection = _normalize_trial_selection  # type: ignore[attr-defined]
    _impl._normalize_trial_selection_seed = _normalize_trial_selection_seed  # type: ignore[attr-defined]
    _impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns_with_trial_selection(_prediction_group_columns_with_alignment())


_install_module_fixes()

globals().update(_impl.__dict__)
