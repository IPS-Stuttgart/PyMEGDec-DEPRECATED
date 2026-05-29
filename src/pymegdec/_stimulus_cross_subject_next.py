# mypy: ignore-errors
"""Additional source-only and low-capacity calibration hooks for BUSH-MEG.

This module intentionally patches the existing composed cross-subject module in
the same style as ``_stimulus_cross_subject_core``.  It avoids touching the
legacy implementation while adding the experiment knobs that are most relevant
after the cue-alignment runs underperformed source-only decoding.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from itertools import product

import numpy as np
from pymegdec.classifiers import get_default_classifier_param, should_use_default_classifier_param, train_multiclass_classifier

DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING = "none"
SAMPLE_WEIGHTING_MODES = ("none", "subject_class_balanced")
DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION = "none"
SCORE_CALIBRATION_MODES = ("none", "inner_class_bias", "inner_class_affine")
DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA = 1.0
DEFAULT_SENSOR_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 70.0))
DEFAULT_SENSOR_TIME_PYRAMID_LEVELS = (1, 2, 4)
BASELINE_WHITENED_EXTENDED_FEATURE_MODES = ("sensor_time_pyramid",)
EXTENDED_FEATURE_MODES = ("sensor_logpower", "sensor_mean_logpower", "sensor_bandpower", "sensor_cov_tangent", "sensor_time_pyramid")
SCORE_CALIBRATION_L2 = 1e-3

_impl = None
_BaseConfig = None
_previous_normalized_config = None
_previous_make_candidate_configs = None
_previous_normalize_feature_mode = None
_previous_extract_window_features = None
_previous_baseline_feature_statistics = None
_previous_normalize_features = None
_previous_normalized_subject_features = None
_previous_fit_outer_fold_model = None
_previous_score_outer_fold_model = None
_previous_candidate_model_scores = None
_previous_align_training_features_by_subject = None
_previous_align_test_features_by_subject = None
_previous_prediction_rows = None
_previous_summarize_smoke = None
_previous_summarize_nested = None
_previous_rank_nested_candidates = None

CrossSubjectStimulusConfig = None


def install(impl) -> None:
    """Install next-method hooks into the composed cross-subject implementation."""

    global _impl, _BaseConfig, CrossSubjectStimulusConfig
    global _previous_normalized_config, _previous_make_candidate_configs, _previous_normalize_feature_mode
    global _previous_extract_window_features, _previous_baseline_feature_statistics, _previous_normalize_features, _previous_normalized_subject_features, _previous_fit_outer_fold_model
    global _previous_score_outer_fold_model, _previous_candidate_model_scores, _previous_align_training_features_by_subject
    global _previous_align_test_features_by_subject, _previous_prediction_rows, _previous_summarize_smoke
    global _previous_summarize_nested, _previous_rank_nested_candidates

    if getattr(impl, "_next_methods_installed", False):
        return

    _impl = impl
    _BaseConfig = impl.CrossSubjectStimulusConfig
    _previous_normalized_config = impl._normalized_config
    _previous_make_candidate_configs = impl.make_cross_subject_candidate_configs
    _previous_normalize_feature_mode = impl._normalize_feature_mode
    _previous_extract_window_features = impl._extract_window_features
    _previous_baseline_feature_statistics = impl._baseline_feature_statistics
    _previous_normalize_features = impl._normalize_features
    _previous_normalized_subject_features = impl._normalized_subject_features
    _previous_fit_outer_fold_model = impl._fit_outer_fold_model
    _previous_score_outer_fold_model = impl._score_outer_fold_model
    _previous_candidate_model_scores = impl._candidate_model_scores
    _previous_align_training_features_by_subject = impl._align_training_features_by_subject
    _previous_align_test_features_by_subject = impl._align_test_features_by_subject
    _previous_prediction_rows = impl._prediction_rows
    _previous_summarize_smoke = impl.summarize_cross_subject_stimulus_smoke
    _previous_summarize_nested = impl.summarize_nested_cross_subject_stimulus
    _previous_rank_nested_candidates = impl._rank_nested_candidates

    @dataclass(frozen=True)
    class NextCrossSubjectStimulusConfig(_BaseConfig):
        sample_weighting: str = DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING
        score_calibration: str = DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION
        alignment_alpha: float = DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA

    CrossSubjectStimulusConfig = NextCrossSubjectStimulusConfig

    impl.DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING = DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING
    impl.SAMPLE_WEIGHTING_MODES = SAMPLE_WEIGHTING_MODES
    impl.DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION = DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION
    impl.SCORE_CALIBRATION_MODES = SCORE_CALIBRATION_MODES
    impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA = DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA
    impl.EXTENDED_FEATURE_MODES = EXTENDED_FEATURE_MODES
    impl.DEFAULT_SENSOR_TIME_PYRAMID_LEVELS = DEFAULT_SENSOR_TIME_PYRAMID_LEVELS
    impl.FEATURE_MODES = tuple(dict.fromkeys((*impl.FEATURE_MODES, *EXTENDED_FEATURE_MODES)))
    impl.CrossSubjectStimulusConfig = NextCrossSubjectStimulusConfig

    impl._normalize_feature_mode = _normalize_feature_mode
    impl._normalized_config = _normalized_config
    impl.make_cross_subject_candidate_configs = make_cross_subject_candidate_configs
    impl._extract_window_features = _extract_window_features
    impl._baseline_feature_statistics = _baseline_feature_statistics
    impl._normalize_features = _normalize_features
    impl._normalized_subject_features = _normalized_subject_features
    impl._fit_outer_fold_model = _fit_outer_fold_model
    impl._score_outer_fold_model = _score_outer_fold_model
    impl._candidate_model_scores = _candidate_model_scores
    impl._apply_score_calibration = _apply_score_calibration
    impl._align_training_features_by_subject = _align_training_features_by_subject
    impl._align_test_features_by_subject = _align_test_features_by_subject
    impl._prediction_rows = _prediction_rows
    impl.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    impl.summarize_nested_cross_subject_stimulus = summarize_nested_cross_subject_stimulus
    impl._rank_nested_candidates = _rank_nested_candidates
    impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns(impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    impl._next_methods_installed = True


def _prediction_group_columns(columns):
    output = list(columns)
    for column in ("sample_weighting", "score_calibration", "alignment_alpha"):
        if column not in output:
            output.append(column)
    return tuple(output)


def _normalize_feature_mode(value):
    token = str(value).strip().lower().replace("-", "_")
    if token in EXTENDED_FEATURE_MODES:
        return token
    return _previous_normalize_feature_mode(value)


def _normalize_sample_weighting(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in SAMPLE_WEIGHTING_MODES:
        raise ValueError(f"sample_weighting must be one of {SAMPLE_WEIGHTING_MODES}.")
    return token


def _normalize_score_calibration(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in SCORE_CALIBRATION_MODES:
        raise ValueError(f"score_calibration must be one of {SCORE_CALIBRATION_MODES}.")
    return token


def _normalize_alignment_alpha(value):
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alignment_alpha must be in [0, 1].")
    return alpha


def _normalized_config(config):
    base = _previous_normalized_config(config)
    kwargs = {field.name: getattr(base, field.name) for field in fields(base)}
    kwargs["sample_weighting"] = _normalize_sample_weighting(getattr(config, "sample_weighting", DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING))
    kwargs["score_calibration"] = _normalize_score_calibration(getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION))
    kwargs["alignment_alpha"] = _normalize_alignment_alpha(getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA))
    return CrossSubjectStimulusConfig(**kwargs)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=None,
    window_size=None,
    baseline_window=None,
    feature_modes=None,
    normalizations=None,
    alignments=None,
    classifiers=None,
    classifier_params=(float("nan"),),
    components_pca_values=None,
    max_trials_per_class_per_participant=None,
    trial_selection=None,
    trial_selection_seed=None,
    sample_weightings=(DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING,),
    score_calibrations=(DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION,),
    alignment_alphas=(DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA,),
    chance_classes=None,
    random_state=0,
    signflip_permutations=10_000,
    signflip_seed=0,
):
    window_centers = _impl.DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS if window_centers is None else window_centers
    window_size = _impl.DEFAULT_CROSS_SUBJECT_WINDOW_SIZE if window_size is None else window_size
    baseline_window = _impl.DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW if baseline_window is None else baseline_window
    feature_modes = (_impl.DEFAULT_CROSS_SUBJECT_FEATURE_MODE,) if feature_modes is None else feature_modes
    normalizations = (_impl.DEFAULT_CROSS_SUBJECT_NORMALIZATION,) if normalizations is None else normalizations
    alignments = (_impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT,) if alignments is None else alignments
    classifiers = (_impl.DEFAULT_CROSS_SUBJECT_CLASSIFIER,) if classifiers is None else classifiers
    components_pca_values = (_impl.DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,) if components_pca_values is None else components_pca_values
    trial_selection = getattr(_impl, "DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION", "random") if trial_selection is None else trial_selection
    trial_selection_seed = getattr(_impl, "DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED", 0) if trial_selection_seed is None else trial_selection_seed
    chance_classes = _impl.DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES if chance_classes is None else chance_classes

    return tuple(
        CrossSubjectStimulusConfig(
            window_center=window_center,
            window_size=window_size,
            baseline_window=baseline_window,
            feature_mode=_normalize_feature_mode(feature_mode),
            normalization=normalization,
            alignment=alignment,
            classifier=classifier,
            classifier_param=classifier_param,
            components_pca=components_pca,
            max_trials_per_class_per_participant=max_trials_per_class_per_participant,
            trial_selection=trial_selection,
            trial_selection_seed=trial_selection_seed,
            sample_weighting=_normalize_sample_weighting(sample_weighting),
            score_calibration=_normalize_score_calibration(score_calibration),
            alignment_alpha=_normalize_alignment_alpha(alignment_alpha),
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, feature_mode, normalization, alignment, classifier, components_pca, sample_weighting, score_calibration, alignment_alpha in product(
            window_centers,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            _impl._components_pca_values_for_grid(components_pca_values),
            sample_weightings,
            score_calibrations,
            alignment_alphas,
        )
        for classifier_param in _impl._classifier_params_for_classifier(classifier, classifier_params)
    )


def _extract_window_features(data, time_window, *, feature_mode, trial_indices=None):
    feature_mode = _normalize_feature_mode(feature_mode)
    if feature_mode not in EXTENDED_FEATURE_MODES:
        return _previous_extract_window_features(data, time_window, feature_mode=feature_mode, trial_indices=trial_indices)

    time_vector = _impl._time_vector(data, 0)
    mask = _impl._time_mask(time_vector, time_window)
    window_time = time_vector[mask]
    features = []
    for trial_idx in _impl._iter_trial_indices(data, trial_indices):
        signal = _impl._trial_signal(data, trial_idx)[:, mask]
        if feature_mode == "sensor_logpower":
            feature = _sensor_logpower_feature(signal)
        elif feature_mode == "sensor_mean_logpower":
            feature = np.concatenate((np.mean(signal, axis=1), _sensor_logpower_feature(signal)))
        elif feature_mode == "sensor_bandpower":
            feature = _sensor_bandpower_feature(signal, window_time)
        elif feature_mode == "sensor_cov_tangent":
            feature = _sensor_cov_tangent_feature(signal)
        elif feature_mode == "sensor_time_pyramid":
            feature = _sensor_time_pyramid_feature(signal)
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    return np.vstack(features), int(np.sum(mask))


def _sensor_logpower_feature(window_signal):
    return np.log(np.mean(np.square(np.asarray(window_signal, dtype=float)), axis=1) + 1e-12)


def _sensor_bandpower_feature(window_signal, window_time):
    signal = np.asarray(window_signal, dtype=float)
    time = np.asarray(window_time, dtype=float).ravel()
    if signal.shape[1] < 2 or time.shape[0] < 2:
        return np.tile(_sensor_logpower_feature(signal), len(DEFAULT_SENSOR_BANDS))
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0 or not np.isfinite(dt):
        return np.tile(_sensor_logpower_feature(signal), len(DEFAULT_SENSOR_BANDS))
    centered = signal - np.mean(signal, axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(centered.shape[1], d=dt)
    spectrum = np.square(np.abs(np.fft.rfft(centered, axis=1)))
    band_features = []
    for low, high in DEFAULT_SENSOR_BANDS:
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            power = np.mean(spectrum[:, mask], axis=1)
        else:
            power = np.zeros(centered.shape[0], dtype=float)
        band_features.append(np.log(power + 1e-12))
    return np.concatenate(band_features)


def _sensor_time_pyramid_feature(window_signal, levels=DEFAULT_SENSOR_TIME_PYRAMID_LEVELS):
    """Concatenate per-sensor means over a short temporal pyramid.

    The 1/2/4-bin default gives seven channel blocks: one full-window mean,
    two half-window means, and four quarter-window means.  This keeps the
    feature width modest while preserving latency and waveform-shape evidence
    that a single mean discards.
    """

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    sample_indices = np.arange(signal.shape[1])
    pieces = []
    for level in levels:
        level = int(level)
        if level <= 0:
            raise ValueError("Temporal-pyramid levels must be positive.")
        for indices in np.array_split(sample_indices, level):
            pieces.append(np.mean(signal[:, indices], axis=1) if indices.size else np.zeros(signal.shape[0], dtype=float))
    return np.concatenate(pieces)


def _sensor_cov_tangent_feature(window_signal):
    signal = np.asarray(window_signal, dtype=float)
    n_channels = int(signal.shape[0])
    if signal.shape[1] < 2:
        covariance = np.eye(n_channels, dtype=float)
    else:
        covariance = np.cov(signal, rowvar=True)
        covariance = 0.5 * (covariance + covariance.T)
    trace = float(np.trace(covariance))
    target = (trace / max(n_channels, 1)) * np.eye(n_channels, dtype=float)
    covariance = 0.9 * covariance + 0.1 * target
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-6, 1e-12)
    log_covariance = (eigenvectors * np.log(np.maximum(eigenvalues, floor))) @ eigenvectors.T
    rows, cols = np.triu_indices(n_channels)
    feature = log_covariance[rows, cols]
    off_diag = rows != cols
    feature = feature.astype(float, copy=True)
    feature[off_diag] *= np.sqrt(2.0)
    return feature


def _baseline_feature_statistics(data, config, n_window_samples, trial_indices):
    if _normalize_feature_mode(config.feature_mode) in EXTENDED_FEATURE_MODES:
        baseline_features, n_baseline_samples = _extract_window_features(data, config.baseline_window, feature_mode=config.feature_mode, trial_indices=trial_indices)
        mean = np.mean(baseline_features, axis=0, keepdims=True)
        std = np.std(baseline_features, axis=0, keepdims=True)
        return mean, _impl._nonzero_std(std), n_baseline_samples
    return _previous_baseline_feature_statistics(data, config, n_window_samples, trial_indices)


def _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix):
    feature_mode = _normalize_feature_mode(config.feature_mode)
    if feature_mode in BASELINE_WHITENED_EXTENDED_FEATURE_MODES and config.normalization == "subject_baseline_whiten":
        if baseline_feature_mean is None or baseline_whitening_matrix is None:
            raise ValueError("sensor_time_pyramid requires baseline feature statistics and a whitening matrix for subject_baseline_whiten.")
        centered = np.asarray(features, dtype=float) - baseline_feature_mean
        return _impl._baseline_whiten_sensor_flat_features(centered, baseline_whitening_matrix)
    if feature_mode in EXTENDED_FEATURE_MODES and config.normalization == "subject_baseline_whiten":
        if baseline_feature_mean is None or baseline_feature_std is None:
            raise ValueError("Extended feature modes use baseline z-scoring when normalization='subject_baseline_whiten'.")
        return (np.asarray(features, dtype=float) - baseline_feature_mean) / baseline_feature_std
    return _previous_normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix)


def _normalized_subject_features(feature_set, config):
    feature_mode = _normalize_feature_mode(config.feature_mode)
    if feature_mode in BASELINE_WHITENED_EXTENDED_FEATURE_MODES and config.normalization == "subject_baseline_whiten":
        if feature_set.normalization == config.normalization:
            return feature_set.features
        if feature_set.baseline_feature_mean is None or feature_set.baseline_whitening_matrix is None:
            raise ValueError("sensor_time_pyramid requires baseline feature statistics and a whitening matrix for subject_baseline_whiten.")
        centered = np.asarray(feature_set.features, dtype=float) - feature_set.baseline_feature_mean
        return _impl._baseline_whiten_sensor_flat_features(centered, feature_set.baseline_whitening_matrix)
    if feature_mode in EXTENDED_FEATURE_MODES and config.normalization == "subject_baseline_whiten":
        if feature_set.normalization == config.normalization:
            return feature_set.features
        if feature_set.baseline_feature_mean is None or feature_set.baseline_feature_std is None:
            raise ValueError("Extended feature modes require baseline feature statistics for subject_baseline_whiten.")
        return (np.asarray(feature_set.features, dtype=float) - feature_set.baseline_feature_mean) / feature_set.baseline_feature_std
    return _previous_normalized_subject_features(feature_set, config)


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    aligned, metadata = _previous_align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config)
    alpha = _normalize_alignment_alpha(getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA))
    if getattr(config, "alignment", "none") != "none" and alpha < 1.0:
        aligned = [(1.0 - alpha) * np.asarray(raw, dtype=float) + alpha * np.asarray(full, dtype=float) for raw, full in zip(features_by_subject, aligned, strict=True)]
    if isinstance(metadata, dict):
        if "metadata" in metadata and isinstance(metadata["metadata"], dict):
            metadata["metadata"]["alignment_alpha"] = alpha
        else:
            metadata["alignment_alpha"] = alpha
    return aligned, metadata


def _align_test_features_by_subject(test_features, test_set, config, alignment_model):
    aligned, metadata = _previous_align_test_features_by_subject(test_features, test_set, config, alignment_model)
    alpha = _normalize_alignment_alpha(getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA))
    if getattr(config, "alignment", "none") != "none" and alpha < 1.0:
        aligned = (1.0 - alpha) * np.asarray(test_features, dtype=float) + alpha * np.asarray(aligned, dtype=float)
    if isinstance(metadata, dict):
        metadata["alignment_alpha"] = alpha
    return aligned, metadata


def _fit_outer_fold_model(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=(), fit_score_calibration=True):
    config = _normalized_config(config)
    train_features_by_subject = [_impl._normalized_subject_features(feature_set, config) for feature_set in train_sets]
    train_label_arrays = [
        _impl._training_labels(feature_set, label_shuffle_seed=label_shuffle_seed, label_shuffle_context=label_shuffle_context)
        for feature_set in train_sets
    ]
    train_features_by_subject, alignment_metadata = _align_training_features_by_subject(train_sets, train_features_by_subject, train_label_arrays, config)
    train_features = np.vstack(train_features_by_subject)
    train_labels_one_based = np.concatenate(train_label_arrays)
    train_labels = train_labels_one_based - 1
    sample_weight = _training_sample_weights(train_sets, train_label_arrays, config)
    feature_transform_metadata = None
    fit_training_feature_transform = getattr(_impl, "_fit_training_feature_transform", None)
    if fit_training_feature_transform is not None:
        train_features, feature_transform_metadata = fit_training_feature_transform(train_features, train_sets, config)
    train_window = _impl._centered_window(config.window_center, config.window_size)
    model_bundle = _impl.fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
            sample_weight=sample_weight,
        ),
        components_pca=config.components_pca,
        train_window=train_window,
    )
    fitted_model = {
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
        "sample_weighting": config.sample_weighting,
    }
    if feature_transform_metadata is not None:
        fitted_model["feature_transform_metadata"] = feature_transform_metadata
    if fit_score_calibration and config.score_calibration in {"inner_class_bias", "inner_class_affine"}:
        fitted_model["score_calibration_metadata"] = _fit_inner_score_calibration(
            train_sets,
            config,
            classifier_param,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
    else:
        fitted_model["score_calibration_metadata"] = {"mode": config.score_calibration}
    return fitted_model


def _training_sample_weights(train_sets, label_arrays, config):
    if _normalize_sample_weighting(getattr(config, "sample_weighting", DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING)) == "none":
        return None
    weights = []
    for labels in label_arrays:
        counts = Counter(np.asarray(labels, dtype=int).tolist())
        weights.extend(1.0 / max(counts[int(label)], 1) for label in labels)
    weights = np.asarray(weights, dtype=float)
    if weights.size and np.sum(weights) > 0.0:
        weights *= weights.size / np.sum(weights)
    return weights


def _fit_inner_score_calibration(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=()):
    mode = _normalize_score_calibration(getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION))
    if len(train_sets) < 3:
        return {"mode": mode, "status": "skipped_not_enough_source_subjects"}
    all_scores = []
    all_labels = []
    class_order = np.arange(int(config.chance_classes), dtype=int)
    inner_config = _config_with(config, score_calibration="none")
    for validation_index, validation_set in enumerate(train_sets):
        inner_train_sets = [feature_set for feature_set in train_sets if int(feature_set.participant) != int(validation_set.participant)]
        inner_model = _fit_outer_fold_model(
            inner_train_sets,
            inner_config,
            classifier_param,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=(*tuple(label_shuffle_context), int(validation_set.participant), validation_index),
            fit_score_calibration=False,
        )
        scores, score_classes = _previous_candidate_model_scores(inner_model, validation_set, inner_config)
        all_scores.append(_align_class_score_columns(scores, score_classes, class_order))
        all_labels.append(np.asarray(validation_set.labels, dtype=int) - 1)
    scores = np.vstack(all_scores)
    labels = np.concatenate(all_labels)
    if mode == "inner_class_affine":
        bias, scale, inner_balanced = _optimize_class_affine(scores, labels, class_order)
    else:
        bias, inner_balanced = _optimize_class_bias(scores, labels, class_order)
        scale = np.ones(class_order.shape[0], dtype=float)
    return {
        "mode": mode,
        "classes": class_order,
        "bias": bias,
        "scale": scale,
        "inner_balanced_accuracy": inner_balanced,
        "l2_penalty": SCORE_CALIBRATION_L2,
    }


def _config_with(config, **updates):
    kwargs = {field.name: getattr(config, field.name) for field in fields(config)}
    kwargs.update(updates)
    return CrossSubjectStimulusConfig(**kwargs)


def _optimize_class_bias(scores, labels, class_order):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    bias = np.zeros(class_order.shape[0], dtype=float)
    best = _bias_objective(scores, labels, class_order, bias)
    for step in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for column in range(bias.shape[0]):
                for direction in (1.0, -1.0):
                    candidate = bias.copy()
                    candidate[column] += direction * step
                    candidate -= np.mean(candidate)
                    value = _bias_objective(scores, labels, class_order, candidate)
                    if value > best + 1e-12:
                        bias = candidate
                        best = value
                        improved = True
    return bias, _balanced_accuracy_for_scores(scores + bias[None, :], labels, class_order)


def _optimize_class_affine(scores, labels, class_order):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    bias, _inner_balanced = _optimize_class_bias(scores, labels, class_order)
    log_scale = np.zeros(class_order.shape[0], dtype=float)
    best = _affine_objective(scores, labels, class_order, bias, log_scale)
    for step in (0.5, 0.25, 0.1, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for column in range(class_order.shape[0]):
                for parameter in ("bias", "log_scale"):
                    for direction in (1.0, -1.0):
                        candidate_bias = bias.copy()
                        candidate_log_scale = log_scale.copy()
                        if parameter == "bias":
                            candidate_bias[column] += direction * step
                            candidate_bias -= np.mean(candidate_bias)
                        else:
                            candidate_log_scale[column] += direction * step
                            candidate_log_scale -= np.mean(candidate_log_scale)
                            candidate_log_scale = np.clip(candidate_log_scale, -1.5, 1.5)
                        value = _affine_objective(scores, labels, class_order, candidate_bias, candidate_log_scale)
                        if value > best + 1e-12:
                            bias = candidate_bias
                            log_scale = candidate_log_scale
                            best = value
                            improved = True
    scale = np.exp(log_scale)
    calibrated = scores * scale[None, :] + bias[None, :]
    return bias, scale, _balanced_accuracy_for_scores(calibrated, labels, class_order)


def _bias_objective(scores, labels, class_order, bias):
    balanced = _balanced_accuracy_for_scores(scores + bias[None, :], labels, class_order)
    return balanced - SCORE_CALIBRATION_L2 * float(np.mean(np.square(bias)))


def _affine_objective(scores, labels, class_order, bias, log_scale):
    scale = np.exp(np.asarray(log_scale, dtype=float))
    calibrated = np.asarray(scores, dtype=float) * scale[None, :] + np.asarray(bias, dtype=float)[None, :]
    balanced = _balanced_accuracy_for_scores(calibrated, labels, class_order)
    penalty = float(np.mean(np.square(bias)) + np.mean(np.square(log_scale)))
    return balanced - SCORE_CALIBRATION_L2 * penalty


def _balanced_accuracy_for_scores(scores, labels, class_order):
    predictions = np.asarray(class_order, dtype=int)[np.argmax(scores, axis=1)]
    return float(_impl.balanced_accuracy_score(labels, predictions))


def _align_class_score_columns(scores, score_classes, class_order):
    scores = np.asarray(scores, dtype=float)
    score_classes = np.asarray(score_classes, dtype=int).ravel()
    class_order = np.asarray(class_order, dtype=int).ravel()
    if scores.ndim != 2:
        return np.zeros((0, class_order.shape[0]), dtype=float)
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return np.zeros((scores.shape[0], class_order.shape[0]), dtype=float)
    aligned = np.zeros((scores.shape[0], class_order.shape[0]), dtype=float)
    finite_min = np.nanmin(np.where(np.isfinite(scores), scores, np.nan), axis=1)
    finite_min = np.where(np.isfinite(finite_min), finite_min - 1.0, -1.0)
    aligned[:] = finite_min[:, None]
    class_to_column = {int(label): column for column, label in enumerate(class_order.tolist())}
    for source_column, label in enumerate(score_classes.tolist()):
        target_column = class_to_column.get(int(label))
        if target_column is not None:
            aligned[:, target_column] = scores[:, source_column]
    return aligned


def _candidate_model_scores(fitted_model, test_set, config):
    scores, classes = _previous_candidate_model_scores(fitted_model, test_set, config)
    return _apply_score_calibration(scores, classes, fitted_model)


def _apply_score_calibration(scores, classes, fitted_model):
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    if metadata.get("mode") not in {"inner_class_bias", "inner_class_affine"} or "bias" not in metadata:
        return scores, classes
    calibration_classes = np.asarray(metadata["classes"], dtype=int)
    bias = np.asarray(metadata["bias"], dtype=float)
    scale = np.asarray(metadata.get("scale", np.ones_like(bias)), dtype=float)
    aligned = _align_class_score_columns(scores, classes, calibration_classes)
    return aligned * scale[None, :] + bias[None, :], calibration_classes


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    config = _normalized_config(config)
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    if metadata.get("mode") not in {"inner_class_bias", "inner_class_affine"} or "bias" not in metadata:
        outer_row, prediction_rows = _previous_score_outer_fold_model(fitted_model, test_set, config, include_predictions=include_predictions)
        _add_next_fields(outer_row, config, fitted_model)
        for row in prediction_rows:
            _add_next_fields(row, config, fitted_model)
        return outer_row, prediction_rows

    outer_row, _unused = _previous_score_outer_fold_model(fitted_model, test_set, config, include_predictions=False)
    test_features = _impl._normalized_subject_features(test_set, config)
    alignment_model = _impl._fitted_alignment_model(fitted_model) if hasattr(_impl, "_fitted_alignment_model") else {"metadata": fitted_model.get("alignment_metadata", {})}
    test_features, test_alignment_metadata = _align_test_features_by_subject(test_features, test_set, config, alignment_model)
    class_scores, score_classes = _impl._model_class_scores(fitted_model["model_bundle"], test_features)
    class_scores, score_classes = _apply_score_calibration(class_scores, score_classes, fitted_model)
    test_labels = np.asarray(test_set.labels, dtype=int) - 1
    predictions = np.asarray(score_classes, dtype=int)[np.argmax(class_scores, axis=1)]
    rank_metrics = _impl._ranked_label_metrics(test_labels, class_scores, score_classes)
    accuracy = float(_impl.accuracy_score(test_labels, predictions))
    balanced_accuracy = float(_impl.balanced_accuracy_score(test_labels, predictions))
    outer_row.update(
        {
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
            "alignment_test_transform": test_alignment_metadata.get("test_transform", ""),
            "alignment_target_centering": test_alignment_metadata.get("target_centering", ""),
        }
    )
    _add_next_fields(outer_row, config, fitted_model)
    prediction_rows = []
    if include_predictions:
        prediction_rows = _prediction_rows(
            test_set,
            test_labels,
            predictions,
            rank_metrics["true_label_ranks"],
            config=config,
            actual_components_pca=fitted_model["model_bundle"].actual_components_pca,
        )
        for row in prediction_rows:
            _add_next_fields(row, config, fitted_model)
    return outer_row, prediction_rows


def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):
    rows = _previous_prediction_rows(test_set, test_labels, predictions, true_label_ranks, config=config, actual_components_pca=actual_components_pca)
    for row in rows:
        _add_config_fields(row, config)
    return rows


def _add_config_fields(row, config):
    row["sample_weighting"] = getattr(config, "sample_weighting", DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING)
    row["score_calibration"] = getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION)
    row["alignment_alpha"] = getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA)


def _add_next_fields(row, config, fitted_model):
    _add_config_fields(row, config)
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    row["score_calibration"] = metadata.get("mode", getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION))
    row["score_calibration_inner_balanced_accuracy"] = metadata.get("inner_balanced_accuracy", "")
    row["score_calibration_status"] = metadata.get("status", "")


def _rank_nested_candidates(inner_rows, *, selection_metric=None):
    if selection_metric is None:
        ranked = _previous_rank_nested_candidates(inner_rows)
    else:
        ranked = _previous_rank_nested_candidates(inner_rows, selection_metric=selection_metric)
    examples = {int(row["candidate_index"]): row for row in inner_rows}
    for row in ranked:
        example = examples.get(int(row["selected_candidate_index"]), {})
        row["selected_sample_weighting"] = example.get("sample_weighting", DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING)
        row["selected_score_calibration"] = example.get("score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION)
        row["selected_alignment_alpha"] = example.get("alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA)
    return ranked


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    rows = _previous_summarize_smoke(outer_rows, config=config)
    config = _normalized_config(config or CrossSubjectStimulusConfig())
    for row in rows:
        _add_config_fields(row, config)
    return rows


def summarize_nested_cross_subject_stimulus(outer_rows, *, signflip_permutations=10_000, signflip_seed=0):
    rows = _previous_summarize_nested(outer_rows, signflip_permutations=signflip_permutations, signflip_seed=signflip_seed)
    if not outer_rows:
        return rows
    for row in rows:
        row["selected_sample_weighting_counts"] = _impl._format_counter(Counter(str(value.get("selected_sample_weighting", value.get("sample_weighting", ""))) for value in outer_rows))
        row["selected_score_calibration_counts"] = _impl._format_counter(Counter(str(value.get("selected_score_calibration", value.get("score_calibration", ""))) for value in outer_rows))
        row["selected_alignment_alpha_counts"] = _impl._format_counter(Counter(str(value.get("selected_alignment_alpha", value.get("alignment_alpha", ""))) for value in outer_rows))
    return rows


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    return classifier_param
