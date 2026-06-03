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
from math import isfinite

import numpy as np
from pymegdec.classifiers import get_default_classifier_param, should_use_default_classifier_param, train_multiclass_classifier

DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING = "none"
SAMPLE_WEIGHTING_MODES = ("none", "subject_class_balanced")
DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION = "none"
DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM = "none"
FEATURE_TRANSFORM_MODES = ("none", "source_anova_scale", "source_anova_sqrt_scale")
SOURCE_ANOVA_FEATURE_TRANSFORM_MODES = frozenset(
    ("source_anova_scale", "source_anova_sqrt_scale")
)
SOURCE_ANOVA_FEATURE_TRANSFORM_POWERS = {
    "source_anova_scale": 1.0,
    "source_anova_sqrt_scale": 0.5,
}
SOURCE_ANOVA_FEATURE_TRANSFORM_MIN_WEIGHT = 0.25
SOURCE_ANOVA_FEATURE_TRANSFORM_MAX_WEIGHT = 4.0
SOURCE_ANOVA_FEATURE_TRANSFORM_EPSILON = 1e-12
SOURCE_ANOVA_FEATURE_TRANSFORM_NORMALIZER_FALLBACK = 1.0
INNER_SCORE_CALIBRATION_MODES = frozenset(
    (
        "inner_class_bias",
        "inner_class_affine",
        "inner_rank_bias",
        "inner_probability_map",
        "inner_rank_probability_map",
        "inner_confusion_blend",
        "inner_margin_confusion_blend",
        "inner_rank_confusion_blend",
        "inner_rank_margin_confusion_blend",
    )
)
GUARDED_INNER_SCORE_CALIBRATION_MODES = frozenset(
    f"{mode}_guarded" for mode in INNER_SCORE_CALIBRATION_MODES
)
TRAIN_SCORE_CALIBRATION_MODES = frozenset(
    ("train_class_bias", "train_class_affine", "train_rank_bias")
)
ACTIVE_SCORE_CALIBRATION_MODES = (
    INNER_SCORE_CALIBRATION_MODES
    | GUARDED_INNER_SCORE_CALIBRATION_MODES
    | TRAIN_SCORE_CALIBRATION_MODES
)
SCORE_CALIBRATION_MODES = (
    "none",
    "inner_class_bias",
    "inner_class_bias_guarded",
    "inner_class_affine",
    "inner_class_affine_guarded",
    "inner_rank_bias",
    "inner_rank_bias_guarded",
    "inner_probability_map",
    "inner_probability_map_guarded",
    "inner_rank_probability_map",
    "inner_rank_probability_map_guarded",
    "inner_confusion_blend",
    "inner_confusion_blend_guarded",
    "inner_margin_confusion_blend",
    "inner_margin_confusion_blend_guarded",
    "inner_rank_confusion_blend",
    "inner_rank_confusion_blend_guarded",
    "inner_rank_margin_confusion_blend",
    "inner_rank_margin_confusion_blend_guarded",
    "train_class_bias",
    "train_class_affine",
    "train_rank_bias",
)
DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA = 1.0
DEFAULT_SENSOR_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 70.0))
DEFAULT_SENSOR_TIME_PYRAMID_LEVELS = (1, 2, 4)
DEFAULT_SENSOR_DCT_COEFFICIENTS = 8
DEFAULT_SENSOR_FLAT_SMOOTH_KERNEL = (0.25, 0.50, 0.25)
DEFAULT_SENSOR_FLAT_TAPER_FLOOR = 0.25
DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_FLOOR = 0.25
DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_SIGMA = 0.50
SENSOR_FLAT_TIME_BIN_FEATURE_MODES = {
    "sensor_flat_time_bins3": 3,
    "sensor_flat_time_bins5": 5,
    "sensor_flat_time_bins7": 7,
}
BASELINE_WHITENED_EXTENDED_FEATURE_MODES = (
    "sensor_flat_logpower",
    "sensor_flat_smooth",
    "sensor_flat_taper",
    "sensor_flat_gaussian_taper",
    "sensor_flat_centered",
    "sensor_flat_delta",
    "sensor_flat_dct",
    "sensor_flat_time_pyramid",
    "sensor_flat_time_pyramid_logpower",
    "sensor_flat_time_pyramid_delta",
    *tuple(SENSOR_FLAT_TIME_BIN_FEATURE_MODES),
    "sensor_dct",
    "sensor_time_pyramid",
    "sensor_time_pyramid_logpower",
    "sensor_time_pyramid_delta",
    "sensor_time_pyramid_delta_logpower",
)
EXTENDED_FEATURE_MODES = (
    "sensor_logpower",
    "sensor_mean_logpower",
    "sensor_flat_logpower",
    "sensor_flat_smooth",
    "sensor_flat_taper",
    "sensor_flat_gaussian_taper",
    "sensor_flat_centered",
    "sensor_flat_delta",
    "sensor_flat_dct",
    "sensor_flat_time_pyramid",
    "sensor_flat_time_pyramid_logpower",
    "sensor_flat_time_pyramid_delta",
    *tuple(SENSOR_FLAT_TIME_BIN_FEATURE_MODES),
    "sensor_dct",
    "sensor_bandpower",
    "sensor_cov_tangent",
    "sensor_time_pyramid",
    "sensor_time_pyramid_logpower",
    "sensor_time_pyramid_delta",
    "sensor_time_pyramid_delta_logpower",
)
SCORE_CALIBRATION_L2 = 1e-3
SCORE_CALIBRATION_MIN_INNER_GAIN = 1e-12
SCORE_CALIBRATION_PROBABILITY_MAP_L2 = 1e-2
SCORE_CALIBRATION_PROBABILITY_MAP_IDENTITY_BLEND = 0.20
SOFT_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    "rank_softmax_inner_confusion_soft": "rank_softmax",
    "rank_softmax_t1_25_inner_confusion_soft": "rank_softmax_t1_25",
    "rank_softmax_t1_5_inner_confusion_soft": "rank_softmax_t1_5",
    "rank_softmax_t1_75_inner_confusion_soft": "rank_softmax_t1_75",
    "rank_softmax_t2_inner_confusion_soft": "rank_softmax_t2",
    "rank_softmax_t3_inner_confusion_soft": "rank_softmax_t3",
    "rank_reciprocal_inner_confusion_soft": "rank_reciprocal",
    "rank_borda_inner_confusion_soft": "rank_borda",
    "rank_margin_blend_inner_confusion_soft": "rank_margin_blend",
    "rank_top2_vote_inner_confusion_soft": "rank_top2_vote",
    "rank_top3_vote_inner_confusion_soft": "rank_top3_vote",
}
TOPK_BORDA_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    # The best BUSH-MEG source-only runs have strong top-2/top-3 signal but weak
    # top-1 separation. Truncated Borda pooling keeps only the near-top classes,
    # while the existing soft guarded inner-confusion correction can re-rank
    # systematic source-validated confusions without forcing a hard quota.
    "rank_top2_borda_inner_confusion_soft": "rank_top2_borda",
    "rank_top3_borda_inner_confusion_soft": "rank_top3_borda",
}
SOFT_GUARDED_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    f"{mode}_guarded": base
    for mode, base in {
        **SOFT_INNER_CONFUSION_SCORE_NORMALIZATION_BASES,
        **TOPK_BORDA_INNER_CONFUSION_SCORE_NORMALIZATION_BASES,
    }.items()
}
SOFT_INNER_BALANCED_CONFUSION_SCORE_NORMALIZATION_BASES = {
    "rank_softmax_inner_balanced_confusion_soft": "rank_softmax",
}
CONFUSION_CALIBRATION_SMOOTHING = 1.0
CONFUSION_CALIBRATION_BLEND_GRID = tuple(
    float(value) for value in np.linspace(0.0, 1.0, 11)
)
CONFUSION_CALIBRATION_MARGIN_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES = {
    # Sub-unit temperatures make the per-candidate rank posterior more
    # top-heavy than the legacy t=1.0 rank_softmax. This is useful for the
    # strong 175 ms / 150 ms BUSH-MEG window where top-1 is now the bottleneck.
    "rank_softmax_t0_5": 0.50,
    "rank_softmax_t0_75": 0.75,
    "rank_softmax_t1_25": 1.25,
    "rank_softmax_t1_5": 1.50,
    "rank_softmax_t1_75": 1.75,
}
INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_BASES = {
    f"{mode}_inner_confusion_soft": mode
    for mode in INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES
}
INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_MARGIN_BASES = {
    f"{mode}_inner_confusion_margin_soft": mode
    for mode in INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES
}
INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_BASES = {
    f"{mode}_inner_balanced": mode
    for mode in INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES
}
INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_CONFUSION_SOFT_BASES = {
    f"{mode}_inner_balanced_confusion_soft": mode
    for mode in INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES
}
INTERMEDIATE_RANK_SOFTMAX_LOG_POOL_MODES = {
    f"{mode}_log_pool": mode
    for mode in INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES
}
INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_BALANCED_CONFUSION_BASES = {
    f"{mode}_guarded": base
    for mode, base in INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_CONFUSION_SOFT_BASES.items()
}
INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_CONFUSION_BASES = {
    f"{mode}_guarded": base
    for mode, base in {
        **INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_BASES,
        **INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_MARGIN_BASES,
    }.items()
}
EXPERIMENTAL_GUARDED_QUOTA_BASE_MODES = (
    # The 175 ms / 150 ms BUSH-MEG source-only run has strong top-2/top-3
    # signal, but its top-1 prediction counts can still collapse toward a few
    # classes. These modes keep the same leakage-safe score path, then apply the
    # existing guarded balanced-quota assignment using only the unlabeled test
    # batch size and the protocol-level balanced-class design.
    "rank_softmax",
    "rank_softmax_t1_25",
    "rank_softmax_t1_5",
    "rank_softmax_t1_75",
    "rank_softmax_inner_balanced",
    "rank_softmax_t1_5_inner_balanced",
    "rank_softmax_inner_confusion_soft_guarded",
    "rank_softmax_t1_5_inner_confusion_soft_guarded",
    "rank_softmax_inner_balanced_confusion_soft_guarded",
    "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded",
    "rank_margin_blend_inner_confusion_margin_soft_guarded",
)
EXPERIMENTAL_GUARDED_QUOTA_SCORE_NORMALIZATIONS = tuple(
    f"{mode}_guarded_balanced_quota" for mode in EXPERIMENTAL_GUARDED_QUOTA_BASE_MODES
)
GUARDED_TEST_PRIOR_BALANCE_SUFFIX = "_guarded_test_prior_balance"
GUARDED_TEST_PRIOR_BALANCE_MARGIN_QUANTILE = 0.50
GUARDED_TEST_PRIOR_BALANCE_BASE_MODES = (
    "rank_softmax",
    "rank_softmax_t0_75",
    "rank_softmax_t1_25",
    "rank_softmax_t1_5",
    "rank_softmax_t1_75",
    "rank_top3_margin_blend",
)
GUARDED_TEST_PRIOR_BALANCE_SCORE_NORMALIZATIONS = tuple(
    f"{mode}{GUARDED_TEST_PRIOR_BALANCE_SUFFIX}"
    for mode in GUARDED_TEST_PRIOR_BALANCE_BASE_MODES
)
TOPK_BORDA_SCORE_NORMALIZATIONS = {
    "rank_top2_borda": 2,
    "rank_top3_borda": 3,
}
TOPK_SCORE_SOFTMAX_SCORE_NORMALIZATIONS = {
    "rank_top2_score_softmax": 2,
    "rank_top3_score_softmax": 3,
}
TOPK_MARGIN_BLEND_SCORE_NORMALIZATIONS = {
    # BUSH-MEG w150 runs now have robust top-2/top-3 signal, but top-1 is
    # still the bottleneck. The tuple is: (truncated Borda k, sharp rank-softmax
    # temperature).
    "rank_top2_margin_blend": (2, 0.75),
    "rank_top3_margin_blend": (3, 0.75),
}
TOPK_SCORE_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    f"{mode}_inner_confusion_soft_guarded": mode
    for mode in TOPK_SCORE_SOFTMAX_SCORE_NORMALIZATIONS
}
TOPK_MARGIN_BLEND_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    f"{mode}_inner_confusion_soft_guarded": mode
    for mode in TOPK_MARGIN_BLEND_SCORE_NORMALIZATIONS
}
ADAPTIVE_RANK_SOFTMAX_MODE = "rank_adaptive_softmax"
ADAPTIVE_RANK_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES = {
    f"{ADAPTIVE_RANK_SOFTMAX_MODE}_inner_confusion_soft_guarded": ADAPTIVE_RANK_SOFTMAX_MODE,
}
ADAPTIVE_RANK_SOFTMAX_LOW_TEMPERATURE = 0.75
ADAPTIVE_RANK_SOFTMAX_HIGH_TEMPERATURE = 1.75
ADAPTIVE_RANK_SOFTMAX_MARGIN_LOW = 0.25
ADAPTIVE_RANK_SOFTMAX_MARGIN_HIGH = 1.25
WORST_CLASS_SELECTION_METRICS = (
    "balanced_worst_class",
    "balanced_worst_class_lcb",
)
WORST_CLASS_SELECTION_WEIGHT = 0.25

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
_previous_class_score_probabilities = None
_previous_candidate_model_scores = None
_previous_align_training_features_by_subject = None
_previous_align_test_features_by_subject = None
_previous_prediction_rows = None
_previous_summarize_smoke = None
_previous_summarize_nested = None
_previous_rank_nested_candidates = None
_previous_without_test_prior_balance_suffix = None
_previous_test_class_prior_balance_mode = None
_previous_test_class_prior_balance_metadata = None
_previous_add_test_class_prior_balance_fields = None

CrossSubjectStimulusConfig = None


def install(impl) -> None:
    """Install next-method hooks into the composed cross-subject implementation."""

    global _impl, _BaseConfig, CrossSubjectStimulusConfig
    global _previous_normalized_config, _previous_make_candidate_configs, _previous_normalize_feature_mode
    global _previous_extract_window_features, _previous_baseline_feature_statistics, _previous_normalize_features, _previous_normalized_subject_features, _previous_fit_outer_fold_model
    global _previous_score_outer_fold_model, _previous_candidate_model_scores, _previous_align_training_features_by_subject
    global _previous_align_test_features_by_subject, _previous_prediction_rows, _previous_summarize_smoke
    global _previous_summarize_nested, _previous_rank_nested_candidates, _previous_class_score_probabilities
    global _previous_without_test_prior_balance_suffix, _previous_test_class_prior_balance_mode
    global _previous_test_class_prior_balance_metadata, _previous_add_test_class_prior_balance_fields

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
    _previous_class_score_probabilities = impl._class_score_probabilities
    _previous_candidate_model_scores = impl._candidate_model_scores
    _previous_align_training_features_by_subject = impl._align_training_features_by_subject
    _previous_align_test_features_by_subject = impl._align_test_features_by_subject
    _previous_prediction_rows = impl._prediction_rows
    _previous_summarize_smoke = impl.summarize_cross_subject_stimulus_smoke
    _previous_summarize_nested = impl.summarize_nested_cross_subject_stimulus
    _previous_rank_nested_candidates = impl._rank_nested_candidates
    _previous_without_test_prior_balance_suffix = impl._without_test_prior_balance_suffix
    _previous_test_class_prior_balance_mode = impl._test_class_prior_balance_mode
    _previous_test_class_prior_balance_metadata = impl._test_class_prior_balance_metadata
    _previous_add_test_class_prior_balance_fields = impl._add_test_class_prior_balance_fields

    @dataclass(frozen=True)
    class NextCrossSubjectStimulusConfig(_BaseConfig):
        sample_weighting: str = DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING
        score_calibration: str = DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION
        feature_transform: str = DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM
        alignment_alpha: float = DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA

    CrossSubjectStimulusConfig = NextCrossSubjectStimulusConfig

    impl.DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING = DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING
    impl.SAMPLE_WEIGHTING_MODES = SAMPLE_WEIGHTING_MODES
    impl.DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION = DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION
    impl.SCORE_CALIBRATION_MODES = SCORE_CALIBRATION_MODES
    impl.GUARDED_INNER_SCORE_CALIBRATION_MODES = GUARDED_INNER_SCORE_CALIBRATION_MODES
    impl.DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM = DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM
    impl.FEATURE_TRANSFORM_MODES = FEATURE_TRANSFORM_MODES
    impl.SCORE_CALIBRATION_MIN_INNER_GAIN = SCORE_CALIBRATION_MIN_INNER_GAIN
    impl.DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA = DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA
    impl.EXTENDED_FEATURE_MODES = EXTENDED_FEATURE_MODES
    impl.DEFAULT_SENSOR_TIME_PYRAMID_LEVELS = DEFAULT_SENSOR_TIME_PYRAMID_LEVELS
    impl.DEFAULT_SENSOR_DCT_COEFFICIENTS = DEFAULT_SENSOR_DCT_COEFFICIENTS
    impl.FEATURE_MODES = tuple(dict.fromkeys((*impl.FEATURE_MODES, *EXTENDED_FEATURE_MODES)))
    impl.CROSS_SUBJECT_SELECTION_METRIC_CHOICES = tuple(
        dict.fromkeys(
            (*impl.CROSS_SUBJECT_SELECTION_METRIC_CHOICES, *WORST_CLASS_SELECTION_METRICS)
        )
    )
    impl.CrossSubjectStimulusConfig = NextCrossSubjectStimulusConfig
    _install_intermediate_rank_softmax_temperatures(impl)
    _install_soft_inner_confusion_score_normalizations(impl)
    _install_guarded_quota_score_normalizations(impl)
    _install_guarded_test_prior_balance_score_normalizations(impl)
    _install_topk_borda_score_normalizations(impl)
    _install_topk_score_softmax_score_normalizations(impl)
    _install_topk_margin_blend_score_normalizations(impl)
    _install_adaptive_rank_softmax_score_normalization(impl)

    impl._normalize_feature_mode = _normalize_feature_mode
    impl._normalized_config = _normalized_config
    impl.make_cross_subject_candidate_configs = make_cross_subject_candidate_configs
    impl._extract_window_features = _extract_window_features
    impl._sensor_flat_gaussian_taper_weights = _sensor_flat_gaussian_taper_weights
    impl._baseline_feature_statistics = _baseline_feature_statistics
    impl._normalize_features = _normalize_features
    impl._normalized_subject_features = _normalized_subject_features
    impl._fit_outer_fold_model = _fit_outer_fold_model
    impl._fit_training_feature_transform = _fit_training_feature_transform
    impl._apply_training_feature_transform = _apply_training_feature_transform
    impl._score_outer_fold_model = _score_outer_fold_model
    impl._without_test_prior_balance_suffix = _without_test_prior_balance_suffix
    impl._test_class_prior_balance_mode = _test_class_prior_balance_mode
    impl._test_class_prior_balance_metadata = _test_class_prior_balance_metadata
    impl._add_test_class_prior_balance_fields = _add_test_class_prior_balance_fields
    impl._guarded_test_prior_balance_probabilities = _guarded_test_prior_balance_probabilities
    impl._class_score_probabilities = _class_score_probabilities
    impl._candidate_model_scores = _candidate_model_scores
    impl._apply_score_calibration = _apply_score_calibration
    impl._score_calibration_base_mode = _score_calibration_base_mode
    impl._guard_inner_score_calibration_metadata = _guard_inner_score_calibration_metadata
    impl._align_training_features_by_subject = _align_training_features_by_subject
    impl._align_test_features_by_subject = _align_test_features_by_subject
    impl._prediction_rows = _prediction_rows
    impl.summarize_cross_subject_stimulus_smoke = summarize_cross_subject_stimulus_smoke
    impl.summarize_nested_cross_subject_stimulus = summarize_nested_cross_subject_stimulus
    impl._rank_nested_candidates = _rank_nested_candidates
    impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns(impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    impl._next_methods_installed = True


def _install_intermediate_rank_softmax_temperatures(impl) -> None:
    """Expose rank-softmax temperatures between the legacy t=1/t=2/t=3 modes."""

    impl.RANK_SOFTMAX_TEMPERATURES.update(INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES)
    impl.INNER_BALANCED_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_BASES
    )
    impl.INNER_BALANCED_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_CONFUSION_SOFT_BASES
    )
    impl.INNER_BALANCED_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_BALANCED_CONFUSION_BASES
    )
    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *INTERMEDIATE_RANK_SOFTMAX_TEMPERATURES,
                *INTERMEDIATE_RANK_SOFTMAX_LOG_POOL_MODES,
                *INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_BASES,
                *INTERMEDIATE_RANK_SOFTMAX_INNER_BALANCED_CONFUSION_SOFT_BASES,
                *INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_BALANCED_CONFUSION_BASES,
            )
        )
    )


def _install_soft_inner_confusion_score_normalizations(impl) -> None:
    """Expose conservative inner-confusion score reranking variants."""

    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        SOFT_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        SOFT_GUARDED_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        TOPK_BORDA_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        TOPK_SCORE_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_MARGIN_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_CONFUSION_BASES
    )
    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        ADAPTIVE_RANK_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.INNER_BALANCED_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        SOFT_INNER_BALANCED_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    extra_modes = (
        *tuple(SOFT_INNER_CONFUSION_SCORE_NORMALIZATION_BASES),
        *tuple(TOPK_BORDA_INNER_CONFUSION_SCORE_NORMALIZATION_BASES),
        *tuple(SOFT_GUARDED_INNER_CONFUSION_SCORE_NORMALIZATION_BASES),
        *tuple(TOPK_SCORE_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES),
        *tuple(INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_BASES),
        *tuple(INTERMEDIATE_RANK_SOFTMAX_INNER_CONFUSION_MARGIN_BASES),
        *tuple(INTERMEDIATE_RANK_SOFTMAX_GUARDED_INNER_CONFUSION_BASES),
        *tuple(ADAPTIVE_RANK_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES),
        *tuple(SOFT_INNER_BALANCED_CONFUSION_SCORE_NORMALIZATION_BASES),
    )
    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys((*impl.ENSEMBLE_SCORE_NORMALIZATION_MODES, *extra_modes))
    )


def _install_guarded_quota_score_normalizations(impl) -> None:
    """Expose guarded balanced-quota wrappers for the w150 source-only branch."""

    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *EXPERIMENTAL_GUARDED_QUOTA_SCORE_NORMALIZATIONS,
            )
        )
    )


def _install_guarded_test_prior_balance_score_normalizations(impl) -> None:
    """Expose margin-gated unlabeled test-prior balancing modes."""

    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *GUARDED_TEST_PRIOR_BALANCE_SCORE_NORMALIZATIONS,
            )
        )
    )


def _install_topk_borda_score_normalizations(impl) -> None:
    """Expose truncated-Borda rank pooling for source-only score ensembles."""

    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *TOPK_BORDA_SCORE_NORMALIZATIONS,
            )
        )
    )


def _install_topk_score_softmax_score_normalizations(impl) -> None:
    """Expose top-k score-softmax rank pooling for source-only ensembles."""

    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *TOPK_SCORE_SOFTMAX_SCORE_NORMALIZATIONS,
            )
        )
    )


def _install_topk_margin_blend_score_normalizations(impl) -> None:
    """Expose confidence-gated top-k Borda/rank-softmax pooling modes."""

    impl.INNER_CONFUSION_ENSEMBLE_SCORE_NORMALIZATION_BASES.update(
        TOPK_MARGIN_BLEND_INNER_CONFUSION_SCORE_NORMALIZATION_BASES
    )
    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                *TOPK_MARGIN_BLEND_SCORE_NORMALIZATIONS,
                *TOPK_MARGIN_BLEND_INNER_CONFUSION_SCORE_NORMALIZATION_BASES,
            )
        )
    )


def _install_adaptive_rank_softmax_score_normalization(impl) -> None:
    """Expose a margin-adaptive rank-softmax normalizer for w150 BUSH runs."""

    impl.ENSEMBLE_SCORE_NORMALIZATION_MODES = tuple(
        dict.fromkeys(
            (
                *impl.ENSEMBLE_SCORE_NORMALIZATION_MODES,
                ADAPTIVE_RANK_SOFTMAX_MODE,
                *ADAPTIVE_RANK_SOFTMAX_INNER_CONFUSION_SCORE_NORMALIZATION_BASES,
            )
        )
    )


def _without_test_prior_balance_suffix(score_normalization):
    score_normalization = str(score_normalization).strip()
    if score_normalization.endswith(GUARDED_TEST_PRIOR_BALANCE_SUFFIX):
        return score_normalization.removesuffix(GUARDED_TEST_PRIOR_BALANCE_SUFFIX)
    return _previous_without_test_prior_balance_suffix(score_normalization)


def _test_class_prior_balance_mode(score_normalization):
    normalized = str(score_normalization).strip().lower().replace("-", "_")
    if (
        normalized in _impl.ENSEMBLE_SCORE_NORMALIZATION_MODES
        and normalized.endswith(GUARDED_TEST_PRIOR_BALANCE_SUFFIX)
    ):
        return normalized
    return _previous_test_class_prior_balance_mode(score_normalization)


def _test_class_prior_balance_metadata(probabilities, class_order, score_normalization):
    """Return metadata for legacy or guarded held-out test-prior balancing."""

    mode = _test_class_prior_balance_mode(score_normalization)
    if mode is None or not str(mode).endswith(GUARDED_TEST_PRIOR_BALANCE_SUFFIX):
        return _previous_test_class_prior_balance_metadata(
            probabilities,
            class_order,
            score_normalization,
        )

    full_adjusted, target_mass, iterations, status = _impl._test_class_prior_balanced_probabilities(probabilities)
    guarded_adjusted, adjusted_trials, margin_threshold = _guarded_test_prior_balance_probabilities(
        probabilities,
        full_adjusted,
    )
    guarded_status = f"{status}_guarded" if str(status).startswith("applied") else status
    class_order = np.asarray(class_order, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float)
    observed_mass = (
        np.sum(probabilities, axis=0)
        if probabilities.ndim == 2
        else np.asarray((), dtype=float)
    )
    adjusted_mass = (
        np.sum(guarded_adjusted, axis=0)
        if guarded_adjusted.ndim == 2
        else np.asarray((), dtype=float)
    )
    return {
        "mode": mode,
        "status": guarded_status,
        "iterations": iterations,
        "class_order": class_order,
        "target_mass": target_mass,
        "observed_mass": observed_mass,
        "adjusted_mass": adjusted_mass,
        "probabilities": guarded_adjusted,
        "guarded": True,
        "guarded_margin_quantile": GUARDED_TEST_PRIOR_BALANCE_MARGIN_QUANTILE,
        "guarded_margin_threshold": margin_threshold,
        "guarded_adjusted_trials": adjusted_trials,
    }


def _guarded_test_prior_balance_probabilities(probabilities, balanced_probabilities):
    """Apply balanced-prior probabilities only to low-margin held-out rows."""

    probabilities = np.asarray(probabilities, dtype=float)
    balanced_probabilities = np.asarray(balanced_probabilities, dtype=float)
    if (
        probabilities.ndim != 2
        or balanced_probabilities.shape != probabilities.shape
        or probabilities.shape[0] == 0
        or probabilities.shape[1] == 0
    ):
        return probabilities, 0, ""

    margins = _probability_margins(probabilities)
    finite_margins = margins[np.isfinite(margins)]
    if finite_margins.size == 0:
        return balanced_probabilities, int(probabilities.shape[0]), ""
    threshold = float(
        np.quantile(
            finite_margins,
            float(GUARDED_TEST_PRIOR_BALANCE_MARGIN_QUANTILE),
        )
    )
    low_margin_rows = margins <= threshold
    adjusted = probabilities.copy()
    adjusted[low_margin_rows] = balanced_probabilities[low_margin_rows]
    row_sums = np.sum(adjusted, axis=1, keepdims=True)
    adjusted = np.divide(
        adjusted,
        row_sums,
        out=np.full_like(adjusted, 1.0 / adjusted.shape[1]),
        where=row_sums > 0.0,
    )
    return adjusted, int(np.sum(low_margin_rows)), threshold


def _probability_margins(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return np.zeros(probabilities.shape[0] if probabilities.ndim == 2 else 0, dtype=float)
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    return ordered[:, 0] - ordered[:, 1]


def _add_test_class_prior_balance_fields(row, metadata):
    _previous_add_test_class_prior_balance_fields(row, metadata)
    row["ensemble_test_class_prior_guarded"] = metadata.get("guarded", "")
    row["ensemble_test_class_prior_guarded_margin_quantile"] = metadata.get(
        "guarded_margin_quantile",
        "",
    )
    row["ensemble_test_class_prior_guarded_margin_threshold"] = metadata.get(
        "guarded_margin_threshold",
        "",
    )
    row["ensemble_test_class_prior_guarded_adjusted_trials"] = metadata.get(
        "guarded_adjusted_trials",
        "",
    )


def _class_score_probabilities(scores, *, score_normalization=None):
    """Return class probabilities, adding BUSH-focused rank normalizers."""

    if score_normalization is None:
        score_normalization = _impl.DEFAULT_CROSS_SUBJECT_ENSEMBLE_SCORE_NORMALIZATION
    base_mode = _impl._base_ensemble_score_normalization(score_normalization)
    if base_mode == ADAPTIVE_RANK_SOFTMAX_MODE:
        return _rank_adaptive_softmax_probabilities(scores)
    top_k = TOPK_BORDA_SCORE_NORMALIZATIONS.get(base_mode)
    if top_k is not None:
        return _rank_topk_borda_probabilities(scores, top_k=top_k)
    top_k = TOPK_SCORE_SOFTMAX_SCORE_NORMALIZATIONS.get(base_mode)
    if top_k is not None:
        return _rank_topk_score_softmax_probabilities(scores, top_k=top_k)
    topk_margin_blend = TOPK_MARGIN_BLEND_SCORE_NORMALIZATIONS.get(base_mode)
    if topk_margin_blend is not None:
        top_k, sharp_temperature = topk_margin_blend
        return _rank_topk_margin_blend_probabilities(
            scores,
            top_k=top_k,
            sharp_temperature=sharp_temperature,
        )
    return _previous_class_score_probabilities(
        scores,
        score_normalization=score_normalization,
    )


def _rank_adaptive_softmax_probabilities(scores):
    """Convert ranks to probabilities with a per-trial adaptive temperature."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(
            "Nested score ensembling requires a non-empty two-dimensional "
            "class-score matrix."
        )

    probabilities = np.empty_like(scores, dtype=float)
    low_temperature = float(ADAPTIVE_RANK_SOFTMAX_LOW_TEMPERATURE)
    high_temperature = float(ADAPTIVE_RANK_SOFTMAX_HIGH_TEMPERATURE)
    if low_temperature <= 0.0 or high_temperature <= 0.0:
        raise ValueError("Adaptive rank-softmax temperatures must be positive.")

    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(
                row.shape[0],
                1.0 / row.shape[0],
                dtype=float,
            )
            continue

        confidence = _adaptive_rank_softmax_confidence(row)
        temperature = high_temperature - confidence * (
            high_temperature - low_temperature
        )
        rank_scores = np.where(finite, row, -np.inf)
        descending_columns = np.argsort(-rank_scores, kind="mergesort")
        ranks = np.empty(row.shape[0], dtype=float)
        ranks[descending_columns] = np.arange(row.shape[0], dtype=float)
        logits = -ranks / float(temperature)
        logits[~finite] = -50.0
        exp_logits = np.exp(logits - np.max(logits))
        probabilities[row_index] = exp_logits / np.sum(exp_logits)
    return probabilities


def _adaptive_rank_softmax_confidence(row):
    """Map z-scored top-1/top-2 score separation to a 0..1 confidence."""

    row = np.asarray(row, dtype=float)
    finite = np.isfinite(row)
    if int(np.sum(finite)) < 2:
        return 1.0
    finite_scores = row[finite]
    centered = finite_scores - np.mean(finite_scores)
    scale = float(np.std(centered))
    if scale > 1e-12:
        centered = centered / scale
    ordered = np.sort(centered)[::-1]
    margin = float(ordered[0] - ordered[1])
    low = float(ADAPTIVE_RANK_SOFTMAX_MARGIN_LOW)
    high = float(ADAPTIVE_RANK_SOFTMAX_MARGIN_HIGH)
    return float(np.clip((margin - low) / max(high - low, 1e-12), 0.0, 1.0))


def _rank_topk_borda_probabilities(scores, *, top_k):
    """Convert class scores to a tapered top-k Borda distribution."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(
            "Nested score ensembling requires a non-empty two-dimensional "
            "class-score matrix."
        )
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be at least one.")

    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        n_finite = int(np.sum(finite))
        if n_finite == 0:
            probabilities[row_index] = np.full(
                row.shape[0],
                1.0 / row.shape[0],
                dtype=float,
            )
            continue
        k = min(top_k, n_finite)
        ordered_columns = np.argsort(
            -np.where(finite, row, -np.inf),
            kind="mergesort",
        )[:k]
        weights = np.zeros(row.shape[0], dtype=float)
        weights[ordered_columns] = np.arange(k, 0, -1, dtype=float)
        probabilities[row_index] = weights / np.sum(weights)
    return probabilities


def _rank_topk_score_softmax_probabilities(scores, *, top_k):
    """Convert class scores to a sparse softmax over only the top-k classes."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(
            "Nested score ensembling requires a non-empty two-dimensional "
            "class-score matrix."
        )
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be at least one.")

    probabilities = np.zeros_like(scores, dtype=float)
    n_classes = int(scores.shape[1])
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        n_finite = int(np.sum(finite))
        if n_finite == 0:
            probabilities[row_index] = np.full(n_classes, 1.0 / n_classes, dtype=float)
            continue

        k = min(top_k, n_finite)
        ordered_columns = np.argsort(
            -np.where(finite, row, -np.inf),
            kind="mergesort",
        )[:k]
        logits = np.asarray(row[ordered_columns], dtype=float)
        logits = logits - float(np.mean(logits))
        scale = float(np.std(logits))
        if isfinite(scale) and scale > 1e-12:
            logits = logits / scale
        logits = logits - float(np.max(logits))
        exp_logits = np.exp(logits)
        total = float(np.sum(exp_logits))
        if not isfinite(total) or total <= 0.0:
            probabilities[row_index, ordered_columns] = 1.0 / k
        else:
            probabilities[row_index, ordered_columns] = exp_logits / total
    return probabilities


def _rank_topk_margin_blend_probabilities(scores, *, top_k, sharp_temperature):
    """Blend sharp rank-softmax with truncated top-k Borda by score margin."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(
            "Nested score ensembling requires a non-empty two-dimensional "
            "class-score matrix."
        )
    sharp = _impl._rank_softmax_probabilities(
        scores,
        temperature=float(sharp_temperature),
    )
    soft = _rank_topk_borda_probabilities(scores, top_k=top_k)
    confidence = np.asarray(_impl._rank_margin_blend_confidence(scores), dtype=float)[
        :, None
    ]
    confidence = np.clip(confidence, 0.0, 1.0)
    blended = confidence * sharp + (1.0 - confidence) * soft
    row_sums = np.sum(blended, axis=1, keepdims=True)
    return np.divide(
        blended,
        row_sums,
        out=np.full_like(blended, 1.0 / blended.shape[1]),
        where=row_sums > 0.0,
    )


def _prediction_group_columns(columns):
    output = list(columns)
    for column in (
        "sample_weighting",
        "score_calibration",
        "feature_transform",
        "alignment_alpha",
    ):
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


def _score_calibration_base_mode(value):
    token = _normalize_score_calibration(value)
    if token in GUARDED_INNER_SCORE_CALIBRATION_MODES:
        return token.removesuffix("_guarded")
    return token


def _normalize_feature_transform(value):
    token = str(value).strip().lower().replace("-", "_")
    if token not in FEATURE_TRANSFORM_MODES:
        raise ValueError(f"feature_transform must be one of {FEATURE_TRANSFORM_MODES}.")
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
    kwargs["feature_transform"] = _normalize_feature_transform(getattr(config, "feature_transform", DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM))
    kwargs["alignment_alpha"] = _normalize_alignment_alpha(getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA))
    return CrossSubjectStimulusConfig(**kwargs)


def make_cross_subject_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    window_centers=None,
    window_size=None,
    window_sizes=None,
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
    feature_transforms=(DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM,),
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

    if window_sizes is None:
        window_sizes = (window_size,)
    window_sizes = tuple(float(value) for value in window_sizes)

    return tuple(
        CrossSubjectStimulusConfig(
            window_center=window_center,
            window_size=grid_window_size,
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
            feature_transform=_normalize_feature_transform(feature_transform),
            alignment_alpha=_normalize_alignment_alpha(alignment_alpha),
            chance_classes=chance_classes,
            random_state=random_state,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        for window_center, grid_window_size, feature_mode, normalization, alignment, classifier, components_pca, sample_weighting, score_calibration, feature_transform, alignment_alpha in product(
            window_centers,
            window_sizes,
            feature_modes,
            normalizations,
            alignments,
            classifiers,
            _impl._components_pca_values_for_grid(components_pca_values),
            sample_weightings,
            score_calibrations,
            feature_transforms,
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
        elif feature_mode == "sensor_flat_logpower":
            feature = _sensor_flat_logpower_feature(signal)
        elif feature_mode == "sensor_flat_smooth":
            feature = _sensor_flat_smooth_feature(signal)
        elif feature_mode == "sensor_flat_taper":
            feature = _sensor_flat_taper_feature(signal)
        elif feature_mode == "sensor_flat_gaussian_taper":
            feature = _sensor_flat_gaussian_taper_feature(signal)
        elif feature_mode == "sensor_flat_centered":
            feature = _sensor_flat_centered_feature(signal)
        elif feature_mode == "sensor_flat_delta":
            feature = _sensor_flat_delta_feature(signal)
        elif feature_mode == "sensor_flat_dct":
            feature = _sensor_flat_dct_feature(signal)
        elif feature_mode == "sensor_flat_time_pyramid":
            feature = _sensor_flat_time_pyramid_feature(signal)
        elif feature_mode == "sensor_flat_time_pyramid_logpower":
            feature = _sensor_flat_time_pyramid_logpower_feature(signal)
        elif feature_mode == "sensor_flat_time_pyramid_delta":
            feature = _sensor_flat_time_pyramid_delta_feature(signal)
        elif feature_mode in SENSOR_FLAT_TIME_BIN_FEATURE_MODES:
            feature = _sensor_flat_time_bins_feature(
                signal,
                n_bins=SENSOR_FLAT_TIME_BIN_FEATURE_MODES[feature_mode],
            )
        elif feature_mode == "sensor_dct":
            feature = _sensor_dct_feature(signal)
        elif feature_mode == "sensor_mean_logpower":
            feature = np.concatenate((np.mean(signal, axis=1), _sensor_logpower_feature(signal)))
        elif feature_mode == "sensor_bandpower":
            feature = _sensor_bandpower_feature(signal, window_time)
        elif feature_mode == "sensor_cov_tangent":
            feature = _sensor_cov_tangent_feature(signal)
        elif feature_mode == "sensor_time_pyramid":
            feature = _sensor_time_pyramid_feature(signal)
        elif feature_mode == "sensor_time_pyramid_logpower":
            feature = np.concatenate((_sensor_time_pyramid_feature(signal), _sensor_logpower_feature(signal)))
        elif feature_mode == "sensor_time_pyramid_delta":
            feature = _sensor_time_pyramid_delta_feature(signal)
        elif feature_mode == "sensor_time_pyramid_delta_logpower":
            feature = np.concatenate((_sensor_time_pyramid_delta_feature(signal), _sensor_logpower_feature(signal)))
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    return np.vstack(features), int(np.sum(mask))


def _sensor_logpower_feature(window_signal):
    return np.log(np.mean(np.square(np.asarray(window_signal, dtype=float)), axis=1) + 1e-12)


def _sensor_flat_logpower_feature(window_signal):
    """Return raw evoked samples plus one per-sensor log-power block.

    The flattened evoked block uses the same channel-block layout as
    ``sensor_flat``: all channels for sample 1, all channels for sample 2, etc.
    Appending log-power as another all-channel block keeps the feature width a
    multiple of the channel count, so subject_baseline_whiten can apply the same
    per-channel whitening matrix blockwise.
    """

    signal = np.asarray(window_signal, dtype=float)
    return np.concatenate((signal.reshape(-1, order="F"), _sensor_logpower_feature(signal)))


def _sensor_flat_smooth_feature(window_signal):
    """Return lightly time-smoothed raw evoked samples in sensor_flat layout."""

    return _temporal_smooth_signal(window_signal).reshape(-1, order="F")


def _sensor_flat_taper_weights(n_samples, floor=DEFAULT_SENSOR_FLAT_TAPER_FLOOR):
    """Return a smooth nonzero temporal taper for flattened evoked samples."""

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("sensor_flat_taper requires at least one time sample.")
    if n_samples == 1:
        return np.ones(1, dtype=float)

    floor = float(floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("DEFAULT_SENSOR_FLAT_TAPER_FLOOR must be in [0, 1].")
    return floor + (1.0 - floor) * np.hanning(n_samples)


def _sensor_flat_taper_feature(window_signal):
    """Return raised-Hann tapered samples in sensor_flat channel-block layout."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    weights = _sensor_flat_taper_weights(signal.shape[1])
    return (signal * weights[None, :]).reshape(-1, order="F")


def _sensor_flat_gaussian_taper_weights(
    n_samples,
    *,
    floor=DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_FLOOR,
    sigma=DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_SIGMA,
):
    """Return a broad Gaussian temporal taper for flattened evoked samples.

    The current best BUSH-MEG source-only result uses a wider 150 ms window,
    suggesting that useful information is spread across the early visual
    response.  This taper keeps the same feature width and channel-block layout
    as ``sensor_flat`` but softly emphasizes the window center instead of giving
    the low-SNR window edges equal weight.
    """

    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("sensor_flat_gaussian_taper requires at least one time sample.")
    if n_samples == 1:
        return np.ones(1, dtype=float)

    floor = float(floor)
    sigma = float(sigma)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_FLOOR must be in [0, 1].")
    if sigma <= 0.0:
        raise ValueError("DEFAULT_SENSOR_FLAT_GAUSSIAN_TAPER_SIGMA must be positive.")

    positions = np.linspace(-1.0, 1.0, n_samples, dtype=float)
    weights = np.exp(-0.5 * np.square(positions / sigma))
    weights /= float(np.max(weights))
    return floor + (1.0 - floor) * weights


def _sensor_flat_gaussian_taper_feature(window_signal):
    """Return Gaussian-tapered samples in sensor_flat channel-block layout."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    weights = _sensor_flat_gaussian_taper_weights(signal.shape[1])
    return (signal * weights[None, :]).reshape(-1, order="F")


def _sensor_flat_centered_feature(window_signal):
    """Return per-trial temporal-mean-centered samples in sensor_flat layout."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    return (signal - np.mean(signal, axis=1, keepdims=True)).reshape(-1, order="F")


def _sensor_flat_time_bins_feature(window_signal, *, n_bins):
    """Return per-sensor means over equal temporal bins in sensor_flat layout."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("sensor_flat_time_bins requires at least one bin.")
    sample_indices = np.arange(signal.shape[1])
    pieces = [
        np.mean(signal[:, indices], axis=1)
        if indices.size
        else np.zeros(signal.shape[0], dtype=float)
        for indices in np.array_split(sample_indices, n_bins)
    ]
    return np.concatenate(pieces)


def _temporal_smooth_signal(window_signal):
    """Apply a short edge-padded temporal smoothing kernel per sensor."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    if signal.shape[1] < 2:
        return signal.copy()

    kernel = np.asarray(DEFAULT_SENSOR_FLAT_SMOOTH_KERNEL, dtype=float)
    if kernel.shape != (3,) or not np.isclose(np.sum(kernel), 1.0):
        raise ValueError(
            "DEFAULT_SENSOR_FLAT_SMOOTH_KERNEL must contain three weights summing to one."
        )
    padded = np.pad(signal, ((0, 0), (1, 1)), mode="edge")
    return (
        kernel[0] * padded[:, :-2]
        + kernel[1] * padded[:, 1:-1]
        + kernel[2] * padded[:, 2:]
    )


def _sensor_flat_delta_feature(window_signal):
    """Return raw evoked samples plus adjacent temporal differences."""

    signal = np.asarray(window_signal, dtype=float)
    flat = signal.reshape(-1, order="F")
    if signal.shape[1] < 2:
        return flat
    deltas = np.diff(signal, axis=1).reshape(-1, order="F")
    return np.concatenate((flat, deltas))


def _sensor_flat_dct_feature(window_signal):
    """Return raw evoked samples plus compact low-order DCT waveform blocks."""

    signal = np.asarray(window_signal, dtype=float)
    return np.concatenate((signal.reshape(-1, order="F"), _sensor_dct_feature(signal)))


def _sensor_flat_time_pyramid_feature(window_signal):
    """Return raw evoked samples plus compact multiscale temporal summaries."""

    signal = np.asarray(window_signal, dtype=float)
    flat = signal.reshape(-1, order="F")
    return np.concatenate((flat, _sensor_time_pyramid_feature(signal)))


def _sensor_flat_time_pyramid_logpower_feature(window_signal):
    """Return raw evoked samples, temporal-pyramid summaries, and log-power."""

    signal = np.asarray(window_signal, dtype=float)
    return np.concatenate(
        (
            _sensor_flat_time_pyramid_feature(signal),
            _sensor_logpower_feature(signal),
        )
    )


def _sensor_flat_time_pyramid_delta_feature(window_signal):
    """Return raw evoked samples plus compact temporal-pyramid summaries."""

    signal = np.asarray(window_signal, dtype=float)
    return np.concatenate(
        (
            signal.reshape(-1, order="F"),
            _sensor_time_pyramid_delta_feature(signal),
        )
    )


def _sensor_dct_feature(
    window_signal,
    n_coefficients=DEFAULT_SENSOR_DCT_COEFFICIENTS,
):
    """Return low-order DCT-II waveform coefficients for each sensor."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    n_samples = int(signal.shape[1])
    if n_samples <= 0:
        raise ValueError("sensor_dct requires at least one time sample.")
    n_coefficients = int(n_coefficients)
    if n_coefficients <= 0:
        raise ValueError("sensor_dct requires at least one coefficient.")

    sample_positions = np.arange(n_samples, dtype=float) + 0.5
    coefficient_indices = np.arange(n_coefficients, dtype=float)[:, None]
    basis = np.cos(
        np.pi * coefficient_indices * sample_positions[None, :] / float(n_samples)
    )
    basis[0] *= np.sqrt(1.0 / float(n_samples))
    if n_coefficients > 1:
        basis[1:] *= np.sqrt(2.0 / float(n_samples))
    coefficients = signal @ basis.T
    return coefficients.reshape(-1, order="F")


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


def _sensor_time_pyramid_delta_feature(
    window_signal, levels=DEFAULT_SENSOR_TIME_PYRAMID_LEVELS
):
    """Concatenate temporal-pyramid means and adjacent-bin deltas per sensor."""

    signal = np.asarray(window_signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("window_signal must be a channel x time matrix.")
    sample_indices = np.arange(signal.shape[1])
    mean_pieces = []
    delta_pieces = []
    for level in levels:
        level = int(level)
        if level <= 0:
            raise ValueError("Temporal-pyramid levels must be positive.")
        level_means = [
            np.mean(signal[:, indices], axis=1)
            if indices.size
            else np.zeros(signal.shape[0], dtype=float)
            for indices in np.array_split(sample_indices, level)
        ]
        mean_pieces.extend(level_means)
        delta_pieces.extend(
            level_means[index + 1] - level_means[index]
            for index in range(len(level_means) - 1)
        )
    return np.concatenate([*mean_pieces, *delta_pieces])


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
    feature_mode = _normalize_feature_mode(config.feature_mode)
    if feature_mode == "sensor_flat_logpower":
        return _sensor_flat_logpower_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_smooth":
        return _sensor_flat_smooth_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_taper":
        return _sensor_flat_taper_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_gaussian_taper":
        return _sensor_flat_gaussian_taper_baseline_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if feature_mode == "sensor_flat_centered":
        return _sensor_flat_centered_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_delta":
        return _sensor_flat_delta_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_dct":
        return _sensor_flat_dct_baseline_statistics(data, config, n_window_samples, trial_indices)
    if feature_mode == "sensor_flat_time_pyramid":
        return _sensor_flat_time_pyramid_baseline_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
            include_logpower=False,
        )
    if feature_mode == "sensor_flat_time_pyramid_logpower":
        return _sensor_flat_time_pyramid_baseline_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
            include_logpower=True,
        )
    if feature_mode == "sensor_flat_time_pyramid_delta":
        return _sensor_flat_time_pyramid_delta_baseline_statistics(
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if feature_mode in SENSOR_FLAT_TIME_BIN_FEATURE_MODES:
        return _sensor_flat_time_bins_baseline_statistics(
            data,
            config,
            trial_indices,
            n_bins=SENSOR_FLAT_TIME_BIN_FEATURE_MODES[feature_mode],
        )
    if feature_mode in EXTENDED_FEATURE_MODES:
        baseline_features, n_baseline_samples = _extract_window_features(data, config.baseline_window, feature_mode=config.feature_mode, trial_indices=trial_indices)
        mean = np.mean(baseline_features, axis=0, keepdims=True)
        std = np.std(baseline_features, axis=0, keepdims=True)
        return mean, _impl._nonzero_std(std), n_baseline_samples
    return _previous_baseline_feature_statistics(data, config, n_window_samples, trial_indices)


def _sensor_flat_logpower_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for sensor_flat plus a same-channel log-power block.

    The decode window and the baseline window usually have different durations.
    For the raw ``sensor_flat`` part we therefore mirror the legacy behavior:
    estimate one baseline mean/std per channel and tile it over the number of
    decode-window samples.  For the appended log-power block, estimate a
    per-channel baseline log-power distribution over baseline trials.
    """

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    logpower_features, _ = _extract_window_features(
        data,
        config.baseline_window,
        feature_mode="sensor_logpower",
        trial_indices=trial_indices,
    )
    flat_mean = np.tile(channel_mean, int(n_window_samples))
    flat_std = np.tile(channel_std, int(n_window_samples))
    mean = np.concatenate((flat_mean, np.mean(logpower_features, axis=0)))[None, :]
    std = np.concatenate((flat_std, np.std(logpower_features, axis=0)))[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_smooth_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for smoothed sensor_flat features."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    mean = np.tile(channel_mean, int(n_window_samples))[None, :]
    std = np.tile(channel_std, int(n_window_samples))[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_taper_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for tapered sensor_flat features."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    n_window_samples = int(n_window_samples)
    weights = _sensor_flat_taper_weights(n_window_samples)
    n_channels = int(channel_mean.shape[0])
    flat_mean = np.tile(channel_mean, n_window_samples) * np.repeat(
        weights,
        n_channels,
    )
    flat_std = np.tile(channel_std, n_window_samples)
    return flat_mean[None, :], _impl._nonzero_std(flat_std[None, :]), n_baseline_samples


def _sensor_flat_gaussian_taper_baseline_statistics(
    data,
    config,
    n_window_samples,
    trial_indices,
):
    """Baseline statistics for Gaussian-tapered sensor_flat features."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    n_window_samples = int(n_window_samples)
    weights = _sensor_flat_gaussian_taper_weights(n_window_samples)
    n_channels = int(channel_mean.shape[0])
    repeated_weights = np.repeat(weights, n_channels)
    flat_mean = np.tile(channel_mean, n_window_samples) * repeated_weights
    flat_std = np.tile(channel_std, n_window_samples) * repeated_weights
    return flat_mean[None, :], _impl._nonzero_std(flat_std[None, :]), n_baseline_samples


def _sensor_flat_centered_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for temporal-mean-centered sensor_flat features."""

    _channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    n_window_samples = int(n_window_samples)
    n_channels = int(channel_std.shape[0])
    mean = np.zeros(n_channels * n_window_samples, dtype=float)
    std = np.tile(channel_std, n_window_samples)
    return mean[None, :], _impl._nonzero_std(std[None, :]), n_baseline_samples


def _sensor_flat_time_bins_baseline_statistics(data, config, trial_indices, *, n_bins):
    """Baseline statistics for binned sensor-flat channel blocks."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("sensor_flat_time_bins requires at least one bin.")
    mean = np.tile(channel_mean, n_bins)[None, :]
    std = np.tile(channel_std, n_bins)[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_delta_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for sensor_flat plus adjacent temporal differences."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    time_vector = _impl._time_vector(data, 0)
    mask = _impl._time_mask(time_vector, config.baseline_window)
    delta_blocks = []
    for trial_idx in _impl._iter_trial_indices(data, trial_indices):
        signal = _impl._trial_signal(data, trial_idx)[:, mask]
        if signal.shape[1] >= 2:
            delta_blocks.append(np.diff(signal, axis=1))

    if delta_blocks:
        baseline_deltas = np.concatenate(delta_blocks, axis=1)
        delta_mean = np.mean(baseline_deltas, axis=1)
        delta_std = np.std(baseline_deltas, axis=1)
    else:
        delta_mean = np.zeros_like(channel_mean, dtype=float)
        delta_std = np.ones_like(channel_std, dtype=float)

    n_window_samples = int(n_window_samples)
    n_delta_samples = max(n_window_samples - 1, 0)
    flat_mean = np.tile(channel_mean, n_window_samples)
    flat_std = np.tile(channel_std, n_window_samples)
    delta_mean_tiled = np.tile(delta_mean, n_delta_samples)
    delta_std_tiled = np.tile(delta_std, n_delta_samples)
    mean = np.concatenate((flat_mean, delta_mean_tiled))[None, :]
    std = np.concatenate((flat_std, delta_std_tiled))[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_dct_baseline_statistics(data, config, n_window_samples, trial_indices):
    """Baseline statistics for raw flat samples plus DCT waveform blocks."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    dct_features, _ = _extract_window_features(
        data,
        config.baseline_window,
        feature_mode="sensor_dct",
        trial_indices=trial_indices,
    )
    flat_mean = np.tile(channel_mean, int(n_window_samples))
    flat_std = np.tile(channel_std, int(n_window_samples))
    mean = np.concatenate((flat_mean, np.mean(dct_features, axis=0)))[None, :]
    std = np.concatenate((flat_std, np.std(dct_features, axis=0)))[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_time_pyramid_baseline_statistics(
    data,
    config,
    n_window_samples,
    trial_indices,
    *,
    include_logpower,
):
    """Baseline statistics for raw samples plus temporal-pyramid blocks."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    pyramid_features, _ = _extract_window_features(
        data,
        config.baseline_window,
        feature_mode="sensor_time_pyramid",
        trial_indices=trial_indices,
    )
    mean_pieces = [
        np.tile(channel_mean, int(n_window_samples)),
        np.mean(pyramid_features, axis=0),
    ]
    std_pieces = [
        np.tile(channel_std, int(n_window_samples)),
        np.std(pyramid_features, axis=0),
    ]
    if include_logpower:
        logpower_features, _ = _extract_window_features(
            data,
            config.baseline_window,
            feature_mode="sensor_logpower",
            trial_indices=trial_indices,
        )
        mean_pieces.append(np.mean(logpower_features, axis=0))
        std_pieces.append(np.std(logpower_features, axis=0))
    mean = np.concatenate(mean_pieces)[None, :]
    std = np.concatenate(std_pieces)[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _sensor_flat_time_pyramid_delta_baseline_statistics(
    data,
    config,
    n_window_samples,
    trial_indices,
):
    """Baseline statistics for raw flat samples plus pyramid-delta blocks."""

    channel_mean, channel_std, n_baseline_samples = _impl._baseline_channel_statistics(
        data,
        config.baseline_window,
        trial_indices,
    )
    pyramid_features, _ = _extract_window_features(
        data,
        config.baseline_window,
        feature_mode="sensor_time_pyramid_delta",
        trial_indices=trial_indices,
    )
    flat_mean = np.tile(channel_mean, int(n_window_samples))
    flat_std = np.tile(channel_std, int(n_window_samples))
    mean = np.concatenate((flat_mean, np.mean(pyramid_features, axis=0)))[None, :]
    std = np.concatenate((flat_std, np.std(pyramid_features, axis=0)))[None, :]
    return mean, _impl._nonzero_std(std), n_baseline_samples


def _normalize_features(features, config, baseline_feature_mean, baseline_feature_std, baseline_whitening_matrix):
    feature_mode = _normalize_feature_mode(config.feature_mode)
    if feature_mode in BASELINE_WHITENED_EXTENDED_FEATURE_MODES and config.normalization == "subject_baseline_whiten":
        if baseline_feature_mean is None or baseline_whitening_matrix is None:
            raise ValueError(f"{feature_mode} requires baseline feature statistics and a whitening matrix for subject_baseline_whiten.")
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
            raise ValueError(f"{feature_mode} requires baseline feature statistics and a whitening matrix for subject_baseline_whiten.")
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


def _fit_training_feature_transform(train_features, train_sets, config, *, train_labels=None):
    """Fit a source-only supervised feature rescaling transform."""

    config = _normalized_config(config)
    mode = _normalize_feature_transform(
        getattr(config, "feature_transform", DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM)
    )
    train_features = np.asarray(train_features, dtype=float)
    if mode == "none":
        return train_features, None
    if mode not in SOURCE_ANOVA_FEATURE_TRANSFORM_MODES:
        raise ValueError(f"Unsupported feature_transform: {mode}")

    labels = _feature_transform_labels(train_sets, train_features, train_labels)
    weights, diagnostics = _source_anova_feature_weights(train_features, labels)
    power = float(SOURCE_ANOVA_FEATURE_TRANSFORM_POWERS[mode])
    if not np.isclose(power, 1.0):
        # ``source_anova_scale`` already normalizes weights to a median of roughly
        # one.  Applying a fractional power shrinks both boosts and suppressions
        # back toward one without changing their ordering, which gives a
        # conservative source-only alternative for high-dimensional MEG windows.
        weights = np.power(weights, power)
    diagnostics = dict(diagnostics)
    diagnostics["feature_transform_status"] = f"fitted_{mode}"
    transformed = train_features * weights[None, :]
    return transformed, {
        "mode": mode,
        "feature_transform_power": power,
        "feature_weights": weights,
        "weight_min": float(np.min(weights)) if weights.size else np.nan,
        "weight_max": float(np.max(weights)) if weights.size else np.nan,
        "weight_mean": float(np.mean(weights)) if weights.size else np.nan,
        "n_features": int(weights.shape[0]),
        **diagnostics,
    }


def _feature_transform_labels(train_sets, train_features, train_labels):
    if train_labels is not None:
        labels = np.asarray(train_labels, dtype=int).ravel()
    else:
        labels = np.concatenate(
            [
                np.asarray(feature_set.labels, dtype=int).ravel() - 1
                for feature_set in train_sets
            ]
        )
    if labels.shape[0] != np.asarray(train_features).shape[0]:
        raise ValueError("feature_transform labels must match train feature rows.")
    return labels


def _source_anova_feature_weights(features, labels):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int).ravel()
    n_samples, n_features = features.shape
    if n_samples == 0 or n_features == 0:
        return np.ones(n_features, dtype=float), {
            "feature_transform_status": "skipped_empty_features"
        }

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return np.ones(n_features, dtype=float), {
            "feature_transform_status": "skipped_one_class"
        }

    total_mean = np.mean(features, axis=0)
    ss_between = np.zeros(n_features, dtype=float)
    ss_within = np.zeros(n_features, dtype=float)
    used_classes = 0
    for label in unique_labels:
        group = features[labels == label]
        if group.shape[0] == 0:
            continue
        group_mean = np.mean(group, axis=0)
        ss_between += float(group.shape[0]) * np.square(group_mean - total_mean)
        ss_within += np.sum(np.square(group - group_mean), axis=0)
        used_classes += 1

    if used_classes < 2:
        return np.ones(n_features, dtype=float), {
            "feature_transform_status": "skipped_one_class"
        }
    df_between = max(used_classes - 1, 1)
    df_within = max(n_samples - used_classes, 1)
    f_scores = (ss_between / float(df_between)) / (
        (ss_within / float(df_within)) + SOURCE_ANOVA_FEATURE_TRANSFORM_EPSILON
    )
    f_scores = np.where(np.isfinite(f_scores) & (f_scores > 0.0), f_scores, 0.0)
    raw_weights = np.sqrt(f_scores)
    positive = raw_weights[np.isfinite(raw_weights) & (raw_weights > 0.0)]
    normalizer = (
        float(np.median(positive))
        if positive.size
        else SOURCE_ANOVA_FEATURE_TRANSFORM_NORMALIZER_FALLBACK
    )
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        normalizer = SOURCE_ANOVA_FEATURE_TRANSFORM_NORMALIZER_FALLBACK
    weights = raw_weights / normalizer
    weights = np.clip(
        weights,
        SOURCE_ANOVA_FEATURE_TRANSFORM_MIN_WEIGHT,
        SOURCE_ANOVA_FEATURE_TRANSFORM_MAX_WEIGHT,
    )
    return weights.astype(float, copy=False), {
        "feature_transform_status": "fitted_source_anova_scale",
        "feature_transform_classes": int(used_classes),
        "feature_transform_normalizer": float(normalizer),
    }


def _has_active_feature_transform_metadata(metadata):
    return (
        isinstance(metadata, dict)
        and metadata.get("mode") in SOURCE_ANOVA_FEATURE_TRANSFORM_MODES
        and "feature_weights" in metadata
    )


def _apply_training_feature_transform(features, metadata):
    features = np.asarray(features, dtype=float)
    if not _has_active_feature_transform_metadata(metadata):
        return features
    weights = np.asarray(metadata["feature_weights"], dtype=float).ravel()
    if features.ndim != 2 or features.shape[1] != weights.shape[0]:
        raise ValueError("Stored feature transform width does not match feature matrix width.")
    return features * weights[None, :]


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
        train_features, feature_transform_metadata = fit_training_feature_transform(
            train_features, train_sets, config, train_labels=train_labels
        )
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
    score_calibration = _normalize_score_calibration(config.score_calibration)
    if fit_score_calibration and score_calibration in (
        INNER_SCORE_CALIBRATION_MODES | GUARDED_INNER_SCORE_CALIBRATION_MODES
    ):
        fitted_model["score_calibration_metadata"] = _fit_inner_score_calibration(
            train_sets,
            config,
            classifier_param,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
    elif fit_score_calibration and score_calibration in TRAIN_SCORE_CALIBRATION_MODES:
        fitted_model["score_calibration_metadata"] = _fit_train_score_calibration(
            model_bundle,
            train_features,
            train_labels,
            config,
        )
    else:
        fitted_model["score_calibration_metadata"] = {"mode": score_calibration}
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
    base_mode = _score_calibration_base_mode(mode)
    guarded = mode in GUARDED_INNER_SCORE_CALIBRATION_MODES
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
        # Source-inner calibration must validate the same candidate-score path
        # that is used for outer scoring. Calling the legacy helper here skips
        # _next hooks such as feature transforms, train-only target alignment,
        # and score wrappers, so guarded calibration can make its apply/skip
        # decision on a representation that differs from the final model path.
        scores, score_classes = _candidate_model_scores(
            inner_model,
            validation_set,
            inner_config,
        )
        all_scores.append(_align_class_score_columns(scores, score_classes, class_order))
        all_labels.append(np.asarray(validation_set.labels, dtype=int) - 1)
    scores = np.vstack(all_scores)
    labels = np.concatenate(all_labels)
    baseline_balanced = _balanced_accuracy_for_scores(scores, labels, class_order)
    if base_mode in {"inner_probability_map", "inner_rank_probability_map"}:
        score_space = "rank" if base_mode == "inner_rank_probability_map" else "raw"
        map_scores = _rank_score_matrix(scores) if score_space == "rank" else scores
        probability_map, inner_balanced = _fit_probability_map(
            map_scores, labels, class_order
        )
        return _guard_inner_score_calibration_metadata(
            _probability_map_metadata(
                mode,
                class_order,
                probability_map,
                inner_balanced,
                score_space=score_space,
            ),
            baseline_balanced,
            guarded=guarded,
        )
    if base_mode in {"inner_confusion_blend", "inner_rank_confusion_blend"}:
        score_space = "rank" if base_mode == "inner_rank_confusion_blend" else "raw"
        calibration_scores = _rank_score_matrix(scores) if score_space == "rank" else scores
        confusion_matrix, blend_alpha, inner_balanced = _fit_confusion_blend(
            calibration_scores, labels, class_order
        )
        return _guard_inner_score_calibration_metadata(
            {
                "mode": mode,
                "score_space": score_space,
                "classes": class_order,
                "confusion_matrix": confusion_matrix,
                "blend_alpha": blend_alpha,
                "inner_balanced_accuracy": inner_balanced,
                "calibration_source": "inner_scores",
                "smoothing": CONFUSION_CALIBRATION_SMOOTHING,
            },
            baseline_balanced,
            guarded=guarded,
        )
    if base_mode in {"inner_margin_confusion_blend", "inner_rank_margin_confusion_blend"}:
        score_space = (
            "rank" if base_mode == "inner_rank_margin_confusion_blend" else "raw"
        )
        calibration_scores = _rank_score_matrix(scores) if score_space == "rank" else scores
        confusion_matrix, blend_alpha, margin_threshold, inner_balanced = (
            _fit_margin_confusion_blend(calibration_scores, labels, class_order)
        )
        return _guard_inner_score_calibration_metadata(
            {
                "mode": mode,
                "score_space": score_space,
                "classes": class_order,
                "confusion_matrix": confusion_matrix,
                "blend_alpha": blend_alpha,
                "margin_threshold": margin_threshold,
                "inner_balanced_accuracy": inner_balanced,
                "calibration_source": "inner_scores",
                "smoothing": CONFUSION_CALIBRATION_SMOOTHING,
                "margin_quantiles": CONFUSION_CALIBRATION_MARGIN_QUANTILES,
            },
            baseline_balanced,
            guarded=guarded,
        )
    score_space = "raw"
    calibration_scores = scores
    if base_mode == "inner_rank_bias":
        score_space = "rank"
        calibration_scores = _rank_score_matrix(scores)
    if base_mode == "inner_class_affine":
        bias, scale, inner_balanced = _optimize_class_affine(
            calibration_scores, labels, class_order
        )
    else:
        bias, inner_balanced = _optimize_class_bias(
            calibration_scores, labels, class_order
        )
        scale = np.ones(class_order.shape[0], dtype=float)
    return _guard_inner_score_calibration_metadata(
        {
            "mode": mode,
            "score_space": score_space,
            "classes": class_order,
            "bias": bias,
            "scale": scale,
            "inner_balanced_accuracy": inner_balanced,
            "l2_penalty": SCORE_CALIBRATION_L2,
        },
        baseline_balanced,
        guarded=guarded,
    )


def _fit_train_score_calibration(model_bundle, train_features, train_labels, config):
    """Fit source-only class score calibration on the final source model."""

    mode = _normalize_score_calibration(
        getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION)
    )
    class_order = np.arange(int(config.chance_classes), dtype=int)
    scores, score_classes = _impl._model_class_scores(model_bundle, train_features)
    scores = _align_class_score_columns(scores, score_classes, class_order)
    labels = np.asarray(train_labels, dtype=int)
    if scores.shape[0] == 0 or labels.shape[0] == 0:
        return {"mode": mode, "status": "skipped_no_training_scores"}
    score_space = "raw"
    calibration_scores = scores
    if mode == "train_rank_bias":
        score_space = "rank"
        calibration_scores = _rank_score_matrix(scores)
    if mode == "train_class_affine":
        bias, scale, source_balanced = _optimize_class_affine(
            calibration_scores, labels, class_order
        )
    else:
        bias, source_balanced = _optimize_class_bias(
            calibration_scores, labels, class_order
        )
        scale = np.ones(class_order.shape[0], dtype=float)
    return {
        "mode": mode,
        "score_space": score_space,
        "classes": class_order,
        "bias": bias,
        "scale": scale,
        "inner_balanced_accuracy": source_balanced,
        "source_balanced_accuracy": source_balanced,
        "calibration_source": "train_scores",
        "l2_penalty": SCORE_CALIBRATION_L2,
    }


def _config_with(config, **updates):
    kwargs = {field.name: getattr(config, field.name) for field in fields(config)}
    kwargs.update(updates)
    return CrossSubjectStimulusConfig(**kwargs)


def _probability_map_metadata(
    mode, class_order, probability_map, inner_balanced, *, score_space="raw"
):
    return {
        "mode": mode,
        "score_space": score_space,
        "classes": np.asarray(class_order, dtype=int),
        "probability_map": np.asarray(probability_map, dtype=float),
        "inner_balanced_accuracy": inner_balanced,
        "calibration_source": "inner_probability_map",
        "probability_map_l2_penalty": SCORE_CALIBRATION_PROBABILITY_MAP_L2,
        "probability_map_identity_blend": (
            SCORE_CALIBRATION_PROBABILITY_MAP_IDENTITY_BLEND
        ),
    }


def _guard_inner_score_calibration_metadata(metadata, baseline_balanced, *, guarded):
    """Disable guarded calibration unless source-inner validation improves."""

    metadata = dict(metadata)
    baseline = float(baseline_balanced)
    inner_balanced = float(metadata.get("inner_balanced_accuracy", np.nan))
    metadata["inner_uncalibrated_balanced_accuracy"] = baseline
    if not guarded:
        return metadata
    if (
        np.isfinite(inner_balanced)
        and np.isfinite(baseline)
        and inner_balanced > baseline + SCORE_CALIBRATION_MIN_INNER_GAIN
    ):
        metadata.setdefault("status", "applied_guarded_inner_gain")
        return metadata
    return {
        "mode": metadata.get("mode", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION),
        "status": "skipped_no_inner_gain",
        "calibration_source": metadata.get("calibration_source", "inner_scores"),
        "score_calibration_base_mode": _score_calibration_base_mode(
            metadata.get("mode", "none")
        ),
        "inner_balanced_accuracy": inner_balanced,
        "inner_uncalibrated_balanced_accuracy": baseline,
    }


def _fit_probability_map(scores, labels, class_order):
    """Fit a source-inner probability remapping from probabilities to labels."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    n_classes = int(class_order.shape[0])
    identity = np.eye(n_classes, dtype=float)
    if scores.shape[0] == 0 or labels.shape[0] == 0 or n_classes == 0:
        return identity, np.nan

    probabilities = _score_softmax_probabilities(scores)
    targets = _one_hot_labels(labels, class_order)
    regularizer = SCORE_CALIBRATION_PROBABILITY_MAP_L2 * identity
    normal_matrix = probabilities.T @ probabilities + regularizer
    rhs = probabilities.T @ targets
    try:
        probability_map = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError:
        probability_map = np.linalg.pinv(normal_matrix) @ rhs

    probability_map = np.maximum(np.asarray(probability_map, dtype=float), 0.0)
    blend = float(SCORE_CALIBRATION_PROBABILITY_MAP_IDENTITY_BLEND)
    probability_map = (1.0 - blend) * probability_map + blend * identity
    probability_map = _row_normalize_probabilities(probability_map)
    calibrated_scores = _probabilities_to_logits(probabilities @ probability_map)
    return probability_map, _balanced_accuracy_for_scores(
        calibrated_scores, labels, class_order
    )


def _fit_confusion_blend(scores, labels, class_order):
    """Fit a source-only predicted-class to true-class re-ranking map."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    confusion_matrix = _confusion_true_given_pred_matrix(
        scores, labels, class_order
    )
    best_alpha = 0.0
    best_balanced = _balanced_accuracy_for_scores(scores, labels, class_order)
    for blend_alpha in CONFUSION_CALIBRATION_BLEND_GRID:
        calibrated_scores = _confusion_blend_scores(
            scores, confusion_matrix, blend_alpha
        )
        balanced = _balanced_accuracy_for_scores(
            calibrated_scores, labels, class_order
        )
        if balanced > best_balanced + 1e-12:
            best_alpha = float(blend_alpha)
            best_balanced = balanced
    return confusion_matrix, best_alpha, best_balanced


def _fit_margin_confusion_blend(scores, labels, class_order):
    """Fit a confusion re-ranker only for low-margin source-inner trials."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    confusion_matrix = _confusion_true_given_pred_matrix(
        scores, labels, class_order
    )
    probabilities = _score_probabilities(scores)
    margins = _top2_probability_margins(probabilities)
    best_alpha = 0.0
    best_threshold = float("inf")
    best_balanced = _balanced_accuracy_for_scores(scores, labels, class_order)
    for margin_threshold in _candidate_margin_thresholds(margins):
        for blend_alpha in CONFUSION_CALIBRATION_BLEND_GRID:
            calibrated_scores = _margin_confusion_blend_scores(
                scores,
                confusion_matrix,
                blend_alpha,
                margin_threshold,
            )
            balanced = _balanced_accuracy_for_scores(
                calibrated_scores, labels, class_order
            )
            if balanced > best_balanced + 1e-12:
                best_alpha = float(blend_alpha)
                best_threshold = float(margin_threshold)
                best_balanced = balanced
    return confusion_matrix, best_alpha, best_threshold, best_balanced


def _candidate_margin_thresholds(margins):
    margins = np.asarray(margins, dtype=float).ravel()
    finite_margins = margins[np.isfinite(margins)]
    if finite_margins.size == 0:
        return (float("inf"),)
    thresholds = [
        float(value)
        for value in np.quantile(
            finite_margins,
            CONFUSION_CALIBRATION_MARGIN_QUANTILES,
        )
    ]
    thresholds.append(float("inf"))
    return tuple(dict.fromkeys(max(0.0, threshold) for threshold in thresholds))


def _confusion_true_given_pred_matrix(scores, labels, class_order):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class_order = np.asarray(class_order, dtype=int)
    n_classes = int(class_order.shape[0])
    matrix = CONFUSION_CALIBRATION_SMOOTHING * np.eye(n_classes, dtype=float)
    if scores.ndim == 2 and scores.shape[0] and scores.shape[1] == n_classes:
        predictions = class_order[np.argmax(scores, axis=1)]
        class_to_column = {
            int(label): column for column, label in enumerate(class_order.tolist())
        }
        for predicted_label, true_label in zip(predictions, labels, strict=True):
            predicted_column = class_to_column.get(int(predicted_label))
            true_column = class_to_column.get(int(true_label))
            if predicted_column is not None and true_column is not None:
                matrix[predicted_column, true_column] += 1.0
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return matrix / row_sums


def _confusion_blend_scores(scores, confusion_matrix, blend_alpha):
    probabilities = _score_probabilities(scores)
    confusion_matrix = np.asarray(confusion_matrix, dtype=float)
    blend_alpha = float(np.clip(float(blend_alpha), 0.0, 1.0))
    corrected = probabilities @ confusion_matrix
    blended = (1.0 - blend_alpha) * probabilities + blend_alpha * corrected
    return _probabilities_to_logits(blended)


def _margin_confusion_blend_scores(
    scores, confusion_matrix, blend_alpha, margin_threshold
):
    probabilities = _score_probabilities(scores)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        return _probabilities_to_logits(probabilities)
    confusion_matrix = np.asarray(confusion_matrix, dtype=float)
    blend_alpha = float(np.clip(float(blend_alpha), 0.0, 1.0))
    corrected = probabilities @ confusion_matrix
    margin_threshold = float(margin_threshold)
    if np.isfinite(margin_threshold):
        margins = _top2_probability_margins(probabilities)
        if margin_threshold <= 1e-12:
            margin_gate = (margins <= margin_threshold).astype(float)
        else:
            margin_gate = np.clip(
                (margin_threshold - margins) / margin_threshold,
                0.0,
                1.0,
            )
        effective_alpha = blend_alpha * margin_gate[:, None]
    else:
        effective_alpha = blend_alpha
    blended = (1.0 - effective_alpha) * probabilities + effective_alpha * corrected
    return _probabilities_to_logits(blended)


def _top2_probability_margins(probabilities):
    probabilities = _row_normalize_probabilities(
        np.asarray(probabilities, dtype=float)
    )
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if probabilities.shape[1] < 2:
        return np.ones(probabilities.shape[0], dtype=float)
    sorted_probabilities = np.sort(probabilities, axis=1)
    return sorted_probabilities[:, -1] - sorted_probabilities[:, -2]


def _score_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    if scores.shape[1] == 0:
        return np.zeros_like(scores, dtype=float)
    row_sums = np.sum(scores, axis=1, keepdims=True)
    probability_like = (
        np.all(np.isfinite(scores), axis=1)
        & np.all(scores >= 0.0, axis=1)
        & (row_sums.ravel() > 0.0)
        & np.isclose(row_sums.ravel(), 1.0, rtol=1e-3, atol=1e-6)
    )
    probabilities = np.zeros_like(scores, dtype=float)
    if np.any(probability_like):
        probabilities[probability_like] = (
            scores[probability_like] / row_sums[probability_like]
        )
    if np.any(~probability_like):
        probabilities[~probability_like] = _score_softmax_probabilities(
            scores[~probability_like]
        )
    return probabilities


def _one_hot_labels(labels, class_order):
    labels = np.asarray(labels, dtype=int).ravel()
    class_order = np.asarray(class_order, dtype=int).ravel()
    targets = np.zeros((labels.shape[0], class_order.shape[0]), dtype=float)
    label_to_column = {
        int(label): column for column, label in enumerate(class_order.tolist())
    }
    for row_index, label in enumerate(labels.tolist()):
        column = label_to_column.get(int(label))
        if column is not None:
            targets[row_index, column] = 1.0
    return targets


def _score_softmax_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        rows = scores.shape[0] if scores.ndim == 2 else 0
        return np.zeros((rows, 0), dtype=float)
    probabilities = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            probabilities[row_index] = np.full(
                row.shape[0], 1.0 / row.shape[0], dtype=float
            )
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


def _row_normalize_probabilities(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros_like(values, dtype=float)
    row_sums = np.sum(values, axis=1, keepdims=True)
    return np.divide(
        values,
        row_sums,
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=row_sums > 1e-12,
    )


def _probabilities_to_logits(probabilities):
    probabilities = _row_normalize_probabilities(
        np.maximum(np.asarray(probabilities, dtype=float), 1e-12)
    )
    return np.log(probabilities)


def _rank_score_matrix(scores):
    """Convert arbitrary class scores into per-row negative-rank scores."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    rank_scores = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            rank_scores[row_index] = 0.0
            continue
        rank_input = np.where(finite, row, -np.inf)
        descending_columns = np.argsort(-rank_input, kind="mergesort")
        ranks = np.empty(row.shape[0], dtype=float)
        ranks[descending_columns] = np.arange(row.shape[0], dtype=float)
        rank_scores[row_index] = -ranks
        rank_scores[row_index, ~finite] = -float(row.shape[0])
    return rank_scores


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


def _model_scores_with_feature_transform(fitted_model, test_set, config):
    config = _normalized_config(config)
    test_features = _impl._normalized_subject_features(test_set, config)
    alignment_model = (
        _impl._fitted_alignment_model(fitted_model)
        if hasattr(_impl, "_fitted_alignment_model")
        else {"metadata": fitted_model.get("alignment_metadata", {})}
    )
    test_features, _test_alignment_metadata = _align_test_features_by_subject(
        test_features, test_set, config, alignment_model
    )
    test_features = _apply_training_feature_transform(
        test_features,
        fitted_model.get("feature_transform_metadata", {})
        if isinstance(fitted_model, dict)
        else {},
    )
    return _impl._model_class_scores(fitted_model["model_bundle"], test_features)


def _candidate_model_scores(fitted_model, test_set, config):
    feature_transform_metadata = (
        fitted_model.get("feature_transform_metadata", {})
        if isinstance(fitted_model, dict)
        else {}
    )
    if _has_active_feature_transform_metadata(feature_transform_metadata):
        scores, classes = _model_scores_with_feature_transform(
            fitted_model, test_set, config
        )
    else:
        scores, classes = _previous_candidate_model_scores(fitted_model, test_set, config)
    return _apply_score_calibration(scores, classes, fitted_model)


def _has_active_score_calibration_metadata(metadata):
    if not isinstance(metadata, dict):
        return False
    mode = metadata.get("mode")
    if mode not in ACTIVE_SCORE_CALIBRATION_MODES:
        return False
    base_mode = _score_calibration_base_mode(mode)
    if base_mode in {
        "inner_confusion_blend",
        "inner_margin_confusion_blend",
        "inner_rank_confusion_blend",
        "inner_rank_margin_confusion_blend",
    }:
        return "classes" in metadata and "confusion_matrix" in metadata
    return "bias" in metadata or "probability_map" in metadata


def _apply_score_calibration(scores, classes, fitted_model):
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    if not _has_active_score_calibration_metadata(metadata):
        return scores, classes
    calibration_classes = np.asarray(metadata["classes"], dtype=int)
    base_mode = _score_calibration_base_mode(metadata.get("mode", "none"))
    if base_mode in {"inner_confusion_blend", "inner_rank_confusion_blend"}:
        aligned = _align_class_score_columns(scores, classes, calibration_classes)
        if metadata.get("score_space") == "rank":
            aligned = _rank_score_matrix(aligned)
        return (
            _confusion_blend_scores(
                aligned,
                metadata["confusion_matrix"],
                metadata.get("blend_alpha", 0.0),
            ),
            calibration_classes,
        )
    if base_mode in {"inner_margin_confusion_blend", "inner_rank_margin_confusion_blend"}:
        aligned = _align_class_score_columns(scores, classes, calibration_classes)
        if metadata.get("score_space") == "rank":
            aligned = _rank_score_matrix(aligned)
        return (
            _margin_confusion_blend_scores(
                aligned,
                metadata["confusion_matrix"],
                metadata.get("blend_alpha", 0.0),
                metadata.get("margin_threshold", float("inf")),
            ),
            calibration_classes,
        )
    if "probability_map" in metadata:
        aligned = _align_class_score_columns(scores, classes, calibration_classes)
        if metadata.get("score_space") == "rank":
            aligned = _rank_score_matrix(aligned)
        probabilities = _score_softmax_probabilities(aligned)
        probability_map = _row_normalize_probabilities(
            np.maximum(np.asarray(metadata["probability_map"], dtype=float), 0.0)
        )
        return _probabilities_to_logits(probabilities @ probability_map), calibration_classes
    bias = np.asarray(metadata["bias"], dtype=float)
    scale = np.asarray(metadata.get("scale", np.ones_like(bias)), dtype=float)
    aligned = _align_class_score_columns(scores, classes, calibration_classes)
    if metadata.get("score_space") == "rank":
        aligned = _rank_score_matrix(aligned)
    return aligned * scale[None, :] + bias[None, :], calibration_classes


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    config = _normalized_config(config)
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    feature_transform_metadata = (
        fitted_model.get("feature_transform_metadata", {})
        if isinstance(fitted_model, dict)
        else {}
    )
    if not _has_active_score_calibration_metadata(metadata) and not _has_active_feature_transform_metadata(feature_transform_metadata):
        outer_row, prediction_rows = _previous_score_outer_fold_model(fitted_model, test_set, config, include_predictions=include_predictions)
        _add_next_fields(outer_row, config, fitted_model)
        for row in prediction_rows:
            _add_next_fields(row, config, fitted_model)
        return outer_row, prediction_rows

    outer_row, _unused = _previous_score_outer_fold_model(fitted_model, test_set, config, include_predictions=False)
    test_features = _impl._normalized_subject_features(test_set, config)
    alignment_model = _impl._fitted_alignment_model(fitted_model) if hasattr(_impl, "_fitted_alignment_model") else {"metadata": fitted_model.get("alignment_metadata", {})}
    test_features, test_alignment_metadata = _align_test_features_by_subject(test_features, test_set, config, alignment_model)
    test_features = _apply_training_feature_transform(test_features, feature_transform_metadata)
    class_scores, score_classes = _impl._model_class_scores(fitted_model["model_bundle"], test_features)
    class_scores, score_classes = _apply_score_calibration(class_scores, score_classes, fitted_model)
    test_labels = np.asarray(test_set.labels, dtype=int) - 1
    predictions = np.asarray(score_classes, dtype=int)[np.argmax(class_scores, axis=1)]
    true_labels_one_based = np.asarray(test_set.labels, dtype=int)
    predicted_labels_one_based = np.asarray(predictions, dtype=int) + 1
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
            "predicted_label_counts": _impl._format_counter(
                Counter(predicted_labels_one_based.tolist())
            ),
            "true_predicted_label_pair_counts": _impl._format_counter(
                _impl._true_predicted_label_pair_counts(
                    true_labels_one_based, predictions
                )
            ),
            "confusion_counts": _impl._format_confusion_counter(
                _impl._confusion_counter(
                    true_labels_one_based, predicted_labels_one_based
                )
            ),
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
    row["feature_transform"] = getattr(config, "feature_transform", DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM)
    row["alignment_alpha"] = getattr(config, "alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA)


def _add_next_fields(row, config, fitted_model):
    _add_config_fields(row, config)
    metadata = fitted_model.get("score_calibration_metadata", {}) if isinstance(fitted_model, dict) else {}
    row["score_calibration"] = metadata.get("mode", getattr(config, "score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION))
    row["score_calibration_inner_balanced_accuracy"] = metadata.get("inner_balanced_accuracy", "")
    row["score_calibration_inner_uncalibrated_balanced_accuracy"] = metadata.get(
        "inner_uncalibrated_balanced_accuracy", ""
    )
    row["score_calibration_status"] = metadata.get("status", "")
    row["score_calibration_source"] = metadata.get("calibration_source", "")
    row["score_calibration_source_balanced_accuracy"] = metadata.get("source_balanced_accuracy", "")
    row["score_calibration_confusion_blend_alpha"] = metadata.get("blend_alpha", "")
    row["score_calibration_confusion_margin_threshold"] = metadata.get(
        "margin_threshold", ""
    )
    row["score_calibration_confusion_smoothing"] = metadata.get("smoothing", "")
    row["score_calibration_probability_map_l2"] = metadata.get(
        "probability_map_l2_penalty", ""
    )
    row["score_calibration_probability_map_identity_blend"] = metadata.get(
        "probability_map_identity_blend", ""
    )
    feature_metadata = fitted_model.get("feature_transform_metadata", {}) if isinstance(fitted_model, dict) else {}
    row["feature_transform"] = feature_metadata.get(
        "mode", getattr(config, "feature_transform", DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM)
    )
    row["feature_transform_status"] = feature_metadata.get("feature_transform_status", "")
    row["feature_transform_weight_min"] = feature_metadata.get("weight_min", "")
    row["feature_transform_weight_max"] = feature_metadata.get("weight_max", "")
    row["feature_transform_weight_mean"] = feature_metadata.get("weight_mean", "")
    row["feature_transform_n_features"] = feature_metadata.get("n_features", "")


def _rank_nested_candidates(inner_rows, *, selection_metric=None):
    metric = (
        None
        if selection_metric is None
        else str(selection_metric).strip().lower().replace("-", "_")
    )
    if metric in WORST_CLASS_SELECTION_METRICS:
        ranked = _previous_rank_nested_candidates(
            inner_rows,
            selection_metric="balanced_accuracy",
        )
        ranked = _rerank_nested_candidates_by_worst_class(ranked, metric)
    elif selection_metric is None:
        ranked = _previous_rank_nested_candidates(inner_rows)
    else:
        ranked = _previous_rank_nested_candidates(inner_rows, selection_metric=selection_metric)
    examples = {int(row["candidate_index"]): row for row in inner_rows}
    for row in ranked:
        example = examples.get(int(row["selected_candidate_index"]), {})
        row["selected_sample_weighting"] = example.get("sample_weighting", DEFAULT_CROSS_SUBJECT_SAMPLE_WEIGHTING)
        row["selected_score_calibration"] = example.get("score_calibration", DEFAULT_CROSS_SUBJECT_SCORE_CALIBRATION)
        row["selected_feature_transform"] = example.get("feature_transform", DEFAULT_CROSS_SUBJECT_FEATURE_TRANSFORM)
        row["selected_alignment_alpha"] = example.get("alignment_alpha", DEFAULT_CROSS_SUBJECT_ALIGNMENT_ALPHA)
    return ranked


def _rerank_nested_candidates_by_worst_class(ranked_rows, selection_metric):
    """Rerank source-inner candidates with a soft per-class recall floor."""

    selection_metric = str(selection_metric).strip().lower().replace("-", "_")
    if selection_metric not in WORST_CLASS_SELECTION_METRICS:
        raise ValueError(f"Unsupported worst-class selection metric: {selection_metric}")

    rows = [dict(row) for row in ranked_rows]
    for row in rows:
        row["selection_metric"] = selection_metric
        worst_recall = _selected_inner_worst_class_recall(row)
        row["selected_inner_worst_class_recall"] = worst_recall
        score = _worst_class_selection_score(row, selection_metric)
        row["selected_inner_selection_score_mean"] = score
        row["selected_inner_selection_score_median"] = score
        row["selected_inner_selection_score_sem"] = 0.0
        row["selected_inner_selection_ranking_score"] = score

    ranked = sorted(rows, key=_worst_class_selection_sort_key, reverse=True)
    if not ranked:
        return ranked

    selected = ranked[0]
    selected_score = float(selected["selected_inner_selection_ranking_score"])
    if len(ranked) > 1:
        second_best_balanced_mean = float(
            ranked[1]["selected_inner_balanced_accuracy_mean"]
        )
        second_best_score = float(ranked[1]["selected_inner_selection_ranking_score"])
        winner_margin = selected_score - second_best_score
    else:
        second_best_balanced_mean = np.nan
        second_best_score = np.nan
        winner_margin = np.nan

    for rank, row in enumerate(ranked, start=1):
        row_score = float(row["selected_inner_selection_ranking_score"])
        row["selected_inner_rank"] = int(rank)
        row["selected_inner_second_best_balanced_accuracy_mean"] = second_best_balanced_mean
        row["selected_inner_second_best_selection_score_mean"] = second_best_score
        row["selected_inner_winner_margin"] = (
            winner_margin if rank == 1 else selected_score - row_score
        )
    return ranked


def _worst_class_selection_sort_key(row):
    return (
        _impl._finite_sort_value(row.get("selected_inner_selection_ranking_score", np.nan)),
        _impl._finite_sort_value(row.get("selected_inner_balanced_accuracy_mean", np.nan)),
        _impl._finite_sort_value(row.get("selected_inner_worst_class_recall", np.nan)),
        _impl._finite_sort_value(row.get("selected_inner_top2_accuracy_mean", np.nan)),
        _impl._finite_sort_value(row.get("selected_inner_top3_accuracy_mean", np.nan)),
        _impl._finite_sort_value(-float(row.get("selected_inner_mean_true_label_rank_mean", np.inf))),
        -int(row["selected_candidate_index"]),
    )


def _worst_class_selection_score(row, selection_metric):
    balanced = float(row.get("selected_inner_balanced_accuracy_mean", np.nan))
    if selection_metric.endswith("_lcb"):
        sem = float(row.get("selected_inner_balanced_accuracy_sem", 0.0))
        balanced -= sem if np.isfinite(sem) else 0.0
    worst_recall = _impl._finite_or(
        row.get("selected_inner_worst_class_recall", np.nan),
        default=balanced,
    )
    weight = float(WORST_CLASS_SELECTION_WEIGHT)
    return float((1.0 - weight) * balanced + weight * worst_recall)


def _selected_inner_worst_class_recall(row):
    counter = _impl._parse_confusion_counter(
        row.get("selected_inner_confusion_counts", "")
    )
    if not counter:
        return np.nan
    labels = sorted(
        {int(true_label) for true_label, _predicted_label in counter}
        | {int(predicted_label) for _true_label, predicted_label in counter}
    )
    if not labels:
        return np.nan
    matrix = _impl._confusion_counter_matrix(counter, labels)
    totals = np.sum(matrix, axis=1)
    valid = totals > 0.0
    if not np.any(valid):
        return np.nan
    recalls = np.diag(matrix)[valid] / totals[valid]
    return float(np.min(recalls))


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
        row["selected_feature_transform_counts"] = _impl._format_counter(Counter(str(value.get("selected_feature_transform", value.get("feature_transform", ""))) for value in outer_rows))
        row["selected_alignment_alpha_counts"] = _impl._format_counter(Counter(str(value.get("selected_alignment_alpha", value.get("alignment_alpha", ""))) for value in outer_rows))
    return rows


def _resolved_classifier_param(config):
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)
    return classifier_param
