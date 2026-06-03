from types import SimpleNamespace

import numpy as np
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    make_cross_subject_candidate_configs,
)


def test_make_candidate_configs_expands_source_anova_feature_transform():
    configs = make_cross_subject_candidate_configs(
        window_centers=(0.175,),
        window_size=0.1,
        feature_modes=("sensor_flat",),
        normalizations=("subject_baseline_whiten",),
        alignments=("none",),
        classifiers=("multinomial-logistic",),
        classifier_params=(1.0,),
        components_pca_values=(128,),
        feature_transforms=("none", "source_anova_scale", "source_anova_sqrt_scale"),
        chance_classes=16,
    )

    assert [config.feature_transform for config in configs] == [
        "none",
        "source_anova_scale",
        "source_anova_sqrt_scale",
    ]


def test_source_anova_feature_transform_scales_discriminative_dimensions():
    train_features = np.asarray(
        [
            [-2.0, -0.2],
            [-1.8, 0.2],
            [1.8, -0.2],
            [2.0, 0.2],
        ],
        dtype=float,
    )
    train_labels = np.asarray([0, 0, 1, 1], dtype=int)
    train_sets = (
        SimpleNamespace(labels=np.asarray([1, 1])),
        SimpleNamespace(labels=np.asarray([2, 2])),
    )
    config = CrossSubjectStimulusConfig(
        feature_transform="source_anova_scale", chance_classes=2
    )

    transformed, metadata = cross_subject._fit_training_feature_transform(  # pylint: disable=protected-access
        train_features, train_sets, config, train_labels=train_labels
    )

    assert metadata["mode"] == "source_anova_scale"
    weights = metadata["feature_weights"]
    assert weights.shape == (2,)
    assert weights[0] > weights[1]
    np.testing.assert_allclose(transformed, train_features * weights[None, :])
    np.testing.assert_allclose(
        cross_subject._apply_training_feature_transform(  # pylint: disable=protected-access
            np.ones((1, 2)), metadata
        ),
        weights[None, :],
    )


def test_source_anova_sqrt_feature_transform_is_conservative():
    train_features = np.asarray(
        [
            [-2.0, -0.4],
            [-1.8, -0.2],
            [1.8, 0.2],
            [2.0, 0.4],
        ],
        dtype=float,
    )
    train_labels = np.asarray([0, 0, 1, 1], dtype=int)
    train_sets = (
        SimpleNamespace(labels=np.asarray([1, 1])),
        SimpleNamespace(labels=np.asarray([2, 2])),
    )

    full_config = CrossSubjectStimulusConfig(
        feature_transform="source_anova_scale", chance_classes=2
    )
    sqrt_config = CrossSubjectStimulusConfig(
        feature_transform="source_anova_sqrt_scale", chance_classes=2
    )
    _full_transformed, full_metadata = cross_subject._fit_training_feature_transform(  # pylint: disable=protected-access
        train_features, train_sets, full_config, train_labels=train_labels
    )
    transformed, metadata = cross_subject._fit_training_feature_transform(  # pylint: disable=protected-access
        train_features, train_sets, sqrt_config, train_labels=train_labels
    )

    assert metadata["mode"] == "source_anova_sqrt_scale"
    assert metadata["feature_transform_power"] == 0.5
    full_weights = full_metadata["feature_weights"]
    sqrt_weights = metadata["feature_weights"]
    assert 1.0 < sqrt_weights[0] < full_weights[0]
    assert full_weights[1] < sqrt_weights[1] < 1.0
    np.testing.assert_allclose(transformed, train_features * sqrt_weights[None, :])
    np.testing.assert_allclose(
        cross_subject._apply_training_feature_transform(  # pylint: disable=protected-access
            np.ones((1, 2)), metadata
        ),
        sqrt_weights[None, :],
    )
