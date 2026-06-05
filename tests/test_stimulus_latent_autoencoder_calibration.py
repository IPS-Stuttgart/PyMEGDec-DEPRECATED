import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _apply_score_calibration,
    _fit_validation_score_calibration,
    _validation_selection_metrics,
)


def test_validation_prediction_bias_penalizes_argmax_collapse():
    classes = np.asarray([1, 2, 3], dtype=int)
    labels = np.asarray([1, 2, 2, 3, 3, 3], dtype=int)
    # Every row initially predicts class 1, even though classes 2 and 3 are
    # present in validation labels.  The hard-prediction prior should therefore
    # push class 1 down and the missing classes up.
    scores = np.asarray(
        [
            [4.0, 3.0, 2.0],
            [4.0, 3.8, 1.0],
            [4.0, 3.7, 1.2],
            [4.0, 1.0, 3.8],
            [4.0, 1.0, 3.7],
            [4.0, 1.5, 3.6],
        ],
        dtype=float,
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_prediction_bias",
        score_calibration_alphas=(0.0, 0.5, 1.0, 2.0),
        score_calibration_smoothing=0.01,
    )

    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_prior_source"] == "argmax_predictions"
    assert metadata["score_calibration_alpha"] > 0.0
    assert bias[0] < 0.0
    assert bias[1] > 0.0
    assert bias[2] > 0.0
    uncalibrated = classes[np.argmax(scores, axis=1)]
    calibrated = classes[np.argmax(scores + bias, axis=1)]
    assert balanced_accuracy_score(labels, calibrated) >= balanced_accuracy_score(labels, uncalibrated)


def test_validation_argmax_class_bias_guarded_uses_hard_prior_with_guard():
    classes = np.asarray([1, 2, 3], dtype=int)
    labels = np.asarray([1, 2, 2, 3, 3, 3], dtype=int)
    # Every validation row initially argmaxes to class 1.  The new guarded hard
    # argmax calibration should use the same collapse-sensitive prior source as
    # validation_prediction_bias, while optimizing the guarded selection metric.
    scores = np.asarray(
        [
            [4.0, 3.0, 2.0],
            [4.0, 3.8, 1.0],
            [4.0, 3.7, 1.2],
            [4.0, 1.0, 3.8],
            [4.0, 1.0, 3.7],
            [4.0, 1.5, 3.6],
        ],
        dtype=float,
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_argmax_class_bias_guarded",
        score_calibration_alphas=(0.0, 0.5, 1.0, 2.0),
        score_calibration_smoothing=0.01,
        score_calibration_selection_metric="balanced_top2_top3_rank_balance",
        score_calibration_guard_tolerance=0.0,
    )

    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_prior_source"] == "argmax_predictions"
    assert metadata["score_calibration_predicted_prior_source"] == "argmax"
    assert metadata["score_calibration_alpha"] > 0.0
    assert metadata["score_calibration_validation_balanced_accuracy"] >= metadata[
        "score_calibration_uncalibrated_validation_balanced_accuracy"
    ]
    assert bias[0] < 0.0
    assert bias[1] > 0.0
    assert bias[2] > 0.0


def test_validation_score_standardize_can_correct_classwise_score_scale_bias():
    classes = np.asarray([1, 2, 3])
    labels = np.asarray([1, 1, 2, 2, 3, 3])
    scores = np.asarray(
        [
            [3.0, 2.0, 1.0],
            [2.8, 1.5, 1.0],
            [3.0, 4.0, 1.0],
            [2.9, 4.1, 1.0],
            [3.0, 1.0, 2.2],
            [2.8, 1.0, 2.1],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_score_standardize",
        score_calibration_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    )

    calibration, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    calibrated = _apply_score_calibration(scores, calibration)

    raw_balanced = _validation_selection_metrics(labels, scores, classes, "balanced_accuracy")["balanced_accuracy"]
    calibrated_balanced = _validation_selection_metrics(labels, calibrated, classes, "balanced_accuracy")[
        "balanced_accuracy"
    ]
    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_predicted_prior_source"] == "score_standardization"
    assert metadata["score_calibration_alpha"] > 0.0
    assert calibrated_balanced > raw_balanced


def test_validation_score_standardize_guard_rejects_balanced_accuracy_drop():
    classes = np.asarray([1, 2, 3])
    labels = np.asarray([1, 2, 3])
    scores = np.asarray(
        [
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_score_standardize_guarded",
        score_calibration_alphas=(0.0, 1.0),
        score_calibration_guard_tolerance=0.0,
    )

    calibration, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    calibrated = _apply_score_calibration(scores, calibration)

    assert metadata["score_calibration_alpha"] == pytest.approx(0.0)
    np.testing.assert_allclose(calibrated, scores)
