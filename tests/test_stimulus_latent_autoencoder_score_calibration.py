import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _apply_score_calibration,
    _bounded_label_smoothing,
    _fit_validation_score_calibration,
    _parse_float_sequence,
)


def test_parse_float_sequence_accepts_commas_and_semicolons():
    assert _parse_float_sequence("0,0.25;0.5") == (0.0, 0.25, 0.5)


def test_bounded_label_smoothing_clamps_to_valid_cross_entropy_range():
    assert _bounded_label_smoothing(-0.25) == 0.0
    assert _bounded_label_smoothing(0.05) == 0.05
    assert _bounded_label_smoothing(1.5) == 0.999


def _fit_calibrated_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    uncalibrated_predictions = classes[np.argmax(scores, axis=1)]
    calibrated_predictions = classes[np.argmax(_apply_score_calibration(scores, bias), axis=1)]
    return uncalibrated_predictions, calibrated_predictions, metadata


def test_validation_class_bias_calibration_improves_validation_balance():
    classes = np.asarray([1, 2])
    labels = np.asarray([1, 1, 2, 2])
    scores = np.asarray(
        [
            [0.9, 0.0],
            [0.8, 0.0],
            [0.1, 0.0],
            [0.0, 0.8],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_class_bias",
        score_calibration_alphas=(0.0, 0.5, 1.0, 2.0),
        score_calibration_smoothing=0.0,
    )
    uncalibrated_predictions, calibrated_predictions, metadata = _fit_calibrated_predictions(
        scores,
        labels,
        classes,
        config,
    )
    assert metadata["score_calibration_status"] == "ok"
    assert np.mean(calibrated_predictions == labels) >= np.mean(uncalibrated_predictions == labels)
    assert metadata["score_calibration_validation_balanced_accuracy"] >= metadata["score_calibration_uncalibrated_validation_balanced_accuracy"]


def test_guarded_validation_class_bias_uses_rank_balance_selection_without_balanced_regression():
    classes = np.asarray([1, 2, 3])
    labels = np.asarray([1, 1, 2, 2, 3, 3])
    scores = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.9, 0.0],
            [2.0, 1.8, 0.0],
            [2.0, 0.0, 1.9],
            [2.0, 0.0, 1.8],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_class_bias_guarded",
        score_calibration_alphas=(0.0, 0.5, 1.0, 2.0),
        score_calibration_smoothing=0.0,
        score_calibration_selection_metric="balanced_top2_top3_rank_balance",
        score_calibration_guard_tolerance=0.0,
    )

    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    calibrated_predictions = classes[np.argmax(_apply_score_calibration(scores, bias), axis=1)]
    uncalibrated_predictions = classes[np.argmax(scores, axis=1)]

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_alpha"] > 0.0
    assert metadata["score_calibration_validation_balanced_accuracy"] >= metadata["score_calibration_uncalibrated_validation_balanced_accuracy"]
    assert metadata["score_calibration_validation_selection_score"] >= metadata["score_calibration_uncalibrated_validation_selection_score"]
    assert len(set(calibrated_predictions.tolist())) > len(set(uncalibrated_predictions.tolist()))


def test_validation_argmax_class_bias_calibration_targets_argmax_collapse():
    classes = np.asarray([1, 2])
    labels = np.asarray([1, 1, 2, 2])
    scores = np.asarray(
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [0.4, 0.3],
            [0.4, 0.35],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_argmax_class_bias",
        score_calibration_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        score_calibration_smoothing=0.1,
    )
    uncalibrated_predictions, calibrated_predictions, metadata = _fit_calibrated_predictions(
        scores,
        labels,
        classes,
        config,
    )
    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_predicted_prior_source"] == "argmax"
    assert np.unique(uncalibrated_predictions).tolist() == [1]
    assert np.mean(calibrated_predictions == labels) > np.mean(uncalibrated_predictions == labels)
    assert metadata["score_calibration_validation_balanced_accuracy"] > metadata["score_calibration_uncalibrated_validation_balanced_accuracy"]


def test_guarded_validation_argmax_class_bias_uses_argmax_prior_without_balanced_regression():
    classes = np.asarray([1, 2, 3])
    labels = np.asarray([1, 1, 2, 2, 3, 3])
    scores = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.9, 0.0],
            [2.0, 1.8, 0.0],
            [2.0, 0.0, 1.9],
            [2.0, 0.0, 1.8],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_argmax_class_bias_guarded",
        score_calibration_alphas=(0.0, 0.5, 1.0, 2.0),
        score_calibration_smoothing=0.1,
        score_calibration_selection_metric="balanced_top2_top3_rank_balance",
        score_calibration_guard_tolerance=0.0,
    )

    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    calibrated_predictions = classes[np.argmax(_apply_score_calibration(scores, bias), axis=1)]
    uncalibrated_predictions = classes[np.argmax(scores, axis=1)]

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_predicted_prior_source"] == "argmax"
    assert metadata["score_calibration_alpha"] > 0.0
    assert metadata["score_calibration_validation_balanced_accuracy"] >= metadata["score_calibration_uncalibrated_validation_balanced_accuracy"]
    assert len(set(calibrated_predictions.tolist())) > len(set(uncalibrated_predictions.tolist()))


def test_validation_confusion_blend_reassigns_systematic_confusions():
    classes = np.asarray([1, 2, 3])
    labels = np.asarray([1, 1, 2, 2, 3, 3])
    scores = np.asarray(
        [
            [0.1, 2.0, 0.0],
            [0.2, 1.8, 0.0],
            [0.0, 0.1, 2.0],
            [0.0, 0.2, 1.8],
            [2.0, 0.0, 0.1],
            [1.8, 0.0, 0.2],
        ]
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_confusion_blend",
        score_calibration_alphas=(0.0, 0.5, 1.0),
        score_calibration_confusion_smoothing=0.0,
    )
    calibration, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    uncalibrated_predictions = classes[np.argmax(scores, axis=1)]
    calibrated_predictions = classes[np.argmax(_apply_score_calibration(scores, calibration), axis=1)]

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_prior_source"] == "validation_confusion_map"
    assert np.mean(calibrated_predictions == labels) > np.mean(uncalibrated_predictions == labels)
