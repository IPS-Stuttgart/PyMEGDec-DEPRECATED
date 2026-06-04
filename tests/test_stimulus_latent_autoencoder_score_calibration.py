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
    bias, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    uncalibrated_predictions = classes[np.argmax(scores, axis=1)]
    calibrated_predictions = classes[np.argmax(_apply_score_calibration(scores, bias), axis=1)]
    assert metadata["score_calibration_status"] == "ok"
    assert np.mean(calibrated_predictions == labels) >= np.mean(uncalibrated_predictions == labels)
    assert metadata["score_calibration_validation_balanced_accuracy"] >= metadata["score_calibration_uncalibrated_validation_balanced_accuracy"]
