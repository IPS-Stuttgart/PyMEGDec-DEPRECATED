import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pymegdec.stimulus_latent_autoencoder import LatentAutoencoderConfig, _fit_validation_score_calibration


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
