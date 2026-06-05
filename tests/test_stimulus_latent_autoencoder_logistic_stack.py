import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _apply_score_calibration,
    _fit_validation_score_calibration,
)


def test_validation_logistic_stack_can_recover_class_interaction_signal():
    classes = np.asarray([1, 2, 3], dtype=int)
    labels = np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=int)
    # Argmax collapses to class 1 because the first column is always largest,
    # but the remaining columns contain linearly decodable class information.
    scores = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [2.9, 0.1, 0.0],
            [2.9, 1.1, 0.1],
            [2.9, 0.1, 1.1],
            [2.8, 0.2, 0.0],
            [2.8, 1.2, 0.2],
            [2.8, 0.2, 1.2],
        ],
        dtype=float,
    )
    config = LatentAutoencoderConfig(
        score_calibration="validation_logistic_stack",
        score_calibration_logistic_c_values=(10.0,),
    )

    calibrator, metadata = _fit_validation_score_calibration(scores, labels, classes, config)
    calibrated_scores = _apply_score_calibration(scores, calibrator)
    predictions = classes[np.argmax(calibrated_scores, axis=1)]

    assert metadata["score_calibration_status"] == "ok"
    assert metadata["score_calibration_prior_source"] == "validation_logistic_stack"
    assert metadata["score_calibration_logistic_c"] == 10.0
    assert predictions.tolist() == labels.tolist()


def test_validation_logistic_stack_reports_missing_validation_classes():
    classes = np.asarray([1, 2, 3], dtype=int)
    labels = np.asarray([1, 1, 2, 2], dtype=int)
    scores = np.asarray([[3.0, 0.0, 0.0], [2.9, 0.1, 0.0], [3.0, 1.0, 0.0], [2.9, 1.1, 0.0]], dtype=float)
    config = LatentAutoencoderConfig(score_calibration="validation_logistic_stack")

    calibrator, metadata = _fit_validation_score_calibration(scores, labels, classes, config)

    assert np.allclose(calibrator, np.zeros(3))
    assert metadata["score_calibration_status"] == "missing_validation_classes"
    assert metadata["score_calibration_prior_source"] == "validation_logistic_stack"
