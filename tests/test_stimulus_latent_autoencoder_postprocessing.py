import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _balanced_assignment_predictions,
    _display_label_map,
    _postprocess_predictions,
    _shrunk_source_prior_class_quotas,
    _source_prior_class_quotas,
)


def test_latent_config_carries_prediction_postprocessing_option():
    config = LatentAutoencoderConfig(prediction_postprocessing="source_prior_balanced_assignment")
    assert config.prediction_postprocessing == "source_prior_balanced_assignment"


def test_source_prior_class_quotas_follow_balanced_bush_fold_counts():
    labels = np.repeat(np.asarray([1, 2, 3, 4]), 10)
    quotas = _source_prior_class_quotas(labels, np.asarray([1, 2, 3, 4]), n_test_trials=20)
    assert quotas.tolist() == [5, 5, 5, 5]


def test_balanced_assignment_predictions_respect_quotas():
    classes = np.asarray([1, 2, 3])
    # Argmax alone would over-predict class 1 for the first two rows.  The
    # quota-constrained assignment should still choose the best feasible one-row
    # allocation for each class.
    scores = np.asarray(
        [
            [3.0, 2.0, 0.0],
            [2.9, 2.8, 0.0],
            [2.0, 0.0, 4.0],
        ]
    )
    predictions, objective_delta = _balanced_assignment_predictions(
        scores,
        classes,
        np.asarray([1, 1, 1]),
    )
    assert sorted(predictions.tolist()) == [1, 2, 3]
    assert objective_delta <= 0.0


def test_postprocess_predictions_source_prior_balanced_assignment():
    classes = np.asarray([1, 2, 3, 4])
    source_labels = np.repeat(classes, 12)
    scores = np.asarray(
        [
            [5.0, 4.0, 0.0, 0.0],
            [4.9, 4.8, 0.0, 0.0],
            [4.7, 0.0, 5.0, 0.0],
            [4.6, 0.0, 0.0, 5.0],
        ]
    )
    predictions, metadata = _postprocess_predictions(
        scores,
        classes,
        source_labels,
        LatentAutoencoderConfig(prediction_postprocessing="source_prior_balanced_assignment"),
    )
    assert sorted(predictions.tolist()) == [1, 2, 3, 4]
    assert metadata["prediction_postprocessing_status"] == "ok"
    assert metadata["prediction_postprocessing_quota_source"] == "source_label_prior"


def test_shrunk_source_prior_class_quotas_interpolate_argmax_and_source_prior():
    classes = np.asarray([1, 2, 3, 4])
    source_labels = np.repeat(classes, 10)
    predicted_labels = np.asarray([1, 1, 1, 1, 2, 2, 3, 4])

    argmax_quotas = _shrunk_source_prior_class_quotas(
        source_labels,
        predicted_labels,
        classes,
        n_test_trials=8,
        shrinkage_alpha=0.0,
    )
    source_quotas = _shrunk_source_prior_class_quotas(
        source_labels,
        predicted_labels,
        classes,
        n_test_trials=8,
        shrinkage_alpha=1.0,
    )

    assert argmax_quotas.tolist() == [4, 2, 1, 1]
    assert source_quotas.tolist() == [2, 2, 2, 2]


def test_postprocess_predictions_validation_guarded_shrunk_assignment_selects_partial_alpha():
    classes = np.asarray([1, 2, 3, 4])
    source_labels = np.repeat(classes, 12)
    scores = np.asarray(
        [
            [5.0, 4.0, 0.0, 0.0],
            [4.9, 4.8, 0.0, 0.0],
            [4.7, 0.0, 5.0, 0.0],
            [4.6, 0.0, 0.0, 5.0],
        ]
    )
    predictions, metadata = _postprocess_predictions(
        scores,
        classes,
        source_labels,
        LatentAutoencoderConfig(
            prediction_postprocessing="validation_guarded_shrunk_source_prior_balanced_assignment",
            prediction_postprocessing_shrinkage_alphas=(0.0, 1.0),
        ),
        validation_scores=scores,
        validation_labels=np.asarray([1, 2, 3, 4]),
    )
    assert sorted(predictions.tolist()) == [1, 2, 3, 4]
    assert metadata["prediction_postprocessing_status"] == "ok"
    assert metadata["prediction_postprocessing_quota_source"] == "shrunk_source_label_prior"
    assert metadata["prediction_postprocessing_shrinkage_alpha"] == 1.0


def test_display_label_map_does_not_shift_one_based_labels():
    assert _display_label_map(np.asarray([1, 2, 3])) == {1: 1, 2: 2, 3: 3}
    assert _display_label_map(np.asarray([0, 1, 2])) == {0: 1, 1: 2, 2: 3}
