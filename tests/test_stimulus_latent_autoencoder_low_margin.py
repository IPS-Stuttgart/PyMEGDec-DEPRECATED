import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _balanced_assignment_predictions,
    _low_margin_balanced_assignment_predictions,
    _validation_balanced_assignment_candidates,
)


def test_low_margin_assignment_respects_quotas_and_keeps_high_margin_rows():
    classes = np.asarray([1, 2, 3])
    scores = np.asarray(
        [
            [5.0, 1.0, 0.0],
            [4.8, 1.0, 0.0],
            [4.6, 1.0, 0.0],
            [2.0, 1.9, 0.0],
            [1.8, 2.0, 0.0],
            [1.8, 1.7, 2.0],
        ]
    )
    quotas = np.asarray([2, 2, 2])

    labels, objective_delta, fixed_predictions = _low_margin_balanced_assignment_predictions(
        scores,
        classes,
        quotas,
        margin_threshold=0.5,
    )

    assert fixed_predictions == 2
    assert labels[:2].tolist() == [1, 1]
    assert {label: int(np.sum(labels == label)) for label in classes} == {1: 2, 2: 2, 3: 2}
    assert objective_delta <= 0.0


def test_low_margin_assignment_releases_overquota_lowest_margin_fixed_rows():
    classes = np.asarray([1, 2])
    scores = np.asarray(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [1.1, 0.0],
        ]
    )
    quotas = np.asarray([2, 2])

    labels, _objective_delta, fixed_predictions = _low_margin_balanced_assignment_predictions(
        scores,
        classes,
        quotas,
        margin_threshold=0.5,
    )

    assert fixed_predictions == 2
    assert labels.tolist().count(1) == 2
    assert labels.tolist().count(2) == 2


def test_validation_candidates_include_low_margin_variants():
    classes = np.asarray([1, 2, 3])
    scores = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.1, 0.2, 0.0],
            [0.1, 0.0, 0.2],
            [0.0, 0.2, 0.1],
            [0.0, 0.1, 0.2],
        ]
    )
    labels = np.asarray([1, 1, 2, 2, 3, 3])
    source_labels = np.asarray([1, 1, 2, 2, 3, 3])
    config = LatentAutoencoderConfig(prediction_postprocessing_margin_thresholds=(0.0, 0.5))

    rows = _validation_balanced_assignment_candidates(scores, labels, source_labels, classes, config)

    assert any(row["selected_method"] == "source_prior_low_margin_balanced_assignment" for row in rows)
    assert all("margin_threshold" in row for row in rows)
    # The original full assignment path is still available for comparison.
    full_labels, _objective_delta = _balanced_assignment_predictions(scores, classes, np.asarray([2, 2, 2]))
    assert full_labels.shape == labels.shape
