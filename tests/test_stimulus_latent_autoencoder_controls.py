import math

import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _final_refit_epochs,
    _prediction_balance_score,
    _split_source_participants,
    _validation_selection_metrics,
)


def test_split_source_participants_spread_does_not_always_take_tail():
    train, validation = _split_source_participants(tuple(range(1, 11)), 2, strategy="spread")

    assert validation != (9, 10)
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)
    assert set(train).union(validation) == set(range(1, 11))


def test_split_source_participants_rotating_uses_anchor():
    _train, validation_a = _split_source_participants(tuple(range(1, 8)), 2, strategy="rotating", anchor=1)
    _train, validation_b = _split_source_participants(tuple(range(1, 8)), 2, strategy="rotating", anchor=2)

    assert validation_a != validation_b
    assert len(validation_a) == 2
    assert len(validation_b) == 2


def test_final_refit_epochs_can_apply_floor_and_multiplier():
    config = LatentAutoencoderConfig(epochs=30, final_epoch_multiplier=2.0, final_min_epochs=8)

    assert _final_refit_epochs(3, config) == 8
    assert _final_refit_epochs(8, config) == 16
    assert _final_refit_epochs(20, config) == 30


def test_prediction_balance_score_detects_collapse():
    classes = np.asarray([1, 2, 3, 4])

    collapsed = _prediction_balance_score(np.asarray([1, 1, 1, 1]), classes)
    balanced = _prediction_balance_score(np.asarray([1, 2, 3, 4]), classes)

    assert 0.0 <= collapsed < balanced <= 1.0


def test_validation_selection_metrics_rank_balance_variant_rewards_balanced_predictions():
    labels = np.asarray([1, 2, 3, 4])
    classes = np.asarray([1, 2, 3, 4])
    balanced_scores = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 0.0, 0.0],
            [0.0, 2.5, 2.0, 1.0],
            [0.0, 1.0, 2.5, 2.0],
        ]
    )
    collapsed_scores = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
        ]
    )

    balanced = _validation_selection_metrics(labels, balanced_scores, classes, "balanced_top2_top3_rank_balance")
    collapsed = _validation_selection_metrics(labels, collapsed_scores, classes, "balanced_top2_top3_rank_balance")

    assert math.isfinite(balanced["selection_score"])
    assert balanced["selection_score"] > collapsed["selection_score"]
