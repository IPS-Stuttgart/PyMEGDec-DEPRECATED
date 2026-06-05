from collections import Counter

import math
import unittest

import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _balanced_epoch_indices,
    _effective_ensemble_seeds,
    _final_refit_epochs,
    _gradient_reverse,
    _make_model_class,
    _prediction_balance_score,
    _postprocess_predictions,
    _split_source_participants,
    _supervised_contrastive_loss,
    _validation_selection_metrics,
)


def _import_torch_or_skip():
    try:
        import torch
    except ImportError as exc:
        raise unittest.SkipTest("torch is not installed") from exc
    return torch


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


def test_effective_ensemble_seeds_defaults_to_seed_and_dedupes():
    assert _effective_ensemble_seeds(LatentAutoencoderConfig(seed=7)) == (7,)
    assert _effective_ensemble_seeds(LatentAutoencoderConfig(seed=7, ensemble_seeds=(1, 2, 1, 3))) == (
        1,
        2,
        3,
    )


def test_prediction_balance_score_detects_collapse():
    classes = np.asarray([1, 2, 3, 4])

    collapsed = _prediction_balance_score(np.asarray([1, 1, 1, 1]), classes)
    balanced = _prediction_balance_score(np.asarray([1, 2, 3, 4]), classes)

    assert 0.0 <= collapsed < balanced <= 1.0


def test_source_prior_balanced_assignment_postprocessing_uses_source_quotas():
    classes = np.asarray([1, 2, 3])
    source_labels = np.asarray([1, 1, 2, 2, 3, 3])
    scores = np.asarray(
        [
            [4.0, 3.0, 0.0],
            [4.0, 2.9, 0.0],
            [4.0, 2.8, 0.0],
            [4.0, 0.0, 3.0],
            [4.0, 0.0, 2.9],
            [4.0, 0.0, 2.8],
        ]
    )
    config = LatentAutoencoderConfig(prediction_postprocessing="source_prior_balanced_assignment")

    predictions, metadata = _postprocess_predictions(scores, classes, source_labels, config)

    assert metadata["prediction_postprocessing_status"] == "ok"
    assert metadata["prediction_postprocessing_quota_source"] == "source_label_prior"
    assert Counter(predictions.tolist()) == Counter({1: 2, 2: 2, 3: 2})


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


def test_balanced_epoch_indices_interleaves_classes_and_preserves_rows():
    labels = np.asarray([0] * 5 + [1] * 3 + [2] * 4)
    rng = np.random.default_rng(0)

    order = _balanced_epoch_indices(labels, rng=rng)

    assert sorted(order.tolist()) == list(range(labels.shape[0]))
    first_cycle_labels = set(labels[order[:3]].tolist())
    assert first_cycle_labels == {0, 1, 2}
    assert np.bincount(labels[order], minlength=3).tolist() == [5, 3, 4]


def test_gradient_reverse_flips_and_scales_gradient_when_torch_is_available():
    torch = _import_torch_or_skip()
    value = torch.tensor([1.0, -2.0], requires_grad=True)

    reversed_value = _gradient_reverse(value, 0.25)
    reversed_value.sum().backward()

    assert torch.allclose(value.grad, torch.tensor([-0.25, -0.25]))


def test_latent_model_maps_sparse_participant_ids_for_subject_adversary_when_torch_is_available():
    torch = _import_torch_or_skip()
    Model = _make_model_class()
    model = Model(
        n_features=4,
        n_classes=3,
        subject_ids=(2, 4, 8),
        hidden_dim=8,
        latent_dim=5,
        dropout=0.0,
    )

    targets = model.subject_targets(torch.tensor([8, 2, 4, 8]))

    assert targets.tolist() == [2, 0, 1, 2]


def test_supervised_contrastive_loss_rewards_same_class_latent_neighbors():
    torch = _import_torch_or_skip()

    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    clustered_latent = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    mixed_latent = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )

    clustered_loss = _supervised_contrastive_loss(clustered_latent, labels, temperature=0.2)
    mixed_loss = _supervised_contrastive_loss(mixed_latent, labels, temperature=0.2)

    assert torch.isfinite(clustered_loss)
    assert clustered_loss < mixed_loss
