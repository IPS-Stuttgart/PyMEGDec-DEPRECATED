from collections import Counter

import math
import unittest

import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _balanced_epoch_indices,
    _class_balanced_focal_cross_entropy,
    _class_margin_loss,
    _confidence_penalty,
    _effective_ensemble_seeds,
    _final_refit_epochs,
    _gradient_reverse,
    _logit_mean_center_loss,
    _make_model_class,
    _prediction_balance_score,
    _postprocess_predictions,
    _split_source_participants,
    _soft_macro_recall_loss,
    _soft_worst_class_recall_loss,
    _supervised_contrastive_loss,
    _validation_selection_metrics,
)


def _import_torch_or_skip():
    try:
        import torch
    except ImportError as exc:
        raise unittest.SkipTest("torch is not installed") from exc
    return torch


def test_latent_config_defaults_to_rotating_validation_sources():
    assert LatentAutoencoderConfig().validation_source_strategy == "rotating"


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


def test_logit_mean_center_loss_penalizes_batch_class_offsets_when_torch_is_available():
    torch = _import_torch_or_skip()
    collapsed = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    balanced = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    assert _logit_mean_center_loss(collapsed) > _logit_mean_center_loss(balanced)
    assert torch.isclose(_logit_mean_center_loss(balanced), torch.tensor(0.0))


def test_confidence_penalty_penalizes_peaked_scores_when_torch_is_available():
    torch = _import_torch_or_skip()
    peaked = torch.tensor([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0]], dtype=torch.float32)
    flat = torch.zeros((2, 3), dtype=torch.float32)

    assert _confidence_penalty(peaked) > _confidence_penalty(flat)
    assert torch.isclose(_confidence_penalty(flat), torch.tensor(0.0))


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


def test_soft_worst_class_recall_loss_targets_collapsed_classes_when_torch_is_available():
    torch = _import_torch_or_skip()
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    collapsed = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    separated = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=torch.float32,
    )

    collapsed_loss = _soft_worst_class_recall_loss(collapsed, labels)
    separated_loss = _soft_worst_class_recall_loss(separated, labels)

    assert collapsed_loss > separated_loss
    assert separated_loss < torch.tensor(0.05)


def test_guarded_source_prior_assignment_policy_uses_source_validation_gain():
    classes = np.asarray([1, 2, 3, 4])
    source_labels = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    validation_labels = np.asarray([1, 2, 3, 4])
    # Argmax collapses onto classes 1 and 3, but a one-per-class assignment
    # recovers the full validation label set from the second-best scores.
    validation_scores = np.asarray(
        [
            [4.0, 3.0, 0.0, 0.0],
            [4.0, 3.5, 0.0, 0.0],
            [0.0, 0.0, 4.0, 3.0],
            [0.0, 0.0, 4.0, 3.5],
        ]
    )
    config = LatentAutoencoderConfig(
        prediction_postprocessing="validation_guarded_source_prior_balanced_assignment"
    )

    _predicted, metadata = _postprocess_predictions(
        validation_scores,
        classes,
        source_labels,
        config,
        validation_scores=validation_scores,
        validation_labels=validation_labels,
    )

    assert metadata["prediction_postprocessing_status"] == "ok"
    assert metadata["prediction_postprocessing_apply"] is True
    assert metadata["prediction_postprocessing_validation_balanced_accuracy"] > metadata[
        "prediction_postprocessing_uncalibrated_validation_balanced_accuracy"
    ]


def test_guarded_source_prior_assignment_falls_back_without_validation_support():
    classes = np.asarray([1, 2, 3, 4])
    scores = np.asarray([[4.0, 3.0, 0.0, 0.0], [4.0, 3.5, 0.0, 0.0]])
    source_labels = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    config = LatentAutoencoderConfig(
        prediction_postprocessing="validation_guarded_source_prior_balanced_assignment"
    )

    predictions, metadata = _postprocess_predictions(
        scores,
        classes,
        source_labels,
        config,
    )

    assert predictions.tolist() == [1, 1]
    assert metadata["prediction_postprocessing_status"] == "no_validation"
    assert metadata["prediction_postprocessing_apply"] is False


def test_gradient_reverse_flips_and_scales_gradient_when_torch_is_available():
    torch = _import_torch_or_skip()
    value = torch.tensor([1.0, -2.0], requires_grad=True)

    reversed_value = _gradient_reverse(value, 0.25)
    reversed_value.sum().backward()

    assert torch.allclose(value.grad, torch.tensor([-0.25, -0.25]))


def test_focal_loss_gamma_zero_matches_weighted_cross_entropy_when_torch_is_available():
    torch = _import_torch_or_skip()
    logits = torch.tensor(
        [
            [3.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    weight = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    focal = _class_balanced_focal_cross_entropy(
        logits,
        targets,
        weight=weight,
        label_smoothing=0.0,
        focal_gamma=0.0,
    )
    expected = torch.nn.functional.cross_entropy(logits, targets, weight=weight)

    assert torch.allclose(focal, expected)


def test_positive_focal_loss_gamma_downweights_easy_examples_when_torch_is_available():
    torch = _import_torch_or_skip()
    logits = torch.tensor([[8.0, -4.0], [0.2, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)
    weight = torch.ones(2, dtype=torch.float32)

    ce = _class_balanced_focal_cross_entropy(logits, targets, weight=weight, label_smoothing=0.0, focal_gamma=0.0)
    focal = _class_balanced_focal_cross_entropy(logits, targets, weight=weight, label_smoothing=0.0, focal_gamma=2.0)

    assert torch.isfinite(focal)
    assert focal < ce


def test_class_margin_loss_penalizes_insufficient_true_class_margin_when_torch_is_available():
    torch = _import_torch_or_skip()
    targets = torch.tensor([0, 1], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    bad_logits = torch.tensor(
        [
            [1.1, 1.0, 0.0],
            [0.0, 1.0, 1.2],
        ],
        dtype=torch.float32,
    )

    good_loss = _class_margin_loss(good_logits, targets, margin=1.0)
    bad_loss = _class_margin_loss(bad_logits, targets, margin=1.0)

    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert good_loss == 0.0
    assert bad_loss > good_loss


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


def test_soft_macro_recall_loss_rewards_per_class_correct_scores():
    torch = _import_torch_or_skip()

    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [3.5, 0.2, 0.0],
            [0.0, 4.0, 0.0],
            [0.1, 3.5, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.1, 3.5],
        ],
        dtype=torch.float32,
    )
    collapsed_logits = torch.tensor([[4.0, 0.0, 0.0]] * 6, dtype=torch.float32)

    good_loss = _soft_macro_recall_loss(good_logits, labels)
    collapsed_loss = _soft_macro_recall_loss(collapsed_logits, labels)

    assert torch.isfinite(good_loss)
    assert torch.isfinite(collapsed_loss)
    assert good_loss < collapsed_loss


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


def test_guarded_source_prior_assignment_applies_when_validation_improves():
    classes = np.asarray([1, 2, 3], dtype=int)
    source_labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)
    scores = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [3.9, 0.0, 0.0],
            [4.0, 3.99, 0.0],
            [4.0, 3.98, 0.0],
            [4.0, 0.0, 3.99],
            [4.0, 0.0, 3.98],
        ],
        dtype=float,
    )
    config = LatentAutoencoderConfig(
        prediction_postprocessing="validation_guarded_source_prior_balanced_assignment"
    )

    predicted, metadata = _postprocess_predictions(
        scores,
        classes,
        source_labels,
        config,
        validation_scores=scores,
        validation_labels=labels,
    )

    assert predicted.tolist() == labels.tolist()
    assert metadata["prediction_postprocessing_status"] == "ok"
    assert metadata["prediction_postprocessing_quota_source"] == "source_label_prior"
    assert metadata["prediction_postprocessing_validation_balanced_accuracy"] == 1.0


def test_guarded_source_prior_assignment_rejects_validation_regression():
    classes = np.asarray([1, 2, 3], dtype=int)
    source_labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)
    scores = np.asarray([[4.0, 0.0, 0.0]] * 6, dtype=float)
    labels = np.asarray([1, 1, 1, 1, 1, 1], dtype=int)
    config = LatentAutoencoderConfig(
        prediction_postprocessing="validation_guarded_source_prior_balanced_assignment"
    )

    predicted, metadata = _postprocess_predictions(
        scores,
        classes,
        source_labels,
        config,
        validation_scores=scores,
        validation_labels=labels,
    )

    assert predicted.tolist() == labels.tolist()
    assert metadata["prediction_postprocessing_status"] == "guard_rejected"
