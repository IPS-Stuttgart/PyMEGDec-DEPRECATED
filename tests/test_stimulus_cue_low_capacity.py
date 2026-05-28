import numpy as np

from pymegdec import stimulus_cue_low_capacity as cue_low_capacity


def test_expert_reliability_scores_can_change_top_k_order():
    similarities = np.asarray([0.90, 0.89, 0.80])
    reliabilities = np.asarray([0.0625, 0.20, 0.30])

    baseline_scores = cue_low_capacity._expert_selection_scores(  # pylint: disable=protected-access
        similarities,
        reliabilities,
        mode="none",
        chance_accuracy=0.0625,
    )
    reliability_scores = cue_low_capacity._expert_selection_scores(  # pylint: disable=protected-access
        similarities,
        reliabilities,
        mode="source-oof-balanced",
        chance_accuracy=0.0625,
    )

    assert baseline_scores.tolist() == similarities.tolist()
    assert int(np.argmax(baseline_scores)) == 0
    assert int(np.argmax(reliability_scores)) == 2


def test_expert_reliability_reweights_similarity_softmax():
    similarities = np.asarray([0.90, 0.89, 0.80])
    reliabilities = np.asarray([0.0625, 0.20, 0.30])

    baseline_weights = cue_low_capacity._expert_weights(  # pylint: disable=protected-access
        similarities,
        reliabilities,
        temperature=0.25,
        reliability_mode="none",
        chance_accuracy=0.0625,
    )
    reliability_weights = cue_low_capacity._expert_weights(  # pylint: disable=protected-access
        similarities,
        reliabilities,
        temperature=0.25,
        reliability_mode="source_oof_balanced",
        chance_accuracy=0.0625,
    )

    np.testing.assert_allclose(baseline_weights.sum(), 1.0)
    np.testing.assert_allclose(reliability_weights.sum(), 1.0)
    assert reliability_weights[2] > baseline_weights[2]
    assert reliability_weights[0] < baseline_weights[0]


def test_expert_fields_record_reliability_provenance():
    fields = cue_low_capacity._expert_fields(  # pylint: disable=protected-access
        4,
        (1, 2),
        np.asarray([0.25, 0.75]),
        np.asarray([0.4, 0.5]),
        np.asarray([0.10, 0.20]),
        np.asarray([0.44, 0.64]),
        8,
        0.25,
        "source-oof-balanced",
    )

    assert fields["cue_expert_reliability"] == "source_oof_balanced"
    assert fields["cue_expert_weights"] == "1:0.25;2:0.75"
    assert fields["cue_expert_reliabilities"] == "1:0.1;2:0.2"
    assert fields["cue_expert_selection_scores"] == "1:0.44;2:0.64"
