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
    tuning = cue_low_capacity._expert_tuning_fields(  # pylint: disable=protected-access
        enabled=True,
        top_k=4,
        temperature=0.1,
        top_k_grid=(4, 8),
        temperature_grid=(0.1, 0.25),
        inner_balanced_accuracy=0.3,
        n_inner_folds=6,
    )
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
        tuning,
    )

    assert fields["cue_expert_reliability"] == "source_oof_balanced"
    assert fields["cue_expert_weights"] == "1:0.25;2:0.75"
    assert fields["cue_expert_reliabilities"] == "1:0.1;2:0.2"
    assert fields["cue_expert_selection_scores"] == "1:0.44;2:0.64"
    assert fields["cue_expert_tuned"] is True
    assert fields["cue_expert_tuned_top_k"] == 4
    assert fields["cue_expert_top_k_grid"] == "4,8"


def test_source_expert_reliabilities_use_candidate_subset_only():
    transfer = np.asarray(
        [
            [np.nan, 0.20, 0.40],
            [0.50, np.nan, 0.70],
            [0.10, 0.30, np.nan],
        ]
    )

    reliabilities = cue_low_capacity._source_expert_reliabilities(  # pylint: disable=protected-access
        (1, 3),
        transfer,
        reliability_participants=(1, 2, 3),
        mode="source_oof_balanced",
    )

    np.testing.assert_allclose(reliabilities, np.asarray([0.40, 0.10]))


def test_select_best_expert_hyperparameters_prefers_inner_score_then_defaults():
    rows = [
        {"top_k": 4, "temperature": 0.10, "validation_participant": 1, "balanced_accuracy": 0.20},
        {"top_k": 4, "temperature": 0.10, "validation_participant": 2, "balanced_accuracy": 0.20},
        {"top_k": 8, "temperature": 0.25, "validation_participant": 1, "balanced_accuracy": 0.20},
        {"top_k": 8, "temperature": 0.25, "validation_participant": 2, "balanced_accuracy": 0.20},
        {"top_k": 12, "temperature": 0.50, "validation_participant": 1, "balanced_accuracy": 0.10},
        {"top_k": 12, "temperature": 0.50, "validation_participant": 2, "balanced_accuracy": 0.10},
    ]

    selected = cue_low_capacity._select_best_expert_hyperparameters(rows)  # pylint: disable=protected-access

    assert selected["top_k"] == 8
    assert selected["temperature"] == 0.25
    assert selected["inner_balanced_accuracy"] == 0.20
    assert selected["n_inner_folds"] == 2


def test_grid_parsers_normalize_csv_values():
    assert cue_low_capacity._parse_int_grid("4, 8,12") == (4, 8, 12)  # pylint: disable=protected-access
    assert cue_low_capacity._parse_float_grid("0.1, 0.25") == (0.1, 0.25)  # pylint: disable=protected-access


def test_expert_summary_fields_count_tuned_settings():
    row = {}

    cue_low_capacity._add_expert_summary_fields(  # pylint: disable=protected-access
        row,
        [
            {"cue_expert_reliability": "source_oof_balanced", "cue_expert_tuned": True, "cue_expert_tuned_top_k": 4},
            {"cue_expert_reliability": "source_oof_balanced", "cue_expert_tuned": True, "cue_expert_tuned_top_k": 8},
            {"cue_expert_reliability": "none", "cue_expert_tuned": False, "cue_expert_tuned_top_k": 8},
        ],
    )

    assert row["cue_expert_reliability_counts"] == "none:1;source_oof_balanced:2"
    assert row["cue_expert_tuned_counts"] == "False:1;True:2"
    assert row["cue_expert_tuned_top_k_counts"] == "4:1;8:2"
