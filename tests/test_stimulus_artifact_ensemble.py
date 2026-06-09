from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from pymegdec.stimulus_artifact_ensemble import PredictionSource, ensemble_prediction_sources, parse_ensemble_spec, resolve_prediction_csv


def _row(participant: int, trial_index: int, true_label: int, predicted_label: int, *, true_label_rank: float = 1.0) -> dict[str, str]:
    return {
        "test_participant": str(participant),
        "test_trial_index": str(trial_index),
        "true_label": str(true_label),
        "predicted_label": str(predicted_label),
        "true_stimulus": str(true_label + 1),
        "predicted_stimulus": str(predicted_label + 1),
        "true_label_rank": str(true_label_rank),
    }


def _source(name: str, rows: list[dict[str, str]]) -> PredictionSource:
    return PredictionSource(name, Path(f"{name}.csv"), rows)


def _scored_row(true_label: int, predicted_label: int, class_0_score: float, class_1_score: float) -> dict[str, str]:
    return {
        **_row(1, 1, true_label, predicted_label, true_label_rank=1.0 if true_label == predicted_label else 2.0),
        "score_class_0": f"{class_0_score:.2f}",
        "score_class_1": f"{class_1_score:.2f}",
    }


def _stimulus_scored_row(true_label: int, predicted_label: int, stimulus_1_score: float, stimulus_2_score: float) -> dict[str, str]:
    return {
        **_row(1, 1, true_label, predicted_label, true_label_rank=1.0 if true_label == predicted_label else 2.0),
        "score_1": f"{stimulus_1_score:.2f}",
        "score_2": f"{stimulus_2_score:.2f}",
    }


def _multi_scored_row(predicted_label: int, scores: list[float]) -> dict[str, str]:
    row = _row(
        1,
        1,
        0,
        predicted_label,
        true_label_rank=1.0 if predicted_label == 0 else 2.0,
    )
    row.update(
        {
            f"score_class_{class_index}": f"{score:.6f}"
            for class_index, score in enumerate(scores)
        }
    )
    return row


def _multi_two_class_scored_row(
    trial_index: int,
    true_label: int,
    predicted_label: int,
    class_0_score: float,
    class_1_score: float,
) -> dict[str, str]:
    row = _row(
        1,
        trial_index,
        true_label,
        predicted_label,
        true_label_rank=1.0 if true_label == predicted_label else 2.0,
    )
    row["score_class_0"] = f"{class_0_score:.2f}"
    row["score_class_1"] = f"{class_1_score:.2f}"
    return row


def _two_class_scored_row(
    trial_index: int,
    true_label: int,
    predicted_label: int,
    class_0_score: float,
    class_1_score: float,
) -> dict[str, str]:
    row = _scored_row(true_label, predicted_label, class_0_score, class_1_score)
    row["test_trial_index"] = str(trial_index)
    return row


def _participant_scored_row(
    participant: int,
    true_label: int,
    predicted_label: int,
    class_0_score: float,
    class_1_score: float,
) -> dict[str, str]:
    row = _scored_row(true_label, predicted_label, class_0_score, class_1_score)
    row["test_participant"] = str(participant)
    return row


def _participant_three_score_row(
    participant: int,
    true_label: int,
    predicted_label: int,
    class_0_score: float,
    class_1_score: float,
    class_2_score: float,
) -> dict[str, str]:
    scores = (class_0_score, class_1_score, class_2_score)
    ranked_labels = sorted(range(len(scores)), key=lambda label: (-scores[label], label))
    row = _row(
        participant,
        1,
        true_label,
        predicted_label,
        true_label_rank=float(ranked_labels.index(true_label) + 1),
    )
    for label, score in enumerate(scores):
        row[f"score_class_{label}"] = f"{score:.2f}"
    return row


def _ranked_row(true_label: int, predicted_label: int, class_0_rank: float, class_1_rank: float) -> dict[str, str]:
    return {
        **_row(1, 1, true_label, predicted_label, true_label_rank=class_0_rank if true_label == 0 else class_1_rank),
        "rank_class_0": f"{class_0_rank:.1f}",
        "rank_class_1": f"{class_1_rank:.1f}",
    }


class TestStimulusArtifactEnsemble(unittest.TestCase):
    def test_parse_ensemble_spec_requires_named_sources(self) -> None:
        self.assertEqual(parse_ensemble_spec("compact_plus=compact,finetune"), ("compact_plus", ("compact", "finetune")))
        with self.assertRaisesRegex(ValueError, "name=source"):
            parse_ensemble_spec("compact,finetune")

    def test_resolve_prediction_csv_prefers_nested_matrix_predictions(self) -> None:
        path = Path("artifact_probe_26870015355")
        if path.exists():
            self.assertEqual(resolve_prediction_csv(path).name, "nested_matrix_predictions.csv")

    def test_single_source_preserves_source_true_label_rank_for_topk_baseline(self) -> None:
        compact = _source(
            "compact",
            [
                _row(1, 1, 0, 1, true_label_rank=2.0),
                _row(1, 2, 1, 1, true_label_rank=1.0),
            ],
        )

        artifacts = ensemble_prediction_sources([compact], [("compact", ("compact",))])
        summary = artifacts["group_summary"][0]

        self.assertEqual(summary["balanced_accuracy_mean"], 0.5)
        self.assertEqual(summary["top2_accuracy_mean"], 1.0)
        self.assertEqual({row["artifact_ensemble_rank_source"] for row in artifacts["predictions"]}, {"source_true_label_rank"})

    def test_hard_vote_ensemble_uses_first_source_as_tie_breaker(self) -> None:
        compact = _source(
            "compact",
            [
                _row(1, 1, 0, 1, true_label_rank=2.0),
                _row(1, 2, 1, 1, true_label_rank=1.0),
            ],
        )
        finetune = _source(
            "finetune",
            [
                _row(1, 1, 0, 0, true_label_rank=1.0),
                _row(1, 2, 1, 0, true_label_rank=2.0),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("compact_finetune", ("compact", "finetune"))],
        )
        predictions = artifacts["predictions"]

        self.assertEqual([row["predicted_label"] for row in predictions], [1, 1])
        self.assertEqual([row["true_label_rank"] for row in predictions], [2.0, 1.0])
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 0.5)
        self.assertEqual(artifacts["group_summary"][0]["top2_accuracy_mean"], 1.0)

    def test_averages_class_scores_when_available(self) -> None:
        compact = _source("compact", [_scored_row(0, 1, 0.40, 0.60)])
        finetune = _source("finetune", [_scored_row(0, 0, 0.90, 0.10)])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("score_mean", ("compact", "finetune"))],
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_mean")
        self.assertAlmostEqual(float(prediction["score_class_0"]), 0.65)
        self.assertAlmostEqual(float(prediction["score_1"]), 0.65)
        self.assertAlmostEqual(float(prediction["prob_class_0"]), 0.65)
        self.assertEqual(prediction["rank_class_0"], 1)
        self.assertEqual(prediction["rank_1"], 1)
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

        stacked = _source("stacked", artifacts["predictions"])
        stacked_artifacts = ensemble_prediction_sources(
            [stacked],
            [("stacked", ("stacked",))],
            aggregation_mode="mean_score",
        )
        stacked_prediction = stacked_artifacts["predictions"][0]
        self.assertEqual(stacked_prediction["predicted_label"], 0)
        self.assertEqual(stacked_prediction["artifact_ensemble_mode"], "class_score_mean")

    def test_display_score_columns_follow_one_based_labels_when_labels_are_one_based(self) -> None:
        def one_based_row(trial_index: int, true_label: int, predicted_label: int, score_1: float, score_2: float) -> dict[str, str]:
            ranked_labels = sorted(((1, score_1), (2, score_2)), key=lambda item: (-item[1], item[0]))
            return {
                "test_participant": "1",
                "test_trial_index": str(trial_index),
                "true_label": str(true_label),
                "predicted_label": str(predicted_label),
                "true_stimulus": str(true_label),
                "predicted_stimulus": str(predicted_label),
                "true_label_rank": str(ranked_labels.index((true_label, score_1 if true_label == 1 else score_2)) + 1),
                "score_1": f"{score_1:.2f}",
                "score_2": f"{score_2:.2f}",
            }

        latent = _source(
            "latent",
            [
                one_based_row(1, 1, 1, 0.90, 0.10),
                one_based_row(2, 2, 2, 0.10, 0.90),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [latent],
            [("latent", ("latent",))],
            aggregation_mode="mean_score",
        )
        predictions = artifacts["predictions"]

        self.assertEqual([row["predicted_label"] for row in predictions], [1, 2])
        self.assertAlmostEqual(float(predictions[0]["score_class_1"]), 0.90)
        self.assertAlmostEqual(float(predictions[1]["score_class_2"]), 0.90)

    def test_display_score_columns_keep_legacy_zero_based_stimulus_shift(self) -> None:
        first = _stimulus_scored_row(0, 0, 0.90, 0.10)
        second = _stimulus_scored_row(1, 1, 0.10, 0.90)
        second["test_trial_index"] = "2"
        legacy = _source("legacy", [first, second])

        artifacts = ensemble_prediction_sources(
            [legacy],
            [("legacy", ("legacy",))],
            aggregation_mode="mean_score",
        )

        self.assertEqual([row["predicted_label"] for row in artifacts["predictions"]], [0, 1])
        self.assertAlmostEqual(float(artifacts["predictions"][0]["score_class_0"]), 0.90)
        self.assertAlmostEqual(float(artifacts["predictions"][1]["score_class_1"]), 0.90)

    def test_log_score_mean_uses_geometric_consensus(self) -> None:
        source_names = tuple(f"source_{index}" for index in range(4))
        sources = []
        for index, source_name in enumerate(source_names):
            scores = [0.19, 0.013333, 0.013333, 0.013333, 0.013333]
            scores[index + 1] = 0.77
            sources.append(_source(source_name, [_multi_scored_row(index + 1, scores)]))

        mean_artifacts = ensemble_prediction_sources(
            sources,
            [("mean", source_names)],
            aggregation_mode="mean_score",
        )
        log_artifacts = ensemble_prediction_sources(
            sources,
            [("log_mean", source_names)],
            aggregation_mode="log_score_mean",
        )

        self.assertNotEqual(mean_artifacts["predictions"][0]["predicted_label"], 0)
        prediction = log_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_log_mean")
        self.assertEqual(log_artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_confidence_weighted_score_mean_trusts_decisive_source(self) -> None:
        confident = _source("confident", [_multi_scored_row(0, [0.95, 0.05])])
        weak_wrong_sources = [
            _source(f"weak_wrong_{index}", [_multi_scored_row(1, [0.40, 0.60])])
            for index in range(5)
        ]
        sources = [confident, *weak_wrong_sources]
        source_names = tuple(source.name for source in sources)

        mean_artifacts = ensemble_prediction_sources(
            sources,
            [("mean", source_names)],
            aggregation_mode="mean_score",
        )
        confidence_artifacts = ensemble_prediction_sources(
            sources,
            [("confidence", source_names)],
            aggregation_mode="confidence_weighted_mean_score",
        )

        self.assertEqual(mean_artifacts["predictions"][0]["predicted_label"], 1)
        prediction = confidence_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_confidence_weighted_mean")

    def test_entropy_weighted_score_mean_downweights_high_entropy_sources(self) -> None:
        confident = _source("confident", [_multi_scored_row(0, [0.99, 0.01])])
        uncertain_wrong_sources = [
            _source(f"uncertain_wrong_{index}", [_multi_scored_row(1, [0.40, 0.60])])
            for index in range(5)
        ]
        sources = [confident, *uncertain_wrong_sources]
        source_names = tuple(source.name for source in sources)

        mean_artifacts = ensemble_prediction_sources(
            sources,
            [("mean", source_names)],
            aggregation_mode="mean_score",
        )
        entropy_artifacts = ensemble_prediction_sources(
            sources,
            [("entropy", source_names)],
            aggregation_mode="entropy_weighted_mean_score",
        )

        self.assertEqual(mean_artifacts["predictions"][0]["predicted_label"], 1)
        prediction = entropy_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_entropy_weighted_mean")
        self.assertEqual(entropy_artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_agreement_weighted_score_mean_downweights_score_outlier(self) -> None:
        outlier_wrong = _source("outlier_wrong", [_multi_scored_row(1, [0.05, 0.95])])
        weak_correct_a = _source("weak_correct_a", [_multi_scored_row(0, [0.65, 0.35])])
        weak_correct_b = _source("weak_correct_b", [_multi_scored_row(0, [0.65, 0.35])])
        sources = [outlier_wrong, weak_correct_a, weak_correct_b]
        source_names = tuple(source.name for source in sources)

        mean_artifacts = ensemble_prediction_sources(
            sources,
            [("mean", source_names)],
            aggregation_mode="mean_score",
        )
        agreement_artifacts = ensemble_prediction_sources(
            sources,
            [("agreement", source_names)],
            aggregation_mode="agreement_weighted_mean_score",
        )

        self.assertEqual(mean_artifacts["predictions"][0]["predicted_label"], 1)
        prediction = agreement_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_agreement_weighted_mean")
        self.assertAlmostEqual(float(prediction["score_class_0"]), 0.508, places=2)
        self.assertEqual(agreement_artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_can_force_hard_vote_when_scores_are_available(self) -> None:
        compact = _source("compact", [_scored_row(0, 1, 0.95, 0.05)])
        finetune = _source("finetune", [_scored_row(0, 0, 0.95, 0.05)])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("hard", ("compact", "finetune"))],
            aggregation_mode="hard_vote",
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 1)
        self.assertEqual(prediction["artifact_ensemble_requested_aggregation_mode"], "hard_vote")
        self.assertEqual(prediction["artifact_ensemble_mode"], "hard_vote_tiebreak_first_source")

    def test_can_force_rank_borda_even_when_scores_are_available(self) -> None:
        compact = _source("compact", [{**_scored_row(0, 1, 0.10, 0.90), "rank_class_0": "1.0", "rank_class_1": "2.0"}])
        finetune = _source("finetune", [{**_scored_row(0, 1, 0.10, 0.90), "rank_class_0": "1.0", "rank_class_1": "2.0"}])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("borda", ("compact", "finetune"))],
            aggregation_mode="borda",
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_requested_aggregation_mode"], "borda")
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_rank_borda")

    def test_score_rank_fusion_combines_score_and_rank_columns(self) -> None:
        compact = _source(
            "compact",
            [{**_scored_row(0, 1, 0.49, 0.51), "rank_class_0": "1.0", "rank_class_1": "2.0"}],
        )
        finetune = _source(
            "finetune",
            [{**_scored_row(0, 1, 0.49, 0.51), "rank_class_0": "1.0", "rank_class_1": "2.0"}],
        )

        mean_artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("mean", ("compact", "finetune"))],
            aggregation_mode="mean_score",
        )
        fusion_artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("fusion", ("compact", "finetune"))],
            aggregation_mode="score_rank_fusion",
        )

        self.assertEqual(mean_artifacts["predictions"][0]["predicted_label"], 1)
        prediction = fusion_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_mean_class_rank_mean_fusion")

    def test_reciprocal_rank_fusion_rewards_repeated_top_ranks(self) -> None:
        strong_a = _source("strong_a", [_ranked_row(0, 1, 1.0, 2.0)])
        strong_b = _source("strong_b", [_ranked_row(0, 1, 1.0, 2.0)])
        outlier = _source("outlier", [_ranked_row(0, 1, 10.0, 1.0)])

        mean_rank_artifacts = ensemble_prediction_sources(
            [strong_a, strong_b, outlier],
            [("mean_rank", ("strong_a", "strong_b", "outlier"))],
            aggregation_mode="mean_rank",
        )
        reciprocal_artifacts = ensemble_prediction_sources(
            [strong_a, strong_b, outlier],
            [("rrf", ("strong_a", "strong_b", "outlier"))],
            aggregation_mode="reciprocal_rank_fusion",
        )

        self.assertEqual(mean_rank_artifacts["predictions"][0]["predicted_label"], 1)
        prediction = reciprocal_artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_reciprocal_rank_fusion")

    def test_mean_rank_can_derive_ranks_from_score_only_artifacts(self) -> None:
        compact = _source("compact", [_scored_row(0, 1, 0.90, 0.10)])
        finetune = _source("finetune", [_scored_row(0, 1, 0.40, 0.60)])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("score_derived_rank", ("compact", "finetune"))],
            aggregation_mode="mean_rank",
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_derived_rank_mean")

    def test_mean_rank_uses_rank_columns(self) -> None:
        compact = _source("compact", [_ranked_row(0, 1, 1.0, 2.0)])
        finetune = _source("finetune", [_ranked_row(0, 1, 1.0, 2.0)])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("rank_mean", ("compact", "finetune"))],
            aggregation_mode="mean_rank",
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 0)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_rank_mean")

    def test_score_tiebreak_first_source_uses_first_source_order(self) -> None:
        compact = _source("compact", [{**_scored_row(0, 1, 0.50, 0.50), "rank_class_0": "2.0", "rank_class_1": "1.0"}])
        finetune = _source("finetune", [{**_scored_row(0, 0, 0.50, 0.50), "rank_class_0": "1.0", "rank_class_1": "2.0"}])

        artifacts = ensemble_prediction_sources(
            [compact, finetune],
            [("score_tie", ("compact", "finetune"))],
            aggregation_mode="score_tiebreak_first_source",
        )

        prediction = artifacts["predictions"][0]
        self.assertEqual(prediction["predicted_label"], 1)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_mean_tiebreak_first_source")

    def test_accepts_one_based_stimulus_score_columns(self) -> None:
        compact = _source("compact", [_stimulus_scored_row(0, 1, 0.40, 0.60)])
        latent = _source("latent", [_stimulus_scored_row(0, 0, 0.90, 0.10)])

        artifacts = ensemble_prediction_sources(
            [compact, latent],
            [("score_mean", ("compact", "latent"))],
        )

        self.assertEqual(artifacts["predictions"][0]["predicted_label"], 0)
        self.assertEqual(artifacts["predictions"][0]["artifact_ensemble_mode"], "class_score_mean")
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_weighted_score_ensemble_can_rank_softmax_scores(self) -> None:
        compact = _source("compact", [_stimulus_scored_row(0, 1, 0.20, 0.80)])
        latent = _source("latent", [_stimulus_scored_row(0, 0, 0.90, 0.10)])

        artifacts = ensemble_prediction_sources(
            [compact, latent],
            [("weighted", ("compact", "latent"), (0.9, 0.1))],
            score_normalization="rank_softmax",
        )

        prediction = artifacts["predictions"][0]
        summary = artifacts["group_summary"][0]
        self.assertEqual(prediction["predicted_label"], 1)
        self.assertEqual(prediction["artifact_ensemble_mode"], "class_score_rank_softmax_weighted_mean")
        self.assertEqual(prediction["artifact_ensemble_source_weights"], "compact:0.9;latent:0.1")
        self.assertEqual(summary["artifact_ensemble_score_normalization"], "rank_softmax")
        self.assertEqual(summary["balanced_accuracy_mean"], 0.0)

    def test_balanced_assignment_aggregation_respects_uniform_class_quotas(self) -> None:
        latent = _source(
            "latent",
            [
                _two_class_scored_row(1, 0, 0, 5.0, 4.0),
                _two_class_scored_row(2, 1, 0, 4.9, 4.8),
                _two_class_scored_row(3, 0, 0, 4.7, 1.0),
                _two_class_scored_row(4, 1, 0, 4.6, 4.5),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [latent],
            [("balanced", ("latent",))],
            aggregation_mode="balanced_assignment",
        )

        predictions = artifacts["predictions"]
        self.assertEqual([row["predicted_label"] for row in predictions], [0, 1, 0, 1])
        self.assertEqual({row["artifact_ensemble_mode"] for row in predictions}, {"class_score_balanced_assignment"})
        second_prediction = predictions[1]
        self.assertEqual(second_prediction["true_label"], 1)
        self.assertEqual(second_prediction["predicted_label"], 1)
        self.assertEqual(second_prediction["true_label_rank"], 1.0)
        self.assertTrue(second_prediction["top2_correct"])
        self.assertEqual(second_prediction["rank_class_1"], 1)
        self.assertTrue(str(second_prediction["vote_ranked_labels"]).startswith("1;"))
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_balanced_assignment_shrinkage_is_less_aggressive_than_uniform_assignment(self) -> None:
        latent = _source(
            "latent",
            [
                _multi_two_class_scored_row(1, 0, 0, 5.0, 4.0),
                _multi_two_class_scored_row(2, 1, 0, 4.9, 4.8),
                _multi_two_class_scored_row(3, 0, 0, 4.7, 1.0),
                _multi_two_class_scored_row(4, 1, 0, 4.6, 4.5),
            ],
        )

        uniform = ensemble_prediction_sources(
            [latent],
            [("balanced", ("latent",))],
            aggregation_mode="balanced_assignment",
        )
        shrink50 = ensemble_prediction_sources(
            [latent],
            [("balanced", ("latent",))],
            aggregation_mode="balanced_assignment_shrink50",
        )

        uniform_counts = Counter(row["predicted_label"] for row in uniform["predictions"])
        shrink_counts = Counter(row["predicted_label"] for row in shrink50["predictions"])
        self.assertEqual(dict(uniform_counts), {0: 2, 1: 2})
        self.assertEqual(dict(shrink_counts), {0: 3, 1: 1})
        self.assertEqual(
            {row["artifact_ensemble_mode"] for row in shrink50["predictions"]},
            {"class_score_balanced_assignment_shrink50"},
        )
        self.assertEqual(
            {
                row["artifact_ensemble_balanced_assignment_uniform_alpha"]
                for row in shrink50["predictions"]
            },
            {"0.5"},
        )

    def test_low_margin_balanced_assignment_preserves_high_margin_predictions(self) -> None:
        latent = _source(
            "latent",
            [
                _multi_two_class_scored_row(1, 0, 0, 5.0, 0.0),
                _multi_two_class_scored_row(2, 0, 0, 4.8, 0.0),
                _multi_two_class_scored_row(3, 1, 0, 3.10, 3.00),
                _multi_two_class_scored_row(4, 1, 0, 3.05, 3.00),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [latent],
            [("low_margin", ("latent",))],
            aggregation_mode="balanced_assignment_low_margin50",
        )

        predictions = artifacts["predictions"]
        self.assertEqual([row["predicted_label"] for row in predictions], [0, 0, 1, 1])
        self.assertEqual(
            {row["artifact_ensemble_mode"] for row in predictions},
            {"class_score_balanced_assignment_low_margin50"},
        )
        self.assertEqual(
            {row["artifact_ensemble_balanced_assignment_margin_threshold"] for row in predictions},
            {"0.5"},
        )
        self.assertEqual(
            {row["artifact_ensemble_balanced_assignment_fixed_predictions"] for row in predictions},
            {2},
        )
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_uniform_prior_shift_debiases_participant_score_distribution(self) -> None:
        biased = _source(
            "biased",
            [
                _two_class_scored_row(1, 0, 0, 0.55, 0.45),
                _two_class_scored_row(2, 1, 0, 0.52, 0.48),
                _two_class_scored_row(3, 0, 0, 0.53, 0.47),
                _two_class_scored_row(4, 1, 0, 0.51, 0.49),
            ],
        )

        unshifted = ensemble_prediction_sources(
            [biased],
            [("unshifted", ("biased",))],
            aggregation_mode="mean_score",
        )
        shifted = ensemble_prediction_sources(
            [biased],
            [("shifted", ("biased",))],
            aggregation_mode="uniform_prior_shift",
        )

        self.assertEqual([row["predicted_label"] for row in unshifted["predictions"]], [0, 0, 0, 0])
        predictions = shifted["predictions"]
        self.assertEqual([row["predicted_label"] for row in predictions], [0, 1, 0, 1])
        self.assertEqual({row["artifact_ensemble_mode"] for row in predictions}, {"class_score_uniform_prior_shift"})
        self.assertEqual({row["artifact_ensemble_uniform_prior_shift_alpha"] for row in predictions}, {"1"})
        self.assertAlmostEqual(
            float(predictions[0]["prob_class_0"]) + float(predictions[0]["prob_class_1"]),
            1.0,
        )
        self.assertEqual(predictions[1]["rank_class_1"], 1)
        self.assertEqual(shifted["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_rejects_misaligned_source_prediction_keys(self) -> None:
        compact = _source("compact", [_row(1, 1, 0, 0)])
        finetune = _source("finetune", [_row(1, 2, 0, 0)])

        with self.assertRaisesRegex(ValueError, "Prediction keys do not match"):
            ensemble_prediction_sources(
                [compact, finetune],
                [("compact_finetune", ("compact", "finetune"))],
            )

    def test_nested_subject_selector_uses_other_subjects_only(self) -> None:
        compact = _source(
            "compact",
            [
                _row(1, 1, 0, 0),
                _row(2, 1, 0, 1),
                _row(3, 1, 0, 0),
            ],
        )
        alt_a = _source(
            "alt_a",
            [
                _row(1, 1, 0, 1),
                _row(2, 1, 0, 0),
                _row(3, 1, 0, 0),
            ],
        )
        alt_b = _source(
            "alt_b",
            [
                _row(1, 1, 0, 1),
                _row(2, 1, 0, 0),
                _row(3, 1, 0, 0),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [compact, alt_a, alt_b],
            [
                ("compact", ("compact",)),
                ("compact_alt", ("compact", "alt_a", "alt_b")),
            ],
            nested_selector_name="nested_subject_selector",
        )

        selections = {
            row["test_participant"]: row["selected_artifact_ensemble"]
            for row in artifacts["nested_selection"]
        }
        self.assertEqual(selections, {"1": "compact_alt", "2": "compact", "3": "compact"})

        nested_summary = next(
            row for row in artifacts["group_summary"] if row["artifact_ensemble"] == "nested_subject_selector"
        )
        self.assertAlmostEqual(nested_summary["balanced_accuracy_mean"], 1.0 / 3.0)
        self.assertEqual(nested_summary["artifact_ensemble_requested_aggregation_mode"], "auto")
        self.assertEqual(nested_summary["selected_artifact_ensemble_counts"], "compact:2;compact_alt:1")
        nested_predictions = [
            row for row in artifacts["predictions"] if row["artifact_ensemble"] == "nested_subject_selector"
        ]
        self.assertEqual([row["predicted_label"] for row in nested_predictions], [1, 1, 0])

    def test_nested_subject_selector_can_use_rank_aware_metric(self) -> None:
        balanced_first = _source(
            "balanced_first",
            [
                _row(1, 1, 0, 0, true_label_rank=1.0),
                _row(2, 1, 0, 0, true_label_rank=1.0),
                _row(3, 1, 0, 1, true_label_rank=3.0),
            ],
        )
        rank_aware = _source(
            "rank_aware",
            [
                _row(1, 1, 0, 1, true_label_rank=2.0),
                _row(2, 1, 0, 1, true_label_rank=2.0),
                _row(3, 1, 0, 0, true_label_rank=1.0),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [balanced_first, rank_aware],
            [
                ("balanced_first", ("balanced_first",)),
                ("rank_aware", ("rank_aware",)),
            ],
            nested_selector_name="nested_subject_selector",
            nested_selection_metric="balanced_top2_top3_rank",
        )

        selections = {
            row["test_participant"]: row["selected_artifact_ensemble"]
            for row in artifacts["nested_selection"]
        }
        self.assertEqual(selections["1"], "rank_aware")
        first_selection = next(row for row in artifacts["nested_selection"] if row["test_participant"] == "1")
        self.assertEqual(first_selection["selection_metric"], "other_subjects_balanced_top2_top3_rank")
        self.assertEqual(first_selection["selection_metric_name"], "balanced_top2_top3_rank")

        nested_summary = next(
            row for row in artifacts["group_summary"] if row["artifact_ensemble"] == "nested_subject_selector"
        )
        self.assertEqual(nested_summary["selection_metric_name"], "balanced_top2_top3_rank")

    def test_nested_subject_selector_can_use_paired_delta_lcb_metric(self) -> None:
        compact = _source(
            "compact",
            [
                _row(1, 1, 0, 0),
                _row(2, 1, 0, 1),
                _row(3, 1, 0, 1),
            ],
        )
        robust_delta = _source(
            "robust_delta",
            [
                _row(1, 1, 0, 1),
                _row(2, 1, 0, 0),
                _row(3, 1, 0, 0),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [compact, robust_delta],
            [
                ("compact", ("compact",)),
                ("robust_delta", ("robust_delta",)),
            ],
            nested_selector_name="nested_subject_selector",
            nested_selection_metric="balanced_accuracy_delta_lcb",
        )

        selections = {
            row["test_participant"]: row["selected_artifact_ensemble"]
            for row in artifacts["nested_selection"]
        }
        self.assertEqual(selections["1"], "robust_delta")
        first_selection = next(row for row in artifacts["nested_selection"] if row["test_participant"] == "1")
        self.assertEqual(first_selection["selection_metric"], "other_subjects_balanced_accuracy_delta_lcb")
        self.assertEqual(first_selection["selection_metric_name"], "balanced_accuracy_delta_lcb")
        self.assertEqual(first_selection["reference_artifact_ensemble"], "compact")

        nested_summary = next(
            row for row in artifacts["group_summary"] if row["artifact_ensemble"] == "nested_subject_selector"
        )
        self.assertEqual(nested_summary["selection_metric_name"], "balanced_accuracy_delta_lcb")
        self.assertEqual(nested_summary["reference_artifact_ensemble"], "compact")

    def test_nested_weight_selector_uses_other_subjects_only(self) -> None:
        source_a = _source(
            "source_a",
            [
                _participant_scored_row(1, 0, 1, 0.10, 0.90),
                _participant_scored_row(2, 0, 0, 0.90, 0.10),
                _participant_scored_row(3, 0, 0, 0.90, 0.10),
            ],
        )
        source_b = _source(
            "source_b",
            [
                _participant_scored_row(1, 0, 0, 0.90, 0.10),
                _participant_scored_row(2, 0, 1, 0.10, 0.90),
                _participant_scored_row(3, 0, 1, 0.10, 0.90),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [source_a, source_b],
            [("source_a_b", ("source_a", "source_b"))],
            aggregation_mode="mean_score",
            nested_weight_selector_name="nested_weight_selector",
            nested_weight_selector_ensemble="source_a_b",
            nested_weight_grid_step=1.0,
        )

        selections = {
            row["test_participant"]: row["selected_source_weights"]
            for row in artifacts["nested_weight_selection"]
        }
        self.assertEqual(selections["1"], "source_a:1;source_b:0")
        nested_predictions = [
            row for row in artifacts["predictions"] if row["artifact_ensemble"] == "nested_weight_selector"
        ]
        participant_1 = next(row for row in nested_predictions if row["test_participant"] == "1")
        self.assertEqual(participant_1["predicted_label"], 1)
        self.assertEqual(participant_1["artifact_ensemble_weight_selection"], "leave_subject_out_grid")
        nested_summary = next(row for row in artifacts["group_summary"] if row["artifact_ensemble"] == "nested_weight_selector")
        self.assertEqual(nested_summary["candidate_source_weight_count"], 2)

    def test_nested_weight_selector_can_use_rank_aware_metric(self) -> None:
        source_a = _source(
            "source_a",
            [
                _participant_three_score_row(1, 0, 1, 0.20, 0.70, 0.10),
                _participant_three_score_row(2, 0, 1, 0.45, 0.50, 0.05),
                _participant_three_score_row(3, 0, 0, 0.85, 0.10, 0.05),
            ],
        )
        source_b = _source(
            "source_b",
            [
                _participant_three_score_row(1, 0, 0, 0.85, 0.10, 0.05),
                _participant_three_score_row(2, 0, 0, 0.85, 0.10, 0.05),
                _participant_three_score_row(3, 0, 2, 0.05, 0.10, 0.85),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [source_a, source_b],
            [("source_a_b", ("source_a", "source_b"))],
            aggregation_mode="mean_score",
            nested_weight_selector_name="nested_weight_selector",
            nested_weight_selector_ensemble="source_a_b",
            nested_weight_grid_step=1.0,
            nested_selection_metric="balanced_top2_top3_rank",
        )

        selections = {
            row["test_participant"]: row["selected_source_weights"]
            for row in artifacts["nested_weight_selection"]
        }
        self.assertEqual(selections["1"], "source_a:1;source_b:0")
        first_selection = next(row for row in artifacts["nested_weight_selection"] if row["test_participant"] == "1")
        self.assertEqual(first_selection["selection_metric"], "other_subjects_balanced_top2_top3_rank")
        self.assertEqual(first_selection["selection_metric_name"], "balanced_top2_top3_rank")
        self.assertAlmostEqual(first_selection["selection_balanced_accuracy"], 0.5)
        nested_predictions = [
            row for row in artifacts["predictions"] if row["artifact_ensemble"] == "nested_weight_selector"
        ]
        participant_1 = next(row for row in nested_predictions if row["test_participant"] == "1")
        self.assertEqual(participant_1["predicted_label"], 1)

    def test_nested_weight_selector_can_use_paired_delta_lcb_metric(self) -> None:
        source_a = _source(
            "source_a",
            [
                _participant_scored_row(1, 0, 1, 0.10, 0.90),
                _participant_scored_row(2, 0, 0, 0.90, 0.10),
                _participant_scored_row(3, 0, 0, 0.90, 0.10),
            ],
        )
        source_b = _source(
            "source_b",
            [
                _participant_scored_row(1, 0, 0, 0.90, 0.10),
                _participant_scored_row(2, 0, 1, 0.10, 0.90),
                _participant_scored_row(3, 0, 1, 0.10, 0.90),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [source_a, source_b],
            [("source_a_b", ("source_a", "source_b"))],
            aggregation_mode="mean_score",
            nested_weight_selector_name="nested_weight_selector",
            nested_weight_selector_ensemble="source_a_b",
            nested_weight_grid_step=1.0,
            nested_selection_metric="balanced_accuracy_delta_lcb",
        )

        selections = {
            row["test_participant"]: row["selected_source_weights"]
            for row in artifacts["nested_weight_selection"]
        }
        self.assertEqual(selections["1"], "source_a:1;source_b:0")
        first_selection = next(row for row in artifacts["nested_weight_selection"] if row["test_participant"] == "1")
        self.assertEqual(first_selection["selection_metric"], "other_subjects_balanced_accuracy_delta_lcb")
        self.assertEqual(first_selection["selection_metric_name"], "balanced_accuracy_delta_lcb")
        self.assertIn("reference_source_weights", first_selection)
        nested_summary = next(row for row in artifacts["group_summary"] if row["artifact_ensemble"] == "nested_weight_selector")
        self.assertEqual(nested_summary["selection_metric_name"], "balanced_accuracy_delta_lcb")

    def test_nested_weight_selector_can_expand_all_multi_source_ensembles(self) -> None:
        source_a = _source(
            "source_a",
            [
                _participant_scored_row(1, 0, 1, 0.10, 0.90),
                _participant_scored_row(2, 0, 0, 0.90, 0.10),
                _participant_scored_row(3, 0, 0, 0.90, 0.10),
            ],
        )
        source_b = _source(
            "source_b",
            [
                _participant_scored_row(1, 0, 0, 0.90, 0.10),
                _participant_scored_row(2, 0, 1, 0.10, 0.90),
                _participant_scored_row(3, 0, 1, 0.10, 0.90),
            ],
        )
        source_c = _source(
            "source_c",
            [
                _participant_scored_row(1, 0, 0, 0.85, 0.15),
                _participant_scored_row(2, 0, 1, 0.15, 0.85),
                _participant_scored_row(3, 0, 1, 0.15, 0.85),
            ],
        )

        artifacts = ensemble_prediction_sources(
            [source_a, source_b, source_c],
            [
                ("source_a_b", ("source_a", "source_b")),
                ("source_a_c", ("source_a", "source_c")),
            ],
            aggregation_mode="mean_score",
            nested_weight_selector_name="nested_weight_selector",
            nested_weight_selector_ensemble="all",
            nested_weight_grid_step=1.0,
        )

        selector_names = {
            row["artifact_ensemble"]
            for row in artifacts["group_summary"]
            if str(row["artifact_ensemble"]).startswith("nested_weight_selector")
        }
        self.assertEqual(
            selector_names,
            {"nested_weight_selector_source_a_b", "nested_weight_selector_source_a_c"},
        )
        selection_names = {row["artifact_ensemble"] for row in artifacts["nested_weight_selection"]}
        self.assertEqual(selection_names, selector_names)


if __name__ == "__main__":
    unittest.main()
