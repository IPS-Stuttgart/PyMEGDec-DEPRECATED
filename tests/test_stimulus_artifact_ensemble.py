from __future__ import annotations

import unittest
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

        self.assertEqual(artifacts["predictions"][0]["predicted_label"], 0)
        self.assertEqual(artifacts["predictions"][0]["artifact_ensemble_mode"], "class_score_mean")
        self.assertEqual(artifacts["group_summary"][0]["balanced_accuracy_mean"], 1.0)

    def test_rejects_misaligned_source_prediction_keys(self) -> None:
        compact = _source("compact", [_row(1, 1, 0, 0)])
        finetune = _source("finetune", [_row(1, 2, 0, 0)])

        with self.assertRaisesRegex(ValueError, "Prediction keys do not match"):
            ensemble_prediction_sources(
                [compact, finetune],
                [("compact_finetune", ("compact", "finetune"))],
            )


if __name__ == "__main__":
    unittest.main()
