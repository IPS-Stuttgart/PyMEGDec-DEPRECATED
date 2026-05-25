from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pymegdec.stimulus_nested_matrix import (
    NestedMatrixShardError,
    aggregate_nested_matrix_outputs,
    discover_nested_matrix_shards,
    validate_nested_matrix_shards,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _outer_row(participant: int, *, balanced: float, bundle_classifier: str = "multinomial-logistic") -> dict:
    return {
        "test_participant": participant,
        "accuracy": balanced,
        "balanced_accuracy": balanced,
        "chance_accuracy": 0.0625,
        "top2_accuracy": min(1.0, balanced + 0.1),
        "top3_accuracy": min(1.0, balanced + 0.2),
        "mean_true_label_rank": 4.0,
        "selected_candidate_index": participant,
        "selected_candidate_indices": str(participant),
        "selected_classifier": bundle_classifier,
        "selected_window_center_s": 0.175,
        "selected_feature_mode": "sensor_flat",
        "selected_normalization": "subject_baseline_whiten",
        "selected_alignment": "none",
        "selected_components_pca": 64,
        "selected_inner_winner_margin": 0.01,
        "max_trials_per_class_per_participant": 10,
        "n_candidates": 2,
        "label_shuffle_control": False,
        "label_shuffle_seed": 0,
        "outer_evaluation_mode": "topk_score_ensemble",
        "selection_ensemble_size": 2,
        "selection_ensemble_diversity": "window_classifier",
        "selection_ensemble_score_normalization": "rank_softmax",
        "selection_ensemble_weighting": "inner_lcb_softmax",
        "selection_ensemble_temperature": 0.02,
    }


def _selected_row(participant: int) -> dict:
    return {
        "test_participant": participant,
        "selected_candidate_index": participant,
        "selected_classifier": "multinomial-logistic",
        "selected_window_center_s": 0.175,
        "selected_inner_balanced_accuracy_mean": 0.2,
        "selected_inner_winner_margin": 0.01,
    }


def _prediction_rows(participant: int) -> list[dict]:
    return [
        {
            "test_participant": participant,
            "true_stimulus": 1,
            "predicted_stimulus": 1,
            "correct": True,
            "window_center_s": 0.175,
            "feature_mode": "sensor_flat",
            "normalization": "subject_baseline_whiten",
            "alignment": "none",
            "classifier": "multinomial-logistic",
            "components_pca": 64,
            "max_trials_per_class_per_participant": 10,
            "label_shuffle_control": False,
            "label_shuffle_seed": 0,
        },
        {
            "test_participant": participant,
            "true_stimulus": 2,
            "predicted_stimulus": 1,
            "correct": False,
            "window_center_s": 0.175,
            "feature_mode": "sensor_flat",
            "normalization": "subject_baseline_whiten",
            "alignment": "none",
            "classifier": "multinomial-logistic",
            "components_pca": 64,
            "max_trials_per_class_per_participant": 10,
            "label_shuffle_control": False,
            "label_shuffle_seed": 0,
        },
    ]


class TestStimulusNestedMatrix(unittest.TestCase):
    def test_discovers_nested_matrix_shards_by_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_outer.csv", [_outer_row(1, balanced=0.1)])
            _write_csv(tmp_path / "nested-matrix-feature-p2" / "matrix_feature_p2_outer.csv", [_outer_row(2, balanced=0.2)])

            shards = discover_nested_matrix_shards(tmp_path)

            self.assertEqual(sorted(shards), ["feature", "logreg"])
            self.assertEqual([path.name for path in shards["logreg"]], ["matrix_logreg_p1_outer.csv"])

    def test_strict_validation_rejects_missing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_outer.csv", [_outer_row(1, balanced=0.1)])

            shards = discover_nested_matrix_shards(tmp_path)

            with self.assertRaisesRegex(NestedMatrixShardError, "missing=matrix_logreg_p1_selected.csv"):
                validate_nested_matrix_shards(shards, required_kinds=("selected", "predictions"))
            with self.assertRaisesRegex(NestedMatrixShardError, "Expected 2 nested matrix outer shard"):
                validate_nested_matrix_shards(shards, expected_shard_count=2, required_kinds=())

    def test_aggregates_nested_matrix_shards_and_recomputes_bundle_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for participant, balanced in [(1, 0.10), (2, 0.20)]:
                stem = tmp_path / f"nested-matrix-logreg-p{participant}" / f"matrix_logreg_p{participant}"
                _write_csv(stem.with_name(f"{stem.name}_outer.csv"), [_outer_row(participant, balanced=balanced)])
                _write_csv(stem.with_name(f"{stem.name}_selected.csv"), [_selected_row(participant)])
                _write_csv(
                    stem.with_name(f"{stem.name}_inner_validation.csv"),
                    [{"test_participant": participant, "candidate_index": participant, "balanced_accuracy": balanced}],
                )
                _write_csv(stem.with_name(f"{stem.name}_predictions.csv"), _prediction_rows(participant))

            artifacts = aggregate_nested_matrix_outputs(
                tmp_path,
                tmp_path / "out",
                output_stem="nested_matrix",
                signflip_permutations=0,
                strict_shards=True,
                expected_shard_count=2,
            )
            summary = list(csv.DictReader((tmp_path / "out" / "nested_matrix_group_summary.csv").open(newline="", encoding="utf-8")))
            selected = list(csv.DictReader((tmp_path / "out" / "nested_matrix_selected.csv").open(newline="", encoding="utf-8")))
            confusion = list(csv.DictReader((tmp_path / "out" / "nested_matrix_confusion.csv").open(newline="", encoding="utf-8")))
            per_stimulus = list(csv.DictReader((tmp_path / "out" / "nested_matrix_per_stimulus.csv").open(newline="", encoding="utf-8")))
            confusion_pairs = list(csv.DictReader((tmp_path / "out" / "nested_matrix_confusion_pairs.csv").open(newline="", encoding="utf-8")))

            self.assertEqual(len(artifacts["outer"]), 2)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["matrix_config_bundle"], "logreg")
            self.assertAlmostEqual(float(summary[0]["balanced_accuracy_mean"]), 0.15)
            self.assertEqual(summary[0]["participants_total"], "2")
            self.assertEqual({row["matrix_config_bundle"] for row in selected}, {"logreg"})
            self.assertEqual({row["matrix_config_bundle"] for row in confusion}, {"logreg"})
            self.assertEqual({row["matrix_config_bundle"] for row in per_stimulus}, {"logreg"})
            self.assertEqual({row["matrix_config_bundle"] for row in confusion_pairs}, {"logreg"})
            self.assertEqual(len(artifacts["confusion"]), len(confusion))
            self.assertEqual(len(artifacts["per_stimulus"]), len(per_stimulus))
            self.assertEqual(len(artifacts["confusion_pairs"]), len(confusion_pairs))

    def test_aggregates_multiple_config_bundles_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_outer.csv", [_outer_row(1, balanced=0.10)])
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_predictions.csv", _prediction_rows(1))
            _write_csv(
                tmp_path / "nested-matrix-feature-p1" / "matrix_feature_p1_outer.csv",
                [_outer_row(1, balanced=0.30, bundle_classifier="shrinkage-lda")],
            )
            _write_csv(tmp_path / "nested-matrix-feature-p1" / "matrix_feature_p1_predictions.csv", _prediction_rows(1))

            artifacts = aggregate_nested_matrix_outputs(
                tmp_path,
                tmp_path / "out",
                output_stem="nested_matrix",
                signflip_permutations=0,
            )
            summary = list(csv.DictReader((tmp_path / "out" / "nested_matrix_group_summary.csv").open(newline="", encoding="utf-8")))
            confusion = list(csv.DictReader((tmp_path / "out" / "nested_matrix_confusion.csv").open(newline="", encoding="utf-8")))

            self.assertEqual([row["matrix_config_bundle"] for row in summary], ["feature", "logreg"])
            self.assertEqual({row["matrix_config_bundle"] for row in confusion}, {"feature", "logreg"})
            self.assertEqual({row["matrix_config_bundle"] for row in artifacts["confusion"]}, {"feature", "logreg"})
            self.assertTrue((tmp_path / "out" / "nested_matrix_feature_group_summary.csv").exists())
            self.assertTrue((tmp_path / "out" / "nested_matrix_feature_confusion.csv").exists())
            self.assertTrue((tmp_path / "out" / "nested_matrix_logreg_group_summary.csv").exists())
            self.assertTrue((tmp_path / "out" / "nested_matrix_logreg_confusion.csv").exists())


if __name__ == "__main__":
    unittest.main()
