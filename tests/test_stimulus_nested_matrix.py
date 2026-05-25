from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pymegdec.stimulus_nested_matrix import aggregate_nested_matrix_outputs, discover_nested_matrix_shards


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


class TestStimulusNestedMatrix(unittest.TestCase):
    def test_discovers_nested_matrix_shards_by_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_outer.csv", [_outer_row(1, balanced=0.1)])
            _write_csv(tmp_path / "nested-matrix-feature-p2" / "matrix_feature_p2_outer.csv", [_outer_row(2, balanced=0.2)])

            shards = discover_nested_matrix_shards(tmp_path)

            self.assertEqual(sorted(shards), ["feature", "logreg"])
            self.assertEqual([path.name for path in shards["logreg"]], ["matrix_logreg_p1_outer.csv"])

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

            artifacts = aggregate_nested_matrix_outputs(
                tmp_path,
                tmp_path / "out",
                output_stem="nested_matrix",
                signflip_permutations=0,
            )
            summary = list(csv.DictReader((tmp_path / "out" / "nested_matrix_group_summary.csv").open(newline="", encoding="utf-8")))
            selected = list(csv.DictReader((tmp_path / "out" / "nested_matrix_selected.csv").open(newline="", encoding="utf-8")))

            self.assertEqual(len(artifacts["outer"]), 2)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["matrix_config_bundle"], "logreg")
            self.assertAlmostEqual(float(summary[0]["balanced_accuracy_mean"]), 0.15)
            self.assertEqual(summary[0]["participants_total"], "2")
            self.assertEqual({row["matrix_config_bundle"] for row in selected}, {"logreg"})

    def test_aggregates_multiple_config_bundles_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_csv(tmp_path / "nested-matrix-logreg-p1" / "matrix_logreg_p1_outer.csv", [_outer_row(1, balanced=0.10)])
            _write_csv(
                tmp_path / "nested-matrix-feature-p1" / "matrix_feature_p1_outer.csv",
                [_outer_row(1, balanced=0.30, bundle_classifier="shrinkage-lda")],
            )

            aggregate_nested_matrix_outputs(
                tmp_path,
                tmp_path / "out",
                output_stem="nested_matrix",
                signflip_permutations=0,
            )
            summary = list(csv.DictReader((tmp_path / "out" / "nested_matrix_group_summary.csv").open(newline="", encoding="utf-8")))

            self.assertEqual([row["matrix_config_bundle"] for row in summary], ["feature", "logreg"])
            self.assertTrue((tmp_path / "out" / "nested_matrix_feature_group_summary.csv").exists())
            self.assertTrue((tmp_path / "out" / "nested_matrix_logreg_group_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
