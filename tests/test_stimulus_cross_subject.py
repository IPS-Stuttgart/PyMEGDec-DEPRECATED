import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    evaluate_cross_subject_stimulus_smoke,
    evaluate_nested_cross_subject_stimulus,
    export_nested_cross_subject_stimulus,
    load_participant_stimulus_features,
    make_cross_subject_candidate_configs,
    summarize_cross_subject_stimulus_smoke,
)
from tests.matlab_fixtures import cell_array


def _mat_data(labels, values):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    time = np.asarray([-0.5, 0.0, 0.1, 0.15, 0.2, 1.5], dtype=float)
    trials = []
    for label, value in zip(labels, values):
        signal = np.zeros((2, time.size), dtype=float)
        signal[:, (time >= 0.15) & (time <= 0.25)] = value
        signal[:, (time >= -0.5) & (time <= 0.0)] = 0.1 * label
        trials.append(signal)
    return {
        "trial": cell_array(trials),
        "time": cell_array([time for _ in trials]),
        "trialinfo": trialinfo,
    }


def _mat_data_from_trials(labels, trials, time):
    return {
        "trial": cell_array([np.asarray(trial, dtype=float) for trial in trials]),
        "time": cell_array([np.asarray(time, dtype=float) for _ in trials]),
        "trialinfo": np.array([[np.asarray(labels, dtype=int)]], dtype=object),
    }


def _loadmat_side_effect(data_by_participant):
    def loadmat(path):
        match = re.search(r"Part(\d+)Data\.mat$", str(path))
        if not match:
            raise AssertionError(f"Unexpected MAT path: {path}")
        participant = int(match.group(1))
        return {"data": np.array([data_by_participant[participant]], dtype=object)}

    return loadmat


def _drop_topk_fields(rows):
    excluded = {
        "top2_accuracy",
        "top2_percent",
        "top3_accuracy",
        "top3_percent",
        "top2_chance_accuracy",
        "top2_chance_percent",
        "top3_chance_accuracy",
        "top3_chance_percent",
        "mean_true_label_rank",
        "median_true_label_rank",
        "chance_mean_rank",
        "true_label_rank",
        "top2_correct",
        "top3_correct",
    }
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]


class TestStimulusCrossSubject(unittest.TestCase):
    def test_load_participant_stimulus_features_uses_main_data_only(self):
        data_by_participant = {1: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -1.0, 1.0])}
        config = CrossSubjectStimulusConfig(window_center=0.2, window_size=0.1, normalization="none", components_pca=float("inf"), chance_classes=2)

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)) as loadmat:
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 2))
        self.assertEqual(feature_set.n_window_samples, 2)
        self.assertEqual(feature_set.labels.tolist(), [1, 2, 1, 2])
        self.assertTrue(str(loadmat.call_args.args[0]).endswith("Part1Data.mat"))
        self.assertNotIn("CueData", str(loadmat.call_args.args[0]))

    def test_sensor_flat_subject_baseline_z_repeats_channel_stats(self):
        data_by_participant = {1: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -1.0, 1.0])}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            feature_mode="sensor_flat",
            normalization="subject_baseline_z",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 4))
        self.assertEqual(feature_set.n_window_samples, 2)
        self.assertEqual(feature_set.n_baseline_samples, 2)
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 4))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 4))
        self.assertTrue(np.allclose(feature_set.baseline_feature_mean[0, :2], feature_set.baseline_feature_mean[0, 2:]))
        self.assertTrue(np.allclose(feature_set.baseline_feature_std[0, :2], feature_set.baseline_feature_std[0, 2:]))

    def test_sensor_flat_subject_trial_z_normalizes_each_trial(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 5.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 6.0, 10.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_flat",
            normalization="subject_trial_z",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 4))
        self.assertTrue(np.allclose(np.mean(feature_set.features, axis=1), 0.0))
        self.assertTrue(np.allclose(np.std(feature_set.features, axis=1), 1.0))

    def test_sensor_flat_subject_baseline_whiten_uses_channel_covariance(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[-1.0, -0.8, 1.0, 1.2], [0.5, 0.7, 3.0, 3.2]],
            [[-0.4, -0.2, 2.0, 2.2], [1.1, 1.3, 4.0, 4.2]],
            [[0.2, 0.4, 3.0, 3.2], [1.7, 1.9, 5.0, 5.2]],
            [[0.8, 1.0, 4.0, 4.2], [2.3, 2.5, 6.0, 6.2]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2, 1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_flat",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 4))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 4))
        self.assertEqual(feature_set.baseline_whitening_matrix.shape, (2, 2))
        self.assertEqual(feature_set.n_baseline_samples, 2)
        self.assertTrue(np.allclose(feature_set.baseline_whitening_matrix, feature_set.baseline_whitening_matrix.T))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_load_participant_stimulus_features_can_cap_trials_per_class(self):
        data_by_participant = {1: _mat_data([1, 2, 1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9, -0.8, 0.8])}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            normalization="none",
            components_pca=float("inf"),
            max_trials_per_class_per_participant=2,
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.labels.tolist(), [1, 2, 1, 2])
        self.assertEqual(feature_set.features.shape[0], 4)
        self.assertEqual(feature_set.max_trials_per_class_per_participant, 2)

    def test_evaluate_cross_subject_stimulus_smoke(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            normalization="none",
            classifier="multiclass-svm",
            classifier_param=0.5,
            components_pca=float("inf"),
            chance_classes=2,
            signflip_permutations=128,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            artifacts = evaluate_cross_subject_stimulus_smoke("unused", [1, 2, 3], config=config)

        self.assertEqual(len(artifacts["outer"]), 3)
        self.assertEqual(len(artifacts["predictions"]), 12)
        self.assertEqual(len(artifacts["group_summary"]), 1)
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top2_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top3_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["mean_true_label_rank"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["top2_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["top3_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["mean_true_label_rank_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["participants_above_chance"], 3)
        self.assertEqual(artifacts["group_summary"][0]["participants_total"], 3)
        self.assertAlmostEqual(artifacts["group_summary"][0]["one_sided_exact_sign_p_value"], 1 / 8)
        self.assertEqual({row["true_stimulus"] for row in artifacts["predictions"]}, {1, 2})
        self.assertEqual({row["predicted_stimulus"] for row in artifacts["predictions"]}, {1, 2})
        self.assertEqual({row["true_label_rank"] for row in artifacts["predictions"]}, {1.0})
        self.assertEqual({row["top2_correct"] for row in artifacts["predictions"]}, {True})
        self.assertEqual({row["top3_correct"] for row in artifacts["predictions"]}, {True})

    def test_summarize_cross_subject_confusion_pairs(self):
        prediction_rows = [
            {"test_participant": 1, "true_stimulus": 1, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 2, "true_stimulus": 1, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 1, "true_stimulus": 2, "predicted_stimulus": 1, "classifier": "logistic"},
            {"test_participant": 2, "true_stimulus": 1, "predicted_stimulus": 1, "classifier": "logistic"},
            {"test_participant": 1, "true_stimulus": 2, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 2, "true_stimulus": 3, "predicted_stimulus": 2, "classifier": "logistic"},
        ]
        metadata_rows = [
            {"stimulus": "1", "name": "apple", "category": "food"},
            {"stimulus": "2", "name": "pear", "category": "food"},
            {"stimulus": "3", "name": "hammer", "category": "tool"},
        ]

        pair_rows = cross_subject.summarize_cross_subject_confusion_pairs(
            prediction_rows,
            stimulus_metadata_rows=metadata_rows,
        )

        self.assertEqual(len(pair_rows), 2)
        first = pair_rows[0]
        self.assertEqual(first["stimulus_a"], 1)
        self.assertEqual(first["stimulus_b"], 2)
        self.assertEqual(first["a_to_b_count"], 2)
        self.assertEqual(first["b_to_a_count"], 1)
        self.assertEqual(first["total_confusions"], 3)
        self.assertEqual(first["n_confused_participants"], 2)
        self.assertAlmostEqual(first["a_to_b_rate"], 2 / 3)
        self.assertAlmostEqual(first["b_to_a_rate"], 1 / 2)
        self.assertAlmostEqual(first["expected_a_to_b_count"], 1.5)
        self.assertAlmostEqual(first["expected_b_to_a_count"], 0.25)
        self.assertAlmostEqual(first["pair_confusion_lift"], 3 / 1.75)
        self.assertAlmostEqual(first["total_confusion_excess"], 1.25)
        self.assertAlmostEqual(first["pair_standardized_residual"], 1.25 / np.sqrt(1.75))
        self.assertEqual(first["stimulus_a_category"], "food")
        self.assertEqual(first["stimulus_b_category"], "food")
        self.assertTrue(first["same_category"])

    def test_nested_cross_subject_selects_from_inner_loso_only(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
            4: _mat_data([1, 2, 1, 2], [-1.1, 1.1, -1.0, 1.0]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.1, 0.2),
            window_size=0.01,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )
        candidate_configs = (
            candidate_configs[0],
            CrossSubjectStimulusConfig(
                window_center=0.2,
                window_size=0.1,
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                components_pca=float("inf"),
                chance_classes=2,
                signflip_permutations=128,
            ),
        )

        with (
            patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)),
            patch("pymegdec.stimulus_cross_subject.fit_reptrace_window_model", wraps=cross_subject.fit_reptrace_window_model) as fit_model,
        ):
            artifacts = evaluate_nested_cross_subject_stimulus("unused", [1, 2, 3, 4], candidate_configs=candidate_configs)

        self.assertEqual(len(artifacts["outer"]), 4)
        self.assertEqual(len(artifacts["inner_validation"]), 24)
        self.assertEqual(len(artifacts["selected"]), 4)
        self.assertEqual(len(artifacts["predictions"]), 16)
        self.assertEqual({row["selected_candidate_index"] for row in artifacts["selected"]}, {2})
        self.assertEqual({row["selected_candidate_index"] for row in artifacts["outer"]}, {2})
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top2_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top3_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["selection_mode"], "nested_loso")
        self.assertEqual(artifacts["group_summary"][0]["n_candidates"], 2)
        self.assertEqual(artifacts["group_summary"][0]["top2_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["top3_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["participants_above_chance"], 4)
        self.assertEqual(artifacts["group_summary"][0]["participants_total"], 4)
        self.assertAlmostEqual(artifacts["group_summary"][0]["one_sided_exact_sign_p_value"], 1 / 16)
        self.assertEqual(fit_model.call_count, 16)

    def test_nested_cross_subject_can_evaluate_outer_subset(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
            4: _mat_data([1, 2, 1, 2], [-1.1, 1.1, -1.0, 1.0]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )
        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            artifacts = evaluate_nested_cross_subject_stimulus(
                "unused",
                [1, 2, 3, 4],
                candidate_configs=candidate_configs,
                outer_participants=[2, 4],
            )

        self.assertEqual({row["test_participant"] for row in artifacts["outer"]}, {2, 4})
        self.assertEqual({row["outer_test_participant"] for row in artifacts["inner_validation"]}, {2, 4})
        self.assertEqual({row["test_participant"] for row in artifacts["selected"]}, {2, 4})
        self.assertEqual(len(artifacts["predictions"]), 8)
        self.assertEqual(artifacts["group_summary"][0]["n_outer_folds"], 2)

    def test_nested_export_resumes_existing_outer_rows(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
            4: _mat_data([1, 2, 1, 2], [-1.1, 1.1, -1.0, 1.0]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )
        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            full_artifacts = evaluate_nested_cross_subject_stimulus("unused", [1, 2, 3, 4], candidate_configs=candidate_configs)

        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            paths = {
                "outer": output_dir / "outer.csv",
                "summary": output_dir / "summary.csv",
                "inner": output_dir / "inner.csv",
                "selected": output_dir / "selected.csv",
                "predictions": output_dir / "predictions.csv",
                "confusion": output_dir / "confusion.csv",
                "per_stimulus": output_dir / "per_stimulus.csv",
            }
            cross_subject.write_alpha_metrics_csv(_drop_topk_fields(row for row in full_artifacts["outer"] if int(row["test_participant"]) == 1), paths["outer"])
            cross_subject.write_alpha_metrics_csv(_drop_topk_fields(row for row in full_artifacts["inner_validation"] if int(row["outer_test_participant"]) == 1), paths["inner"])
            cross_subject.write_alpha_metrics_csv([row for row in full_artifacts["selected"] if int(row["test_participant"]) == 1], paths["selected"])
            cross_subject.write_alpha_metrics_csv(_drop_topk_fields(row for row in full_artifacts["predictions"] if int(row["test_participant"]) == 1), paths["predictions"])
            progress_messages = []
            with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
                resumed_artifacts = export_nested_cross_subject_stimulus(
                    "unused",
                    [1, 2, 3, 4],
                    candidate_configs=candidate_configs,
                    outer_output_path=paths["outer"],
                    group_summary_output_path=paths["summary"],
                    inner_validation_output_path=paths["inner"],
                    selected_output_path=paths["selected"],
                    predictions_output_path=paths["predictions"],
                    confusion_output_path=paths["confusion"],
                    per_stimulus_output_path=paths["per_stimulus"],
                    resume=True,
                    write_incremental=True,
                    progress=progress_messages.append,
                )

        self.assertEqual(len(resumed_artifacts["outer"]), 4)
        self.assertEqual({int(row["test_participant"]) for row in resumed_artifacts["outer"]}, {1, 2, 3, 4})
        self.assertIn("SKIP outer_test_participant=1 resume=complete", progress_messages)

    def test_summarize_cross_subject_stimulus_smoke_signflip(self):
        config = CrossSubjectStimulusConfig(chance_classes=2, signflip_permutations=128)
        rows = [
            {"balanced_accuracy": 0.75, "accuracy": 0.75, "chance_accuracy": 0.5},
            {"balanced_accuracy": 1.0, "accuracy": 1.0, "chance_accuracy": 0.5},
        ]

        summary = summarize_cross_subject_stimulus_smoke(rows, config=config)

        self.assertEqual(summary[0]["n_outer_folds"], 2)
        self.assertAlmostEqual(summary[0]["balanced_accuracy_mean"], 0.875)
        self.assertEqual(summary[0]["participants_above_chance"], 2)
        self.assertEqual(summary[0]["participants_total"], 2)
        self.assertAlmostEqual(summary[0]["one_sided_exact_sign_p_value"], 0.25)
        self.assertLessEqual(summary[0]["one_sided_signflip_p_value"], 1.0)

    def test_summarize_cross_subject_stimulus_smoke_exact_sign_all_23(self):
        config = CrossSubjectStimulusConfig(chance_classes=16, signflip_permutations=128)
        rows = [
            {"balanced_accuracy": 0.10, "accuracy": 0.10, "chance_accuracy": 1 / 16}
            for _ in range(23)
        ]

        summary = summarize_cross_subject_stimulus_smoke(rows, config=config)

        self.assertEqual(summary[0]["participants_above_chance"], 23)
        self.assertEqual(summary[0]["participants_total"], 23)
        self.assertAlmostEqual(summary[0]["one_sided_exact_sign_p_value"], 1 / (2**23))


if __name__ == "__main__":
    unittest.main()
