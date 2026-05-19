import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import (
    AUTO_CLASSIFIER_PARAM_GRID_TOKEN,
    AUTO_COMPONENTS_PCA_GRID_TOKEN,
    CLASSIFIER_AUTO_PARAM_GRIDS,
    COMPONENTS_PCA_AUTO_GRID,
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

    def test_sensor_mean_slope_keeps_channel_mean_and_temporal_trend(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean_slope",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 4))
        np.testing.assert_allclose(feature_set.features[0], np.asarray([2.0, 4.0, 2.0, 4.0]))
        np.testing.assert_allclose(feature_set.features[1], np.asarray([3.0, 6.0, 2.0, 4.0]))

    def test_sensor_mean_slope_std_keeps_channel_summary_moments(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean_slope_std",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(feature_set.features[0], np.asarray([2.0, 4.0, 2.0, 4.0, 1.0, 2.0]))
        np.testing.assert_allclose(feature_set.features[1], np.asarray([3.0, 6.0, 2.0, 4.0, 1.0, 2.0]))

    def test_sensor_mean_slope_supports_baseline_whitening(self):
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
            feature_mode="sensor_mean_slope",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 4))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 4))
        self.assertEqual(feature_set.baseline_whitening_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_mean_slope_std_supports_baseline_whitening(self):
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
            feature_mode="sensor_mean_slope_std",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_whitening_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

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
            trial_selection="first",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.labels.tolist(), [1, 2, 1, 2])
        self.assertEqual(feature_set.features.shape[0], 4)
        self.assertEqual(feature_set.max_trials_per_class_per_participant, 2)

    def test_trial_cap_random_selection_is_seeded_and_not_file_order(self):
        labels = np.asarray([1, 2, 1, 2, 1, 2], dtype=int)

        selected = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="random",
            seed=0,
            participant=1,
        )
        repeated = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="random",
            seed=0,
            participant=1,
        )
        legacy = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="first",
            seed=0,
            participant=1,
        )

        self.assertEqual(selected.tolist(), [1, 2, 3, 4])
        self.assertEqual(repeated.tolist(), selected.tolist())
        self.assertEqual(legacy.tolist(), [0, 1, 2, 3])

    def test_random_trial_cap_preserves_original_trial_indices(self):
        data_by_participant = {1: _mat_data([1, 2, 1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9, -0.8, 0.8])}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            normalization="none",
            components_pca=float("inf"),
            max_trials_per_class_per_participant=2,
            trial_selection="random",
            trial_selection_seed=0,
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.trial_indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(feature_set.labels.tolist(), [2, 1, 2, 1])

    def test_auto_classifier_param_grid_expands_per_classifier(self):
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.2,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm", "shrinkage-lda", "regularized-qda"),
            classifier_params=(AUTO_CLASSIFIER_PARAM_GRID_TOKEN,),
            components_pca_values=(64,),
        )

        params_by_classifier = {
            classifier: tuple(config.classifier_param for config in candidate_configs if config.classifier == classifier)
            for classifier in ("multiclass-svm", "shrinkage-lda", "regularized-qda")
        }

        self.assertEqual(len(candidate_configs), 10)
        self.assertEqual(params_by_classifier["multiclass-svm"], CLASSIFIER_AUTO_PARAM_GRIDS["multiclass-svm"])
        self.assertEqual(params_by_classifier["shrinkage-lda"], CLASSIFIER_AUTO_PARAM_GRIDS["shrinkage-lda"])
        self.assertEqual(params_by_classifier["regularized-qda"], CLASSIFIER_AUTO_PARAM_GRIDS["regularized-qda"])

    def test_auto_classifier_param_grid_preserves_explicit_classifier_params_once(self):
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.2,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(AUTO_CLASSIFIER_PARAM_GRID_TOKEN, 1.0, 100.0),
            components_pca_values=(64,),
        )

        self.assertEqual(tuple(config.classifier_param for config in candidate_configs), (0.1, 1.0, 10.0, 100.0))

    def test_auto_components_pca_grid_expands_candidate_configs(self):
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.2,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(AUTO_COMPONENTS_PCA_GRID_TOKEN,),
        )

        self.assertEqual(tuple(config.components_pca for config in candidate_configs), COMPONENTS_PCA_AUTO_GRID)

    def test_auto_components_pca_grid_preserves_explicit_values_once(self):
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.2,),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(AUTO_COMPONENTS_PCA_GRID_TOKEN, 64, 256),
        )

        self.assertEqual(tuple(config.components_pca for config in candidate_configs), (32, 64, 128, 256))

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

    def test_summarize_cross_subject_confusion_category_enrichment(self):
        prediction_rows = [
            {"test_participant": 1, "true_stimulus": 1, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 2, "true_stimulus": 1, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 1, "true_stimulus": 2, "predicted_stimulus": 1, "classifier": "logistic"},
            {"test_participant": 2, "true_stimulus": 3, "predicted_stimulus": 2, "classifier": "logistic"},
            {"test_participant": 3, "true_stimulus": 4, "predicted_stimulus": 3, "classifier": "logistic"},
            {"test_participant": 3, "true_stimulus": 4, "predicted_stimulus": 4, "classifier": "logistic"},
        ]
        metadata_rows = [
            {"stimulus": "1", "name": "apple", "category": "food"},
            {"stimulus": "2", "name": "pear", "category": "food"},
            {"stimulus": "3", "name": "hammer", "category": "tool"},
            {"stimulus": "4", "name": "saw", "category": "tool"},
        ]

        enrichment_rows = cross_subject.summarize_cross_subject_confusion_category_enrichment(
            prediction_rows,
            stimulus_metadata_rows=metadata_rows,
            category_columns=("category",),
            n_permutations=128,
            seed=0,
        )
        matrix_rows = cross_subject.summarize_cross_subject_confusion_category_matrix(
            prediction_rows,
            stimulus_metadata_rows=metadata_rows,
            category_columns=("category",),
        )

        self.assertEqual(len(enrichment_rows), 1)
        enrichment = enrichment_rows[0]
        self.assertEqual(enrichment["category_column"], "category")
        self.assertEqual(enrichment["n_errors_with_category"], 5)
        self.assertEqual(enrichment["same_category_errors"], 4)
        self.assertAlmostEqual(enrichment["expected_same_category_errors"], 14 / 5)
        self.assertAlmostEqual(enrichment["same_category_lift"], 4 / (14 / 5))
        self.assertEqual(enrichment["n_participants_with_category_errors"], 3)
        self.assertEqual(enrichment["n_participants_with_same_category_errors"], 3)
        self.assertLessEqual(enrichment["same_category_permutation_p_value"], 1.0)

        food_to_food = next(row for row in matrix_rows if row["true_category"] == "food" and row["predicted_category"] == "food")
        self.assertTrue(food_to_food["same_category"])
        self.assertEqual(food_to_food["count"], 3)
        self.assertAlmostEqual(food_to_food["expected_count"], 12 / 5)
        self.assertAlmostEqual(food_to_food["category_confusion_lift"], 3 / (12 / 5))

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
        self.assertTrue(all(row["selected_inner_winner_margin"] > 0.0 for row in artifacts["selected"]))
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top2_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual({row["top3_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["selection_mode"], "nested_loso")
        self.assertEqual(artifacts["group_summary"][0]["n_candidates"], 2)
        self.assertEqual(artifacts["group_summary"][0]["selected_classifier_counts"], "multiclass-svm:4")
        self.assertEqual(artifacts["group_summary"][0]["selected_window_center_counts"], "0.2:4")
        self.assertEqual(artifacts["group_summary"][0]["selected_feature_mode_counts"], "sensor_mean:4")
        self.assertEqual(artifacts["group_summary"][0]["selected_normalization_counts"], "none:4")
        self.assertEqual(artifacts["group_summary"][0]["selected_alignment_counts"], "none:4")
        self.assertEqual(artifacts["group_summary"][0]["selected_components_pca_counts"], "inf:4")
        self.assertGreater(artifacts["group_summary"][0]["inner_winner_margin_mean"], 0.0)
        self.assertGreater(artifacts["group_summary"][0]["inner_winner_margin_median"], 0.0)
        self.assertGreater(artifacts["group_summary"][0]["inner_winner_margin_min"], 0.0)
        self.assertEqual(artifacts["group_summary"][0]["top2_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["top3_accuracy_mean"], 1.0)
        self.assertEqual(artifacts["group_summary"][0]["participants_above_chance"], 4)
        self.assertEqual(artifacts["group_summary"][0]["participants_total"], 4)
        self.assertAlmostEqual(artifacts["group_summary"][0]["one_sided_exact_sign_p_value"], 1 / 16)
        self.assertEqual(fit_model.call_count, 16)

    def test_nested_cross_subject_can_ensemble_top_inner_candidates(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
            4: _mat_data([1, 2, 1, 2], [-1.1, 1.1, -1.0, 1.0]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.150, 0.200),
            window_size=0.1,
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )

        with (
            patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)),
            patch("pymegdec.stimulus_cross_subject.fit_reptrace_window_model", wraps=cross_subject.fit_reptrace_window_model) as fit_model,
        ):
            artifacts = evaluate_nested_cross_subject_stimulus(
                "unused",
                [1, 2, 3, 4],
                candidate_configs=candidate_configs,
                selection_ensemble_size=2,
            )

        self.assertEqual(len(artifacts["outer"]), 4)
        self.assertEqual({row["classifier"] for row in artifacts["outer"]}, {"nested_topk_score_ensemble"})
        self.assertEqual({row["selection_ensemble_size"] for row in artifacts["selected"]}, {2})
        self.assertEqual({row["selection_ensemble_score_normalization"] for row in artifacts["selected"]}, {"row_z_softmax"})
        self.assertEqual({row["selection_ensemble_weighting"] for row in artifacts["selected"]}, {"uniform"})
        self.assertTrue(all(";" in row["selected_candidate_indices"] for row in artifacts["selected"]))
        self.assertTrue(all(row["selected_ensemble_weights"] in {"1:0.5;2:0.5", "2:0.5;1:0.5"} for row in artifacts["selected"]))
        self.assertTrue(all(row["ensemble_score_normalization"] == "row_z_softmax" for row in artifacts["outer"]))
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["outer_evaluation_mode"], "topk_score_ensemble")
        self.assertEqual(artifacts["group_summary"][0]["selection_ensemble_size"], 2)
        self.assertEqual(artifacts["group_summary"][0]["selection_ensemble_score_normalization"], "row_z_softmax")
        self.assertEqual(artifacts["group_summary"][0]["selection_ensemble_weighting"], "uniform")
        self.assertIn("1:", artifacts["group_summary"][0]["selected_ensemble_candidate_counts"])
        self.assertIn("2:", artifacts["group_summary"][0]["selected_ensemble_candidate_counts"])
        self.assertEqual(fit_model.call_count, 20)

    def test_nested_ensemble_weights_can_follow_inner_validation_scores(self):
        rows = [
            {"selected_inner_balanced_accuracy_mean": 0.70, "selected_inner_balanced_accuracy_sem": 0.07},
            {"selected_inner_balanced_accuracy_mean": 0.66, "selected_inner_balanced_accuracy_sem": 0.01},
            {"selected_inner_balanced_accuracy_mean": 0.62, "selected_inner_balanced_accuracy_sem": 0.00},
        ]

        uniform = cross_subject._nested_ensemble_weights(rows, weighting="uniform", temperature=0.02)  # pylint: disable=protected-access
        weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_softmax", temperature=0.02)  # pylint: disable=protected-access
        lcb_weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_lcb_softmax", temperature=0.02)  # pylint: disable=protected-access

        np.testing.assert_allclose(uniform, np.asarray([1 / 3, 1 / 3, 1 / 3]))
        self.assertAlmostEqual(float(np.sum(weighted)), 1.0)
        self.assertGreater(weighted[0], weighted[1])
        self.assertGreater(weighted[1], weighted[2])
        self.assertGreater(weighted[0], 0.80)
        self.assertAlmostEqual(float(np.sum(lcb_weighted)), 1.0)
        self.assertGreater(lcb_weighted[1], lcb_weighted[0])
        self.assertGreater(lcb_weighted[0], lcb_weighted[2])

    def test_nested_ensemble_can_prefer_diverse_candidate_windows(self):
        configs = (
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, normalization="none", classifier="shrinkage-lda"),
            CrossSubjectStimulusConfig(window_center=0.20, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
        )

        def inner_row(candidate_index, balanced_accuracy, window_center, classifier):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": window_center,
                "window_size_s": 0.1,
                "window_start_s": window_center - 0.05,
                "window_stop_s": window_center + 0.05,
                "feature_mode": "sensor_mean",
                "normalization": "none",
                "alignment": "none",
                "classifier": classifier,
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
            }

        inner_rows = [
            inner_row(1, 0.90, 0.10, "multiclass-svm"),
            inner_row(2, 0.85, 0.10, "shrinkage-lda"),
            inner_row(3, 0.80, 0.20, "multiclass-svm"),
        ]

        top_two, _top_two_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )
        diverse, _diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(top_two["selected_candidate_indices"], "1;2")
        self.assertEqual(diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(diverse["selection_ensemble_diversity"], "window")
        self.assertEqual(diverse["selected_ensemble_window_center_counts"], "0.1:1;0.2:1")

    def test_rank_softmax_score_normalization_ignores_score_scale(self):
        small_scale = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            np.asarray([[0.30, 0.10, 0.20]], dtype=float),
            score_normalization="rank_softmax",
        )
        large_scale = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            np.asarray([[300.0, 100.0, 200.0]], dtype=float),
            score_normalization="rank-softmax",
        )
        row_z = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            np.asarray([[300.0, 100.0, 200.0]], dtype=float),
            score_normalization="row_z_softmax",
        )

        np.testing.assert_allclose(small_scale, large_scale)
        self.assertEqual(int(np.argmax(large_scale[0])), 0)
        self.assertEqual(int(np.argmax(row_z[0])), 0)
        self.assertGreater(large_scale[0, 0], large_scale[0, 2])
        self.assertGreater(large_scale[0, 2], large_scale[0, 1])

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

    def test_nested_cross_subject_can_try_train_class_procrustes_alignment(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
            4: _mat_data([1, 2, 1, 2], [-1.1, 1.1, -1.0, 1.0]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            window_size=0.1,
            feature_modes=("sensor_flat",),
            normalizations=("subject_trial_z",),
            alignments=("none", "train_class_procrustes"),
            classifiers=("multiclass-svm",),
            classifier_params=(0.5,),
            components_pca_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            artifacts = evaluate_nested_cross_subject_stimulus("unused", [1, 2, 3, 4], candidate_configs=candidate_configs)

        selected_alignments = {row["selected_alignment"] for row in artifacts["selected"]}
        self.assertTrue(selected_alignments <= {"none", "train_class_procrustes"})
        self.assertEqual({row["alignment"] for row in artifacts["inner_validation"]}, {"none", "train_class_procrustes"})
        self.assertIn("selected_alignment_counts", artifacts["group_summary"][0])
        self.assertTrue(all("alignment_common_classes" in row for row in artifacts["outer"]))

    def test_nested_cross_subject_label_shuffle_control_marks_outputs(self):
        data_by_participant = {
            1: _mat_data([1, 2, 1, 2], [-1.2, 1.2, -1.1, 1.1]),
            2: _mat_data([1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9]),
            3: _mat_data([1, 2, 1, 2], [-1.3, 1.3, -1.2, 1.2]),
        }
        candidate_configs = make_cross_subject_candidate_configs(
            window_centers=(0.1, 0.2),
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
                [1, 2, 3],
                candidate_configs=candidate_configs,
                label_shuffle_control=True,
                label_shuffle_seed=11,
            )

        self.assertEqual({row["label_shuffle_control"] for row in artifacts["outer"]}, {True})
        self.assertEqual({row["label_shuffle_seed"] for row in artifacts["outer"]}, {11})
        self.assertEqual({row["label_shuffle_control"] for row in artifacts["inner_validation"]}, {True})
        self.assertEqual({row["label_shuffle_seed"] for row in artifacts["inner_validation"]}, {11})
        self.assertEqual(artifacts["group_summary"][0]["label_shuffle_control"], True)
        self.assertEqual(artifacts["group_summary"][0]["label_shuffle_seed"], 11)
        self.assertEqual({row["true_stimulus"] for row in artifacts["predictions"]}, {1, 2})

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
        rows = [{"balanced_accuracy": 0.10, "accuracy": 0.10, "chance_accuracy": 1 / 16} for _ in range(23)]

        summary = summarize_cross_subject_stimulus_smoke(rows, config=config)

        self.assertEqual(summary[0]["participants_above_chance"], 23)
        self.assertEqual(summary[0]["participants_total"], 23)
        self.assertAlmostEqual(summary[0]["one_sided_exact_sign_p_value"], 1 / (2**23))


if __name__ == "__main__":
    unittest.main()
