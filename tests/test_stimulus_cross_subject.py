import re
import tempfile
import unittest
from collections import Counter
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
    def test_fractional_rank_softmax_temperature_modes_are_supported(self):
        scores = np.asarray([[3.0, 2.0, 1.0]], dtype=float)
        default_probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )
        softer_probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax_t1_5",
        )

        self.assertEqual(
            cross_subject._normalize_ensemble_score_normalization("rank-softmax-t1-5"),  # pylint: disable=protected-access
            "rank_softmax_t1_5",
        )
        self.assertLess(softer_probabilities[0, 0], default_probabilities[0, 0])
        self.assertGreater(softer_probabilities[0, 1], default_probabilities[0, 1])

    def test_trial_entropy_ensemble_weighting_prefers_sharp_model_per_trial(self):
        score_matrices = (
            np.asarray(
                [
                    [8.0, 1.0, 0.0],
                    [1.0, 0.9, 0.8],
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    [1.0, 0.9, 0.8],
                    [0.0, 1.0, 8.0],
                ],
                dtype=float,
            ),
        )

        weights = cross_subject._trial_margin_ensemble_weights(  # pylint: disable=protected-access
            score_matrices,
            np.asarray([0.5, 0.5], dtype=float),
            "inner_lcb_trial_entropy_softmax",
        )

        self.assertEqual(weights.shape, (2, 2))
        self.assertGreater(weights[0, 0], weights[1, 0])
        self.assertGreater(weights[1, 1], weights[0, 1])
        np.testing.assert_allclose(np.sum(weights, axis=0), np.ones(2))

    def test_fractional_rank_softmax_inner_confusion_modes_are_supported(self):
        modes = [
            "rank_softmax_t1_5_inner_confusion_soft_guarded",
            "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded_guarded_balanced_quota",
        ]

        for mode in modes:
            with self.subTest(mode=mode):
                self.assertEqual(
                    cross_subject._normalize_ensemble_score_normalization(mode),  # pylint: disable=protected-access
                    mode,
                )
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
                    "rank_softmax_t1_5",
                )

    def test_top3_adaptive_score_softmax_inner_recall_bias_mode_is_supported(self):
        mode = "rank_top3_adaptive_score_softmax_inner_recall_bias"

        self.assertEqual(
            cross_subject._normalize_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            mode,
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_adaptive_score_softmax",
        )

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

    def test_sensor_flat_time_bins7_keeps_fine_temporal_shape(self):
        time = np.asarray([-0.5, 0.0, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16], dtype=float)
        trials = [
            [
                [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            ],
            [
                [0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                [0.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
            ],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.13,
            window_size=0.06,
            feature_mode="sensor_flat_time_bins7",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 14))
        self.assertEqual(feature_set.n_window_samples, 7)
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0]),
        )

    def test_sensor_temporal_pyramid_keeps_multiscale_channel_means(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 4.0, 8.0, 12.0, 16.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.25,
            window_size=0.3,
            feature_mode="sensor_temporal_pyramid",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 14))
        self.assertEqual(feature_set.n_window_samples, 4)
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        np.testing.assert_allclose(
            feature_set.features[1],
            np.asarray([5.0, 10.0, 3.0, 6.0, 7.0, 14.0, 2.0, 4.0, 4.0, 8.0, 6.0, 12.0, 8.0, 16.0]),
        )

    def test_sensor_temporal_pyramid_supports_baseline_whitening(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
        trials = [
            [[-1.0, -0.8, 1.0, 1.2, 1.4, 1.6], [0.5, 0.7, 3.0, 3.2, 3.4, 3.6]],
            [[-0.4, -0.2, 2.0, 2.2, 2.4, 2.6], [1.1, 1.3, 4.0, 4.2, 4.4, 4.6]],
            [[0.2, 0.4, 3.0, 3.2, 3.4, 3.6], [1.7, 1.9, 5.0, 5.2, 5.4, 5.6]],
            [[0.8, 1.0, 4.0, 4.2, 4.4, 4.6], [2.3, 2.5, 6.0, 6.2, 6.4, 6.6]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2, 1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.25,
            window_size=0.3,
            feature_mode="sensor_temporal_pyramid",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 14))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 14))
        self.assertEqual(feature_set.baseline_whitening_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_inner_confusion_complement_diversity_prefers_distinct_errors(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            window_size=0.1,
            feature_modes=("sensor_flat",),
            normalizations=("subject_baseline_whiten",),
            alignments=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(0.3, 1.0, 2.0),
            components_pca_values=(128,),
            chance_classes=4,
        )
        ranked_rows = [
            {
                "selected_candidate_index": 1,
                "selected_inner_selection_ranking_score": 0.1200,
                "selected_inner_selection_score_mean": 0.1200,
                "selected_inner_balanced_accuracy_mean": 0.1200,
                "selected_inner_top2_accuracy_mean": 0.2200,
                "selected_inner_top3_accuracy_mean": 0.3200,
                "selected_inner_mean_true_label_rank_mean": 2.0,
                "selected_inner_accuracy_mean": 0.1200,
                "selected_inner_confusion_counts": "1>2:10;2>1:10",
            },
            {
                "selected_candidate_index": 2,
                "selected_inner_selection_ranking_score": 0.1190,
                "selected_inner_selection_score_mean": 0.1190,
                "selected_inner_balanced_accuracy_mean": 0.1190,
                "selected_inner_top2_accuracy_mean": 0.2190,
                "selected_inner_top3_accuracy_mean": 0.3190,
                "selected_inner_mean_true_label_rank_mean": 2.1,
                "selected_inner_accuracy_mean": 0.1190,
                "selected_inner_confusion_counts": "1>2:10;2>1:10",
            },
            {
                "selected_candidate_index": 3,
                "selected_inner_selection_ranking_score": 0.1180,
                "selected_inner_selection_score_mean": 0.1180,
                "selected_inner_balanced_accuracy_mean": 0.1180,
                "selected_inner_top2_accuracy_mean": 0.2180,
                "selected_inner_top3_accuracy_mean": 0.3180,
                "selected_inner_mean_true_label_rank_mean": 2.2,
                "selected_inner_accuracy_mean": 0.1180,
                "selected_inner_confusion_counts": "3>4:10;4>3:10",
            },
        ]

        selected = cross_subject._select_diverse_nested_rows(  # pylint: disable=protected-access
            ranked_rows,
            requested_size=2,
            candidate_configs=configs,
            diversity="inner_confusion_complement",
        )

        self.assertEqual(
            [row["selected_candidate_index"] for row in selected],
            [1, 3],
        )

    def test_sensor_dct_keeps_low_order_temporal_coefficients(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 4.0, 8.0, 12.0, 16.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.25,
            window_size=0.3,
            feature_mode="sensor_dct",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertIn("sensor_dct", cross_subject.FEATURE_MODES)
        self.assertEqual(feature_set.features.shape, (2, 16))
        self.assertEqual(feature_set.n_window_samples, 4)
        np.testing.assert_allclose(feature_set.features[0, :2], np.asarray([8.0, 10.0]))
        np.testing.assert_allclose(feature_set.features[1, :2], np.asarray([10.0, 20.0]))

    def test_sensor_dct_supports_baseline_whitening(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
        trials = [
            [[-1.0, -0.8, 1.0, 1.2, 1.4, 1.6], [0.5, 0.7, 3.0, 3.2, 3.4, 3.6]],
            [[-0.4, -0.2, 2.0, 2.2, 2.4, 2.6], [1.1, 1.3, 4.0, 4.2, 4.4, 4.6]],
            [[0.2, 0.4, 3.0, 3.2, 3.4, 3.6], [1.7, 1.9, 5.0, 5.2, 5.4, 5.6]],
            [[0.8, 1.0, 4.0, 4.2, 4.4, 4.6], [2.3, 2.5, 6.0, 6.2, 6.4, 6.6]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2, 1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.25,
            window_size=0.3,
            feature_mode="sensor_dct",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 16))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 16))
        self.assertEqual(feature_set.baseline_whitening_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_rank_reciprocal_score_normalization_softens_rank_probabilities(self):
        self.assertIn("rank_reciprocal", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        scores = np.asarray(
            [
                [3.0, 2.0, 1.0, 0.0],
                [np.nan, 10.0, 9.0, 8.0],
            ],
            dtype=float,
        )

        probabilities = cross_subject._class_score_probabilities(scores, score_normalization="rank_reciprocal")

        expected = np.asarray([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=float)
        np.testing.assert_allclose(probabilities[0], expected / np.sum(expected))
        self.assertEqual(probabilities[1, 0], 0.0)
        self.assertGreater(probabilities[1, 1], probabilities[1, 2])
        self.assertGreater(probabilities[1, 2], probabilities[1, 3])
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))

    def test_rank_topk_vote_score_normalization_votes_for_near_top_classes(self):
        self.assertIn("rank_top2_vote", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn("rank_top3_vote", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        scores = np.asarray(
            [
                [4.0, 3.0, 2.0, 1.0],
                [np.nan, 10.0, 9.0, 8.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=float,
        )

        top2 = cross_subject._class_score_probabilities(scores, score_normalization="rank_top2_vote")
        top3 = cross_subject._class_score_probabilities(scores, score_normalization="rank_top3_vote")

        np.testing.assert_allclose(top2[0], np.asarray([0.5, 0.5, 0.0, 0.0]))
        np.testing.assert_allclose(top2[1], np.asarray([0.0, 0.5, 0.5, 0.0]))
        np.testing.assert_allclose(top2[2], np.full(4, 0.25))
        np.testing.assert_allclose(top3[0], np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]))
        np.testing.assert_allclose(top3[1], np.asarray([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        np.testing.assert_allclose(np.sum(top2, axis=1), np.ones(3))
        np.testing.assert_allclose(np.sum(top3, axis=1), np.ones(3))

    def test_rank_consensus_ensemble_score_normalization_rewards_agreement(self):
        self.assertIn("rank_consensus", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        score_matrices = np.asarray(
            [
                [[4.0, 3.9, 1.0], [4.0, 1.0, 3.9]],
                [[1.0, 4.0, 3.0], [1.0, 3.8, 4.0]],
                [[1.0, 4.0, 3.0], [1.0, 3.7, 4.0]],
            ],
            dtype=float,
        )

        probabilities = cross_subject._rank_consensus_ensemble_probabilities(  # pylint: disable=protected-access
            score_matrices,
            np.full(3, 1.0 / 3.0, dtype=float),
        )

        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
        self.assertGreater(probabilities[0, 1], probabilities[0, 0])
        self.assertGreater(probabilities[0, 1], probabilities[0, 2])
        self.assertGreater(probabilities[1, 2], probabilities[1, 0])
        self.assertGreater(probabilities[1, 2], probabilities[1, 1])

    def test_prediction_balance_selection_metric_penalizes_collapsed_predictions(self):
        metric = "balanced_top2_top3_rank_prediction_balance"
        collapsed = {
            "balanced_accuracy": 0.55,
            "accuracy": 0.55,
            "top2_accuracy": 0.90,
            "top3_accuracy": 1.00,
            "mean_true_label_rank": 1.20,
            "chance_mean_rank": 1.50,
            "chance_classes": 2,
            "test_label_counts": "1:10;2:10",
            "predicted_label_counts": "1:20",
        }
        balanced = dict(collapsed)
        balanced["predicted_label_counts"] = "1:10;2:10"

        self.assertIn(metric, cross_subject.CROSS_SUBJECT_SELECTION_METRIC_CHOICES)
        self.assertAlmostEqual(
            cross_subject._inner_prediction_balance_score(balanced),
            1.0,
        )
        self.assertLess(
            cross_subject._inner_prediction_balance_score(collapsed),
            1.0,
        )
        self.assertGreater(
            cross_subject._nested_row_selection_score(balanced, metric),
            cross_subject._nested_row_selection_score(collapsed, metric),
        )

    def test_inner_balanced_suffix_is_valid_for_soft_rank_normalizations(self):
        scores = np.asarray([[3.0, 1.0, 2.0], [0.0, 2.0, 1.0]], dtype=float)
        balanced_to_base = {
            "rank_softmax_inner_balanced": "rank_softmax",
            "rank_reciprocal_inner_balanced": "rank_reciprocal",
            "rank_borda_inner_balanced": "rank_borda",
            "rank_top2_vote_inner_balanced": "rank_top2_vote",
            "rank_top3_vote_inner_balanced": "rank_top3_vote",
            "rank_z_blend_inner_balanced": "rank_z_blend",
        }

        for balanced, base in balanced_to_base.items():
            with self.subTest(score_normalization=balanced):
                self.assertIn(balanced, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
                expected = cross_subject._class_score_probabilities(
                    scores, score_normalization=base
                )
                actual = cross_subject._class_score_probabilities(
                    scores, score_normalization=balanced
                )
                np.testing.assert_allclose(actual, expected)

    def test_inner_balanced_suffix_enables_prior_balance_for_rank_reciprocal(self):
        selected_rows = (
            {
                "selected_inner_test_label_counts": "1:10;2:10",
                "selected_inner_predicted_label_counts": "1:18;2:2",
            },
        )

        metadata = cross_subject._inner_class_prior_balance_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            "rank_reciprocal_inner_balanced",
        )

        adjusted = cross_subject._apply_inner_class_prior_balance(
            np.asarray([[0.60, 0.40]], dtype=float), metadata
        )

        self.assertEqual(metadata["mode"], "rank_reciprocal_inner_balanced")
        self.assertLess(adjusted[0, 0], 0.60)
        self.assertGreater(adjusted[0, 1], 0.40)

    def test_balanced_quota_rank_softmax_is_valid_base_normalization(self):
        self.assertIn(
            "rank_softmax_balanced_quota",
            cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES,
        )
        scores = np.asarray([[3.0, 1.0, 2.0], [0.0, 2.0, 1.0]], dtype=float)

        expected = cross_subject._class_score_probabilities(
            scores, score_normalization="rank_softmax"
        )
        actual = cross_subject._class_score_probabilities(
            scores, score_normalization="rank_softmax_balanced_quota"
        )

        np.testing.assert_allclose(actual, expected)

    def test_topk_vote_balanced_quota_modes_are_valid_base_normalizations(self):
        scores = np.asarray(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]], dtype=float
        )
        quota_to_base = {
            "rank_top2_vote_balanced_quota": "rank_top2_vote",
            "rank_top3_vote_balanced_quota": "rank_top3_vote",
            "rank_top2_vote_guarded_balanced_quota": "rank_top2_vote",
            "rank_top3_vote_guarded_balanced_quota": "rank_top3_vote",
        }

        for quota_mode, base_mode in quota_to_base.items():
            with self.subTest(score_normalization=quota_mode):
                self.assertIn(
                    quota_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES
                )
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(quota_mode),
                    base_mode,
                )
                expected = cross_subject._class_score_probabilities(
                    scores, score_normalization=base_mode
                )
                actual = cross_subject._class_score_probabilities(
                    scores, score_normalization=quota_mode
                )
                np.testing.assert_allclose(actual, expected)

    def test_inner_balanced_quota_suffix_combines_prior_balance_and_assignment(self):
        mode = "rank_reciprocal_inner_balanced_balanced_quota"
        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        scores = np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=float)

        expected = cross_subject._class_score_probabilities(
            scores,
            score_normalization="rank_reciprocal",
        )
        actual = cross_subject._class_score_probabilities(
            scores,
            score_normalization=mode,
        )

        np.testing.assert_allclose(actual, expected)
        selected_rows = (
            {
                "selected_inner_test_label_counts": "1:10;2:10",
                "selected_inner_predicted_label_counts": "1:18;2:2",
            },
        )
        metadata = cross_subject._inner_class_prior_balance_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            mode,
        )
        adjusted = cross_subject._apply_inner_class_prior_balance(
            np.asarray([[0.60, 0.40]], dtype=float),
            metadata,
        )
        quota = cross_subject._balanced_quota_metadata(
            np.asarray([[0.95, 0.05], [0.60, 0.40]], dtype=float),
            np.asarray([0, 1], dtype=int),
            mode,
        )

        self.assertEqual(metadata["mode"], mode)
        self.assertEqual(metadata["inner_mode"], "rank_reciprocal_inner_balanced")
        self.assertLess(adjusted[0, 0], 0.60)
        self.assertGreater(adjusted[0, 1], 0.40)
        self.assertEqual(quota["status"], "applied")
        self.assertEqual(
            np.bincount(quota["predictions"], minlength=2).tolist(), [1, 1]
        )

    def test_inner_confusion_quota_suffix_combines_correction_and_assignment(self):
        mode = "rank_softmax_inner_confusion_balanced_quota"
        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        selected_rows = (
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:2;2001:8;2002:10"
                ),
            },
        )
        metadata = cross_subject._inner_confusion_correction_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            mode,
        )
        probabilities = np.asarray([[0.70, 0.30], [0.65, 0.35]], dtype=float)
        adjusted = cross_subject._apply_inner_confusion_correction(
            probabilities, metadata
        )
        quota = cross_subject._balanced_quota_metadata(
            adjusted, np.asarray([0, 1], dtype=int), mode
        )

        self.assertEqual(metadata["mode"], mode)
        self.assertEqual(metadata["inner_mode"], "rank_softmax_inner_confusion")
        self.assertEqual(metadata["status"], "applied")
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])
        self.assertEqual(quota["status"], "applied")
        self.assertEqual(
            np.bincount(quota["predictions"], minlength=2).tolist(), [1, 1]
        )

    def test_guarded_inner_confusion_skips_near_chance_correction(self):
        mode = "rank_softmax_inner_confusion_guarded"
        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        selected_rows = (
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:10;1002:10;2001:10;2002:10"
                ),
            },
        )
        metadata = cross_subject._inner_confusion_correction_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            mode,
        )
        probabilities = np.asarray([[0.70, 0.30], [0.20, 0.80]], dtype=float)
        adjusted = cross_subject._apply_inner_confusion_correction(
            probabilities,
            metadata,
        )
        expected_base = cross_subject._class_score_probabilities(
            np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=float),
            score_normalization="rank_softmax",
        )
        guarded_base = cross_subject._class_score_probabilities(
            np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=float),
            score_normalization=mode,
        )

        self.assertEqual(metadata["status"], "skipped_low_inner_confusion_reliability")
        self.assertTrue(metadata["guarded"])
        self.assertEqual(metadata["guarded_reliability"], 0.0)
        np.testing.assert_allclose(adjusted, probabilities)
        np.testing.assert_allclose(guarded_base, expected_base)

    def test_balanced_quota_assignment_enforces_known_batch_balance(self):
        probabilities = np.asarray(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.70, 0.30],
                [0.60, 0.40],
            ],
            dtype=float,
        )

        metadata = cross_subject._balanced_quota_metadata(
            probabilities,
            np.asarray([0, 1], dtype=int),
            "rank_softmax_balanced_quota",
        )

        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["predictions"].tolist(), [0, 0, 1, 1])
        self.assertEqual(
            np.bincount(metadata["predictions"], minlength=2).tolist(),
            [2, 2],
        )

    def test_inner_class_prior_balance_downweights_overpredicted_class(self):
        probabilities = np.asarray([[0.60, 0.40]], dtype=float)
        selected_rows = (
            {
                "selected_inner_test_label_counts": "1:10;2:10",
                "selected_inner_predicted_label_counts": "1:18;2:2",
            },
        )
        metadata = cross_subject._inner_class_prior_balance_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            "rank_softmax_inner_balanced",
        )

        adjusted = cross_subject._apply_inner_class_prior_balance(
            probabilities, metadata
        )
        self.assertLess(adjusted[0, 0], probabilities[0, 0])
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])

    def test_inner_recall_bias_boosts_under_recalled_class(self):
        mode = "rank_softmax_inner_recall_bias"
        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax",
        )
        selected_rows = (
            {
                # True class 2 is mostly misclassified as class 1 in source-inner
                # validation, so the recall-bias correction should increase the
                # held-out posterior mass for class 2.
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:12;2001:10;2002:2"
                ),
            },
        )
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            mode,
        )
        probabilities = np.asarray([[0.55, 0.45]], dtype=float)

        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )
        row = {}
        cross_subject._add_inner_class_prior_balance_fields(row, metadata)  # pylint: disable=protected-access

        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["recall_bias_status"], "applied")
        self.assertLess(metadata["log_adjustment"][0], 0.0)
        self.assertGreater(metadata["log_adjustment"][1], 0.0)
        self.assertLess(adjusted[0, 0], probabilities[0, 0])
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])
        self.assertEqual(row["ensemble_inner_recall_bias_status"], "applied")

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

    def test_combined_inner_prior_confusion_score_normalization_is_supported(self):
        mode = "rank_softmax_inner_balanced_confusion"
        quota_mode = "rank_softmax_inner_balanced_confusion_balanced_quota"

        self.assertEqual(cross_subject._normalize_ensemble_score_normalization(mode), mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._normalize_ensemble_score_normalization(quota_mode), quota_mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._base_ensemble_score_normalization(mode), "rank_softmax")  # pylint: disable=protected-access
        self.assertEqual(cross_subject._base_ensemble_score_normalization(quota_mode), "rank_softmax")  # pylint: disable=protected-access
        self.assertEqual(cross_subject._inner_class_prior_balance_mode(mode), mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._inner_class_prior_balance_mode(quota_mode), mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._inner_confusion_correction_mode(mode), mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._inner_confusion_correction_mode(quota_mode), mode)  # pylint: disable=protected-access

    def test_log_pool_score_normalization_uses_geometric_probability_pool(self):
        mode = "rank_softmax_log_pool"
        self.assertEqual(cross_subject._normalize_ensemble_score_normalization(mode), mode)  # pylint: disable=protected-access
        self.assertEqual(cross_subject._base_ensemble_score_normalization(mode), "rank_softmax")  # pylint: disable=protected-access

        probability_matrices = (
            np.asarray([[0.90, 0.05, 0.05]], dtype=float),
            np.asarray([[0.05, 0.70, 0.25]], dtype=float),
            np.asarray([[0.05, 0.65, 0.30]], dtype=float),
        )
        weights = np.full(3, 1.0 / 3.0, dtype=float)

        arithmetic = cross_subject._pool_ensemble_probability_matrices(  # pylint: disable=protected-access
            probability_matrices,
            weights,
            "rank_softmax",
        )
        log_pool = cross_subject._pool_ensemble_probability_matrices(  # pylint: disable=protected-access
            probability_matrices,
            weights,
            mode,
        )

        np.testing.assert_allclose(np.sum(log_pool, axis=1), np.asarray([1.0]))
        self.assertGreater(log_pool[0, 1] - log_pool[0, 0], arithmetic[0, 1] - arithmetic[0, 0])

    def test_intermediate_rank_softmax_temperatures_are_supported(self):
        scores = np.asarray([[3.0, 2.0, 1.0]], dtype=float)
        modes = (
            "rank_softmax",
            "rank_softmax_t1_25",
            "rank_softmax_t1_5",
            "rank_softmax_t1_75",
            "rank_softmax_t2",
        )

        top_class_probabilities = []
        for mode in modes:
            with self.subTest(mode=mode):
                self.assertEqual(
                    cross_subject._normalize_ensemble_score_normalization(mode),
                    mode,
                )  # pylint: disable=protected-access
                probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
                    scores,
                    score_normalization=mode,
                )
                self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
                top_class_probabilities.append(float(probabilities[0, 0]))

        self.assertEqual(top_class_probabilities, sorted(top_class_probabilities, reverse=True))

    def test_guarded_balanced_quota_preserves_high_margin_argmax_trials(self):
        probabilities = np.asarray(
            [
                [0.99, 0.01],
                [0.98, 0.02],
                [0.51, 0.49],
                [0.50, 0.50],
            ],
            dtype=float,
        )
        class_order = np.asarray([0, 1], dtype=int)

        predictions, quotas, status, fixed_trials = cross_subject._guarded_balanced_quota_predictions(  # pylint: disable=protected-access
            probabilities,
            class_order,
        )

        self.assertEqual(status, "applied_guarded")
        self.assertEqual(quotas.tolist(), [2, 2])
        self.assertEqual(fixed_trials, 2)
        self.assertEqual(predictions[:2].tolist(), [0, 0])
        self.assertEqual(Counter(predictions.tolist()), Counter({0: 2, 1: 2}))

    def test_guarded_balanced_quota_normalization_uses_underlying_rank_base(self):
        mode = "rank_z_blend_guarded_balanced_quota"

        self.assertEqual(
            cross_subject._normalize_ensemble_score_normalization(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),
            "rank_z_blend",
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_class_prior_balance_mode(mode),
            None,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_confusion_correction_mode(mode),
            None,
        )  # pylint: disable=protected-access

    def test_soft_inner_confusion_score_normalizations_are_supported(self):
        expected_bases = {
            "rank_softmax_inner_confusion_soft": "rank_softmax",
            "rank_softmax_t1_25_inner_confusion_soft": "rank_softmax_t1_25",
            "rank_softmax_t1_5_inner_confusion_soft": "rank_softmax_t1_5",
            "rank_softmax_t1_75_inner_confusion_soft": "rank_softmax_t1_75",
            "rank_softmax_t2_inner_confusion_soft": "rank_softmax_t2",
            "rank_softmax_t3_inner_confusion_soft": "rank_softmax_t3",
            "rank_reciprocal_inner_confusion_soft": "rank_reciprocal",
            "rank_borda_inner_confusion_soft": "rank_borda",
            "rank_top2_vote_inner_confusion_soft": "rank_top2_vote",
            "rank_top3_vote_inner_confusion_soft": "rank_top3_vote",
        }

        for mode, expected_base in expected_bases.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    cross_subject._normalize_ensemble_score_normalization(mode),
                    mode,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(mode),
                    expected_base,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._inner_confusion_correction_mode(mode),
                    mode,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._inner_confusion_correction_blend(mode),
                    cross_subject.INNER_CONFUSION_CORRECTION_SOFT_BLEND,
                )  # pylint: disable=protected-access

    def test_soft_guarded_inner_confusion_score_normalizations_are_supported(self):
        expected_bases = {
            "rank_softmax_inner_confusion_soft_guarded": "rank_softmax",
            "rank_softmax_t1_25_inner_confusion_soft_guarded": "rank_softmax_t1_25",
            "rank_softmax_t1_5_inner_confusion_soft_guarded": "rank_softmax_t1_5",
            "rank_softmax_t1_75_inner_confusion_soft_guarded": "rank_softmax_t1_75",
            "rank_softmax_t2_inner_confusion_soft_guarded": "rank_softmax_t2",
            "rank_softmax_t3_inner_confusion_soft_guarded": "rank_softmax_t3",
            "rank_reciprocal_inner_confusion_soft_guarded": "rank_reciprocal",
            "rank_borda_inner_confusion_soft_guarded": "rank_borda",
            "rank_top2_vote_inner_confusion_soft_guarded": "rank_top2_vote",
            "rank_top3_vote_inner_confusion_soft_guarded": "rank_top3_vote",
        }

        for mode, expected_base in expected_bases.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    cross_subject._normalize_ensemble_score_normalization(mode),
                    mode,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(mode),
                    expected_base,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._inner_confusion_correction_mode(mode),
                    mode,
                )  # pylint: disable=protected-access
                self.assertEqual(
                    cross_subject._inner_confusion_correction_blend(mode),
                    cross_subject.INNER_CONFUSION_CORRECTION_SOFT_BLEND,
                )  # pylint: disable=protected-access
                self.assertTrue(
                    cross_subject._inner_confusion_correction_is_guarded(mode)
                )  # pylint: disable=protected-access

    def test_soft_inner_balanced_confusion_score_normalization_is_supported(self):
        mode = "rank_softmax_inner_balanced_confusion_soft"

        self.assertEqual(
            cross_subject._normalize_ensemble_score_normalization(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),
            "rank_softmax",
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_class_prior_balance_mode(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_confusion_correction_mode(mode),
            mode,
        )  # pylint: disable=protected-access

    def test_soft_guarded_inner_balanced_confusion_score_normalization_is_supported(
        self,
    ):
        mode = "rank_softmax_inner_balanced_confusion_soft_guarded"

        self.assertEqual(
            cross_subject._normalize_ensemble_score_normalization(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),
            "rank_softmax",
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_class_prior_balance_mode(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_confusion_correction_mode(mode),
            mode,
        )  # pylint: disable=protected-access
        self.assertEqual(
            cross_subject._inner_confusion_correction_blend(mode),
            cross_subject.INNER_CONFUSION_CORRECTION_SOFT_BLEND,
        )  # pylint: disable=protected-access
        self.assertTrue(
            cross_subject._inner_confusion_correction_is_guarded(mode)
        )  # pylint: disable=protected-access

    def test_margin_blend_supports_guarded_margin_inner_confusion(self):
        mode = "rank_margin_blend_inner_confusion_margin_soft_guarded"
        scores = np.asarray([[3.0, 1.0], [1.2, 1.0]], dtype=float)
        selected_rows = (
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:18;1002:2;2001:6;2002:14"
                ),
            },
        )

        expected_base = cross_subject._class_score_probabilities(
            scores,
            score_normalization="rank_margin_blend",
        )
        guarded_base = cross_subject._class_score_probabilities(
            scores,
            score_normalization=mode,
        )
        metadata = cross_subject._inner_confusion_correction_metadata(
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            mode,
        )
        probabilities = np.asarray([[0.55, 0.45], [0.90, 0.10]], dtype=float)
        adjusted = cross_subject._apply_inner_confusion_correction(
            probabilities,
            metadata,
        )

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        np.testing.assert_allclose(guarded_base, expected_base)
        self.assertEqual(metadata["mode"], mode)
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["status"], "applied")
        self.assertTrue(metadata["guarded"])
        self.assertTrue(metadata["margin_gated"])
        self.assertGreater(metadata["guarded_reliability"], 0.0)
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])
        np.testing.assert_allclose(adjusted[1], probabilities[1])

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

    def test_rank_median_consensus_prefers_majority_top_rank(self):
        score_matrices = (
            np.asarray([[4.0, 3.0, 2.0], [1.0, 5.0, 4.0]]),
            np.asarray([[1.0, 5.0, 4.0], [5.0, 4.0, 1.0]]),
            np.asarray([[0.0, 6.0, 5.0], [4.0, 6.0, 2.0]]),
        )

        probabilities = cross_subject._rank_median_consensus_ensemble_probabilities(  # pylint: disable=protected-access
            score_matrices,
            np.asarray([1 / 3, 1 / 3, 1 / 3]),
        )

        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(
            np.sum(probabilities, axis=1),
            np.ones(2),
        )
        self.assertGreater(probabilities[0, 1], probabilities[0, 0])
        self.assertGreater(probabilities[0, 1], probabilities[0, 2])
        self.assertGreater(probabilities[1, 1], probabilities[1, 0])
        self.assertGreater(probabilities[1, 1], probabilities[1, 2])

    def test_nested_ensemble_weights_can_follow_inner_validation_scores(self):
        rows = [
            {
                "selected_inner_balanced_accuracy_mean": 0.70,
                "selected_inner_balanced_accuracy_sem": 0.07,
                "selected_inner_selection_score_mean": 0.74,
                "selected_inner_selection_score_sem": 0.01,
            },
            {
                "selected_inner_balanced_accuracy_mean": 0.66,
                "selected_inner_balanced_accuracy_sem": 0.01,
                "selected_inner_selection_score_mean": 0.80,
                "selected_inner_selection_score_sem": 0.10,
            },
            {
                "selected_inner_balanced_accuracy_mean": 0.62,
                "selected_inner_balanced_accuracy_sem": 0.00,
                "selected_inner_selection_score_mean": 0.60,
                "selected_inner_selection_score_sem": 0.00,
            },
        ]

        uniform = cross_subject._nested_ensemble_weights(rows, weighting="uniform", temperature=0.02)  # pylint: disable=protected-access
        weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_softmax", temperature=0.02)  # pylint: disable=protected-access
        lcb_weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_lcb_softmax", temperature=0.02)  # pylint: disable=protected-access
        selection_weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_selection_softmax", temperature=0.02)  # pylint: disable=protected-access
        selection_lcb_weighted = cross_subject._nested_ensemble_weights(rows, weighting="inner_selection_lcb_softmax", temperature=0.02)  # pylint: disable=protected-access

        np.testing.assert_allclose(uniform, np.asarray([1 / 3, 1 / 3, 1 / 3]))
        self.assertAlmostEqual(float(np.sum(weighted)), 1.0)
        self.assertGreater(weighted[0], weighted[1])
        self.assertGreater(weighted[1], weighted[2])
        self.assertGreater(weighted[0], 0.80)
        self.assertAlmostEqual(float(np.sum(lcb_weighted)), 1.0)
        self.assertGreater(lcb_weighted[1], lcb_weighted[0])
        self.assertGreater(lcb_weighted[0], lcb_weighted[2])
        self.assertAlmostEqual(float(np.sum(selection_weighted)), 1.0)
        self.assertGreater(selection_weighted[1], selection_weighted[0])
        self.assertGreater(selection_weighted[0], selection_weighted[2])
        self.assertAlmostEqual(float(np.sum(selection_lcb_weighted)), 1.0)
        self.assertGreater(selection_lcb_weighted[0], selection_lcb_weighted[1])
        self.assertGreater(selection_lcb_weighted[1], selection_lcb_weighted[2])

    def test_balanced_accuracy_lcb_selection_penalizes_unstable_candidates(self):
        self.assertIn(
            "balanced_accuracy_lcb",
            cross_subject.CROSS_SUBJECT_SELECTION_METRIC_CHOICES,
        )

        def row(candidate_index, validation_participant, balanced_accuracy):
            return {
                "selection_mode": "nested_loso",
                "selection_metric": "balanced_accuracy_lcb",
                "outer_test_participant": 99,
                "inner_validation_participant": validation_participant,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "top2_accuracy": balanced_accuracy,
                "top3_accuracy": balanced_accuracy,
                "mean_true_label_rank": 1.0,
                "chance_mean_rank": 1.5,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.175,
                "window_size_s": 0.1,
                "window_start_s": 0.125,
                "window_stop_s": 0.225,
                "feature_mode": "sensor_flat",
                "normalization": "subject_baseline_whiten",
                "alignment": "none",
                "classifier": "multinomial-logistic",
                "classifier_param": 1.0,
                "components_pca": 128,
                "max_trials_per_class_per_participant": "",
                "label_shuffle_control": False,
                "label_shuffle_seed": "",
            }

        ranked = cross_subject._rank_nested_candidates(  # pylint: disable=protected-access
            [
                row(1, 1, 0.70),
                row(1, 2, 0.70),
                row(1, 3, 0.40),
                row(2, 1, 0.56),
                row(2, 2, 0.56),
                row(2, 3, 0.56),
            ],
            selection_metric="balanced_accuracy_lcb",
        )

        self.assertEqual(ranked[0]["selected_candidate_index"], 2)
        self.assertGreater(
            ranked[0]["selected_inner_selection_ranking_score"],
            ranked[1]["selected_inner_selection_ranking_score"],
        )

    def test_rank_softmax_temperature_modes_soften_rank_mass(self):
        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)

        sharp = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )[0]
        t2 = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax_t2",
        )[0]
        t3 = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax_t3_inner_balanced",
        )[0]

        np.testing.assert_allclose(np.sum(sharp), 1.0)
        np.testing.assert_allclose(np.sum(t2), 1.0)
        np.testing.assert_allclose(np.sum(t3), 1.0)
        self.assertGreater(sharp[0], t2[0])
        self.assertGreater(t2[0], t3[0])
        self.assertGreater(t2[1], sharp[1])
        self.assertGreater(t3[2], t2[2])
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank_softmax_t2_inner_balanced"
            ),
            "rank_softmax_t2",
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank_softmax_t3_inner_confusion"
            ),
            "rank_softmax_t3",
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank-softmax-t2-balanced-quota"
            ),
            "rank_softmax_t2",
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank-softmax-t2-test-prior-balance"
            ),
            "rank_softmax_t2",
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank-softmax-inner-balanced-test-prior-balance"
            ),
            "rank_softmax",
        )

    def test_rank_margin_blend_softens_only_low_margin_trials(self):
        scores = np.asarray(
            [
                [1.00, 0.99, 0.00],
                [3.00, 0.00, -1.00],
            ],
            dtype=float,
        )

        sharp = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )
        blended = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_margin_blend",
        )

        self.assertEqual(blended.shape, scores.shape)
        np.testing.assert_allclose(np.sum(blended, axis=1), np.ones(scores.shape[0]))
        self.assertGreater(
            blended[0, 1],
            sharp[0, 1],
            "low-margin rows should give the runner-up more mass than rank_softmax",
        )
        self.assertGreater(blended[1, 0], blended[0, 0])

    def test_test_class_prior_balance_equalizes_unlabeled_class_mass(self):
        probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.70, 0.30],
                [0.60, 0.40],
            ],
            dtype=float,
        )

        adjusted, target_mass, iterations, status = cross_subject._test_class_prior_balanced_probabilities(  # pylint: disable=protected-access
            probabilities
        )

        self.assertIn(status, {"applied", "applied_max_iterations"})
        self.assertGreater(iterations, 0)
        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(4))
        np.testing.assert_allclose(np.sum(adjusted, axis=0), target_mass, atol=1e-5)
        np.testing.assert_allclose(target_mass, np.asarray([2.0, 2.0]))

        metadata = cross_subject._test_class_prior_balance_metadata(  # pylint: disable=protected-access
            probabilities,
            np.asarray([0, 1], dtype=int),
            "rank_softmax_test_prior_balance",
        )
        applied = cross_subject._apply_test_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )
        np.testing.assert_allclose(applied, adjusted)
        self.assertEqual(metadata["mode"], "rank_softmax_test_prior_balance")
        self.assertEqual(metadata["class_order"].tolist(), [0, 1])

    def test_rank_borda_score_normalization_uses_linear_rank_weights(self):
        scores = np.asarray(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, np.nan, 3.0, 2.0],
            ],
            dtype=float,
        )

        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_borda",
        )
        balanced = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_borda_inner_balanced",
        )

        np.testing.assert_allclose(probabilities[0], [0.4, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(
            probabilities[1], [1.0 / 6.0, 0.0, 0.5, 1.0 / 3.0]
        )
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
        np.testing.assert_allclose(balanced, probabilities)

    def test_rank_borda_score_normalization_uses_full_rank_order(self):
        scores = np.asarray(
            [
                [4.0, 3.0, 1.0, 0.0],
                [0.0, 2.0, 5.0, 1.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=float,
        )

        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_borda",
        )

        np.testing.assert_allclose(
            probabilities[0],
            np.asarray([4.0, 3.0, 2.0, 1.0]) / 10.0,
        )
        np.testing.assert_allclose(
            probabilities[1],
            np.asarray([1.0, 3.0, 4.0, 2.0]) / 10.0,
        )
        np.testing.assert_allclose(probabilities[2], np.full(4, 0.25))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(3))

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

    def test_nested_ensemble_can_diversify_by_window_feature_and_classifier(self):
        configs = (
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, feature_mode="sensor_flat", normalization="none", classifier="multiclass-svm", classifier_param=0.5),
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, feature_mode="sensor_flat", normalization="none", classifier="multiclass-svm", classifier_param=1.0),
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, feature_mode="sensor_mean_logpower", normalization="none", classifier="multiclass-svm", classifier_param=0.5),
        )

        def inner_row(candidate_index, balanced_accuracy, feature_mode, classifier_param):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10,
                "window_size_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "feature_mode": feature_mode,
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": classifier_param,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
            }

        inner_rows = [
            inner_row(1, 0.90, "sensor_flat", 0.5),
            inner_row(2, 0.85, "sensor_flat", 1.0),
            inner_row(3, 0.80, "sensor_mean_logpower", 0.5),
        ]

        feature_diverse, _feature_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(feature_diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(feature_diverse["selection_ensemble_diversity"], "window_feature_classifier")
        self.assertEqual(feature_diverse["selected_ensemble_feature_mode_counts"], "sensor_flat:1;sensor_mean_logpower:1")

    def test_nested_ensemble_can_diversify_by_score_calibration(self):
        configs = (
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                score_calibration="none",
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                score_calibration="inner_class_affine",
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_mean_logpower",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                score_calibration="none",
            ),
        )

        def inner_row(candidate_index, balanced_accuracy, feature_mode, score_calibration):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10,
                "window_size_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "feature_mode": feature_mode,
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
                "score_calibration": score_calibration,
            }

        inner_rows = [
            inner_row(1, 0.90, "sensor_flat", "none"),
            inner_row(2, 0.85, "sensor_flat", "inner_class_affine"),
            inner_row(3, 0.80, "sensor_mean_logpower", "none"),
        ]

        feature_diverse, _feature_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )
        calibration_diverse, _calibration_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_score_calibration",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(feature_diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(calibration_diverse["selected_candidate_indices"], "1;2")
        self.assertEqual(calibration_diverse["selection_ensemble_diversity"], "window_feature_classifier_score_calibration")
        self.assertEqual(calibration_diverse["selected_ensemble_score_calibration_counts"], "inner_class_affine:1;none:1")

    def test_nested_ensemble_can_diversify_by_sample_weighting_and_score_calibration(self):
        configs = (
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="none",
                score_calibration="none",
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="subject_class_balanced",
                score_calibration="none",
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="none",
                score_calibration="inner_class_affine",
            ),
        )

        def inner_row(candidate_index, balanced_accuracy, sample_weighting, score_calibration):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10,
                "window_size_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "feature_mode": "sensor_flat",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
                "sample_weighting": sample_weighting,
                "score_calibration": score_calibration,
            }

        inner_rows = [
            inner_row(1, 0.90, "none", "none"),
            inner_row(2, 0.85, "subject_class_balanced", "none"),
            inner_row(3, 0.80, "none", "inner_class_affine"),
        ]

        calibration_diverse, _calibration_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_score_calibration",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )
        sample_weight_diverse, _sample_weight_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_sample_weighting_score_calibration",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(calibration_diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(sample_weight_diverse["selected_candidate_indices"], "1;2")
        self.assertEqual(
            sample_weight_diverse["selection_ensemble_diversity"],
            "window_feature_classifier_sample_weighting_score_calibration",
        )
        self.assertEqual(sample_weight_diverse["selected_ensemble_sample_weighting_counts"], "none:1;subject_class_balanced:1")

    def test_nested_ensemble_can_diversify_by_pca_components(self):
        configs = (
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="none",
                score_calibration="none",
                components_pca=64,
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="none",
                score_calibration="none",
                components_pca=128,
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.5,
                sample_weighting="subject_class_balanced",
                score_calibration="none",
                components_pca=64,
            ),
        )

        def inner_row(candidate_index, balanced_accuracy, components_pca, sample_weighting):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10,
                "window_size_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "feature_mode": "sensor_flat",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": components_pca,
                "max_trials_per_class_per_participant": "",
                "sample_weighting": sample_weighting,
                "score_calibration": "none",
            }

        inner_rows = [
            inner_row(1, 0.90, 64, "none"),
            inner_row(2, 0.85, 128, "none"),
            inner_row(3, 0.80, 64, "subject_class_balanced"),
        ]

        sample_weight_diverse, _sample_weight_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_sample_weighting_score_calibration",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )
        pca_diverse, _pca_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_sample_weighting_score_calibration_pca",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(sample_weight_diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(pca_diverse["selected_candidate_indices"], "1;2")
        self.assertEqual(
            pca_diverse["selection_ensemble_diversity"],
            "window_feature_classifier_sample_weighting_score_calibration_pca",
        )
        self.assertEqual(pca_diverse["selected_ensemble_components_pca_counts"], "128:1;64:1")

    def test_nested_ensemble_can_diversify_by_classifier_param(self):
        configs = (
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.1,
                sample_weighting="none",
                score_calibration="none",
                components_pca=64,
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=1.0,
                sample_weighting="none",
                score_calibration="none",
                components_pca=64,
            ),
            CrossSubjectStimulusConfig(
                window_center=0.10,
                window_size=0.1,
                feature_mode="sensor_flat",
                normalization="none",
                classifier="multiclass-svm",
                classifier_param=0.1,
                sample_weighting="subject_class_balanced",
                score_calibration="none",
                components_pca=64,
            ),
        )

        def inner_row(candidate_index, balanced_accuracy, classifier_param, sample_weighting):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10,
                "window_size_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "feature_mode": "sensor_flat",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": classifier_param,
                "components_pca": 64,
                "max_trials_per_class_per_participant": "",
                "sample_weighting": sample_weighting,
                "score_calibration": "none",
            }

        inner_rows = [
            inner_row(1, 0.90, 0.1, "none"),
            inner_row(2, 0.85, 1.0, "none"),
            inner_row(3, 0.80, 0.1, "subject_class_balanced"),
        ]

        pca_diverse, _pca_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_sample_weighting_score_calibration_pca",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )
        param_diverse, _param_diverse_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=2,
            selection_ensemble_diversity="window_feature_classifier_param_sample_weighting_score_calibration_pca",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            candidate_configs=configs,
        )

        self.assertEqual(pca_diverse["selected_candidate_indices"], "1;3")
        self.assertEqual(param_diverse["selected_candidate_indices"], "1;2")
        self.assertEqual(
            param_diverse["selection_ensemble_diversity"],
            "window_feature_classifier_param_sample_weighting_score_calibration_pca",
        )
        self.assertEqual(param_diverse["selected_ensemble_classifier_param_counts"], "0.1:1;1.0:1")

    def test_nested_selection_metric_can_use_topk_signal(self):
        configs = (
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
            CrossSubjectStimulusConfig(window_center=0.20, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
        )

        def inner_row(candidate_index, balanced_accuracy, top2_accuracy, top3_accuracy, mean_true_label_rank):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "top2_accuracy": top2_accuracy,
                "top3_accuracy": top3_accuracy,
                "mean_true_label_rank": mean_true_label_rank,
                "chance_mean_rank": 1.5,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10 * candidate_index,
                "window_size_s": 0.1,
                "window_start_s": 0.10 * candidate_index - 0.05,
                "window_stop_s": 0.10 * candidate_index + 0.05,
                "feature_mode": "sensor_mean",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
            }

        inner_rows = [
            inner_row(1, 0.70, 0.72, 0.74, 1.30),
            inner_row(2, 0.64, 0.95, 0.99, 1.05),
        ]

        balanced, _balanced_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_accuracy",
            candidate_configs=configs,
        )
        rank_aware, _rank_aware_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2",
            candidate_configs=configs,
        )
        topk_aware, _topk_aware_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2_top3",
            candidate_configs=configs,
        )

        self.assertEqual(balanced["selected_candidate_index"], 1)
        self.assertEqual(rank_aware["selected_candidate_index"], 2)
        self.assertEqual(rank_aware["selection_metric"], "balanced_top2")
        self.assertIn("2:", rank_aware["selected_ensemble_inner_selection_score_means"])
        self.assertGreater(rank_aware["selected_inner_selection_score_mean"], rank_aware["selected_inner_balanced_accuracy_mean"])
        self.assertEqual(rank_aware["selected_inner_selection_score_sem"], 0.0)
        self.assertEqual(topk_aware["selected_candidate_index"], 2)
        self.assertEqual(topk_aware["selection_metric"], "balanced_top2_top3")
        self.assertGreater(topk_aware["selected_inner_selection_score_mean"], rank_aware["selected_inner_selection_score_mean"])

    def test_nested_selection_metric_can_use_topk_and_rank_signal(self):
        configs = (
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
            CrossSubjectStimulusConfig(window_center=0.20, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
        )

        def inner_row(candidate_index, balanced_accuracy, top2_accuracy, top3_accuracy, mean_true_label_rank):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "top2_accuracy": top2_accuracy,
                "top3_accuracy": top3_accuracy,
                "mean_true_label_rank": mean_true_label_rank,
                "chance_mean_rank": 8.5,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10 * candidate_index,
                "window_size_s": 0.1,
                "window_start_s": 0.10 * candidate_index - 0.05,
                "window_stop_s": 0.10 * candidate_index + 0.05,
                "feature_mode": "sensor_mean",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
            }

        inner_rows = [
            inner_row(1, 0.75, 0.80, 0.90, 5.5),
            inner_row(2, 0.70, 0.74, 0.78, 2.0),
        ]

        topk_aware, _topk_aware_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2_top3",
            candidate_configs=configs,
        )
        rank_composite, _rank_composite_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2_top3_rank",
            candidate_configs=configs,
        )

        self.assertEqual(topk_aware["selected_candidate_index"], 1)
        self.assertEqual(rank_composite["selected_candidate_index"], 2)
        self.assertEqual(rank_composite["selection_metric"], "balanced_top2_top3_rank")
        self.assertGreater(rank_composite["selected_inner_rank_score_mean"], 0.8)

    def test_nested_selection_metric_can_use_lcb_topk_rank_signal(self):
        configs = (
            CrossSubjectStimulusConfig(window_center=0.10, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
            CrossSubjectStimulusConfig(window_center=0.20, window_size=0.1, normalization="none", classifier="multiclass-svm", classifier_param=0.5),
        )

        def inner_row(candidate_index, balanced_accuracy, top2_accuracy, top3_accuracy, mean_true_label_rank):
            return {
                "outer_test_participant": 4,
                "candidate_index": candidate_index,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": balanced_accuracy,
                "top2_accuracy": top2_accuracy,
                "top3_accuracy": top3_accuracy,
                "mean_true_label_rank": mean_true_label_rank,
                "chance_mean_rank": 8.5,
                "train_participants": "1,2,3",
                "n_train_participants": 3,
                "window_center_s": 0.10 * candidate_index,
                "window_size_s": 0.1,
                "window_start_s": 0.10 * candidate_index - 0.05,
                "window_stop_s": 0.10 * candidate_index + 0.05,
                "feature_mode": "sensor_mean",
                "normalization": "none",
                "alignment": "none",
                "classifier": "multiclass-svm",
                "classifier_param": 0.5,
                "components_pca": float("inf"),
                "max_trials_per_class_per_participant": "",
            }

        inner_rows = [
            inner_row(1, 1.00, 1.00, 1.00, 1.0),
            inner_row(1, 0.55, 0.57, 0.59, 6.5),
            inner_row(2, 0.70, 0.72, 0.74, 3.0),
            inner_row(2, 0.69, 0.71, 0.73, 3.1),
        ]

        rank_composite, _rank_composite_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2_top3_rank",
            candidate_configs=configs,
        )
        lcb_composite, _lcb_composite_rows = cross_subject._select_nested_candidate_ensemble(  # pylint: disable=protected-access
            inner_rows,
            selection_ensemble_size=1,
            selection_ensemble_diversity="none",
            selection_ensemble_score_normalization="row_z_softmax",
            selection_ensemble_weighting="uniform",
            selection_ensemble_temperature=0.02,
            selection_metric="balanced_top2_top3_rank_lcb",
            candidate_configs=configs,
        )

        self.assertEqual(rank_composite["selected_candidate_index"], 1)
        self.assertEqual(lcb_composite["selected_candidate_index"], 2)
        self.assertEqual(lcb_composite["selection_metric"], "balanced_top2_top3_rank_lcb")
        self.assertLess(
            lcb_composite["selected_inner_selection_ranking_score"],
            lcb_composite["selected_inner_selection_score_mean"],
        )

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

    def test_rank_z_blend_score_normalization_keeps_rank_and_margin_signal(self):
        scores = np.asarray([[1.00, 0.99, -3.00], [10.0, -1.0, 0.0]], dtype=float)

        rank_only = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )
        row_z = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="row_z_softmax",
        )
        blended = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank-z-blend",
        )

        np.testing.assert_allclose(np.sum(blended, axis=1), np.ones(2))
        self.assertEqual(int(np.argmax(blended[0])), 0)
        self.assertEqual(int(np.argmax(blended[1])), 0)
        self.assertGreater(blended[0, 1], blended[0, 2])
        self.assertGreater(blended[1, 2], blended[1, 1])
        self.assertLess(blended[0, 0] - blended[0, 1], rank_only[0, 0] - rank_only[0, 1])
        self.assertGreater(blended[1, 0] - blended[1, 2], rank_only[1, 0] - rank_only[1, 2])
        np.testing.assert_allclose(blended, 0.5 * rank_only + 0.5 * row_z)

    def test_inner_confusion_correction_uses_source_validation_confusions(self):
        class_order = np.asarray([0, 1], dtype=int)
        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:1;1002:9;2001:9;2002:1"
                )
            }
        ]
        probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=float)

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            selected_rows,
            class_order,
            np.ones(1),
            "rank_softmax_inner_confusion",
        )
        adjusted = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        self.assertEqual(metadata["mode"], "rank_softmax_inner_confusion")
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])
        self.assertGreater(adjusted[1, 0], probabilities[1, 0])
        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(2))

    def test_soft_inner_confusion_correction_uses_conservative_blend(self):
        class_order = np.asarray([0, 1], dtype=int)
        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:1;1002:9;2001:9;2002:1"
                )
            }
        ]
        probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=float)

        hard_metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            selected_rows,
            class_order,
            np.ones(1),
            "rank_z_blend_inner_confusion",
        )
        soft_metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            selected_rows,
            class_order,
            np.ones(1),
            "rank_z_blend_inner_confusion_soft",
        )
        hard_adjusted = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            probabilities,
            hard_metadata,
        )
        soft_adjusted = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            probabilities,
            soft_metadata,
        )

        self.assertEqual(soft_metadata["mode"], "rank_z_blend_inner_confusion_soft")
        self.assertAlmostEqual(soft_metadata["blend"], 0.35)
        self.assertGreater(hard_metadata["blend"], soft_metadata["blend"])
        np.testing.assert_allclose(
            soft_adjusted, 0.65 * probabilities + 0.35 * hard_adjusted
        )
        np.testing.assert_allclose(np.sum(soft_adjusted, axis=1), np.ones(2))

    def test_confusion_counter_round_trips_pair_counts(self):
        counter = Counter({(1, 2): 3, (2, 1): 1})

        parsed = cross_subject._parse_confusion_counter(  # pylint: disable=protected-access
            cross_subject._format_confusion_counter(counter)  # pylint: disable=protected-access
        )

        self.assertEqual(parsed[(1, 2)], 3.0)
        self.assertEqual(parsed[(2, 1)], 1.0)

    def test_inner_confusion_normalization_maps_to_base_rank_mode(self):
        self.assertIn(
            "rank_reciprocal_inner_confusion",
            cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES,
        )
        self.assertIn(
            "rank_z_blend_inner_confusion_soft",
            cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES,
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank_reciprocal_inner_confusion"
            ),
            "rank_reciprocal",
        )
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(  # pylint: disable=protected-access
                "rank_z_blend_inner_confusion_soft"
            ),
            "rank_z_blend",
        )

    def test_inner_confusion_correction_uses_source_validation_map(self):
        selected_rows = (
            {
                "selected_inner_confusion_counts": "1>1:1;1>2:9;2>1:9;2>2:1",
            },
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.asarray([0, 1], dtype=int),
            np.asarray([1.0], dtype=float),
            "rank_softmax_inner_confusion",
        )
        corrected = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            np.asarray([[0.8, 0.2]], dtype=float),
            metadata,
        )

        self.assertEqual(metadata["mode"], "rank_softmax_inner_confusion")
        self.assertGreater(corrected[0, 1], corrected[0, 0])
        self.assertAlmostEqual(float(np.sum(corrected[0])), 1.0)

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
