import unittest
from unittest.mock import patch

import numpy as np
from pymegdec import _stimulus_cross_subject_next as next_hooks
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig, load_participant_stimulus_features, make_cross_subject_candidate_configs
from tests.matlab_fixtures import loadmat_side_effect, mat_data_from_trials


class TestStimulusCrossSubjectNext(unittest.TestCase):
    def test_soft_guarded_inner_confusion_score_normalizations_are_exported(self):
        mode = "rank_softmax_t2_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t2",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:2;1002:1;2002:3"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertLess(metadata["blend"], 1.0)

    def test_margin_gated_inner_confusion_score_normalization_is_exported(self):
        mode = "rank_z_blend_inner_confusion_margin_soft"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_z_blend",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:3;1002:2;2002:4"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["margin_gated"])
        self.assertEqual(metadata["margin_quantile"], 0.5)
        self.assertLess(metadata["blend"], 1.0)

    def test_rank_margin_blend_inner_confusion_soft_is_exported_without_margin_gate(self):
        mode = "rank_margin_blend_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_margin_blend",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertFalse(metadata["margin_gated"])
        self.assertLess(metadata["blend"], 1.0)

    def test_margin_gated_inner_confusion_leaves_high_margin_rows_unchanged(self):
        probabilities = np.asarray([[0.51, 0.49], [0.95, 0.05]], dtype=float)
        metadata = {
            "mode": "rank_softmax_inner_confusion_margin_soft",
            "true_given_predicted": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            "blend": 1.0,
            "margin_gated": True,
            "margin_quantile": 0.5,
        }

        adjusted = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(adjusted[0], np.asarray([0.49, 0.51]))
        np.testing.assert_allclose(adjusted[1], probabilities[1])
        self.assertAlmostEqual(metadata["margin_threshold"], 0.46)

    def test_extended_feature_modes_are_exported(self):
        self.assertIn("sensor_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_mean_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_pyramid_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_bandpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_cov_tangent", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_delta_logpower", cross_subject.FEATURE_MODES)

    def test_sensor_mean_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 4))
        np.testing.assert_allclose(feature_set.features[0, :2], np.asarray([2.0, 4.0]))
        np.testing.assert_allclose(feature_set.features[1, :2], np.asarray([3.0, 6.0]))
        self.assertTrue(np.all(np.isfinite(feature_set.features[:, 2:])))

    def test_sensor_flat_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_flat_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(feature_set.features[0, :4], np.asarray([1.0, 2.0, 3.0, 6.0]))
        np.testing.assert_allclose(feature_set.features[0, 4:], np.log(np.asarray([5.0, 20.0]) + 1e-12))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 6.0], [0.0, 0.0, 2.0, 5.0, 9.0]],
            [[0.0, 0.0, 2.0, 4.0, 7.0], [0.0, 0.0, 4.0, 7.0, 11.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 10))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 2.0, 3.0, 3.0, 4.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_delta_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.1, 0.2, 0.3, 1.0, 3.0, 6.0], [0.0, 0.4, 0.5, 0.6, 2.0, 5.0, 9.0]],
            [[0.0, 0.2, 0.3, 0.4, 2.0, 4.0, 7.0], [0.0, 0.5, 0.6, 0.7, 4.0, 7.0, 11.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_delta",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 10))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 10))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 10))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_time_pyramid_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 28))
        np.testing.assert_allclose(
            feature_set.features[0, :6],
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_delta_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_time_pyramid_delta",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 26))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 26))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 26))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_logpower_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_logpower",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_cov_tangent_feature_width(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 6.0, 8.0], [0.0, 0.0, 3.0, 1.0, 0.5]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 4.0, 8.0, 9.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_cov_tangent",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_time_pyramid_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 14))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ]),
        )

    def test_sensor_time_pyramid_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 22))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                4.0, 4.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ]),
        )

    def test_sensor_time_pyramid_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 16))
        np.testing.assert_allclose(
            feature_set.features[0, :14],
            np.asarray([
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ]),
        )
        np.testing.assert_allclose(feature_set.features[0, 14:], np.log(np.asarray([21.0, 30.0]) + 1e-12))

    def test_candidate_grid_expands_next_knobs(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            alignments=("none", "train_class_procrustes"),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            sample_weightings=("none", "subject_class_balanced"),
            score_calibrations=("none", "inner_class_bias"),
            alignment_alphas=(0.25, 1.0),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 16)
        self.assertEqual({config.sample_weighting for config in configs}, {"none", "subject_class_balanced"})
        self.assertEqual({config.score_calibration for config in configs}, {"none", "inner_class_bias"})
        self.assertEqual({config.alignment_alpha for config in configs}, {0.25, 1.0})

    def test_candidate_grid_expands_window_sizes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.150, 0.175),
            window_sizes=(0.125, 0.150),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            alignments=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(config.window_center, config.window_size) for config in configs},
            {
                (0.150, 0.125),
                (0.150, 0.150),
                (0.175, 0.125),
                (0.175, 0.150),
            },
        )

    def test_candidate_grid_accepts_inner_class_affine_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_class_affine",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].score_calibration, "inner_class_affine")

    def test_candidate_grid_accepts_rank_bias_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_rank_bias", "train_rank_bias"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_rank_bias", "train_rank_bias"},
        )

    def test_candidate_grid_accepts_inner_probability_map_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_probability_map", "inner_rank_probability_map"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_probability_map", "inner_rank_probability_map"},
        )

    def test_candidate_grid_accepts_inner_confusion_blend_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_confusion_blend",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].score_calibration, "inner_confusion_blend")

    def test_candidate_grid_accepts_inner_margin_confusion_blend_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_margin_confusion_blend",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(
            configs[0].score_calibration,
            "inner_margin_confusion_blend",
        )

    def test_candidate_grid_accepts_rank_confusion_blend_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=(
                "inner_rank_confusion_blend",
                "inner_rank_margin_confusion_blend",
                "inner_rank_confusion_blend_guarded",
            ),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 3)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {
                "inner_rank_confusion_blend",
                "inner_rank_margin_confusion_blend",
                "inner_rank_confusion_blend_guarded",
            },
        )

    def test_candidate_grid_accepts_guarded_inner_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=(
                "inner_class_bias_guarded",
                "inner_probability_map_guarded",
            ),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_class_bias_guarded", "inner_probability_map_guarded"},
        )

    def test_candidate_grid_accepts_train_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("train_class_bias", "train_class_affine"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"train_class_bias", "train_class_affine"},
        )

    def test_inner_class_bias_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_class_bias",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_class_bias")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        np.testing.assert_allclose(metadata["scale"], np.ones(2))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_class_affine_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_class_affine",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_class_affine")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        self.assertTrue(np.all(metadata["scale"] > 0.0))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_probability_map_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_probability_map",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_probability_map")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["probability_map"].shape, (2, 2))
        self.assertTrue(np.all(metadata["probability_map"] >= 0.0))
        np.testing.assert_allclose(np.sum(metadata["probability_map"], axis=1), np.ones(2))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_confusion_blend_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_confusion_blend",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_confusion_blend")
        self.assertEqual(metadata["calibration_source"], "inner_scores")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["confusion_matrix"].shape, (2, 2))
        np.testing.assert_allclose(np.sum(metadata["confusion_matrix"], axis=1), np.ones(2))
        self.assertGreaterEqual(metadata["blend_alpha"], 0.0)
        self.assertLessEqual(metadata["blend_alpha"], 1.0)
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_train_class_bias_score_calibration_fits_source_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="train_class_bias",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "train_class_bias")
        self.assertEqual(metadata["calibration_source"], "train_scores")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        np.testing.assert_allclose(metadata["scale"], np.ones(2))
        self.assertTrue(np.isfinite(metadata["source_balanced_accuracy"]))

    def test_inner_class_affine_score_calibration_applies_scale_and_bias(self):
        scores = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_class_affine",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.5, -0.5]),
                "scale": np.asarray([2.0, 0.25]),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_allclose(calibrated, np.asarray([[2.5, 0.0], [6.5, 0.5]]))

    def test_rank_bias_score_calibration_applies_bias_to_rank_scores(self):
        scores = np.asarray([[10.0, 1.0], [0.0, 3.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_bias",
                "score_space": "rank",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.0, 2.0]),
                "scale": np.ones(2),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_allclose(calibrated, np.asarray([[0.0, 1.0], [-1.0, 2.0]]))

    def test_inner_probability_map_score_calibration_applies_probability_map(self):
        scores = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_probability_map",
                "classes": np.asarray([0, 1]),
                "probability_map": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_array_equal(np.argmax(calibrated, axis=1), np.asarray([1, 0]))
        np.testing.assert_allclose(np.sum(np.exp(calibrated), axis=1), np.ones(2))

    def test_rank_probability_map_score_calibration_uses_rank_space(self):
        scores = np.asarray([[100.0, 99.0, 0.0], [1.0, 0.0, 100.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_probability_map",
                "score_space": "rank",
                "classes": np.asarray([0, 1, 2]),
                "probability_map": np.eye(3, dtype=float),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1, 2]),
            fitted_model,
        )

        rank_scores = np.asarray(
            [[0.0, -1.0, -2.0], [-1.0, -2.0, 0.0]], dtype=float
        )
        centered = rank_scores - np.mean(rank_scores, axis=1, keepdims=True)
        centered /= np.std(centered, axis=1, keepdims=True)
        probabilities = np.exp(centered - np.max(centered, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        expected = np.log(probabilities)

        np.testing.assert_array_equal(classes, np.asarray([0, 1, 2]))
        np.testing.assert_allclose(calibrated, expected)

    def test_inner_confusion_blend_score_calibration_applies_confusion_matrix(self):
        scores = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_confusion_blend",
                "classes": np.asarray([0, 1]),
                "confusion_matrix": np.asarray(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=float
                ),
                "blend_alpha": 1.0,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        self.assertGreater(calibrated[0, 1], calibrated[0, 0])
        self.assertGreater(calibrated[1, 0], calibrated[1, 1])

    def test_rank_confusion_blend_score_calibration_uses_rank_space(self):
        scores = np.asarray([[100.0, 99.0, 0.0], [1.0, 0.0, 100.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_confusion_blend",
                "score_space": "rank",
                "classes": np.asarray([0, 1, 2]),
                "confusion_matrix": np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                ),
                "blend_alpha": 1.0,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1, 2]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(np.argmax(calibrated, axis=1), np.asarray([1, 2]))

    def test_inner_margin_confusion_blend_only_reranks_low_margin_trials(self):
        scores = np.asarray(
            [[0.99, 0.01], [0.51, 0.49], [0.01, 0.99]],
            dtype=float,
        )
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_margin_confusion_blend",
                "classes": np.asarray([0, 1]),
                "confusion_matrix": np.asarray(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=float
                ),
                "blend_alpha": 1.0,
                "margin_threshold": 0.1,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_array_equal(
            np.argmax(calibrated, axis=1),
            np.asarray([0, 1, 1]),
        )

    def test_guarded_inner_score_calibration_skips_without_inner_gain(self):
        metadata = next_hooks._guard_inner_score_calibration_metadata(  # pylint: disable=protected-access
            {
                "mode": "inner_class_bias_guarded",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.5, -0.5]),
                "scale": np.ones(2),
                "inner_balanced_accuracy": 0.5,
                "calibration_source": "inner_scores",
            },
            0.5,
            guarded=True,
        )

        self.assertEqual(metadata["status"], "skipped_no_inner_gain")
        self.assertEqual(metadata["inner_uncalibrated_balanced_accuracy"], 0.5)
        self.assertNotIn("bias", metadata)
        self.assertFalse(next_hooks._has_active_score_calibration_metadata(metadata))  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
