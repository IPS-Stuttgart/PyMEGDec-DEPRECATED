import unittest
from unittest.mock import patch

import numpy as np
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig, load_participant_stimulus_features, make_cross_subject_candidate_configs
from tests.matlab_fixtures import loadmat_side_effect, mat_data_from_trials


class TestStimulusCrossSubjectNext(unittest.TestCase):
    def test_extended_feature_modes_are_exported(self):
        self.assertIn("sensor_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_mean_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_bandpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_cov_tangent", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid", cross_subject.FEATURE_MODES)

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


if __name__ == "__main__":
    unittest.main()
