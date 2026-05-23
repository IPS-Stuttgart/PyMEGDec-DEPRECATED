import re
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig, load_participant_stimulus_features, make_cross_subject_candidate_configs
from tests.matlab_fixtures import cell_array


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
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
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
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_cov_tangent",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_time_pyramid_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: _mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
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


if __name__ == "__main__":
    unittest.main()
