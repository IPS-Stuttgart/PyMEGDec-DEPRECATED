import os
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec.preprocessing import downsample_data, extract_windows
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig, load_participant_stimulus_features
from tests.matlab_fixtures import cell_array


def _data(trials, times):
    return {
        "trial": cell_array(trials),
        "time": cell_array(times),
    }


def _mat_data(labels, trials, times):
    return {
        "trial": cell_array([np.asarray(trial, dtype=float) for trial in trials]),
        "time": cell_array([np.asarray(time, dtype=float) for time in times]),
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


class TestTimeAxisHandling(unittest.TestCase):
    def test_extract_windows_uses_each_trial_time_vector(self):
        first_time = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2]])
        second_time = np.array([[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]])
        first_trial = np.array([[0, 1, 2, 3, 4]], dtype=float)
        second_trial = np.array([[10, 11, 12, 13, 14, 15]], dtype=float)
        data = _data([first_trial, second_trial], [first_time, second_time])

        stimuli, null = extract_windows(data, (-0.1, 0.1), (np.nan, np.nan))

        np.testing.assert_array_equal(stimuli[0].ravel(), [1, 2, 3])
        np.testing.assert_array_equal(stimuli[1].ravel(), [12, 13, 14])
        self.assertEqual(null, [])

    def test_downsample_uses_each_trial_time_support_without_extrapolation(self):
        first_time = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        second_time = np.array([[0.125, 0.375, 0.625, 0.875, 1.125]])
        first_trial = np.array([[0, 1, 0, -1, 0]], dtype=float)
        second_trial = np.array([[10, 11, 10, 9, 10]], dtype=float)
        data = _data([first_trial, second_trial], [first_time, second_time])

        downsampled = downsample_data(data, 2)

        np.testing.assert_allclose(downsampled["time"][0][0][0], [[0.0, 0.5, 1.0]])
        np.testing.assert_allclose(downsampled["time"][0][0][1], [[0.125, 0.625, 1.125]])
        self.assertEqual(downsampled["trial"][0][0][0].shape, (1, 3))
        self.assertEqual(downsampled["trial"][0][0][1].shape, (1, 3))

    def test_cross_subject_features_use_each_trial_time_vector(self):
        first_time = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
        second_time = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
        first_trial = np.array([[0, 1, 2, 3, 4]], dtype=float)
        second_trial = np.array([[10, 11, 12, 13, 14, 15]], dtype=float)
        data_by_participant = {1: _mat_data([1, 2], [first_trial, second_trial], [first_time, second_time])}
        config = CrossSubjectStimulusConfig(
            window_center=0.0,
            window_size=0.2,
            feature_mode="sensor_mean",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        np.testing.assert_allclose(feature_set.features.ravel(), [2.0, 13.0])
        self.assertEqual(feature_set.n_window_samples, 3)

    def test_cross_subject_legacy_rejects_inconsistent_window_sample_counts(self):
        """Avoid silently stacking features from physically different sample counts."""

        from pymegdec import _stimulus_cross_subject_legacy as cross_subject

        first_time = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
        second_time = np.array([-0.2, 0.0, 0.2], dtype=float)
        first_trial = np.array([[0, 1, 2, 3, 4]], dtype=float)
        second_trial = np.array([[10, 11, 12]], dtype=float)
        data = {
            "trial": cell_array([first_trial, second_trial]),
            "time": cell_array([first_time, second_time]),
        }

        with self.assertRaisesRegex(ValueError, "time_window for trial 1 contains 1 samples; expected 3"):
            cross_subject._extract_window_features(
                data, (-0.1, 0.1), feature_mode="sensor_mean", trial_indices=None
            )

    def test_legacy_cross_subject_extractors_use_each_trial_time_vector_without_public_shim(self):
        """Guard the implementation path, not only the public facade import."""

        code = r'''
import numpy as np
from pymegdec import _stimulus_cross_subject_legacy as cross_subject


def cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value
    outer = np.empty((1,), dtype=object)
    outer[0] = inner
    return outer


first_time = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
second_time = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
first_trial = np.array([[0, 1, 2, 3, 4]], dtype=float)
second_trial = np.array([[10, 11, 12, 13, 14, 15]], dtype=float)
data = {
    "trial": cell_array([first_trial, second_trial]),
    "time": cell_array([first_time, second_time]),
}

features, n_window_samples = cross_subject._extract_window_features(
    data, (-0.1, 0.1), feature_mode="sensor_mean", trial_indices=None
)
np.testing.assert_allclose(features.ravel(), [2.0, 13.0])
assert n_window_samples == 3

mean, _std, n_baseline_samples = cross_subject._baseline_channel_statistics(
    data, (-0.1, 0.1), trial_indices=None
)
np.testing.assert_allclose(mean, [7.5])
assert n_baseline_samples == 3
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
