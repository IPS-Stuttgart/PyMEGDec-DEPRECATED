import re
import unittest
from unittest.mock import patch

import numpy as np

from pymegdec.stimulus_covariance_features import (
    CovarianceStimulusConfig,
    evaluate_nested_covariance_stimulus,
    load_participant_covariance_features,
    make_covariance_candidate_configs,
)
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


def _covariance_trials(labels, *, scale, time):
    base_patterns = {
        1: np.asarray([[1.0, -1.0, 1.0, -1.0], [0.2, 0.1, 0.2, 0.1], [0.0, 0.1, 0.0, 0.1]]),
        2: np.asarray([[0.2, 0.1, 0.2, 0.1], [1.0, -1.0, 1.0, -1.0], [0.1, 0.0, 0.1, 0.0]]),
    }
    trials = []
    for trial_index, label in enumerate(labels):
        signal = np.zeros((3, time.size), dtype=float)
        signal[:, time < 0.0] = 0.05 * (trial_index + 1)
        signal[:, time >= 0.0] = scale * base_patterns[int(label)]
        trials.append(signal)
    return trials


class TestStimulusCovarianceFeatures(unittest.TestCase):
    def test_load_participant_covariance_features_uses_main_data_only(self):
        time = np.asarray([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03], dtype=float)
        labels = [1, 2]
        trials = _covariance_trials(labels, scale=1.0, time=time)
        data_by_participant = {1: _mat_data_from_trials(labels, trials, time)}
        config = CovarianceStimulusConfig(
            time_window=(0.0, 0.03),
            baseline_window=(-0.03, -0.005),
            normalization="none",
            covariance_feature_mode="logeuclidean_covariance",
            projection="none",
            n_components=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_covariance_features.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)) as loadmat:
            feature_set = load_participant_covariance_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.n_channels, 3)
        self.assertEqual(feature_set.n_time_bins, 4)
        self.assertEqual(feature_set.covariance_feature_mode, "logeuclidean_covariance")
        self.assertTrue(np.all(np.isfinite(feature_set.features)))
        self.assertTrue(str(loadmat.call_args.args[0]).endswith("Part1Data.mat"))
        self.assertNotIn("CueData", str(loadmat.call_args.args[0]))

    def test_covariance_feature_modes_have_expected_widths(self):
        time = np.asarray([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03], dtype=float)
        labels = [1, 2]
        trials = _covariance_trials(labels, scale=1.0, time=time)
        data_by_participant = {1: _mat_data_from_trials(labels, trials, time)}

        widths = {}
        with patch("pymegdec.stimulus_covariance_features.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            for mode in ("covariance_upper", "correlation_upper", "variance"):
                config = CovarianceStimulusConfig(
                    time_window=(0.0, 0.03),
                    normalization="none",
                    covariance_feature_mode=mode,
                    projection="none",
                    n_components=float("inf"),
                    chance_classes=2,
                )
                widths[mode] = load_participant_covariance_features("unused", 1, config=config).features.shape[1]

        self.assertEqual(widths["covariance_upper"], 6)
        self.assertEqual(widths["correlation_upper"], 6)
        self.assertEqual(widths["variance"], 3)

    def test_nested_covariance_logistic_selects_candidate_without_target_labels(self):
        time = np.asarray([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03], dtype=float)
        data_by_participant = {}
        labels = [1, 2, 1, 2]
        for participant, scale in [(1, 1.00), (2, 1.05), (3, 0.95), (4, 1.10)]:
            trials = _covariance_trials(labels, scale=scale, time=time)
            data_by_participant[participant] = _mat_data_from_trials(labels, trials, time)
        candidate_configs = make_covariance_candidate_configs(
            time_windows=((0.0, 0.03),),
            normalizations=("none",),
            covariance_feature_modes=("logeuclidean_covariance",),
            covariance_shrinkages=(0.1,),
            covariance_epsilons=(1e-6,),
            projections=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_values=(float("inf"),),
            chance_classes=2,
            signflip_permutations=128,
        )

        with patch("pymegdec.stimulus_covariance_features.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            artifacts = evaluate_nested_covariance_stimulus("unused", [1, 2, 3, 4], candidate_configs=candidate_configs)

        self.assertEqual(len(artifacts["outer"]), 4)
        self.assertEqual(len(artifacts["inner_validation"]), 12)
        self.assertEqual(len(artifacts["predictions"]), 16)
        self.assertEqual({row["feature_family"] for row in artifacts["outer"]}, {"covariance"})
        self.assertEqual({row["feature_mode"] for row in artifacts["outer"]}, {"logeuclidean_covariance"})
        self.assertEqual({row["selected_covariance_feature_mode"] for row in artifacts["outer"]}, {"logeuclidean_covariance"})


if __name__ == "__main__":
    unittest.main()
