import unittest
from unittest.mock import patch

import numpy as np

from pymegdec import stimulus_full_epoch_lowrank as full_epoch
from pymegdec.stimulus_full_epoch_lowrank import (
    FullEpochLowRankConfig,
    evaluate_nested_full_epoch_lowrank_stimulus,
    load_participant_full_epoch_features,
    make_full_epoch_lowrank_candidate_configs,
)
from tests.matlab_fixtures import loadmat_side_effect, mat_data_from_trials


def _participant_trials(values, *, time):
    labels = [1, 2, 1, 2]
    trials = []
    for label, value in zip(labels, values, strict=True):
        signal = np.zeros((2, time.size), dtype=float)
        signal[:, time < 0.0] = 0.1 * label
        signal[:, time >= 0.0] = value
        signal[1, time >= 0.0] = -value
        trials.append(signal)
    return labels, trials


class TestStimulusFullEpochLowRank(unittest.TestCase):
    def test_load_participant_full_epoch_features_bins_time_axis(self):
        time = np.asarray([-0.05, 0.00, 0.01, 0.02, 0.03, 0.04], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0, 9.0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]],
            [[0.0, 2.0, 6.0, 10.0, 14.0, 18.0], [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = FullEpochLowRankConfig(
            time_window=(0.0, 0.04),
            time_bin_size=0.02,
            normalization="none",
            projection="none",
            n_components=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_full_epoch_lowrank.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_full_epoch_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 4))
        self.assertEqual(feature_set.n_time_bins, 2)
        np.testing.assert_allclose(feature_set.features[0], np.asarray([2.0, 3.0, 7.0, 8.0]))
        np.testing.assert_allclose(feature_set.features[1], np.asarray([4.0, 6.0, 14.0, 16.0]))

    def test_load_participant_full_epoch_features_can_append_first_differences(self):
        time = np.asarray([-0.05, 0.00, 0.01, 0.02, 0.03, 0.04], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0, 9.0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]],
            [[0.0, 2.0, 6.0, 10.0, 14.0, 18.0], [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = FullEpochLowRankConfig(
            time_window=(0.0, 0.04),
            time_bin_size=0.02,
            temporal_feature_mode="mean+d1",
            normalization="none",
            projection="none",
            n_components=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_full_epoch_lowrank.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_full_epoch_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 8))
        self.assertEqual(feature_set.temporal_feature_mode, "mean_d1")
        self.assertEqual(feature_set.n_time_bins, 2)
        np.testing.assert_allclose(feature_set.features[0], np.asarray([2.0, 3.0, 7.0, 8.0, 0.0, 0.0, 5.0, 5.0]))
        np.testing.assert_allclose(feature_set.features[1], np.asarray([4.0, 6.0, 14.0, 16.0, 0.0, 0.0, 10.0, 10.0]))

    def test_full_epoch_subject_baseline_whiten_keeps_channel_blocks(self):
        time = np.asarray([-0.04, -0.02, 0.00, 0.01, 0.02, 0.03], dtype=float)
        labels, trials = _participant_trials([-1.0, 1.0, -0.8, 0.8], time=time)
        data_by_participant = {1: mat_data_from_trials(labels, trials, time)}
        config = FullEpochLowRankConfig(
            time_window=(0.0, 0.04),
            time_bin_size=0.02,
            baseline_window=(-0.04, -0.01),
            normalization="subject_baseline_whiten",
            projection="none",
            n_components=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_full_epoch_lowrank.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_full_epoch_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 4))
        self.assertEqual(feature_set.n_baseline_samples, 2)
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_nested_full_epoch_pls_logistic_selects_candidate_without_target_labels(self):
        time = np.asarray([-0.04, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04], dtype=float)
        data_by_participant = {}
        for participant, scale in [(1, 1.2), (2, 1.0), (3, 1.3), (4, 1.1)]:
            labels, trials = _participant_trials([-scale, scale, -0.9 * scale, 0.9 * scale], time=time)
            data_by_participant[participant] = mat_data_from_trials(labels, trials, time)
        candidate_configs = make_full_epoch_lowrank_candidate_configs(
            time_windows=((0.0, 0.04),),
            time_bin_size=0.02,
            baseline_window=(-0.05, -0.01),
            normalizations=("none",),
            projections=("pls",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_values=(1,),
            chance_classes=2,
            signflip_permutations=128,
        )

        with patch("pymegdec.stimulus_full_epoch_lowrank.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            artifacts = evaluate_nested_full_epoch_lowrank_stimulus("unused", [1, 2, 3, 4], candidate_configs=candidate_configs)

        self.assertEqual(len(artifacts["outer"]), 4)
        self.assertEqual(len(artifacts["inner_validation"]), 12)
        self.assertEqual(len(artifacts["selected"]), 4)
        self.assertEqual(len(artifacts["predictions"]), 16)
        self.assertEqual({row["projection"] for row in artifacts["outer"]}, {"pls"})
        self.assertEqual({row["feature_mode"] for row in artifacts["outer"]}, {full_epoch.FULL_EPOCH_FEATURE_MODE})
        self.assertEqual({row["temporal_feature_mode"] for row in artifacts["outer"]}, {"mean"})
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["selection_mode"], "nested_loso")
        self.assertEqual(artifacts["group_summary"][0]["selected_projection_counts"], "pls:4")
        self.assertEqual(artifacts["group_summary"][0]["participants_above_chance"], 4)


if __name__ == "__main__":
    unittest.main()
