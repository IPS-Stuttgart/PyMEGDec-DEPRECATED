import re
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    evaluate_cross_subject_stimulus_smoke,
    load_participant_stimulus_features,
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


def _loadmat_side_effect(data_by_participant):
    def loadmat(path):
        match = re.search(r"Part(\d+)Data\.mat$", str(path))
        if not match:
            raise AssertionError(f"Unexpected MAT path: {path}")
        participant = int(match.group(1))
        return {"data": np.array([data_by_participant[participant]], dtype=object)}

    return loadmat


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
        self.assertEqual(artifacts["group_summary"][0]["participants_above_chance"], 3)
        self.assertEqual({row["true_stimulus"] for row in artifacts["predictions"]}, {1, 2})
        self.assertEqual({row["predicted_stimulus"] for row in artifacts["predictions"]}, {1, 2})

    def test_summarize_cross_subject_stimulus_smoke_signflip(self):
        config = CrossSubjectStimulusConfig(chance_classes=2, signflip_permutations=128)
        rows = [
            {"balanced_accuracy": 0.75, "accuracy": 0.75, "chance_accuracy": 0.5},
            {"balanced_accuracy": 1.0, "accuracy": 1.0, "chance_accuracy": 0.5},
        ]

        summary = summarize_cross_subject_stimulus_smoke(rows, config=config)

        self.assertEqual(summary[0]["n_outer_folds"], 2)
        self.assertAlmostEqual(summary[0]["balanced_accuracy_mean"], 0.875)
        self.assertLessEqual(summary[0]["one_sided_signflip_p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
