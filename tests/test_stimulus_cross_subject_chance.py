import re
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    evaluate_cross_subject_stimulus_smoke,
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


class TestStimulusCrossSubjectChance(unittest.TestCase):
    def test_cross_subject_chance_uses_held_out_classes(self):
        data_by_participant = {
            1: _mat_data([1, 2, 3, 1, 2, 3], [-1.2, 0.0, 1.2, -1.1, 0.1, 1.1]),
            2: _mat_data([1, 2, 3, 1, 2, 3], [-1.0, 0.0, 1.0, -0.9, 0.1, 0.9]),
            3: _mat_data([1, 2, 3, 1, 2, 3], [-1.3, 0.0, 1.3, -1.2, 0.1, 1.2]),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            normalization="none",
            classifier="multiclass-svm",
            classifier_param=0.5,
            components_pca=float("inf"),
            chance_classes=16,
            signflip_permutations=128,
        )

        with patch(
            "pymegdec.stimulus_cross_subject.sio.loadmat",
            side_effect=_loadmat_side_effect(data_by_participant),
        ):
            artifacts = evaluate_cross_subject_stimulus_smoke(
                "unused",
                [1, 2, 3],
                config=config,
            )

        self.assertEqual({row["n_test_classes"] for row in artifacts["outer"]}, {3})
        self.assertEqual({row["chance_classes"] for row in artifacts["outer"]}, {3})
        for row in artifacts["outer"]:
            self.assertAlmostEqual(row["chance_accuracy"], 1 / 3)
            self.assertAlmostEqual(row["chance_percent"], 100 / 3)
            self.assertAlmostEqual(row["top2_chance_accuracy"], 2 / 3)
            self.assertAlmostEqual(row["top3_chance_accuracy"], 1.0)
            self.assertAlmostEqual(row["chance_mean_rank"], 2.0)
        self.assertAlmostEqual(artifacts["group_summary"][0]["chance_accuracy"], 1 / 3)
        self.assertAlmostEqual(artifacts["group_summary"][0]["chance_accuracy_min"], 1 / 3)
        self.assertAlmostEqual(artifacts["group_summary"][0]["chance_accuracy_max"], 1 / 3)
        self.assertEqual(artifacts["group_summary"][0]["chance_classes_counts"], "3:3")

    def test_summarize_cross_subject_stimulus_smoke_uses_per_fold_chance(self):
        config = CrossSubjectStimulusConfig(chance_classes=16, signflip_permutations=128)
        rows = [
            {
                "balanced_accuracy": 0.40,
                "accuracy": 0.40,
                "chance_accuracy": 0.50,
                "chance_classes": 2,
            },
            {
                "balanced_accuracy": 0.30,
                "accuracy": 0.30,
                "chance_accuracy": 0.25,
                "chance_classes": 4,
            },
        ]

        summary = summarize_cross_subject_stimulus_smoke(rows, config=config)

        self.assertAlmostEqual(summary[0]["chance_accuracy"], 0.375)
        self.assertEqual(summary[0]["chance_classes_counts"], "2:1;4:1")
        self.assertAlmostEqual(summary[0]["mean_above_chance"], (-0.10 + 0.05) / 2)
        self.assertEqual(summary[0]["participants_above_chance"], 1)
        self.assertEqual(summary[0]["participants_at_or_below_chance"], 1)
        self.assertAlmostEqual(summary[0]["one_sided_exact_sign_p_value"], 0.75)


if __name__ == "__main__":
    unittest.main()
