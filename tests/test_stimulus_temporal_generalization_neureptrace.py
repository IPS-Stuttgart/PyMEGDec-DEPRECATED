import unittest

from pymegdec.stimulus_temporal_generalization_neureptrace import summarize_stimulus_temporal_generalization_with_neureptrace


class TestStimulusTemporalGeneralizationNeuRepTrace(unittest.TestCase):
    def test_summary_delegates_to_neureptrace_shape_with_pymegdec_compatibility(self):
        rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "is_diagonal": True,
                "accuracy": 0.25,
                "chance_accuracy": 0.0625,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
            },
            {
                "participant": 2,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "is_diagonal": True,
                "accuracy": 0.50,
                "chance_accuracy": 0.0625,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
            },
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.1,
                "is_diagonal": False,
                "accuracy": 0.10,
                "chance_accuracy": 0.0625,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
            },
        ]

        summary = summarize_stimulus_temporal_generalization_with_neureptrace(rows)

        self.assertEqual(len(summary), 2)
        diagonal = [row for row in summary if row["is_diagonal"]][0]
        off_diagonal = [row for row in summary if not row["is_diagonal"]][0]
        self.assertEqual(diagonal["n_participants"], 2)
        self.assertAlmostEqual(diagonal["accuracy_mean"], 0.375)
        self.assertAlmostEqual(diagonal["percent_mean"], 37.5)
        self.assertEqual(diagonal["above_chance_count"], 2)
        self.assertEqual(off_diagonal["n_participants"], 1)
        self.assertAlmostEqual(off_diagonal["accuracy_mean"], 0.10)

    def test_empty_summary_is_empty_list(self):
        self.assertEqual(summarize_stimulus_temporal_generalization_with_neureptrace([]), [])


if __name__ == "__main__":
    unittest.main()
