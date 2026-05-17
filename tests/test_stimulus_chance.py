import unittest
from unittest.mock import patch

import numpy as np
from pymegdec.stimulus_decoding import (
    DEFAULT_CHANCE_CLASSES,
    StimulusDecodingConfig,
    evaluate_participant_time_resolved_stimulus_transfer,
)
from tests.matlab_fixtures import cell_array


def _mat_data(labels, trial_values, time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    return {
        "trial": cell_array([np.asarray([[0.0, value]], dtype=float) for value in trial_values]),
        "time": cell_array([np.asarray([time], dtype=float) for _ in trial_values]),
        "trialinfo": trialinfo,
    }


class TestStimulusChanceLevel(unittest.TestCase):
    def _evaluate(self, config):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows = evaluate_participant_time_resolved_stimulus_transfer("unused", 1, config=config)
        return rows[0]

    def test_auto_chance_uses_validation_class_count(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.5)
        self.assertEqual(row["chance_percent"], 50.0)

    def test_legacy_default_chance_classes_is_treated_as_auto_for_cli_defaults(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=DEFAULT_CHANCE_CLASSES,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.5)

    def test_non_default_chance_classes_remain_explicit_override(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=4,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 0.25)
        self.assertEqual(row["chance_percent"], 25.0)

    def test_inference_can_be_disabled_to_force_sixteen_way_chance(self):
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=DEFAULT_CHANCE_CLASSES,
            infer_chance_classes=False,
        )

        row = self._evaluate(config)

        self.assertEqual(row["n_validation_classes"], 2)
        self.assertEqual(row["chance_accuracy"], 1.0 / DEFAULT_CHANCE_CLASSES)
        self.assertEqual(row["chance_percent"], 100.0 / DEFAULT_CHANCE_CLASSES)


if __name__ == "__main__":
    unittest.main()
