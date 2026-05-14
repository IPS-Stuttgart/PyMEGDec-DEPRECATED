import unittest
from dataclasses import dataclass

import numpy as np
from pymegdec.stimulus_cross_subject_controls import (
    LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
    LABEL_CONTROL_NONE,
    LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
    controlled_labels,
    controlled_training_sets,
    exact_one_sided_sign_p_value,
    normalize_label_control,
)


@dataclass(frozen=True)
class _FeatureSet:
    participant: int
    labels: np.ndarray


class CrossSubjectLabelControlsTest(unittest.TestCase):
    def test_exact_sign_test_all_above_chance(self):
        self.assertAlmostEqual(exact_one_sided_sign_p_value(3, 3), 1 / 8)
        self.assertAlmostEqual(exact_one_sided_sign_p_value(23, 23), 1 / (2**23))

    def test_shuffle_within_subject_preserves_class_counts(self):
        labels = np.array([1, 1, 1, 2, 2, 3])

        controlled = controlled_labels(
            labels,
            participant=7,
            label_control=LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
            label_control_seed=13,
        )

        self.assertCountEqual(controlled.tolist(), labels.tolist())
        np.testing.assert_array_equal(labels, np.array([1, 1, 1, 2, 2, 3]))

    def test_circular_shift_preserves_multiset_and_changes_positions(self):
        labels = np.arange(1, 9)

        controlled = controlled_labels(
            labels,
            participant=3,
            label_control=LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
            label_control_seed=5,
        )

        self.assertCountEqual(controlled.tolist(), labels.tolist())
        self.assertFalse(np.array_equal(controlled, labels))

    def test_none_control_is_copy(self):
        labels = np.array([1, 2, 3])

        controlled = controlled_labels(labels, participant=1, label_control=LABEL_CONTROL_NONE, label_control_seed=0)

        np.testing.assert_array_equal(controlled, labels)
        self.assertIsNot(controlled, labels)

    def test_controlled_training_sets_do_not_mutate_original_sets(self):
        original = _FeatureSet(participant=1, labels=np.array([1, 1, 2, 2]))

        controlled = controlled_training_sets(
            [original],
            label_control=LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
            label_control_seed=0,
        )

        self.assertIsNot(controlled[0], original)
        self.assertCountEqual(controlled[0].labels.tolist(), original.labels.tolist())
        np.testing.assert_array_equal(original.labels, np.array([1, 1, 2, 2]))

    def test_label_control_aliases(self):
        self.assertEqual(normalize_label_control("shuffle"), LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT)
        self.assertEqual(normalize_label_control("circular_shift"), LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT)
        self.assertEqual(normalize_label_control(""), LABEL_CONTROL_NONE)


if __name__ == "__main__":
    unittest.main()
