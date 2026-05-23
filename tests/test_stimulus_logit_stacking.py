import unittest

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pymegdec import stimulus_logit_stacking as stack


class TestStimulusLogitStacking(unittest.TestCase):
    def test_align_score_columns_fills_missing_classes(self):
        scores = np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=float)
        aligned = stack._align_score_columns(scores, np.asarray([0, 2]), np.arange(4))  # pylint: disable=protected-access

        self.assertEqual(aligned.shape, (2, 4))
        np.testing.assert_allclose(aligned[:, 0], scores[:, 0])
        np.testing.assert_allclose(aligned[:, 2], scores[:, 1])
        self.assertTrue(np.all(aligned[:, 1] < np.min(scores, axis=1)))
        self.assertTrue(np.all(aligned[:, 3] < np.min(scores, axis=1)))

    def test_rank_normalization_preserves_score_order(self):
        scores = np.asarray([[0.1, 0.3, 0.2]], dtype=float)
        ranked = stack._normalize_scores(scores, mode="rank")  # pylint: disable=protected-access

        self.assertEqual(tuple(np.argsort(-ranked[0])), (1, 2, 0))

    def test_class_bias_is_fitted_from_source_oof_scores(self):
        class_order = np.arange(2)
        labels = np.asarray([0, 0, 1, 1], dtype=int)
        scores = np.asarray(
            [
                [0.60, 0.50],
                [0.60, 0.50],
                [0.55, 0.50],
                [0.55, 0.50],
            ],
            dtype=float,
        )
        unbiased = stack._balanced_accuracy_for_scores(scores, labels, class_order)  # pylint: disable=protected-access
        bias, biased = stack._optimize_class_bias(scores, labels, class_order, l2=1e-4)  # pylint: disable=protected-access

        self.assertEqual(unbiased, 0.5)
        self.assertGreater(biased, unbiased)
        self.assertAlmostEqual(biased, 1.0)
        self.assertGreater(bias[1], bias[0])

    def test_greedy_stacker_prefers_source_oof_winner(self):
        class_order = np.arange(3)
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=int)
        good = np.full((labels.shape[0], class_order.shape[0]), -1.0, dtype=float)
        good[np.arange(labels.shape[0]), labels] = 3.0
        bad = np.roll(good, shift=1, axis=1)
        weak = np.zeros_like(good)
        score_cube = np.stack([bad, good, weak], axis=0)
        inner_balanced = np.asarray(
            [balanced_accuracy_score(labels, class_order[np.argmax(scores, axis=1)]) for scores in score_cube],
            dtype=float,
        )

        weights, selected = stack._fit_stacker_weights(  # pylint: disable=protected-access
            score_cube,
            labels,
            class_order,
            inner_balanced,
            weighting="greedy_balanced",
            temperature=0.02,
            max_base_models=1,
        )
        stacked = stack._weighted_score_average(score_cube, weights)  # pylint: disable=protected-access

        self.assertEqual(selected, (1,))
        self.assertAlmostEqual(weights[1], 1.0)
        self.assertAlmostEqual(stack._balanced_accuracy_for_scores(stacked, labels, class_order), 1.0)  # pylint: disable=protected-access

    def test_candidate_grid_expands_base_model_axes(self):
        configs = stack.make_logit_stack_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_flat",),
            normalizations=("none",),
            alignments=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(32, 64),
            sample_weightings=("none", "subject_class_balanced"),
            chance_classes=3,
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual({config.components_pca for config in configs}, {32, 64})
        self.assertEqual({config.sample_weighting for config in configs}, {"none", "subject_class_balanced"})


if __name__ == "__main__":
    unittest.main()
