import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pymegdec import _stimulus_cross_subject_next as next_hooks
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig, load_participant_stimulus_features, make_cross_subject_candidate_configs
from tests.matlab_fixtures import loadmat_side_effect, mat_data_from_trials


class TestStimulusCrossSubjectNext(unittest.TestCase):
    def test_sensor_flat_gaussian_taper_time_pyramid_feature_is_exported(self):
        mode = "sensor_flat_gaussian_taper_time_pyramid"

        self.assertEqual(cross_subject._normalize_feature_mode(mode), mode)  # pylint: disable=protected-access

        signal = np.arange(15, dtype=float).reshape(3, 5)
        feature = next_hooks._sensor_flat_gaussian_taper_time_pyramid_feature(signal)  # pylint: disable=protected-access
        tapered = next_hooks._sensor_flat_gaussian_taper_feature(signal)  # pylint: disable=protected-access
        pyramid = next_hooks._sensor_time_pyramid_feature(signal)  # pylint: disable=protected-access

        self.assertEqual(feature.shape, (3 * (5 + 1 + 2 + 4),))
        np.testing.assert_allclose(feature, np.concatenate((tapered, pyramid)))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_soft_guarded_inner_confusion_score_normalizations_are_exported(self):
        mode = "rank_softmax_t2_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t2",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:2;1002:1;2002:3"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertLess(metadata["blend"], 1.0)

    def test_margin_gated_inner_confusion_score_normalization_is_exported(self):
        mode = "rank_z_blend_inner_confusion_margin_soft"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_z_blend",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:3;1002:2;2002:4"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["margin_gated"])
        self.assertEqual(metadata["margin_quantile"], 0.5)
        self.assertLess(metadata["blend"], 1.0)

    def test_rank_margin_blend_inner_confusion_soft_is_exported_without_margin_gate(self):
        mode = "rank_margin_blend_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_margin_blend",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertFalse(metadata["margin_gated"])
        self.assertLess(metadata["blend"], 1.0)

    def test_topk_borda_soft_guarded_inner_confusion_modes_are_exported(self):
        mode = "rank_top3_borda_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_borda",
        )

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )
        self.assertEqual(np.count_nonzero(probabilities[0]), 3)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])

    def test_topk_score_softmax_modes_are_exported(self):
        mode = "rank_top3_score_softmax"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )

        self.assertEqual(np.count_nonzero(probabilities[0]), 3)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))
        self.assertGreater(probabilities[0, 0], probabilities[0, 1])
        self.assertGreater(probabilities[0, 1], probabilities[0, 2])
        self.assertEqual(probabilities[0, 3], 0.0)

    def test_topk_score_softmax_log_pool_modes_are_exported(self):
        modes = (
            "rank_top2_score_softmax_log_pool",
            "rank_top3_score_softmax_log_pool",
            "rank_top3_adaptive_score_softmax_log_pool",
        )
        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)

        for mode in modes:
            with self.subTest(mode=mode):
                base_mode = mode.removesuffix("_log_pool")
                self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
                    base_mode,
                )
                self.assertEqual(
                    cross_subject._ensemble_log_pool_mode(mode),  # pylint: disable=protected-access
                    mode,
                )

                probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
                    scores,
                    score_normalization=mode,
                )
                np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))
                self.assertGreater(np.count_nonzero(probabilities[0]), 0)

    def test_topk_adaptive_score_softmax_modes_are_exported(self):
        mode = "rank_top3_adaptive_score_softmax"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)

        confident_scores = np.asarray([[6.0, 1.0, 0.0, -1.0]], dtype=float)
        ambiguous_scores = np.asarray([[3.0, 2.9, 2.8, 0.0]], dtype=float)
        confident = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            confident_scores,
            score_normalization=mode,
        )
        ambiguous = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            ambiguous_scores,
            score_normalization=mode,
        )

        self.assertEqual(np.count_nonzero(confident[0]), 3)
        self.assertEqual(np.count_nonzero(ambiguous[0]), 3)
        np.testing.assert_allclose(np.sum(confident, axis=1), np.ones(1))
        np.testing.assert_allclose(np.sum(ambiguous, axis=1), np.ones(1))
        self.assertEqual(confident[0, 3], 0.0)
        self.assertEqual(ambiguous[0, 3], 0.0)
        self.assertGreater(confident[0, 0], ambiguous[0, 0])

    def test_topk_score_borda_blend_modes_are_exported_and_margin_gated(self):
        mode = "rank_top3_score_borda_blend"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            mode,
        )

        confident_scores = np.asarray([[6.0, 1.0, 0.0, -1.0]], dtype=float)
        ambiguous_scores = np.asarray([[3.0, 2.9, 2.8, 0.0]], dtype=float)
        confident = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            confident_scores,
            score_normalization=mode,
        )
        ambiguous = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            ambiguous_scores,
            score_normalization=mode,
        )

        np.testing.assert_allclose(np.sum(confident, axis=1), np.ones(1))
        np.testing.assert_allclose(np.sum(ambiguous, axis=1), np.ones(1))
        self.assertEqual(np.count_nonzero(confident[0]), 3)
        self.assertEqual(np.count_nonzero(ambiguous[0]), 3)
        self.assertEqual(confident[0, 3], 0.0)
        self.assertEqual(ambiguous[0, 3], 0.0)

        ambiguous_borda = next_hooks._rank_topk_borda_probabilities(  # pylint: disable=protected-access
            ambiguous_scores,
            top_k=3,
        )
        ambiguous_score_softmax = (
            next_hooks._rank_topk_score_softmax_probabilities(  # pylint: disable=protected-access
                ambiguous_scores,
                top_k=3,
            )
        )
        self.assertLess(
            np.linalg.norm(ambiguous - ambiguous_borda),
            np.linalg.norm(ambiguous_score_softmax - ambiguous_borda),
        )

    def test_topk_agreement_log_pool_blends_on_near_top_consensus(self):
        mode = "rank_top3_score_softmax_top3_agreement_log_pool"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_score_softmax",
        )

        stacked = np.asarray(
            [
                [
                    [0.40, 0.30, 0.20, 0.05, 0.03, 0.02],
                    [0.45, 0.25, 0.20, 0.05, 0.03, 0.02],
                ],
                [
                    [0.30, 0.40, 0.20, 0.05, 0.03, 0.02],
                    [0.05, 0.03, 0.02, 0.45, 0.25, 0.20],
                ],
                [
                    [0.20, 0.30, 0.40, 0.05, 0.03, 0.02],
                    [0.03, 0.45, 0.02, 0.05, 0.25, 0.20],
                ],
            ],
            dtype=float,
        )
        weights = np.full(3, 1.0 / 3.0, dtype=float)

        blend = next_hooks._topk_agreement_log_pool_blend_weights(stacked, weights, top_k=3)  # pylint: disable=protected-access
        self.assertGreater(blend[0], blend[1])
        self.assertGreater(blend[0], 0.75)

        pooled = cross_subject._pool_ensemble_probability_matrices(tuple(stacked), weights, mode)  # pylint: disable=protected-access
        self.assertEqual(pooled.shape, (2, 6))
        np.testing.assert_allclose(np.sum(pooled, axis=1), np.ones(2))

    def test_topk_score_softmax_balanced_quota_modes_are_exported_and_constrained(self):
        mode = "rank_top2_score_softmax_balanced_quota"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top2_score_softmax",
        )

        probabilities = np.asarray(
            [
                [0.90, 0.10, 0.00],
                [0.80, 0.20, 0.00],
                [0.10, 0.80, 0.10],
                [0.00, 0.70, 0.30],
                [0.20, 0.00, 0.80],
                [0.10, 0.00, 0.90],
            ],
            dtype=float,
        )
        metadata = cross_subject._balanced_quota_metadata(  # pylint: disable=protected-access
            probabilities,
            np.arange(3, dtype=int),
            mode,
        )

        self.assertEqual(metadata["status"], "applied_top2_constrained")
        np.testing.assert_array_equal(metadata["quota_counts"], np.asarray([2, 2, 2]))
        np.testing.assert_array_equal(metadata["predictions"], np.asarray([0, 0, 1, 1, 2, 2]))
        self.assertEqual(metadata["top_k_constrained"], 2)

    def test_topk_score_softmax_inner_confusion_guarded_mode_is_exported(self):
        mode = "rank_top3_score_softmax_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_score_softmax",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])

    def test_top2_score_softmax_inner_recall_bias_mode_is_exported(self):
        mode = "rank_top2_score_softmax_inner_recall_bias"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top2_score_softmax",
        )

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )
        self.assertEqual(np.count_nonzero(probabilities[0]), 2)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))

        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:4;2002:8"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["recall_bias_status"], "applied")
        self.assertEqual(metadata["log_adjustment"].shape, (2,))
        self.assertGreater(metadata["log_adjustment"][0], metadata["log_adjustment"][1])

    def test_topk_gated_inner_recall_bias_only_adjusts_near_top_classes(self):
        mode = "rank_softmax_inner_recall_bias_top2"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax",
        )

        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_true_predicted_label_pair_counts": (
                        "1001:8;1002:2;2002:8;3003:1;3004:5;4004:8"
                    ),
                    "selected_inner_confusion_counts": "",
                }
            ],
            np.arange(4, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["recall_bias_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k"], 2)

        probabilities = np.asarray(
            [[0.50, 0.30, 0.15, 0.05]],
            dtype=float,
        )
        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(1))
        self.assertFalse(np.allclose(adjusted, probabilities))
        # Classes 3 and 4 are outside the row's original top-2.  The top-k gate
        # may rescale them during row renormalization, but it must not apply
        # different class-bias multipliers to them.
        self.assertAlmostEqual(adjusted[0, 2] / adjusted[0, 3], 3.0)
        self.assertEqual(metadata["inner_bias_top_k_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k_adjusted_trials"], 1)

    def test_guarded_inner_recall_bias_is_exported_and_margin_gated(self):
        mode = "rank_top2_score_softmax_inner_recall_bias_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top2_score_softmax",
        )

        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_true_predicted_label_pair_counts": "1001:4;1002:4;2002:8",
                    "selected_inner_confusion_counts": "",
                }
            ],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["recall_bias_status"], "applied")
        self.assertTrue(metadata["recall_bias_guarded"])
        self.assertTrue(metadata["inner_bias_guarded"])

        probabilities = np.asarray(
            [[0.49, 0.51], [0.95, 0.05]],
            dtype=float,
        )
        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(2))
        self.assertFalse(np.allclose(adjusted[0], probabilities[0]))
        np.testing.assert_allclose(adjusted[1], probabilities[1])
        self.assertEqual(metadata["inner_bias_guard_status"], "applied")
        self.assertEqual(metadata["inner_bias_guard_adjusted_trials"], 1)
        self.assertAlmostEqual(metadata["inner_bias_guard_margin_quantile"], 0.50)

    def test_topk_adaptive_score_softmax_inner_confusion_guarded_mode_is_exported(self):
        mode = "rank_top3_adaptive_score_softmax_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_adaptive_score_softmax",
        )

        ambiguous_scores = np.asarray([[3.0, 2.9, 2.8, 0.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            ambiguous_scores,
            score_normalization=mode,
        )
        self.assertEqual(np.count_nonzero(probabilities[0]), 3)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])

    def test_top2_margin_recall_bias_mode_is_exported_and_scaled(self):
        mode = "rank_top2_margin_blend_inner_recall_bias"
        strong_mode = "rank_top2_margin_blend_inner_recall_bias_s100"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn(strong_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top2_margin_blend",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:6;1002:2;2001:5;2002:3",
                "selected_inner_confusion_counts": "",
            }
        ]
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertAlmostEqual(metadata["recall_bias_strength"], 0.50)
        self.assertGreater(metadata["log_adjustment"][1], metadata["log_adjustment"][0])

    def test_inner_recall_bias_strength_variants_are_exported_and_scaled(self):
        weak_mode = "rank_softmax_t0_75_inner_recall_bias_s25"
        default_mode = "rank_softmax_t0_75_inner_recall_bias"
        strong_mode = "rank_softmax_t0_75_inner_recall_bias_s100"

        self.assertIn(weak_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn(strong_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(weak_mode),  # pylint: disable=protected-access
            "rank_softmax_t0_75",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:6;1002:2;2001:5;2002:3",
                "selected_inner_confusion_counts": "",
            }
        ]
        weak_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), weak_mode
        )
        default_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), default_mode
        )
        strong_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), strong_mode
        )

        self.assertEqual(weak_metadata["inner_mode"], weak_mode)
        self.assertAlmostEqual(weak_metadata["recall_bias_strength"], 0.25)
        self.assertAlmostEqual(default_metadata["recall_bias_strength"], 0.50)
        self.assertAlmostEqual(strong_metadata["recall_bias_strength"], 1.00)
        self.assertGreater(weak_metadata["log_adjustment"][1], weak_metadata["log_adjustment"][0])
        self.assertLess(np.ptp(weak_metadata["log_adjustment"]), np.ptp(default_metadata["log_adjustment"]))
        self.assertLess(np.ptp(default_metadata["log_adjustment"]), np.ptp(strong_metadata["log_adjustment"]))

    def test_inner_prediction_bias_is_exported_and_uses_predicted_mass(self):
        mode = "rank_softmax_t0_75_inner_prediction_bias_s25"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t0_75",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:4;2001:4;2002:4",
                "selected_inner_confusion_counts": "",
            }
        ]
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["prediction_bias_status"], "applied")
        self.assertAlmostEqual(metadata["prediction_bias_strength"], 0.25)
        self.assertGreater(metadata["log_adjustment"][1], metadata["log_adjustment"][0])

        probabilities = np.asarray([[0.55, 0.45]], dtype=float)
        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )
        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(1))
        self.assertGreater(adjusted[0, 1], probabilities[0, 1])

    def test_topk_gated_inner_prediction_bias_only_adjusts_near_top_classes(self):
        mode = "rank_softmax_inner_prediction_bias_top2"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:4;2001:4;2002:4;3001:4;3003:4;4004:4"
                ),
                "selected_inner_confusion_counts": "",
            }
        ]
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(4, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["prediction_bias_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k"], 2)

        probabilities = np.asarray(
            [[0.50, 0.30, 0.15, 0.05]],
            dtype=float,
        )
        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(1))
        self.assertFalse(np.allclose(adjusted, probabilities))
        # Classes 3 and 4 are outside the row's original top-2.  They may be
        # rescaled by row renormalization, but they must keep their original
        # relative odds because no class-specific prediction-bias multiplier was
        # applied to either of them.
        self.assertAlmostEqual(adjusted[0, 2] / adjusted[0, 3], 3.0)
        self.assertEqual(metadata["inner_bias_top_k_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k_adjusted_trials"], 1)

    def test_top2_precision_recall_bias_modes_are_exported_and_conservative(self):
        mode = "rank_top2_margin_blend_inner_precision_recall_bias"
        weak_mode = "rank_top2_margin_blend_inner_precision_recall_bias_s25"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn(weak_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top2_margin_blend",
        )

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )
        self.assertEqual(np.count_nonzero(probabilities[0]), 2)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:8;2001:6;2002:2",
                "selected_inner_confusion_counts": "",
            }
        ]
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        weak_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            weak_mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["precision_recall_bias_status"], "applied")
        self.assertAlmostEqual(metadata["precision_recall_bias_recall_strength"], 0.35)
        self.assertAlmostEqual(weak_metadata["precision_recall_bias_recall_strength"], 0.25)
        self.assertLess(np.ptp(weak_metadata["log_adjustment"]), np.ptp(metadata["log_adjustment"]))

    def test_inner_precision_recall_bias_is_exported_and_conservative(self):
        mode = "rank_top3_margin_blend_inner_precision_recall_bias"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_margin_blend",
        )

        # Class 2 is under-recalled but relatively precise; class 1 is a noisy
        # false-positive sink.  The precision/recall bias should therefore boost
        # class 2 and downweight class 1 using only source-inner confusion counts.
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_true_predicted_label_pair_counts": "1001:8;2001:6;2002:2",
                    "selected_inner_confusion_counts": "",
                }
            ],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["precision_recall_bias_status"], "applied")
        self.assertAlmostEqual(metadata["precision_recall_bias_recall_strength"], 0.35)
        self.assertAlmostEqual(metadata["precision_recall_bias_precision_strength"], 0.35)
        self.assertGreater(metadata["log_adjustment"][1], metadata["log_adjustment"][0])
        self.assertLessEqual(np.max(np.abs(metadata["log_adjustment"])), 1.0)
        self.assertGreater(
            metadata["precision_recall_bias_precisions"][1],
            metadata["precision_recall_bias_precisions"][0],
        )

    def test_topk_precision_recall_bias_expanded_modes_are_exported(self):
        expectations = {
            "rank_top2_score_softmax_inner_precision_recall_bias": (
                "rank_top2_score_softmax",
                2,
                0.35,
                0.35,
            ),
            "rank_top2_margin_blend_inner_precision_recall_bias_s25": (
                "rank_top2_margin_blend",
                4,
                0.25,
                0.25,
            ),
            "rank_top3_adaptive_score_softmax_inner_precision_recall_bias_s25": (
                "rank_top3_adaptive_score_softmax",
                3,
                0.25,
                0.25,
            ),
        }
        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:8;2001:6;2002:2",
                "selected_inner_confusion_counts": "",
            }
        ]

        for mode, (base_mode, expected_nonzero, recall_strength, precision_strength) in expectations.items():
            with self.subTest(mode=mode):
                self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
                self.assertEqual(
                    cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
                    base_mode,
                )

                probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
                    scores,
                    score_normalization=mode,
                )
                np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))
                self.assertEqual(np.count_nonzero(probabilities[0]), expected_nonzero)

                metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
                    selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), mode
                )
                self.assertEqual(metadata["inner_mode"], mode)
                self.assertEqual(metadata["precision_recall_bias_status"], "applied")
                self.assertAlmostEqual(metadata["precision_recall_bias_recall_strength"], recall_strength)
                self.assertAlmostEqual(metadata["precision_recall_bias_precision_strength"], precision_strength)
                self.assertGreater(metadata["log_adjustment"][1], metadata["log_adjustment"][0])

    def test_topk_gated_inner_precision_recall_bias_only_adjusts_near_top_classes(self):
        mode = "rank_top3_margin_blend_inner_precision_recall_bias_top2"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_margin_blend",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": (
                    "1001:8;1002:2;2001:6;2002:2;3003:8;4004:8"
                ),
                "selected_inner_confusion_counts": "",
            }
        ]
        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows,
            np.arange(4, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["precision_recall_bias_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k"], 2)

        probabilities = np.asarray(
            [[0.50, 0.30, 0.15, 0.05]],
            dtype=float,
        )
        adjusted = cross_subject._apply_inner_class_prior_balance(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(1))
        self.assertFalse(np.allclose(adjusted, probabilities))
        # Classes 3 and 4 are outside the row's original top-2. They may be
        # rescaled during row renormalization, but no class-specific precision /
        # recall multiplier should be applied to either of them.
        self.assertAlmostEqual(adjusted[0, 2] / adjusted[0, 3], 3.0)
        self.assertEqual(metadata["inner_bias_top_k_status"], "applied")
        self.assertEqual(metadata["inner_bias_top_k_adjusted_trials"], 1)

        guarded_mode = "rank_top3_margin_blend_inner_precision_recall_bias_s25_top2_guarded"
        self.assertIn(guarded_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(guarded_mode),  # pylint: disable=protected-access
            "rank_top3_margin_blend",
        )
        guarded_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(4, dtype=int), np.ones(1, dtype=float), guarded_mode
        )
        self.assertTrue(guarded_metadata["inner_bias_guarded"])
        self.assertEqual(guarded_metadata["inner_bias_top_k"], 2)
        self.assertAlmostEqual(guarded_metadata["precision_recall_bias_recall_strength"], 0.25)
        self.assertAlmostEqual(guarded_metadata["precision_recall_bias_precision_strength"], 0.25)

    def test_inner_precision_recall_bias_strength_variants_are_exported_and_scaled(self):
        weak_mode = "rank_top3_margin_blend_inner_precision_recall_bias_s25"
        default_mode = "rank_top3_margin_blend_inner_precision_recall_bias"
        recall_heavy_mode = "rank_top3_margin_blend_inner_precision_recall_bias_recall_s50"
        precision_heavy_mode = "rank_top3_margin_blend_inner_precision_recall_bias_precision_s50"

        self.assertIn(weak_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn(recall_heavy_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn(precision_heavy_mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(weak_mode),  # pylint: disable=protected-access
            "rank_top3_margin_blend",
        )

        selected_rows = [
            {
                "selected_inner_true_predicted_label_pair_counts": "1001:8;2001:6;2002:2",
                "selected_inner_confusion_counts": "",
            }
        ]
        weak_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), weak_mode
        )
        default_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), default_mode
        )
        recall_heavy_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), recall_heavy_mode
        )
        precision_heavy_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            selected_rows, np.arange(2, dtype=int), np.ones(1, dtype=float), precision_heavy_mode
        )

        self.assertEqual(weak_metadata["inner_mode"], weak_mode)
        self.assertAlmostEqual(weak_metadata["precision_recall_bias_recall_strength"], 0.25)
        self.assertAlmostEqual(weak_metadata["precision_recall_bias_precision_strength"], 0.25)
        self.assertLess(np.ptp(weak_metadata["log_adjustment"]), np.ptp(default_metadata["log_adjustment"]))
        self.assertAlmostEqual(recall_heavy_metadata["precision_recall_bias_recall_strength"], 0.50)
        self.assertAlmostEqual(recall_heavy_metadata["precision_recall_bias_precision_strength"], 0.25)
        self.assertAlmostEqual(precision_heavy_metadata["precision_recall_bias_recall_strength"], 0.25)
        self.assertAlmostEqual(precision_heavy_metadata["precision_recall_bias_precision_strength"], 0.50)
        self.assertEqual(
            recall_heavy_metadata["precision_recall_bias_strength_mode"],
            recall_heavy_mode,
        )

    def test_guarded_test_prior_balance_is_exported_and_margin_gated(self):
        mode = "rank_softmax_guarded_test_prior_balance"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax",
        )
        self.assertEqual(
            cross_subject._test_class_prior_balance_mode(mode),  # pylint: disable=protected-access
            mode,
        )

        probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.60, 0.40],
                [0.55, 0.45],
                [0.30, 0.70],
            ],
            dtype=float,
        )
        metadata = cross_subject._test_class_prior_balance_metadata(  # pylint: disable=protected-access
            probabilities,
            np.arange(2, dtype=int),
            mode,
        )
        adjusted = cross_subject._apply_test_class_prior_balance(probabilities, metadata)  # pylint: disable=protected-access

        self.assertEqual(metadata["mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertEqual(metadata["guarded_margin_quantile"], 0.5)
        self.assertEqual(metadata["guarded_adjusted_trials"], 2)
        np.testing.assert_allclose(np.sum(adjusted, axis=1), np.ones(4))
        np.testing.assert_allclose(adjusted[0], probabilities[0])
        self.assertFalse(np.allclose(adjusted[2], probabilities[2]))

    def test_margin_gated_inner_confusion_leaves_high_margin_rows_unchanged(self):
        probabilities = np.asarray([[0.51, 0.49], [0.95, 0.05]], dtype=float)
        metadata = {
            "mode": "rank_softmax_inner_confusion_margin_soft",
            "true_given_predicted": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            "blend": 1.0,
            "margin_gated": True,
            "margin_quantile": 0.5,
        }

        adjusted = cross_subject._apply_inner_confusion_correction(  # pylint: disable=protected-access
            probabilities,
            metadata,
        )

        np.testing.assert_allclose(adjusted[0], np.asarray([0.49, 0.51]))
        np.testing.assert_allclose(adjusted[1], probabilities[1])
        self.assertAlmostEqual(metadata["margin_threshold"], 0.46)

    def test_sensor_flat_delta2_feature_mode_is_exported_and_channel_blocked(self):
        mode = "sensor_flat_delta2"

        self.assertIn(mode, cross_subject.FEATURE_MODES)
        self.assertEqual(
            cross_subject._normalize_feature_mode("sensor-flat-delta2"),  # pylint: disable=protected-access
            mode,
        )

        signal = np.asarray([[1.0, 2.0, 4.0, 7.0], [0.0, 1.0, 1.0, 2.0]], dtype=float)
        feature = next_hooks._sensor_flat_delta2_feature(signal)  # pylint: disable=protected-access
        expected = np.concatenate(
            (
                signal.reshape(-1, order="F"),
                np.diff(signal, axis=1).reshape(-1, order="F"),
                np.diff(signal, n=2, axis=1).reshape(-1, order="F"),
            )
        )
        np.testing.assert_allclose(feature, expected)
        self.assertEqual(feature.shape[0] % signal.shape[0], 0)

    def test_trial_margin_ensemble_weighting_is_exported_and_trialwise(self):
        mode = "inner_lcb_trial_margin_softmax"

        self.assertIn(mode, cross_subject.SELECTION_ENSEMBLE_WEIGHTING_MODES)

        score_matrices = [
            np.asarray([[5.0, 4.0], [0.0, 0.0]], dtype=float),
            np.asarray([[0.0, 0.0], [5.0, 4.0]], dtype=float),
        ]
        trial_weights = cross_subject._trial_margin_ensemble_weights(  # pylint: disable=protected-access
            score_matrices,
            np.asarray([0.5, 0.5], dtype=float),
            mode,
        )

        self.assertEqual(trial_weights.shape, (2, 2))
        np.testing.assert_allclose(np.sum(trial_weights, axis=0), np.ones(2))
        self.assertGreater(trial_weights[0, 0], trial_weights[1, 0])
        self.assertGreater(trial_weights[1, 1], trial_weights[0, 1])

        probabilities = np.stack(
            [np.eye(2, dtype=float), np.flipud(np.eye(2, dtype=float))],
            axis=0,
        )
        pooled = cross_subject._weighted_probability_pool(probabilities, trial_weights)  # pylint: disable=protected-access
        self.assertEqual(pooled.shape, (2, 2))

    def test_inner_score_calibration_uses_next_candidate_scoring_path(self):
        feature_sets = tuple(
            SimpleNamespace(
                participant=participant,
                labels=np.asarray([1, 2, 1, 2], dtype=int),
            )
            for participant in (1, 2, 3)
        )
        config = CrossSubjectStimulusConfig(
            chance_classes=2,
            score_calibration="inner_class_bias",
        )
        seen_validation_participants = []

        def fake_fit_outer_fold_model(
            *_args,
            **_kwargs,
        ):
            return {
                "model_bundle": object(),
                "score_calibration_metadata": {"mode": "none"},
            }

        def fake_candidate_scores(_model, validation_set, _config):
            seen_validation_participants.append(int(validation_set.participant))
            labels = np.asarray(validation_set.labels, dtype=int) - 1
            scores = np.full((labels.shape[0], 2), -1.0, dtype=float)
            scores[np.arange(labels.shape[0]), labels] = 1.0
            return scores, np.arange(2, dtype=int)

        with (
            patch.object(
                next_hooks,
                "_fit_outer_fold_model",
                side_effect=fake_fit_outer_fold_model,
            ),
            patch.object(
                next_hooks,
                "_candidate_model_scores",
                side_effect=fake_candidate_scores,
            ),
            patch.object(
                next_hooks,
                "_previous_candidate_model_scores",
                side_effect=AssertionError("legacy score path used"),
            ),
        ):
            metadata = next_hooks._fit_inner_score_calibration(  # pylint: disable=protected-access
                feature_sets,
                config,
                1.0,
            )

        self.assertEqual(seen_validation_participants, [1, 2, 3])
        self.assertEqual(metadata["mode"], "inner_class_bias")
        self.assertIn("inner_balanced_accuracy", metadata)

    def test_subunit_rank_softmax_temperatures_are_exported(self):
        self.assertIn("rank_softmax_t0_5", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn("rank_softmax_t0_75", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        default = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )
        sharp = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax_t0_75",
        )
        sharper = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax_t0_5",
        )

        np.testing.assert_allclose(np.sum(default, axis=1), np.ones(1))
        np.testing.assert_allclose(np.sum(sharp, axis=1), np.ones(1))
        np.testing.assert_allclose(np.sum(sharper, axis=1), np.ones(1))
        self.assertGreater(sharp[0, 0], default[0, 0])
        self.assertGreater(sharper[0, 0], sharp[0, 0])

    def test_intermediate_rank_softmax_log_pool_modes_are_exported(self):
        mode = "rank_softmax_t0_75_log_pool"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t0_75",
        )
        self.assertEqual(
            cross_subject._ensemble_log_pool_mode(mode),  # pylint: disable=protected-access
            mode,
        )

        scores = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=float)
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores, score_normalization=mode
        )
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(1))

    def test_agreement_log_pool_is_exported_and_agreement_gated(self):
        mode = "rank_softmax_agreement_log_pool"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax",
        )

        probability_matrices = [
            np.asarray(
                [[0.70, 0.20, 0.10], [0.70, 0.20, 0.10]],
                dtype=float,
            ),
            np.asarray(
                [[0.60, 0.30, 0.10], [0.20, 0.70, 0.10]],
                dtype=float,
            ),
            np.asarray(
                [[0.65, 0.25, 0.10], [0.20, 0.10, 0.70]],
                dtype=float,
            ),
        ]
        weights = np.full(3, 1.0 / 3.0, dtype=float)

        legacy = cross_subject._pool_ensemble_probability_matrices(  # pylint: disable=protected-access
            probability_matrices,
            weights,
            "rank_softmax",
        )
        agreement = cross_subject._pool_ensemble_probability_matrices(  # pylint: disable=protected-access
            probability_matrices,
            weights,
            mode,
        )

        np.testing.assert_allclose(np.sum(agreement, axis=1), np.ones(2))
        self.assertGreater(agreement[0, 0], legacy[0, 0])
        np.testing.assert_allclose(agreement[1], legacy[1])

    def test_subunit_rank_softmax_soft_guarded_inner_confusion_modes_are_exported(self):
        mode = "rank_softmax_t0_75_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t0_75",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertLess(metadata["blend"], 1.0)

    def test_intermediate_rank_softmax_inner_confusion_margin_modes_are_exported(self):
        mode = "rank_softmax_t1_5_inner_confusion_margin_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:3;1002:2;2002:4"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertTrue(metadata["margin_gated"])
        self.assertEqual(metadata["margin_quantile"], 0.5)
        self.assertLess(metadata["blend"], 1.0)

    def test_intermediate_rank_softmax_inner_prior_modes_are_exported(self):
        mode = "rank_softmax_t1_5_inner_balanced"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5",
        )

        metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_test_label_counts": "1:8;2:8",
                    "selected_inner_predicted_label_counts": "1:12;2:4",
                }
            ],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertEqual(metadata["log_adjustment"].shape, (2,))
        self.assertGreater(metadata["log_adjustment"][1], metadata["log_adjustment"][0])

    def test_intermediate_rank_softmax_inner_prior_confusion_modes_are_exported(self):
        mode = "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5",
        )

        prior_metadata = cross_subject._inner_class_prior_balance_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_test_label_counts": "1:8;2:8",
                    "selected_inner_predicted_label_counts": "1:12;2:4",
                    "selected_inner_true_predicted_label_pair_counts": "1001:3;1002:2;2002:4",
                }
            ],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        confusion_metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [
                {
                    "selected_inner_test_label_counts": "1:8;2:8",
                    "selected_inner_predicted_label_counts": "1:12;2:4",
                    "selected_inner_true_predicted_label_pair_counts": "1001:3;1002:2;2002:4",
                }
            ],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(prior_metadata["inner_mode"], mode)
        self.assertEqual(confusion_metadata["inner_mode"], mode)
        self.assertTrue(confusion_metadata["guarded"])
        self.assertLess(confusion_metadata["blend"], 1.0)

    def test_guarded_quota_wraps_intermediate_source_inner_modes(self):
        mode = "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded_guarded_balanced_quota"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5",
        )
        self.assertEqual(
            cross_subject._inner_class_prior_balance_mode(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded",
        )
        self.assertEqual(
            cross_subject._inner_confusion_correction_mode(mode),  # pylint: disable=protected-access
            "rank_softmax_t1_5_inner_balanced_confusion_soft_guarded",
        )

        quota_metadata = cross_subject._balanced_quota_metadata(  # pylint: disable=protected-access
            np.asarray(
                [
                    [0.90, 0.10],
                    [0.80, 0.20],
                    [0.40, 0.60],
                    [0.30, 0.70],
                ],
                dtype=float,
            ),
            np.arange(2, dtype=int),
            mode,
        )

        self.assertEqual(quota_metadata["mode"], mode)
        self.assertEqual(quota_metadata["status"], "applied_guarded")
        np.testing.assert_array_equal(quota_metadata["quota_counts"], np.asarray([2, 2]))

    def test_worst_class_selection_metrics_are_exported(self):
        self.assertIn("balanced_worst_class", cross_subject.CROSS_SUBJECT_SELECTION_METRIC_CHOICES)
        self.assertIn("balanced_worst_class_lcb", cross_subject.CROSS_SUBJECT_SELECTION_METRIC_CHOICES)

    def test_worst_class_lcb_selection_prefers_less_collapsed_class(self):
        inner_rows = [
            self._inner_candidate_row(
                candidate_index=1,
                balanced_accuracy=0.80,
                confusion_counts="1>1:10;2>1:8;2>2:2",
            ),
            self._inner_candidate_row(
                candidate_index=2,
                balanced_accuracy=0.78,
                confusion_counts="1>1:8;1>2:2;2>1:2;2>2:8",
            ),
        ]

        balanced_ranked = cross_subject._rank_nested_candidates(  # pylint: disable=protected-access
            inner_rows,
            selection_metric="balanced_accuracy",
        )
        worst_class_ranked = cross_subject._rank_nested_candidates(  # pylint: disable=protected-access
            inner_rows,
            selection_metric="balanced_worst_class_lcb",
        )

        self.assertEqual(balanced_ranked[0]["selected_candidate_index"], 1)
        self.assertEqual(worst_class_ranked[0]["selected_candidate_index"], 2)
        self.assertAlmostEqual(
            worst_class_ranked[0]["selected_inner_worst_class_recall"],
            0.8,
        )
        self.assertEqual(worst_class_ranked[0]["selection_metric"], "balanced_worst_class_lcb")

    @staticmethod
    def _inner_candidate_row(*, candidate_index, balanced_accuracy, confusion_counts):
        return {
            "outer_test_participant": 1,
            "inner_validation_participant": 2,
            "candidate_index": candidate_index,
            "balanced_accuracy": balanced_accuracy,
            "accuracy": balanced_accuracy,
            "top2_accuracy": min(1.0, balanced_accuracy + 0.1),
            "top3_accuracy": min(1.0, balanced_accuracy + 0.2),
            "mean_true_label_rank": 1.5,
            "chance_mean_rank": 1.5,
            "chance_classes": 2,
            "test_label_counts": "1:10;2:10",
            "predicted_label_counts": "1:10;2:10",
            "true_predicted_label_pair_counts": "",
            "confusion_counts": confusion_counts,
            "window_center_s": 0.175,
            "window_size_s": 0.1,
            "window_start_s": 0.125,
            "window_stop_s": 0.225,
            "feature_mode": "sensor_flat",
            "normalization": "subject_baseline_whiten",
            "alignment": "none",
            "classifier": "multinomial-logistic",
            "classifier_param": 1.0,
            "components_pca": 128,
            "max_trials_per_class_per_participant": "",
            "label_shuffle_control": False,
            "label_shuffle_seed": "",
        }

    def test_topk_borda_score_normalizations_are_exported(self):
        self.assertIn("rank_top2_borda", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn("rank_top3_borda", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)

        scores = np.asarray(
            [
                [4.0, 3.0, 2.0, 1.0],
                [0.0, 5.0, 4.0, 3.0],
            ],
            dtype=float,
        )

        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_top3_borda",
        )

        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
        np.testing.assert_allclose(probabilities[0], np.asarray([3.0, 2.0, 1.0, 0.0]) / 6.0)
        np.testing.assert_allclose(probabilities[1], np.asarray([0.0, 3.0, 2.0, 1.0]) / 6.0)

    def test_topk_margin_blend_score_normalizations_are_exported(self):
        self.assertIn("rank_top2_margin_blend", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertIn("rank_top3_margin_blend", cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)

        scores = np.asarray(
            [
                [4.00, 3.95, 3.90, 0.00],
                [4.00, 1.00, 0.00, -1.00],
            ],
            dtype=float,
        )

        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_top3_margin_blend",
        )

        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertEqual(int(np.argmax(probabilities[0])), 0)
        self.assertEqual(int(np.argmax(probabilities[1])), 0)
        self.assertGreater(probabilities[0, 1], probabilities[1, 1])
        self.assertGreater(probabilities[1, 0], probabilities[0, 0])

    def test_topk_margin_blend_soft_guarded_inner_confusion_mode_is_exported(self):
        mode = "rank_top3_margin_blend_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_top3_margin_blend",
        )

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )
        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertLess(metadata["blend"], 1.0)

    def test_adaptive_rank_softmax_is_exported_and_margin_sensitive(self):
        mode = "rank_adaptive_softmax"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            mode,
        )

        scores = np.asarray(
            [
                [4.0, 3.95, 2.0, 1.0],
                [4.0, 0.0, -1.0, -2.0],
            ],
            dtype=float,
        )
        adaptive = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )
        default = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization="rank_softmax",
        )

        np.testing.assert_allclose(np.sum(adaptive, axis=1), np.ones(2))
        self.assertLess(adaptive[0, 0], default[0, 0])
        self.assertGreater(adaptive[1, 0], default[1, 0])

    def test_adaptive_rank_softmax_soft_guarded_inner_confusion_mode_is_exported(self):
        mode = "rank_adaptive_softmax_inner_confusion_soft_guarded"

        self.assertIn(mode, cross_subject.ENSEMBLE_SCORE_NORMALIZATION_MODES)
        self.assertEqual(
            cross_subject._base_ensemble_score_normalization(mode),  # pylint: disable=protected-access
            "rank_adaptive_softmax",
        )

        scores = np.asarray(
            [
                [4.0, 3.95, 2.0, 1.0],
                [4.0, 0.0, -1.0, -2.0],
            ],
            dtype=float,
        )
        probabilities = cross_subject._class_score_probabilities(  # pylint: disable=protected-access
            scores,
            score_normalization=mode,
        )
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))

        metadata = cross_subject._inner_confusion_correction_metadata(  # pylint: disable=protected-access
            [{"selected_inner_true_predicted_label_pair_counts": "1001:4;1002:2;2002:5"}],
            np.arange(2, dtype=int),
            np.ones(1, dtype=float),
            mode,
        )

        self.assertEqual(metadata["inner_mode"], mode)
        self.assertTrue(metadata["guarded"])
        self.assertFalse(metadata["margin_gated"])
        self.assertLess(metadata["blend"], 1.0)

    def test_extended_feature_modes_are_exported(self):
        self.assertIn("sensor_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_mean_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_smooth", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_taper", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_gaussian_taper", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_centered", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_dct", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_pyramid", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_pyramid_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_pyramid_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_bins3", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_flat_time_bins5", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_bandpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_cov_tangent", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_logpower", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_delta", cross_subject.FEATURE_MODES)
        self.assertIn("sensor_time_pyramid_delta_logpower", cross_subject.FEATURE_MODES)

    def test_sensor_mean_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 4))
        np.testing.assert_allclose(feature_set.features[0, :2], np.asarray([2.0, 4.0]))
        np.testing.assert_allclose(feature_set.features[1, :2], np.asarray([3.0, 6.0]))
        self.assertTrue(np.all(np.isfinite(feature_set.features[:, 2:])))

    def test_sensor_flat_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 2.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_flat_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(feature_set.features[0, :4], np.asarray([1.0, 2.0, 3.0, 6.0]))
        np.testing.assert_allclose(feature_set.features[0, 4:], np.log(np.asarray([5.0, 20.0]) + 1e-12))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_smooth_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_smooth",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([1.5, 2.5, 3.0, 4.0, 4.5, 5.5]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_taper_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_taper",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([0.25, 0.50, 3.00, 4.00, 1.25, 1.50]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_bins_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
        trials = [
            [
                [0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
                [0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            ],
            [
                [0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                [0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
            ],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.3,
            window_size=0.4,
            feature_mode="sensor_flat_time_bins3",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_cross_subject.sio.loadmat",
            side_effect=loadmat_side_effect(data_by_participant),
        ):
            feature_set = load_participant_stimulus_features(
                "unused", 1, config=config
            )

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.n_window_samples, 5)
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([2.0, 3.0, 6.0, 7.0, 9.0, 10.0]),
        )
        np.testing.assert_allclose(
            feature_set.features[1],
            np.asarray([3.0, 2.0, 7.0, 6.0, 10.0, 9.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_gaussian_taper_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_gaussian_taper",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        weights = cross_subject._sensor_flat_gaussian_taper_weights(3)  # pylint: disable=protected-access
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * np.repeat(weights, 2)
        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(feature_set.features[0], expected)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_centered_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_centered",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_centered_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0, 5.0], [0.4, 0.5, 0.6, 2.0, 4.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0, 6.0], [0.5, 0.6, 0.7, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_centered",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(feature_set.baseline_feature_std > 0.0))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_taper_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0, 5.0], [0.4, 0.5, 0.6, 2.0, 4.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0, 6.0], [0.5, 0.6, 0.7, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_taper",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(feature_set.baseline_feature_std > 0.0))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_gaussian_taper_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0, 5.0], [0.4, 0.5, 0.6, 2.0, 4.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0, 6.0], [0.5, 0.6, 0.7, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_gaussian_taper",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(feature_set.baseline_feature_std > 0.0))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_smooth_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0, 5.0], [0.4, 0.5, 0.6, 2.0, 4.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0, 6.0], [0.5, 0.6, 0.7, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_smooth",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_dct_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 6.0], [0.0, 0.0, 2.0, 5.0, 9.0]],
            [[0.0, 0.0, 2.0, 4.0, 7.0], [0.0, 0.0, 4.0, 7.0, 11.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_dct",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 22))
        np.testing.assert_allclose(
            feature_set.features[0, :6],
            np.asarray([1.0, 2.0, 3.0, 5.0, 6.0, 9.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_dct_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_dct",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 20))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 20))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 20))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 6.0], [0.0, 0.0, 2.0, 5.0, 9.0]],
            [[0.0, 0.0, 2.0, 4.0, 7.0], [0.0, 0.0, 4.0, 7.0, 11.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 10))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 2.0, 3.0, 3.0, 4.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_delta_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.1, 0.2, 0.3, 1.0, 3.0, 6.0], [0.0, 0.4, 0.5, 0.6, 2.0, 5.0, 9.0]],
            [[0.0, 0.2, 0.3, 0.4, 2.0, 4.0, 7.0], [0.0, 0.5, 0.6, 0.7, 4.0, 7.0, 11.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_delta",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 10))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 10))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 10))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_flat_time_pyramid",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 22))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 4.0, 6.0]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_flat_time_pyramid_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 28))
        np.testing.assert_allclose(
            feature_set.features[0, :6],
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_delta_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_time_pyramid_delta",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 26))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 26))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 26))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_time_pyramid_logpower_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_time_pyramid_logpower",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 20))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 20))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 20))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_flat_logpower_baseline_whiten_allows_different_baseline_duration(self):
        time = np.asarray([-0.3, -0.2, -0.1, 0.1, 0.2], dtype=float)
        trials = [
            [[0.1, 0.2, 0.3, 1.0, 3.0], [0.4, 0.5, 0.6, 2.0, 6.0]],
            [[0.2, 0.3, 0.4, 2.0, 4.0], [0.5, 0.6, 0.7, 4.0, 8.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            baseline_window=(-0.3, -0.1),
            feature_mode="sensor_flat_logpower",
            normalization="subject_baseline_whiten",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertEqual(feature_set.baseline_feature_mean.shape, (1, 6))
        self.assertEqual(feature_set.baseline_feature_std.shape, (1, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_cov_tangent_feature_width(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 0.0, 1.0, 3.0, 5.0], [0.0, 0.0, 2.0, 6.0, 8.0], [0.0, 0.0, 3.0, 1.0, 0.5]],
            [[0.0, 0.0, 2.0, 4.0, 6.0], [0.0, 0.0, 4.0, 8.0, 9.0], [0.0, 0.0, 1.0, 3.0, 5.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.2,
            feature_mode="sensor_cov_tangent",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(feature_set.features)))

    def test_sensor_time_pyramid_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
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

    def test_sensor_time_pyramid_delta_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid_delta",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 22))
        np.testing.assert_allclose(
            feature_set.features[0],
            np.asarray([
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                4.0, 4.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ]),
        )

    def test_sensor_time_pyramid_logpower_feature(self):
        time = np.asarray([-0.5, 0.0, 0.1, 0.2, 0.3], dtype=float)
        trials = [
            [[0.0, 1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]],
            [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 3.0, 5.0, 7.0]],
        ]
        data_by_participant = {1: mat_data_from_trials([1, 2], trials, time)}
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.3,
            feature_mode="sensor_time_pyramid_logpower",
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (2, 16))
        np.testing.assert_allclose(
            feature_set.features[0, :14],
            np.asarray([
                4.0, 5.0,
                2.0, 3.0, 6.0, 7.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ]),
        )
        np.testing.assert_allclose(feature_set.features[0, 14:], np.log(np.asarray([21.0, 30.0]) + 1e-12))

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

    def test_candidate_grid_expands_window_sizes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.150, 0.175),
            window_sizes=(0.125, 0.150),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            alignments=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(config.window_center, config.window_size) for config in configs},
            {
                (0.150, 0.125),
                (0.150, 0.150),
                (0.175, 0.125),
                (0.175, 0.150),
            },
        )

    def test_candidate_grid_accepts_inner_class_affine_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_class_affine",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].score_calibration, "inner_class_affine")

    def test_candidate_grid_accepts_rank_bias_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_rank_bias", "train_rank_bias"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_rank_bias", "train_rank_bias"},
        )

    def test_candidate_grid_accepts_inner_probability_map_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_probability_map", "inner_rank_probability_map"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_probability_map", "inner_rank_probability_map"},
        )

    def test_candidate_grid_accepts_inner_confusion_blend_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_confusion_blend",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].score_calibration, "inner_confusion_blend")

    def test_candidate_grid_accepts_inner_margin_confusion_blend_score_calibration(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("inner_margin_confusion_blend",),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(
            configs[0].score_calibration,
            "inner_margin_confusion_blend",
        )

    def test_candidate_grid_accepts_rank_confusion_blend_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=(
                "inner_rank_confusion_blend",
                "inner_rank_margin_confusion_blend",
                "inner_rank_confusion_blend_guarded",
            ),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 3)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {
                "inner_rank_confusion_blend",
                "inner_rank_margin_confusion_blend",
                "inner_rank_confusion_blend_guarded",
            },
        )

    def test_candidate_grid_accepts_guarded_inner_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=(
                "inner_class_bias_guarded",
                "inner_probability_map_guarded",
            ),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"inner_class_bias_guarded", "inner_probability_map_guarded"},
        )

    def test_candidate_grid_accepts_train_score_calibration_modes(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            feature_modes=("sensor_mean",),
            normalizations=("none",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_pca_values=(64,),
            score_calibrations=("train_class_bias", "train_class_affine"),
            chance_classes=2,
        )

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config.score_calibration for config in configs},
            {"train_class_bias", "train_class_affine"},
        )

    def test_inner_class_bias_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_class_bias",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_class_bias")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        np.testing.assert_allclose(metadata["scale"], np.ones(2))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_class_affine_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_class_affine",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_class_affine")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        self.assertTrue(np.all(metadata["scale"] > 0.0))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_probability_map_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_probability_map",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_probability_map")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["probability_map"].shape, (2, 2))
        self.assertTrue(np.all(metadata["probability_map"] >= 0.0))
        np.testing.assert_allclose(np.sum(metadata["probability_map"], axis=1), np.ones(2))
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_inner_confusion_blend_score_calibration_fits_source_fold_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
            3: mat_data_from_trials(labels, [[[-1.2, -1.2, -1.2, -1.2]], [[1.2, 1.2, 1.2, 1.2]], [[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="inner_confusion_blend",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2, 3)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "inner_confusion_blend")
        self.assertEqual(metadata["calibration_source"], "inner_scores")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["confusion_matrix"].shape, (2, 2))
        np.testing.assert_allclose(np.sum(metadata["confusion_matrix"], axis=1), np.ones(2))
        self.assertGreaterEqual(metadata["blend_alpha"], 0.0)
        self.assertLessEqual(metadata["blend_alpha"], 1.0)
        self.assertTrue(np.isfinite(metadata["inner_balanced_accuracy"]))

    def test_train_class_bias_score_calibration_fits_source_metadata(self):
        time = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
        labels = [1, 2, 1, 2]
        data_by_participant = {
            1: mat_data_from_trials(labels, [[[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]], [[-0.9, -0.9, -0.9, -0.9]], [[0.9, 0.9, 0.9, 0.9]]], time),
            2: mat_data_from_trials(labels, [[[-1.1, -1.1, -1.1, -1.1]], [[1.1, 1.1, 1.1, 1.1]], [[-1.0, -1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0, 1.0]]], time),
        }
        config = CrossSubjectStimulusConfig(
            window_center=0.15,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multinomial-logistic",
            classifier_param=1.0,
            components_pca=float("inf"),
            score_calibration="train_class_bias",
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat_side_effect(data_by_participant)):
            train_sets = [load_participant_stimulus_features("unused", participant, config=config) for participant in (1, 2)]

        fitted_model = cross_subject._fit_outer_fold_model(train_sets, config, 1.0)  # pylint: disable=protected-access
        metadata = fitted_model["score_calibration_metadata"]

        self.assertEqual(metadata["mode"], "train_class_bias")
        self.assertEqual(metadata["calibration_source"], "train_scores")
        np.testing.assert_array_equal(metadata["classes"], np.asarray([0, 1]))
        self.assertEqual(metadata["bias"].shape, (2,))
        self.assertEqual(metadata["scale"].shape, (2,))
        np.testing.assert_allclose(metadata["scale"], np.ones(2))
        self.assertTrue(np.isfinite(metadata["source_balanced_accuracy"]))

    def test_inner_class_affine_score_calibration_applies_scale_and_bias(self):
        scores = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_class_affine",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.5, -0.5]),
                "scale": np.asarray([2.0, 0.25]),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_allclose(calibrated, np.asarray([[2.5, 0.0], [6.5, 0.5]]))

    def test_rank_bias_score_calibration_applies_bias_to_rank_scores(self):
        scores = np.asarray([[10.0, 1.0], [0.0, 3.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_bias",
                "score_space": "rank",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.0, 2.0]),
                "scale": np.ones(2),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_allclose(calibrated, np.asarray([[0.0, 1.0], [-1.0, 2.0]]))

    def test_inner_probability_map_score_calibration_applies_probability_map(self):
        scores = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_probability_map",
                "classes": np.asarray([0, 1]),
                "probability_map": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_array_equal(np.argmax(calibrated, axis=1), np.asarray([1, 0]))
        np.testing.assert_allclose(np.sum(np.exp(calibrated), axis=1), np.ones(2))

    def test_rank_probability_map_score_calibration_uses_rank_space(self):
        scores = np.asarray([[100.0, 99.0, 0.0], [1.0, 0.0, 100.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_probability_map",
                "score_space": "rank",
                "classes": np.asarray([0, 1, 2]),
                "probability_map": np.eye(3, dtype=float),
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1, 2]),
            fitted_model,
        )

        rank_scores = np.asarray(
            [[0.0, -1.0, -2.0], [-1.0, -2.0, 0.0]], dtype=float
        )
        centered = rank_scores - np.mean(rank_scores, axis=1, keepdims=True)
        centered /= np.std(centered, axis=1, keepdims=True)
        probabilities = np.exp(centered - np.max(centered, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        expected = np.log(probabilities)

        np.testing.assert_array_equal(classes, np.asarray([0, 1, 2]))
        np.testing.assert_allclose(calibrated, expected)

    def test_inner_confusion_blend_score_calibration_applies_confusion_matrix(self):
        scores = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_confusion_blend",
                "classes": np.asarray([0, 1]),
                "confusion_matrix": np.asarray(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=float
                ),
                "blend_alpha": 1.0,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        self.assertGreater(calibrated[0, 1], calibrated[0, 0])
        self.assertGreater(calibrated[1, 0], calibrated[1, 1])

    def test_rank_confusion_blend_score_calibration_uses_rank_space(self):
        scores = np.asarray([[100.0, 99.0, 0.0], [1.0, 0.0, 100.0]], dtype=float)
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_rank_confusion_blend",
                "score_space": "rank",
                "classes": np.asarray([0, 1, 2]),
                "confusion_matrix": np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                ),
                "blend_alpha": 1.0,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1, 2]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(np.argmax(calibrated, axis=1), np.asarray([1, 2]))

    def test_inner_margin_confusion_blend_only_reranks_low_margin_trials(self):
        scores = np.asarray(
            [[0.99, 0.01], [0.51, 0.49], [0.01, 0.99]],
            dtype=float,
        )
        fitted_model = {
            "score_calibration_metadata": {
                "mode": "inner_margin_confusion_blend",
                "classes": np.asarray([0, 1]),
                "confusion_matrix": np.asarray(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=float
                ),
                "blend_alpha": 1.0,
                "margin_threshold": 0.1,
            }
        }

        calibrated, classes = cross_subject._apply_score_calibration(  # pylint: disable=protected-access
            scores,
            np.asarray([0, 1]),
            fitted_model,
        )

        np.testing.assert_array_equal(classes, np.asarray([0, 1]))
        np.testing.assert_array_equal(
            np.argmax(calibrated, axis=1),
            np.asarray([0, 1, 1]),
        )

    def test_guarded_inner_score_calibration_skips_without_inner_gain(self):
        metadata = next_hooks._guard_inner_score_calibration_metadata(  # pylint: disable=protected-access
            {
                "mode": "inner_class_bias_guarded",
                "classes": np.asarray([0, 1]),
                "bias": np.asarray([0.5, -0.5]),
                "scale": np.ones(2),
                "inner_balanced_accuracy": 0.5,
                "calibration_source": "inner_scores",
            },
            0.5,
            guarded=True,
        )

        self.assertEqual(metadata["status"], "skipped_no_inner_gain")
        self.assertEqual(metadata["inner_uncalibrated_balanced_accuracy"], 0.5)
        self.assertNotIn("bias", metadata)
        self.assertFalse(next_hooks._has_active_score_calibration_metadata(metadata))  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
