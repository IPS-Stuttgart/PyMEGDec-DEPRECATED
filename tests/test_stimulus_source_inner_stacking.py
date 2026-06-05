from __future__ import annotations

import unittest

import numpy as np

from pymegdec import stimulus_source_inner_stacking as stacking


class SourceInnerStackingTests(unittest.TestCase):
    def test_align_columns_can_fill_missing_probability_classes(self):
        scores = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)
        aligned = stacking._align_columns(  # pylint: disable=protected-access
            scores,
            score_classes=(1, 3),
            class_order=np.asarray([1, 2, 3], dtype=int),
            fill_value=0.0,
        )

        np.testing.assert_allclose(
            aligned,
            np.asarray([[0.7, 0.0, 0.3], [0.2, 0.0, 0.8]], dtype=float),
        )

    def test_align_columns_can_fill_missing_logit_classes_below_row_min(self):
        scores = np.asarray([[2.0, -4.0], [-3.0, -2.0]], dtype=float)
        aligned = stacking._align_columns(  # pylint: disable=protected-access
            scores,
            score_classes=(1, 3),
            class_order=np.asarray([1, 2, 3], dtype=int),
            fill_value="row_min",
        )

        np.testing.assert_allclose(
            aligned,
            np.asarray([[2.0, -5.0, -4.0], [-3.0, -4.0, -2.0]], dtype=float),
        )

    def test_scalar_stacker_selects_latent_when_source_inner_latent_is_better(self):
        labels = np.asarray([1, 2, 3, 1, 2, 3], dtype=int)
        class_order = np.asarray([1, 2, 3], dtype=int)
        compact_probabilities = np.asarray(
            [
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
            ],
            dtype=float,
        )
        latent_logits = np.asarray(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=float,
        )

        selected_weight, selected_temperature, selected_score_mode, rows = stacking._fit_scalar_stacker(  # pylint: disable=protected-access
            compact_probabilities,
            latent_logits,
            labels,
            class_order,
            weight_grid=(0.0, 1.0),
            latent_temperature_grid=(1.0,),
            mode="compact_probability_latent_logit",
            chance_classes=3,
        )

        self.assertEqual(selected_weight, 0.0)
        self.assertEqual(selected_temperature, 1.0)
        self.assertEqual(selected_score_mode, "compact_probability_latent_logit")
        self.assertEqual(max(row["balanced_accuracy"] for row in rows), 1.0)

    def test_scalar_stacker_selects_compact_when_source_inner_compact_is_better(self):
        labels = np.asarray([1, 2, 3, 1, 2, 3], dtype=int)
        class_order = np.asarray([1, 2, 3], dtype=int)
        compact_probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=float,
        )
        latent_logits = np.asarray(
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        selected_weight, selected_temperature, selected_score_mode, rows = stacking._fit_scalar_stacker(  # pylint: disable=protected-access
            compact_probabilities,
            latent_logits,
            labels,
            class_order,
            weight_grid=(0.0, 1.0),
            latent_temperature_grid=(1.0,),
            mode="compact_probability_latent_logit",
            chance_classes=3,
        )

        self.assertEqual(selected_weight, 1.0)
        self.assertEqual(selected_temperature, 1.0)
        self.assertEqual(selected_score_mode, "compact_probability_latent_logit")
        self.assertEqual(max(row["balanced_accuracy"] for row in rows), 1.0)

    def test_scalar_stacker_selects_latent_temperature_from_source_inner_grid(self):
        labels = np.asarray([1, 2, 1, 2], dtype=int)
        class_order = np.asarray([1, 2], dtype=int)
        compact_probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.10, 0.90],
                [0.90, 0.10],
                [0.10, 0.90],
            ],
            dtype=float,
        )
        latent_logits = np.asarray(
            [
                [0.0, 3.0],
                [3.0, 0.0],
                [0.0, 3.0],
                [3.0, 0.0],
            ],
            dtype=float,
        )

        selected_weight, selected_temperature, selected_score_mode, rows = stacking._fit_scalar_stacker(  # pylint: disable=protected-access
            compact_probabilities,
            latent_logits,
            labels,
            class_order,
            weight_grid=(0.5,),
            latent_temperature_grid=(0.5, 5.0),
            mode="compact_logprob_latent_logit",
            chance_classes=2,
        )

        self.assertEqual(selected_weight, 0.5)
        self.assertEqual(selected_temperature, 5.0)
        self.assertEqual(selected_score_mode, "compact_logprob_latent_logit")
        selected = next(row for row in rows if row["latent_temperature"] == selected_temperature)
        self.assertEqual(selected["balanced_accuracy"], 1.0)

    def test_scalar_stacker_can_select_rank_safe_fallback_over_raw_logits(self):
        labels = np.asarray([1, 2, 1, 2], dtype=int)
        class_order = np.asarray([1, 2], dtype=int)
        compact_probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.10, 0.90],
                [0.90, 0.10],
                [0.10, 0.90],
            ],
            dtype=float,
        )
        latent_logits = np.asarray(
            [
                [0.0, 100.0],
                [100.0, 0.0],
                [0.0, 100.0],
                [100.0, 0.0],
            ],
            dtype=float,
        )

        selected_weight, selected_temperature, selected_score_mode, rows = stacking._fit_scalar_stacker(  # pylint: disable=protected-access
            compact_probabilities,
            latent_logits,
            labels,
            class_order,
            weight_grid=(0.5,),
            latent_temperature_grid=(1.0,),
            score_modes=("raw_logit_mix", "z_softmax_mix", "rank_softmax_mix"),
            chance_classes=2,
        )

        self.assertEqual(selected_weight, 0.5)
        self.assertEqual(selected_temperature, 1.0)
        self.assertEqual(selected_score_mode, "z_softmax_mix")
        by_mode = {row["stacker_score_mode"]: row["balanced_accuracy"] for row in rows}
        self.assertEqual(by_mode["raw_logit_mix"], 0.0)
        self.assertEqual(by_mode["z_softmax_mix"], 1.0)

    def test_validate_block_alignment_rejects_mismatched_trials(self):
        block = stacking.ScoreBlock(
            scores=np.ones((2, 2)),
            labels=np.asarray([1, 2], dtype=int),
            class_order=np.asarray([1, 2], dtype=int),
            trial_indices=np.asarray([0, 1], dtype=int),
        )
        mismatched = stacking.ScoreBlock(
            scores=np.ones((2, 2)),
            labels=np.asarray([1, 2], dtype=int),
            class_order=np.asarray([1, 2], dtype=int),
            trial_indices=np.asarray([0, 2], dtype=int),
        )

        with self.assertRaisesRegex(ValueError, "trial indices"):
            stacking._validate_block_alignment(block, mismatched)  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
