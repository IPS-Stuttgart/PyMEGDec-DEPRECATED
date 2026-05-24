import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pymegdec import stimulus_logit_stacking as stack


def _probability_rows(candidate: str, probabilities: list[tuple[float, float]], labels=(0, 1, 0, 1)) -> pd.DataFrame:
    rows = []
    for sample_index, ((p0, p1), label) in enumerate(zip(probabilities, labels, strict=True)):
        rows.append(
            {
                "decoder": candidate,
                "subject": "source" if sample_index < 2 else "source2",
                "fold": sample_index % 2,
                "sample_index": sample_index,
                "true_label": label,
                "true_class": f"class_{label}",
                "class_0": "class_0",
                "class_1": "class_1",
                "prob_class_0": p0,
                "prob_class_1": p1,
            }
        )
    return pd.DataFrame(rows)


def _source_observations() -> pd.DataFrame:
    good = _probability_rows("good", [(0.95, 0.05), (0.10, 0.90), (0.80, 0.20), (0.20, 0.80)])
    weak = _probability_rows("weak", [(0.55, 0.45), (0.45, 0.55), (0.52, 0.48), (0.48, 0.52)])
    return pd.concat([good, weak], ignore_index=True)


def _target_observations() -> pd.DataFrame:
    good = _probability_rows("good", [(0.90, 0.10), (0.15, 0.85), (0.75, 0.25), (0.30, 0.70)])
    weak = _probability_rows("weak", [(0.51, 0.49), (0.49, 0.51), (0.51, 0.49), (0.49, 0.51)])
    good["subject"] = "target"
    weak["subject"] = "target"
    return pd.concat([good, weak], ignore_index=True)


class TestStimulusLogitStacking(unittest.TestCase):
    def test_weighting_aliases_delegate_to_neureptrace_names(self):
        self.assertEqual(stack._normalize_weighting("greedy-balanced"), "stacked")  # pylint: disable=protected-access
        self.assertEqual(stack._normalize_weighting("inner_softmax"), "softmax")  # pylint: disable=protected-access
        self.assertEqual(stack._normalize_weighting("uniform"), "uniform")  # pylint: disable=protected-access

    def test_stack_source_oof_observations_uses_neureptrace_probability_stacker(self):
        stacked = stack.stack_source_oof_observations(
            _source_observations(),
            _target_observations(),
            weighting="stacked",
            output_decoder=stack.LOGIT_STACK_CLASSIFIER,
        )

        self.assertEqual(set(stacked["decoder"]), {stack.LOGIT_STACK_CLASSIFIER})
        self.assertIn("source_oof_weights", stacked.columns)
        self.assertIn("model_hash", stacked.columns)
        self.assertGreaterEqual(float(stacked["prob_class_0"].iloc[0]), 0.5)
        self.assertTrue(np.allclose(stacked[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0))

    def test_run_source_oof_probability_stacking_writes_compatibility_outputs(self):
        with tempfile.TemporaryDirectory(prefix="pymegdec-logit-stack-test-") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.csv"
            target = tmp_path / "target.csv"
            predictions = tmp_path / "stacked.csv"
            metrics = tmp_path / "metrics.csv"
            summary = tmp_path / "summary.csv"
            selected = tmp_path / "selected.csv"
            confusion = tmp_path / "confusion.csv"
            per_stimulus = tmp_path / "per_stimulus.csv"
            confusion_pairs = tmp_path / "confusion_pairs.csv"

            _source_observations().to_csv(source, index=False)
            _target_observations().to_csv(target, index=False)

            artifacts = stack.run_source_oof_probability_stacking(
                source_oof_paths=[source],
                target_paths=[target],
                predictions_output_path=predictions,
                metrics_output_path=metrics,
                group_summary_output_path=summary,
                selected_output_path=selected,
                confusion_output_path=confusion,
                per_stimulus_output_path=per_stimulus,
                confusion_pairs_output_path=confusion_pairs,
                weighting="uniform",
            )

            self.assertEqual(set(artifacts), {"predictions", "metrics", "selected"})
            self.assertTrue(predictions.exists())
            self.assertTrue(metrics.exists())
            self.assertTrue(summary.exists())
            self.assertTrue(selected.exists())
            self.assertTrue(confusion.exists())
            self.assertTrue(per_stimulus.exists())
            self.assertIn("source_oof_candidates", pd.read_csv(selected).columns)

    def test_cli_requires_observation_tables_instead_of_raw_data(self):
        with self.assertRaises(SystemExit) as cm:
            stack.stimulus_cross_subject_logit_stack(["--participants", "1-4"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_cli_delegates_to_probability_stacking(self):
        with tempfile.TemporaryDirectory(prefix="pymegdec-logit-stack-cli-test-") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.csv"
            target = tmp_path / "target.csv"
            out = tmp_path / "stacked.csv"
            summary = tmp_path / "summary.csv"
            selected = tmp_path / "selected.csv"

            _source_observations().to_csv(source, index=False)
            _target_observations().to_csv(target, index=False)

            status = stack.stimulus_cross_subject_logit_stack(
                [
                    "--source-oof",
                    str(source),
                    "--target",
                    str(target),
                    "--out",
                    str(out),
                    "--summary-output",
                    str(summary),
                    "--selected-output",
                    str(selected),
                    "--stacker-weighting",
                    "greedy_balanced",
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue(out.exists())
            self.assertEqual(set(pd.read_csv(out)["decoder"]), {stack.LOGIT_STACK_CLASSIFIER})
            self.assertTrue(summary.exists())
            self.assertTrue(selected.exists())

    def test_deprecated_raw_data_api_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "NeuRepTrace probability observation"):
            stack.make_logit_stack_candidate_configs()
        with self.assertRaisesRegex(RuntimeError, "Part\\*Data"):
            stack.evaluate_cross_subject_logit_stacking("unused", [1, 2, 3, 4], candidate_configs=())


if __name__ == "__main__":
    unittest.main()
