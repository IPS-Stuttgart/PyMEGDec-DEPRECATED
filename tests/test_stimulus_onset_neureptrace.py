import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pymegdec import stimulus_onset_neureptrace as onset


def _observation_rows() -> pd.DataFrame:
    rows = []
    for sequence_id, label in [("trial0", 0), ("trial1", 1)]:
        for time, confidence in [(-0.10, 0.20), (-0.05, 0.25), (0.00, 0.95), (0.05, 0.90)]:
            p_true = confidence
            p_other = 1.0 - confidence
            if label == 0:
                probs = (p_true, p_other)
                predicted = 0 if p_true >= p_other else 1
            else:
                probs = (p_other, p_true)
                predicted = 1 if p_true >= p_other else 0
            rows.append(
                {
                    "subject": "s1",
                    "decoder": "demo",
                    "emission_mode": "calibrated",
                    "sequence_id": sequence_id,
                    "sample_index": sequence_id,
                    "time": time,
                    "window_start": time - 0.0125,
                    "window_stop": time + 0.0125,
                    "true_label": label,
                    "true_class": f"class_{label}",
                    "class_0": "class_0",
                    "class_1": "class_1",
                    "predicted_label": predicted,
                    "predicted_class": f"class_{predicted}",
                    "prob_class_0": probs[0],
                    "prob_class_1": probs[1],
                    "confidence": max(probs),
                }
            )
    return pd.DataFrame(rows)


class TestStimulusOnsetNeuRepTrace(unittest.TestCase):
    def test_run_neureptrace_onset_scan_writes_pymegdec_compat_outputs(self):
        with tempfile.TemporaryDirectory(prefix="pymegdec-onset-test-") as tmp:
            root = Path(tmp)
            observations = root / "observations.csv"
            scan = root / "scan.csv"
            events = root / "events.csv"
            summary = root / "summary.csv"
            event_summary = root / "event_summary.csv"
            _observation_rows().to_csv(observations, index=False)

            scan_rows, event_rows, summary_rows, event_summary_rows = onset.run_neureptrace_onset_scan(
                [observations],
                output_path=scan,
                events_output_path=events,
                summary_output_path=summary,
                event_summary_output_path=event_summary,
                threshold_window=(-0.10, -0.05),
                threshold_quantile=1.0,
                threshold_method="point",
                detection_start_s=0.0,
            )

            self.assertTrue(scan.exists())
            self.assertTrue(events.exists())
            self.assertTrue(summary.exists())
            self.assertTrue(event_summary.exists())
            self.assertEqual(len(scan_rows), 8)
            self.assertEqual(len(event_rows), 2)
            self.assertTrue(all(row["detected"] for row in event_rows))
            self.assertIn("scan_window_center_s", pd.read_csv(scan).columns)
            self.assertIn("detection_window_center_s", pd.read_csv(events).columns)
            self.assertTrue(summary_rows)
            self.assertTrue(event_summary_rows)

    def test_cli_requires_observation_csvs(self):
        with self.assertRaises(SystemExit) as caught:
            onset.stimulus_onset_scan(["--participants", "1-2"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_cli_delegates_to_neureptrace_observation_csvs(self):
        with tempfile.TemporaryDirectory(prefix="pymegdec-onset-cli-test-") as tmp:
            root = Path(tmp)
            observations = root / "observations.csv"
            scan = root / "scan.csv"
            events = root / "events.csv"
            summary = root / "summary.csv"
            event_summary = root / "event_summary.csv"
            _observation_rows().to_csv(observations, index=False)

            status = onset.stimulus_onset_scan(
                [
                    "--observation-csv",
                    str(observations),
                    "--threshold-window",
                    "-0.10,-0.05",
                    "--threshold-quantile",
                    "1.0",
                    "--detection-start-s",
                    "0.0",
                    "--output",
                    str(scan),
                    "--events-output",
                    str(events),
                    "--summary-output",
                    str(summary),
                    "--event-summary-output",
                    str(event_summary),
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue(scan.exists())
            self.assertTrue(events.exists())
            self.assertTrue(summary.exists())
            self.assertTrue(event_summary.exists())


if __name__ == "__main__":
    unittest.main()
