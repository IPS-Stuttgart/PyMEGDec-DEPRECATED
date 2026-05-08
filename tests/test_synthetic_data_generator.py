import tempfile
import unittest
from pathlib import Path

import scipy.io as sio

from pymegdec.synthetic_data import SyntheticDataConfig, write_synthetic_dataset
from pymegdec.synthetic_data_cli import make_synthetic_data


class TestSyntheticDataGenerator(unittest.TestCase):
    def test_write_synthetic_dataset_creates_participant_mat_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = write_synthetic_dataset(
                tmp_dir,
                SyntheticDataConfig(
                    participant_id=7,
                    n_classes=2,
                    main_repeats_per_class=4,
                    cue_repeats_per_class=2,
                    n_channels=4,
                    n_times=101,
                    tmax=0.5,
                    noise_scale=0.01,
                    alpha_scale=0.0,
                ),
            )

            self.assertTrue(output.main_path.exists())
            self.assertTrue(output.cue_path.exists())
            self.assertTrue(output.manifest_path.exists())
            self.assertEqual(output.main_trials, 8)
            self.assertEqual(output.cue_trials, 4)

            data = sio.loadmat(output.main_path)["data"][0]
            self.assertIn("trial", data.dtype.names)
            self.assertIn("time", data.dtype.names)
            self.assertIn("trialinfo", data.dtype.names)
            self.assertIn("label", data.dtype.names)
            self.assertIn("grad", data.dtype.names)
            self.assertEqual(len(data["trial"][0][0]), 8)
            self.assertEqual(data["trial"][0][0][0].shape, (4, 101))
            self.assertEqual(data["trialinfo"][0][0].tolist(), [1, 2, 1, 2, 1, 2, 1, 2])

    def test_write_synthetic_dataset_refuses_to_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SyntheticDataConfig(n_classes=2, main_repeats_per_class=1, cue_repeats_per_class=1)
            write_synthetic_dataset(tmp_dir, config)

            with self.assertRaises(FileExistsError):
                write_synthetic_dataset(tmp_dir, config)

            output = write_synthetic_dataset(tmp_dir, config, overwrite=True)
            self.assertTrue(output.main_path.exists())

    def test_make_synthetic_data_cli(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code = make_synthetic_data(
                [
                    "--out",
                    tmp_dir,
                    "--participant",
                    "3",
                    "--classes",
                    "2",
                    "--main-repeats",
                    "3",
                    "--cue-repeats",
                    "2",
                    "--channels",
                    "4",
                    "--times",
                    "101",
                    "--tmax",
                    "0.5",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmp_dir) / "Part3Data.mat").exists())
            self.assertTrue((Path(tmp_dir) / "Part3CueData.mat").exists())
            self.assertTrue((Path(tmp_dir) / "synthetic_data_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
