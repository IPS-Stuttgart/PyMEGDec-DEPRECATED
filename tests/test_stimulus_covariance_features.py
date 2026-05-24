import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pymegdec import stimulus_covariance_features as covariance
from pymegdec.stimulus_covariance_features import (
    covariance_feature_vector,
    make_covariance_candidate_configs,
    normalize_covariance_feature_mode,
)


class TestStimulusCovarianceFeatures(unittest.TestCase):
    def test_covariance_feature_vector_delegates_to_neureptrace_modes(self):
        signal = np.asarray(
            [
                [1.0, -1.0, 1.0, -1.0],
                [0.2, 0.1, 0.2, 0.1],
                [0.0, 0.1, 0.0, 0.1],
            ],
            dtype=float,
        )

        self.assertEqual(covariance_feature_vector(signal, "covariance_upper").shape, (6,))
        self.assertEqual(covariance_feature_vector(signal, "correlation_upper").shape, (6,))
        self.assertEqual(covariance_feature_vector(signal, "variance").shape, (3,))
        self.assertTrue(np.all(np.isfinite(covariance_feature_vector(signal, "logeuclidean_covariance"))))
        self.assertEqual(normalize_covariance_feature_mode("correlation"), "correlation_upper")

    def test_make_covariance_candidate_configs_returns_neureptrace_specs(self):
        candidates = make_covariance_candidate_configs(
            time_windows=((0.0, 0.03),),
            covariance_feature_modes=("logeuclidean_covariance", "variance"),
            covariance_shrinkages=(0.1,),
            covariance_epsilons=(1e-6,),
            covariance_max_channels=(8,),
            projections=("pca",),
            classifiers=("multinomial-logistic",),
            classifier_params=(1.0,),
            components_values=(32,),
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate.covariance_feature_mode for candidate in candidates}, {"logeuclidean_covariance", "variance"})
        self.assertEqual({candidate.covariance_max_channels for candidate in candidates}, {8})
        self.assertEqual({candidate.pca_components for candidate in candidates}, {32})

    def test_build_neureptrace_covariance_config_maps_legacy_grid(self):
        config = covariance.build_neureptrace_covariance_config(
            data_folder="/data/bush-meg",
            participants="1-2,4",
            time_windows=((0.05, 0.30), (0.08, 0.35)),
            baseline_window=(-0.35, -0.05),
            normalizations=("subject_baseline_whiten",),
            feature_modes=("logeuclidean_covariance", "variance"),
            covariance_shrinkages=(0.05, 0.1),
            covariance_epsilons=(1e-6,),
            covariance_max_channels=(32, 64),
            projections=("pca",),
            classifiers=("multinomial-logistic",),
            classifier_params=(0.1, 1.0),
            components_values=(32, 64),
            label_shuffle_control=True,
            label_shuffle_seed=7,
        )

        self.assertEqual(config["dataset"]["root"], "/data/bush-meg")
        self.assertEqual(config["participants"]["ids"], "1-2,4")
        self.assertEqual(config["preprocessing"]["normalization"], "subject_baseline_whiten")
        self.assertEqual(config["preprocessing"]["baseline_window"], [-0.35, -0.05])
        self.assertEqual(config["preprocessing"]["tmin"], -0.35)
        self.assertEqual(config["preprocessing"]["tmax"], 0.35)
        grid = config["covariance_loso"]["candidate_grid"]
        self.assertEqual(len(grid["time_windows"]), 2)
        self.assertEqual(grid["feature_modes"], ["logeuclidean_covariance", "variance"])
        self.assertEqual(grid["covariance_shrinkages"], [0.05, 0.1])
        self.assertEqual(grid["covariance_max_channels"], [32, 64])
        self.assertEqual(grid["pca_components"], [32, 64])
        self.assertTrue(config["covariance_loso"]["label_shuffle_control"])
        self.assertEqual(config["covariance_loso"]["label_shuffle_seed"], 7)

    def test_write_neureptrace_covariance_config_uses_json(self):
        with self.subTest("roundtrip"):
            config = covariance.build_neureptrace_covariance_config(data_folder="/tmp/data", participants="1")
            out = self._tmp_json_path()
            try:
                covariance.write_neureptrace_covariance_config(config, out)
                loaded = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(loaded["dataset"]["root"], "/tmp/data")
                self.assertEqual(loaded["covariance_loso"]["candidate_grid"]["feature_modes"], ["logeuclidean_covariance"])
            finally:
                if out.exists():
                    out.unlink()
                if out.parent.exists():
                    out.parent.rmdir()

    def test_cli_delegates_to_neureptrace_runner(self):
        calls = []

        def fake_run(config_path, *, out_path=None, inner_cv_out_path=None, predictions_out_path=None):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            calls.append((config, out_path, inner_cv_out_path, predictions_out_path))
            summary = pd.DataFrame(
                [
                    {
                        "outer_test_subject": "1",
                        "candidate": "cov_00",
                        "balanced_accuracy": 0.25,
                        "covariance_feature_mode": "variance",
                        "covariance_shrinkage": 0.1,
                        "covariance_epsilon": 1e-6,
                        "feature_preprocessor": "pca",
                        "pca_components": 32,
                    }
                ]
            )
            pd.DataFrame(
                [
                    {
                        "true_label": 0,
                        "predicted_label": 0,
                        "prob_class_0": 0.9,
                        "prob_class_1": 0.1,
                    }
                ]
            ).to_csv(predictions_out_path, index=False)
            return summary

        with (
            patch.object(covariance, "resolve_data_folder", return_value="/data/bush"),
            patch.object(covariance._nrt_covariance, "run_bushmeg_covariance_loso", side_effect=fake_run),
        ):
            with self._tmp_output_paths() as paths:
                status = covariance.stimulus_cross_subject_covariance(
                    [
                        "--data-dir",
                        "/unused",
                        "--participants",
                        "1-2",
                        "--time-windows",
                        "0.05:0.30",
                        "--feature-modes",
                        "variance",
                        "--covariance-shrinkages",
                        "0.1",
                        "--classifier-params",
                        "1.0",
                        "--components-values",
                        "32",
                        "--outer-output",
                        str(paths.outer),
                        "--summary-output",
                        str(paths.summary),
                        "--inner-validation-output",
                        str(paths.inner),
                        "--selected-output",
                        str(paths.selected),
                        "--predictions-output",
                        str(paths.predictions),
                        "--confusion-output",
                        str(paths.confusion),
                        "--per-stimulus-output",
                        str(paths.per_stimulus),
                        "--confusion-pairs-output",
                        str(paths.confusion_pairs),
                    ]
                )

                self.assertEqual(status, 0)
                self.assertEqual(len(calls), 1)
                generated_config = calls[0][0]
                self.assertEqual(generated_config["dataset"]["root"], "/data/bush")
                self.assertEqual(generated_config["covariance_loso"]["candidate_grid"]["feature_modes"], ["variance"])
                self.assertTrue(paths.summary.exists())
                self.assertTrue(paths.selected.exists())
                self.assertTrue(paths.confusion.exists())
                self.assertTrue(paths.per_stimulus.exists())

    def test_deprecated_native_loader_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "NeuRepTrace"):
            covariance.load_participant_covariance_features("unused", 1)
        with self.assertRaisesRegex(RuntimeError, "neureptrace"):
            covariance.evaluate_nested_covariance_stimulus("unused", [1, 2, 3], candidate_configs=())

    def _tmp_json_path(self):
        import tempfile
        from pathlib import Path

        root = Path(tempfile.mkdtemp(prefix="pymegdec-cov-test-"))
        return root / "config.json"

    class _OutputPaths:
        def __init__(self):
            import tempfile
            from pathlib import Path

            self.root = Path(tempfile.mkdtemp(prefix="pymegdec-cov-cli-test-"))
            self.outer = self.root / "outer.csv"
            self.summary = self.root / "summary.csv"
            self.inner = self.root / "inner.csv"
            self.selected = self.root / "selected.csv"
            self.predictions = self.root / "predictions.csv"
            self.confusion = self.root / "confusion.csv"
            self.per_stimulus = self.root / "per_stimulus.csv"
            self.confusion_pairs = self.root / "confusion_pairs.csv"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            for path in sorted(self.root.glob("*"), reverse=True):
                path.unlink()
            self.root.rmdir()

    def _tmp_output_paths(self):
        return self._OutputPaths()


if __name__ == "__main__":
    unittest.main()
