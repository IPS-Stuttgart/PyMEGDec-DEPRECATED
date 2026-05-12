import os
import subprocess  # nosec B404
import sys
import unittest
from pathlib import Path

import numpy as np
from pymegdec.classifiers import (
    get_default_classifier_param,
    train_multiclass_classifier,
)


class TestClassifiers(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [1.0, 1.0],
                [1.0, 1.2],
            ]
        )
        self.labels = np.array([0, 0, 1, 1])

    def test_fast_sklearn_classifier_trains(self):
        model = train_multiclass_classifier(
            self.features,
            self.labels,
            "multiclass-svm",
            get_default_classifier_param("multiclass-svm"),
        )

        predictions = model.predict(self.features)

        self.assertEqual(len(predictions), len(self.labels))

    def test_default_params_for_cross_subject_baseline_classifiers(self):
        self.assertIsNone(get_default_classifier_param("correlation-prototype"))
        self.assertEqual(get_default_classifier_param("multinomial-logistic"), 1.0)
        self.assertIsNone(get_default_classifier_param("shrinkage-lda"))

    def test_optional_ml_dependencies_are_lazy_imported(self):
        env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(  # nosec B603
            [
                sys.executable,
                "-c",
                ("import sys; " "import pymegdec.classifiers; " "print('xgboost' in sys.modules, " "'torch' in sys.modules, " "'pytorch_lightning' in sys.modules)"),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual("False False False", result.stdout.strip())

    def test_unsupported_classifier_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported classifier"):
            train_multiclass_classifier(self.features, self.labels, "unknown", None)


if __name__ == "__main__":
    unittest.main()
