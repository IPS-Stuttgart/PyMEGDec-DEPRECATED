import os
import subprocess  # nosec B404
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymegdec.cli import parse_classifier_param
from reptrace.decoding.classifiers import (
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

    def test_train_multiclass_classifier_encodes_nonzero_labels_and_decodes_outputs(self):
        class EncodedBinaryModel:
            def predict(self, features):
                return np.arange(np.asarray(features).shape[0], dtype=int) % 2

            def decision_function(self, features):
                return np.asarray([0.25, -0.50], dtype=float)[: np.asarray(features).shape[0]]

        seen = {}

        def fake_train_classifier(_features, labels, _classifier, _classifier_param, *, random_state=None, registry=None):
            del random_state, registry
            seen["labels"] = np.asarray(labels, dtype=int).copy()
            return EncodedBinaryModel()

        with patch("reptrace.decoding.classifiers.train_classifier", side_effect=fake_train_classifier):
            model = train_multiclass_classifier(
                self.features[:2],
                np.asarray([10, 20], dtype=int),
                "dummy-requires-zero-based-labels",
                None,
            )

        np.testing.assert_array_equal(seen["labels"], np.asarray([0, 1], dtype=int))
        np.testing.assert_array_equal(model.classes_, np.asarray([10, 20], dtype=int))
        np.testing.assert_array_equal(model.predict(self.features[:2]), np.asarray([10, 20], dtype=int))
        np.testing.assert_allclose(model.decision_function(self.features[:2]), np.asarray([[-0.25, 0.25], [0.50, -0.50]], dtype=float))

    def test_pymegdec_classifier_module_is_compatibility_shim(self):
        import pymegdec.classifiers as classifiers

        self.assertIs(classifiers.train_multiclass_classifier, train_multiclass_classifier)

    def test_default_params_for_cross_subject_baseline_classifiers(self):
        self.assertIsNone(get_default_classifier_param("correlation-prototype"))
        self.assertEqual(get_default_classifier_param("multinomial-logistic"), 1.0)
        self.assertIsNone(get_default_classifier_param("shrinkage-lda"))

    def test_parse_classifier_param_accepts_auto(self):
        self.assertEqual(parse_classifier_param("auto"), "auto")

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
