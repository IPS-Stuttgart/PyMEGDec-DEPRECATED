import os
import subprocess  # nosec B404
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymegdec.cli import parse_classifier_param
from pymegdec.classifiers import (
    CLASSIFIER_REGISTRY,
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

        with patch("neureptrace.decoding.classifiers.train_classifier", side_effect=fake_train_classifier):
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

    def test_pymegdec_classifier_module_extends_upstream_registry(self):
        import pymegdec.classifiers as classifiers

        self.assertIs(classifiers.train_multiclass_classifier, train_multiclass_classifier)
        self.assertIn("gaussian-naive-bayes", classifiers.CLASSIFIER_REGISTRY)
        self.assertIn("multinomial-logistic-weighted", classifiers.CLASSIFIER_REGISTRY)
        self.assertIn("regularized-qda", classifiers.CLASSIFIER_REGISTRY)
        self.assertIn("shrinkage-prototype", classifiers.CLASSIFIER_REGISTRY)

    def test_default_params_for_cross_subject_baseline_classifiers(self):
        self.assertIsNone(get_default_classifier_param("correlation-prototype"))
        self.assertEqual(get_default_classifier_param("gaussian-naive-bayes"), 1e-9)
        self.assertEqual(get_default_classifier_param("multinomial-logistic"), 1.0)
        self.assertEqual(get_default_classifier_param("multinomial-logistic-weighted"), 1.0)
        self.assertEqual(get_default_classifier_param("regularized-qda"), 0.5)
        self.assertEqual(get_default_classifier_param("shrinkage-prototype"), 0.25)
        self.assertIsNone(get_default_classifier_param("shrinkage-lda"))

    def test_gaussian_naive_bayes_trains_and_predicts_probabilities(self):
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
                [1.1, 1.0],
                [2.0, 0.0],
                [2.0, 0.1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)

        model = train_multiclass_classifier(features, labels, "gaussian-naive-bayes", 1e-8)
        probabilities = model.predict_proba(features[:3])

        self.assertIn("gaussian-naive-bayes", CLASSIFIER_REGISTRY)
        self.assertEqual(model.model.var_smoothing, 1e-8)
        self.assertEqual(probabilities.shape, (3, 3))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(3))

    def test_regularized_qda_trains_and_predicts_probabilities(self):
        rng = np.random.default_rng(13)
        features = np.vstack(
            [
                rng.normal(loc=(-1.0, -1.0, 0.0), scale=0.2, size=(8, 3)),
                rng.normal(loc=(1.0, 0.5, 0.0), scale=0.2, size=(8, 3)),
                rng.normal(loc=(0.0, 1.2, 1.0), scale=0.2, size=(8, 3)),
            ]
        )
        labels = np.repeat(np.arange(3, dtype=int), 8)

        model = train_multiclass_classifier(features, labels, "regularized-qda", 0.25)
        probabilities = model.predict_proba(features[:4])

        self.assertIn("regularized-qda", CLASSIFIER_REGISTRY)
        self.assertEqual(model.model.reg_param, 0.25)
        self.assertEqual(probabilities.shape, (4, 3))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(4))

    def test_regularized_qda_rejects_invalid_regularization(self):
        with self.assertRaisesRegex(ValueError, "regularized-qda classifier_param"):
            train_multiclass_classifier(self.features, self.labels, "regularized-qda", 1.5)

    def test_weighted_multinomial_logistic_trains(self):
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [0.1, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 0, 1, 2], dtype=int)

        model = train_multiclass_classifier(features, labels, "multinomial-logistic-weighted", 1.0, random_state=13)

        self.assertIn("multinomial-logistic-weighted", CLASSIFIER_REGISTRY)
        self.assertEqual(model.model.class_weight, "balanced")
        self.assertEqual(len(model.predict(features)), len(labels))

    def test_shrinkage_prototype_classifier_trains_and_scores(self):
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [1.0, 1.0],
                [1.1, 1.0],
                [2.0, 0.0],
                [2.1, 0.1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)

        model = train_multiclass_classifier(features, labels, "shrinkage-prototype", 0.1)
        predictions = model.predict(np.asarray([[0.0, 0.1], [1.1, 1.0], [2.1, 0.0]], dtype=float))
        scores = model.decision_function(features[:3])

        self.assertIn("shrinkage-prototype", CLASSIFIER_REGISTRY)
        self.assertEqual(model.model.shrinkage, 0.1)
        np.testing.assert_array_equal(predictions, np.asarray([0, 1, 2], dtype=int))
        self.assertEqual(scores.shape, (3, 3))

    def test_shrinkage_prototype_rejects_invalid_shrinkage(self):
        with self.assertRaisesRegex(ValueError, "shrinkage-prototype classifier_param"):
            train_multiclass_classifier(self.features, self.labels, "shrinkage-prototype", 1.5)

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
