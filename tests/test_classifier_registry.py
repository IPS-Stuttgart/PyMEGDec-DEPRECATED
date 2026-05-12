import sys
import unittest
import warnings

import numpy as np
from pymegdec.classifiers import (
    CLASSIFIER_REGISTRY,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from sklearn.exceptions import ConvergenceWarning


class TestClassifierRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [1.0, 1.0],
                [1.0, 1.2],
                [2.0, 2.0],
                [2.0, 2.2],
            ]
        )
        self.labels = np.array([0, 0, 1, 1, 2, 2])

    def test_registry_contains_supported_classifiers(self):
        self.assertEqual(
            {
                "always1Dummy",
                "correlation-prototype",
                "gradient-boosting",
                "knn",
                "mostFrequentDummy",
                "multinomial-logistic",
                "multiclass-svm",
                "multiclass-svm-weighted",
                "pytorch-mlp",
                "random-forest",
                "scikit-mlp",
                "shrinkage-lda",
                "xgboost",
            },
            set(CLASSIFIER_REGISTRY),
        )

    def test_registry_trains_fast_sklearn_classifiers(self):
        classifier_params = {
            "always1Dummy": None,
            "correlation-prototype": None,
            "gradient-boosting": 5,
            "knn": 1,
            "mostFrequentDummy": None,
            "multinomial-logistic": 1.0,
            "multiclass-svm": 1.0,
            "multiclass-svm-weighted": 1.0,
            "random-forest": 5,
            "scikit-mlp": (5, 50),
            "shrinkage-lda": None,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for classifier, classifier_param in classifier_params.items():
                with self.subTest(classifier=classifier):
                    model = train_multiclass_classifier(
                        self.features,
                        self.labels,
                        classifier,
                        classifier_param,
                        random_state=13,
                    )
                    predictions = model.predict(self.features)
                    self.assertEqual(len(predictions), len(self.labels))

    def test_correlation_prototype_predicts_by_nearest_class_pattern(self):
        features = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
            ]
        )
        labels = np.array([0, 0, 1, 1, 2, 2])
        model = train_multiclass_classifier(
            features,
            labels,
            "correlation-prototype",
            None,
        )

        predictions = model.predict(np.array([[0.9, 0.1, 0.0], [0.0, 0.2, 1.0]]))

        np.testing.assert_array_equal(predictions, np.array([0, 2]))

    def test_random_state_reproduces_stochastic_classifier_predictions(self):
        model_a = train_multiclass_classifier(self.features, self.labels, "random-forest", 5, random_state=7)
        model_b = train_multiclass_classifier(self.features, self.labels, "random-forest", 5, random_state=7)

        np.testing.assert_array_equal(
            model_a.predict(self.features),
            model_b.predict(self.features),
        )

    def test_default_classifier_param_detection_handles_non_numeric_values(self):
        self.assertTrue(should_use_default_classifier_param(np.nan))
        self.assertFalse(should_use_default_classifier_param(None))
        self.assertFalse(should_use_default_classifier_param({"hidden_dim": 10}))

    def test_optional_ml_dependencies_are_lazy_imported(self):
        self.assertNotIn("xgboost", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("pytorch_lightning", sys.modules)

    def test_unsupported_classifier_error_lists_supported_names(self):
        with self.assertRaisesRegex(ValueError, "Supported classifiers"):
            train_multiclass_classifier(self.features, self.labels, "unknown", None)


if __name__ == "__main__":
    unittest.main()
