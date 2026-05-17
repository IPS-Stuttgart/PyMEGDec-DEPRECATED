import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pymegdec.classifiers import train_multiclass_classifier
from pymegdec.data_config import resolve_data_folder
from pymegdec.model_transfer import (
    evaluate_model_transfer,
    get_original_feature_importance,
)
from tests.matlab_fixtures import cell_array


class TestLinearSvmFeatures(unittest.TestCase):
    def test_multiclass_svm_uses_linear_kernel(self):
        features = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0, 1, 1])

        model = train_multiclass_classifier(features, labels, "multiclass-svm", 1.0)

        self.assertEqual(model[-1].kernel, "linear")
        self.assertTrue(hasattr(model[-1], "coef_"))

    def test_original_feature_importance_without_pca(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        np.testing.assert_allclose(
            get_original_feature_importance(Model()),
            [[2.0, 3.0]],
        )

    def test_original_feature_importance_maps_matlab_style_pca_loadings(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        pca_components = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        np.testing.assert_allclose(
            get_original_feature_importance(Model(), pca_components),
            [[2.0, 3.0, 5.0]],
        )

    def test_original_feature_importance_maps_sklearn_style_pca_components(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        pca_components = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        np.testing.assert_allclose(
            get_original_feature_importance(Model(), pca_components),
            [[2.0, 3.0, 5.0]],
        )

    def test_original_feature_importance_rejects_incompatible_pca_shape(self):
        class Model:
            coef_ = np.array([[2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "incompatible"):
            get_original_feature_importance(Model(), np.ones((3, 4)))

    def test_original_feature_importance_uses_pipeline_scale(self):
        class Scaler:
            scale_ = np.array([2.0, 4.0])

        class Classifier:
            coef_ = np.array([[2.0, 8.0]])

        class Model:
            steps = [("standardscaler", Scaler()), ("svc", Classifier())]

        np.testing.assert_allclose(
            get_original_feature_importance(Model()),
            [[1.0, 2.0]],
        )

    def test_original_feature_importance_requires_coefficients(self):
        with self.assertRaises(ValueError):
            get_original_feature_importance(object())


def _mat_data_with_time(time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.array([1, 2])
    return {
        "time": cell_array([np.array([time])]),
        "trialinfo": trialinfo,
    }


def _mat_data_with_trialinfo(time, labels):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.array(labels)
    return {
        "time": cell_array([np.array([time])]),
        "trialinfo": trialinfo,
    }


class TestEvaluateModelTransfer(unittest.TestCase):
    def setUp(self) -> None:
        self.parts = 2
        try:
            self.data_folder = resolve_data_folder(
                required=True,
                required_files=[
                    f"Part{self.parts}Data.mat",
                    f"Part{self.parts}CueData.mat",
                ],
            )
        except FileNotFoundError as exc:
            if os.getenv("PYMEGDEC_REQUIRE_DATA"):
                self.fail(str(exc))
            self.skipTest(str(exc))

        self.null_window_center = np.nan

    def test_evaluate_model_transfer_accuracy_svm(self):
        classifier = "multiclass-svm"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

    def test_evaluate_model_transfer_accuracy_scikit_mlp(self):
        classifier = "scikit-mlp"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.09, "Accuracy should be at least 0.09")

    def test_evaluate_model_transfer_accuracy_pytorch_mlp(self):
        classifier = "pytorch-mlp"
        accuracy = evaluate_model_transfer(
            self.data_folder,
            self.parts,
            null_window_center=self.null_window_center,
            classifier=classifier,
            components_pca=200,
        )

        self.assertGreaterEqual(accuracy, 0.09, "Accuracy should be at least 0.09")


class TestEvaluateModelTransferSynthetic(unittest.TestCase):
    def test_evaluate_model_transfer_rejects_sampling_rate_mismatch(self):
        train_data = _mat_data_with_time([0.0, 0.1])
        val_data = _mat_data_with_time([0.0, 0.2])

        with patch(
            "pymegdec.model_transfer.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([val_data], dtype=object)},
            ],
        ):
            with self.assertRaisesRegex(ValueError, "Sampling rate"):
                evaluate_model_transfer("unused", 1)

    def test_evaluate_model_transfer_flattens_matlab_trialinfo_column_vectors(self):
        train_data = _mat_data_with_trialinfo([0.0, 0.1], [[1], [2]])
        val_data = _mat_data_with_trialinfo([0.0, 0.1], [[1], [2]])
        captured_labels = {}

        def fake_evaluate_feature_transfer(
            features_train,
            labels_train,
            features_val,
            labels_val,
            **kwargs,
        ):
            del features_train, features_val, kwargs
            captured_labels["train"] = labels_train.copy()
            captured_labels["val"] = labels_val.copy()
            return SimpleNamespace(accuracy=0.5)

        with patch(
            "pymegdec.model_transfer.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([val_data], dtype=object)},
            ],
        ), patch(
            "pymegdec.model_transfer.preprocess_features",
            side_effect=[
                ([np.array([[1.0, 2.0]])], []),
                ([np.array([[3.0, 4.0]])], []),
            ],
        ), patch(
            "pymegdec.model_transfer.evaluate_feature_transfer",
            side_effect=fake_evaluate_feature_transfer,
        ):
            accuracy = evaluate_model_transfer(
                "unused",
                1,
                null_window_center=np.nan,
                components_pca=float("inf"),
            )

        self.assertEqual(accuracy, 0.5)
        self.assertEqual(captured_labels["train"].shape, (2,))
        self.assertEqual(captured_labels["val"].shape, (2,))
        np.testing.assert_array_equal(captured_labels["train"], np.array([0, 1]))
        np.testing.assert_array_equal(captured_labels["val"], np.array([0, 1]))
        np.testing.assert_array_equal(train_data["trialinfo"][0][0], np.array([[1], [2]]))
        np.testing.assert_array_equal(val_data["trialinfo"][0][0], np.array([[1], [2]]))

    def test_evaluate_model_transfer_does_not_mutate_trialinfo_labels(self):
        train_data = _mat_data_with_time([0.0, 0.1])
        val_data = _mat_data_with_time([0.0, 0.1])
        captured_labels = {}

        def fake_evaluate_feature_transfer(
            features_train,
            labels_train,
            features_val,
            labels_val,
            **kwargs,
        ):
            del features_train, features_val, kwargs
            captured_labels["train"] = labels_train.copy()
            captured_labels["val"] = labels_val.copy()
            return SimpleNamespace(accuracy=0.5)

        with patch(
            "pymegdec.model_transfer.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([val_data], dtype=object)},
            ],
        ), patch(
            "pymegdec.model_transfer.preprocess_features",
            side_effect=[
                ([np.array([[1.0, 2.0]])], []),
                ([np.array([[3.0, 4.0]])], []),
            ],
        ), patch(
            "pymegdec.model_transfer.evaluate_feature_transfer",
            side_effect=fake_evaluate_feature_transfer,
        ):
            accuracy = evaluate_model_transfer(
                "unused",
                1,
                null_window_center=np.nan,
                components_pca=float("inf"),
            )

        self.assertEqual(accuracy, 0.5)
        np.testing.assert_array_equal(captured_labels["train"], np.array([0, 1]))
        np.testing.assert_array_equal(captured_labels["val"], np.array([0, 1]))
        np.testing.assert_array_equal(train_data["trialinfo"][0][0], np.array([1, 2]))
        np.testing.assert_array_equal(val_data["trialinfo"][0][0], np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
