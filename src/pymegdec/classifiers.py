"""Backward-compatible classifier adapters.

Classifier implementations now live in :mod:`reptrace.decoding.classifiers`.
This module preserves historical ``pymegdec.classifiers`` imports and adds
PyMEGDec-local registry entries that are useful for the stimulus benchmarks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import reptrace.decoding.classifiers as reptrace_classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from reptrace.decoding.classifiers import (
    ClassifierSpec,
    CorrelationPrototypeClassifier,
    DecodedLabelClassifier,
    _build_pytorch_data_loaders,
    encode_classifier_labels,
    positive_class_score,
    prediction_scores,
    should_use_default_classifier_param,
    train_binary_svm,
    train_for_stimulus_lasso_glm,
    train_gradient_boosting,
    train_lasso_logistic,
)
from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY as REPTRACE_CLASSIFIER_REGISTRY,
)
from reptrace.decoding.classifiers import (
    DEFAULT_CLASSIFIER_PARAMS as REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
)
from reptrace.decoding.classifiers import (
    get_default_classifier_param as get_reptrace_default_classifier_param,
)

_DecodedLabelClassifier = DecodedLabelClassifier
_encode_classifier_labels = encode_classifier_labels

PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS = {
    "gaussian-naive-bayes": 1e-9,
    "multinomial-logistic-weighted": 1.0,
    "regularized-qda": 0.5,
    "shrinkage-prototype": 0.25,
}
DEFAULT_CLASSIFIER_PARAMS = {
    **REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
    **PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS,
}


def _build_weighted_multinomial_logistic(_features, _labels, classifier_param, random_state):
    return LogisticRegression(
        C=float(classifier_param),
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


def _build_gaussian_naive_bayes(_features, _labels, classifier_param, _random_state):
    return GaussianNB(var_smoothing=float(classifier_param))


def _build_regularized_qda(_features, _labels, classifier_param, _random_state):
    reg_param = PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS["regularized-qda"] if classifier_param is None else float(classifier_param)
    if not 0.0 <= reg_param <= 1.0:
        raise ValueError("regularized-qda classifier_param must be a numeric regularization in [0, 1].")
    return QuadraticDiscriminantAnalysis(reg_param=reg_param)


class ShrinkagePrototypeClassifier:
    """Classify by distance to class prototypes shrunk toward the grand mean."""

    def __init__(self, shrinkage: float = 0.25):
        self.shrinkage = float(shrinkage)
        self.classes_: np.ndarray | None = None
        self.class_prototypes_: np.ndarray | None = None
        self.global_prototype_: np.ndarray | None = None
        self.shrunk_prototypes_: np.ndarray | None = None

    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels).ravel()
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix.")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have the same length as features.")
        if features.shape[0] == 0:
            raise ValueError("At least one training row is required.")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage-prototype classifier_param must be a numeric shrinkage in [0, 1].")

        self.classes_ = np.unique(labels)
        self.class_prototypes_ = np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in self.classes_])
        self.global_prototype_ = np.mean(features, axis=0, keepdims=True)
        self.shrunk_prototypes_ = (1.0 - self.shrinkage) * self.class_prototypes_ + self.shrinkage * self.global_prototype_
        return self

    def decision_function(self, features):
        if self.shrunk_prototypes_ is None:
            raise RuntimeError("ShrinkagePrototypeClassifier must be fitted before scoring.")
        features = np.asarray(features, dtype=float)
        squared_distances = np.sum(np.square(features[:, None, :] - self.shrunk_prototypes_[None, :, :]), axis=2)
        return -squared_distances

    def predict(self, features):
        if self.classes_ is None:
            raise RuntimeError("ShrinkagePrototypeClassifier must be fitted before prediction.")
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]


def _normalize_shrinkage_prototype_param(classifier_param):
    if classifier_param is None:
        return PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS["shrinkage-prototype"]
    shrinkage = float(classifier_param)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage-prototype classifier_param must be a numeric shrinkage in [0, 1].")
    return shrinkage


def _build_shrinkage_prototype(_features, _labels, classifier_param, _random_state):
    return ShrinkagePrototypeClassifier(shrinkage=_normalize_shrinkage_prototype_param(classifier_param))


CLASSIFIER_REGISTRY = {
    **REPTRACE_CLASSIFIER_REGISTRY,
    "gaussian-naive-bayes": ClassifierSpec(_build_gaussian_naive_bayes),
    "multinomial-logistic-weighted": ClassifierSpec(_build_weighted_multinomial_logistic),
    "regularized-qda": ClassifierSpec(_build_regularized_qda),
    "shrinkage-prototype": ClassifierSpec(_build_shrinkage_prototype),
}

__all__ = [
    "CLASSIFIER_REGISTRY",
    "DEFAULT_CLASSIFIER_PARAMS",
    "ClassifierSpec",
    "CorrelationPrototypeClassifier",
    "DecodedLabelClassifier",
    "ShrinkagePrototypeClassifier",
    "_build_pytorch_data_loaders",
    "get_default_classifier_param",
    "positive_class_score",
    "prediction_scores",
    "should_use_default_classifier_param",
    "train_binary_svm",
    "train_classifier",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
    "train_lasso_logistic",
    "train_multiclass_classifier",
]


def get_default_classifier_param(classifier: str) -> Any:
    """Return a defensive copy of the PyMEGDec classifier default."""

    if classifier in DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    return get_reptrace_default_classifier_param(classifier)


def train_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Build and fit a classifier from the PyMEGDec-extended registry."""

    return reptrace_classifiers.train_classifier(
        features,
        labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY if registry is None else registry,
    )


def train_multiclass_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Train a multiclass classifier from the PyMEGDec-extended registry."""

    classes, encoded_labels = encode_classifier_labels(labels)
    model = train_classifier(
        features,
        encoded_labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY if registry is None else registry,
    )
    return DecodedLabelClassifier(model, classes)


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from reptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
