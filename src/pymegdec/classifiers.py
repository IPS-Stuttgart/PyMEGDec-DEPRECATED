"""Backward-compatible classifier adapters.

Classifier implementations now live in :mod:`reptrace.decoding.classifiers`.
This module preserves historical ``pymegdec.classifiers`` imports and adds
PyMEGDec-local registry entries that are useful for the stimulus benchmarks.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import reptrace.decoding.classifiers as reptrace_classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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
    "one-vs-one-linear-svm": 1.0,
    "one-vs-rest-linear-svm": 1.0,
    "ecoc-linear-svm": {"C": 1.0, "code_size": 2.0, "random_state": 0},
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


def _linear_svc(classifier_param, random_state):
    return LinearSVC(C=float(classifier_param), max_iter=5000, random_state=random_state)


def _build_one_vs_one_linear_svm(_features, _labels, classifier_param, random_state):
    return OneVsOneClassifier(_linear_svc(classifier_param, random_state))


def _build_one_vs_rest_linear_svm(_features, _labels, classifier_param, random_state):
    return OneVsRestClassifier(_linear_svc(classifier_param, random_state))


class SampleWeightedECOCLinearSVM:
    """Small ECOC classifier with LinearSVC binary learners and sample weights.

    scikit-learn's ``OutputCodeClassifier`` is convenient, but older supported
    versions do not consistently propagate ``sample_weight``.  This local ECOC
    implementation keeps the BUSH-MEG source-subject weighting experiments from
    silently dropping weights.
    """

    def __init__(self, *, C: float = 1.0, code_size: float = 2.0, random_state: int | None = 0):
        self.C = float(C)
        self.code_size = float(code_size)
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.code_book_: np.ndarray | None = None
        self.models_: list[LinearSVC] | None = None

    def fit(self, features, labels, sample_weight=None):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels).ravel()
        self.classes_ = np.unique(labels)
        if self.classes_.size < 2:
            raise ValueError("ECOC requires at least two classes.")
        rng = np.random.default_rng(self.random_state)
        n_bits = max(1, int(np.ceil(self.code_size * self.classes_.size)))
        self.code_book_ = self._make_code_book(rng, self.classes_.size, n_bits)
        class_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        encoded = np.asarray([class_to_index[label] for label in labels], dtype=int)
        self.models_ = []
        for bit_index in range(n_bits):
            binary_labels = (self.code_book_[encoded, bit_index] > 0).astype(int)
            model = LinearSVC(C=self.C, class_weight="balanced", max_iter=5000, random_state=None if self.random_state is None else int(self.random_state) + bit_index)
            if sample_weight is None:
                model.fit(features, binary_labels)
            else:
                model.fit(features, binary_labels, sample_weight=np.asarray(sample_weight, dtype=float))
            self.models_.append(model)
        return self

    def decision_function(self, features):
        if self.models_ is None or self.code_book_ is None:
            raise RuntimeError("SampleWeightedECOCLinearSVM must be fitted before scoring.")
        features = np.asarray(features, dtype=float)
        margins = np.column_stack([model.decision_function(features) for model in self.models_])
        distances = np.sum(np.square(margins[:, None, :] - self.code_book_[None, :, :]), axis=2)
        return -distances

    def predict(self, features):
        if self.classes_ is None:
            raise RuntimeError("SampleWeightedECOCLinearSVM must be fitted before prediction.")
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]

    @staticmethod
    def _make_code_book(rng, n_classes: int, n_bits: int) -> np.ndarray:
        for _attempt in range(256):
            code_book = rng.choice(np.asarray([-1.0, 1.0]), size=(n_classes, n_bits))
            if np.unique(code_book, axis=0).shape[0] == n_classes and np.all(np.any(code_book > 0, axis=0)) and np.all(np.any(code_book < 0, axis=0)):
                return code_book
        # Deterministic fallback: one-vs-rest bits plus random extra bits.
        code_book = -np.ones((n_classes, max(n_bits, n_classes)), dtype=float)
        for class_index in range(n_classes):
            code_book[class_index, class_index] = 1.0
        if n_bits < n_classes:
            return code_book[:, :n_bits]
        return code_book


def _normalize_ecoc_param(classifier_param, random_state):
    if classifier_param is None:
        classifier_param = PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS["ecoc-linear-svm"]
    if isinstance(classifier_param, dict):
        C = float(classifier_param.get("C", 1.0))
        code_size = float(classifier_param.get("code_size", 2.0))
        seed = classifier_param.get("random_state", random_state)
    else:
        C = float(classifier_param)
        code_size = 2.0
        seed = random_state
    return C, code_size, None if seed is None else int(seed)


def _build_ecoc_linear_svm(_features, _labels, classifier_param, random_state):
    C, code_size, seed = _normalize_ecoc_param(classifier_param, random_state)
    return SampleWeightedECOCLinearSVM(C=C, code_size=code_size, random_state=seed)


CLASSIFIER_REGISTRY = {
    **REPTRACE_CLASSIFIER_REGISTRY,
    "gaussian-naive-bayes": ClassifierSpec(_build_gaussian_naive_bayes),
    "multinomial-logistic-weighted": ClassifierSpec(_build_weighted_multinomial_logistic),
    "regularized-qda": ClassifierSpec(_build_regularized_qda),
    "shrinkage-prototype": ClassifierSpec(_build_shrinkage_prototype),
    "one-vs-one-linear-svm": ClassifierSpec(_build_one_vs_one_linear_svm),
    "one-vs-rest-linear-svm": ClassifierSpec(_build_one_vs_rest_linear_svm),
    "ecoc-linear-svm": ClassifierSpec(_build_ecoc_linear_svm),
}

__all__ = [
    "CLASSIFIER_REGISTRY",
    "DEFAULT_CLASSIFIER_PARAMS",
    "ClassifierSpec",
    "CorrelationPrototypeClassifier",
    "DecodedLabelClassifier",
    "ShrinkagePrototypeClassifier",
    "SampleWeightedECOCLinearSVM",
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


def _fit_with_optional_sample_weight(model, features, labels, sample_weight=None):
    if sample_weight is None:
        return model.fit(features, labels)
    sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    if sample_weight.shape[0] != np.asarray(labels).shape[0]:
        raise ValueError("sample_weight must contain one value per training row.")
    try:
        if isinstance(model, Pipeline):
            final_step_name = model.steps[-1][0]
            return model.fit(features, labels, **{f"{final_step_name}__sample_weight": sample_weight})
        return model.fit(features, labels, sample_weight=sample_weight)
    except TypeError:
        warnings.warn(
            f"Classifier {model.__class__.__name__} does not accept sample_weight; fitting without weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model.fit(features, labels)


def train_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
    sample_weight=None,
):
    """Build and fit a classifier from the PyMEGDec-extended registry."""

    registry = CLASSIFIER_REGISTRY if registry is None else registry
    features = np.asarray(features)
    labels = np.asarray(labels).ravel()
    try:
        classifier_spec = registry[classifier]
    except KeyError:
        # Preserve upstream error text for callers that rely on it.
        return reptrace_classifiers.train_classifier(features, labels, classifier, classifier_param, random_state=random_state, registry=registry)
    model = classifier_spec.builder(features, labels, classifier_param, random_state)
    if classifier_spec.fits_in_builder:
        if sample_weight is not None:
            warnings.warn(f"Classifier {classifier!r} fits inside its builder; sample_weight was not passed.", RuntimeWarning, stacklevel=2)
        return model
    return _fit_with_optional_sample_weight(model, features, labels, sample_weight=sample_weight)


def train_multiclass_classifier(
    features,
    labels,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
    sample_weight=None,
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
        sample_weight=sample_weight,
    )
    return DecodedLabelClassifier(model, classes)


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from reptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
