"""Backward-compatible classifier imports.

Classifier implementations now live in :mod:`reptrace.decoding.classifiers`.
This module preserves historical ``pymegdec.classifiers`` imports without
owning classifier implementations.
"""

from __future__ import annotations

from typing import Any

from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY,
    DEFAULT_CLASSIFIER_PARAMS,
    ClassifierSpec,
    CorrelationPrototypeClassifier,
    DecodedLabelClassifier,
    _build_pytorch_data_loaders,
    encode_classifier_labels,
    get_default_classifier_param,
    positive_class_score,
    prediction_scores,
    should_use_default_classifier_param,
    train_binary_svm,
    train_classifier,
    train_for_stimulus_lasso_glm,
    train_gradient_boosting,
    train_lasso_logistic,
    train_multiclass_classifier,
)

_DecodedLabelClassifier = DecodedLabelClassifier
_encode_classifier_labels = encode_classifier_labels

__all__ = [
    "CLASSIFIER_REGISTRY",
    "DEFAULT_CLASSIFIER_PARAMS",
    "ClassifierSpec",
    "CorrelationPrototypeClassifier",
    "DecodedLabelClassifier",
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


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from reptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
