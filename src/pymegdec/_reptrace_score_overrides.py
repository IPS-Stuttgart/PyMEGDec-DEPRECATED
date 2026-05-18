"""Thin PyMEGDec adapters for upstream RepTrace class-score helpers."""

from __future__ import annotations

import numpy as np
from reptrace.decoding.class_scores import class_score_matrix, predict_window_class_scores
from reptrace.metrics import rank_class_scores


def mcca_score_matrix(bundle, features):
    """Return upstream RepTrace per-class scores for a fitted M-CCA window bundle."""

    if hasattr(bundle, "pca_coeff") and hasattr(bundle, "train_features_mean"):
        return predict_window_class_scores(bundle, features)
    return class_score_matrix(bundle.model, features, fallback_labels=getattr(bundle, "train_labels", None))


def mcca_rank_metrics(scores, classes, y_true):
    """Return the legacy PyMEGDec tuple shape from RepTrace rank summaries."""

    summary = rank_class_scores(scores, classes, y_true, top_k=(2, 3), row_top_k=3, class_column="stimulus")
    top_k_accuracy = summary["top_k_accuracy"]
    return top_k_accuracy[2], top_k_accuracy[3], summary["mean_true_label_rank"], summary["rows"]


def hyperalignment_class_score_matrix(model_bundle, features):
    """Return class scores for hyperalignment, including predict-only models."""

    return predict_window_class_scores(model_bundle, features, predict_fallback=True)


def true_label_ranks(true_labels, class_scores, score_classes):
    """Return only true-label ranks using RepTrace's ranking semantics."""

    summary = rank_class_scores(class_scores, score_classes, true_labels, top_k=(), row_top_k=0)
    return summary["true_label_ranks"]


def topk_and_rank_metrics(true_labels, class_scores, score_classes):
    """Return PyMEGDec's top-k metric dict using upstream RepTrace ranking."""

    summary = rank_class_scores(class_scores, score_classes, true_labels, top_k=(2, 3), row_top_k=0)
    top_k_accuracy = summary["top_k_accuracy"]
    return {
        "top2_accuracy": top_k_accuracy[2],
        "top3_accuracy": top_k_accuracy[3],
        "mean_true_label_rank": summary["mean_true_label_rank"],
    }


def cross_subject_model_class_scores(model_bundle, features):
    """Return class scores in the legacy cross-subject empty-matrix shape."""

    scores, classes = predict_window_class_scores(model_bundle, features)
    if scores is None or classes is None:
        return np.full((np.asarray(features).shape[0], 0), np.nan, dtype=float), np.asarray([], dtype=int)
    return scores, classes


def cross_subject_ranked_label_metrics(true_labels, class_scores, score_classes):
    """Return PyMEGDec cross-subject rank metrics using RepTrace summaries."""

    summary = rank_class_scores(class_scores, score_classes, true_labels, top_k=(2, 3), row_top_k=0)
    top_k_accuracy = summary["top_k_accuracy"]
    return {
        "true_label_ranks": summary["true_label_ranks"],
        "top2_accuracy": top_k_accuracy[2],
        "top3_accuracy": top_k_accuracy[3],
        "mean_true_label_rank": summary["mean_true_label_rank"],
        "median_true_label_rank": summary["median_true_label_rank"],
    }


def install_mcca(impl):
    """Install upstream-backed M-CCA scoring helpers into the legacy module."""

    impl._score_matrix = mcca_score_matrix
    impl._rank_metrics = mcca_rank_metrics


def install_hyperalignment(impl):
    """Install upstream-backed hyperalignment scoring helpers."""

    impl._class_score_matrix = hyperalignment_class_score_matrix
    impl._topk_and_rank_metrics = topk_and_rank_metrics
    impl._true_label_ranks = true_label_ranks


def install_cross_subject(impl):
    """Install upstream-backed cross-subject scoring helpers."""

    impl._model_class_scores = cross_subject_model_class_scores
    impl._ranked_label_metrics = cross_subject_ranked_label_metrics
    impl._true_label_ranks = true_label_ranks
