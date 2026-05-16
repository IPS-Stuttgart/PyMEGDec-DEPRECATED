"""Conservative top-k rank metrics for cross-subject stimulus decoding."""

from __future__ import annotations

import numpy as np


def _cross_subject_ranked_label_metrics(true_labels, class_scores, score_classes):
    from pymegdec import stimulus_cross_subject as cross_subject

    true_label_ranks = cross_subject._true_label_ranks(true_labels, class_scores, score_classes)
    finite_ranks = true_label_ranks[np.isfinite(true_label_ranks)]

    # A score matrix with no scored classes means top-k metrics are undefined.
    # Missing true-label ranks in a non-empty score matrix, however, are trials
    # whose true class was not scoreable by the fitted model. Those trials must
    # stay in the top-k denominator and count as top-k failures instead of being
    # silently removed.
    if true_label_ranks.size == 0 or class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return {
            "true_label_ranks": true_label_ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    return {
        "true_label_ranks": true_label_ranks,
        "top2_accuracy": float(np.mean(true_label_ranks <= 2)),
        "top3_accuracy": float(np.mean(true_label_ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _hyperalignment_topk_and_rank_metrics(true_labels, class_scores, score_classes):
    from pymegdec import stimulus_hyperalignment as hyperalignment

    if class_scores.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    ranks = np.asarray(hyperalignment._true_label_ranks(true_labels, class_scores, score_classes), dtype=float)
    finite_ranks = ranks[np.isfinite(ranks)]

    # In a non-empty score matrix, NaN ranks identify trials whose true class is
    # absent from the model's score classes. Keep those trials in the top-k
    # denominator and count them as top-k failures.
    if ranks.size == 0:
        return {"top2_accuracy": np.nan, "top3_accuracy": np.nan, "mean_true_label_rank": np.nan}
    return {
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
    }


def apply_conservative_topk_metrics():
    """Install conservative rank metrics in decoding modules.

    The public decoding functions look up these private helpers at runtime, so
    replacing the helpers here updates the corresponding evaluation paths while
    keeping no-score-matrix cases explicitly undefined.
    """

    from pymegdec import stimulus_cross_subject as cross_subject
    from pymegdec import stimulus_hyperalignment as hyperalignment

    cross_subject._ranked_label_metrics = _cross_subject_ranked_label_metrics
    hyperalignment._topk_and_rank_metrics = _hyperalignment_topk_and_rank_metrics
