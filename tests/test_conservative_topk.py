import numpy as np

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec import stimulus_hyperalignment as hyperalignment


def test_cross_subject_topk_metrics_count_unscoreable_true_labels_as_failures():
    metrics = cross_subject._ranked_label_metrics(
        np.asarray([0, 1, 2]),
        np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        ),
        np.asarray([0, 1]),
    )

    assert np.allclose(metrics["true_label_ranks"][:2], [1.0, 1.0])
    assert np.isnan(metrics["true_label_ranks"][2])
    assert metrics["top2_accuracy"] == 2 / 3
    assert metrics["top3_accuracy"] == 2 / 3
    assert metrics["mean_true_label_rank"] == 1.0
    assert metrics["median_true_label_rank"] == 1.0


def test_cross_subject_topk_metrics_without_score_columns_are_undefined():
    metrics = cross_subject._ranked_label_metrics(
        np.asarray([0, 1]),
        np.empty((2, 0)),
        np.asarray([], dtype=int),
    )

    assert np.isnan(metrics["top2_accuracy"])
    assert np.isnan(metrics["top3_accuracy"])
    assert np.isnan(metrics["mean_true_label_rank"])
    assert np.isnan(metrics["median_true_label_rank"])


def test_hyperalignment_topk_metrics_count_unscoreable_true_labels_as_failures():
    metrics = hyperalignment._topk_and_rank_metrics(
        np.asarray([0, 1, 2]),
        np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        ),
        np.asarray([0, 1]),
    )

    assert metrics["top2_accuracy"] == 2 / 3
    assert metrics["top3_accuracy"] == 2 / 3
    assert metrics["mean_true_label_rank"] == 1.0
