from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pymegdec import stimulus_mcca as mcca


def _run_score_matrix(model, features, *, train_labels=(1, 2, 1, 2)):
    bundle = SimpleNamespace(model=model, train_labels=np.asarray(train_labels))
    with patch.object(mcca, "transform_window_features", lambda _bundle, _features: np.asarray(_features)):
        return mcca._score_matrix(bundle, np.asarray(features))


def test_mcca_score_matrix_expands_binary_decision_function_to_class_scores():
    class BinaryDecisionModel:
        classes_ = np.asarray([1, 2])

        def decision_function(self, features):
            return np.asarray([-2.0, 0.5, 1.5])[: features.shape[0]]

    scores, classes = _run_score_matrix(BinaryDecisionModel(), np.zeros((3, 2)))

    np.testing.assert_array_equal(classes, np.asarray([1, 2]))
    np.testing.assert_allclose(scores, np.asarray([[2.0, -2.0], [-0.5, 0.5], [-1.5, 1.5]]))

    top2, top3, mean_rank, rows = mcca._rank_metrics(scores, classes, np.asarray([1, 2, 1]))
    assert top2 == 1.0
    assert top3 == 1.0
    assert np.isclose(mean_rank, 4.0 / 3.0)
    assert rows[0]["rank1_stimulus"] == 1
    assert rows[1]["rank1_stimulus"] == 2
    assert rows[2]["true_label_rank"] == 2


def test_mcca_score_matrix_uses_train_label_fallback_for_binary_decision_scores():
    class BinaryDecisionModelWithoutClasses:
        def decision_function(self, features):
            return np.asarray([0.25])[: features.shape[0]]

    scores, classes = _run_score_matrix(
        BinaryDecisionModelWithoutClasses(),
        np.zeros((1, 2)),
        train_labels=(5, 9, 5, 9),
    )

    np.testing.assert_array_equal(classes, np.asarray([5, 9]))
    np.testing.assert_allclose(scores, np.asarray([[-0.25, 0.25]]))


def test_mcca_score_matrix_falls_back_to_probabilities_when_decision_shape_is_invalid():
    class InvalidDecisionValidProbabilityModel:
        classes_ = np.asarray([1, 2, 3])

        def decision_function(self, features):
            return np.ones((features.shape[0], 2), dtype=float)

        def predict_proba(self, features):
            return np.tile(np.asarray([[0.1, 0.7, 0.2]]), (features.shape[0], 1))

    scores, classes = _run_score_matrix(InvalidDecisionValidProbabilityModel(), np.zeros((2, 2)))

    np.testing.assert_array_equal(classes, np.asarray([1, 2, 3]))
    np.testing.assert_allclose(scores, np.asarray([[0.1, 0.7, 0.2], [0.1, 0.7, 0.2]]))


def test_mcca_score_matrix_rejects_one_dimensional_nonbinary_decision_scores():
    class NonBinaryDecisionModel:
        classes_ = np.asarray([1, 2, 3])

        def decision_function(self, features):
            return np.arange(features.shape[0], dtype=float)

    scores, classes = _run_score_matrix(NonBinaryDecisionModel(), np.zeros((2, 2)))

    assert scores is None
    assert classes is None
