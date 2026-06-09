from __future__ import annotations

import numpy as np

from pymegdec.stimulus_latent_autoencoder import LatentAutoencoderConfig, _prediction_rows


def test_prediction_rows_emit_artifact_ensemble_ready_score_and_rank_aliases_for_one_based_labels():
    rows = _prediction_rows(
        test_participant=1,
        true_labels=np.asarray([1, 2]),
        predicted_labels=np.asarray([2, 2]),
        scores=np.asarray(
            [
                [0.40, 0.60],
                [0.10, 0.90],
            ]
        ),
        classes=np.asarray([1, 2]),
        config=LatentAutoencoderConfig(),
        pca_components=2,
        pca_explained_variance_percent=100.0,
    )

    first, second = rows
    assert first["trial"] == 0
    assert first["test_trial_index"] == 0
    assert first["true_label"] == 1
    assert first["true_stimulus"] == 1
    assert first["predicted_label"] == 2
    assert first["predicted_stimulus"] == 2
    assert first["true_label_rank"] == 2.0
    assert first["top2_correct"] is True
    assert first["top3_correct"] is True
    assert first["score_class_1"] == 0.40
    assert first["score_1"] == 0.40
    assert first["score_class_2"] == 0.60
    assert first["score_2"] == 0.60
    assert first["rank_class_1"] == 2
    assert first["rank_1"] == 2
    assert first["rank_class_2"] == 1
    assert first["rank_2"] == 1
    assert second["true_label_rank"] == 1.0
    assert second["correct"] is True


def test_prediction_rows_keep_display_labels_shifted_for_zero_based_labels():
    rows = _prediction_rows(
        test_participant=1,
        true_labels=np.asarray([0]),
        predicted_labels=np.asarray([0]),
        scores=np.asarray([[0.75, 0.25]]),
        classes=np.asarray([0, 1]),
        config=LatentAutoencoderConfig(),
        pca_components=2,
        pca_explained_variance_percent=100.0,
    )

    row = rows[0]
    assert row["true_label"] == 0
    assert row["true_stimulus"] == 1
    assert row["score_class_0"] == 0.75
    assert row["score_1"] == 0.75
    assert row["rank_class_0"] == 1
    assert row["rank_1"] == 1
