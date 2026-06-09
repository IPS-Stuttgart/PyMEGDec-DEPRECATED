import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _fit_latent_logistic_head,
    _logistic_head_score_matrix,
)


def test_source_logistic_head_refit_returns_class_aligned_scores():
    classes = np.asarray([1, 2, 3], dtype=int)
    train_latent = np.asarray(
        [
            [-2.0, 0.0],
            [-1.6, 0.1],
            [0.0, 1.8],
            [0.1, 1.4],
            [1.8, -0.2],
            [1.5, 0.0],
        ]
    )
    train_labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)
    config = LatentAutoencoderConfig(latent_head_refit="source_logistic", latent_head_refit_c_values=(0.3,))

    model, metadata = _fit_latent_logistic_head(train_latent, train_labels, None, None, classes, config)
    scores = _logistic_head_score_matrix(model, train_latent, classes)

    assert model is not None
    assert "standardscaler" in model.named_steps
    assert metadata["latent_head_refit_status"] == "ok"
    assert metadata["latent_head_refit_selected_c"] == 0.3
    assert scores.shape == (6, 3)
    assert classes[np.argmax(scores, axis=1)].tolist() == train_labels.tolist()


def test_validation_selected_source_logistic_uses_source_validation_metric():
    classes = np.asarray([1, 2, 3], dtype=int)
    train_latent = np.asarray([[-2.0, 0.0], [-1.5, 0.2], [0.0, 2.0], [0.2, 1.5], [2.0, 0.0], [1.6, -0.2]])
    train_labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)
    validation_latent = np.asarray([[-1.8, 0.0], [0.0, 1.7], [1.7, 0.0]])
    validation_labels = np.asarray([1, 2, 3], dtype=int)
    config = LatentAutoencoderConfig(latent_head_refit="validation_selected_source_logistic", latent_head_refit_c_values=(0.03, 1.0))

    model, metadata = _fit_latent_logistic_head(train_latent, train_labels, validation_latent, validation_labels, classes, config)

    assert model is not None
    assert "standardscaler" in model.named_steps
    assert metadata["latent_head_refit_status"] == "ok"
    assert metadata["latent_head_refit_selected_c"] in {0.03, 1.0}
    assert metadata["latent_head_refit_validation_balanced_accuracy"] >= 2.0 / 3.0
