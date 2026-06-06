import numpy as np

from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _blend_score_matrices,
    _fit_latent_logistic_head,
)


def test_blend_score_matrices_preserves_endpoints():
    neural_scores = np.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    logistic_scores = np.asarray([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])

    np.testing.assert_allclose(_blend_score_matrices(neural_scores, logistic_scores, 0.0), neural_scores)
    np.testing.assert_allclose(_blend_score_matrices(neural_scores, logistic_scores, 1.0), logistic_scores)
    blended = _blend_score_matrices(neural_scores, logistic_scores, 0.5)

    assert blended.shape == neural_scores.shape
    assert np.all(np.isfinite(blended))


def test_validation_selected_source_logistic_blend_can_keep_neural_head():
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
    # The logistic probe will be poor on this deliberately flipped validation
    # latent arrangement, while the supplied neural head scores are perfect.
    validation_latent = np.asarray([[1.8, -0.2], [0.0, 1.7], [-1.8, 0.0]])
    validation_labels = np.asarray([1, 2, 3], dtype=int)
    validation_neural_scores = np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    config = LatentAutoencoderConfig(
        latent_head_refit="validation_selected_source_logistic_blend",
        latent_head_refit_c_values=(1.0,),
        latent_head_refit_blend_alphas=(0.0, 1.0),
    )

    model, metadata = _fit_latent_logistic_head(
        train_latent,
        train_labels,
        validation_latent,
        validation_labels,
        classes,
        config,
        validation_base_scores=validation_neural_scores,
    )

    assert model is not None
    assert metadata["latent_head_refit_selected_blend_alpha"] == 0.0
    assert metadata["latent_head_refit_validation_balanced_accuracy"] == 1.0
