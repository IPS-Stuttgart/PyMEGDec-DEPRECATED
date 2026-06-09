import numpy as np

from pymegdec.stimulus_latent_autoencoder import LatentAutoencoderConfig
from pymegdec.stimulus_source_inner_stacking import (
    SourceInnerStackConfig,
    _postprocess_stacked_predictions,
)


def _stack_config(**kwargs):
    return SourceInnerStackConfig(
        compact_candidate_configs=(),
        latent_config=LatentAutoencoderConfig(),
        **kwargs,
    )


def _postprocessing_inputs():
    classes = np.asarray([1, 2, 3])
    scores = np.asarray(
        [
            [3.0, 2.0, 0.0],
            [2.9, 2.8, 0.0],
            [2.0, 0.0, 4.0],
        ]
    )
    source_labels = np.repeat(classes, 4)
    return scores, classes, source_labels


def test_stacker_postprocessing_none_returns_argmax_predictions():
    scores, classes, source_labels = _postprocessing_inputs()

    predictions, metadata = _postprocess_stacked_predictions(
        scores,
        classes,
        source_labels,
        _stack_config(),
    )

    assert predictions.tolist() == [1, 1, 3]
    assert metadata["stacker_prediction_postprocessing_status"] == "not_requested"


def test_stacker_postprocessing_can_reuse_source_prior_balanced_assignment():
    scores, classes, source_labels = _postprocessing_inputs()

    predictions, metadata = _postprocess_stacked_predictions(
        scores,
        classes,
        source_labels,
        _stack_config(prediction_postprocessing="source_prior_balanced_assignment"),
    )

    assert sorted(predictions.tolist()) == [1, 2, 3]
    assert metadata["stacker_prediction_postprocessing_status"] == "ok"
    assert metadata["stacker_prediction_postprocessing_selected_method"] == "source_prior_balanced_assignment"
