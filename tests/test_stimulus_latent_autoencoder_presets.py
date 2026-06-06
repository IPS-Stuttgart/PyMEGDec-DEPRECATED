from pymegdec.stimulus_latent_autoencoder import (
    LatentAutoencoderConfig,
    _apply_latent_training_preset,
)


def test_anti_collapse_train_preset_enables_source_only_regularizers():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_train",
    )

    assert config.training_preset == "anti_collapse_train"
    assert config.balanced_batch_sampling is True
    assert config.label_smoothing >= 0.05
    assert config.prediction_balance_weight >= 0.02
    assert config.prediction_balance_temperature <= 0.10
    assert config.logit_mean_center_weight >= 0.003
    assert config.soft_macro_recall_weight >= 0.02
    assert config.validation_source_count >= 4
    assert config.validation_prediction_balance_weight >= 0.03
    assert config.validation_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.final_min_epochs >= 8
    assert config.score_calibration == "none"
    assert config.prediction_postprocessing == "none"


def test_anti_collapse_calibrated_preset_adds_guarded_source_validation_corrections():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_calibrated",
    )

    assert config.training_preset == "anti_collapse_calibrated"
    assert config.score_calibration == "validation_selected_guarded"
    assert config.score_calibration_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.prediction_postprocessing == "validation_selected_balanced_assignment"
    assert config.prediction_postprocessing_guard_tolerance == 0.0


def test_anti_collapse_refit_preset_adds_source_only_latent_logistic_probe():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_refit",
    )

    assert config.training_preset == "anti_collapse_refit"
    assert config.balanced_batch_sampling is True
    assert config.validation_source_count >= 4
    assert config.validation_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.latent_head_refit == "validation_selected_source_logistic"
    assert config.latent_head_refit_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.score_calibration == "validation_selected_guarded"
    assert config.score_calibration_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.score_calibration_final_refit is True
    assert config.prediction_postprocessing == "validation_selected_balanced_assignment"


def test_anti_collapse_head_refit_preset_adds_source_validation_logistic_head():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_head_refit",
    )

    assert config.training_preset == "anti_collapse_head_refit"
    assert config.score_calibration == "validation_selected_guarded"
    assert config.prediction_postprocessing == "validation_selected_balanced_assignment"
    assert config.latent_head_refit == "validation_selected_source_logistic"
    assert config.latent_head_refit_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.latent_head_refit_c_values == (0.03, 0.1, 0.3, 1.0, 3.0)


def test_none_preset_preserves_explicit_config_values():
    original = LatentAutoencoderConfig(
        label_smoothing=0.2,
        validation_source_count=6,
    )

    assert _apply_latent_training_preset(original, "none") == original
