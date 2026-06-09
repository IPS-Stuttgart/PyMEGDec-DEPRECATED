from pathlib import Path

from pymegdec.stimulus_latent_autoencoder import (
    LATENT_TRAINING_PRESET_CHOICES,
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
    assert config.input_dropout >= 0.05
    assert config.prediction_balance_weight >= 0.02
    assert config.prediction_balance_temperature <= 0.10
    assert config.logit_mean_center_weight >= 0.003
    assert config.class_bias_l2_weight >= 0.003
    assert config.soft_macro_recall_weight >= 0.02
    assert config.validation_source_count >= 4
    assert config.validation_prediction_balance_weight >= 0.03
    assert config.validation_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.validation_min_epochs >= 6
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
    assert config.validation_min_epochs >= 6
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
    assert config.class_bias_l2_weight >= 0.003


def test_anti_collapse_head_blend_preset_adds_source_validation_head_blend():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_head_blend",
    )

    assert config.training_preset == "anti_collapse_head_blend"
    assert config.score_calibration == "validation_selected_guarded"
    assert config.prediction_postprocessing == "validation_selected_balanced_assignment"
    assert config.latent_head_refit == "validation_selected_source_logistic_blend"
    assert config.latent_head_refit_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.latent_head_refit_c_values == (0.03, 0.1, 0.3, 1.0, 3.0)
    assert config.latent_head_refit_blend_alphas == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_manual_workflow_exposes_all_latent_training_presets():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "stimulus-latent-autoencoder.yml"
    workflow_text = workflow.read_text(encoding="utf-8")
    for preset in LATENT_TRAINING_PRESET_CHOICES:
        assert f"          - {preset}" in workflow_text


def test_anti_collapse_contrastive_preset_adds_source_only_latent_clustering():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_contrastive",
    )

    assert config.training_preset == "anti_collapse_contrastive"
    assert config.balanced_batch_sampling is True
    assert config.subject_class_balanced_batch_sampling is True
    assert config.label_smoothing >= 0.05
    assert config.prediction_balance_weight >= 0.02
    assert config.soft_macro_recall_weight >= 0.02
    assert config.validation_source_count >= 4
    assert config.validation_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.validation_min_epochs >= 6
    assert config.final_min_epochs >= 8
    assert config.supervised_contrastive_weight >= 0.02
    assert config.supervised_contrastive_temperature <= 0.20


def test_anti_collapse_rank_rescue_preset_adds_low_margin_rescue_stack():
    config = _apply_latent_training_preset(
        LatentAutoencoderConfig(),
        "anti_collapse_rank_rescue",
    )

    assert config.training_preset == "anti_collapse_rank_rescue"
    assert config.balanced_batch_sampling is True
    assert config.subject_class_balanced_batch_sampling is True
    assert config.validation_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.soft_worst_class_recall_weight >= 0.01
    assert config.margin_loss_weight >= 0.005
    assert config.confidence_penalty_weight >= 0.002
    assert config.latent_head_refit == "validation_selected_source_logistic_blend"
    assert config.latent_head_refit_selection_metric == "balanced_top2_top3_rank_balance"
    assert config.score_calibration == "validation_selected_guarded"
    assert config.prediction_postprocessing == "validation_selected_balanced_assignment"
    assert config.prediction_postprocessing_margin_thresholds == (0.1, 0.2, 0.3, 0.5, 0.75)


def test_none_preset_preserves_explicit_config_values():
    original = LatentAutoencoderConfig(
        label_smoothing=0.2,
        validation_source_count=6,
        validation_min_epochs=5,
        input_dropout=0.07,
    )

    assert _apply_latent_training_preset(original, "none") == original
