"""Utilities for MEG decoding experiments."""

from pymegdec.alpha_metrics import (
    AlphaMetricConfig,
    compute_alpha_metrics,
    export_participant_alpha_metrics,
)
from pymegdec.alpha_movement import (
    AlphaMovementConfig,
    compute_alpha_movement,
    export_alpha_movement,
)
from pymegdec.alpha_movement_analysis import (
    AlphaMovementAnalysisConfig,
    analyze_alpha_movement_windows,
    export_alpha_movement_analysis,
    summarize_alpha_movement_effects,
)
from pymegdec.alpha_signal import extract_phase, extract_time_basis
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import DATA_DIR_ENV_VAR, resolve_data_folder
from pymegdec import _stimulus_trial_sampling as _stimulus_trial_sampling  # noqa: F401
from pymegdec.model_transfer import (
    evaluate_model_transfer,
    get_original_feature_importance,
)
from pymegdec.reaction_time_analysis import (
    AlphaReactionTimeExportConfig,
    ReactionTimeCsvConfig,
    ReactionTimeUnavailableError,
    analyze_alpha_reaction_times,
    join_alpha_reaction_times,
)
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    evaluate_cross_subject_stimulus_smoke,
    evaluate_nested_cross_subject_stimulus,
    export_cross_subject_stimulus_smoke,
    export_nested_cross_subject_stimulus,
    make_cross_subject_candidate_configs,
    summarize_cross_subject_confusion_category_enrichment,
    summarize_cross_subject_confusion_category_matrix,
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_stimulus_smoke,
    summarize_nested_cross_subject_stimulus,
)
from pymegdec.stimulus_decoding import (
    TRANSFER_DIRECTIONS,
    StimulusDecodingConfig,
    evaluate_participant_stimulus_decoding_diagnostics,
    evaluate_participant_stimulus_onset_scan,
    evaluate_participant_stimulus_temporal_generalization,
    evaluate_time_resolved_stimulus_transfer,
    export_stimulus_onset_scan,
    export_stimulus_temporal_generalization,
    export_time_resolved_stimulus_decoding,
    summarize_stimulus_decoding,
    summarize_stimulus_decoding_peaks,
    summarize_stimulus_onset_events,
    summarize_stimulus_onset_scan,
    summarize_stimulus_prediction_diagnostics,
    summarize_stimulus_temporal_generalization,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATA_DIR_ENV_VAR",
    "AlphaMetricConfig",
    "AlphaMovementConfig",
    "AlphaMovementAnalysisConfig",
    "AlphaReactionTimeExportConfig",
    "ReactionTimeCsvConfig",
    "ReactionTimeUnavailableError",
    "StimulusDecodingConfig",
    "CrossSubjectStimulusConfig",
    "TRANSFER_DIRECTIONS",
    "analyze_alpha_reaction_times",
    "analyze_alpha_movement_windows",
    "compute_alpha_movement",
    "compute_alpha_metrics",
    "cross_validate_single_dataset",
    "evaluate_participant_stimulus_decoding_diagnostics",
    "evaluate_participant_stimulus_onset_scan",
    "evaluate_participant_stimulus_temporal_generalization",
    "evaluate_cross_subject_stimulus_smoke",
    "evaluate_nested_cross_subject_stimulus",
    "evaluate_model_transfer",
    "evaluate_time_resolved_stimulus_transfer",
    "export_alpha_movement",
    "export_alpha_movement_analysis",
    "export_participant_alpha_metrics",
    "export_stimulus_onset_scan",
    "export_stimulus_temporal_generalization",
    "export_time_resolved_stimulus_decoding",
    "export_cross_subject_stimulus_smoke",
    "export_nested_cross_subject_stimulus",
    "extract_phase",
    "extract_time_basis",
    "get_original_feature_importance",
    "join_alpha_reaction_times",
    "make_cross_subject_candidate_configs",
    "resolve_data_folder",
    "summarize_stimulus_decoding",
    "summarize_stimulus_decoding_peaks",
    "summarize_stimulus_onset_events",
    "summarize_stimulus_onset_scan",
    "summarize_stimulus_prediction_diagnostics",
    "summarize_stimulus_temporal_generalization",
    "summarize_cross_subject_confusion_pairs",
    "summarize_cross_subject_confusion_category_enrichment",
    "summarize_cross_subject_confusion_category_matrix",
    "summarize_cross_subject_stimulus_smoke",
    "summarize_nested_cross_subject_stimulus",
    "summarize_alpha_movement_effects",
]
