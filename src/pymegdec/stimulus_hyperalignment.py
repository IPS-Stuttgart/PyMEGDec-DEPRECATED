"""Public facade for NeuRepTrace-backed Procrustes hyperalignment stimulus decoding."""

from __future__ import annotations

from dataclasses import replace
import sys

from neureptrace.decoding.hyperalignment import (
    CLASS_ALIGNMENT_SAMPLE_MODES,
    class_alignment_matrices as _nrt_class_alignment_matrices,
    fit_class_hyperalignment as _nrt_fit_class_hyperalignment,
    fit_projection_to_hyperalignment as _nrt_fit_projection_to_hyperalignment,
)
from neureptrace.decoding.hyperalignment_initialization import (
    HYPERALIGNMENT_INITIALIZATION_MODES,
    fit_class_hyperalignment as _fit_initialized_class_hyperalignment,
    normalize_hyperalignment_initialization,
)
from neureptrace.decoding.windowed import (
    fit_window_model as _nrt_fit_window_model,
    predict_window_model as _nrt_predict_window_model,
    transform_window_features as _nrt_transform_window_features,
)

from pymegdec import _stimulus_hyperalignment_legacy as _impl
from pymegdec._reptrace_score_overrides import install_hyperalignment

_ORIGINAL_EVALUATE_HYPERALIGNMENT_OUTER_FOLD = _impl._evaluate_hyperalignment_outer_fold
_ORIGINAL_NORMALIZED_HYPERALIGNMENT_CONFIG = _impl._normalized_hyperalignment_config


def _normalized_hyperalignment_config(config):
    normalized = _ORIGINAL_NORMALIZED_HYPERALIGNMENT_CONFIG(config)
    initialization = normalize_hyperalignment_initialization(normalized.hyperalignment_initialization)
    if initialization == normalized.hyperalignment_initialization:
        return normalized
    return replace(normalized, hyperalignment_initialization=initialization)


def _evaluate_hyperalignment_outer_fold(*args, **kwargs):
    config = args[4] if len(args) >= 5 else kwargs.get("config")
    initialization = normalize_hyperalignment_initialization(config.hyperalignment_initialization)

    original_fit_class_hyperalignment = _impl.fit_class_hyperalignment

    def configured_fit_class_hyperalignment(*fit_args, **fit_kwargs):
        fit_kwargs.setdefault("initialization", initialization)
        return _fit_initialized_class_hyperalignment(*fit_args, **fit_kwargs)

    _impl.fit_class_hyperalignment = configured_fit_class_hyperalignment
    try:
        return _ORIGINAL_EVALUATE_HYPERALIGNMENT_OUTER_FOLD(*args, **kwargs)
    finally:
        _impl.fit_class_hyperalignment = original_fit_class_hyperalignment


# Bind the legacy BUSH-MEG orchestration to NeuRepTrace-owned reusable kernels.
_impl.CLASS_ALIGNMENT_SAMPLE_MODES = CLASS_ALIGNMENT_SAMPLE_MODES
_impl.class_alignment_matrices = _nrt_class_alignment_matrices
_impl.fit_class_hyperalignment = _nrt_fit_class_hyperalignment
_impl.fit_projection_to_hyperalignment = _nrt_fit_projection_to_hyperalignment
_impl.fit_window_model = _nrt_fit_window_model
_impl.predict_window_model = _nrt_predict_window_model
_impl.transform_window_features = _nrt_transform_window_features
_impl.HYPERALIGNMENT_INITIALIZATION_MODES = HYPERALIGNMENT_INITIALIZATION_MODES
_impl._normalize_hyperalignment_initialization = normalize_hyperalignment_initialization
_impl._normalized_hyperalignment_config = _normalized_hyperalignment_config
_impl._evaluate_hyperalignment_outer_fold = _evaluate_hyperalignment_outer_fold
install_hyperalignment(_impl)

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
