"""Public facade for NeuRepTrace M-CCA stimulus decoding."""

from __future__ import annotations

import sys

from neureptrace.decoding.mcca import CLASS_ALIGNMENT_SAMPLE_MODES, fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.windowed import (
    fit_window_model as _nrt_fit_window_model,
    predict_window_model as _nrt_predict_window_model,
    transform_window_features as _nrt_transform_window_features,
)

from pymegdec import _stimulus_mcca_legacy as _impl
from pymegdec._reptrace_score_overrides import install_mcca

# Bind the legacy BUSH-MEG orchestration to NeuRepTrace-owned reusable kernels.
_impl.CLASS_ALIGNMENT_SAMPLE_MODES = CLASS_ALIGNMENT_SAMPLE_MODES
_impl.fit_class_mcca = fit_class_mcca
_impl.class_alignment_matrix = class_alignment_matrix
_impl.fit_target_mcca_projection = fit_target_mcca_projection
_impl.fit_window_model = _nrt_fit_window_model
_impl.predict_window_model = _nrt_predict_window_model
_impl.transform_window_features = _nrt_transform_window_features

install_mcca(_impl)

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
