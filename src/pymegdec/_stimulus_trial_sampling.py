"""NeuRepTrace-backed trial-selection bridge for stimulus decoding."""

from __future__ import annotations

from neureptrace.decoding.sampling import (
    normalize_class_limit_seed,
    normalize_class_limit_selection,
    select_class_limited_indices,
)

from pymegdec import _stimulus_cross_subject_core as _core

DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"
DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0
TRIAL_SELECTION_MODES = ("random", "first")


def _selected_trial_indices(
    labels,
    max_trials_per_class,
    *,
    selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    participant=None,
    seed_context=None,
):
    """Select trial indices through NeuRepTrace's generic class-limited sampler."""

    if seed_context is None and participant is not None:
        seed_context = participant
    return select_class_limited_indices(
        labels,
        max_trials_per_class,
        selection=selection,
        seed=seed,
        seed_context=seed_context,
    )


def _normalize_trial_selection(value):
    return normalize_class_limit_selection(value)


def _normalize_trial_selection_seed(value):
    return normalize_class_limit_seed(value)


def _install_reptrace_sampling_bridge():
    _impl = _core._impl
    for module in (_core, _impl):
        setattr(module, "DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)
        setattr(module, "DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)
        setattr(module, "TRIAL_SELECTION_MODES", TRIAL_SELECTION_MODES)
        setattr(module, "_selected_trial_indices", _selected_trial_indices)
        setattr(module, "_normalize_trial_selection", _normalize_trial_selection)
        setattr(module, "_normalize_trial_selection_seed", _normalize_trial_selection_seed)


_install_reptrace_sampling_bridge()
