"""Warnings and helpers for PyMEGDec workflows kept as legacy scripts."""

from __future__ import annotations

import warnings


class PyMEGDecLegacyWorkflowWarning(FutureWarning):
    """Warning for workflows that are intentionally not part of the NeuRepTrace migration."""


_ALPHA_PHASEOUT_MESSAGE = (
    "{command} is a legacy PyMEGDec alpha-analysis workflow. It is kept for "
    "reproducing the Bush-MEG paper-specific alpha/movement/reaction-time "
    "analyses, but it is not planned as a NeuRepTrace migration target. "
    "For the PyMEGDec phase-out, migrate generic decoding and FieldTrip/MAT "
    "loading to NeuRepTrace, and keep or archive alpha analyses in a "
    "project-specific analysis repository."
)


def warn_legacy_alpha_workflow(command: str) -> None:
    """Warn that an alpha workflow is legacy and project-specific.

    The warning category subclasses ``FutureWarning`` rather than
    ``DeprecationWarning`` so command-line users see the phase-out message by
    default.  The commands continue to run unchanged; this marks the ownership
    decision needed to phase out PyMEGDec without silently moving paper-specific
    alpha analyses into NeuRepTrace.
    """

    warnings.warn(
        _ALPHA_PHASEOUT_MESSAGE.format(command=command),
        PyMEGDecLegacyWorkflowWarning,
        stacklevel=2,
    )
