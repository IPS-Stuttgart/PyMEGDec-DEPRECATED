"""Deprecation helpers for the PyMEGDec-to-NeuRepTrace transition.

PyMEGDec is kept as a compatibility shell while dataset loading, decoding, and
probability-observation exports move into NeuRepTrace. CLI warnings are emitted
as ``FutureWarning`` rather than ``DeprecationWarning`` so end users see them in
normal command-line use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import TypeVar
import warnings

T = TypeVar("T")
CommandHandler = Callable[[Sequence[str] | None, str | None], int]

PYMEGDEC_DEPRECATION_MESSAGE = (
    "PyMEGDec is now a compatibility layer. Prefer NeuRepTrace dataset YAML "
    "workflows for new analyses; PyMEGDec-specific commands will be removed "
    "after migration parity has been established."
)

_REPLACEMENTS: dict[str, str] = {
    "pymegdec cross-validate": "neureptrace dataset run <dataset.yml> --analysis cross_validate",
    "pymegdec transfer": "neureptrace dataset run <dataset.yml> --analysis transfer_decode",
    "pymegdec stimulus decoding": "neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue",
    "pymegdec stimulus-decoding": "neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue",
    "pymegdec stimulus predictions": "neureptrace dataset run <dataset.yml> --analysis stimulus_predictions",
    "pymegdec stimulus-predictions": "neureptrace dataset run <dataset.yml> --analysis stimulus_predictions",
    "pymegdec stimulus temporal-generalization": (
        "neureptrace dataset run <dataset.yml> --analysis stimulus_temporal_generalization"
    ),
    "pymegdec stimulus-temporal-generalization": (
        "neureptrace dataset run <dataset.yml> --analysis stimulus_temporal_generalization"
    ),
    "pymegdec stimulus onset-scan": "neureptrace dataset run <dataset.yml> --analysis stimulus_onset_scan",
    "pymegdec stimulus-onset-scan": "neureptrace dataset run <dataset.yml> --analysis stimulus_onset_scan",
    "pymegdec stimulus robustness": "neureptrace dataset run <dataset.yml> --analysis stimulus_robustness",
    "pymegdec stimulus-robustness": "neureptrace dataset run <dataset.yml> --analysis stimulus_robustness",
    "pymegdec stimulus cross-subject-smoke": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_smoke"
    ),
    "pymegdec stimulus-cross-subject-smoke": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_smoke"
    ),
    "pymegdec stimulus cross-subject-cue-calibrated": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_cue_calibrated"
    ),
    "pymegdec stimulus-cross-subject-cue-calibrated": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_cue_calibrated"
    ),
    "pymegdec stimulus cross-subject-hyperalignment": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_hyperalignment"
    ),
    "pymegdec stimulus-cross-subject-hyperalignment": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_hyperalignment"
    ),
    "pymegdec stimulus cross-subject-mcca": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_mcca"
    ),
    "pymegdec stimulus-cross-subject-mcca": "neureptrace dataset run <dataset.yml> --analysis cross_subject_mcca",
    "pymegdec stimulus cross-subject-nested": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_nested"
    ),
    "pymegdec stimulus-cross-subject-nested": (
        "neureptrace dataset run <dataset.yml> --analysis cross_subject_nested"
    ),
    "pymegdec alpha metrics": "neureptrace meg alpha-metrics <dataset.yml>",
    "pymegdec alpha-metrics": "neureptrace meg alpha-metrics <dataset.yml>",
    "pymegdec alpha movement": "neureptrace meg alpha-movement <dataset.yml>",
    "pymegdec alpha-movement": "neureptrace meg alpha-movement <dataset.yml>",
    "pymegdec alpha movement-results": "neureptrace meg alpha-movement-results <dataset.yml>",
    "pymegdec alpha-movement-results": "neureptrace meg alpha-movement-results <dataset.yml>",
    "pymegdec alpha reaction-time": "examples/bush_meg/run_alpha_reaction_time.py <dataset.yml>",
    "pymegdec alpha-reaction-time": "examples/bush_meg/run_alpha_reaction_time.py <dataset.yml>",
    "pymegdec alpha rt": "examples/bush_meg/run_alpha_reaction_time.py <dataset.yml>",
    "pymegdec alpha-rt": "examples/bush_meg/run_alpha_reaction_time.py <dataset.yml>",
    "pymegdec data download": "dataset-specific download helper or documented external data source",
    "pymegdec download-meg-data": "dataset-specific download helper or documented external data source",
    "pymegdec make-synthetic-data": "neureptrace fieldtrip make-synthetic-data or tests/fixtures helper",
}


def _normalize_command(command: str | None) -> str | None:
    if command is None:
        return None
    return " ".join(command.strip().split())


def replacement_for(command: str | None) -> str | None:
    """Return the preferred replacement command for a PyMEGDec command."""

    normalized = _normalize_command(command)
    if normalized is None:
        return None
    return _REPLACEMENTS.get(normalized)


def deprecation_text(command: str | None = None, replacement: str | None = None) -> str:
    """Build a human-readable deprecation warning for CLI users."""

    normalized = _normalize_command(command)
    subject = f"`{normalized}`" if normalized else "This PyMEGDec command"
    resolved_replacement = replacement if replacement is not None else replacement_for(normalized)
    message = f"{subject} is deprecated. {PYMEGDEC_DEPRECATION_MESSAGE}"
    if resolved_replacement:
        message = f"{message} Suggested replacement: `{resolved_replacement}`."
    return message


def warn_pymegdec_deprecated(command: str | None = None, replacement: str | None = None) -> None:
    """Warn that a PyMEGDec CLI command is a temporary compatibility shim."""

    warnings.warn(
        deprecation_text(command, replacement),
        FutureWarning,
        stacklevel=2,
    )


def deprecated_handler(command: str, handler: CommandHandler, replacement: str | None = None) -> CommandHandler:
    """Wrap a grouped-CLI command handler with a visible deprecation warning."""

    @wraps(handler)
    def wrapped(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
        warn_pymegdec_deprecated(command, replacement)
        return handler(argv, prog)

    return wrapped


def deprecated_nullary_entrypoint(command: str, func: Callable[[], T], replacement: str | None = None) -> T:
    """Warn before invoking a legacy console-script function that takes no arguments."""

    warn_pymegdec_deprecated(command, replacement)
    return func()
