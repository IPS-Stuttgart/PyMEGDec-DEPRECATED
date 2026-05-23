"""Compatibility bridge to NeuRepTrace-owned configuration workflows.

PyMEGDec is being reduced to dataset recipes and temporary aliases.  New
configuration-driven workflows should live in NeuRepTrace; this module keeps the
old PyMEGDec command surface usable while making the ownership boundary explicit.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

CommandHandler = Callable[[Sequence[str] | None, str | None], int]

_COMMANDS: dict[str, str] = {
    "dataset": "Run NeuRepTrace dataset-spec validation and manifest helpers.",
    "dataset-manifest": "Run NeuRepTrace dataset-manifest helpers.",
    "dataset-spec": "Run NeuRepTrace dataset-spec helpers.",
    "decode-from-config": "Run NeuRepTrace config-driven decoding.",
    "epoch-transfer-decode": "Run NeuRepTrace epoch-transfer decoding.",
    "event-detect": "Run NeuRepTrace event detection.",
    "fieldtrip-to-mne": "Convert FieldTrip MAT files through NeuRepTrace.",
    "mne-time-decode": "Run NeuRepTrace MNE time-resolved decoding.",
    "mne-time-decode-base": "Run the base NeuRepTrace MNE time decoder.",
    "mne-time-decode-ensemble": "Run NeuRepTrace MNE time-decoding ensembles.",
    "onset-detect": "Run NeuRepTrace onset detection.",
    "plot-time-decode": "Plot NeuRepTrace time-decoding outputs.",
    "probability-stacking": "Run NeuRepTrace source-OOF probability stacking.",
    "pymegdec-bushmeg-spec": "Write the canonical NeuRepTrace PyMEGDec/BUSH-MEG dataset spec.",
    "results": "Run NeuRepTrace result table helpers.",
    "source-oof-stacking": "Run NeuRepTrace source-OOF probability stacking.",
    "synthetic-fieldtrip": "Write private-data-free synthetic FieldTrip fixtures.",
    "time-transfer-decode": "Run NeuRepTrace time-transfer decoding.",
    "transfer-from-config": "Run NeuRepTrace config-driven transfer decoding.",
    "validate-manifest": "Validate a NeuRepTrace benchmark manifest.",
    "validate-observations": "Validate NeuRepTrace observation tables.",
}


@contextmanager
def _patched_argv(prog: str, argv: Sequence[str] | None) -> Iterator[None]:
    previous = sys.argv[:]
    sys.argv = [prog, *(list(argv) if argv is not None else [])]
    try:
        yield
    finally:
        sys.argv = previous


def _module_name_for_command(command: str) -> str:
    """Return the installed NeuRepTrace module for a grouped command."""

    cli = importlib.import_module("neureptrace.cli")
    command_modules = getattr(cli, "COMMAND_MODULES")
    try:
        return command_modules[command]
    except KeyError as exc:
        raise RuntimeError(
            f"PyMEGDec exposes NeuRepTrace command {command!r}, but the installed NeuRepTrace package does not. "
            "Upgrade NeuRepTrace or remove the stale PyMEGDec compatibility alias."
        ) from exc


def _run_neureptrace(command: str, argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    module_name = _module_name_for_command(command)
    warnings.warn(
        (
            f"pymegdec {command} is a temporary compatibility alias. "
            f"Use the corresponding NeuRepTrace command instead: neureptrace {command}."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module(module_name)
    entry_point = getattr(module, "main")
    with _patched_argv(prog or f"pymegdec {command}", argv):
        result = entry_point()
    return int(result or 0)


def validate_manifest(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("validate-manifest", argv, prog)


def mne_time_decode(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("mne-time-decode", argv, prog)


def plot_time_decode(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("plot-time-decode", argv, prog)


def _generic_handler(command: str) -> CommandHandler:
    def handler(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
        return _run_neureptrace(command, argv, prog)

    handler.__name__ = command.replace("-", "_")
    return handler


def handlers() -> dict[str, CommandHandler]:
    explicit_handlers: dict[str, CommandHandler] = {
        "validate-manifest": validate_manifest,
        "mne-time-decode": mne_time_decode,
        "plot-time-decode": plot_time_decode,
    }
    return {command: explicit_handlers.get(command, _generic_handler(command)) for command in _COMMANDS}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility bridge to NeuRepTrace workflows.")
    parser.add_argument("command", nargs="?", choices=sorted(_COMMANDS), help="NeuRepTrace command to run.")
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.command is None:
        parser.print_help()
        return 0
    return handlers()[parsed.command](remaining, f"pymegdec-neureptrace {parsed.command}")
