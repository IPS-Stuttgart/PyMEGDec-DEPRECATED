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

_COMMANDS: dict[str, tuple[str, str, str]] = {
    "validate-manifest": (
        "neureptrace.validate_manifest",
        "main",
        "Validate a NeuRepTrace benchmark manifest.",
    ),
    "mne-time-decode": (
        "neureptrace.mne_time_decode",
        "main",
        "Run NeuRepTrace MNE time-resolved decoding.",
    ),
    "plot-time-decode": (
        "neureptrace.plot_time_decode",
        "main",
        "Plot NeuRepTrace time-decoding outputs.",
    ),
}


@contextmanager
def _patched_argv(prog: str, argv: Sequence[str] | None) -> Iterator[None]:
    previous = sys.argv[:]
    sys.argv = [prog, *(list(argv) if argv is not None else [])]
    try:
        yield
    finally:
        sys.argv = previous


def _run_neureptrace(command: str, argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    module_name, function_name, _description = _COMMANDS[command]
    warnings.warn(
        (
            f"pymegdec {command} is a temporary compatibility alias. "
            f"Use the corresponding NeuRepTrace command/module instead: {module_name}.{function_name}."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module(module_name)
    entry_point = getattr(module, function_name)
    with _patched_argv(prog or f"pymegdec {command}", argv):
        result = entry_point()
    return int(result or 0)


def validate_manifest(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("validate-manifest", argv, prog)


def mne_time_decode(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("mne-time-decode", argv, prog)


def plot_time_decode(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return _run_neureptrace("plot-time-decode", argv, prog)


def handlers() -> dict[str, CommandHandler]:
    return {
        "validate-manifest": validate_manifest,
        "mne-time-decode": mne_time_decode,
        "plot-time-decode": plot_time_decode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility bridge to NeuRepTrace workflows.")
    parser.add_argument("command", nargs="?", choices=sorted(_COMMANDS), help="NeuRepTrace command to run.")
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.command is None:
        parser.print_help()
        return 0
    return handlers()[parsed.command](remaining, f"pymegdec-neureptrace {parsed.command}")
