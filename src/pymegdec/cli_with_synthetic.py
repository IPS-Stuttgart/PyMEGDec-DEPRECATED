"""Grouped PyMEGDec CLI wrapper with synthetic demo-data support."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import cli as legacy_cli
from .synthetic_data_cli import make_synthetic_data

COMMANDS = (
    "cross-validate",
    "transfer",
    "stimulus-decoding",
    "make-synthetic-data",
    "alpha-movement-results",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch grouped PyMEGDec commands.

    Existing workflow implementations remain in :mod:`pymegdec.cli`; this
    wrapper adds the synthetic data generator while preserving the legacy
    command behavior and compatibility entry points.
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        help="Workflow to run.",
    )
    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "make-synthetic-data":
        return make_synthetic_data(remaining, prog="pymegdec make-synthetic-data")

    return legacy_cli.main([args.command, *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
