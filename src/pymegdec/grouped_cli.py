"""Grouped command-line dispatcher for PyMEGDec workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from pymegdec import alpha_cli
from pymegdec import cli as legacy_cli
from pymegdec import stimulus_cli
from pymegdec.data_download import download_meg_data_files
from pymegdec.synthetic_data_cli import make_synthetic_data

CommandHandler = Callable[[Sequence[str] | None, str | None], int]


def _dispatch_group(group: str, description: str, handlers: dict[str, CommandHandler], argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog=f"pymegdec {group}", description=description)
        parser.add_argument("subcommand", nargs="?", choices=sorted(handlers), help="Subcommand to run.")
        parser.print_help()
        return 0

    subcommand, *remaining = argv
    if subcommand not in handlers:
        parser = argparse.ArgumentParser(prog=f"pymegdec {group}", description=description)
        parser.error(f"Unsupported {group} subcommand: {subcommand}")
    return handlers[subcommand](remaining, f"pymegdec {group} {subcommand}")


def _stimulus_handlers() -> dict[str, CommandHandler]:
    return {
        "cross-subject-nested": stimulus_cli.stimulus_cross_subject_nested,
        "cross-subject-smoke": stimulus_cli.stimulus_cross_subject_smoke,
        "decoding": legacy_cli.stimulus_decoding,
        "predictions": stimulus_cli.stimulus_predictions,
        "robustness": stimulus_cli.stimulus_robustness,
        "temporal-generalization": stimulus_cli.stimulus_temporal_generalization,
        "onset-scan": stimulus_cli.stimulus_onset_scan,
    }


def _alpha_handlers() -> dict[str, CommandHandler]:
    return {
        "metrics": alpha_cli.alpha_metrics,
        "movement": alpha_cli.alpha_movement,
        "movement-results": legacy_cli.alpha_movement_results,
        "reaction-time": alpha_cli.alpha_reaction_time,
        "rt": alpha_cli.alpha_reaction_time,
    }


def _data_handlers() -> dict[str, CommandHandler]:
    return {"download": download_meg_data_files}


def _top_level_handlers() -> dict[str, CommandHandler]:
    return {
        "cross-validate": legacy_cli.cross_validate,
        "transfer": legacy_cli.transfer,
        "make-synthetic-data": make_synthetic_data,
        # Backward-compatible top-level aliases. Prefer grouped forms in new docs.
        "stimulus-decoding": legacy_cli.stimulus_decoding,
        "stimulus-cross-subject-nested": stimulus_cli.stimulus_cross_subject_nested,
        "stimulus-cross-subject-smoke": stimulus_cli.stimulus_cross_subject_smoke,
        "stimulus-predictions": stimulus_cli.stimulus_predictions,
        "stimulus-robustness": stimulus_cli.stimulus_robustness,
        "stimulus-temporal-generalization": stimulus_cli.stimulus_temporal_generalization,
        "stimulus-onset-scan": stimulus_cli.stimulus_onset_scan,
        "alpha-metrics": alpha_cli.alpha_metrics,
        "alpha-movement": alpha_cli.alpha_movement,
        "alpha-movement-results": legacy_cli.alpha_movement_results,
        "alpha-reaction-time": alpha_cli.alpha_reaction_time,
        "alpha-rt": alpha_cli.alpha_reaction_time,
        "download-meg-data": download_meg_data_files,
    }


def _print_main_help() -> None:
    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument("command", nargs="?", help="Command or command group to run.")
    parser.print_help()
    print(
        "\nCommand groups:\n"
        "  pymegdec stimulus <cross-subject-nested|cross-subject-smoke|decoding|predictions|robustness|temporal-generalization|onset-scan>\n"
        "  pymegdec alpha <metrics|movement|movement-results|reaction-time|rt>\n"
        "  pymegdec data <download>\n"
        "\nCore commands:\n"
        "  pymegdec cross-validate ...\n"
        "  pymegdec transfer ...\n"
        "  pymegdec make-synthetic-data ...\n"
        "\nCompatibility aliases such as pymegdec stimulus-decoding and pymegdec alpha-metrics remain available."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_main_help()
        return 0

    command, *remaining = argv
    if command == "stimulus":
        return _dispatch_group("stimulus", "Stimulus decoding and diagnostics.", _stimulus_handlers(), remaining)
    if command == "alpha":
        return _dispatch_group("alpha", "Alpha metric, movement, and reaction-time analyses.", _alpha_handlers(), remaining)
    if command == "data":
        return _dispatch_group("data", "Data download and data-management helpers.", _data_handlers(), remaining)

    handlers = _top_level_handlers()
    if command in handlers:
        return handlers[command](remaining, f"pymegdec {command}")

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
