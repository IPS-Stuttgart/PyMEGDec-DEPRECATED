"""Grouped command-line dispatcher for PyMEGDec workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from pymegdec import alpha_cli, neureptrace_compat
from pymegdec import cli as legacy_cli
from pymegdec import stimulus_cli, stimulus_hyperalignment, stimulus_mcca
from pymegdec.neureptrace_dataset_spec import write_neureptrace_dataset_spec
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
        "cross-subject-cue-calibrated": stimulus_cli.stimulus_cross_subject_cue_calibrated,
        "cross-subject-hyperalignment": stimulus_hyperalignment.stimulus_cross_subject_hyperalignment,
        "cross-subject-mcca": stimulus_mcca.stimulus_cross_subject_mcca,
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
        "movement-results": alpha_cli.alpha_movement_results,
        "reaction-time": alpha_cli.alpha_reaction_time,
        "rt": alpha_cli.alpha_reaction_time,
    }


def _config_handlers() -> dict[str, CommandHandler]:
    return neureptrace_compat.handlers()


def _data_handlers() -> dict[str, CommandHandler]:
    return {"write-neureptrace-spec": write_neureptrace_dataset_spec}


def _top_level_handlers() -> dict[str, CommandHandler]:
    return {
        "cross-validate": legacy_cli.cross_validate,
        "transfer": legacy_cli.transfer,
        "make-synthetic-data": make_synthetic_data,
        "write-neureptrace-spec": write_neureptrace_dataset_spec,
        # Backward-compatible top-level aliases. Prefer grouped forms in new docs.
        "stimulus-decoding": legacy_cli.stimulus_decoding,
        "stimulus-cross-subject-cue-calibrated": stimulus_cli.stimulus_cross_subject_cue_calibrated,
        "stimulus-cross-subject-hyperalignment": stimulus_hyperalignment.stimulus_cross_subject_hyperalignment,
        "stimulus-cross-subject-mcca": stimulus_mcca.stimulus_cross_subject_mcca,
        "stimulus-cross-subject-nested": stimulus_cli.stimulus_cross_subject_nested,
        "stimulus-cross-subject-smoke": stimulus_cli.stimulus_cross_subject_smoke,
        "stimulus-predictions": stimulus_cli.stimulus_predictions,
        "stimulus-robustness": stimulus_cli.stimulus_robustness,
        "stimulus-temporal-generalization": stimulus_cli.stimulus_temporal_generalization,
        "stimulus-onset-scan": stimulus_cli.stimulus_onset_scan,
        "alpha-metrics": alpha_cli.alpha_metrics,
        "alpha-movement": alpha_cli.alpha_movement,
        "alpha-movement-results": alpha_cli.alpha_movement_results,
        "alpha-reaction-time": alpha_cli.alpha_reaction_time,
        "alpha-rt": alpha_cli.alpha_reaction_time,
        # Transitional aliases for NeuRepTrace-owned, config-oriented workflows.
        "validate-manifest": neureptrace_compat.validate_manifest,
        "mne-time-decode": neureptrace_compat.mne_time_decode,
        "plot-time-decode": neureptrace_compat.plot_time_decode,
    }


def _print_main_help() -> None:
    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.add_argument("command", nargs="?", help="Command or command group to run.")
    parser.print_help()
    print(
        "\nCommand groups:\n"
        "  pymegdec stimulus <cross-subject-cue-calibrated|cross-subject-hyperalignment|cross-subject-mcca|cross-subject-nested|cross-subject-smoke|"
        "decoding|predictions|robustness|temporal-generalization|onset-scan>\n"
        "  pymegdec alpha <metrics|movement|movement-results|reaction-time|rt>  # legacy paper-specific analyses\n"
        "  pymegdec config <validate-manifest|mne-time-decode|plot-time-decode>\n"
        "  pymegdec data <write-neureptrace-spec>\n"
        "\nCore commands:\n"
        "  pymegdec cross-validate ...\n"
        "  pymegdec transfer ...\n"
        "  pymegdec make-synthetic-data ...\n"
        "\nCompatibility aliases such as pymegdec stimulus-decoding, pymegdec alpha-metrics, and pymegdec validate-manifest remain available.\n"
        "Alpha commands are retained as legacy Bush-MEG analysis scripts and are not NeuRepTrace migration targets."
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
    if command == "config":
        return _dispatch_group("config", "NeuRepTrace-owned configuration workflows.", _config_handlers(), remaining)
    if command == "data":
        return _dispatch_group("data", "Data configuration and migration helpers.", _data_handlers(), remaining)
    handlers = _top_level_handlers()
    if command in handlers:
        return handlers[command](remaining, f"pymegdec {command}")

    parser = argparse.ArgumentParser(description="PyMEGDec command-line interface.")
    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
