"""Consolidated command dispatcher for PyMEGDec workflows.

The command-line surface is centralized here while older script files remain
available as compatibility entry points.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from pymegdec import cli as legacy_cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_COMMANDS = {
    ("alpha-metrics",): ("alpha-metrics", "export_alpha_metrics.py"),
    ("alpha", "metrics"): ("alpha metrics", "export_alpha_metrics.py"),
    ("alpha-movement",): ("alpha-movement", "analyze_alpha_movement.py"),
    ("alpha", "movement"): ("alpha movement", "analyze_alpha_movement.py"),
    ("alpha-reaction-time",): ("alpha-reaction-time", "analyze_alpha_reaction_time.py"),
    ("alpha", "reaction-time"): ("alpha reaction-time", "analyze_alpha_reaction_time.py"),
    ("alpha", "rt"): ("alpha rt", "analyze_alpha_reaction_time.py"),
    ("stimulus-predictions",): ("stimulus-predictions", "scripts/export_stimulus_predictions.py"),
    ("stimulus", "predictions"): ("stimulus predictions", "scripts/export_stimulus_predictions.py"),
    ("stimulus-robustness",): ("stimulus-robustness", "scripts/export_stimulus_robustness.py"),
    ("stimulus", "robustness"): ("stimulus robustness", "scripts/export_stimulus_robustness.py"),
    ("stimulus-temporal-generalization",): (
        "stimulus-temporal-generalization",
        "scripts/export_stimulus_temporal_generalization.py",
    ),
    ("stimulus", "temporal-generalization"): (
        "stimulus temporal-generalization",
        "scripts/export_stimulus_temporal_generalization.py",
    ),
    ("stimulus-onset-scan",): ("stimulus-onset-scan", "scripts/export_stimulus_onset_scan.py"),
    ("stimulus", "onset-scan"): ("stimulus onset-scan", "scripts/export_stimulus_onset_scan.py"),
}
_LEGACY_COMMANDS = {
    ("cross-validate",): ("cross-validate", legacy_cli.cross_validate),
    ("transfer",): ("transfer", legacy_cli.transfer),
    ("stimulus-decoding",): ("stimulus-decoding", legacy_cli.stimulus_decoding),
    ("stimulus", "decoding"): ("stimulus decoding", legacy_cli.stimulus_decoding),
    ("alpha-movement-results",): ("alpha-movement-results", legacy_cli.alpha_movement_results),
    ("alpha", "movement-results"): ("alpha movement-results", legacy_cli.alpha_movement_results),
}


def _script_path(relative_path: str) -> Path:
    path = _REPO_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot locate {relative_path!r}. This command currently requires the "
            "source-tree compatibility script to be available."
        )
    return path


def _run_script(relative_path: str, argv: Sequence[str], prog: str) -> int:
    original_argv = sys.argv
    original_path = list(sys.path)
    sys.argv = [prog, *argv]
    for path in (_REPO_ROOT, _REPO_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    try:
        runpy.run_path(str(_script_path(relative_path)), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
    return 0


def _resolve_command(argv: Sequence[str]):
    two_token = tuple(argv[:2])
    if two_token in _LEGACY_COMMANDS:
        command_display, handler = _LEGACY_COMMANDS[two_token]
        return command_display, handler, list(argv[2:])
    if two_token in _SCRIPT_COMMANDS:
        command_display, script = _SCRIPT_COMMANDS[two_token]
        return command_display, script, list(argv[2:])

    one_token = (argv[0],)
    if one_token in _LEGACY_COMMANDS:
        command_display, handler = _LEGACY_COMMANDS[one_token]
        return command_display, handler, list(argv[1:])
    if one_token in _SCRIPT_COMMANDS:
        command_display, script = _SCRIPT_COMMANDS[one_token]
        return command_display, script, list(argv[1:])
    return None


def _print_group_help(group: str) -> None:
    if group == "alpha":
        print(
            "usage: pymegdec alpha <command> [options]\n\n"
            "Alpha workflows:\n"
            "  metrics           Export exploratory prestimulus alpha metrics.\n"
            "  movement          Export sensor-level alpha movement trajectories.\n"
            "  movement-results  Analyze exported alpha movement summaries.\n"
            "  reaction-time     Analyze alpha metrics against reaction time.\n"
        )
    elif group == "stimulus":
        print(
            "usage: pymegdec stimulus <command> [options]\n\n"
            "Stimulus workflows:\n"
            "  decoding                  Run time-resolved stimulus decoding.\n"
            "  predictions               Export trial-level stimulus predictions.\n"
            "  robustness                Export robustness-control predictions and summaries.\n"
            "  temporal-generalization   Export train-time/test-time temporal generalization.\n"
            "  onset-scan                Export onset-blind stimulus identity scans.\n"
        )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="PyMEGDec command-line interface.",
        epilog=(
            "Preferred grouped commands:\n"
            "  pymegdec alpha metrics ...\n"
            "  pymegdec alpha movement ...\n"
            "  pymegdec alpha movement-results ...\n"
            "  pymegdec alpha reaction-time ...\n"
            "  pymegdec stimulus decoding ...\n"
            "  pymegdec stimulus predictions ...\n"
            "  pymegdec stimulus robustness ...\n"
            "  pymegdec stimulus temporal-generalization ...\n"
            "  pymegdec stimulus onset-scan ...\n\n"
            "Compatibility aliases remain available, for example:\n"
            "  pymegdec cross-validate ...\n"
            "  pymegdec transfer ...\n"
            "  pymegdec stimulus-decoding ...\n"
            "  pymegdec alpha-movement-results ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="Command to run. Pass '<command> --help' for command-specific options.")

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if argv[0] in {"alpha", "stimulus"} and (len(argv) == 1 or argv[1] in {"-h", "--help"}):
        _print_group_help(argv[0])
        return 0

    resolved = _resolve_command(argv)
    if resolved is None:
        parser.error(f"Unsupported command: {' '.join(argv[:2]) if len(argv) > 1 else argv[0]}")
        return 2

    command_display, handler_or_script, remaining = resolved
    prog = f"pymegdec {command_display}"
    if isinstance(handler_or_script, str):
        return _run_script(handler_or_script, remaining, prog)
    return handler_or_script(remaining, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
