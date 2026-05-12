"""Backward-compatible wrapper for the cross-subject stimulus smoke command."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.stimulus_cli import stimulus_cross_subject_smoke  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(stimulus_cross_subject_smoke())
