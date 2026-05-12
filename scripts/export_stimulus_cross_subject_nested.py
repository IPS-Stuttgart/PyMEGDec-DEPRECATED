"""Backward-compatible wrapper for the nested cross-subject stimulus command."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.stimulus_cli import stimulus_cross_subject_nested  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(stimulus_cross_subject_nested())
