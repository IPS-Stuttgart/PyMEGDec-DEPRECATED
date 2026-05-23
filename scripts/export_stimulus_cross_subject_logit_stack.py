"""Backward-compatible wrapper for source-OOF logit-stack cross-subject stimulus decoding."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec.stimulus_logit_stacking import stimulus_cross_subject_logit_stack  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(stimulus_cross_subject_logit_stack())
