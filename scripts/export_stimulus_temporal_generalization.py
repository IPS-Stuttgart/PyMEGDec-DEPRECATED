"""Backward-compatible wrapper for the NeuRepTrace-backed temporal-generalization command."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.stimulus_temporal_generalization_neureptrace import stimulus_temporal_generalization  # noqa: E402


def main() -> int:
    return stimulus_temporal_generalization()


if __name__ == "__main__":
    raise SystemExit(main())
