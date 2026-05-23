"""Compatibility wrapper for NeuRepTrace-owned PyMEGDec/BUSH-MEG specs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neureptrace.datasets.pymegdec import (
    DEFAULT_ENV_VAR,
    DEFAULT_PARTICIPANTS,
    build_pymegdec_bushmeg_dataset_spec_text,
    write_pymegdec_bushmeg_dataset_spec,
)


def build_neureptrace_dataset_spec_text(
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
) -> str:
    """Return the NeuRepTrace-owned YAML spec for PyMEGDec-style files."""

    return build_pymegdec_bushmeg_dataset_spec_text(
        participants=participants,
        env_var=env_var,
        data_dir=data_dir,
    )


def write_neureptrace_dataset_spec(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Write the NeuRepTrace-owned PyMEGDec/BUSH-MEG YAML dataset spec."""

    return write_pymegdec_bushmeg_dataset_spec(argv, prog)


if __name__ == "__main__":
    raise SystemExit(write_neureptrace_dataset_spec())


__all__ = [
    "DEFAULT_ENV_VAR",
    "DEFAULT_PARTICIPANTS",
    "build_neureptrace_dataset_spec_text",
    "write_neureptrace_dataset_spec",
]
