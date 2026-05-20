#!/usr/bin/env python3
"""Repository-local wrapper for private MEG data downloads."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_IMPL = _ROOT / "scripts" / "download_private_meg_data.py"

_SPEC = spec_from_file_location("_pymegdec_private_data_download", _IMPL)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load {_IMPL}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
download_meg_data_files = _MODULE.download_meg_data_files


def main() -> int:
    return download_meg_data_files()


if __name__ == "__main__":
    raise SystemExit(main())
