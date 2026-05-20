from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace
from unittest.mock import patch

from pymegdec import neureptrace_compat


def test_validate_manifest_delegates_with_patched_argv() -> None:
    calls: list[list[str]] = []

    def fake_main() -> None:
        calls.append(sys.argv[:])

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "neureptrace.validate_manifest"
        return SimpleNamespace(main=fake_main)

    with (
        patch.object(neureptrace_compat.importlib, "import_module", fake_import_module),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        status = neureptrace_compat.validate_manifest(["manifest.csv", "--label-column", "condition"], "pymegdec config validate-manifest")

    assert status == 0
    assert calls == [["pymegdec config validate-manifest", "manifest.csv", "--label-column", "condition"]]
    assert len(record) == 1
    assert issubclass(record[0].category, DeprecationWarning)
    assert "temporary compatibility alias" in str(record[0].message)


def test_main_dispatches_selected_command() -> None:
    seen: dict[str, object] = {}

    def fake_validate(argv, prog):
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 7

    with patch.object(neureptrace_compat, "validate_manifest", fake_validate):
        assert neureptrace_compat.main(["validate-manifest", "manifest.csv"]) == 7
    assert seen == {"argv": ["manifest.csv"], "prog": "pymegdec-neureptrace validate-manifest"}
