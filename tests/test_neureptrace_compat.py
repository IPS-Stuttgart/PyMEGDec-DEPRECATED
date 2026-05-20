from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pymegdec import neureptrace_compat


def test_validate_manifest_delegates_with_patched_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_main() -> None:
        calls.append(sys.argv[:])

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "neureptrace.validate_manifest"
        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(neureptrace_compat.importlib, "import_module", fake_import_module)

    with pytest.warns(DeprecationWarning, match="temporary compatibility alias"):
        status = neureptrace_compat.validate_manifest(
            ["manifest.csv", "--label-column", "condition"],
            "pymegdec config validate-manifest",
        )

    assert status == 0
    assert calls == [["pymegdec config validate-manifest", "manifest.csv", "--label-column", "condition"]]


def test_main_dispatches_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_validate(argv, prog):
        seen["argv"] = list(argv)
        seen["prog"] = prog
        return 7

    monkeypatch.setattr(neureptrace_compat, "validate_manifest", fake_validate)

    assert neureptrace_compat.main(["validate-manifest", "manifest.csv"]) == 7
    assert seen == {"argv": ["manifest.csv"], "prog": "pymegdec-neureptrace validate-manifest"}
