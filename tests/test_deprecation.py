from __future__ import annotations

import warnings

from pymegdec.deprecation import deprecated_handler, deprecation_text, replacement_for


def test_replacement_for_known_pymegdec_command() -> None:
    assert replacement_for("pymegdec stimulus decoding") == (
        "neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue"
    )


def test_deprecation_text_includes_replacement() -> None:
    message = deprecation_text("pymegdec cross-validate")

    assert "PyMEGDec is now a compatibility layer" in message
    assert "neureptrace dataset run <dataset.yml> --analysis cross_validate" in message


def test_deprecated_handler_warns_and_forwards_args() -> None:
    calls = []

    def handler(argv, prog):
        calls.append((argv, prog))
        return 0

    wrapped = deprecated_handler("pymegdec transfer", handler)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = wrapped(["--participant", "10"], "pymegdec transfer")

    assert result == 0
    assert calls == [(["--participant", "10"], "pymegdec transfer")]
    assert len(records) == 1
    assert issubclass(records[0].category, FutureWarning)
    assert "pymegdec transfer" in str(records[0].message)
    assert "transfer_decode" in str(records[0].message)
