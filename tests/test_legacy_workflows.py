from __future__ import annotations

import warnings

from pymegdec.legacy import PyMEGDecLegacyWorkflowWarning, warn_legacy_alpha_workflow


def test_warn_legacy_alpha_workflow_is_visible_to_cli_users() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warn_legacy_alpha_workflow("pymegdec alpha metrics")

    assert len(record) == 1
    message = str(record[0].message)
    assert "pymegdec alpha metrics" in message
    assert "legacy PyMEGDec alpha-analysis workflow" in message
    assert "not planned as a NeuRepTrace migration target" in message
    assert issubclass(PyMEGDecLegacyWorkflowWarning, FutureWarning)
