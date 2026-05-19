"""Console-script wrappers for deprecated PyMEGDec entry points."""

from __future__ import annotations

from collections.abc import Sequence

from pymegdec.deprecation import warn_pymegdec_deprecated


def cross_validate(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec cross-validate")
    from pymegdec.cli import cross_validate as _cross_validate

    return _cross_validate(argv, prog)


def transfer(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec transfer")
    from pymegdec.cli import transfer as _transfer

    return _transfer(argv, prog)


def stimulus_decoding(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec stimulus-decoding")
    from pymegdec.cli import stimulus_decoding as _stimulus_decoding

    return _stimulus_decoding(argv, prog)


def stimulus_cross_subject_cue_calibrated(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec stimulus-cross-subject-cue-calibrated")
    from pymegdec.stimulus_cli import stimulus_cross_subject_cue_calibrated as _cue_calibrated

    return _cue_calibrated(argv, prog)


def stimulus_cross_subject_hyperalignment(argv: Sequence[str] | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec stimulus-cross-subject-hyperalignment")
    from pymegdec.stimulus_hyperalignment import main as _main

    return _main(argv)


def stimulus_cross_subject_mcca(argv: Sequence[str] | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec stimulus-cross-subject-mcca")
    from pymegdec.stimulus_mcca import main as _main

    return _main(argv)


def alpha_movement_results(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec alpha-movement-results")
    from pymegdec.cli import alpha_movement_results as _alpha_movement_results

    return _alpha_movement_results(argv, prog)


def make_synthetic_data(argv: Sequence[str] | None = None) -> int:
    warn_pymegdec_deprecated("pymegdec make-synthetic-data")
    from pymegdec.synthetic_data_cli import main as _main

    return _main(argv)
