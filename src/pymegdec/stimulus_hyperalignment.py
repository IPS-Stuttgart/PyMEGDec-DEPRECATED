"""Public facade for Procrustes hyperalignment stimulus decoding."""

from __future__ import annotations

import sys

from pymegdec import _stimulus_hyperalignment_legacy as _impl
from pymegdec._reptrace_score_overrides import install_hyperalignment

install_hyperalignment(_impl)

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
