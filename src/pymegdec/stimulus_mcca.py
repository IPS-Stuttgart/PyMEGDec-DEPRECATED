"""Public facade for NeuRepTrace M-CCA stimulus decoding."""

from __future__ import annotations

import sys

from pymegdec import _stimulus_mcca_legacy as _impl
from pymegdec._reptrace_score_overrides import install_mcca

install_mcca(_impl)

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
