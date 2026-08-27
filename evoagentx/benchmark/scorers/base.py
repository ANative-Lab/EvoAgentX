"""Backward-compatible re-export of the scoring protocol.

``Scorer`` / ``ScoreRequest`` / ``ScoreResult`` / ``as_scorer`` / ``CompositeScorer`` are
engine-facing protocols and now live in ``optimizers.core.scoring`` (they must be
importable from a package built out of ``optimizers/`` alone). This module keeps
``evoagentx.benchmark.scorers.base`` — and therefore ``evoagentx.benchmark.scorers`` —
resolving to the same objects so existing call sites (``from evoagentx.benchmark.scorers
import Scorer``) keep working. After the package split this file is the host-side
adapter point; it does not ship with ``optimizers/``.
"""

from ...optimizers.core.scoring import (
    CompositeScorer,
    Metrics,
    ScoreRequest,
    ScoreResult,
    Scorer,
    as_scorer,
)

__all__ = [
    "Scorer",
    "ScoreRequest",
    "ScoreResult",
    "CompositeScorer",
    "as_scorer",
    "Metrics",
]
