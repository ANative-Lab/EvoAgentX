"""Scoring-side contracts for the optimization engine.

A ``Scorer`` answers a single question: given everything about one observation, what are
the evaluation metrics? ``ScoreRequest`` / ``ScoreResult`` are the sole boundary object on
each side, so a scorer never has to guess what a caller happened to pass in and a caller
never has to sniff what a scorer happens to return.

These are engine-facing protocols and generic composition helpers — zero business logic,
zero dataset knowledge. They live in ``optimizers/core`` (not ``evoagentx.benchmark``)
because they appear in engine-facing signatures (``DataEvaluator.__init__`` /
``Evaluator.metric_keys``); a package built from this one must stay importable without the
host application. Concrete scorers (QA / math / code) live in ``evoagentx.benchmark.scorers``
and import these protocols, never the other way around.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "ObservationKey",
    "Metrics",
    "ScoreRequest",
    "ScoreResult",
    "Scorer",
    "CompositeScorer",
    "as_scorer",
]


# (example_id, repeat) — the primary key of one observation, shared by ItemContext /
# ItemResult / ScoreRequest. Canonical home is ``core/results.py`` (frozen alongside
# ``ItemResult``); defined here first because ``ScoreRequest`` needs it and this module
# has no dependency on ``results.py``.
ObservationKey = Tuple[str, int]

Metrics = Dict[str, float]


@dataclass
class ScoreRequest:
    """Everything a scorer needs to score one observation — one object, zero sniffing."""

    key: ObservationKey                  # (example_id, repeat) — judge knows which retry this is
    inputs: Any                          # raw input to the program — judge / rubric need it
    prediction: Any                      # the (post output_fn) prediction to score
    reference: Any                       # generalized label: answer | rubric list | checklist
    trace: Any = None                    # process trace, for judges that look beyond the final answer


@dataclass
class ScoreResult:
    """A scorer's output. ``metrics`` is comparable; ``feedback`` / ``details`` are not."""

    metrics: Metrics                     # -> comparable signal (Objective / best pointer / state.json)
    feedback: Optional[str] = None       # -> interpretable signal, report.jsonl only, never compared
    details: Dict[str, Any] = field(default_factory=dict)  # -> traceable signal (rubric line items, etc.)


class Scorer(ABC):
    """Base class for all scorers. One async entry point, no sync/async twin."""

    @abstractmethod
    async def score(self, req: ScoreRequest) -> ScoreResult:
        """Score a single observation."""
        raise NotImplementedError

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        """The metric keys this scorer produces, independent of any data.

        Declaring them makes the metric set a *contract of the scorer* rather than
        something inferred from successful items: an all-crashed trial still scores
        ``{key: 0.0}`` for each declared key instead of collapsing to an empty ``{}``,
        which would otherwise leave the objective undefined.

        Return ``()`` (the default) if the keys are not knowable ahead of time; callers
        fall back to inferring them from successful items.
        """
        return ()


class _CallableScorer(Scorer):
    """Adapter that wraps a plain ``(prediction, reference) -> dict`` callable into a ``Scorer``.

    Supports sync and async callables alike; a sync callable is offloaded to a thread so it
    never blocks the event loop that concurrent evaluation runs on.
    """

    def __init__(self, fn: Callable[[Any, Any], Any], metric_keys: Tuple[str, ...] = ()):
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self._metric_keys = tuple(metric_keys)

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        # A plain callable can't be introspected for its keys; empty unless declared
        # explicitly via ``as_scorer(fn, metric_keys=...)``.
        return self._metric_keys

    async def score(self, req: ScoreRequest) -> ScoreResult:
        if self._is_async:
            metrics = await self._fn(req.prediction, req.reference)
        else:
            metrics = await asyncio.to_thread(self._fn, req.prediction, req.reference)
        return ScoreResult(metrics=metrics)


class CompositeScorer(Scorer):
    """Run multiple scorers against the same request and merge their outputs.

    Later scorers win on metric-key collisions. Optionally prefix each scorer's metric
    keys to avoid collisions (e.g. ``{"qa/f1": ..., "code/pass@1": ...}``). Feedback is
    concatenated (child scorers with no feedback contribute nothing); details are kept
    side by side, one entry per (prefixed) scorer.
    """

    def __init__(self, scorers: List[Union[Scorer, Callable]], prefixes: Optional[List[str]] = None):
        self.scorers = [as_scorer(s) for s in scorers]
        self.prefixes = prefixes

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        # Merge child keys with the same prefixing scheme as ``_merge``. If any child
        # declares no keys, the whole set is unknown (``()``) — we can't partially
        # promise a contract we can't fully back.
        keys: List[str] = []
        for i, scorer in enumerate(self.scorers):
            child = scorer.metric_keys
            if not child:
                return ()
            prefix = self.prefixes[i] + "/" if self.prefixes else ""
            keys.extend(f"{prefix}{k}" for k in child)
        return tuple(keys)

    def _prefix(self, i: int) -> str:
        return self.prefixes[i] + "/" if self.prefixes else ""

    async def score(self, req: ScoreRequest) -> ScoreResult:
        results = await asyncio.gather(*(s.score(req) for s in self.scorers))
        metrics: Metrics = {}
        feedback_parts: List[str] = []
        details: Dict[str, Any] = {}
        for i, result in enumerate(results):
            prefix = self._prefix(i)
            for k, v in result.metrics.items():
                metrics[f"{prefix}{k}"] = v
            if result.feedback:
                feedback_parts.append(f"[{prefix.rstrip('/')}] {result.feedback}" if prefix else result.feedback)
            if result.details:
                details[prefix.rstrip("/") or str(i)] = result.details
        return ScoreResult(
            metrics=metrics,
            feedback="\n".join(feedback_parts) if feedback_parts else None,
            details=details,
        )


def as_scorer(obj: Union[Scorer, Callable], metric_keys: Tuple[str, ...] = ()) -> Scorer:
    """Normalize a ``Scorer`` or a plain ``(prediction, reference) -> dict`` callable into a ``Scorer``.

    ``metric_keys`` only applies when wrapping a plain callable, letting the caller
    declare the callable's metric keys (see ``Scorer.metric_keys``); it is ignored when
    ``obj`` is already a ``Scorer`` (that scorer owns its own contract).
    """
    if isinstance(obj, Scorer):
        return obj
    if callable(obj):
        return _CallableScorer(obj, metric_keys=metric_keys)
    raise TypeError(f"Expected a Scorer or a callable, got {type(obj)!r}.")
