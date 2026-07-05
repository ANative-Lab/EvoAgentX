"""Scorer base class and helpers.

A ``Scorer`` answers a single question: given a ``prediction`` and a ``label``, what are
the evaluation metrics? It is decoupled from any specific ``Benchmark`` so that the same
scorer (e.g. ``QAScorer``) can be reused across datasets.

Design:
    - Sync core: subclasses implement ``score(prediction, label) -> dict[str, float]``.
    - Async access: ``ascore`` defaults to running ``score`` in a worker thread; scorers
      that are natively async (e.g. an LLM-judge) may override it.
    - The pipeline accepts a plain ``Callable`` too; use ``as_scorer`` to normalize it.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union


Metrics = Dict[str, float]


class Scorer(ABC):
    """Base class for all scorers."""

    @abstractmethod
    def score(self, prediction: Any, label: Any) -> Metrics:
        """Compute metrics for a single (prediction, label) pair."""
        raise NotImplementedError

    async def ascore(self, prediction: Any, label: Any) -> Metrics:
        """Async variant. Defaults to offloading the sync ``score`` to a thread."""
        return await asyncio.to_thread(self.score, prediction, label)


class _CallableScorer(Scorer):
    """Adapter that wraps a plain callable into a ``Scorer``.

    Supports both sync ``fn(prediction, label) -> dict`` and async coroutine functions.
    """

    def __init__(self, fn: Callable[[Any, Any], Any]):
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)

    def score(self, prediction: Any, label: Any) -> Metrics:
        if self._is_async:
            # Best-effort sync bridge for an async callable used on the sync path.
            return asyncio.run(self._fn(prediction, label))
        return self._fn(prediction, label)

    async def ascore(self, prediction: Any, label: Any) -> Metrics:
        if self._is_async:
            return await self._fn(prediction, label)
        return await asyncio.to_thread(self._fn, prediction, label)


class CompositeScorer(Scorer):
    """Run multiple scorers and merge their metric dicts.

    Later scorers win on key collisions. Optionally prefix each scorer's keys to avoid
    collisions (e.g. ``{"qa/f1": ..., "code/pass@1": ...}``).
    """

    def __init__(self, scorers: List[Union[Scorer, Callable]], prefixes: List[str] = None):
        self.scorers = [as_scorer(s) for s in scorers]
        self.prefixes = prefixes

    def _merge(self, results: List[Metrics]) -> Metrics:
        merged: Metrics = {}
        for i, metrics in enumerate(results):
            prefix = self.prefixes[i] + "/" if self.prefixes else ""
            for k, v in metrics.items():
                merged[f"{prefix}{k}"] = v
        return merged

    def score(self, prediction: Any, label: Any) -> Metrics:
        return self._merge([s.score(prediction, label) for s in self.scorers])

    async def ascore(self, prediction: Any, label: Any) -> Metrics:
        results = await asyncio.gather(*(s.ascore(prediction, label) for s in self.scorers))
        return self._merge(list(results))


def as_scorer(obj: Union[Scorer, Callable]) -> Scorer:
    """Normalize a ``Scorer`` or a plain callable into a ``Scorer`` instance."""
    if isinstance(obj, Scorer):
        return obj
    if callable(obj):
        return _CallableScorer(obj)
    raise TypeError(f"Expected a Scorer or a callable, got {type(obj)!r}.")
