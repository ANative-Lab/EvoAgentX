"""Evaluation result types — two altitudes, one bridge.

This module is the single home for the two evaluation-result shapes the framework uses,
so the ``optimizers`` engine and the ``evaluators`` pipeline can be chained without a
naming clash or a dependency cycle (``optimizers -> evaluators``, never the reverse).

    - ``ItemResult``        : everything that happened for **one** example.
    - ``EvaluationReport``  : the **per-item** run log produced by ``EvaluationPipeline``
                              (a list of ``ItemResult``). This is what a plain-evaluation
                              user inspects.
    - ``EvaluationResult``  : the **trial-level**, objective-facing summary the Optimizer
                              consumes (aggregated ``metrics`` plus opaque ``traces`` /
                              ``artifacts`` / ``metadata``). It is recorded into a
                              ``TrialRecord`` and therefore must round-trip as pure JSON.

The bridge is ``EvaluationReport.to_result(...)``: explicit, policy-driven, with a full
``builder`` escape hatch so callers decide exactly what to keep (the framework cannot know
in advance which traces/artifacts/metadata matter for a given program).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import Field

from ..core.logging import logger
from ..core.module import BaseModule


# (field_name, value_type_name) pairs already warned about, so a large run does not
# emit one warning per item when e.g. every prediction is a non-serializable object.
_JSON_DEGRADE_WARNED = set()


def _json_safe(value: Any, field_name: str) -> Any:
    """Return ``value`` if it is JSON-serializable, else degrade to ``repr(value)``."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        key = (field_name, type(value).__name__)
        if key not in _JSON_DEGRADE_WARNED:
            _JSON_DEGRADE_WARNED.add(key)
            logger.warning(
                f"ItemResult.{field_name} of type {type(value).__name__} is not JSON-serializable; "
                f"storing repr() in the persisted report instead."
            )
        return repr(value)


# --------------------------------------------------------------------------- #
# Trial-level, objective-facing result (consumed by the Optimizer engine)
# --------------------------------------------------------------------------- #
class EvaluationResult(BaseModule):
    """Structured result returned by evaluation functions.

    ``metrics`` are objective-facing and aggregated. ``traces`` / ``artifacts`` /
    ``metadata`` are opaque payloads recorded into a ``TrialRecord``; because the run
    state is checkpointed, every value placed here must be a pure JSON data tree (no
    live objects, no ``class_name`` dicts). Serialize heavy trajectories to files and
    keep only their paths here.
    """
    metrics: Dict[str, Any] = Field(description="Objective-facing metrics for this trial.")
    traces: Optional[List[Any]] = Field(default_factory=list, description="Optional execution traces or trajectory data collected during evaluation.")
    artifacts: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional produced artifacts such as summaries, retrieved memories, file paths, generated skills, or debug payloads.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional evaluation metadata such as split name, cost, latency, seeds, or evaluator configuration.")


# --------------------------------------------------------------------------- #
# Per-item result
# --------------------------------------------------------------------------- #
@dataclass
class ItemResult:
    """Outcome of evaluating one example.

    A successful item has ``error is None`` and a populated ``metrics``. A failed item
    carries an ``error`` message and (depending on which stage failed) may still have a
    ``prediction``. ``trace`` / ``artifacts`` are optional, opt-in payloads (see the
    pipeline's ``trace_fn``).
    """
    id: Any
    prediction: Any = None
    metrics: Optional[Dict[str, float]] = None
    trace: Any = None
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    elapsed_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_json_dict(self) -> Dict[str, Any]:
        """One JSONL record. Non-serializable payloads degrade to ``repr`` (with a warning)."""
        return {
            "id": _json_safe(self.id, "id"),
            "prediction": _json_safe(self.prediction, "prediction"),
            "metrics": _json_safe(self.metrics, "metrics"),
            "trace": _json_safe(self.trace, "trace"),
            "artifacts": _json_safe(self.artifacts, "artifacts"),
            "error": self.error,
            "traceback": self.traceback,
            "elapsed_time": self.elapsed_time,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "ItemResult":
        return cls(
            id=data.get("id"),
            prediction=data.get("prediction"),
            metrics=data.get("metrics"),
            trace=data.get("trace"),
            artifacts=data.get("artifacts"),
            error=data.get("error"),
            traceback=data.get("traceback"),
            elapsed_time=data.get("elapsed_time", 0.0),
        )


# --------------------------------------------------------------------------- #
# Per-item run report (produced by EvaluationPipeline)
# --------------------------------------------------------------------------- #
ReduceFn = Union[str, Callable[[List[float]], float]]
OnError = str  # "zero" | "ignore" | "raise"


@dataclass
class EvaluationReport:
    """The full per-item log of one evaluation run.

    Aggregation and conversion to a trial-level :class:`EvaluationResult` are explicit,
    so the same report serves plain evaluation (inspect ``items`` / call ``aggregate``)
    and self-evolution (call ``to_result`` inside an ``evaluate_fn``).
    """

    items: List[ItemResult] = field(default_factory=list)
    elapsed_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Persistence (JSONL: one ItemResult per line)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        """Write all items to ``path`` as JSONL (one serialized ItemResult per line)."""
        dirname = os.path.dirname(os.path.abspath(path))
        os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as f:
            for item in self.items:
                f.write(json.dumps(item.to_json_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "EvaluationReport":
        """Load a report previously written by ``save`` (or by the pipeline's item log)."""
        items: List[ItemResult] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(ItemResult.from_json_dict(json.loads(line)))
        return cls(items=items)

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    @property
    def successes(self) -> List[ItemResult]:
        return [it for it in self.items if it.ok]

    @property
    def failures(self) -> List[ItemResult]:
        return [it for it in self.items if not it.ok]

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def total_errors(self) -> int:
        return len(self.failures)

    @property
    def predictions(self) -> Dict[Any, Any]:
        """id -> prediction for successful items (kept for inspection/back-compat)."""
        return {it.id: it.prediction for it in self.successes}

    @property
    def errors(self) -> List[Tuple[Any, str]]:
        """(id, message) for failed items."""
        return [(it.id, it.error) for it in self.failures]

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, reduce: ReduceFn = "mean", on_error: OnError = "zero") -> Dict[str, float]:
        """Reduce per-item metrics into a single objective-facing dict.

        ``on_error`` decides how failed items count toward the score:
            - ``"zero"``   : failures contribute 0.0 for every metric key (penalized).
                             This is the default — a program that crashes on half the
                             dataset should not look as good as one that solves the other
                             half. Denominator = all items.
            - ``"ignore"`` : average over successful items only (denominator = successes).
                             Use for pure inspection where failures are "no data".
            - ``"raise"``  : raise if any item failed.

        ``reduce`` is ``"mean"`` or a callable ``List[float] -> float`` applied per key
        over the chosen denominator (failed items appear as 0.0 in that list).
        """
        if on_error == "raise" and self.failures:
            raise RuntimeError(
                f"{self.total_errors}/{self.total_items} items failed and on_error='raise'."
            )
        successes = self.successes
        if not successes:
            logger.warning("No successful metrics to aggregate. Returning an empty dict.")
            return {}

        keys = {k for it in successes for k in (it.metrics or {})}
        if on_error == "ignore":
            denom_items = successes
        elif on_error == "zero":
            denom_items = self.items
        else:
            raise ValueError(f"Unsupported on_error: {on_error!r} (use 'zero', 'ignore', or 'raise').")

        if reduce == "mean":
            reducer = lambda xs: (sum(xs) / len(xs) if xs else 0.0)  # noqa: E731
        elif callable(reduce):
            reducer = reduce
        else:
            raise ValueError(f"Unsupported reduce: {reduce!r}")

        return {
            k: reducer([float((it.metrics or {}).get(k, 0.0)) for it in denom_items])
            for k in keys
        }

    # ------------------------------------------------------------------ #
    # Bridge to the optimizer-facing result
    # ------------------------------------------------------------------ #
    def to_result(
        self,
        reduce: ReduceFn = "mean",
        on_error: OnError = "zero",
        include_traces: Union[bool, Callable[[ItemResult], Any]] = False,
        include_artifacts: Union[bool, Callable[["EvaluationReport"], Dict[str, Any]]] = False,
        extra_metadata: Optional[Union[Dict[str, Any], Callable[["EvaluationReport"], Dict[str, Any]]]] = None,
        builder: Optional[Callable[["EvaluationReport"], EvaluationResult]] = None,
    ) -> EvaluationResult:
        """Convert this report into an objective-facing :class:`EvaluationResult`.

        The default packs only aggregated ``metrics`` plus run statistics in ``metadata``
        (all pure-JSON, safe to checkpoint). Opt in to richer payloads:

            - ``include_traces=True``      -> ``traces = [item.trace for item in items]``.
              Pass a callable ``ItemResult -> Any`` to map each item to a trace entry.
            - ``include_artifacts=True``   -> ``artifacts = {id: item.artifacts}``.
              Pass a callable ``report -> dict`` to build artifacts yourself.
            - ``extra_metadata``           -> dict (or ``report -> dict``) merged into metadata.
            - ``builder``                  -> full override; receives the report and returns
              the ``EvaluationResult`` to use (all other args ignored). This is the escape
              hatch for keeping anything the framework can't anticipate.

        NOTE: whatever ends up in ``traces`` / ``artifacts`` / ``metadata`` is persisted
        with the trial state and must be pure JSON. Heavy per-item payloads are better
        left on disk (the pipeline's ``output_dir`` item log) with only paths kept here.
        """
        if builder is not None:
            return builder(self)

        metrics = self.aggregate(reduce=reduce, on_error=on_error)

        traces: List[Any] = []
        if include_traces:
            mapper = include_traces if callable(include_traces) else (lambda it: it.trace)
            traces = [mapper(it) for it in self.items]

        artifacts: Dict[str, Any] = {}
        if include_artifacts:
            if callable(include_artifacts):
                artifacts = include_artifacts(self)
            else:
                artifacts = {it.id: it.artifacts for it in self.items if it.artifacts is not None}

        metadata: Dict[str, Any] = {
            "total_items": self.total_items,
            "total_errors": self.total_errors,
            "elapsed_time": self.elapsed_time,
            "error_ids": [it.id for it in self.failures],
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata(self) if callable(extra_metadata) else extra_metadata)

        return EvaluationResult(metrics=metrics, traces=traces, artifacts=artifacts, metadata=metadata)
