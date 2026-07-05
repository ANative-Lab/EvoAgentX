"""QA-style scorers (exact-match / F1 / accuracy).

Covers HotPotQA / NQ / RealMMRAG / BIGBenchHard. Supports single- or multi-reference
labels and an em-only mode (BBH).
"""

from typing import Any, List, Tuple

from ..measures import exact_match_score, ems, f1_score, acc_score
from .base import Scorer, Metrics


class QAScorer(Scorer):
    """Exact-match / F1 / accuracy scorer for QA tasks.

    Emits a configurable subset of :attr:`METRIC_KEYS`.

    Args:
        multi_reference: if True, ``label`` is a list of acceptable answers; metrics are
            aggregated over references (EM via ``ems``, F1 via max over references).
        metrics: which metrics to compute, subset of :attr:`METRIC_KEYS` (defaults to all).
        normalize_fn: optional task-specific normalization applied to prediction & label
            before scoring (e.g. dyck-languages whitespace stripping for BBH).
    """

    METRIC_KEYS: Tuple[str, ...] = ("em", "f1", "acc")

    def __init__(
        self,
        multi_reference: bool = False,
        metrics: List[str] = None,
        normalize_fn=None,
    ):
        self.multi_reference = multi_reference
        self.metrics = list(metrics) if metrics else list(self.METRIC_KEYS)
        self.normalize_fn = normalize_fn

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        return tuple(self.metrics)

    def score(self, prediction: Any, label: Any) -> Metrics:
        if self.normalize_fn is not None:
            prediction, label = self.normalize_fn(prediction), self.normalize_fn(label)

        refs: List[Any] = label if self.multi_reference else [label]
        out: Metrics = {}

        if "em" in self.metrics:
            if self.multi_reference:
                out["em"] = ems(prediction=prediction, ground_truths=refs)
            else:
                out["em"] = exact_match_score(prediction=prediction, ground_truth=label)
        if "f1" in self.metrics:
            out["f1"] = max(f1_score(prediction=prediction, ground_truth=r) for r in refs)
        if "acc" in self.metrics:
            out["acc"] = acc_score(prediction=prediction, ground_truths=refs)

        return out
