"""QA-style scorers (exact-match / F1 / accuracy).

Covers HotPotQA / NQ / RealMMRAG / BIGBenchHard. Supports single- or multi-reference
labels and an em-only mode (BBH).
"""

from typing import Any, List

from ..measures import exact_match_score, ems, f1_score, acc_score
from .base import Scorer, Metrics


class QAScorer(Scorer):
    """Exact-match / F1 / accuracy scorer for QA tasks.

    Args:
        multi_reference: if True, ``label`` is a list of acceptable answers; metrics are
            aggregated over references (EM via ``ems``, F1 via max over references).
        metrics: which metrics to compute, subset of {"em", "f1", "acc"}.
        normalize_fn: optional task-specific normalization applied to prediction & label
            before scoring (e.g. dyck-languages whitespace stripping for BBH).
    """

    def __init__(
        self,
        multi_reference: bool = False,
        metrics: List[str] = None,
        normalize_fn=None,
    ):
        self.multi_reference = multi_reference
        self.metrics = metrics or ["em", "f1", "acc"]
        self.normalize_fn = normalize_fn

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
