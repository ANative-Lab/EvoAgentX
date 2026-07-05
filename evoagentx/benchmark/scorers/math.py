"""Math scorers (numeric / symbolic answer matching).

Covers GSM8K (numeric: last-number match) and MATH (symbolic: sympy equality).

This is a structural skeleton: the actual answer-extraction and equality logic still
lives on the legacy GSM8K / MATH benchmarks and should be migrated into the methods
marked ``TODO`` below.
"""

from typing import Any, Tuple

from .base import Scorer, Metrics


class MathScorer(Scorer):
    """Score a math answer against the ground truth.

    Emits a single fixed metric, :attr:`METRIC_KEYS` (``solve_rate``) — the convention the
    GSM8K / MATH benchmarks already use.

    Args:
        mode: "numeric" (GSM8K-style: extract last number, compare within ``tol``) or
            "symbolic" (MATH-style: extract answer, compare with sympy equality).
        tol: absolute tolerance for numeric comparison.
    """

    METRIC_KEYS: Tuple[str, ...] = ("solve_rate",)

    def __init__(self, mode: str = "numeric", tol: float = 1e-6):
        assert mode in ("numeric", "symbolic"), f"Invalid mode: {mode}"
        self.mode = mode
        self.tol = tol

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        return self.METRIC_KEYS

    # --- TODO: migrate from GSM8K.extract_last_number --------------------- #
    def _extract_numeric(self, text: Any):
        raise NotImplementedError("Migrate GSM8K.extract_last_number here.")

    # --- TODO: migrate from MATH.extract_answer / math_equal -------------- #
    def _extract_symbolic(self, text: Any):
        raise NotImplementedError("Migrate MATH.extract_answer here.")

    def _symbolic_equal(self, pred: Any, gold: Any) -> bool:
        raise NotImplementedError("Migrate MATH.math_equal here.")

    def score(self, prediction: Any, label: Any) -> Metrics:
        if self.mode == "numeric":
            pred = self._extract_numeric(prediction)
            gold = self._extract_numeric(label)
            solved = pred is not None and gold is not None and abs(pred - gold) < self.tol
        else:
            pred = self._extract_symbolic(prediction)
            gold = self._extract_symbolic(label)
            solved = self._symbolic_equal(pred, gold)
        return {self.METRIC_KEYS[0]: 1.0 if solved else 0.0}
