"""Math scorers (numeric / symbolic answer matching).

Covers GSM8K (numeric: last-number match) and MATH (symbolic: sympy equality).
"""

from math import isclose
from typing import Any, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

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

    def _extract_numeric(self, text: Any):
        matches = regex.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "").strip()
            try:
                return float(last_number)
            except ValueError:
                return None
        return None

    def _extract_symbolic(self, text: Any):
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = regex.findall(pattern, str(text), regex.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = regex.split(sentence_end_pattern, str(text))
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def _symbolic_equal(self, pred: Any, gold: Any) -> bool:
        if str(pred) == str(gold):
            return True

        try:
            if self._is_digit(pred) and self._is_digit(gold):
                pred = self._parse_digits(pred)
                gold = self._parse_digits(gold)
                return isclose(pred, gold, abs_tol=self.tol)
        except Exception:
            pass

        try:
            return self._symbolic_expression_equal(pred, gold)
        except Exception:
            pass

        return False

    def _is_digit(self, num: Any) -> bool:
        return self._parse_digits(num) is not None

    def _parse_digits(self, num: Any):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except Exception:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except Exception:
                    pass
        return None

    def _symbolic_expression_equal(self, a: Any, b: Any) -> bool:
        def _parse(s: Any) -> Any:
            for parser in (parse_latex, parse_expr):
                try:
                    return parser(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=self.tol):
                return True
        except Exception:
            pass
        return False

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
