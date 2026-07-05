"""Code-generation scorers (pass@k via sandboxed execution).

Covers HumanEval / MBPP (``CodeScorer``) and LiveCodeBench (``LiveCodeBenchScorer``).

This is a structural skeleton. The execution + pass@k logic currently lives on the
legacy ``CodingBenchmark`` and should be migrated into the methods marked ``TODO``:
    - ``check_solution`` / ``handle_special_cases`` / ``_check_evaluation_inputs``
    - ``compute_pass_at_k``
along with the ``SUCCESS / FAILED / TIMEOUT`` constants.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...core.callbacks import timeout, TimeoutException  # noqa: F401  (used after migration)
from ...utils.sanitize import sanitize  # noqa: F401  (used after migration)
from ..lcb_utils.evaluation import estimate_pass_at_k
from .base import Scorer, Metrics


class CodeScorer(Scorer):
    """pass@k scorer for HumanEval / MBPP-style tasks.

    Unlike the fixed-metric QA/Math scorers, the emitted keys are parametric: one
    ``pass@{k}`` per configured ``k`` (see :attr:`metric_keys`).

    Args:
        k: a single k or a list of k values for pass@k.
        timeout: per-test execution timeout (seconds).
        prompt_joiner: how the dataset prompt is concatenated with the solution before
            execution (HumanEval: ``prompt + solution``; MBPP: ``prompt + "\\n" + solution``).
    """

    SUCCESS, FAILED, TIMEOUT = 0, 1, 2

    def __init__(self, k: Union[int, List[int]] = 1, timeout: int = 60, prompt_joiner: str = ""):
        self.k = k
        self.timeout = timeout
        self.prompt_joiner = prompt_joiner

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        k_list = self.k if isinstance(self.k, (list, tuple)) else [self.k]
        return tuple(f"pass@{k}" for k in k_list)

    # --- TODO: migrate from CodingBenchmark ------------------------------- #
    def _check_inputs(self, prediction: Any, label: Any) -> Tuple[List[str], List[dict]]:
        raise NotImplementedError("Migrate CodingBenchmark._check_evaluation_inputs here.")

    def handle_special_cases(self, task_id: str, solution: str, test: str):
        return solution, test

    def check_solution(
        self,
        task_id: str,
        solution: str,
        test: str,
        entry_point: Optional[str] = None,
        use_entrypoint_as_input: bool = True,
    ) -> Tuple[int, str]:
        raise NotImplementedError("Migrate CodingBenchmark.check_solution here.")

    def compute_pass_at_k(self, results: List[bool], k_list: List[int]) -> Dict[str, float]:
        pass_at_k: Dict[str, float] = {}
        n, c = len(results), sum(results)
        for k in k_list:
            if n >= k:
                pass_at_k[f"pass@{k}"] = float(estimate_pass_at_k(np.array([n]), np.array([c]), k)[0])
        return pass_at_k

    def score(self, prediction: Any, label: Any) -> Metrics:
        # prediction: solution code (or list of samples); label: unit test spec(s).
        # TODO: run check_solution per sample, then compute_pass_at_k.
        raise NotImplementedError("Wire up CodeScorer.score after migrating CodingBenchmark.")


class LiveCodeBenchScorer(Scorer):
    """LiveCodeBench scorer (delegates to ``lcb_utils`` metrics).

    Handles the three scenarios: code_generation / test_output_prediction / code_execution.
    """

    def __init__(self, scenario: str = "code_generation", k=1, timeout: int = 60, num_process: int = 1):
        self.scenario = scenario
        self.k = k
        self.timeout = timeout
        self.num_process = num_process

    def score(self, prediction: Any, label: Any) -> Metrics:
        # TODO: migrate the scenario-specific lcb_utils metric calls from LiveCodeBench.evaluate.
        raise NotImplementedError("Wire up LiveCodeBenchScorer.score after migrating LiveCodeBench.")
