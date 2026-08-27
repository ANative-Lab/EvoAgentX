"""Code-generation scorers (pass@k via sandboxed execution).

Covers HumanEval / MBPP (``CodeScorer``) and LiveCodeBench
(``LiveCodeBenchScorer``).
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ...core.callbacks import timeout, TimeoutException
from ...core.module_utils import extract_code_blocks
from ...utils.sanitize import sanitize
from ...optimizers.core.scoring import Scorer, ScoreRequest, ScoreResult, Metrics
from ..lcb_utils.evaluation import (
    code_execution_metrics,
    codegen_metrics,
    estimate_pass_at_k,
    test_output_metrics,
)
from ..lcb_utils.utils import extract_execution_code, extract_test_output_code


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

    def __init__(
        self,
        k: Union[int, List[int]] = 1,
        timeout: int = 60,
        prompt_joiner: str = "",
        prompt_getter: Optional[Callable[[Any], str]] = None,
        special_cases_handler: Optional[Callable[[Any, str, str], Tuple[str, str]]] = None,
        use_entrypoint_as_input: bool = True,
    ):
        self.k = k
        self.timeout = timeout
        self.prompt_joiner = prompt_joiner
        self.prompt_getter = prompt_getter
        self.special_cases_handler = special_cases_handler
        self.use_entrypoint_as_input = use_entrypoint_as_input

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        k_list = self.k if isinstance(self.k, (list, tuple)) else [self.k]
        return tuple(f"pass@{k}" for k in k_list)

    def _check_inputs(self, prediction: Any, label: Any) -> Tuple[List[str], List[dict]]:
        assert isinstance(prediction, (str, list)), (
            "prediction must be a string or a list of strings, but got {}".format(type(prediction))
        )
        assert isinstance(label, (dict, list)), (
            "label must be a dict or a list of dicts, but got {}".format(type(label))
        )
        prediction = [prediction] if isinstance(prediction, str) else prediction
        label = [label] if isinstance(label, dict) else label
        return prediction, label

    def handle_special_cases(self, task_id: str, solution: str, test: str):
        if self.special_cases_handler is not None:
            return self.special_cases_handler(task_id, solution, test)
        return solution, test

    def check_solution(
        self,
        task_id: str,
        solution: str,
        test: str,
        entry_point: Optional[str] = None,
        use_entrypoint_as_input: bool = True,
    ) -> Tuple[int, str]:
        solution = sanitize(solution, entrypoint=entry_point)

        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }
            solution, test = self.handle_special_cases(task_id=task_id, solution=solution, test=test)
            exec(solution, global_dict)
            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} not found in the solution code.")
            exec(test, global_dict)
            unit_test_func = global_dict["check"]
            with timeout(seconds=self.timeout):
                if use_entrypoint_as_input:
                    unit_test_func(global_dict[entry_point])
                else:
                    unit_test_func()
            result = (self.SUCCESS, "The solution passed the unit test.")

        except TimeoutException:
            result = (self.TIMEOUT, "Execution timed out.")

        except Exception as e:
            error_msg = f"An error occurred: {e}\nSolution:\n{solution}\nTest:\n{test}"
            result = (self.FAILED, error_msg)

        return result

    def compute_pass_at_k(self, results: List[bool], k_list: List[int]) -> Dict[str, float]:
        pass_at_k: Dict[str, float] = {}
        n, c = len(results), sum(results)
        for k in k_list:
            if n >= k:
                pass_at_k[f"pass@{k}"] = float(estimate_pass_at_k(np.array([n]), np.array([c]), k)[0])
        return pass_at_k

    async def score(self, req: ScoreRequest) -> ScoreResult:
        return ScoreResult(metrics=await asyncio.to_thread(self._compute, req.prediction, req.reference))

    def _compute(self, prediction: Any, label: Any) -> Metrics:
        prediction, label = self._check_inputs(prediction, label)

        results = []
        for solution in prediction:
            solution_states = []
            for label_data in label:
                task_id = label_data["task_id"]
                prompt = label_data.get("prompt")
                if prompt is None and self.prompt_getter is not None:
                    prompt = self.prompt_getter(task_id)
                if prompt is None:
                    prompt = ""

                state, message = self.check_solution(
                    task_id=task_id,
                    solution=prompt + self.prompt_joiner + solution,
                    test=label_data["test"],
                    entry_point=label_data["entry_point"],
                    use_entrypoint_as_input=self.use_entrypoint_as_input,
                )
                if state != self.SUCCESS:
                    break
                solution_states.append(state)
            results.append(len(solution_states) == len(label) and all(state == self.SUCCESS for state in solution_states))

        k_list = [self.k] if isinstance(self.k, int) else self.k
        return self.compute_pass_at_k(results, k_list)


class LiveCodeBenchScorer(Scorer):
    """LiveCodeBench scorer (delegates to ``lcb_utils`` metrics).

    Handles the three scenarios: code_generation / test_output_prediction / code_execution.
    """

    VALID_SCENARIO = ["code_generation", "test_output_prediction", "code_execution"]

    def __init__(
        self,
        scenario: str = "code_generation",
        k=1,
        timeout: int = 60,
        num_process: int = 1,
        use_cot_for_execution: bool = False,
    ):
        assert scenario in self.VALID_SCENARIO, (
            f"Invalid scenario: {scenario}. Available choices: {self.VALID_SCENARIO}."
        )
        self.scenario = scenario
        self.k = k
        self.timeout = timeout
        self.num_process = num_process
        self.use_cot_for_execution = use_cot_for_execution

    @property
    def metric_keys(self) -> Tuple[str, ...]:
        if self.scenario == "code_execution":
            return ("pass@1",)
        k_list = self.k if isinstance(self.k, (list, tuple)) else [self.k]
        return tuple(f"pass@{k}" for k in k_list)

    def _check_inputs(self, prediction: Any, label: Any) -> Tuple[List[str], List[dict]]:
        assert isinstance(prediction, (str, list)), (
            "prediction must be a string or a list of strings, but got {}".format(type(prediction))
        )
        assert isinstance(label, (dict, list)), (
            "label must be a dict or a list of dicts, but got {}".format(type(label))
        )
        prediction = [prediction] if isinstance(prediction, str) else prediction
        label = [label] if isinstance(label, dict) else label
        return prediction, label

    async def score(self, req: ScoreRequest) -> ScoreResult:
        return ScoreResult(metrics=await asyncio.to_thread(self._compute, req.prediction, req.reference))

    def _compute(self, prediction: Any, label: Any) -> Metrics:
        prediction, label = self._check_inputs(prediction, label)
        k_list = [self.k] if isinstance(self.k, int) else self.k

        if self.scenario == "code_generation":
            solutions: List[str] = [extract_code_blocks(pred)[0] for pred in prediction]
            metrics, results, metadatas = codegen_metrics(
                samples_list=label,
                generations_list=[solutions],
                k_list=k_list,
                num_process_evaluate=self.num_process,
                timeout=self.timeout,
            )
        elif self.scenario == "test_output_prediction":
            pred_outputs = [extract_test_output_code(pred) for pred in prediction]
            metrics, results = test_output_metrics(
                samples=label,
                generations=[pred_outputs],
                k_list=k_list,
            )
        elif self.scenario == "code_execution":
            pred_outputs = [extract_execution_code(pred, self.use_cot_for_execution) for pred in prediction]
            metrics, results = code_execution_metrics(
                samples=label,
                generations=[pred_outputs],
            )
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Available choices: {self.VALID_SCENARIO}.")

        return {key: float(metrics[key]) for key in self.metric_keys if key in metrics}
