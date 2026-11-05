# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/optimizer.py) under MIT License 

import os 
import re 
import shutil
import asyncio
import numpy as np
from tqdm import tqdm
from typing import List, Any, Optional
from pydantic import Field

from ..core.logging import logger
from ..core.module import BaseModule
from ..models.base_model import BaseLLM, LLMOutputParser
from ..benchmark.benchmark import Benchmark
from ..utils.aflow_utils.data_utils import DataUtils
from ..utils.aflow_utils.experience_utils import ExperienceUtils
from ..utils.aflow_utils.evaluation_utils import EvaluationUtils
from ..utils.aflow_utils.graph_utils import GraphUtils, OPERATOR_MAP
from ..utils.aflow_utils.convergence_utils import ConvergenceUtils
from .aflow import ReliableTrialConfig


class GraphOptimizeOutput(LLMOutputParser):

    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class AFlowOptimizer(BaseModule):

    """
    AFlow Optimizer for workflow optimization.
    
    This optimizer iteratively improves workflows through multiple rounds of optimization
    using large language models. It evaluates workflow performance, identifies improvement
    opportunities, and applies optimizations based on experience and convergence metrics.
    
    Attributes:
        question_type: Type of task to optimize for (e.g., qa, match, code)
        graph_path: Path to the workflow graph directory (must contain graph.py and prompt.py)
        optimized_path: Path to save optimized workflows (defaults to graph_path)
        initial_round: Starting round number for optimization
        optimizer_llm: LLM used for generating optimizations
        executor_llm: LLM used for executing the workflow
        operators: List of operators available for optimization
        sample: Number of rounds to sample from for optimization
        max_rounds: Maximum number of optimization rounds to perform
        validation_rounds: Number of validation runs per optimization round
        eval_rounds: Number of evaluation runs for test mode
        check_convergence: Whether to check for optimization convergence
    """
    question_type: str = Field(description="The type of question to optimize the workflow for, e.g., qa, match, code, etc.")
    graph_path: str = Field(description="The folder of the workflow graph. This folder must contain a `graph.py` file that defines the workflow structure, and a `prompt.py` file that defines the prompt for the workflow.")
    optimized_path: str = Field(default=None, description="The path to save the optimized workflow. If not provided, the optimized path will be the same as the graph path.")
    initial_round: int = Field(default=0, description="The round number to start or continue optimization from. If not provided, will start from round 0 using the `graph.py` file in `graph_path`.")
    optimizer_llm: BaseLLM = Field(default=None, description="The LLM to use for optimization.")
    executor_llm: BaseLLM = Field(default=None, description="The LLM to use for execution.")

    operators: List[str] = Field(default_factory=lambda: list(OPERATOR_MAP.keys()), description="The operators to use for optimization. If not provided, will use all operators in OPERATOR_MAP.")
    sample: int = Field(default=4, description="The number of rounds to sample from the top scores.")
    max_rounds: int = Field(default=20, description="The maximum number of rounds to optimize the workflow.")
    validation_rounds: int = Field(default=5, description="Run the workflow for `validation_rounds` times to evaluate the performance on the validation set.")
    eval_rounds: int = Field(default=3, description="Run the workflow for `eval_rounds` times to evaluate the performance on the test set.")
    check_convergence: bool = Field(default=True, description="Whether to check for convergence.")
    reliable_trials: Optional[ReliableTrialConfig] = Field(default=None, description="Opt-in durable AFlow trial policy.")
    max_proposal_attempts: int = Field(default=20, ge=1, description="Bound duplicate proposals in legacy mode.")

    def init_module(self, **kwargs):

        self.root_path = self.optimized_path or self.graph_path
        os.makedirs(self.root_path, exist_ok=True)

        # Initialize utilities
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

        self.graph = None
        self.round = self.initial_round
        self._baseline_complete = False
        if self.reliable_trials is not None and self.initial_round != 0:
            raise ValueError("Reliable trials resume by experiment identity, not initial_round")
        if self.round == 0 and self.reliable_trials is None:
            round_zero_path = os.path.join(self.root_path, f"round_{self.round}")
            os.makedirs(round_zero_path, exist_ok=True)
            shutil.copy2(os.path.join(self.graph_path, "graph.py"), os.path.join(round_zero_path, "graph.py"))
            shutil.copy2(os.path.join(self.graph_path, "prompt.py"), os.path.join(round_zero_path, "prompt.py"))
            self.graph_utils.update_prompt_import(os.path.join(round_zero_path, "graph.py"), round_zero_path)
        
        if self.reliable_trials is None and not os.path.exists(os.path.join(self.root_path, f"round_{self.round}")):
            raise ValueError(f"Round {self.round} does not exist in {self.root_path}")
        
        if self.optimizer_llm is None:
            raise ValueError("optimizer_llm is not provided") 
        if self.executor_llm is None:
            self.executor_llm = self.optimizer_llm

    def optimize(self, benchmark: Benchmark):
        """Run the optimization process on the workflow.
        
        Performs multiple rounds of optimization, evaluating each round against
        the benchmark and checking for convergence. Continues until convergence
        is detected or the maximum number of rounds is reached.
        
        Args:
            benchmark: The benchmark to evaluate the workflow against
        """
        self.benchmark = benchmark
        if self.reliable_trials is not None:
            return asyncio.run(self.optimize_async(benchmark))
        for _ in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                score = loop.run_until_complete(self._execute_with_retry(self._optimize_graph))
            finally:
                loop.close()
            if score is None:
                logger.warning("Optimization round exhausted without completed evidence")
                break
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")
            if self._check_convergence():
                break
            if self.round >= self.max_rounds:
                logger.info(f"Max rounds reached: {self.max_rounds}, stopping optimization.")
                break
    
    async def optimize_async(self, benchmark: Benchmark):
        """Run or resume opt-in reliable evolution in the caller's event loop."""
        if self.reliable_trials is None:
            raise ValueError("optimize_async requires reliable_trials configuration")
        from .aflow.runner import ReliableAFlowRunner
        self.benchmark = benchmark
        self.trial_result = await ReliableAFlowRunner(self, benchmark).run()
        if self.trial_result.best_round is not None:
            self.round = self.trial_result.best_round
        return self.trial_result

    def test(self, benchmark: Benchmark, test_rounds: List[int] = None):
        """Run the test evaluation on optimized workflows.
        
        Evaluates specified rounds (or the best round if none specified) against
        the benchmark multiple times and logs the results.
        
        Args:
            benchmark: The benchmark to evaluate against
            test_rounds: Specific round numbers to test, or None to use the best round
        """
        self.benchmark = benchmark
        if self.reliable_trials is not None:
            return asyncio.run(self.test_async(benchmark, test_rounds))
        if test_rounds is None:
            best_round = self._load_best_round()
            logger.info(f"No test rounds provided, using best round: {best_round}")
            test_rounds = [best_round]
        for _ in tqdm(range(self.eval_rounds)):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_test(test_rounds))
            finally:
                loop.close()

    async def test_async(self, benchmark: Benchmark, test_rounds: List[int] = None):
        """Evaluate a separate reliable test plan without changing parent evidence."""
        if self.reliable_trials is None:
            raise ValueError("test_async requires reliable_trials configuration")
        from .aflow.runner import ReliableAFlowRunner
        self.benchmark = benchmark
        return await ReliableAFlowRunner(self, benchmark).test(test_rounds)
    
    async def _execute_with_retry(self, func: callable, max_retries: int = 3) -> Any:

        retry_count = 0
        while retry_count < max_retries:
            try:
                return await func()
            except Exception as e:
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    logger.info("Max retries reached.")
                    return None
                await asyncio.sleep(5 * retry_count)
        
        return None
    
    def _check_convergence(self) -> bool:
        if not self.check_convergence:
            return False

        converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)
        if converged:
            logger.info(
                f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
            )
            self.convergence_utils.print_results()
            return True
        return False

    async def _optimize_graph(self) -> float:
        """Optimize the graph for one round"""
        validation_n = self.validation_rounds
        graph_path = self.root_path
        data = self.data_utils.load_results(graph_path)

        if self.round == 0 and not self._baseline_complete:
            self.avg_score = await self._handle_initial_round(graph_path, validation_n, data)
            self._baseline_complete = True
        
        return await self._handle_optimization_round(graph_path, validation_n, data)

    async def _handle_initial_round(self, graph_path: str, validation_n: int, data: list) -> float:
        """Handle the initial round of optimization"""
        self.graph_utils.create_round_directory(graph_path, self.round)
        self.graph = self.graph_utils.load_graph(self.round, graph_path)
        return await self.evaluation_utils.evaluate_graph_async(self, validation_n, data, initial=True)

    async def _handle_optimization_round(self, graph_path: str, validation_n: int, data: list) -> float:

        directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

        for _ in range(self.max_proposal_attempts):
            sample = self._get_optimization_sample()
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)
            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])
            operator_description = self.graph_utils.load_operators_description(self.operators, self.optimizer_llm)
            log_data = self.data_utils.load_log(sample["round"])
            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.question_type, log_data
            )
            # response = await self.optimizer_llm.async_generate(prompt=graph_optimize_prompt, parser=GraphOptimizeOutput, parse_mode="xml")
            # response = response.get_structured_data()
            response = await self.optimizer_llm.async_generate(prompt=graph_optimize_prompt, parse_mode="str")
            print(response.content)
            try:
                parsed_response = GraphOptimizeOutput.parse(response.content, parse_mode="xml")
                response = parsed_response.get_structured_data()
            except Exception:
                response = self._parse_optimizer_llm_output(response.content, orig_graph=graph[0], orig_prompt=prompt)

            if self.experience_utils.check_modification(processed_experience, response['modification'], sample["round"]):
                break
        else:
            raise RuntimeError("Duplicate proposal attempt budget exhausted")
        
        # Save and evaluate results
        avg_score = await self._evaluate_and_save_optimization_results(directory, response, sample, data, validation_n)
        return avg_score
    
    def _get_optimization_sample(self) -> dict:

        top_rounds = self.data_utils.get_top_rounds(self.sample)
        return self.data_utils.select_round(top_rounds)

    def _parse_optimizer_llm_output(self, content: str, orig_graph: str, orig_prompt: str) -> dict:

        response = {"modification": "", "graph": "", "prompt": ""}

        # Extract content between <modification> tags
        modification_pattern = r'<modification>(.*?)</modification>'
        modification_match = re.search(modification_pattern, content, re.DOTALL)
        if modification_match:
            response["modification"] = modification_match.group(1).strip()
        
        # extract code block
        code_block_pattern = r'```(?:python)?(.*?)```'
        code_blocks = re.finditer(code_block_pattern, content, re.DOTALL)

        # Process found code blocks
        for block in code_blocks:
            code = block.group(1).strip()
            # If code contains graph-related content, store in graph
            if 'class' in code or 'workflow' in code.lower():
                response["graph"] = code
            # If code contains prompt-related content, store in prompt
            # elif 'PROMPT' in code or 'prompt' in code.lower():
            #     response["prompt"] = code
            else:
                response["prompt"] = code
        
        if not response["graph"] and not response["prompt"]:
            response["modification"] = "No modification due to error in LLM output"
            response["graph"] = orig_graph
            response["prompt"] = orig_prompt 
        
        return response
    
    async def _evaluate_and_save_optimization_results(self, directory: str, response: dict, sample: dict, data: list, validation_n: int):

        # Write optimized files
        self.graph_utils.write_graph_files(directory, response)

        experience = self.experience_utils.create_experience_data(sample, response['modification'])

        self.graph = self.graph_utils.load_graph(self.round + 1, self.root_path)

        # evaluate the graph 
        avg_score = await self.evaluation_utils.evaluate_graph_async(self, validation_n, data, initial=False)
        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    def _load_best_round(self) -> int:
        """Load the best round"""
        if self.reliable_trials is not None:
            from .aflow.runner import ReliableAFlowRunner
            from .aflow.store import TrialStore
            from .aflow.evidence import EvidenceView
            runner = ReliableAFlowRunner(self, self.benchmark)
            with TrialStore(runner.path, runner.plan) as store:
                best = EvidenceView(store).best()
                if best is None:
                    raise ValueError("No eligible reliable validation trial")
                return best.round
        ranked_scores = self.data_utils._load_scores()
        return ranked_scores[0]["round"]

    async def _run_test(self, test_rounds: List[int]):
        """Run test evaluation"""

        logger.info("Running test evaluation...")

        graph_path = self.root_path
        json_file_path = os.path.join(graph_path, "test_results.json")
        import json
        data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, encoding="utf-8") as stream:
                data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError("Invalid test results")
        scores = [] 

        # for round in tqdm(test_rounds, desc="Testing"):
        for round in test_rounds:

            logger.info(f"Running test for round {round}...")

            self.graph = self.graph_utils.load_graph(round, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test_async(self)
            scores.append(score)

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            new_data['split'] = 'test'
            data.append(new_data)

            logger.info(f"Test round {round} score: {score}, avg_cost: {avg_cost}, total_cost: {total_cost}")

            self.data_utils.save_results(json_file_path, data)

        logger.info(f"Test round {round} avg_score: {np.mean(scores)}")
        return np.mean(scores)
