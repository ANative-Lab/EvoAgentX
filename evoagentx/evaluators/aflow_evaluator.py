# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/evaluator.py) under MIT License

import asyncio
from typing import Callable, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio

from ..benchmark.benchmark import Benchmark
from ..core.logging import logger
from ..models.base_model import BaseLLM
from ..models.model_utils import cost_manager
from ..optimizers.aflow.evidence import CostLedger

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above

class AFlowEvaluator:

    """
    AFlow-specific evaluator for workflow performance assessment. 
    This evaluator measures the performance of AFlow workflow graphs against benchmarks.
    
    Attributes:
        llm: The language model to use for evaluation, if needed by the graph
    """

    def __init__( self, llm: Optional[BaseLLM] = None):
        self.llm = llm 
    
    def _configure_graph(self, graph, benchmark):
        return graph(name=benchmark.name, llm_config=self.llm.config, benchmark=benchmark)
    
    async def graph_evaluate_async(self, benchmark: Benchmark, graph: Callable, is_test: bool = False, max_concurrent_tasks: int = 20) -> Tuple[float, float, float]:
        """Asynchronously evaluate a workflow graph against a benchmark.
        
        Configures the graph with benchmark settings, processes all examples in the
        dataset concurrently (up to max_concurrent_tasks), and calculates
        performance metrics including average score, cost per example, and total cost.
        
        Args:
            benchmark: The benchmark to evaluate against
            graph: The workflow graph to evaluate
            is_test: Whether to use test data (True) or validation data (False)
            max_concurrent_tasks: Maximum number of concurrent evaluation tasks
            
        Returns:
            A tuple containing:
              - average_metrics: Mean performance score across all examples
              - avg_cost: Average cost per example
              - total_cost: Total cost for all examples
              - all_failed: Boolean indicating if all evaluations failed
        """
        
        configured_graph = self._configure_graph(graph=graph, benchmark=benchmark)

        # Get evaluation data
        data = benchmark.get_test_data() if is_test else benchmark.get_dev_data()
        if not data:
            logger.warning("No data to evaluate. Returning zeros.")
            return (0.0, 0.0, 0.0, True)
        
        # get total cost before evaluation
        ledger = CostLedger()
        
        # Create a shared semaphore
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def evaluate_with_semaphore(example):
            async with semaphore:
                try:
                    return await benchmark.async_evaluate(configured_graph, example)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {str(e)}")
                    return None
        
        # Create tasks for concurrent execution with semaphore
        tasks = [evaluate_with_semaphore(example) for example in data]

        # Wait for all tasks to complete
        with cost_manager.observe(ledger.observe):
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Evaluating {benchmark.name} problems",
                total=len(data)
            )

        # Replace failed evaluations (None results) with 0
        valid_results = [0.0 if r is None else r for r in results]
        all_failed = all(r is None for r in results)

        # get total cost after evaluation
        total_cost = ledger.close()['known_cost']
        avg_cost = total_cost / len(data)

        if not valid_results:
            logger.warning("No valid results. Returning zeros.")
            avg_metrics = 0.0
        else:
            avg_metrics = sum(valid_results) / len(valid_results)
        
        return avg_metrics, avg_cost, total_cost, all_failed
