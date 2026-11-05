# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/optimizer_utils/evaluation_utils.py) under MIT License 

from ...evaluators.aflow_evaluator import AFlowEvaluator
from ...core.callbacks import suppress_logger_info
from ...core.logging import logger

class EvaluationUtils:

    def __init__(self, root_path: str):
        self.root_path = root_path
    
    async def evaluate_graph_async(self, optimizer, validation_n, data, initial=False):

        evaluator = AFlowEvaluator(llm=optimizer.executor_llm)
        sum_score = 0
        completed = 0
        if validation_n < 1:
            raise ValueError("validation_n must be positive")
        
        for _ in range(validation_n):

            with suppress_logger_info():
                score, avg_cost, total_cost, all_failed = await evaluator.graph_evaluate_async(optimizer.benchmark, optimizer.graph, is_test=False)
            cur_round = optimizer.round + 1 if initial is False else optimizer.round 
            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(self.root_path)
            optimizer.data_utils.save_results(result_path, data)
            
            sum_score += score
            completed += 1

            if all_failed:
                logger.warning(f"All test cases failed in round {cur_round}. Stopping evaluation for this round.")
                break 
            
        # Legacy selection averages persisted replicates; feedback must use the
        # same denominator. Reliable mode separately requires complete coverage.
        return sum_score / completed

    async def evaluate_graph_test_async(self, optimizer):

        evaluator = AFlowEvaluator(llm=optimizer.executor_llm)
        with suppress_logger_info():
            score, avg_cost, total_cost, all_failed = await evaluator.graph_evaluate_async(optimizer.benchmark, optimizer.graph, is_test=True)
        return score, avg_cost, total_cost
