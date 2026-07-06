from dotenv import load_dotenv

from evoagentx.agents.agent_manager import AgentManager
from evoagentx.benchmark import MATH
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.core.logging import logger
from evoagentx.evaluators import Evaluator
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.prompts import StringTemplate
from evoagentx.workflow import SequentialWorkFlowGraph

load_dotenv()

class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into train, dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 10 samples for train, 40 for dev, and 100 for test
        self._train_data = [full_test_data[idx] for idx in permutation[:10]]
        self._dev_data = [full_test_data[idx] for idx in permutation[10:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]


def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{123}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt_template": StringTemplate(instruction="Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}\n"),
            "parse_mode": "str"
        }
    ] 
}


def main(): 

    executor_config = OpenAILLMConfig(model="gpt-4o-mini")
    executor_llm = OpenAILLM(config=executor_config)

    optimizer_config = OpenAILLMConfig(model="gpt-4o")
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, executor_llm.config)

    evaluator = Evaluator(
        llm=executor_llm, 
        agent_manager=agent_manager, 
        collate_func=collate_func, 
        num_workers=20, 
        verbose=True
    )

    textgrad_optimizer = TextGradOptimizer(
        graph=workflow_graph, 
        optimize_mode="all",
        executor_llm=executor_llm, 
        optimizer_llm=optimizer_llm,
        batch_size=3,
        max_steps=20,
        evaluator=evaluator,
        eval_every_n_steps=1,
        eval_rounds=1,
        save_interval=None,
        save_path="./",
        rollback=True,
        constraints=[]
    )

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (before optimization): {results}")

    logger.info("Optimizing workflow...")
    textgrad_optimizer.optimize(benchmark, seed=8)
    textgrad_optimizer.restore_best_graph()

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (after optimization): {results}")


if __name__ == "__main__":
    main() 
