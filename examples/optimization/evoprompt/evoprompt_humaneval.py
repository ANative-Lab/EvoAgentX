import asyncio
import os

import numpy as np
from dotenv import load_dotenv

from evoagentx.benchmark import HumanEval
from evoagentx.core.logging import logger
from evoagentx.evaluators import DataLoader, EvaluationPipeline
from evoagentx.models import OpenRouterConfig, OpenRouterLLM
from evoagentx.optimizers.engine.adapter import TrialContext
from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.evoprompt_optimizer import EvoPromptGAOptimizer
from evoagentx.optimizers.sew_optimizer import SEWWorkFlowAdapter
from evoagentx.prompts import ChatTemplate
from evoagentx.workflow import SequentialWorkFlowGraph, WorkflowResult

# ---------------------------------------------------------------------------
# Tunables (kept small so the example is cheap to run end-to-end).
# ---------------------------------------------------------------------------
DEV_SAMPLE_NUM = 30        # examples used to score each candidate during optimization
TEST_SAMPLE_NUM = 50       # examples used for the before/after report
POPULATION_SIZE = 4        # configurations kept in the evolving population
MAX_TRIALS = 16            # total evaluations: (POPULATION_SIZE - 1) init + POPULATION_SIZE per generation
EVAL_CONCURRENCY = 10      # how many dev examples to run concurrently per evaluation
EXECUTION_MODEL = "openai/gpt-5.4-mini"          # OpenRouter model running the workflow being optimized
EVOLUTION_MODEL = "anthropic/claude-sonnet-4.6"  # OpenRouter model doing paraphrase / crossover / mutation
SAVE_DIR = "debug/evoprompt_humaneval"           # where the run state and per-trial results are saved


# A single-node coding workflow whose one bare prompt is the optimizable PROMPT unit.
# SEWWorkFlowAdapter is the generic "prompts of a workflow graph" adapter and is reused
# here unchanged; only the optimizer differs from the SEW example.
coding_graph_data = {
    "goal": "Generate functional and correct Python code that completes the given problem.",
    "tasks": [
        {
            "name": "code_generation",
            "description": "Generate the Python code that solves the given coding problem.",
            "inputs": [
                {"name": "question", "type": "str", "required": True, "description": "The coding problem (function signature and docstring)."}
            ],
            "outputs": [
                {"name": "code", "type": "str", "required": True, "description": "The generated Python code."}
            ],
            "prompt_template": ChatTemplate(instruction="Write code for the problem."),
            "parse_mode": "str",
        },
    ],
}


class HumanEvalSplits(HumanEval):
    """HumanEval with a deterministic dev/test split (HumanEval ships only a test set)."""

    def _load_data(self):
        super()._load_data()
        np.random.seed(42)
        num_dev_samples = int(len(self._test_data) * 0.2)
        permutation = np.random.permutation(len(self._test_data))
        self._dev_data = [self._test_data[i] for i in permutation[:num_dev_samples]]
        self._test_data = [self._test_data[i] for i in permutation[num_dev_samples:]]


def _example_to_input(example: dict) -> dict:
    return {"question": example["prompt"]}


def _extract_code(execution_result: WorkflowResult) -> str:
    if execution_result.status == "success":
        result = execution_result.result
        return result.get("code", "") if isinstance(result, dict) else result
    return execution_result.error_msg or ""


def make_evaluate_fn(benchmark: HumanEval, split: str, sample_num: int, concurrency: int):
    """Build the async ``evaluate_fn`` the optimizer calls on every (baseline + trial) adapter.

    Identical wiring to the SEW example: an ``EvaluationPipeline`` over a deterministic
    ``DataLoader`` subset, scoring with ``benchmark.evaluate``. Per-item failures count
    as 0 toward the score. When the optimizer supplies a ``TrialContext``, the full
    per-item report is archived under that trial's directory.
    """
    pipeline = EvaluationPipeline(max_concurrency=concurrency, verbose=False)

    def scorer(prediction, label) -> dict:
        return benchmark.evaluate(prediction=prediction, label=label)

    async def evaluate_fn(adapter: SEWWorkFlowAdapter, ctx: TrialContext = None):
        loader = DataLoader(benchmark, split=split, sample_k=sample_num, seed=42)
        report = await pipeline.arun(
            adapter,
            loader,
            scorer,
            input_fn=_example_to_input,
            output_fn=_extract_code,
            output_dir=os.path.join(ctx.trial_dir, f"eval_{split}") if ctx and ctx.trial_dir else None,
        )
        return report.to_result()

    return evaluate_fn


async def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not found (drives both the workflow and the evolution LLM).")

    # 1) The program to optimize: a bare single-prompt coding workflow.
    llm_config = OpenRouterConfig(
        model=EXECUTION_MODEL,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        top_p=0.85,
        temperature=0.2,
    )
    llm = OpenRouterLLM(config=llm_config)
    workflow_graph = SequentialWorkFlowGraph.from_dict(coding_graph_data)

    # 2) Wrap it: the (single) agent prompt becomes an optimizable PROMPT unit.
    adapter = SEWWorkFlowAdapter(graph=workflow_graph, llm=llm, llm_config=llm_config)
    logger.info(f"Optimizable prompts: {list(adapter.prompts.keys())}")

    # 3) Benchmark + evaluation/objective wiring.
    benchmark = HumanEvalSplits()
    objective = ScalarObjective(metric="pass@1", direction="maximize")
    dev_evaluate_fn = make_evaluate_fn(benchmark, "dev", DEV_SAMPLE_NUM, EVAL_CONCURRENCY)
    test_evaluate_fn = make_evaluate_fn(benchmark, "test", TEST_SAMPLE_NUM, EVAL_CONCURRENCY)

    # 4) The optimizer: GA-style EvoPrompt. Swap in EvoPromptDEOptimizer (same
    # constructor) for the differential-evolution variant.
    evolution_llm = OpenRouterLLM(config=OpenRouterConfig(
        model=EVOLUTION_MODEL,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    ))
    optimizer = EvoPromptGAOptimizer(
        adapter=adapter,
        llm=evolution_llm,
        population_size=POPULATION_SIZE,
        early_stopping_patience=2,
        seed=42,
    )

    # Evaluate the un-optimized workflow on the held-out test split.
    before = (await test_evaluate_fn(adapter)).metrics
    logger.info(f"Test pass@1 BEFORE optimization: {before['pass@1']:.4f}")

    # 5) Optimize. The first POPULATION_SIZE - 1 trials build the initial population
    # (paraphrased variants); each later generation evaluates POPULATION_SIZE children.
    best_adapter = await optimizer.async_optimize(
        evaluate_fn=dev_evaluate_fn,
        objective=objective,
        max_trials=MAX_TRIALS,
        save_dir=SAVE_DIR,
        execution_mode="concurrent",
        max_workers=POPULATION_SIZE,
    )

    # (Optional) load optimized adapter from a saved state
    # best_adapter = optimizer.load_optimized(SAVE_DIR)

    # Evaluate the optimized workflow on the same test split.
    after = (await test_evaluate_fn(best_adapter)).metrics
    logger.info(f"Test pass@1 AFTER optimization:  {after['pass@1']:.4f}")

    logger.info("Optimized prompts:")
    for name, prompt in best_adapter.prompts.items():
        logger.info(f"--- {name} ---\n{prompt}\n")


if __name__ == "__main__":
    asyncio.run(main())
