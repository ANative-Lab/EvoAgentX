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
from evoagentx.optimizers.textgrad_optimizer import TextGradOptimizer, TextGradWorkFlowAdapter
from evoagentx.prompts import ChatTemplate
from evoagentx.workflow import SequentialWorkFlowGraph, WorkflowResult

# ---------------------------------------------------------------------------
# Tunables (kept small so the example is cheap to run end-to-end).
# ---------------------------------------------------------------------------
DEV_SAMPLE_NUM = 30        # examples used to score the prompts after each TextGrad step
TEST_SAMPLE_NUM = 50       # examples used for the before/after report
MAX_TRIALS = 5             # evaluated trials; each trial = GRADIENT_STEPS_PER_TRIAL TextGrad steps
GRADIENT_STEPS_PER_TRIAL = 1  # TextGrad steps folded into one evaluated trial (legacy eval_every_n_steps)
BATCH_SIZE = 3             # training examples per TextGrad step (forward + textual backward)
EVAL_CONCURRENCY = 10      # how many dev examples to run concurrently per evaluation
EXECUTION_MODEL = "openai/gpt-5.4-mini"          # OpenRouter model that runs the workflow (the program being optimized)
OPTIMIZER_MODEL = "anthropic/claude-sonnet-4.6"  # OpenRouter model that computes textual gradients and prompt updates
SAVE_DIR = "debug/textgrad_humaneval"            # where the run state and per-trial results are saved


# A single-node coding workflow. TextGrad optimizes BOTH the agent's system prompt and its
# prompt template's instruction (two PROMPT units); both start deliberately bare so the
# gain from optimization is easy to see.
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
    """HumanEval with deterministic train/dev/test splits (HumanEval ships only a test set).

    TextGrad additionally needs a train split: its forward/backward passes consume labeled
    training batches, separate from the dev examples used to score each trial.
    """

    def _load_data(self):
        super()._load_data()
        np.random.seed(42)
        num_train_samples = 10
        num_dev_samples = int(len(self._test_data) * 0.2)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        self._train_data = [full_test_data[i] for i in permutation[:num_train_samples]]
        self._dev_data = [full_test_data[i] for i in permutation[num_train_samples:num_train_samples + num_dev_samples]]
        self._test_data = [full_test_data[i] for i in permutation[num_train_samples + num_dev_samples:]]


# --- The small hooks that adapt the black-box program to the benchmark I/O ----------- #
# collate/input_fn: benchmark example -> the dict TextGradWorkFlowAdapter.execute expects.
# output_fn: the raw WorkflowResult -> the prediction string the scorer/benchmark wants.
def _example_to_input(example: dict) -> dict:
    return {"question": example["prompt"]}


def _extract_code(execution_result: WorkflowResult) -> str:
    if execution_result.status == "success":
        result = execution_result.result
        return result.get("code", "") if isinstance(result, dict) else result
    return execution_result.error_msg or ""


def make_evaluate_fn(benchmark: HumanEval, split: str, sample_num: int, concurrency: int):
    """Build the async ``evaluate_fn`` the optimizer calls on every (baseline + trial) adapter.

    Identical wiring to the SEW example: DataLoader (which examples) + program (the adapter)
    + scorer (benchmark.evaluate), run through a reusable ``EvaluationPipeline``. When the
    optimizer passes a ``TrialContext``, the full per-item log is archived under that trial's
    directory in ``<save_dir>/trials/``.
    """
    pipeline = EvaluationPipeline(max_concurrency=concurrency, verbose=False)

    def scorer(prediction, label) -> dict:
        return benchmark.evaluate(prediction=prediction, label=label)

    async def evaluate_fn(adapter: TextGradWorkFlowAdapter, ctx: TrialContext = None):
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
        raise ValueError("OPENROUTER_API_KEY not found (drives both the workflow and the TextGrad LLM).")

    # 1) The program to optimize: a single-agent coding workflow.
    llm_config = OpenRouterConfig(
        model=EXECUTION_MODEL,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        top_p=0.85,
        temperature=0.2,
    )
    llm = OpenRouterLLM(config=llm_config)
    workflow_graph = SequentialWorkFlowGraph.from_dict(coding_graph_data)

    # 2) Wrap it: every agent contributes two PROMPT units (system prompt + instruction).
    adapter = TextGradWorkFlowAdapter(graph=workflow_graph, llm=llm, llm_config=llm_config)
    logger.info(f"Optimizable units: {[unit.uid for unit in adapter.units]}")

    # 3) Benchmark + evaluation/objective wiring.
    benchmark = HumanEvalSplits()
    objective = ScalarObjective(metric="pass@1", direction="maximize")
    dev_evaluate_fn = make_evaluate_fn(benchmark, "dev", DEV_SAMPLE_NUM, EVAL_CONCURRENCY)
    test_evaluate_fn = make_evaluate_fn(benchmark, "test", TEST_SAMPLE_NUM, EVAL_CONCURRENCY)

    # 4) The optimizer: textual gradient descent over the workflow's prompts. Each trial
    #    runs the workflow on a labeled train batch, backpropagates a textual loss (for this
    #    coding benchmark the label embeds the generated code's unit-test result), updates
    #    the prompts, and hands them to the engine for evaluation on the dev split.
    optimizer_llm = OpenRouterLLM(
        config=OpenRouterConfig(model=OPTIMIZER_MODEL, openrouter_key=os.getenv("OPENROUTER_API_KEY"))
    )
    optimizer = TextGradOptimizer(
        adapter=adapter,
        optimizer_llm=optimizer_llm,
        benchmark=benchmark,
        collate_func=_example_to_input,
        optimize_mode="all",       # optimize both system prompts and instructions
        batch_size=BATCH_SIZE,
        use_answers=True,
        gradient_steps_per_trial=GRADIENT_STEPS_PER_TRIAL,
        rollback=True,             # seed each trial from the best snapshot found so far
        seed=42,
    )

    # Evaluate the un-optimized workflow on the held-out test split.
    before = (await test_evaluate_fn(adapter)).metrics
    logger.info(f"Test pass@1 BEFORE optimization: {before['pass@1']:.4f}")

    # 5) Optimize. Returns the adapter rebuilt from the best-scoring snapshot.
    best_adapter = await optimizer.async_optimize(
        evaluate_fn=dev_evaluate_fn,
        objective=objective,
        max_trials=MAX_TRIALS,
        save_dir=SAVE_DIR,
    )

    # (Optional) load optimized adapter from a saved state
    # best_adapter = optimizer.load_optimized(SAVE_DIR)

    # Evaluate the optimized workflow on the same test split.
    after = (await test_evaluate_fn(best_adapter)).metrics
    logger.info(f"Test pass@1 AFTER optimization:  {after['pass@1']:.4f}")

    logger.info("Optimized prompts:")
    for name in best_adapter.prompts:
        logger.info(f"--- {name} (system prompt) ---\n{best_adapter.system_prompts[name]}\n")
        logger.info(f"--- {name} (instruction) ---\n{best_adapter.prompts[name]}\n")


if __name__ == "__main__":
    asyncio.run(main())
