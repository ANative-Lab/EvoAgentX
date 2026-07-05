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
from evoagentx.optimizers.sew_optimizer import SEWOptimizer, SEWWorkFlowAdapter
from evoagentx.prompts import ChatTemplate
from evoagentx.workflow import SequentialWorkFlowGraph, WorkflowResult

# ---------------------------------------------------------------------------
# Tunables (kept small so the example is cheap to run end-to-end).
# ---------------------------------------------------------------------------
DEV_SAMPLE_NUM = 30        # examples used to score each prompt variant during optimization
TEST_SAMPLE_NUM = 50       # examples used for the before/after report
MAX_TRIALS = 10            # total prompt variants the optimizer is allowed to evaluate
NUM_VARIANTS_PER_STEP = 2  # variants generated per round (the "λ" in (1 + λ))
EVAL_CONCURRENCY = 10      # how many dev examples to run concurrently per evaluation
EXECUTION_MODEL = "openai/gpt-5.4-mini" # OpenRouter model used to run the workflow (the program being optimized)
OPTIMIZER_MODEL = "anthropic/claude-sonnet-4.6"  # OpenRouter model id used to mutate the prompts
SAVE_DIR = "debug/sew_humaneval_final" # where to save the intermediate variants and final best state (SEWOptimizer.async_optimize's save_dir)


# A single-node coding workflow defined as a SequentialWorkFlowGraph. Its one prompt is kept
# deliberately bare so the gain from SEW's prompt optimization is easy to see. That prompt
# template's ``instruction`` is the single optimizable PROMPT unit.
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


# --- The three small hooks that adapt the black-box program to the benchmark I/O ----- #
# input_fn:  benchmark example  -> the dict SEWWorkFlowAdapter.execute expects.
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

    The whole evaluation is delegated to a reusable ``EvaluationPipeline``:
        DataLoader (which examples) + program (the adapter) + scorer (benchmark.evaluate).
    The adapter is the ``program``; the pipeline calls it via ``__call__`` (-> async_execute
    inside the running loop). Per-item failures are captured and, via ``on_error="zero"``
    (the ``to_result`` default), count as 0 toward the score — matching SEW's semantics that
    a crashing solution scores 0 rather than being silently dropped.

    ``evaluate_fn`` takes an optional second ``TrialContext`` argument: when the optimizer
    calls it, ``ctx.trial_dir`` points at this trial's persistent directory under
    ``<save_dir>/trials/``, and the pipeline streams the full per-item log (predictions,
    metrics, tracebacks) there as ``report.jsonl``. Called directly (e.g. the before/after
    test evaluation below), ``ctx`` is None and nothing is written to disk.
    """
    # A new DataLoader per call keeps each evaluation independent; ``sample_k``+``seed`` make
    # the dev/test subset deterministic across every trial so candidates are compared fairly.
    pipeline = EvaluationPipeline(max_concurrency=concurrency, verbose=False)

    def scorer(prediction, label) -> dict:
        return benchmark.evaluate(prediction=prediction, label=label)

    async def evaluate_fn(adapter: SEWWorkFlowAdapter, ctx: TrialContext=None):
        # No suppress_logger_info() needed: SEWWorkFlowAdapter.async_execute suppresses
        # the workflow's per-step logs itself (contextvar-scoped, so it's concurrency-safe).
        loader = DataLoader(benchmark, split=split, sample_k=sample_num, seed=42)
        report = await pipeline.arun(
            adapter,
            loader,
            scorer,
            input_fn=_example_to_input,
            output_fn=_extract_code,
            output_dir=os.path.join(ctx.trial_dir, f"eval_{split}") if ctx and ctx.trial_dir else None,
        )
        # Explicit bridge: per-item report -> objective-facing EvaluationResult.
        return report.to_result()

    return evaluate_fn


async def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY not found (drives the SEW prompt-mutation model).")

    # 1) The program to optimize: the built-in SEW coding workflow.
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

    # 4) The optimizer: re-mutate every prompt each round, steered by the workflow goal.
    optimizer = SEWOptimizer(
        adapter=adapter,
        optimizer_model=OPTIMIZER_MODEL,
        task_description=workflow_graph.goal,
        num_variants_per_step=NUM_VARIANTS_PER_STEP,
        order="first-order",  # "second-order" synthesizes a role-aware mutation directive first
        seed=42,
    )

    # Evaluate the un-optimized workflow on the held-out test split. evaluate_fn now returns
    # an EvaluationResult (the optimizer-facing summary); its aggregated metrics are in .metrics.
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
    for name, prompt in best_adapter.prompts.items():
        logger.info(f"--- {name} ---\n{prompt}\n")


if __name__ == "__main__":
    asyncio.run(main())
