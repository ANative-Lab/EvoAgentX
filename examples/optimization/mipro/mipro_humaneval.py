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
from evoagentx.optimizers.mipro_optimizer import MiproOptimizer, MiproWorkFlowAdapter
from evoagentx.prompts import ChatTemplate
from evoagentx.workflow import SequentialWorkFlowGraph, WorkflowResult

# ---------------------------------------------------------------------------
# Tunables (kept small so the example is cheap to run end-to-end).
# ---------------------------------------------------------------------------
TRAIN_SAMPLE_NUM = 20      # examples used to bootstrap few-shot demos and ground proposals
DEV_SAMPLE_NUM = 30        # examples used to score each candidate during optimization
TEST_SAMPLE_NUM = 50       # examples used for the before/after report
MAX_TRIALS = 10            # total (instruction, demos) combinations the TPE sampler may evaluate
NUM_INSTRUCTION_CANDIDATES = 4  # instruction candidates per agent (index 0 = the original)
NUM_DEMO_CANDIDATES = 4    # few-shot demo sets per agent (index 0 = the baseline, usually empty)
MAX_BOOTSTRAPPED_DEMOS = 3 # bootstrapped demos per set
MAX_LABELED_DEMOS = 2      # labeled demos used to pad each set
EVAL_CONCURRENCY = 10      # how many dev examples to run concurrently per evaluation
EXECUTION_MODEL = "openai/gpt-5.4-mini"          # OpenRouter model that runs the workflow (the program being optimized)
OPTIMIZER_MODEL = "anthropic/claude-sonnet-4.6"  # OpenRouter model that proposes instructions / summarizes the dataset
SAVE_DIR = "debug/mipro_humaneval"  # where the run state, trials, and best snapshot are saved


# A single-node coding workflow defined as a SequentialWorkFlowGraph. Its one prompt is kept
# deliberately bare so the gain from MIPRO's joint instruction + few-shot optimization is
# easy to see. The template's ``instruction`` becomes the PROMPT unit and its
# ``demonstrations`` become the paired demos (FIELD) unit.
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
    """HumanEval with deterministic train/dev/test splits (HumanEval ships only a test set)."""

    def _load_data(self):
        super()._load_data()
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        num_train = int(len(self._test_data) * 0.15)
        num_dev = int(len(self._test_data) * 0.2)
        self._train_data = [self._test_data[i] for i in permutation[:num_train]]
        self._dev_data = [self._test_data[i] for i in permutation[num_train:num_train + num_dev]]
        self._test_data = [self._test_data[i] for i in permutation[num_train + num_dev:]]


# --- The small hooks that adapt the black-box program to the benchmark I/O ----------- #
# input_fn:  benchmark example -> the dict MiproWorkFlowAdapter.execute expects.
# output_fn: the raw WorkflowResult -> the prediction string the scorer/benchmark wants.
# label_fn:  benchmark example -> gold output fields, used to build labeled few-shot demos.
def _example_to_input(example: dict) -> dict:
    return {"question": example["prompt"]}


def _extract_code(execution_result: WorkflowResult) -> str:
    if execution_result.status == "success":
        result = execution_result.result
        return result.get("code", "") if isinstance(result, dict) else result
    return execution_result.error_msg or ""


def _example_to_label(example: dict) -> dict:
    # The demo's "code" field must be a full solution: signature + canonical body.
    return {"code": example["prompt"] + example["canonical_solution"]}


def make_evaluate_fn(benchmark: HumanEval, split: str, sample_num: int, concurrency: int):
    """Build the async ``evaluate_fn`` the optimizer calls on every (baseline + trial) adapter.

    Identical to the SEW example: the evaluation is delegated to an ``EvaluationPipeline``
    (DataLoader + adapter-as-program + benchmark scorer). ``sample_k``+``seed`` keep the
    dev/test subset fixed across trials so candidates are compared fairly, and per-item
    failures score 0 instead of being dropped.
    """
    pipeline = EvaluationPipeline(max_concurrency=concurrency, verbose=False)

    def scorer(prediction, label) -> dict:
        return benchmark.evaluate(prediction=prediction, label=label)

    async def evaluate_fn(adapter: MiproWorkFlowAdapter, ctx: TrialContext = None):
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
        raise ValueError("OPENROUTER_API_KEY not found (drives the MIPRO proposal model).")

    # 1) The program to optimize: a bare single-agent coding workflow.
    llm_config = OpenRouterConfig(
        model=EXECUTION_MODEL,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        top_p=0.85,
        temperature=0.2,
    )
    llm = OpenRouterLLM(config=llm_config)
    workflow_graph = SequentialWorkFlowGraph.from_dict(coding_graph_data)

    # 2) Wrap it: each agent contributes an instruction (PROMPT) unit and a few-shot
    #    demos (FIELD) unit, both searched jointly by the optimizer.
    adapter = MiproWorkFlowAdapter(graph=workflow_graph, llm=llm, llm_config=llm_config)
    logger.info(f"Optimizable units: {[unit.uid for unit in adapter.units]}")

    # 3) Benchmark + evaluation/objective wiring.
    benchmark = HumanEvalSplits()
    objective = ScalarObjective(metric="pass@1", direction="maximize")
    dev_evaluate_fn = make_evaluate_fn(benchmark, "dev", DEV_SAMPLE_NUM, EVAL_CONCURRENCY)
    test_evaluate_fn = make_evaluate_fn(benchmark, "test", TEST_SAMPLE_NUM, EVAL_CONCURRENCY)

    # 4) The optimizer: bootstrap demos on the train split, propose instructions with the
    #    optimizer model, then let TPE search (instruction, demo-set) combinations.
    def metric_fn(prediction: str, example: dict) -> float:
        return benchmark.evaluate(prediction=prediction, label=benchmark.get_label(example))["pass@1"]

    optimizer = MiproOptimizer(
        adapter=adapter,
        optimizer_model=OPTIMIZER_MODEL,
        task_description=workflow_graph.goal,
        trainset=benchmark.get_train_data(sample_k=TRAIN_SAMPLE_NUM, seed=42),
        input_fn=_example_to_input,
        output_fn=_extract_code,
        metric_fn=metric_fn,
        label_fn=_example_to_label,
        num_instruction_candidates=NUM_INSTRUCTION_CANDIDATES,
        num_demo_candidates=NUM_DEMO_CANDIDATES,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        bootstrap_concurrency=EVAL_CONCURRENCY,
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

    # (Optional) load the optimized adapter from a saved state:
    # best_adapter = optimizer.load_optimized(SAVE_DIR)

    # Evaluate the optimized workflow on the same test split.
    after = (await test_evaluate_fn(best_adapter)).metrics
    logger.info(f"Test pass@1 AFTER optimization:  {after['pass@1']:.4f}")

    logger.info("Optimized instructions and demos:")
    for name, prompt in best_adapter.prompts.items():
        logger.info(f"--- {name} (instruction) ---\n{prompt}\n")
        demos = best_adapter.demos.get(name, [])
        logger.info(f"--- {name} ({len(demos)} few-shot demo(s)) ---")
        for i, demo in enumerate(demos):
            logger.info(f"[demo {i}] {demo}")


if __name__ == "__main__":
    asyncio.run(main())
