import asyncio
import os
from typing import Any, Dict, List, Optional

import pytest

from evoagentx.optimizers.engine.adapter import ProgramAdapter, SnapShot
from evoagentx.optimizers.engine.base import (
    ChangeOperation,
    OptimizationUnit,
    OptimizationUnitType,
    UnitChange,
)
from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.mipro_optimizer import (
    DEMOS_UID_SUFFIX,
    MiproOptimizer,
    MiproWorkFlowAdapter,
)


# ---------------------------------------------------------------------------
# Synthetic in-memory adapter (no LLM, no workflow).
# ---------------------------------------------------------------------------
class FakeQAAdapter(ProgramAdapter):
    """A toy program that answers a question by reversing it; demos are optimizable."""

    def __init__(self, instruction: str = "Answer the question.", demos: Optional[List[Dict[str, Any]]] = None):
        self.instruction = instruction
        self.demos = [dict(demo) for demo in (demos or [])]

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name="qa",
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string"},
                metadata={"role": "Answer the user's question."},
            ),
            OptimizationUnit(
                name=f"qa{DEMOS_UID_SUFFIX}",
                unit_type=OptimizationUnitType.FIELD,
                json_schema={"type": "array", "items": {"type": "object"}},
                metadata={
                    "kind": "demos",
                    "paired_prompt_uid": "qa",
                    "input_names": ["q"],
                    "output_names": ["a"],
                },
            ),
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"qa": self.instruction, f"qa{DEMOS_UID_SUFFIX}": [dict(d) for d in self.demos]})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        new_values = dict(snapshot.unit_values)
        for change in changes:
            new_values[change.uid] = change.value
        return SnapShot(unit_values=new_values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "FakeQAAdapter":
        return type(self)(
            instruction=snapshot.unit_values["qa"],
            demos=snapshot.unit_values.get(f"qa{DEMOS_UID_SUFFIX}") or [],
        )

    def execute(self, inputs: Optional[dict] = None, **kwargs) -> str:
        return (inputs or {}).get("q", "")[::-1]

    def execute_with_trace(self, inputs: Optional[dict] = None, **kwargs):
        prediction = self.execute(inputs)
        return prediction, {f"qa{DEMOS_UID_SUFFIX}": {"q": (inputs or {}).get("q", ""), "a": prediction}}

    async def async_execute_with_trace(self, inputs: Optional[dict] = None, **kwargs):
        return self.execute_with_trace(inputs, **kwargs)


TRAINSET = [{"q": "ab"}, {"q": "cd"}, {"q": "ef"}, {"q": "gh"}, {"q": "ij"}]


def make_optimizer(adapter: FakeQAAdapter, **overrides) -> MiproOptimizer:
    params = dict(
        adapter=adapter,
        optimizer_model="test/model",
        task_description="Reverse the question to answer it.",
        trainset=TRAINSET,
        input_fn=lambda ex: {"q": ex["q"]},
        output_fn=lambda result: result,
        metric_fn=lambda prediction, ex: prediction == ex["q"][::-1],
        label_fn=lambda ex: {"a": ex["q"][::-1]},
        # 1 instruction candidate == the original only -> no proposal LLM calls needed.
        num_instruction_candidates=1,
        num_demo_candidates=3,
        max_bootstrapped_demos=2,
        max_labeled_demos=1,
        use_dataset_summary=False,
        trials_per_batch=2,
        openrouter_key="test-key",
        seed=7,
    )
    params.update(overrides)
    return MiproOptimizer(**params)


def evaluate_fn(adapter: FakeQAAdapter) -> Dict[str, float]:
    # More demos -> better score, so the search should move away from the empty baseline.
    return {"score": 0.5 + 0.1 * len(adapter.demos)}


def test_mipro_optimize_end_to_end(tmp_path):
    adapter = FakeQAAdapter()
    optimizer = make_optimizer(adapter)
    objective = ScalarObjective(metric="score", direction="maximize")

    best = optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=objective,
        max_trials=4,
        save_dir=str(tmp_path),
    )

    # Demo candidate sets were built: baseline (empty) + labeled + bootstrapped.
    demo_sets = optimizer._demo_candidates[f"qa{DEMOS_UID_SUFFIX}"]
    assert len(demo_sets) == 3
    assert demo_sets[0] == []  # baseline set
    assert all(demo["a"] == demo["q"][::-1] for demo_set in demo_sets[1:] for demo in demo_set)

    # The best adapter should carry a non-empty demo set (they strictly improve the score).
    assert isinstance(best, FakeQAAdapter)
    assert len(best.demos) > 0

    # The run state was persisted with the candidates and optuna history.
    assert os.path.isfile(os.path.join(str(tmp_path), "optimization_state.json"))
    assert optimizer._history, "optuna trial history should be recorded"

    # load_optimized restores the same best configuration.
    reloaded = optimizer.load_optimized(str(tmp_path))
    assert reloaded.demos == best.demos
    assert reloaded.instruction == best.instruction


def test_mipro_resume_skips_candidate_generation(tmp_path):
    adapter = FakeQAAdapter()
    objective = ScalarObjective(metric="score", direction="maximize")
    optimizer = make_optimizer(adapter)
    optimizer.optimize(evaluate_fn=evaluate_fn, objective=objective, max_trials=2, save_dir=str(tmp_path))
    first_candidates = optimizer._demo_candidates

    # Resume with a fresh optimizer whose bootstrap path would fail if invoked again.
    resumed = make_optimizer(adapter, input_fn=lambda ex: (_ for _ in ()).throw(RuntimeError("no rollouts on resume")))
    best = resumed.optimize(
        evaluate_fn=evaluate_fn,
        objective=objective,
        max_trials=4,
        resume_from=str(tmp_path),
    )
    assert resumed._demo_candidates == first_candidates
    assert isinstance(best, FakeQAAdapter)


def test_mipro_resume_rejects_changed_candidate_config(tmp_path):
    adapter = FakeQAAdapter()
    objective = ScalarObjective(metric="score", direction="maximize")
    make_optimizer(adapter).optimize(
        evaluate_fn=evaluate_fn, objective=objective, max_trials=2, save_dir=str(tmp_path)
    )

    # The persisted optuna history indexes into the persisted candidate lists, so a
    # resume with different candidate-generation settings must fail fast instead of
    # silently reusing (or regenerating) candidates.
    changed = make_optimizer(adapter, num_demo_candidates=5, task_description="A different goal.")
    with pytest.raises(ValueError, match="num_demo_candidates.*|task_description.*"):
        changed.optimize(
            evaluate_fn=evaluate_fn,
            objective=objective,
            max_trials=4,
            resume_from=str(tmp_path),
        )


def test_mipro_async_optimize_concurrent(tmp_path):
    adapter = FakeQAAdapter()
    optimizer = make_optimizer(adapter)
    objective = ScalarObjective(metric="score", direction="maximize")

    async def async_evaluate_fn(adapter: FakeQAAdapter) -> Dict[str, float]:
        return evaluate_fn(adapter)

    best = asyncio.run(
        optimizer.async_optimize(
            evaluate_fn=async_evaluate_fn,
            objective=objective,
            max_trials=4,
            save_dir=str(tmp_path),
            execution_mode="concurrent",
            max_workers=2,
        )
    )
    assert isinstance(best, FakeQAAdapter)
    assert len(best.demos) > 0


def test_mipro_requires_scalar_objective(tmp_path):
    from evoagentx.optimizers.engine.objective import ParetoObjective

    adapter = FakeQAAdapter()
    optimizer = make_optimizer(adapter)
    with pytest.raises(TypeError, match="ScalarObjective"):
        optimizer.optimize(
            evaluate_fn=evaluate_fn,
            objective=ParetoObjective(metrics=["score"]),
            max_trials=2,
            save_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Action-level prompt layout (Agent-instance-style serialized dicts).
# ---------------------------------------------------------------------------
def test_agent_prompt_helpers_support_action_level_templates():
    from evoagentx.optimizers.engine.adapter import WorkFlowAdapter

    # Agent-instance-style dict: prompts live on actions, and every agent carries a
    # promptless ContextExtraction action that must be skipped.
    agent = {
        "name": "coder",
        "actions": [
            {"class_name": "ContextExtraction"},
            {"class_name": "CodeAction", "prompt_template": {"instruction": "Write code."}},
        ],
    }
    assert WorkFlowAdapter._agent_prompt_text(agent) == "Write code."
    WorkFlowAdapter._set_agent_prompt_text(agent, "Write better code.")
    assert agent["actions"][1]["prompt_template"]["instruction"] == "Write better code."

    # Raw-prompt action variant.
    agent_raw = {"name": "coder", "actions": [{"class_name": "ContextExtraction"}, {"prompt": "Do it."}]}
    assert WorkFlowAdapter._agent_prompt_text(agent_raw) == "Do it."
    WorkFlowAdapter._set_agent_prompt_text(agent_raw, "Do it well.")
    assert agent_raw["actions"][1]["prompt"] == "Do it well."

    # Agents whose actions carry no prompt expose nothing to optimize.
    assert WorkFlowAdapter._agent_prompt_text({"name": "x", "actions": [{"class_name": "ContextExtraction"}]}) is None

    # CustomizeAgent-style (top-level prompt) is unaffected.
    assert WorkFlowAdapter._agent_prompt_text({"name": "y", "prompt": "Top-level."}) == "Top-level."

    # Multiple prompt-bearing actions are ambiguous (which prompt does the workflow run?)
    # and rejected outright, matching legacy MIPRO's fail-fast behavior.
    ambiguous = {
        "name": "coder",
        "actions": [{"prompt": "First."}, {"prompt_template": {"instruction": "Second."}}],
    }
    with pytest.raises(ValueError, match="prompt-bearing actions"):
        WorkFlowAdapter._agent_prompt_text(ambiguous)


# ---------------------------------------------------------------------------
# MiproWorkFlowAdapter over a real (unexecuted) workflow graph.
# ---------------------------------------------------------------------------
@pytest.fixture
def workflow_adapter():
    from evoagentx.models import OpenAILLM, OpenAILLMConfig
    from evoagentx.prompts import ChatTemplate
    from evoagentx.workflow import SequentialWorkFlowGraph

    graph_data = {
        "goal": "Generate correct Python code for the given problem.",
        "tasks": [
            {
                "name": "code_generation",
                "description": "Generate Python code that solves the problem.",
                "inputs": [{"name": "question", "type": "str", "required": True, "description": "The problem."}],
                "outputs": [{"name": "code", "type": "str", "required": True, "description": "The code."}],
                "prompt_template": ChatTemplate(instruction="Write code for the problem."),
                "parse_mode": "str",
            },
        ],
    }
    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="fake-key-for-tests")
    llm = OpenAILLM(config=llm_config)
    graph = SequentialWorkFlowGraph.from_dict(graph_data)
    return MiproWorkFlowAdapter(graph=graph, llm=llm, llm_config=llm_config)


def test_workflow_adapter_units_and_snapshot(workflow_adapter):
    unit_uids = {unit.uid for unit in workflow_adapter.units}
    prompt_uids = {u.uid for u in workflow_adapter.units if u.unit_type == OptimizationUnitType.PROMPT}
    demo_units = [u for u in workflow_adapter.units if (u.metadata or {}).get("kind") == "demos"]

    assert len(prompt_uids) == 1
    assert len(demo_units) == 1
    (prompt_uid,) = prompt_uids
    demos_unit = demo_units[0]
    assert demos_unit.uid == f"{prompt_uid}{DEMOS_UID_SUFFIX}"
    assert demos_unit.metadata["input_names"] == ["question"]
    assert demos_unit.metadata["output_names"] == ["code"]
    assert unit_uids == {prompt_uid, demos_unit.uid}

    snapshot = workflow_adapter.take_snapshot()
    assert snapshot.unit_values[prompt_uid] == "Write code for the problem."
    assert snapshot.unit_values[demos_unit.uid] == []


def test_workflow_adapter_apply_injects_instruction_and_demos(workflow_adapter):
    prompt_unit = next(u for u in workflow_adapter.units if u.unit_type == OptimizationUnitType.PROMPT)
    demos_unit = next(u for u in workflow_adapter.units if (u.metadata or {}).get("kind") == "demos")
    demos = [{"question": "add two numbers", "code": "def add(a, b): return a + b"}]

    snapshot = workflow_adapter.take_snapshot()
    changes = [
        UnitChange.create(unit=prompt_unit, new_value="You are an expert Python engineer.", operation=ChangeOperation.REPLACE),
        UnitChange.create(unit=demos_unit, new_value=demos, operation=ChangeOperation.REPLACE),
    ]
    result = workflow_adapter.apply(snapshot, changes)
    assert result.ok, result.error

    new_adapter = result.adapter
    assert isinstance(new_adapter, MiproWorkFlowAdapter)
    assert new_adapter.prompts[prompt_unit.uid] == "You are an expert Python engineer."
    assert new_adapter.demos[prompt_unit.uid] == demos

    # The rebuilt graph's template actually carries the new instruction + demonstrations.
    node = new_adapter.graph.nodes[0]
    agent = node.agents[0]
    template = agent["prompt_template"]
    instruction = template["instruction"] if isinstance(template, dict) else template.instruction
    template_demos = template["demonstrations"] if isinstance(template, dict) else template.demonstrations
    assert instruction == "You are an expert Python engineer."
    assert template_demos == demos

    # Round-trip: the new adapter's snapshot reproduces the applied values.
    new_snapshot = new_adapter.take_snapshot()
    assert new_snapshot.unit_values[prompt_unit.uid] == "You are an expert Python engineer."
    assert new_snapshot.unit_values[demos_unit.uid] == demos

    # And the original adapter was not mutated.
    assert workflow_adapter.prompts[prompt_unit.uid] == "Write code for the problem."
    assert workflow_adapter.demos[prompt_unit.uid] == []
