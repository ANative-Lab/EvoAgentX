import unittest

from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.optimizers.engine.base import ChangeOperation, UnitChange
from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.textgrad_optimizer import (
    INSTRUCTION_KIND,
    SYSTEM_PROMPT_KIND,
    TextGradOptimizer,
    TextGradWorkFlowAdapter,
)
from evoagentx.prompts import StringTemplate
from evoagentx.prompts.utils import DEFAULT_SYSTEM_PROMPT
from evoagentx.workflow import SequentialWorkFlowGraph


class DummyBenchmark(Benchmark):

    def __init__(self):
        super().__init__(name="dummy", path="")

    def _load_data(self):
        self._train_data = [{"id": i, "problem": f"problem {i}", "answer": f"answer {i}"} for i in range(4)]
        self._dev_data = []
        self._test_data = []

    def _get_id(self, example):
        return example["id"]

    def _get_label(self, example):
        return example["answer"]


def build_graph():
    graph_data = {
        "goal": "Answer the question.",
        "tasks": [
            {
                "name": "planner",
                "description": "Plan how to solve the problem.",
                "inputs": [
                    {"name": "problem", "type": "str", "required": True, "description": "The problem."}
                ],
                "outputs": [
                    {"name": "plan", "type": "str", "required": True, "description": "The plan."}
                ],
                "prompt_template": StringTemplate(instruction="Make a plan."),
                "system_prompt": "You are a planner.",
                "parse_mode": "str",
            },
            {
                "name": "solver",
                "description": "Solve the problem following the plan.",
                "inputs": [
                    {"name": "plan", "type": "str", "required": True, "description": "The plan."}
                ],
                "outputs": [
                    {"name": "answer", "type": "str", "required": True, "description": "The answer."}
                ],
                "prompt_template": StringTemplate(instruction="Solve it."),
                "parse_mode": "str",
            },
        ],
    }
    return SequentialWorkFlowGraph.from_dict(graph_data)


class TestTextGradWorkFlowAdapter(unittest.TestCase):

    def setUp(self):
        self.llm = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX"))
        self.adapter = TextGradWorkFlowAdapter(graph=build_graph(), llm=self.llm)

    def test_register_units(self):
        units = self.adapter.units
        self.assertEqual(len(units), 2 * len(self.adapter.prompts))
        kinds = {unit.uid: unit.metadata["kind"] for unit in units}
        for name in self.adapter.prompts:
            self.assertEqual(kinds[f"{name}::system_prompt"], SYSTEM_PROMPT_KIND)
            self.assertEqual(kinds[f"{name}::instruction"], INSTRUCTION_KIND)

    def test_take_snapshot(self):
        snapshot = self.adapter.take_snapshot()
        instructions = {
            name: snapshot.unit_values[f"{name}::instruction"] for name in self.adapter.prompts
        }
        self.assertEqual(instructions, self.adapter.prompts)
        system_prompt_values = sorted(
            snapshot.unit_values[f"{name}::system_prompt"] for name in self.adapter.prompts
        )
        # One agent declares its own system prompt; the other falls back to the default.
        self.assertIn("You are a planner.", system_prompt_values)
        self.assertIn(DEFAULT_SYSTEM_PROMPT, system_prompt_values)

    def test_apply_round_trip(self):
        snapshot = self.adapter.take_snapshot()
        names = list(self.adapter.prompts)
        changes = [
            UnitChange(
                uid=f"{names[0]}::system_prompt",
                value="You are an expert planner.",
                operation=ChangeOperation.REPLACE,
            ),
            UnitChange(
                uid=f"{names[1]}::instruction",
                value="Solve the problem step by step.",
                operation=ChangeOperation.REPLACE,
            ),
        ]
        result = self.adapter.apply(snapshot, changes)
        self.assertTrue(result.ok, result.error)
        new_adapter = result.adapter
        self.assertIsInstance(new_adapter, TextGradWorkFlowAdapter)
        self.assertEqual(new_adapter.system_prompts[names[0]], "You are an expert planner.")
        self.assertEqual(new_adapter.prompts[names[1]], "Solve the problem step by step.")
        # untouched units keep their values
        self.assertEqual(new_adapter.prompts[names[0]], self.adapter.prompts[names[0]])
        self.assertEqual(new_adapter.system_prompts[names[1]], self.adapter.system_prompts[names[1]])
        # the original adapter is unchanged
        self.assertEqual(self.adapter.system_prompts[names[0]], "You are a planner.")
        # the rebuilt graph carries the new prompts
        rebuilt_agent = new_adapter.graph.nodes[0].agents[0]
        self.assertEqual(rebuilt_agent["system_prompt"], "You are an expert planner.")

    def test_rejects_full_agent_config_with_actions(self):
        # The adapter only supports CustomizeAgent-style dicts (top-level prompt_template);
        # full Agent configs carrying an 'actions' list must be rejected at validation.
        from types import SimpleNamespace
        node = SimpleNamespace(agents=[{
            "name": "full_agent",
            "actions": [{"class_name": "SomeAction", "prompt_template": {"instruction": "do it"}}],
        }])
        graph = SimpleNamespace(nodes=[node])
        with self.assertRaises(ValueError) as ctx:
            TextGradWorkFlowAdapter._validate_textgrad_compatibility(graph)
        self.assertIn("actions", str(ctx.exception))

    def test_rejects_agent_without_prompt_template(self):
        from types import SimpleNamespace
        node = SimpleNamespace(agents=[{"name": "bare_agent", "prompt": "just a prompt"}])
        graph = SimpleNamespace(nodes=[node])
        with self.assertRaises(ValueError):
            TextGradWorkFlowAdapter._validate_textgrad_compatibility(graph)

    def test_snapshot_round_trip_through_from_snapshot(self):
        snapshot = self.adapter.take_snapshot()
        restored = self.adapter.from_snapshot(snapshot)
        self.assertEqual(restored.take_snapshot().unit_values, snapshot.unit_values)


class TestTextGradOptimizer(unittest.TestCase):

    def setUp(self):
        self.llm = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX"))
        self.adapter = TextGradWorkFlowAdapter(graph=build_graph(), llm=self.llm)
        self.benchmark = DummyBenchmark()

    def make_optimizer(self, **kwargs):
        return TextGradOptimizer(
            adapter=self.adapter,
            optimizer_llm=self.llm,
            benchmark=self.benchmark,
            collate_func=lambda example: {"problem": example["problem"]},
            **kwargs,
        )

    def test_optimize_mode_filters_target_units(self):
        optimizer = self.make_optimizer(optimize_mode="system_prompt")
        self.assertTrue(optimizer.target_units)
        self.assertTrue(all(unit.metadata["kind"] == SYSTEM_PROMPT_KIND for unit in optimizer.target_units))

        optimizer = self.make_optimizer(optimize_mode="instruction")
        self.assertTrue(all(unit.metadata["kind"] == INSTRUCTION_KIND for unit in optimizer.target_units))

        optimizer = self.make_optimizer(optimize_mode="all")
        self.assertEqual(len(optimizer.target_units), len(self.adapter.units))

    def test_invalid_optimize_mode(self):
        with self.assertRaises(ValueError):
            self.make_optimizer(optimize_mode="everything")

    def test_requires_train_data(self):
        benchmark = DummyBenchmark()
        benchmark._train_data = []
        with self.assertRaises(ValueError):
            TextGradOptimizer(
                adapter=self.adapter,
                optimizer_llm=self.llm,
                benchmark=benchmark,
                collate_func=lambda example: {"problem": example["problem"]},
            )

    def test_propose_builds_changes_from_gradient_step(self):
        optimizer = self.make_optimizer(optimize_mode="instruction", batch_size=2, seed=0)
        state = optimizer._init_run_state()
        baseline = state.snapshots[0]

        recorded = {}

        def fake_gradient_step(adapter, inputs, labels):
            recorded["inputs"] = inputs
            recorded["labels"] = labels
            # rewrite every target instruction
            return {
                unit.uid: f"IMPROVED: {baseline.unit_values[unit.uid]}"
                for unit in optimizer.target_units
            }

        optimizer._gradient_step = fake_gradient_step
        proposal = optimizer.propose(state, ScalarObjective(metric="score", direction="maximize"))

        self.assertEqual(proposal.source_snapshot_id, baseline.snapshot_id)
        self.assertEqual(len(recorded["inputs"]), 2)
        self.assertEqual(len(recorded["labels"]), 2)
        self.assertEqual(len(proposal.changes), len(optimizer.target_units))
        for change in proposal.changes:
            self.assertTrue(change.uid.endswith("::instruction"))
            self.assertTrue(change.value.startswith("IMPROVED: "))
            self.assertEqual(change.old_value, baseline.unit_values[change.uid])
        # applying the proposal on the baseline yields a consistent new adapter
        result = self.adapter.apply(baseline, proposal.changes)
        self.assertTrue(result.ok, result.error)

    def test_create_textgrad_agents(self):
        optimizer = self.make_optimizer(optimize_mode="system_prompt")
        snapshot = self.adapter.take_snapshot()
        trial_adapter = self.adapter.load_snapshot(snapshot)
        tg_agents, variables_by_uid = optimizer._create_textgrad_agents(trial_adapter)

        self.assertEqual(len(tg_agents), len(trial_adapter.graph.nodes))
        # only the targeted (system prompt) variables become optimization parameters
        self.assertEqual(set(variables_by_uid), set(optimizer.target_units_by_uid))
        for uid, variable in variables_by_uid.items():
            self.assertEqual(variable.value, snapshot.unit_values[uid])
            self.assertTrue(variable.requires_grad)
        # instruction variables exist on the agents but do not require grad
        for tg_agent in tg_agents.values():
            self.assertFalse(tg_agent.instruction.requires_grad)
            self.assertTrue(tg_agent.system_prompt.requires_grad)

    def test_data_cursor_persists_through_optimizer_state(self):
        # 4 train examples, batch_size 3 -> after one batch the cursor sits at epoch 0
        # offset 3; after two batches the epoch rolls over.
        optimizer = self.make_optimizer(batch_size=3, seed=0)
        state = optimizer._init_run_state()

        first_batch, _ = optimizer._next_batch()
        self.assertEqual(len(first_batch), 3)
        state.optimizer_state = optimizer.serialize_optimizer_state(state)
        self.assertEqual(state.optimizer_state["data_cursor"], {"epoch": 0, "offset": 3})

        # a fresh optimizer resumed from that state continues with the tail of the epoch
        resumed = self.make_optimizer(batch_size=3, seed=0)
        resumed.load_optimizer_state(state)
        resumed_batch, _ = resumed._next_batch()
        continued_batch, _ = optimizer._next_batch()
        self.assertEqual(resumed_batch, continued_batch)
        self.assertEqual(len(resumed_batch), 1)
        # both rolled over into epoch 1
        self.assertEqual(
            resumed.serialize_optimizer_state(state)["data_cursor"],
            {"epoch": 1, "offset": 0},
        )

    def test_propose_no_change_yields_empty_proposal(self):
        optimizer = self.make_optimizer(batch_size=1)
        state = optimizer._init_run_state()
        optimizer._gradient_step = lambda adapter, inputs, labels: dict(state.snapshots[0].unit_values)
        proposal = optimizer.propose(state, ScalarObjective(metric="score", direction="maximize"))
        self.assertEqual(proposal.changes, [])


if __name__ == "__main__":
    unittest.main()
