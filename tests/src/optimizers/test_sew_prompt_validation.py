import unittest

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow.workflow_graph import SEWWorkFlowGraph
from evoagentx.optimizers.sew_optimizer import SEWOptimizer

# A real-world failure case observed with deepseek-chat: the model echoed the
# refinement meta-instructions verbatim, and the optimizer adopted the echo as
# the new task prompt, corrupting the workflow (HumanEval pass@1 dropped 0.4 -> 0.1).
ECHOED_META_PROMPT = (
    "Given the above information, please refine the instruction for the 1-th task.\n"
    "Note that you should always use bracket (e.g. `{input_name}`) to wrap the inputs "
    "of the tasks in your refined instruction.\n"
    "Only output the refined instruction and DON'T include any other text!"
)


class _StubBreeder:
    """Prompt breeder stub that returns a fixed response."""

    def __init__(self, response: str):
        self.response = response

    def generate_prompt(self, **kwargs) -> str:
        return self.response


class _StubOptimizer:
    """Duck-typed stand-in exposing only what `_wfg_prompt_optimization_step` uses."""

    _validate_refined_prompt = staticmethod(SEWOptimizer._validate_refined_prompt)
    repr_scheme = "python"
    order = "zero-order"

    def __init__(self, breeder: _StubBreeder):
        self._prompt_breeder = breeder


class TestSEWPromptValidation(unittest.TestCase):

    def setUp(self):
        self.model = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini", openai_key="XXX"))
        self.graph = SEWWorkFlowGraph(llm=self.model)

    def test_validator_rejects_echoed_meta_instructions(self):
        self.assertFalse(
            SEWOptimizer._validate_refined_prompt(ECHOED_META_PROMPT, "{question}", ["question"])
        )

    def test_validator_rejects_empty_prompt(self):
        self.assertFalse(SEWOptimizer._validate_refined_prompt("", "{question}", ["question"]))
        self.assertFalse(SEWOptimizer._validate_refined_prompt("   \n", "{question}", ["question"]))

    def test_validator_rejects_dropped_placeholder(self):
        self.assertFalse(
            SEWOptimizer._validate_refined_prompt(
                "Summarize the task in detail.", "{question}", ["question"]
            )
        )

    def test_validator_accepts_legitimate_refinement(self):
        self.assertTrue(
            SEWOptimizer._validate_refined_prompt(
                "Carefully parse the coding question below and summarize it.\n\nQuestion: {question}",
                "{question}",
                ["question"],
            )
        )

    def test_validator_ignores_placeholders_absent_from_original(self):
        # placeholders that were never in the original prompt must not be required
        self.assertTrue(
            SEWOptimizer._validate_refined_prompt("Do the task.", "Do it.", ["question"])
        )

    def test_step_keeps_original_prompts_on_echoed_response(self):
        original_prompts = [t["prompt"] for t in self.graph.get_graph_info()["tasks"]]
        stub = _StubOptimizer(_StubBreeder(ECHOED_META_PROMPT))
        new_graph = SEWOptimizer._wfg_prompt_optimization_step(stub, self.graph)
        new_prompts = [t["prompt"] for t in new_graph.get_graph_info()["tasks"]]
        self.assertEqual(original_prompts, new_prompts)

    def test_step_adopts_valid_refinement(self):
        refined = "Refined instruction: answer {question} using the summary {parsed_task}."
        stub = _StubOptimizer(_StubBreeder(refined))
        new_graph = SEWOptimizer._wfg_prompt_optimization_step(stub, self.graph)
        new_prompts = [t["prompt"] for t in new_graph.get_graph_info()["tasks"]]
        self.assertEqual(new_prompts, [refined] * len(new_prompts))


if __name__ == "__main__":
    unittest.main()
