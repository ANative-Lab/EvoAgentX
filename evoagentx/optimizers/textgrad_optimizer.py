# -----------------------------------------------------------------------------
# This file re-implements algorithms from the TextGrad project:
#   Repo: https://github.com/zou-group/textgrad
#   Paper: "TextGrad: Automatic 'Differentiation' via Text -- using large language models to backpropagate textual gradients."
#   Authors: Zou et al.
#
# Re-implementation integrated into EvoAgentX with permission from the authors.
# All mistakes or modifications are our own.
#
# Code of Conduct: This project follows the Microsoft Open Source Code of Conduct.
#   https://opensource.microsoft.com/codeofconduct/
# -----------------------------------------------------------------------------
"""TextGrad optimizer on the new optimization engine.

Migrated from ``evoagentx.optimizers.legacy.textgrad_optimizer``. The TextGrad
algorithm itself (forward pass over the workflow as a textgrad computation
graph, textual loss, backward, textual gradient descent) is preserved; the
surrounding run management is delegated to the engine:

* ``TextGradWorkFlowAdapter`` exposes each workflow agent's system prompt and
  instruction as two ``PROMPT`` optimization units, and knows how to snapshot /
  rebuild the workflow graph from unit values.
* ``TextGradOptimizer.propose`` performs one (or more) TextGrad gradient steps
  on a training batch and emits the updated prompts as a single
  ``OptimizationProposal``; the engine applies it, evaluates it with the
  user-provided ``evaluate_fn``, tracks the best snapshot, checkpoints, and
  resumes. The legacy ``rollback`` behavior maps to seeding each step from the
  engine-tracked best snapshot; legacy ``eval_every_n_steps`` maps to
  ``gradient_steps_per_trial`` (gradient steps folded into one evaluated trial).
"""

# ruff: noqa: E402
import asyncio
import os
import shutil
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

# Check if logs folder exists before importing textgrad
log_folder_exists = os.path.exists("./logs")

import textgrad as tg
from textgrad import EngineLM, Variable
from textgrad import logger as tg_logger
from textgrad import sh as tg_file_handler
from textgrad.autograd import StringBasedFunction
from textgrad.loss import MultiFieldEvaluation, TextLoss
from textgrad.optimizer import TextualGradientDescent

from ..agents import Agent, CustomizeAgent
from ..benchmark.benchmark import Benchmark
from ..benchmark.scorers import CodeScorer
from ..core.callbacks import suppress_logger_info
from ..core.logging import logger
from ..models.base_model import BaseLLM
from ..prompts.optimizers.textgrad_optimizer import (
    CODE_LOSS_PROMPT,
    CODE_REVIEW_EXAMPLE,
    FITNESS_COACH_EXAMPLE,
    GENERAL_LOSS_PROMPT,
    NO_ANSWER_LOSS_PROMPT,
    OPTIMIZER_CONSTRAINTS,
    OPTIMIZER_SYSTEM_PROMPT,
    PERSONAL_FINANCE_ADVISOR_EXAMPLE,
)
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode
from .engine.adapter import SnapShot
from .engine.base import (
    ChangeOperation,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    UnitChange,
)
from .engine.objective import Objective
from .engine.optimizer import OptimizationRunState, Optimizer
from .engine.utils import get_best_snapshot
from .sew_optimizer import SEWWorkFlowAdapter

tg_logger.removeHandler(tg_file_handler)
# remove the logs folder created by textgrad
if not log_folder_exists and os.path.exists("./logs"):
    shutil.rmtree("./logs")


SYSTEM_PROMPT_KIND = "system_prompt"
INSTRUCTION_KIND = "instruction"


def _system_prompt_uid(agent_name: str) -> str:
    return f"{agent_name}::{SYSTEM_PROMPT_KIND}"


def _instruction_uid(agent_name: str) -> str:
    return f"{agent_name}::{INSTRUCTION_KIND}"


class TextGradEngine(EngineLM):
    """Adapts an EvoAgentX ``BaseLLM`` to textgrad's ``EngineLM`` interface."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt: str, system_prompt: str = None, **kwargs):
        with suppress_logger_info():
            response = self.llm.generate(prompt, system_prompt=system_prompt, **kwargs)
            return response.content

    def __call__(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)


class CustomAgentCall:
    """A custom agent call with textgrad.Variable inputs and output."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.last_outputs: dict[str, str] = dict()

    def __call__(
        self,
        instruction: Variable,
        system_prompt: Variable,
        **inputs: Variable,
    ) -> Variable:

        # The single user-defined action (get_all_actions filters out ContextExtraction).
        action = self.agent.get_all_actions()[0]
        input_names = action.inputs_format.get_attrs()

        agent_inputs = {}

        for key, input_variable in inputs.items():
            if key in input_names:
                agent_inputs[key] = input_variable.value
            else:
                parsed_inputs: dict[str, str] = {k: v for k, v in input_variable.parsed_outputs.items() if k in input_names}
                agent_inputs.update(parsed_inputs)

        with suppress_logger_info():
            outputs = self.agent.execute(action_name=action.name, action_input_data=agent_inputs).content

        self.last_outputs = outputs.to_dict()

        return outputs.content


class TextGradAgent:
    """An agent that takes textgrad.Variable inputs and returns a textgrad.Variable response.
    This class wraps an EvoAgentX Agent to allow TextGrad optimization of its prompts.
    """

    def __init__(
        self,
        agent: Agent,
        system_prompt_requires_grad: bool = True,
        instruction_requires_grad: bool = True,
    ):
        self.name = agent.name
        # The single user-defined action (get_all_actions filters out ContextExtraction).
        action = agent.get_all_actions()[0]

        self.system_prompt = Variable(
            agent.system_prompt,
            requires_grad=system_prompt_requires_grad,
            role_description=f"{self.name}'s system prompt",
        )

        self.instruction = Variable(
            action.prompt_template.instruction,
            requires_grad=instruction_requires_grad,
            role_description=f"{self.name}'s instruction prompt",
        )

        self._agent_call = CustomAgentCall(agent)
        self.forward = StringBasedFunction(self._agent_call, agent.description)
        self.output_description = " and ".join(action.outputs_format.get_attr_descriptions().values())
        self.last_output = None

    def __call__(self, inputs: dict[str, Variable]) -> Variable:
        """Given textgrad.Variable inputs, generates a textgrad.Variable output."""

        forward_inputs: dict[str, Variable] = {
            "instruction": self.instruction,
            "system_prompt": self.system_prompt,
            **inputs,
        }

        output_variable = self.forward(forward_inputs, self.output_description)
        output_variable.parsed_outputs = self._agent_call.last_outputs
        self.last_output = output_variable
        return output_variable


# ---------------------------------------------------------------------------
# ProgramAdapter paired with TextGradOptimizer.
# ---------------------------------------------------------------------------
class TextGradWorkFlowAdapter(SEWWorkFlowAdapter):
    """
    A workflow adapter that exposes, for every agent in the graph, TWO ``PROMPT``
    optimization units: the agent's system prompt (``<agent>::system_prompt``) and its
    prompt-template instruction (``<agent>::instruction``). This matches what TextGrad
    optimizes; :class:`SEWWorkFlowAdapter` only exposes the instruction text.

    Compatibility requirements (validated at construction): every node holds exactly
    one agent, defined as a CustomizeAgent-style dict with a top-level
    ``prompt_template`` — the forward pass reads/writes ``prompt_template.instruction``
    and ``system_prompt``. Full ``Agent`` configs carrying an ``"actions"`` list are
    NOT supported (the legacy optimizer accepted them; use it for such workflows).

    Execution (``execute`` / ``async_execute``), graph copying, and prompt injection
    are reused from :class:`SEWWorkFlowAdapter`; snapshot / reconstruct are overridden
    to carry both unit kinds.
    """

    def __init__(
        self,
        graph: "WorkFlowGraph",
        tools: Optional[List[Any]] = None,
        llm: Optional[Any] = None,
        llm_config: Optional[Any] = None,
        max_execution_steps: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            graph=graph,
            tools=tools,
            llm=llm,
            llm_config=llm_config,
            max_execution_steps=max_execution_steps,
            **kwargs,
        )
        self._validate_textgrad_compatibility(self.graph)
        self.system_prompts: Dict[str, str] = self._extract_system_prompts(self.graph)

    # -- graph compatibility --------------------------------------------------
    @staticmethod
    def _validate_textgrad_compatibility(graph: "WorkFlowGraph") -> None:
        """Checks if the graph is compatible with TextGrad optimization.

        Unlike the legacy optimizer, this adapter deliberately supports only
        CustomizeAgent-style agent dicts (a top-level ``prompt_template``): the prompt
        extraction and write-back inherited from :class:`SEWWorkFlowAdapter` operate on
        the agent dict's top level, so full ``Agent`` configs carrying an ``"actions"``
        list are rejected here instead of being silently mis-handled.
        """
        for node in graph.nodes:
            if len(node.agents) > 1:
                raise ValueError("TextGrad optimization only supports workflows where every node has a single agent.")
            agent = node.agents[0]
            if not isinstance(agent, dict):
                raise ValueError(f"Unsupported agent type {type(agent)}. Expected 'dict'.")
            if "actions" in agent:
                raise ValueError(
                    f"Agent {agent.get('name')!r} carries a full Agent config with an 'actions' list, "
                    "which TextGradWorkFlowAdapter does not support: it only optimizes "
                    "CustomizeAgent-style agent dicts with a top-level 'prompt_template' "
                    "(e.g. workflows built via SequentialWorkFlowGraph). Use the legacy "
                    "TextGradOptimizer (evoagentx.optimizers.legacy.textgrad_optimizer) for such workflows."
                )
            if agent.get("prompt_template") is None:
                raise ValueError(f"Please provide a PromptTemplate for {agent['name']}.")

    @staticmethod
    def _extract_system_prompts(graph: "WorkFlowGraph") -> Dict[str, str]:
        """Build ``{agent_name: system_prompt}``; agents without one get the default
        that ``CustomizeAgent`` would apply at execution time."""
        system_prompts: Dict[str, str] = {}
        for node in graph.nodes:
            agent = node.agents[0]
            name = agent.get("name")
            if not name:
                continue
            system_prompt = agent.get("system_prompt")
            system_prompts[name] = (
                system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
            )
        return system_prompts

    # -- declare optimizable units -------------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        units: List[OptimizationUnit] = []
        for name in self.prompts:
            metadata_base = {"agent": name}
            if name in self.roles:
                metadata_base["role"] = self.roles[name]
            units.append(
                OptimizationUnit(
                    name=_system_prompt_uid(name),
                    uid=_system_prompt_uid(name),
                    unit_type=OptimizationUnitType.PROMPT,
                    json_schema={"type": "string", "description": f"System prompt of agent '{name}'."},
                    allowed_operations=[ChangeOperation.REPLACE],
                    metadata={**metadata_base, "kind": SYSTEM_PROMPT_KIND},
                )
            )
            units.append(
                OptimizationUnit(
                    name=_instruction_uid(name),
                    uid=_instruction_uid(name),
                    unit_type=OptimizationUnitType.PROMPT,
                    json_schema={"type": "string", "description": f"Instruction prompt of agent '{name}'."},
                    allowed_operations=[ChangeOperation.REPLACE],
                    metadata={**metadata_base, "kind": INSTRUCTION_KIND},
                )
            )
        return units

    # -- snapshot / reconstruct ----------------------------------------------
    def take_snapshot(self) -> SnapShot:
        unit_values: Dict[str, str] = {}
        for name in self.prompts:
            unit_values[_system_prompt_uid(name)] = self.system_prompts[name]
            unit_values[_instruction_uid(name)] = self.prompts[name]
        return SnapShot(unit_values=unit_values)

    # merge_changes is inherited from SEWProgramAdapter: REPLACE-only dict update on
    # unit_values, which is correct for both unit kinds here.

    def _split_unit_values(self, unit_values: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Split ``{uid: value}`` into ``({agent: instruction}, {agent: system_prompt})``."""
        instructions: Dict[str, str] = {}
        system_prompts: Dict[str, str] = {}
        for unit in self.units:
            value = unit_values[unit.uid]
            agent_name = unit.metadata["agent"]
            if unit.metadata["kind"] == SYSTEM_PROMPT_KIND:
                system_prompts[agent_name] = value
            else:
                instructions[agent_name] = value
        return instructions, system_prompts

    def _rebuild_graph(self, prompts: Dict[str, str], system_prompts: Optional[Dict[str, str]] = None) -> "WorkFlowGraph":
        """Deep-copy the base graph and inject the instructions and system prompts."""
        new_graph = super()._rebuild_graph(prompts)
        if system_prompts:
            for node in new_graph.nodes:
                agent = node.agents[0]
                name = agent.get("name")
                if name in system_prompts:
                    agent["system_prompt"] = system_prompts[name]
        return new_graph

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "TextGradWorkFlowAdapter":
        instructions, system_prompts = self._split_unit_values(snapshot.unit_values)
        new_graph = self._rebuild_graph(instructions, system_prompts)
        return type(self)(
            graph=new_graph,
            tools=self.tools,
            llm=self.llm,
            llm_config=self.llm_config,
            max_execution_steps=self.max_execution_steps,
        )


# ---------------------------------------------------------------------------
# Optimizer: textual gradient descent over the workflow's prompts.
# ---------------------------------------------------------------------------
class TextGradOptimizer(Optimizer):
    """Uses TextGrad to optimize agents' system prompts and instructions in a multi-agent workflow.
    For more information on TextGrad, see https://github.com/zou-group/textgrad.

    Each engine trial performs ``gradient_steps_per_trial`` TextGrad steps: run the
    workflow on a training batch as a textgrad computation graph, compute a textual
    loss (optionally against the batch labels), backpropagate textual gradients, and
    apply textual gradient descent to the target prompts. The updated prompts are
    emitted as one ``OptimizationProposal``, which the engine applies and evaluates
    with the ``evaluate_fn`` passed to ``optimize`` / ``async_optimize``.

    Args:
        adapter: The workflow to optimize, wrapped in a :class:`TextGradWorkFlowAdapter`.
            The adapter's ``llm`` is used to execute the workflow during the forward pass
            (the legacy ``executor_llm``).
        optimizer_llm: The LLM driving TextGrad itself (loss evaluation, gradient
            computation, and prompt updates).
        benchmark: Provides the training batches (``get_train_data`` / ``get_labels``)
            used for the forward/backward passes. For coding benchmarks (a ``CodeScorer``
            scorer), labels are enriched with the generated code's unit-test result.
        collate_func: Maps one benchmark training example to the workflow's input dict
            (e.g. ``lambda example: {"question": example["prompt"]}``).
        optimize_mode: Which prompt kinds to optimize: ``"all"``, ``"system_prompt"``,
            or ``"instruction"``. Sugar for ``target_unit_uids`` filtering.
        batch_size: The number of training examples per gradient step.
        use_answers: Whether to use the labels of the training set in the textual loss.
            If False, the training set does not need labels.
        gradient_steps_per_trial: How many TextGrad steps to fold into one evaluated
            trial (the legacy ``eval_every_n_steps``). The engine evaluates once per
            trial, after all steps have been applied.
        rollback: When True (default), each trial starts from the engine-tracked best
            snapshot, so a step that degraded the metrics is discarded (the legacy
            ``rollback=True``). When False, each trial continues from the most recent
            snapshot — the pure gradient-descent trajectory.
        constraints: Extra natural-language constraints for the TextGrad optimizer,
            appended to the built-in ``OPTIMIZER_CONSTRAINTS``.
        seed: Optional random seed used to shuffle the training data each epoch.

    Note:
        The training-batch cursor (epoch and in-epoch offset) is checkpointed through the
        engine's ``optimizer_state``, so a resumed run continues from the same position in
        the training stream. With a ``seed`` the per-epoch shuffle is deterministic and the
        resumed run sees exactly the batches the interrupted run would have; with
        ``seed=None`` the epoch is re-shuffled on resume and only the position is preserved.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset(
        {OptimizationUnitType.PROMPT}
    )

    def __init__(
        self,
        adapter: TextGradWorkFlowAdapter,
        optimizer_llm: BaseLLM,
        benchmark: Benchmark,
        collate_func: Callable[[dict], dict],
        optimize_mode: Literal["all", "system_prompt", "instruction"] = "all",
        batch_size: int = 1,
        use_answers: bool = True,
        gradient_steps_per_trial: int = 1,
        rollback: bool = True,
        constraints: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not isinstance(adapter, TextGradWorkFlowAdapter):
            raise TypeError(
                f"`adapter` must be a TextGradWorkFlowAdapter, got {type(adapter).__name__}."
            )
        if optimize_mode not in ("all", "system_prompt", "instruction"):
            raise ValueError(
                f"optimize_mode must be one of 'all', 'system_prompt', 'instruction', got {optimize_mode!r}"
            )
        # `optimize_mode` is sugar for target-unit filtering; explicit selectors win.
        if optimize_mode != "all" and kwargs.get("target_unit_uids") is None:
            kwargs["target_unit_uids"] = [
                unit.uid for unit in adapter.units if (unit.metadata or {}).get("kind") == optimize_mode
            ]
        super().__init__(adapter, **kwargs)

        if not isinstance(optimizer_llm, BaseLLM):
            raise TypeError(f"`optimizer_llm` must be a BaseLLM instance, got {type(optimizer_llm).__name__}.")
        if not isinstance(benchmark, Benchmark):
            raise TypeError(f"`benchmark` must be a Benchmark instance, got {type(benchmark).__name__}.")
        if not callable(collate_func):
            raise TypeError("`collate_func` must be a callable mapping a benchmark example to workflow inputs.")
        if not benchmark.get_data_by_mode("train"):
            raise ValueError(
                f"Benchmark {type(benchmark).__name__} has no train data; TextGradOptimizer "
                "draws its forward/backward batches from the train split."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if gradient_steps_per_trial < 1:
            raise ValueError("gradient_steps_per_trial must be >= 1")

        self.benchmark = benchmark
        self.collate_func = collate_func
        self.optimize_mode = optimize_mode
        self.batch_size = batch_size
        self.use_answers = use_answers
        self.gradient_steps_per_trial = gradient_steps_per_trial
        self.rollback = rollback
        # Copy instead of extending the module-level list, so repeated construction
        # does not accumulate constraints globally (a legacy bug).
        self.constraints = list(OPTIMIZER_CONSTRAINTS) + list(constraints or [])
        self.optimizer_engine = TextGradEngine(optimizer_llm)
        self.seed = seed
        # Training-batch cursor; persisted via serialize/load_optimizer_state.
        self._epoch = 0
        self._offset = 0
        self._epoch_data: Optional[List[Any]] = None
        self._loss_fn = self._build_loss_fn()

        # Disable TextGrad's short variable value so the optimizer receives the full value.
        def disable_short_variable_value(self, n_words_offset: int = 10):
            return self.value
        Variable.get_short_value = disable_short_variable_value

    # -- textgrad setup -------------------------------------------------------
    def _is_coding_benchmark(self) -> bool:
        return isinstance(getattr(self.benchmark, "scorer", None), CodeScorer)

    def _build_loss_fn(self):
        if self.use_answers:
            if self._is_coding_benchmark():
                loss_prompt = CODE_LOSS_PROMPT
                role_descriptions = [
                    "code snippet to evaluate",
                    "the task, the test result of the code snippet, and the correct code",
                ]
            else:
                loss_prompt = GENERAL_LOSS_PROMPT
                role_descriptions = ["response to evaluate", "correct answer"]
            evaluation_instruction = Variable(loss_prompt, requires_grad=False, role_description="evaluation instruction")
            return MultiFieldEvaluation(evaluation_instruction, role_descriptions, self.optimizer_engine)

        loss_prompt = NO_ANSWER_LOSS_PROMPT
        evaluation_instruction = Variable(loss_prompt, requires_grad=False, role_description="evaluation instruction")
        return TextLoss(evaluation_instruction, self.optimizer_engine)

    # -- training data --------------------------------------------------------
    # The batch cursor (epoch, offset) is plain optimizer state, persisted through the
    # engine's optimizer_state hooks so a resumed run continues from the same position
    # in the training stream. TextGrad itself is stateless between steps (no momentum),
    # so this cursor is the only optimizer state worth checkpointing.
    def _epoch_train_data(self) -> List[Any]:
        """The train split, shuffled for the current epoch (deterministic when seeded)."""
        train_data = self.benchmark.get_data_by_mode("train")
        effective_seed = self.seed + self._epoch if self.seed is not None else None
        return self.benchmark.get_train_data(sample_k=len(train_data), seed=effective_seed)

    def _next_batch(self) -> Tuple[List[Dict[str, str]], Optional[List[Union[str, Dict[str, str]]]]]:
        if self._epoch_data is None:
            self._epoch_data = self._epoch_train_data()
        batch = self._epoch_data[self._offset:self._offset + self.batch_size]
        self._offset += self.batch_size
        if self._offset >= len(self._epoch_data):
            # epoch exhausted: reshuffle on next access
            self._epoch += 1
            self._offset = 0
            self._epoch_data = None
        inputs = [self.collate_func(example) for example in batch]
        labels = self.benchmark.get_labels(batch) if self.use_answers else None
        return inputs, labels

    def serialize_optimizer_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        serialized = dict(state.optimizer_state or {})
        serialized["data_cursor"] = {"epoch": self._epoch, "offset": self._offset}
        return serialized

    def load_optimizer_state(self, state: OptimizationRunState) -> None:
        cursor = (state.optimizer_state or {}).get("data_cursor")
        if cursor:
            self._epoch = int(cursor.get("epoch", 0))
            self._offset = int(cursor.get("offset", 0))
            self._epoch_data = None

    # -- forward pass over the workflow (ported from the legacy optimizer) ----
    def _create_textgrad_agents(
        self, adapter: TextGradWorkFlowAdapter
    ) -> Tuple[Dict[str, TextGradAgent], Dict[str, Variable]]:
        """Wrap every node's agent as a TextGradAgent over ``adapter``'s graph.

        Returns ``(agents_by_node_name, target_variables_by_uid)``; the latter holds the
        textgrad Variables for exactly the units this optimizer targets, and becomes the
        parameter list of the textual gradient descent step.
        """
        target_uids: Set[str] = set(self.target_units_by_uid)
        tg_agents: Dict[str, TextGradAgent] = {}
        variables_by_uid: Dict[str, Variable] = {}
        for node in adapter.graph.nodes:
            agent_dict = node.agents[0]
            if agent_dict.get("llm") is None and agent_dict.get("llm_config") is None:
                # The adapter's llm executes the workflow (the legacy `executor_llm`).
                # `adapter.graph` is already a per-trial copy, so this mutation is local.
                agent_dict["llm"] = adapter.llm
            # Adapter validation guarantees CustomizeAgent-style dicts (top-level
            # prompt_template, no 'actions' list), so this always builds a CustomizeAgent.
            agent: CustomizeAgent = CustomizeAgent.from_dict(agent_dict)
            agent_name = agent_dict.get("name")
            system_prompt_uid = _system_prompt_uid(agent_name)
            instruction_uid = _instruction_uid(agent_name)
            tg_agent = TextGradAgent(
                agent,
                system_prompt_requires_grad=system_prompt_uid in target_uids,
                instruction_requires_grad=instruction_uid in target_uids,
            )
            tg_agents[node.name] = tg_agent
            if system_prompt_uid in target_uids:
                variables_by_uid[system_prompt_uid] = tg_agent.system_prompt
            if instruction_uid in target_uids:
                variables_by_uid[instruction_uid] = tg_agent.instruction
        return tg_agents, variables_by_uid

    @staticmethod
    def _create_output_lookup(graph: "WorkFlowGraph") -> Dict[str, str]:
        """Creates a lookup table from output names to node names."""
        output_name_to_node_name = {}
        for node in graph.nodes:
            for output in node.outputs:
                output_name_to_node_name[output.name] = node.name
        return output_name_to_node_name

    @staticmethod
    def _initial_inputs_to_variables(graph: "WorkFlowGraph", initial_inputs: Dict[str, str]) -> Dict[str, Variable]:
        """Converts inputs to the initial nodes to textgrad variables."""
        variables = {}
        initial_nodes = graph.find_initial_nodes()
        for initial_node in initial_nodes:
            for key, value in initial_inputs.items():
                for input in graph.get_node(initial_node).inputs:
                    if input.name == key:
                        initial_input_variable = Variable(
                            value,
                            requires_grad=False,
                            role_description=input.description,
                        )
                        variables[key] = initial_input_variable
                        if len(variables) == len(initial_inputs):
                            return variables
        missing_inputs = set(initial_inputs.keys()) - set(variables.keys())
        raise ValueError(f"Initial inputs do not match the inputs of the initial nodes. Missing inputs: {missing_inputs}")

    def _compute_node(
        self,
        graph: "WorkFlowGraph",
        tg_agents: Dict[str, TextGradAgent],
        output_lookup: Dict[str, str],
        node: Union[str, "WorkFlowNode"],
        initial_inputs: Dict[str, Variable],
        visited: Set[str],
    ) -> Variable:
        """Computes the output of a node in the workflow graph by recursively computing the required inputs."""
        if isinstance(node, str):
            node = graph.get_node(node)

        if node.name in visited:
            return tg_agents[node.name].last_output

        input_variables: Dict[str, Variable] = {}   # inputs to TextGradAgent
        input_node_names: Set[str] = set()           # which nodes we need to compute the output of

        for input in node.inputs:
            if input.name in initial_inputs:
                input_variables[input.name] = initial_inputs[input.name]
            else:
                input_node_names.add(output_lookup[input.name])

        # if the input is from another node, compute the output of that node
        for node_name in input_node_names:
            input_variables[node_name] = self._compute_node(
                graph, tg_agents, output_lookup, node_name, initial_inputs, visited
            )

        output_variable = tg_agents[node.name](input_variables)
        visited.add(node.name)
        return output_variable

    def _forward(
        self,
        graph: "WorkFlowGraph",
        tg_agents: Dict[str, TextGradAgent],
        output_lookup: Dict[str, str],
        inputs: Dict[str, str],
    ) -> Variable:
        """Returns the final output from the workflow as a textgrad Variable."""
        end_node = graph.find_end_nodes()[0]
        input_variables = self._initial_inputs_to_variables(graph, inputs)
        return self._compute_node(graph, tg_agents, output_lookup, end_node, input_variables, visited=set())

    # -- loss helpers ----------------------------------------------------------
    def _format_code_label(self, code: str, label: Dict[str, str]) -> str:
        """Formats the label for coding tasks to include the task, the test result, and the correct code.

        Args:
            code: The code to evaluate.
            label: A dictionary with keys "task_id", "test", "entry_point", and "canonical_solution".

        Returns:
            The formatted label which includes the task, the test result, and the correct code.
        """
        task_id = label["task_id"]
        prompt = self.benchmark.get_example_by_id(task_id)["prompt"]
        test = label["test"]
        entry_point = label["entry_point"]

        scorer: CodeScorer = self.benchmark.scorer
        state, message = scorer.check_solution(
            task_id=task_id,
            solution=prompt + scorer.prompt_joiner + code,
            test=test,
            entry_point=entry_point,
            use_entrypoint_as_input=scorer.use_entrypoint_as_input,
        )

        if state != scorer.SUCCESS:
            message = message.replace("Solution", "Failed Code")

        formatted_label = (
            f"## Task:\n{prompt}\n\n## Result on test:\n{message}\n\n"
            f"## Correct Solution:\n{label['canonical_solution']}"
        )
        return formatted_label

    def _make_label_variable(
        self,
        graph: "WorkFlowGraph",
        output: Variable,
        label: Union[str, Dict[str, str]],
    ) -> Variable:
        if isinstance(label, str):
            return Variable(label, requires_grad=False, role_description="correct answer for the query")
        if isinstance(label, dict):
            if not self._is_coding_benchmark():
                raise ValueError("Label must be a string for non-coding benchmarks.")
            end_node = graph.get_node(graph.find_end_nodes()[0])
            output_name = end_node.outputs[0].name
            code = output.parsed_outputs[output_name]
            return Variable(
                self._format_code_label(code, label),
                requires_grad=False,
                role_description="the task, the test result, and the correct code",
            )
        raise ValueError(f"Unsupported label type {type(label).__name__}. Expected 'str' or 'dict'.")

    # -- one TextGrad step ------------------------------------------------------
    def _gradient_step(
        self,
        adapter: TextGradWorkFlowAdapter,
        inputs: List[Dict[str, str]],
        labels: Optional[List[Union[str, Dict[str, str]]]],
    ) -> Dict[str, str]:
        """Run forward / loss / backward / step on one batch; return the updated unit values."""
        graph = adapter.graph
        tg_agents, variables_by_uid = self._create_textgrad_agents(adapter)
        output_lookup = self._create_output_lookup(graph)

        losses = []
        if self.use_answers:
            if labels is None:
                raise ValueError("Labels must be provided if `use_answers` is True.")
            for input, label in zip(inputs, labels, strict=True):
                output = self._forward(graph, tg_agents, output_lookup, input)
                label_variable = self._make_label_variable(graph, output, label)
                losses.append(self._loss_fn([output, label_variable]))
        else:
            for input in inputs:
                output = self._forward(graph, tg_agents, output_lookup, input)
                losses.append(self._loss_fn(output))

        # TextualGradientDescent is stateless across steps (no momentum), so a fresh
        # instance per step over freshly built Variables is equivalent to the legacy loop.
        textgrad_optimizer = TextualGradientDescent(
            parameters=list(variables_by_uid.values()),
            engine=self.optimizer_engine,
            constraints=self.constraints,
            optimizer_system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            in_context_examples=[PERSONAL_FINANCE_ADVISOR_EXAMPLE, FITNESS_COACH_EXAMPLE, CODE_REVIEW_EXAMPLE],
        )
        total_loss = tg.sum(losses)
        total_loss.backward(self.optimizer_engine)
        textgrad_optimizer.step()

        return {uid: variable.value for uid, variable in variables_by_uid.items()}

    # -- proposal generation -----------------------------------------------------
    def _seed_snapshot(self, state: OptimizationRunState) -> SnapShot:
        if self.rollback:
            return get_best_snapshot(state)
        # Pure gradient trajectory: continue from the most recently recorded snapshot
        # (falls back to the baseline before any trial has completed).
        return state.snapshots[-1]

    def propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        **kwargs,
    ) -> OptimizationProposal:
        snapshot = self._seed_snapshot(state)
        current_values: Dict[str, str] = dict(snapshot.unit_values)

        for step in range(self.gradient_steps_per_trial):
            trial_snapshot = SnapShot(unit_values=dict(current_values), program_config=snapshot.program_config)
            trial_adapter = self.adapter.load_snapshot(trial_snapshot)
            inputs, labels = self._next_batch()
            logger.info(
                f"TextGrad gradient step {step + 1}/{self.gradient_steps_per_trial} "
                f"on a batch of {len(inputs)} example(s)..."
            )
            new_values = self._gradient_step(trial_adapter, inputs, labels)
            current_values.update(new_values)

        changes: List[UnitChange] = []
        for unit in self.target_units:
            old_value = snapshot.unit_values[unit.uid]
            new_value = current_values[unit.uid]
            if not new_value or new_value == old_value:
                continue
            changes.append(
                UnitChange.create(
                    unit=unit,
                    new_value=new_value,
                    old_value=old_value,
                    operation=ChangeOperation.REPLACE,
                    metadata={"strategy": "textgrad", "optimize_mode": self.optimize_mode},
                )
            )
        if not changes:
            logger.warning("TextGrad step produced no prompt changes; the trial will re-evaluate the seed prompts.")
        return OptimizationProposal(
            source_snapshot_id=snapshot.snapshot_id,
            changes=changes,
            metadata={
                "source": "TextGradOptimizer",
                "optimize_mode": self.optimize_mode,
                "batch_size": self.batch_size,
                "gradient_steps": self.gradient_steps_per_trial,
                "target_uids": self.target_unit_uids,
            },
        )

    async def async_batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        # The TextGrad forward/backward pass is synchronous (textgrad's API is sync);
        # run it in a worker thread so async_optimize's event loop is not blocked.
        return await asyncio.to_thread(
            self.batch_propose, state, objective, budget_remaining=budget_remaining, **kwargs
        )


__all__ = ["TextGradOptimizer", "TextGradWorkFlowAdapter", "TextGradAgent", "TextGradEngine"]
