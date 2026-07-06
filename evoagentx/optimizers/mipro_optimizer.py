# -----------------------------------------------------------------------------
# This file natively re-implements the MIPROv2 algorithm from the DSPy project:
#   Repo: https://github.com/stanfordnlp/dspy
#   Paper: "Optimizing Instructions and Demonstrations for Multi-Stage Language
#           Model Programs" (Opsahl-Ong et al.)
#
# Re-implementation integrated into EvoAgentX with permission from the authors.
# All mistakes or modifications are our own.
# -----------------------------------------------------------------------------
"""
MIPRO(v2)-style optimizer on the new optimization engine.

The algorithm has three phases, mapped onto the engine's hooks:

1. **Bootstrap few-shot demo candidates** (``on_run_start``): the current program is run
   on training examples; runs whose final prediction passes ``metric_fn`` contribute the
   per-agent input/output values observed *during* that run as demo candidates. Several
   candidate sets are built (baseline demos, labeled-only, and bootstrapped sets with
   different shuffles/sizes), mirroring dspy's ``create_n_fewshot_demo_sets``.
2. **Propose instruction candidates** (``on_run_start``): for every prompt unit, an LLM
   proposes new instructions grounded in a dataset summary, a program overview, the
   unit's role, bootstrapped task demos, and a randomly selected prompting tip
   (dspy's ``GroundedProposer``). Candidate 0 is always the original instruction.
3. **Bayesian search** (``batch_propose`` / ``observe``): an optuna TPE sampler picks one
   (instruction index, demo-set index) combination per agent for each trial; the engine
   applies it on top of the *baseline* snapshot, evaluates it with the user's
   ``evaluate_fn``, and the observed score is fed back to the sampler.

Unlike dspy's MIPROv2 there is no minibatch evaluation: every trial is scored by the
same ``evaluate_fn`` (which itself decides how many dev examples to use), so the
engine's best-snapshot selection is directly trustworthy.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple

import optuna

from ..core.logging import logger
from ..models.model_configs import OpenRouterConfig
from ..models.openrouter_model import OpenRouterLLM
from .engine.adapter import ProgramAdapter, SnapShot, WorkFlowAdapter
from .engine.base import (
    ChangeOperation,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    TrialRecord,
    UnitChange,
)
from .engine.objective import Objective, ScalarObjective
from .engine.optimizer import Optimizer, OptimizationRunState

# Suffix appended to an agent name to form the uid of its few-shot demos unit.
DEMOS_UID_SUFFIX = "__demos"

# When the run is zero-shot (no demo search), a few demos are still bootstrapped purely
# to ground the instruction proposals (dspy: BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT).
DEMOS_FOR_PROPOSAL_ONLY = 3

# Prompting tips randomly injected into instruction proposals (from dspy's GroundedProposer).
PROPOSER_TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...").',
}

_DATASET_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst. You will be shown a small sample of examples from a dataset "
    "used to train and evaluate an AI system. Write a concise 2-3 sentence observation of "
    "the dataset: the kind of task it captures, the style/format of its inputs and outputs, "
    "and anything notable a prompt engineer should know. Respond with ONLY the summary text."
)

_INSTRUCTION_PROPOSAL_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. Use the information below to learn about a task "
    "that an AI program is trying to solve, then write a new instruction that will be used "
    "to prompt a language model to better solve the task. Preserve the original intent and "
    "any required input/output format — including placeholder tokens such as {variable}. "
    "Respond with ONLY the new instruction text — no explanations, no commentary, no "
    "surrounding quotes or markdown fences."
)


def _jsonify_value(value: Any) -> Any:
    """Coerce an execution-data value into a JSON-safe scalar for demo storage."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _render_demo(demo: Dict[str, Any]) -> str:
    """Render one demo dict as 'field: value' lines for use inside proposer prompts."""
    return "\n".join(f"{key}: {value}" for key, value in demo.items())


# ---------------------------------------------------------------------------
# WorkFlowGraph-backed adapter paired with MiproOptimizer.
# ---------------------------------------------------------------------------
class MiproWorkFlowAdapter(WorkFlowAdapter):
    """
    A :class:`WorkFlowAdapter` that additionally exposes each agent's few-shot
    demonstrations as an optimizable unit, paired with :class:`MiproOptimizer`.

    Per agent this adapter derives:

    * one ``PROMPT`` unit for its instruction (inherited from ``WorkFlowAdapter``); and
    * one ``FIELD`` unit (uid ``"<agent>__demos"``) holding the agent's few-shot
      demonstrations as a JSON list of ``{field: value}`` dicts — but only for agents
      whose active prompt slot is a ``prompt_template``, since demos are rendered
      through :meth:`PromptTemplate.render_demonstrations` at execution time. Agents
      with only a raw ``prompt`` string keep their instruction unit but get no demos
      unit (a warning is logged).

    The demos unit's metadata records the node's input/output names
    (``metadata["input_names"]`` / ``metadata["output_names"]``): these are the fields a
    demo of that agent may carry, and :meth:`async_execute_with_trace` uses them to slice
    the workflow's execution data into per-agent demos when :class:`MiproOptimizer`
    bootstraps demo candidates.
    """

    def __init__(
        self,
        graph: Any,
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
        self.demos, self.demo_fields = self._extract_demos_and_fields(self.graph)
        for agent_name in self.prompts:
            if agent_name not in self.demos:
                logger.warning(
                    f"Agent {agent_name!r} uses a raw `prompt` string (no prompt_template); "
                    "its few-shot demos cannot be rendered, so only its instruction will be optimized."
                )

    # -- demos extraction -----------------------------------------------------
    @staticmethod
    def _template_demonstrations(template: Any) -> List[Dict[str, Any]]:
        """Read the current demonstrations from a prompt template (dict or PromptTemplate)."""
        demos = template.get("demonstrations") if isinstance(template, dict) else getattr(template, "demonstrations", None)
        if not isinstance(demos, list):
            return []
        return [dict(demo) for demo in demos if isinstance(demo, dict)]

    @staticmethod
    def _set_template_demonstrations(template: Any, demos: List[Dict[str, Any]]) -> None:
        """Write ``demos`` back as the prompt template's ``demonstrations`` (dict or PromptTemplate)."""
        value = [dict(demo) for demo in demos] or None
        if isinstance(template, dict):
            template["demonstrations"] = value
        elif hasattr(template, "set_demonstrations"):
            template.set_demonstrations(value)
        else:
            setattr(template, "demonstrations", value)

    @classmethod
    def _extract_demos_and_fields(
        cls, graph: Any
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, List[str]]]]:
        """Build ``{agent: demos}`` and ``{agent: {input_names, output_names}}`` for template agents."""
        demos: Dict[str, List[Dict[str, Any]]] = {}
        fields: Dict[str, Dict[str, List[str]]] = {}
        for node in graph.nodes:
            for agent_name, agent in cls._iter_prompt_agents(node):
                container = cls._agent_prompt_container(agent)
                if container is None or container.get("prompt_template") is None:
                    continue
                demos[agent_name] = [
                    {key: _jsonify_value(value) for key, value in demo.items()}
                    for demo in cls._template_demonstrations(container["prompt_template"])
                ]
                fields[agent_name] = {
                    "input_names": list(node.get_input_names()),
                    "output_names": list(node.get_output_names()),
                }
        return demos, fields

    # -- declare optimizable units ---------------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        units = super().register_units()
        for agent_name, demos_fields in self.demo_fields.items():
            units.append(
                OptimizationUnit(
                    name=f"{agent_name}{DEMOS_UID_SUFFIX}",
                    uid=f"{agent_name}{DEMOS_UID_SUFFIX}",
                    unit_type=OptimizationUnitType.FIELD,
                    json_schema={
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Few-shot demonstrations as a list of {field: value} dicts.",
                    },
                    allowed_operations=[ChangeOperation.REPLACE],
                    metadata={
                        "kind": "demos",
                        "agent": agent_name,
                        "paired_prompt_uid": agent_name,
                        "input_names": demos_fields["input_names"],
                        "output_names": demos_fields["output_names"],
                    },
                )
            )
        return units

    # -- snapshot / reconstruct --------------------------------------------------
    def take_snapshot(self) -> SnapShot:
        unit_values: Dict[str, Any] = dict(self.prompts)
        for agent_name, demos in self.demos.items():
            unit_values[f"{agent_name}{DEMOS_UID_SUFFIX}"] = [dict(demo) for demo in demos]
        return SnapShot(unit_values=unit_values)

    def _rebuild_graph(self, unit_values: Dict[str, Any]) -> Any:
        new_graph = super()._rebuild_graph(unit_values)
        for node in new_graph.nodes:
            for agent_name, agent in self._iter_prompt_agents(node):
                demos_uid = f"{agent_name}{DEMOS_UID_SUFFIX}"
                container = self._agent_prompt_container(agent)
                if demos_uid in unit_values and container is not None and container.get("prompt_template") is not None:
                    self._set_template_demonstrations(container["prompt_template"], unit_values[demos_uid] or [])
        return new_graph

    def fingerprint(self) -> Dict[str, Any]:
        # The demos units' candidate pools and traced rollouts are only valid for the
        # node input/output names they were built against; include them so a resumed run
        # against a same-shaped graph with changed IO is rejected instead of silently
        # reusing stale demo candidates.
        fingerprint = super().fingerprint()
        fingerprint["demo_fields"] = self.demo_fields
        return fingerprint

    # -- traced execution for demo bootstrapping ---------------------------------
    def _collect_demo_traces(self, workflow: Any) -> Dict[str, Dict[str, Any]]:
        """Slice the workflow's execution data into one demo dict per demos unit."""
        execution_data = workflow.environment.get_all_execution_data()
        traces: Dict[str, Dict[str, Any]] = {}
        for agent_name, demo_fields in self.demo_fields.items():
            demo = {
                name: _jsonify_value(execution_data[name])
                for name in demo_fields["input_names"] + demo_fields["output_names"]
                if name in execution_data
            }
            # A demo without any observed output teaches nothing; skip it.
            if any(name in demo for name in demo_fields["output_names"]):
                traces[f"{agent_name}{DEMOS_UID_SUFFIX}"] = demo
        return traces

    def execute_with_trace(self, inputs: Optional[dict] = None, **kwargs) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
        """Run the workflow and also return per-agent demo dicts observed during the run.

        Returns:
            ``(result, traces)`` where ``traces`` maps each demos-unit uid to the
            ``{field: value}`` dict observed for that agent during this execution.
        """
        from ..core.callbacks import suppress_logger_info

        workflow = self._build_workflow()
        with suppress_logger_info():
            result = workflow.execute(inputs=dict(inputs or {}), **kwargs)
        return result, self._collect_demo_traces(workflow)

    async def async_execute_with_trace(self, inputs: Optional[dict] = None, **kwargs) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
        """Async variant of :meth:`execute_with_trace`."""
        from ..core.callbacks import suppress_logger_info

        workflow = self._build_workflow()
        with suppress_logger_info():
            result = await workflow.async_execute(inputs=dict(inputs or {}), **kwargs)
        return result, self._collect_demo_traces(workflow)


# ---------------------------------------------------------------------------
# Optimizer: bootstrap demos -> propose instructions -> TPE search.
# ---------------------------------------------------------------------------
class MiproOptimizer(Optimizer):
    """
    Native MIPRO(v2) optimizer over an adapter's instruction (``PROMPT``) and few-shot
    demo (``FIELD``, ``metadata["kind"] == "demos"``) units.

    Designed for :class:`MiproWorkFlowAdapter`, but works with any adapter that follows
    the same unit conventions. Demo bootstrapping additionally requires the adapter to
    implement ``execute_with_trace`` / ``async_execute_with_trace`` returning
    ``(result, {demos_uid: demo_dict})``; without it (or without demos units), the
    optimizer degrades gracefully to instruction-only search.

    Args:
        adapter: The program adapter to optimize.
        optimizer_model: OpenRouter model id used for dataset summaries and instruction
            proposals (e.g. ``"anthropic/claude-sonnet-4.6"``). An ``OpenRouterLLM`` is
            built internally; the API key is read from ``openrouter_key`` or
            ``$OPENROUTER_API_KEY``.
        task_description: Description of the overall goal the program serves (typically
            the workflow goal); grounds every instruction proposal.
        trainset: Raw training examples (list of dicts) used to bootstrap demos, sample
            labeled demos, and summarize the dataset.
        input_fn: Maps one raw training example to the ``inputs`` dict expected by
            ``adapter.execute`` / ``async_execute``.
        output_fn: Extracts the final prediction from the adapter's execution result.
        metric_fn: ``metric_fn(prediction, example) -> float | bool`` judging one
            bootstrap rollout. A rollout is kept as a demo source when the returned
            value is truthy, or ``>= metric_threshold`` when a threshold is set.
        metric_threshold: Optional numeric acceptance threshold for ``metric_fn``.
        label_fn: Optional ``label_fn(example) -> {field: value}`` mapping a raw example
            to gold output fields. Labeled demos are built as
            ``{**input_fn(example), **label_fn(example)}`` filtered to each unit's
            declared fields; when omitted, labeled demos fall back to the intersection
            of the raw example's keys with those fields.
        num_instruction_candidates: Instruction candidates per prompt unit (including
            the original instruction at index 0).
        num_demo_candidates: Demo candidate sets per demos unit (including the baseline
            set at index 0).
        max_bootstrapped_demos: Max bootstrapped demos per set (0 disables bootstrapping;
            with ``max_labeled_demos == 0`` this makes the run zero-shot).
        max_labeled_demos: Max labeled demos used to pad each set.
        bootstrap_concurrency: How many bootstrap rollouts to run concurrently
            (``async_optimize`` only; sync bootstrapping is sequential).
        trials_per_batch: Proposals generated per engine batch. Values > 1 let
            ``async_optimize(..., execution_mode="concurrent")`` evaluate several TPE
            suggestions in parallel at some cost to sampler informedness.
        init_temperature: Sampling temperature for instruction proposals.
        use_dataset_summary / use_program_summary / use_tips / use_task_demos: Toggle the
            corresponding grounding sections of the instruction-proposal prompt
            (dspy's data/program/tip/fewshot-aware proposer flags).
        view_data_batch_size: How many training examples the dataset summarizer sees.
        openrouter_key: Optional explicit API key; falls back to ``$OPENROUTER_API_KEY``.
        seed: RNG seed for shuffles, tip selection, and the TPE sampler.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset(
        {OptimizationUnitType.PROMPT, OptimizationUnitType.FIELD}
    )

    def __init__(
        self,
        adapter: ProgramAdapter,
        optimizer_model: str,
        task_description: str,
        trainset: List[dict],
        input_fn: Callable[[dict], dict],
        output_fn: Callable[[Any], Any],
        metric_fn: Callable[[Any, dict], Any],
        metric_threshold: Optional[float] = None,
        label_fn: Optional[Callable[[dict], dict]] = None,
        num_instruction_candidates: int = 6,
        num_demo_candidates: int = 6,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        bootstrap_concurrency: int = 4,
        trials_per_batch: int = 1,
        init_temperature: float = 1.0,
        use_dataset_summary: bool = True,
        use_program_summary: bool = True,
        use_tips: bool = True,
        use_task_demos: bool = True,
        view_data_batch_size: int = 10,
        openrouter_key: Optional[str] = None,
        seed: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if not isinstance(optimizer_model, str) or not optimizer_model:
            raise ValueError("`optimizer_model` must be a non-empty OpenRouter model id string.")
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("`task_description` must be a non-empty string describing the program's goal.")
        if not trainset or not all(isinstance(example, dict) for example in trainset):
            raise ValueError("`trainset` must be a non-empty list of dict examples.")
        if num_instruction_candidates < 1 or num_demo_candidates < 1:
            raise ValueError("num_instruction_candidates and num_demo_candidates must be >= 1.")
        if max_bootstrapped_demos < 0 or max_labeled_demos < 0:
            raise ValueError("max_bootstrapped_demos and max_labeled_demos must be >= 0.")
        if trials_per_batch < 1:
            raise ValueError("trials_per_batch must be >= 1.")

        self.task_description = task_description.strip()
        self.trainset = list(trainset)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.metric_fn = metric_fn
        self.metric_threshold = metric_threshold
        self.label_fn = label_fn
        self.num_instruction_candidates = num_instruction_candidates
        self.num_demo_candidates = num_demo_candidates
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.bootstrap_concurrency = max(1, bootstrap_concurrency)
        self.trials_per_batch = trials_per_batch
        self.init_temperature = init_temperature
        self.use_dataset_summary = use_dataset_summary
        self.use_program_summary = use_program_summary
        self.use_tips = use_tips
        self.use_task_demos = use_task_demos
        self.view_data_batch_size = view_data_batch_size
        self.seed = seed
        self.optimizer_model_id = optimizer_model
        self._rng = random.Random(seed)
        self.model = self._build_model(optimizer_model, openrouter_key)

        # Split target units into instruction (PROMPT) and demos (FIELD, kind=demos) units.
        self.prompt_units = [u for u in self.target_units if u.unit_type == OptimizationUnitType.PROMPT]
        self.demo_units = [
            u for u in self.target_units
            if u.unit_type == OptimizationUnitType.FIELD and (u.metadata or {}).get("kind") == "demos"
        ]
        if not self.prompt_units:
            raise ValueError("MiproOptimizer requires at least one PROMPT unit to optimize.")

        # Live (non-serializable) search state; rebuilt from optimizer_state on resume.
        self._study: Optional[optuna.Study] = None
        self._pending_trials: Dict[int, optuna.trial.Trial] = {}
        self._instruction_candidates: Dict[str, List[str]] = {}
        self._demo_candidates: Dict[str, List[List[Dict[str, Any]]]] = {}
        self._dataset_summary: Optional[str] = None
        self._history: List[Dict[str, Any]] = []
        self._baseline_snapshot_id: Optional[str] = None

    @staticmethod
    def _build_model(optimizer_model: str, openrouter_key: Optional[str]) -> OpenRouterLLM:
        key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "No OpenRouter API key found. Pass `openrouter_key=...` or set $OPENROUTER_API_KEY."
            )
        config = OpenRouterConfig(model=optimizer_model, openrouter_key=key, output_response=True, stream=True)
        return OpenRouterLLM(config=config)

    # =========================================================================
    # Phase 1: bootstrap few-shot demo candidate sets
    # =========================================================================
    @property
    def _zeroshot(self) -> bool:
        return self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

    def _adapter_supports_trace(self) -> bool:
        return hasattr(self.adapter, "async_execute_with_trace") or hasattr(self.adapter, "execute_with_trace")

    def _bootstrap_success(self, score: Any) -> bool:
        if self.metric_threshold is not None:
            try:
                return score >= self.metric_threshold
            except TypeError:
                return False
        return bool(score)

    def _demo_unit_fields(self, unit: OptimizationUnit) -> List[str]:
        metadata = unit.metadata or {}
        return list(metadata.get("input_names") or []) + list(metadata.get("output_names") or [])

    def _labeled_demo(self, unit: OptimizationUnit, example: dict) -> Optional[Dict[str, Any]]:
        """Build a labeled demo for `unit` from a raw training example, or None if impossible."""
        fields = self._demo_unit_fields(unit)
        source: Dict[str, Any] = dict(self.input_fn(example) or {})
        if self.label_fn is not None:
            source.update(self.label_fn(example) or {})
        else:
            source.update(example)
        demo = {name: _jsonify_value(source[name]) for name in fields if name in source}
        output_names = list((unit.metadata or {}).get("output_names") or [])
        if not demo or not any(name in demo for name in output_names):
            return None
        return demo

    def _labeled_demo_set(self, examples: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
        """Labeled-only demos per demos-unit uid, sampled from `examples`."""
        result: Dict[str, List[Dict[str, Any]]] = {unit.uid: [] for unit in self.demo_units}
        for unit in self.demo_units:
            for example in examples:
                if len(result[unit.uid]) >= self.max_labeled_demos:
                    break
                demo = self._labeled_demo(unit, example)
                if demo is not None:
                    result[unit.uid].append(demo)
        return result

    async def _async_bootstrap_one_set(
        self, examples: List[dict], max_demos: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run the program over `examples` until `max_demos` rollouts pass the metric."""
        collected: Dict[str, List[Dict[str, Any]]] = {unit.uid: [] for unit in self.demo_units}
        successes = 0

        async def run_one(example: dict) -> Optional[Dict[str, Dict[str, Any]]]:
            try:
                result, traces = await self.adapter.async_execute_with_trace(inputs=self.input_fn(example))
                prediction = self.output_fn(result)
                score = self.metric_fn(prediction, example)
            except Exception as exc:
                logger.warning(f"MIPRO bootstrap rollout failed and is skipped: {exc}")
                return None
            return traces if self._bootstrap_success(score) else None

        for start in range(0, len(examples), self.bootstrap_concurrency):
            if successes >= max_demos:
                break
            chunk = examples[start:start + self.bootstrap_concurrency]
            for traces in await asyncio.gather(*[run_one(example) for example in chunk]):
                if traces is None or successes >= max_demos:
                    continue
                successes += 1
                for uid, demo in traces.items():
                    if uid in collected:
                        collected[uid].append(demo)
        return collected

    def _bootstrap_one_set(self, examples: List[dict], max_demos: int) -> Dict[str, List[Dict[str, Any]]]:
        """Sync variant of `_async_bootstrap_one_set` (sequential rollouts)."""
        collected: Dict[str, List[Dict[str, Any]]] = {unit.uid: [] for unit in self.demo_units}
        successes = 0
        for example in examples:
            if successes >= max_demos:
                break
            try:
                result, traces = self.adapter.execute_with_trace(inputs=self.input_fn(example))
                prediction = self.output_fn(result)
                score = self.metric_fn(prediction, example)
            except Exception as exc:
                logger.warning(f"MIPRO bootstrap rollout failed and is skipped: {exc}")
                continue
            if not self._bootstrap_success(score):
                continue
            successes += 1
            for uid, demo in traces.items():
                if uid in collected:
                    collected[uid].append(demo)
        return collected

    def _plan_demo_sets(self) -> List[Dict[str, Any]]:
        """Plan the candidate sets to build, mirroring dspy's create_n_fewshot_demo_sets.

        Set 0 is always the baseline's current demos. Then (budget permitting): one
        labeled-only set, one unshuffled bootstrapped set, and shuffled bootstrapped
        sets of random size. In zero-shot mode demos are excluded from the search space,
        but one small bootstrapped set is still built purely to ground the instruction
        proposals (dspy does the same).
        """
        if self._zeroshot:
            return [
                {"kind": "baseline"},
                {"kind": "bootstrap", "shuffle": False, "size": DEMOS_FOR_PROPOSAL_ONLY},
            ]
        plans: List[Dict[str, Any]] = [{"kind": "baseline"}]
        if self.max_labeled_demos > 0 and len(plans) < self.num_demo_candidates:
            plans.append({"kind": "labeled"})
        if self.max_bootstrapped_demos > 0 and len(plans) < self.num_demo_candidates:
            plans.append({"kind": "bootstrap", "shuffle": False, "size": self.max_bootstrapped_demos})
        while self.max_bootstrapped_demos > 0 and len(plans) < self.num_demo_candidates:
            plans.append({
                "kind": "bootstrap",
                "shuffle": True,
                "size": self._rng.randint(1, self.max_bootstrapped_demos),
            })
        return plans

    def _assemble_demo_candidates(
        self,
        baseline_snapshot: SnapShot,
        plans: List[Dict[str, Any]],
        bootstrapped_sets: List[Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Combine baseline/labeled/bootstrapped materials into per-uid candidate set lists."""
        labeled = self._labeled_demo_set(self.trainset) if self.max_labeled_demos > 0 else {}
        candidates: Dict[str, List[List[Dict[str, Any]]]] = {unit.uid: [] for unit in self.demo_units}
        bootstrap_index = 0
        for plan in plans:
            if plan["kind"] == "baseline":
                for unit in self.demo_units:
                    baseline_demos = baseline_snapshot.unit_values.get(unit.uid) or []
                    candidates[unit.uid].append([dict(demo) for demo in baseline_demos])
            elif plan["kind"] == "labeled":
                for unit in self.demo_units:
                    candidates[unit.uid].append(list(labeled.get(unit.uid, [])))
            else:  # bootstrap: pad with labeled demos up to max_labeled_demos (dspy's _train)
                bootstrapped = bootstrapped_sets[bootstrap_index]
                bootstrap_index += 1
                for unit in self.demo_units:
                    demos = list(bootstrapped.get(unit.uid, []))
                    pad = [
                        demo for demo in labeled.get(unit.uid, [])
                        if demo not in demos
                    ][:max(0, self.max_labeled_demos - len(demos))]
                    candidates[unit.uid].append(demos + pad)
        return candidates

    def _bootstrap_plan_examples(self, plan: Dict[str, Any]) -> List[dict]:
        examples = list(self.trainset)
        if plan.get("shuffle"):
            self._rng.shuffle(examples)
        return examples

    # =========================================================================
    # Phase 2: propose instruction candidates
    # =========================================================================
    def _program_summary(self) -> str:
        """Render an adapter-agnostic overview of the program's modules for the proposer."""
        lines = []
        for unit in self.prompt_units:
            metadata = unit.metadata or {}
            role = metadata.get("role")
            demos_unit = next(
                (u for u in self.demo_units if (u.metadata or {}).get("paired_prompt_uid") == unit.uid),
                None,
            )
            io = ""
            if demos_unit is not None:
                demo_metadata = demos_unit.metadata or {}
                io = (
                    f" (inputs: {', '.join(demo_metadata.get('input_names') or []) or 'n/a'}; "
                    f"outputs: {', '.join(demo_metadata.get('output_names') or []) or 'n/a'})"
                )
            lines.append(f"- module '{unit.name}'{io}: {role or 'no description provided'}")
        return "The program consists of the following LLM module(s):\n" + "\n".join(lines)

    def _dataset_sample_text(self) -> str:
        examples = self.trainset[: self.view_data_batch_size]
        return "\n\n".join(_render_demo({k: _jsonify_value(v) for k, v in ex.items()}) for ex in examples)

    def _proposal_task_demos(self, unit: OptimizationUnit, candidate_index: int) -> str:
        """Pick grounding demos for one instruction proposal from the bootstrapped sets."""
        if not self.use_task_demos:
            return ""
        demos_unit = next(
            (u for u in self.demo_units if (u.metadata or {}).get("paired_prompt_uid") == unit.uid),
            None,
        )
        if demos_unit is None:
            return ""
        sets = self._demo_candidates.get(demos_unit.uid) or []
        if not sets:
            return ""
        # Rotate the starting set with the candidate index so different candidates are
        # grounded in different demos (dspy pairs candidate i with demo set i).
        ordered = sets[candidate_index % len(sets):] + sets[:candidate_index % len(sets)]
        gathered: List[str] = []
        for demo_set in ordered:
            for demo in demo_set:
                gathered.append(_render_demo(demo))
                if len(gathered) >= DEMOS_FOR_PROPOSAL_ONLY:
                    return "\n\n".join(gathered)
        return "\n\n".join(gathered)

    def _build_instruction_proposal_prompt(
        self, unit: OptimizationUnit, current_instruction: str, candidate_index: int
    ) -> str:
        sections: List[str] = [f"TASK DESCRIPTION:\n{self.task_description}"]
        if self.use_dataset_summary and self._dataset_summary:
            sections.append(f"DATASET SUMMARY:\n{self._dataset_summary}")
        if self.use_program_summary:
            sections.append(f"PROGRAM DESCRIPTION:\n{self._program_summary()}")
        role = (unit.metadata or {}).get("role")
        module_desc = f"MODULE:\nThe instruction to improve belongs to module '{unit.name}'."
        if role:
            module_desc += f" Its role in the program is: {role}"
        sections.append(module_desc)
        task_demos = self._proposal_task_demos(unit, candidate_index)
        sections.append(f"TASK DEMO(S):\n{task_demos or 'No task demos provided.'}")
        sections.append(f"BASIC INSTRUCTION:\n{current_instruction}")
        if self.use_tips:
            tip = PROPOSER_TIPS[self._rng.choice(list(PROPOSER_TIPS.keys()))]
            if tip:
                sections.append(f"TIP:\n{tip}")
        sections.append("PROPOSED INSTRUCTION:")
        return "\n\n".join(sections)

    @staticmethod
    def _clean_instruction(text: str) -> str:
        """Strip stray markdown fences / wrapping quotes the LLM may add."""
        cleaned = (text or "").strip()
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        cleaned = re.sub(r"^PROPOSED INSTRUCTION:\s*", "", cleaned)
        return cleaned

    def _summarize_dataset(self) -> Optional[str]:
        if not self.use_dataset_summary:
            return None
        try:
            summary = self.model.single_generate(
                messages=[
                    {"role": "system", "content": _DATASET_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"DATASET SAMPLE:\n{self._dataset_sample_text()}"},
                ],
                temperature=0.5,
            )
            return summary.strip() or None
        except Exception as exc:
            logger.warning(f"MIPRO dataset summary failed; proposing without it: {exc}")
            return None

    async def _async_summarize_dataset(self) -> Optional[str]:
        if not self.use_dataset_summary:
            return None
        try:
            summary = await self.model.single_generate_async(
                messages=[
                    {"role": "system", "content": _DATASET_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"DATASET SAMPLE:\n{self._dataset_sample_text()}"},
                ],
                temperature=0.5,
                output_response=False,
            )
            return summary.strip() or None
        except Exception as exc:
            logger.warning(f"MIPRO dataset summary failed; proposing without it: {exc}")
            return None

    def _propose_instructions(self, baseline_snapshot: SnapShot) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {}
        for unit in self.prompt_units:
            current = baseline_snapshot.unit_values[unit.uid]
            candidates[unit.uid] = [current]
            for candidate_index in range(1, self.num_instruction_candidates):
                proposal = self.model.single_generate(
                    messages=[
                        {"role": "system", "content": _INSTRUCTION_PROPOSAL_SYSTEM_PROMPT},
                        {"role": "user", "content": self._build_instruction_proposal_prompt(unit, current, candidate_index)},
                    ],
                    temperature=self.init_temperature,
                )
                cleaned = self._clean_instruction(proposal)
                candidates[unit.uid].append(cleaned or current)
        return candidates

    async def _async_propose_instructions(self, baseline_snapshot: SnapShot) -> Dict[str, List[str]]:
        index: List[Tuple[str, int]] = []
        tasks = []
        for unit in self.prompt_units:
            current = baseline_snapshot.unit_values[unit.uid]
            for candidate_index in range(1, self.num_instruction_candidates):
                # Prompts are built up-front (sequentially) so RNG-driven tips stay
                # deterministic for a given seed regardless of task scheduling.
                messages = [
                    {"role": "system", "content": _INSTRUCTION_PROPOSAL_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_instruction_proposal_prompt(unit, current, candidate_index)},
                ]
                index.append((unit.uid, candidate_index))
                tasks.append(
                    self.model.single_generate_async(
                        messages=messages, temperature=self.init_temperature, output_response=False
                    )
                )
        results = await asyncio.gather(*tasks)

        candidates: Dict[str, List[str]] = {
            unit.uid: [baseline_snapshot.unit_values[unit.uid]] for unit in self.prompt_units
        }
        by_uid: Dict[str, Dict[int, str]] = {}
        for (uid, candidate_index), text in zip(index, results):
            by_uid.setdefault(uid, {})[candidate_index] = text
        for unit in self.prompt_units:
            current = candidates[unit.uid][0]
            for candidate_index in range(1, self.num_instruction_candidates):
                cleaned = self._clean_instruction(by_uid.get(unit.uid, {}).get(candidate_index, ""))
                candidates[unit.uid].append(cleaned or current)
        return candidates

    # =========================================================================
    # Engine hooks
    # =========================================================================
    def _baseline_snapshot(self, state: OptimizationRunState) -> SnapShot:
        baseline_record = state.get_baseline_record()
        snapshot = state.get_snapshot_by_id(baseline_record.snapshot_id) if baseline_record else None
        if snapshot is None:
            raise RuntimeError("MiproOptimizer requires the baseline snapshot to be present in the run state.")
        return snapshot

    def _require_scalar_objective(self, objective: Objective) -> ScalarObjective:
        if not isinstance(objective, ScalarObjective):
            raise TypeError(
                "MiproOptimizer requires a ScalarObjective (the TPE sampler needs a single "
                f"scalar score per trial); got {type(objective).__name__}."
            )
        return objective

    def _search_space(self) -> Dict[str, optuna.distributions.CategoricalDistribution]:
        space: Dict[str, optuna.distributions.CategoricalDistribution] = {}
        for unit in self.prompt_units:
            n = len(self._instruction_candidates.get(unit.uid) or [])
            if n > 1:
                space[f"{unit.uid}||instruction"] = optuna.distributions.CategoricalDistribution(list(range(n)))
        if not self._zeroshot:
            for unit in self.demo_units:
                n = len(self._demo_candidates.get(unit.uid) or [])
                if n > 1:
                    space[f"{unit.uid}||demos"] = optuna.distributions.CategoricalDistribution(list(range(n)))
        return space

    def _create_study(self, objective: ScalarObjective) -> optuna.Study:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        return optuna.create_study(direction=objective.direction, sampler=sampler)

    def _register_history_trial(self, params: Dict[str, int], value: Optional[float], failed: bool = False) -> None:
        self._history.append({"params": params, "value": value, "failed": failed})

    def _replay_history(self) -> None:
        """Re-add persisted completed trials to a fresh study (resume path)."""
        space = self._search_space()
        for entry in self._history:
            if entry.get("failed") or entry.get("value") is None:
                continue
            params = {k: v for k, v in entry["params"].items() if k in space}
            self._study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions={k: space[k] for k in params},
                    value=entry["value"],
                )
            )

    def _finish_run_start(self, state: OptimizationRunState, objective: ScalarObjective) -> None:
        """Persist candidates into optimizer_state and set up the TPE study."""
        # Checkpoint right away: bootstrapping and instruction proposals are the expensive
        # part of a MIPRO run, and the engine's next checkpoint only happens after the
        # first trial batch. Without this, a crash between candidate generation and that
        # first batch would force a resumed run to redo all of it.
        self._checkpoint(state)
        self._study = self._create_study(objective)
        self._replay_history()

        # Seed the sampler with the baseline configuration (all indices 0) and score.
        baseline_record = state.get_baseline_record()
        baseline_score = (baseline_record.metrics or {}).get(objective.metric) if baseline_record else None
        space = self._search_space()
        if baseline_score is not None and space and not self._history:
            params = {name: 0 for name in space}
            self._study.add_trial(
                optuna.trial.create_trial(params=params, distributions=space, value=baseline_score)
            )
            self._register_history_trial(params, baseline_score)

        if not space:
            logger.warning(
                "MIPRO search space is empty (no instruction or demo candidates with more than "
                "one option); every proposal would repeat the baseline."
            )

    def _candidates_fingerprint(self) -> Dict[str, Any]:
        """Configuration that determines the generated candidate pools.

        Persisted alongside the candidates so a resumed run can detect that it was
        constructed with different candidate-generation settings. The check must be
        fail-fast rather than regenerate-on-mismatch: the persisted optuna history
        indexes into the persisted candidate lists, so new candidates would silently
        invalidate every recorded trial.
        """
        trainset_digest = hashlib.md5(
            json.dumps(self.trainset, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "task_description": self.task_description,
            "trainset_digest": trainset_digest,
            "trainset_size": len(self.trainset),
            "optimizer_model": self.optimizer_model_id,
            "num_instruction_candidates": self.num_instruction_candidates,
            "num_demo_candidates": self.num_demo_candidates,
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "metric_threshold": self.metric_threshold,
            "use_dataset_summary": self.use_dataset_summary,
            "use_program_summary": self.use_program_summary,
            "use_tips": self.use_tips,
            "use_task_demos": self.use_task_demos,
            "view_data_batch_size": self.view_data_batch_size,
            "seed": self.seed,
        }

    def _restore_candidates(self, state: OptimizationRunState) -> bool:
        """Load persisted candidates from a resumed run. Returns True when present.

        Raises:
            ValueError: If the persisted candidates were generated under a different
                candidate-generation configuration than this optimizer's.
        """
        persisted = state.optimizer_state or {}
        if not persisted.get("instruction_candidates"):
            return False
        saved_fingerprint = persisted.get("candidates_fingerprint")
        if saved_fingerprint is not None:
            current_fingerprint = self._candidates_fingerprint()
            if saved_fingerprint != current_fingerprint:
                changed = sorted(
                    key for key in set(saved_fingerprint) | set(current_fingerprint)
                    if saved_fingerprint.get(key) != current_fingerprint.get(key)
                )
                raise ValueError(
                    "Cannot resume MIPRO run: the saved instruction/demo candidates were "
                    f"generated under a different configuration (changed: {', '.join(changed)}). "
                    "Revert the changed settings, or start a fresh run with a new save_dir."
                )
        self._instruction_candidates = persisted["instruction_candidates"]
        self._demo_candidates = persisted.get("demo_candidates") or {}
        self._dataset_summary = persisted.get("dataset_summary")
        self._history = list(persisted.get("history") or [])
        return True

    def _prepare_demo_materials(self) -> List[Dict[str, Any]]:
        """Shared pre-bootstrap validation/planning. Returns the demo-set build plans."""
        if self.demo_units and not self._adapter_supports_trace():
            logger.warning(
                f"{type(self.adapter).__name__} does not implement execute_with_trace / "
                "async_execute_with_trace; demo bootstrapping is disabled (labeled demos only)."
            )
        return self._plan_demo_sets() if self.demo_units else []

    def on_run_start(self, state: OptimizationRunState, objective: Objective, **kwargs) -> None:
        objective = self._require_scalar_objective(objective)
        baseline_snapshot = self._baseline_snapshot(state)
        self._baseline_snapshot_id = baseline_snapshot.snapshot_id

        if not self._restore_candidates(state):
            plans = self._prepare_demo_materials()
            bootstrap_plans = [p for p in plans if p["kind"] == "bootstrap"]
            bootstrapped_sets = []
            if bootstrap_plans and hasattr(self.adapter, "execute_with_trace"):
                logger.info(f"MIPRO step 1: bootstrapping {len(bootstrap_plans)} demo candidate set(s)...")
                bootstrapped_sets = [
                    self._bootstrap_one_set(self._bootstrap_plan_examples(plan), plan["size"])
                    for plan in bootstrap_plans
                ]
            else:
                plans = [p for p in plans if p["kind"] != "bootstrap"]
            self._demo_candidates = self._assemble_demo_candidates(baseline_snapshot, plans, bootstrapped_sets)
            logger.info("MIPRO step 2: proposing instruction candidates...")
            self._dataset_summary = self._summarize_dataset()
            self._instruction_candidates = self._propose_instructions(baseline_snapshot)

        self._finish_run_start(state, objective)

    async def async_on_run_start(self, state: OptimizationRunState, objective: Objective, **kwargs) -> None:
        objective = self._require_scalar_objective(objective)
        baseline_snapshot = self._baseline_snapshot(state)
        self._baseline_snapshot_id = baseline_snapshot.snapshot_id

        if not self._restore_candidates(state):
            plans = self._prepare_demo_materials()
            bootstrap_plans = [p for p in plans if p["kind"] == "bootstrap"]
            bootstrapped_sets = []
            if bootstrap_plans and hasattr(self.adapter, "async_execute_with_trace"):
                logger.info(f"MIPRO step 1: bootstrapping {len(bootstrap_plans)} demo candidate set(s)...")
                for plan in bootstrap_plans:
                    bootstrapped_sets.append(
                        await self._async_bootstrap_one_set(self._bootstrap_plan_examples(plan), plan["size"])
                    )
            else:
                plans = [p for p in plans if p["kind"] != "bootstrap"]
            self._demo_candidates = self._assemble_demo_candidates(baseline_snapshot, plans, bootstrapped_sets)
            logger.info("MIPRO step 2: proposing instruction candidates...")
            self._dataset_summary = await self._async_summarize_dataset()
            self._instruction_candidates = await self._async_propose_instructions(baseline_snapshot)

        self._finish_run_start(state, objective)

    def _proposal_from_trial(
        self, trial: optuna.trial.Trial, baseline_snapshot: SnapShot
    ) -> OptimizationProposal:
        """Turn one optuna trial's categorical picks into an engine proposal."""
        changes: List[UnitChange] = []
        raw_params: Dict[str, int] = {}
        for unit in self.prompt_units:
            param = f"{unit.uid}||instruction"
            options = self._instruction_candidates.get(unit.uid) or []
            if len(options) <= 1:
                continue
            idx = trial.suggest_categorical(param, list(range(len(options))))
            raw_params[param] = idx
            changes.append(
                UnitChange.create(
                    unit=unit,
                    new_value=options[idx],
                    old_value=baseline_snapshot.unit_values.get(unit.uid),
                    operation=ChangeOperation.REPLACE,
                    metadata={"strategy": "mipro", "candidate_index": idx, "optuna_trial_number": trial.number},
                )
            )
        if not self._zeroshot:
            for unit in self.demo_units:
                param = f"{unit.uid}||demos"
                sets = self._demo_candidates.get(unit.uid) or []
                if len(sets) <= 1:
                    continue
                idx = trial.suggest_categorical(param, list(range(len(sets))))
                raw_params[param] = idx
                changes.append(
                    UnitChange.create(
                        unit=unit,
                        new_value=[dict(demo) for demo in sets[idx]],
                        old_value=baseline_snapshot.unit_values.get(unit.uid),
                        operation=ChangeOperation.REPLACE,
                        metadata={"strategy": "mipro", "candidate_index": idx, "optuna_trial_number": trial.number},
                    )
                )
        return OptimizationProposal(
            source_snapshot_id=baseline_snapshot.snapshot_id,
            changes=changes,
            metadata={
                "source": "MiproOptimizer",
                "optuna_trial_number": trial.number,
                "params": raw_params,
            },
        )

    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        if self._study is None:
            raise RuntimeError("MiproOptimizer.batch_propose called before on_run_start initialized the study.")
        space = self._search_space()
        if not space:
            return []  # nothing to search; stop the run
        baseline_snapshot = self._baseline_snapshot(state)
        n = self.trials_per_batch if budget_remaining is None else max(0, min(self.trials_per_batch, budget_remaining))
        proposals: List[OptimizationProposal] = []
        for _ in range(n):
            trial = self._study.ask()
            proposal = self._proposal_from_trial(trial, baseline_snapshot)
            self._pending_trials[trial.number] = trial
            proposals.append(proposal)
        return proposals

    def observe(
        self,
        state: OptimizationRunState,
        trial_records: List[TrialRecord],
        objective: Objective,
        **kwargs,
    ) -> None:
        objective = self._require_scalar_objective(objective)
        for record in trial_records:
            trial_number = next(
                (
                    (change.metadata or {}).get("optuna_trial_number")
                    for change in record.changes
                    if (change.metadata or {}).get("optuna_trial_number") is not None
                ),
                None,
            )
            if trial_number is None or trial_number not in self._pending_trials:
                continue
            trial = self._pending_trials.pop(trial_number)
            score = (record.metrics or {}).get(objective.metric) if record.status == "completed" else None
            if score is None:
                self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self._register_history_trial(dict(trial.params), None, failed=True)
            else:
                self._study.tell(trial, score)
                self._register_history_trial(dict(trial.params), score)

    # -- persistence -----------------------------------------------------------
    def serialize_optimizer_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        return {
            "instruction_candidates": self._instruction_candidates,
            "demo_candidates": self._demo_candidates,
            "dataset_summary": self._dataset_summary,
            "history": self._history,
            "candidates_fingerprint": self._candidates_fingerprint(),
        }


__all__ = ["MiproOptimizer", "MiproWorkFlowAdapter", "DEMOS_UID_SUFFIX"]
