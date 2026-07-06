import abc
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final, List, Any, Optional, Dict, Literal, Iterable, Tuple

from .base import ChangeOperation, EvaluationResult, OptimizationUnit, OptimizationUnitType, SnapShot, UnitChange, ValidationResult

if TYPE_CHECKING:
    from ...workflow.workflow_graph import WorkFlowGraph, WorkFlowNode


@dataclass
class ApplyResult:
    """Return value of ProgramAdapter.apply, carrying either a successful (adapter, snapshot) pair or a failure description."""
    status: Literal["success", "failed"]
    adapter: Optional["ProgramAdapter"] = None
    snapshot: Optional[SnapShot] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class TrialWorkspace:
    """Filesystem sandbox for one optimization trial."""
    root_dir: str
    trial_id: int
    source_snapshot_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    keep: Optional[bool] = True

    @classmethod
    def create(
        cls,
        root_dir: str,
        trial_id: int,
        source_snapshot_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        keep: Optional[bool] = True,
        clean: bool = True,
    ) -> "TrialWorkspace":
        workspace = cls(
            root_dir=os.path.abspath(root_dir),
            trial_id=trial_id,
            source_snapshot_id=source_snapshot_id,
            metadata=metadata or {},
            keep=keep,
        )
        # A trial workspace is a fresh per-trial sandbox. When `keep=True` the
        # directory survives across runs, and trial paths can repeat (a baseline
        # retried after a failure, or a resumed run that recomputes the same
        # trial_id+snapshot_id). Reusing the path with `exist_ok=True` alone would
        # leak stale files from the prior attempt into a file-backed adapter, so we
        # remove any existing directory first unless the caller opts out.
        if clean and os.path.exists(workspace.root_dir):
            shutil.rmtree(workspace.root_dir)
        os.makedirs(workspace.root_dir, exist_ok=True)
        return workspace

    def path(self, *parts: str) -> str:
        """Return an absolute path inside this workspace."""
        path = os.path.abspath(os.path.join(self.root_dir, *parts))
        if os.path.commonpath([self.root_dir, path]) != self.root_dir:
            raise ValueError(f"Workspace path escapes root_dir: {path}")
        return path

    def ensure_dir(self, *parts: str) -> str:
        """Create and return a directory inside the workspace."""
        path = self.path(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def cleanup(self) -> None:
        """Remove the workspace when `keep` is False."""
        if not self.keep and os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def as_artifact(self) -> Dict[str, Any]:
        return {
            "root_dir": self.root_dir,
            "trial_id": self.trial_id,
            "source_snapshot_id": self.source_snapshot_id,
            "metadata": self.metadata,
            "kept": self.keep,
        }


@dataclass
class TrialContext:
    """Per-trial context the engine passes to ``evaluate_fn`` as an optional second argument.

    An ``evaluate_fn`` may be declared either as ``fn(adapter)`` or ``fn(adapter, context)``.
    Receiving the context is opt-in by signature: the engine passes it only when the second
    positional parameter is named ``ctx`` / ``context`` / ``trial_context`` or is annotated
    as ``TrialContext`` (so e.g. ``fn(adapter, split="dev")`` keeps its single-argument
    behavior instead of silently receiving a context).

    Attributes:
        trial_id: The trial being evaluated (``BASELINE_TRIAL_ID`` for the baseline).
        trial_dir: Persistent per-trial directory under ``<save_dir>/trials/`` where the
            evaluator can archive results (e.g. an ``EvaluationPipeline`` ``output_dir``).
            None when the run has no ``save_dir``. Unlike a ``TrialWorkspace``, this
            directory is never cleaned up by the engine.
        workspace: The adapter's trial workspace (file sandbox), if workspace isolation
            is enabled for this trial. Subject to ``keep_trial_workspaces`` cleanup.
        metadata: Proposal metadata for trial evaluations; ``{"baseline": True}`` for
            the baseline evaluation.
    """
    trial_id: int
    trial_dir: Optional[str] = None
    workspace: Optional[TrialWorkspace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgramAdapter(abc.ABC):

    #: Whether trials need an isolated, per-trial filesystem workspace.
    #:
    #: A trial workspace is only meaningful for adapters that materialize file-backed
    #: state (file/code/skills adapters that override `prepare_workspace`). Purely
    #: in-memory adapters — prompts, model names, config scalars — have nothing to
    #: write into it, so the engine skips workspace creation for them and never spawns
    #: empty per-trial directories. File-backed adapters should set this to True (or
    #: override the `uses_workspace` property). An explicit `workspace_root` passed to
    #: `optimize()` always forces workspace creation regardless of this flag.
    uses_workspace: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Ensure that subclasses implement either execute() or async_execute()
        has_execute = cls.execute is not ProgramAdapter.execute
        has_async_execute = cls.async_execute is not ProgramAdapter.async_execute
        if not has_execute and not has_async_execute:
            raise TypeError(
                f"{cls.__name__} must implement execute() or async_execute()."
            )

    def _validate_units(self, units: List[OptimizationUnit]) -> List[OptimizationUnit]:
        if not units:
            raise ValueError(f"{self.__class__.__name__} requires at least one optimization unit.")
        if not all(isinstance(unit, OptimizationUnit) for unit in units):
            raise TypeError("register_units() must return a list of OptimizationUnit objects.")

        unit_ids = [unit.uid for unit in units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("OptimizationUnit uid values must be unique.")

        unit_names = [unit.name for unit in units]
        if len(unit_names) != len(set(unit_names)):
            raise ValueError("OptimizationUnit name values must be unique.")

        return units

    @property
    def units(self) -> List[OptimizationUnit]:
        if not hasattr(self, "_units"):
            self._units = self._validate_units(self.register_units())
        return self._units

    def get_unit(self, uid: str) -> Optional[OptimizationUnit]:
        """Return a registered optimization unit by uid, or None if it is unknown."""
        for unit in self.units:
            if unit.uid == uid:
                return unit
        return None

    def select_units(
        self,
        *,
        unit_types: Optional[Iterable[OptimizationUnitType]] = None,
        uids: Optional[Iterable[str]] = None,
    ) -> List[OptimizationUnit]:
        """
        Return registered units filtered by type and/or uid.

        Args:
            unit_types: Optional set of unit types to include.
            uids: Optional set of unit uids to include.

        Returns:
            Matching units in adapter registration order.
        """
        unit_type_set = set(unit_types) if unit_types is not None else None
        uid_set = set(uids) if uids is not None else None
        return [
            unit for unit in self.units
            if (unit_type_set is None or unit.unit_type in unit_type_set)
            and (uid_set is None or unit.uid in uid_set)
        ]

    def fingerprint(self) -> Dict[str, Any]:
        """
        Return a stable adapter compatibility fingerprint for persisted optimization runs.

        Subclasses may override when reconstruction depends on additional adapter-level
        contracts. The default intentionally avoids live object identity and includes only
        declared optimization-unit shape.
        """
        return {
            "adapter_class": self.__class__.__qualname__,
            "units": [
                {
                    "uid": unit.uid,
                    "name": unit.name,
                    "unit_type": unit.unit_type.value,
                    "json_schema": unit.json_schema,
                    "allowed_operations": [
                        op.value if isinstance(op, ChangeOperation) else str(op)
                        for op in unit.allowed_operations
                    ],
                    "operation_schemas": unit.operation_schemas,
                }
                for unit in self.units
            ],
        }

    def _validate_changes(self, changes: List[UnitChange]) -> None:
        if not isinstance(changes, list):
            raise TypeError("changes must be a list of UnitChange objects.")
        if not all(isinstance(change, UnitChange) for change in changes):
            raise TypeError("changes must be a list of UnitChange objects.")

        units_by_uid = {unit.uid: unit for unit in self.units}
        unknown_uids = [change.uid for change in changes if change.uid not in units_by_uid]
        if unknown_uids:
            raise ValueError(
                "Changes reference unknown OptimizationUnit uid values: "
                f"{', '.join(unknown_uids)}."
            )

        for change in changes:
            unit = units_by_uid[change.uid]
            UnitChange.validate_value(change.value, unit, operation=change.operation)

    def _validate_snapshot(self, snapshot: SnapShot, context: str = "snapshot") -> None:
        """Validate registered unit values in a snapshot against their json_schema."""
        if not isinstance(snapshot, SnapShot):
            raise TypeError(f"{context} must be a SnapShot instance, got {type(snapshot).__name__}.")
        if not isinstance(snapshot.unit_values, dict):
            raise TypeError(f"{context}.unit_values must be a dict.")

        missing_uids = [unit.uid for unit in self.units if unit.uid not in snapshot.unit_values]
        if missing_uids:
            raise ValueError(
                f"{context} is missing value(s) for OptimizationUnit uid(s): "
                f"{', '.join(missing_uids)}."
            )

        for unit in self.units:
            unit.validate_value(snapshot.unit_values[unit.uid], context=context)

    @property
    def workspace(self) -> Optional[TrialWorkspace]:
        """Workspace bound to this adapter for the current trial, if any."""
        return getattr(self, "_workspace", None)

    def bind_workspace(self, workspace: Optional[TrialWorkspace]) -> None:
        """Attach a trial workspace to this adapter before validation/evaluation."""
        self._workspace = workspace

    def prepare_workspace(
        self,
        workspace: TrialWorkspace,
        snapshot: SnapShot,
        **kwargs,
    ) -> None:
        """
        Optional hook to materialize file-backed state into a trial workspace.

        File/code/skills adapters can override this to write snapshot contents into
        `workspace` and then execute against those isolated files.
        """
        pass

    def validate_trial(
        self,
        snapshot: SnapShot,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> List[ValidationResult]:
        """
        Optional validation pipeline run after apply/workspace preparation and before evaluation.

        For proposed trials, `snapshot` is the post-apply snapshot produced after
        `changes` have already been merged. For baseline evaluation, `changes` is
        empty and `snapshot` is the baseline snapshot.

        Override this for static checks, import checks, smoke tests, schema checks, or
        adapter-specific consistency validation. The engine records all returned results
        and skips evaluation when any result has status="failed".
        """
        return []

    async def async_validate_trial(
        self,
        snapshot: SnapShot,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> List[ValidationResult]:
        """Async variant of `validate_trial`."""
        return self.validate_trial(snapshot, changes, workspace=workspace, **kwargs)

    def capture_after_eval(
        self,
        snapshot: SnapShot,
        evaluation: EvaluationResult,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> Optional[SnapShot]:
        """
        Optional hook to capture program state that changed *during* evaluation.

        The engine normally records the snapshot produced by `merge_changes` as the
        trial result. That snapshot reflects the state *before* `evaluate_fn` ran, so
        any mutation the program performs on itself while being evaluated — online
        memory growth, accumulated experience, learned counters, a skill library that
        self-extends during a rollout — would otherwise be lost.

        Override this to return a fresh SnapShot reflecting the post-evaluation state.
        When a SnapShot is returned, the engine records *it* as the trial's result
        snapshot (and therefore as the candidate the objective may select as best, and
        the snapshot future proposals can branch from). Returning ``None`` keeps the
        default behavior (the pre-evaluation snapshot is recorded).

        This runs before the trial workspace is cleaned up, so workspace files written
        during evaluation are still readable here. Keep it lightweight and avoid raising:
        an exception here fails the trial even though evaluation itself succeeded.

        Args:
            snapshot: The pre-evaluation snapshot produced by `merge_changes`.
            evaluation: The normalized EvaluationResult returned by the evaluator.
            changes: The changes that were applied to produce `snapshot`.
            workspace: The trial workspace, if workspace isolation is enabled.
            **kwargs: Subclass-specific arguments forwarded from the trial runner.

        Returns:
            A new SnapShot to record instead of `snapshot`, or None to keep `snapshot`.
        """
        return None

    async def async_capture_after_eval(
        self,
        snapshot: SnapShot,
        evaluation: EvaluationResult,
        changes: List[UnitChange],
        workspace: Optional[TrialWorkspace] = None,
        **kwargs,
    ) -> Optional[SnapShot]:
        """Async variant of `capture_after_eval`."""
        return self.capture_after_eval(snapshot, evaluation, changes, workspace=workspace, **kwargs)

    @final
    def apply(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> ApplyResult:
        """
        Apply a changeset on top of a snapshot and return an ApplyResult.

        Subclasses should NOT override this method. Use `pre_apply_hook` / `post_apply_hook`
        for lifecycle customization, and `from_snapshot` / `merge_changes` for the core logic.

        The original adapter and the input snapshot are never modified. On success the returned
        ApplyResult carries both the new adapter and the SnapShot produced by `merge_changes`,
        so the caller never needs to call `take_snapshot()` again to obtain a consistent snapshot.

        Args:
            snapshot: The baseline snapshot to apply changes on top of.
            changes:  List of UnitChange objects specifying which units to update.

        Returns:
            ApplyResult with status="success" and (adapter, snapshot) on success, or
            status="failed" and an error message string on any exception.
        """
        try:
            processed_changes = self.pre_apply_hook(snapshot, changes, **kwargs)
            self._validate_changes(processed_changes)
            new_snapshot = self.merge_changes(snapshot, processed_changes, **kwargs)
            if not isinstance(new_snapshot, SnapShot):
                raise TypeError(
                    f"merge_changes() must return a SnapShot instance, "
                    f"got {type(new_snapshot).__name__}."
                )
            self._validate_snapshot(new_snapshot, context="post-merge snapshot")
            new_adapter = self.from_snapshot(new_snapshot, **kwargs)
            if not isinstance(new_adapter, ProgramAdapter):
                raise TypeError(
                    f"from_snapshot() must return a ProgramAdapter instance, "
                    f"got {type(new_adapter).__name__}."
                )
            self.post_apply_hook(new_adapter, processed_changes, **kwargs)
            return ApplyResult(status="success", adapter=new_adapter, snapshot=new_snapshot)
        except Exception as e:
            return ApplyResult(status="failed", error=str(e))

    @final
    def load_snapshot(self, snapshot: SnapShot, **kwargs) -> "ProgramAdapter":
        """
        Reconstruct a new adapter from a snapshot, delegating to `from_snapshot`.

        Called once at the end of `Optimizer.optimize()` to restore the best-found adapter.
        The current adapter is not modified.

        Args:
            snapshot: A SnapShot previously produced by `take_snapshot`.
            **kwargs: Forwarded to `from_snapshot`.

        Returns:
            A new ProgramAdapter instance whose state matches the given snapshot.
        """
        return self.from_snapshot(snapshot, **kwargs)

    def execute(self, *args, **kwargs) -> Any:
        """Run the adapted program synchronously.

        Reentrancy contract: the engine passes one trial adapter instance to
        ``evaluate_fn``. Evaluators may call this method or ``async_execute``
        concurrently on that same instance when they fan out over a dataset, and may
        also run directly against adapter-owned execution state. Adapter
        implementations that support such evaluators must keep per-run state local to
        the call, or isolate it with copies/workspaces/locks. If an adapter
        intentionally mutates shared runtime state, its evaluator should run it serially
        or capture the resulting state through ``capture_after_eval``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement execute().")

    async def async_execute(self, *args, **kwargs) -> Any:
        """Run the adapted program asynchronously.

        The same reentrancy contract as ``execute`` applies.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement async_execute().")

    @abc.abstractmethod
    def register_units(self) -> List[OptimizationUnit]:
        """
        Declare the optimization units exposed by this adapter.

        Called once during initialization; the result is cached in `self._units`.
        Each unit describes one optimizable component (e.g. a prompt string, model name,
        temperature value).

        Subclasses must ensure:
            - At least one unit is returned.
            - All unit `uid` and `name` values are unique.
            - If a unit has a `json_schema`, it reflects the expected type/structure
              of the value in a corresponding `UnitChange`.

        Returns:
            A list of OptimizationUnit objects.

        Example:
            def register_units(self) -> List[OptimizationUnit]:
                return [
                    OptimizationUnit(name="system_prompt", unit_type=OptimizationUnitType.PROMPT),
                    OptimizationUnit(name="model_name", unit_type=OptimizationUnitType.MODEL),
                ]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def take_snapshot(self) -> SnapShot:
        """
        Capture the current state of the program as a SnapShot.

        Must populate `unit_values` as a dict mapping each registered unit's `uid`
        to its current value. If `from_snapshot` needs additional context beyond
        `unit_values` to reconstruct the adapter (e.g. non-optimizable model settings,
        pipeline paths), also populate `program_config`. All values in `program_config`
        must be JSON-serializable (str, int, float, bool, None, list, dict) so the
        snapshot can be persisted and restored without data loss.

        Returns:
            A SnapShot capturing the current value of every registered optimization unit.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        """
        Produce a new SnapShot reflecting the proposed changes applied on top of the given snapshot.

        The input snapshot must remain unmodified. Return a fresh SnapShot whose
        `unit_values` (and optionally `program_config`) reflect the result of applying
        each change. The simplest implementation is a plain dict update, but you may
        apply custom merging semantics (e.g. additive updates, cross-unit dependencies,
        config recomputation).

        Called by `apply` after validation and before `from_snapshot`.

        Args:
            snapshot: The baseline snapshot to merge changes into.
            changes:  Validated list of UnitChange objects (all uids are guaranteed registered).
            **kwargs: Optional subclass-specific arguments forwarded from `apply`.

        Returns:
            A new SnapShot whose state reflects the applied changes.

        Example:
            def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
                new_values = dict(snapshot.unit_values)
                for change in changes:
                    new_values[change.uid] = change.value
                return SnapShot(unit_values=new_values, program_config=snapshot.program_config)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "ProgramAdapter":
        """
        Build a new ProgramAdapter whose state matches the given SnapShot.

        This is the single construction point shared by `apply` (produces a trial adapter)
        and `load_snapshot` (restores the best adapter after optimization). Keeping all
        reconstruction logic here avoids duplication and makes the contract clear:
        given a snapshot, produce a fully initialized adapter.

        The current adapter must remain unmodified. Create and return a fresh instance
        initialized from `snapshot.unit_values` (and `snapshot.program_config` if needed).

        IMPORTANT: return an instance of *your own* adapter subclass, not a base class.
        Every trial (and the baseline) is evaluated against the adapter produced here, so
        if a subclass returns its parent type, all of the subclass's overrides —
        `validate_trial`, `prepare_workspace`, `execute`, lifecycle and `capture_after_eval`
        hooks — are silently lost at evaluation time. When subclassing an existing adapter,
        override `from_snapshot` to construct the subclass.

        Args:
            snapshot: A SnapShot previously produced by `take_snapshot`.
            **kwargs: Optional subclass-specific arguments.

        Returns:
            A new instance of this adapter's concrete type whose state matches the snapshot.
        """
        raise NotImplementedError

    def pre_apply_hook(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> List[UnitChange]:
        """
        Optional hook called before changes are validated and merged.
        Override to preprocess or transform the changeset.
        Returns the changeset unchanged by default.

        Args:
            snapshot: The baseline snapshot to merge changes into.
            changes: The original list of UnitChange objects.
            **kwargs: Optional subclass-specific arguments forwarded from `apply`.

        Returns:
            A (possibly modified) list of UnitChange objects.
        """
        return changes
    
    def post_apply_hook(self, new_adapter: "ProgramAdapter", changes: List[UnitChange], **kwargs) -> None:  # noqa: ARG002
        """
        Optional hook called after `from_snapshot`.
        Override to perform post-processing on the newly created adapter `new_adapter`.

        Args:
            new_adapter: The ProgramAdapter instance returned by `from_snapshot`.
            changes:     The list of UnitChange objects that were applied (after preprocessing using `pre_apply_hook`).
            **kwargs:    Optional subclass-specific arguments forwarded from `apply`.
        """
        pass


# ---------------------------------------------------------------------------
# WorkFlowGraph-backed adapter shared by prompt-level optimizers.
# ---------------------------------------------------------------------------
class WorkFlowAdapter(ProgramAdapter):
    """
    General-purpose adapter whose optimizable prompts are drawn from the agents of a
    :class:`~evoagentx.workflow.workflow_graph.WorkFlowGraph`.

    Given a workflow graph, this adapter automatically derives one ``PROMPT`` unit per agent
    that carries editable prompt text (keyed by agent name, since a node may hold several
    agents). For each agent the optimizable text is its ``prompt``, or — when a
    ``prompt_template`` is present (which takes precedence, mirroring ``CustomizeAgent``) —
    the template's ``instruction``. The agent's (or node's) description becomes that prompt's
    role (exposed via ``unit.metadata["role"]``) so prompt optimizers can rewrite each prompt
    for *its* job, and the graph's ``goal`` is the overall system goal. ``execute`` /
    ``async_execute`` run the graph end-to-end: they build a fresh ``AgentManager`` +
    ``WorkFlow`` and call ``workflow.execute`` / ``workflow.async_execute``
    (cf. ``examples/workflow``).

    Subclasses can expose additional per-agent units (e.g. few-shot demos) by overriding
    ``register_units`` / ``take_snapshot`` / ``_rebuild_graph``; ``from_snapshot`` funnels
    the full ``unit_values`` mapping through ``_rebuild_graph`` so subclass units are
    reconstructed alongside the prompts.

    Concurrency: each call runs on its own copy of the graph (and fresh agents), so a
    single adapter instance is safe to execute concurrently — e.g. an ``evaluate_fn``
    that gathers ``async_execute`` over a dataset, or fans ``execute`` out across a
    thread pool. The ``llm``, ``tools``, and ``llm_config`` dependencies are still
    shared by reference across those concurrent runs (deep-copying LLM clients /
    toolkits is undesirable), so they must themselves be safe for concurrent use.
    Evaluators that bypass ``execute`` and run ``adapter.graph`` directly must copy the
    graph first; workflow execution mutates node status.

    Args:
        graph: The ``WorkFlowGraph`` to optimize. Its nodes' agent prompts become the
            optimizable units.
        tools: Optional list of toolkits passed to the ``AgentManager`` at execution time.
        llm: The ``BaseLLM`` instance driving the ``WorkFlow`` (scheduling / output
            extraction).
        llm_config: The ``LLMConfig`` used to instantiate each node's agent.
        max_execution_steps: Forwarded to ``WorkFlow.max_execution_steps``.
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
        from ...workflow.workflow_graph import WorkFlowGraph

        if not isinstance(graph, WorkFlowGraph):
            raise TypeError(f"`graph` must be a WorkFlowGraph instance, got {type(graph).__name__}.")
        self.graph = graph
        self.tools = tools
        self.llm, self.llm_config = self._reconcile_llm(llm, llm_config)
        self.max_execution_steps = max_execution_steps

        prompts, roles = self._extract_prompts_and_roles(graph)
        if not prompts:
            raise ValueError(
                "No optimizable prompts found in the workflow graph: none of its agents "
                "carry an editable `prompt` or `prompt_template` instruction."
            )
        self.prompts: Dict[str, str] = prompts
        self.roles: Dict[str, str] = roles

    @staticmethod
    def _reconcile_llm(llm: Optional[Any], llm_config: Optional[Any]) -> Tuple[Any, Any]:
        """Return a consistent ``(llm, llm_config)`` pair, deriving whichever is missing.

        Both are used when running the workflow, so they must agree. ``llm`` wins when
        both are supplied; ``llm_config`` is then taken from ``llm.config``.
        """
        from ...core.logging import logger
        from ...models.base_model import BaseLLM
        from ...models.model_utils import create_llm_instance

        if llm is None and llm_config is None:
            raise ValueError("Provide at least one of `llm` or `llm_config`.")
        if llm is not None:
            if not isinstance(llm, BaseLLM):
                raise TypeError(f"`llm` must be a BaseLLM instance, got {type(llm).__name__}.")
            if llm_config is not None and llm_config is not llm.config:
                logger.warning(
                    f"Both `llm` and `llm_config` were provided to {type(llm).__name__}; "
                    "using `llm` and ignoring the passed `llm_config` (taking llm.config)."
                )
            return llm, llm.config
        return create_llm_instance(llm_config), llm_config

    # -- prompt extraction --------------------------------------------------
    @staticmethod
    def _template_instruction(template: Any) -> Optional[str]:
        """Read the optimizable ``instruction`` from a prompt template (dict or PromptTemplate)."""
        if template is None:
            return None
        instruction = template.get("instruction") if isinstance(template, dict) else getattr(template, "instruction", None)
        return instruction if isinstance(instruction, str) and instruction.strip() else None

    @staticmethod
    def _set_template_instruction(template: Any, text: str) -> None:
        """Write ``text`` back as the prompt template's ``instruction`` (dict or PromptTemplate)."""
        if isinstance(template, dict):
            template["instruction"] = text
        elif hasattr(template, "set_instruction"):
            template.set_instruction(text)
        else:
            setattr(template, "instruction", text)

    @classmethod
    def _agent_prompt_container(cls, agent: Any) -> Optional[Dict[str, Any]]:
        """Return the dict that carries the agent's editable ``prompt`` / ``prompt_template``.

        Two serialized agent layouts exist:

        * ``CustomizeAgent``-style: ``prompt`` / ``prompt_template`` live directly on the
          agent dict — the container is the agent itself.
        * ``Agent``-instance-style: prompts live on the agent's actions
          (``agent["actions"][j]``). Every agent carries a ``ContextExtraction`` action
          with no prompt, so the container is the single prompt-bearing action. Multiple
          prompt-bearing actions are rejected: units are keyed per agent, so with several
          prompt actions the optimizer could silently tune a prompt the workflow never
          executes.

        Returns None for agents (e.g. bare string references) with no editable prompt slot.

        Raises:
            ValueError: If the agent carries more than one prompt-bearing action.
        """
        if not isinstance(agent, dict):
            return None
        actions = agent.get("actions")
        if not actions:
            return agent
        prompt_actions = [
            action for action in actions
            if isinstance(action, dict)
            and (action.get("prompt_template") is not None or isinstance(action.get("prompt"), str))
        ]
        if not prompt_actions:
            return None
        if len(prompt_actions) > 1:
            raise ValueError(
                f"Agent {agent.get('name')!r} has {len(prompt_actions)} prompt-bearing actions; "
                "prompt units are keyed per agent, so only single-prompt-action agents can be "
                "optimized unambiguously. Split the agent or remove the extra prompt actions."
            )
        return prompt_actions[0]

    @classmethod
    def _agent_prompt_text(cls, agent: Any) -> Optional[str]:
        """The active optimizable prompt text of an agent config dict.

        Mirrors ``CustomizeAgent``: when both are present ``prompt_template`` wins, so the
        template's ``instruction`` is the optimizable text; otherwise the raw ``prompt`` is.
        For ``Agent``-instance-style dicts the prompt is read from the prompt-bearing
        action (see ``_agent_prompt_container``). Returns ``None`` for agents with no
        editable prompt.
        """
        container = cls._agent_prompt_container(agent)
        if container is None:
            return None
        if container.get("prompt_template") is not None:
            return cls._template_instruction(container["prompt_template"])
        prompt = container.get("prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None

    @classmethod
    def _set_agent_prompt_text(cls, agent: Dict[str, Any], text: str) -> None:
        """Inject ``text`` into the agent's active prompt slot (template instruction or prompt)."""
        container = cls._agent_prompt_container(agent)
        if container is None:
            return
        if container.get("prompt_template") is not None:
            cls._set_template_instruction(container["prompt_template"], text)
        else:
            container["prompt"] = text

    @classmethod
    def _iter_prompt_agents(cls, node: "WorkFlowNode") -> List[Tuple[str, Dict[str, Any]]]:
        """Yield ``(agent_name, agent_dict)`` for every agent on ``node`` with editable prompt text."""
        result: List[Tuple[str, Dict[str, Any]]] = []
        for agent in node.agents or []:
            if not isinstance(agent, dict) or cls._agent_prompt_text(agent) is None:
                continue
            name = agent.get("name")
            if name:
                result.append((name, agent))
        return result

    @classmethod
    def _extract_prompts_and_roles(
        cls, graph: "WorkFlowGraph"
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build ``{agent_name: prompt}`` and ``{agent_name: role}`` across all nodes' agents.

        A node may carry several agents, so units are keyed by (workflow-unique) agent name
        rather than node name. Each agent's role defaults to its own description, falling back
        to the node's description.
        """
        prompts: Dict[str, str] = {}
        roles: Dict[str, str] = {}
        for node in graph.nodes:
            for agent_name, agent in cls._iter_prompt_agents(node):
                if agent_name in prompts:
                    # Units are keyed by agent name; a duplicate would make reconstruction
                    # ambiguous (which agent does a mutated prompt belong to?).
                    raise ValueError(
                        f"Duplicate agent name {agent_name!r} found in the workflow graph. "
                        "Each agent must have a unique name so its prompt can be optimized "
                        "and reconstructed unambiguously."
                    )
                prompts[agent_name] = cls._agent_prompt_text(agent)
                role = agent.get("description") or node.description
                if role:
                    roles[agent_name] = role
        return prompts, roles

    # -- declare optimizable units -------------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name=name,
                uid=name,
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string", "description": "Full prompt text."},
                allowed_operations=[ChangeOperation.REPLACE],
                metadata={"role": self.roles[name]} if name in self.roles else {},
            )
            for name in self.prompts
        ]

    # -- snapshot / merge / reconstruct ---------------------------------------
    def take_snapshot(self) -> SnapShot:
        # Only the prompts are optimizable; everything needed to rebuild the graph
        # (structure, non-optimizable agent config, tools, llm) is carried on the live
        # adapter and re-applied in `from_snapshot`.
        return SnapShot(unit_values=dict(self.prompts))

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        new_values = dict(snapshot.unit_values)
        for change in changes:
            new_values[change.uid] = change.value  # REPLACE-only
        return SnapShot(unit_values=new_values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "WorkFlowAdapter":
        # Reconstruct the graph by injecting the snapshot's unit values into a copy of the
        # current graph, then build a fresh adapter around it. tools / llm / llm_config are
        # non-serializable runtime dependencies carried over from the live adapter.
        new_graph = self._rebuild_graph(dict(snapshot.unit_values))
        return type(self)(
            graph=new_graph,
            tools=self.tools,
            llm=self.llm,
            llm_config=self.llm_config,
            max_execution_steps=self.max_execution_steps,
        )

    def _copy_graph(self) -> "WorkFlowGraph":
        """Return an independent copy of the graph (its nodes/edges are deep-copied)."""
        from ...workflow.workflow_graph import WorkFlowGraph

        # Constructing from an existing graph deep-copies its nodes and edges while building
        # a fresh internal MultiDiGraph + lock, so the original is never mutated and we avoid
        # deep-copying the un-pickleable threading.Lock the graph holds. (cf. Evaluator.)
        return WorkFlowGraph(goal=self.graph.goal, graph=self.graph)

    def _rebuild_graph(self, unit_values: Dict[str, Any]) -> "WorkFlowGraph":
        """Deep-copy the base graph and inject the (possibly mutated) unit values into its agents.

        The base implementation consumes only the prompt units (keyed by agent name);
        subclasses that register additional units should override this, call ``super()``,
        and inject their own values into the returned graph.
        """
        new_graph = self._copy_graph()
        for node in new_graph.nodes:
            for agent_name, agent in self._iter_prompt_agents(node):
                if agent_name in unit_values:
                    self._set_agent_prompt_text(agent, unit_values[agent_name])
        new_graph.reset_graph()
        return new_graph

    # -- run the workflow ------------------------------------------------------
    def _build_workflow(self):
        """Assemble a fresh AgentManager + WorkFlow over a per-call graph copy."""
        from ...agents import AgentManager
        from ...workflow import WorkFlow

        # Run each execution on its own graph copy (not the shared self.graph), so
        # concurrent execute() / async_execute() calls on a single adapter — e.g. an
        # evaluate_fn that gathers over a dataset — never share mutable node-status
        # state. WorkFlow.execute_task mutates the graph (step()/completed()), so a
        # shared graph would let parallel runs clobber each other.
        graph = self._copy_graph()
        graph.reset_graph()
        agent_manager = AgentManager()
        agent_manager.add_agents_from_workflow(graph, llm_config=self.llm_config, tools=self.tools)
        return WorkFlow(
            graph=graph,
            agent_manager=agent_manager,
            llm=self.llm,
            max_execution_steps=self.max_execution_steps,
        )

    def execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        # Suppress the workflow's per-step INFO logs so an evaluate_fn running many
        # executions doesn't flood the console. suppress_logger_info is contextvar-scoped
        # to this thread / asyncio-task, so it stays correct under concurrent execution.
        from ...core.callbacks import suppress_logger_info

        workflow = self._build_workflow()
        with suppress_logger_info():
            return workflow.execute(inputs=dict(inputs or {}), **kwargs)

    async def async_execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        from ...core.callbacks import suppress_logger_info

        workflow = self._build_workflow()
        with suppress_logger_info():
            return await workflow.async_execute(inputs=dict(inputs or {}), **kwargs)
