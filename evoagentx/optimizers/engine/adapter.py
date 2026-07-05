import abc
import os
import shutil
from dataclasses import dataclass, field
from typing import final, List, Any, Optional, Dict, Literal, Iterable

from .base import ChangeOperation, EvaluationResult, OptimizationUnit, OptimizationUnitType, SnapShot, UnitChange, ValidationResult


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
