import abc
import asyncio
import inspect
import os
import time
from typing import Any, Awaitable, Callable, ClassVar, FrozenSet, Optional, List, Dict, Literal, Set, Tuple, Iterable, Union

from ...core.callbacks import silence_cost_logs
from ...core.logging import logger
from ...models.model_utils import cost_manager
from .base import (
    BASELINE_TRIAL_ID,
    EvaluationResult,
    OptimizationProposal,
    OptimizationRunState,
    OptimizationUnit,
    OptimizationUnitType,
    SnapShot,
    TrialRecord,
    ValidationResult,
)
from .adapter import ProgramAdapter, ApplyResult, TrialContext, TrialWorkspace
from .objective import Objective
from .utils import OptimizationProgress, format_optimization_report, format_trial_progress_message

EvaluationReturn = Union[Dict[str, Any], EvaluationResult]

_SYNC_CONCURRENT_ERROR = (
    "Synchronous optimize/run_batch does not support execution_mode='concurrent'. "
    "Use execution_mode='sequential' with optimize(), or use "
    "async_optimize(..., execution_mode='concurrent') for concurrent evaluations."
)


def _sync_best(state: OptimizationRunState, objective: Objective) -> None:
    """Update state.best_snapshot_id and state.best_metrics from objective."""
    best_record = objective.select_best_trial_record(state.trial_records)
    if best_record is not None:
        state.best_snapshot_id = best_record.snapshot_id
        state.best_metrics = best_record.metrics


def _validate_metrics(metrics: Any, trial_id: int) -> None:
    """Raise TypeError if evaluate_fn did not return a dict."""
    if not isinstance(metrics, dict):
        raise TypeError(
            f"evaluate_fn must return a dict, got {type(metrics).__name__!r} "
            f"(trial_id={trial_id})"
        )


def _normalize_evaluation_result(result: Any, trial_id: int) -> EvaluationResult:
    """
    Normalize legacy metric dictionaries and structured evaluation results.

    Existing evaluators can keep returning `dict` metrics. Newer evaluators may
    return `EvaluationResult` to attach traces, artifacts, and metadata without
    overloading objective-facing metrics.
    """
    if isinstance(result, EvaluationResult):
        _validate_metrics(result.metrics, trial_id)
        return result
    _validate_metrics(result, trial_id)
    return EvaluationResult(metrics=result)


def _normalize_validation_results(results: Any, trial_id: int) -> List[ValidationResult]:
    """Validate and normalize adapter validation output."""
    if results is None:
        return []
    if isinstance(results, ValidationResult):
        return [results]
    if not isinstance(results, list):
        raise TypeError(
            f"validate_trial must return a ValidationResult, a list of ValidationResult objects, "
            f"or None; got {type(results).__name__!r} (trial_id={trial_id})"
        )
    if not all(isinstance(result, ValidationResult) for result in results):
        raise TypeError(
            f"validate_trial must return only ValidationResult objects (trial_id={trial_id})"
        )
    return results


def _validation_failure_message(results: List[ValidationResult]) -> Optional[str]:
    failures = [result for result in results if result.status == "failed"]
    if not failures:
        return None
    return "; ".join(
        f"{failure.validator}: {failure.message or 'validation failed'}"
        for failure in failures
    )


_CONTEXT_PARAM_NAMES = frozenset({"ctx", "context", "trial_context"})


def _accepts_trial_context(evaluate_fn: Callable) -> bool:
    """Return True when `evaluate_fn` should be called as (adapter, context).

    Deliberately opt-in rather than "any second positional parameter": an evaluator like
    ``def evaluate(adapter, split="dev")`` must NOT silently receive a TrialContext as
    ``split``. The context is passed only when the second positional parameter is named
    ``ctx`` / ``context`` / ``trial_context`` or is annotated as ``TrialContext``, or when
    the callable takes ``*args`` after a single positional parameter.
    """
    try:
        sig = inspect.signature(evaluate_fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        second = positional[1]
        if second.name in _CONTEXT_PARAM_NAMES:
            return True
        annotation = second.annotation
        if annotation is TrialContext:
            return True
        # String annotations (e.g. under `from __future__ import annotations`).
        if isinstance(annotation, str) and annotation.split(".")[-1].strip() == "TrialContext":
            return True
        return False
    return any(p.kind == p.VAR_POSITIONAL for p in params)


def _call_evaluate_fn(evaluate_fn: Callable, adapter: ProgramAdapter, context: TrialContext) -> Any:
    """Invoke `evaluate_fn` with or without the TrialContext, based on its signature.

    Works for both sync and async evaluators: the async caller simply awaits the
    returned coroutine.
    """
    if _accepts_trial_context(evaluate_fn):
        return evaluate_fn(adapter, context)
    return evaluate_fn(adapter)


def _prepare_trial_dirs(
    adapter: ProgramAdapter,
    snapshot: SnapShot,
    trial_id: int,
    trials_root: Optional[str],
    workspace_root: Optional[str],
    keep_trial_workspaces: bool,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Optional[str], Optional[TrialWorkspace]]:
    """Create the persistent trial directory and (when enabled) the adapter workspace.

    Two distinct locations with different lifecycles:
      - trial_dir (``<trials_root>/trial_XXXX_<snapshot_id>``): the trial's archive
        directory, created whenever the run persists state (`save_dir`/`resume_from`).
        Handed to `evaluate_fn` via TrialContext so evaluation results land in a
        per-trial location; never cleaned up by the engine.
      - workspace: the adapter's file sandbox, created only when the adapter declares
        `uses_workspace` (placed inside trial_dir) or an explicit `workspace_root`
        forces it. Subject to `keep_trial_workspaces` cleanup.
    """
    trial_dir: Optional[str] = None
    if trials_root is not None:
        trial_dir = os.path.join(trials_root, f"trial_{trial_id:04d}_{snapshot.snapshot_id}")
        os.makedirs(trial_dir, exist_ok=True)

    if workspace_root is not None:
        workspace_dir = os.path.join(workspace_root, f"trial_{trial_id:04d}_{snapshot.snapshot_id}")
    elif adapter.uses_workspace and trial_dir is not None:
        workspace_dir = os.path.join(trial_dir, "workspace")
    else:
        adapter.bind_workspace(None)
        return trial_dir, None

    workspace = TrialWorkspace.create(
        root_dir=workspace_dir,
        trial_id=trial_id,
        source_snapshot_id=snapshot.snapshot_id,
        metadata=metadata,
        keep=keep_trial_workspaces,
    )
    adapter.bind_workspace(workspace)
    adapter.prepare_workspace(workspace, snapshot, **kwargs)
    return trial_dir, workspace


def _run_trial(
    adapter: ProgramAdapter,
    base_snapshot: SnapShot,
    proposal: OptimizationProposal,
    trial_id: int,
    evaluate_fn: Callable[..., EvaluationReturn],
    trials_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    keep_trial_workspaces: bool = True,
    isolate_snapshots: bool = True,
    **kwargs,
) -> Tuple[Optional[SnapShot], TrialRecord]:
    """
    Apply a proposal and evaluate it synchronously.

    Returns (snapshot, record): snapshot is None when the trial failed.
    Does not touch shared state — callers are responsible for merging the result.
    """
    if base_snapshot is None:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=f"source_snapshot_id '{proposal.source_snapshot_id}' not found in run state",
        )
    base = base_snapshot if not isolate_snapshots else base_snapshot.model_copy(deep=True)
    result: ApplyResult = adapter.apply(base, proposal.changes, **kwargs)
    if not result.ok:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=result.error,
        )
    trial_dir: Optional[str] = None
    workspace: Optional[TrialWorkspace] = None
    validation_results: List[ValidationResult] = []
    try:
        trial_dir, workspace = _prepare_trial_dirs(
            result.adapter,
            result.snapshot,
            trial_id,
            trials_root,
            workspace_root,
            keep_trial_workspaces,
            metadata=proposal.metadata,
            **kwargs,
        )
        validation_results = _normalize_validation_results(
            result.adapter.validate_trial(
                result.snapshot,
                proposal.changes,
                workspace=workspace,
                **kwargs,
            ),
            trial_id,
        )
        validation_error = _validation_failure_message(validation_results)
        if validation_error is not None:
            return None, TrialRecord(
                trial_id=trial_id,
                changes=proposal.changes,
                source_snapshot_id=proposal.source_snapshot_id,
                status="failed",
                validation_results=validation_results,
                trial_dir=trial_dir,
                workspace_dir=workspace.root_dir if workspace is not None else None,
                error=validation_error,
            )
        context = TrialContext(
            trial_id=trial_id,
            trial_dir=trial_dir,
            workspace=workspace,
            metadata=proposal.metadata or {},
        )
        evaluation = _normalize_evaluation_result(
            _call_evaluate_fn(evaluate_fn, result.adapter, context), trial_id
        )
        captured_snapshot = result.adapter.capture_after_eval(
            snapshot=result.snapshot,
            evaluation=evaluation,
            changes=proposal.changes,
            workspace=workspace,
            **kwargs,
        )
        if captured_snapshot is not None:
            result.adapter._validate_snapshot(captured_snapshot, context="captured trial snapshot")
    except Exception as exc:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            validation_results=validation_results,
            trial_dir=trial_dir,
            workspace_dir=workspace.root_dir if workspace is not None else None,
            error=str(exc),
        )
    finally:
        if workspace is not None:
            workspace.cleanup()
    final_snapshot = captured_snapshot if captured_snapshot is not None else result.snapshot
    return final_snapshot, TrialRecord(
        trial_id=trial_id,
        changes=proposal.changes,
        source_snapshot_id=proposal.source_snapshot_id,
        status="completed",
        snapshot_id=final_snapshot.snapshot_id,
        metrics=evaluation.metrics,
        traces=evaluation.traces,
        artifacts=evaluation.artifacts,
        metadata=evaluation.metadata,
        validation_results=validation_results,
        trial_dir=trial_dir,
        workspace_dir=workspace.root_dir if workspace is not None else None,
    )


async def _async_run_trial(
    adapter: ProgramAdapter,
    base_snapshot: SnapShot,
    proposal: OptimizationProposal,
    trial_id: int,
    evaluate_fn: Callable[..., Awaitable[EvaluationReturn]],
    sem: Optional[asyncio.Semaphore] = None,
    trials_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    keep_trial_workspaces: bool = True,
    isolate_snapshots: bool = True,
    **kwargs,
) -> Tuple[Optional[SnapShot], TrialRecord]:
    """
    Apply a proposal and evaluate it asynchronously.

    `sem` throttles concurrent evaluate calls when set; pass None for uncapped concurrency.
    Returns (snapshot, record): snapshot is None when the trial failed.
    Does not touch shared state — callers are responsible for merging the result.
    """
    if base_snapshot is None:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=f"source_snapshot_id '{proposal.source_snapshot_id}' not found in run state",
        )
    base = base_snapshot if not isolate_snapshots else base_snapshot.model_copy(deep=True)
    result: ApplyResult = adapter.apply(base, proposal.changes, **kwargs)
    if not result.ok:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            error=result.error,
        )
    trial_dir: Optional[str] = None
    workspace: Optional[TrialWorkspace] = None
    validation_results: List[ValidationResult] = []
    try:
        trial_dir, workspace = _prepare_trial_dirs(
            result.adapter,
            result.snapshot,
            trial_id,
            trials_root,
            workspace_root,
            keep_trial_workspaces,
            metadata=proposal.metadata,
            **kwargs,
        )
        validation_results = _normalize_validation_results(
            await result.adapter.async_validate_trial(
                result.snapshot,
                proposal.changes,
                workspace=workspace,
                **kwargs,
            ),
            trial_id,
        )
        validation_error = _validation_failure_message(validation_results)
        if validation_error is not None:
            return None, TrialRecord(
                trial_id=trial_id,
                changes=proposal.changes,
                source_snapshot_id=proposal.source_snapshot_id,
                status="failed",
                validation_results=validation_results,
                trial_dir=trial_dir,
                workspace_dir=workspace.root_dir if workspace is not None else None,
                error=validation_error,
            )
        context = TrialContext(
            trial_id=trial_id,
            trial_dir=trial_dir,
            workspace=workspace,
            metadata=proposal.metadata or {},
        )
        if sem is not None:
            async with sem:
                evaluation = _normalize_evaluation_result(
                    await _call_evaluate_fn(evaluate_fn, result.adapter, context), trial_id
                )
        else:
            evaluation = _normalize_evaluation_result(
                await _call_evaluate_fn(evaluate_fn, result.adapter, context), trial_id
            )
        captured_snapshot = await result.adapter.async_capture_after_eval(
            snapshot=result.snapshot,
            evaluation=evaluation,
            changes=proposal.changes,
            workspace=workspace,
            **kwargs,
        )
        if captured_snapshot is not None:
            result.adapter._validate_snapshot(captured_snapshot, context="captured trial snapshot")
    except Exception as exc:
        return None, TrialRecord(
            trial_id=trial_id,
            changes=proposal.changes,
            source_snapshot_id=proposal.source_snapshot_id,
            status="failed",
            validation_results=validation_results,
            trial_dir=trial_dir,
            workspace_dir=workspace.root_dir if workspace is not None else None,
            error=str(exc),
        )
    finally:
        if workspace is not None:
            workspace.cleanup()
    final_snapshot = captured_snapshot if captured_snapshot is not None else result.snapshot
    return final_snapshot, TrialRecord(
        trial_id=trial_id,
        changes=proposal.changes,
        source_snapshot_id=proposal.source_snapshot_id,
        status="completed",
        snapshot_id=final_snapshot.snapshot_id,
        metrics=evaluation.metrics,
        traces=evaluation.traces,
        artifacts=evaluation.artifacts,
        metadata=evaluation.metadata,
        validation_results=validation_results,
        trial_dir=trial_dir,
        workspace_dir=workspace.root_dir if workspace is not None else None,
    )


def _merge_outcomes(
    state: OptimizationRunState,
    outcomes: List[Tuple[Optional[SnapShot], TrialRecord]],
    objective: Objective,
) -> OptimizationRunState:
    """Append (snapshot, record) pairs to state and sync the best pointer."""
    for snapshot, record in outcomes:
        if snapshot is not None:
            state.snapshots.append(snapshot)
        state.trial_records.append(record)
    _sync_best(state, objective)
    return state


class TrialRuntime:

    def __init__(self, adapter: ProgramAdapter) -> None:
        self.adapter = adapter

    def run(
        self,
        state: OptimizationRunState,
        proposal: OptimizationProposal,
        evaluate_fn: Callable[..., EvaluationReturn],
        objective: Objective,
        trials_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> OptimizationRunState:
        """
        Execute a single optimization trial and return the updated run state.

        Calls `_run_trial` to apply the proposal and evaluate it, then merges the
        result into `state`. `state.current_step` is NOT incremented here; the
        caller (`Optimizer.optimize`) is responsible for that.

        Args:
            state: The current optimization run state.
            proposal: The proposed changeset and its source snapshot.
            evaluate_fn: Callable that executes the adapted program and returns metrics.
            objective: Used to update `best_snapshot_id` after the trial completes.
            **kwargs: Forwarded to `adapter.apply`.

        Returns:
            Updated OptimizationRunState with the new snapshot, trial record, and
            (if the trial succeeded) a potentially updated best_snapshot_id.
        """
        base_snapshot = state.get_snapshot_by_id(proposal.source_snapshot_id)
        trial_id = max((r.trial_id for r in state.trial_records), default=-1) + 1
        outcome = _run_trial(
            self.adapter,
            base_snapshot,
            proposal,
            trial_id,
            evaluate_fn,
            trials_root=trials_root,
            workspace_root=workspace_root,
            keep_trial_workspaces=keep_trial_workspaces,
            isolate_snapshots=isolate_snapshots,
            **kwargs,
        )
        return _merge_outcomes(state, [outcome], objective)

    async def async_run(
        self,
        state: OptimizationRunState,
        proposal: OptimizationProposal,
        evaluate_fn: Callable[..., Awaitable[EvaluationReturn]],
        objective: Objective,
        trials_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> OptimizationRunState:
        """
        Async variant of `run`. Identical pipeline but `evaluate_fn` is awaited.

        Args:
            state: The current optimization run state.
            proposal: The proposed changeset and its source snapshot.
            evaluate_fn: Async callable that executes the adapted program and returns metrics.
            objective: Used to update `best_snapshot_id` after the trial completes.
            **kwargs: Forwarded to `adapter.apply`.

        Returns:
            Updated OptimizationRunState (see `run` for full pipeline description).
        """
        base_snapshot = state.get_snapshot_by_id(proposal.source_snapshot_id)
        trial_id = max((r.trial_id for r in state.trial_records), default=-1) + 1
        outcome = await _async_run_trial(
            self.adapter,
            base_snapshot,
            proposal,
            trial_id,
            evaluate_fn,
            trials_root=trials_root,
            workspace_root=workspace_root,
            keep_trial_workspaces=keep_trial_workspaces,
            isolate_snapshots=isolate_snapshots,
            **kwargs,
        )
        return _merge_outcomes(state, [outcome], objective)

    def run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[..., EvaluationReturn],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        trials_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> OptimizationRunState:
        """
        Run a batch of proposals and return the updated state.

        Sequential mode evaluates proposals one after another; each trial sees the
        state already updated by its predecessors.

        Synchronous batch execution is intentionally sequential. Use
        `async_run_batch(..., execution_mode="concurrent")` when proposals should
        be evaluated concurrently.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Sync callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: Must be "sequential" for synchronous execution.
            max_workers: Use None or 1 in sequential mode.
            **kwargs: Forwarded to each trial.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        if execution_mode == "concurrent":
            raise ValueError(_SYNC_CONCURRENT_ERROR)

        for proposal in proposals:
            state = self.run(
                state,
                proposal,
                evaluate_fn,
                objective,
                trials_root=trials_root,
                workspace_root=workspace_root,
                keep_trial_workspaces=keep_trial_workspaces,
                isolate_snapshots=isolate_snapshots,
                **kwargs,
            )
        return state

    async def async_run_batch(
        self,
        state: OptimizationRunState,
        proposals: List[OptimizationProposal],
        evaluate_fn: Callable[..., Awaitable[EvaluationReturn]],
        objective: Objective,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        trials_root: Optional[str] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> OptimizationRunState:
        """
        Async variant of `run_batch`.

        Sequential mode evaluates proposals one after another, so each trial sees the
        state already updated by its predecessors.

        Concurrent mode launches all trials via `asyncio.gather` from the same read-only
        snapshot of the current state, then merges results in order. `max_workers` caps
        the number of coroutines that may call `evaluate_fn` simultaneously via a semaphore.

        Args:
            state: Current optimization run state.
            proposals: List of proposals to evaluate.
            evaluate_fn: Async callable returning evaluation metrics.
            objective: Used to update `best_snapshot_id` after the batch.
            execution_mode: "sequential" (default) or "concurrent".
            max_workers: Maximum number of coroutines running concurrently in concurrent mode.
                         Required and must be greater than 1 in concurrent mode.
                         Use None or 1 in sequential mode.
            **kwargs: Forwarded to each trial.

        Returns:
            Updated OptimizationRunState after all proposals have been evaluated.
        """
        if execution_mode == "sequential":
            for proposal in proposals:
                state = await self.async_run(
                    state,
                    proposal,
                    evaluate_fn,
                    objective,
                    trials_root=trials_root,
                    workspace_root=workspace_root,
                    keep_trial_workspaces=keep_trial_workspaces,
                    isolate_snapshots=isolate_snapshots,
                    **kwargs,
                )
            return state
        if not proposals:
            return state

        # concurrent: all trials launched together; semaphore throttles evaluate calls.
        base_trial_id = max((r.trial_id for r in state.trial_records), default=-1) + 1
        effective_workers = min(max_workers, len(proposals))
        sem = asyncio.Semaphore(effective_workers)

        outcomes = await asyncio.gather(*[
            _async_run_trial(
                self.adapter,
                state.get_snapshot_by_id(proposal.source_snapshot_id),
                proposal,
                base_trial_id + idx,
                evaluate_fn,
                sem,
                trials_root=trials_root,
                workspace_root=workspace_root,
                keep_trial_workspaces=keep_trial_workspaces,
                isolate_snapshots=isolate_snapshots,
                **kwargs,
            )
            for idx, proposal in enumerate(proposals)
        ])

        return _merge_outcomes(state, outcomes, objective)


class Optimizer(abc.ABC):
    """
    Abstract base class for all optimizers.

    Subclasses MUST define `supported_unit_types` as a class-level frozenset::

        class MyOptimizer(Optimizer):
            supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
                OptimizationUnitType.PROMPT,
            })

    Defining it as an instance property is not allowed: `Optimizer.__init__` reads
    `supported_unit_types` before the subclass constructor body has run, so any property
    that depends on instance state will observe a partially-constructed object.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "supported_unit_types" in cls.__dict__:
            val = cls.__dict__["supported_unit_types"]
            if not isinstance(val, frozenset):
                raise TypeError(
                    f"{cls.__name__}.supported_unit_types must be a frozenset, "
                    f"got {type(val).__name__!r}"
                )

    def __init__(
        self,
        adapter: ProgramAdapter,
        runtime: Optional[TrialRuntime] = None,
        target_unit_uids: Optional[Iterable[str]] = None,
        target_unit_types: Optional[Union[OptimizationUnitType, str, Iterable[Union[OptimizationUnitType, str]]]] = None,
        **kwargs
    ) -> None:

        if not isinstance(adapter, ProgramAdapter):
            raise TypeError(f"Expected adapter to be an instance of ProgramAdapter, got {type(adapter)}")

        self.adapter = adapter
        self.runtime = runtime or TrialRuntime(adapter)
        self.kwargs = kwargs

        if not isinstance(getattr(type(self), "supported_unit_types", None), frozenset):
            raise TypeError(
                f"{self.__class__.__name__}.supported_unit_types must be defined as a "
                f"frozenset[OptimizationUnitType] class attribute"
            )

        cls = type(self)
        if (
            cls.propose is Optimizer.propose
            and cls.batch_propose is Optimizer.batch_propose
            and cls.async_batch_propose is Optimizer.async_batch_propose
        ):
            raise TypeError(
                f"{cls.__name__} must override at least one of propose(), batch_propose(), "
                f"or async_batch_propose(); none was provided."
            )

        self.target_units = self._select_target_units(
            target_unit_uids=target_unit_uids,
            target_unit_types=target_unit_types,
        )
        self.target_units_by_uid = {unit.uid: unit for unit in self.target_units}

        unsupported = {unit.unit_type for unit in self.target_units if unit.unit_type not in self.supported_unit_types}
        if unsupported:
            raise TypeError(
                f"{self.__class__.__name__} supports unit type(s): "
                f"{{{', '.join(t.value for t in sorted(self.supported_unit_types, key=lambda t: t.value))}}}; "
                f"but the selected target unit(s) include unsupported type(s): "
                f"{{{', '.join(t.value for t in sorted(unsupported, key=lambda t: t.value))}}}."
            )

    @staticmethod
    def _coerce_unit_type(unit_type: Union[OptimizationUnitType, str]) -> OptimizationUnitType:
        if isinstance(unit_type, OptimizationUnitType):
            return unit_type
        try:
            return OptimizationUnitType(unit_type)
        except ValueError as exc:
            raise ValueError(f"Unknown OptimizationUnitType: {unit_type!r}") from exc

    @classmethod
    def _coerce_unit_types(
        cls,
        unit_types: Optional[Union[OptimizationUnitType, str, Iterable[Union[OptimizationUnitType, str]]]],
    ) -> Optional[FrozenSet[OptimizationUnitType]]:
        if unit_types is None:
            return None
        if isinstance(unit_types, (OptimizationUnitType, str)):
            return frozenset({cls._coerce_unit_type(unit_types)})
        return frozenset(cls._coerce_unit_type(unit_type) for unit_type in unit_types)

    def _select_target_units(
        self,
        target_unit_uids: Optional[Iterable[str]] = None,
        target_unit_types: Optional[Union[OptimizationUnitType, str, Iterable[Union[OptimizationUnitType, str]]]] = None,
    ) -> List[OptimizationUnit]:
        """
        Select the units this optimizer is allowed to propose changes for.

        By default, a mixed adapter is allowed: the optimizer targets all adapter
        units whose type is in `supported_unit_types` and ignores unrelated units.
        Passing explicit `target_unit_uids` and/or `target_unit_types` narrows that
        set further.
        """
        requested_types: Optional[FrozenSet[OptimizationUnitType]] = self._coerce_unit_types(target_unit_types)
        if requested_types is None and target_unit_uids is None:
            requested_types = self.supported_unit_types
        requested_uids = set(target_unit_uids) if target_unit_uids is not None else None

        unknown_uids = requested_uids - {unit.uid for unit in self.adapter.units} if requested_uids is not None else set()
        if unknown_uids:
            raise ValueError(f"target_unit_uids reference unknown unit uid(s): {sorted(unknown_uids)}")

        target_units = self.adapter.select_units(unit_types=requested_types, uids=requested_uids)
        if not target_units:
            requested_type_values = sorted(t.value for t in requested_types) if requested_types is not None else None
            raise ValueError(
                f"{self.__class__.__name__} found no target units. "
                f"requested types={requested_type_values}, "
                f"requested uids={sorted(requested_uids) if requested_uids is not None else None}."
            )
        return target_units

    @property
    def target_unit_uids(self) -> List[str]:
        """Return target unit uids in adapter registration order."""
        return [unit.uid for unit in self.target_units]

    def _check_adapter_fingerprint(self, state: OptimizationRunState, context: str) -> None:
        """Raise if the saved run's adapter fingerprint does not match the current adapter."""
        current_fingerprint = self.adapter.fingerprint()
        if state.adapter_fingerprint is not None and state.adapter_fingerprint != current_fingerprint:
            raise ValueError(
                f"Cannot {context}: adapter fingerprint does not match the saved run state."
            )

    def _init_run_state(
        self,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> OptimizationRunState:
        """Load or create OptimizationRunState and seed the initial snapshot and baseline record."""
        if resume_from:
            state = OptimizationRunState.load_state(resume_from)
            self._check_adapter_fingerprint(state, context="resume optimization")
            # When both resume_from and save_dir are given and differ, read the state
            # from resume_from but redirect all subsequent checkpoints to save_dir,
            # leaving the original run untouched.
            #
            # Note: only the serialized run state (snapshots, trial records, optimizer
            # state) is relocated. On-disk artifacts that records merely reference by
            # path — `TrialRecord.workspace_dir` and any file paths stored in
            # `artifacts` — still live under the original run's directory and are NOT
            # copied into save_dir. For file-backed adapters this means the relocated
            # state.json points at files outside save_dir.
            if save_dir is not None and save_dir != state.save_dir:
                logger.warning(
                    f"resume_from ('{state.save_dir}') and save_dir ('{save_dir}') differ; "
                    f"resuming from the saved state but writing future checkpoints to save_dir."
                )
                state.save_dir = save_dir
                state.save_state()
        else:
            state = OptimizationRunState(save_dir=save_dir or "./")
            state.adapter_fingerprint = self.adapter.fingerprint()

        # Seed the initial snapshot so propose() always has a valid source_snapshot_id,
        # including the first trial of a fresh run.
        if not state.snapshots:
            initial_snapshot = self.adapter.take_snapshot()
            self.adapter._validate_snapshot(initial_snapshot, context="initial snapshot")
            state.snapshots.append(initial_snapshot)
            state.trial_records.append(TrialRecord(
                trial_id=BASELINE_TRIAL_ID,
                changes=[],
                source_snapshot_id=initial_snapshot.snapshot_id,
                status="completed",
                snapshot_id=initial_snapshot.snapshot_id,
                metrics=None,
            ))

        return state

    @staticmethod
    def _validate_optimize_args(
        objective: Any,
        max_trials: Any,
        execution_mode: Any,
        max_workers: Any,
        allow_concurrent: bool,
    ) -> Tuple[Literal["sequential", "concurrent"], Optional[int]]:
        """Validate optimize / async_optimize arguments and return normalized execution settings."""
        if not isinstance(objective, Objective):
            raise TypeError(f"`objective` must be an instance of Objective, got {type(objective).__name__!r}")
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError(f"`max_trials` must be a positive integer, got {max_trials}")

        if execution_mode not in ("sequential", "concurrent"):
            raise ValueError(f"execution_mode must be 'sequential' or 'concurrent', got {execution_mode!r}")

        if max_workers is not None and (
            isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1
        ):
            raise ValueError(f"max_workers must be a positive integer or None, got {max_workers!r}")

        if execution_mode == "sequential":
            if max_workers not in (None, 1):
                raise ValueError(
                    "max_workers is only valid with execution_mode='concurrent'; "
                    "use max_workers=None or 1 for sequential execution."
                )
            return "sequential", None

        if not allow_concurrent:
            raise ValueError(_SYNC_CONCURRENT_ERROR)
        if max_workers is None:
            raise ValueError("max_workers must be explicitly provided when execution_mode='concurrent'.")
        if max_workers <= 1:
            raise ValueError("max_workers must be greater than 1 when execution_mode='concurrent'.")

        return "concurrent", max_workers

    @staticmethod
    def _resolve_trials_root(
        state: OptimizationRunState,
        save_dir: Optional[str],
        resume_from: Optional[str],
    ) -> Optional[str]:
        """
        Resolve the root under which per-trial archive directories are created.

        Whenever the run persists state (`save_dir` or `resume_from` given), every trial
        (and the baseline) gets a durable directory `<save_dir>/trials/trial_XXXX_<snap>`,
        handed to `evaluate_fn` through `TrialContext.trial_dir` so evaluation results
        from different trials never collide. Runs without persistence get no trial dirs.

        The adapter workspace (file sandbox) is separate: it is created inside the trial
        dir only for adapters that declare `uses_workspace`, or wherever an explicit
        `workspace_root` argument forces it (see `_prepare_trial_dirs`).
        """
        if save_dir is not None or resume_from is not None:
            return os.path.join(state.save_dir or "./", "trials")
        return None

    def on_run_start(
        self,
        _state: OptimizationRunState,
        _objective: Objective,
        **_kwargs,
    ) -> None:
        """
        Hook called after baseline evaluation and best syncing, before trial proposals.

        Optimizers with persistent algorithm state can initialize
        `state.optimizer_state` here without reimplementing `optimize`.
        """
        pass

    async def async_on_run_start(
        self,
        state: OptimizationRunState,
        objective: Objective,
        **kwargs,
    ) -> None:
        """Async variant of `on_run_start`."""
        self.on_run_start(state, objective, **kwargs)

    def format_trial_progress(
        self,
        state: OptimizationRunState,
        record: TrialRecord,
        objective: Objective,
        baseline_record: Optional[TrialRecord],
        **_kwargs,
    ) -> Optional[str]:
        """
        Return the progress message for a completed baseline/trial record.

        Subclasses can override this to customize optimizer-specific progress output.
        Return None to suppress output for a record.
        """
        return format_trial_progress_message(
            record=record,
            objective=objective,
            baseline_record=baseline_record,
            best_snapshot_id=state.best_snapshot_id,
        )

    def _report_trial_progress(
        self,
        progress: OptimizationProgress,
        state: OptimizationRunState,
        record: TrialRecord,
        objective: Objective,
        baseline_record: Optional[TrialRecord],
        **kwargs,
    ) -> None:
        message = self.format_trial_progress(state, record, objective, baseline_record, **kwargs)
        if message is not None:
            progress.write(message)

    def format_optimization_report(
        self,
        state: OptimizationRunState,
        objective: Objective,
        baseline_record: Optional[TrialRecord],
        elapsed_seconds: float,
        start_step: int,
        max_trials: int,
        total_cost: Optional[float] = None,
        **_kwargs,
    ) -> Optional[str]:
        """
        Return the final optimization report emitted after optimize finishes.

        Subclasses can override this to customize report content. Return None to
        suppress the final report.
        """
        return format_optimization_report(
            state=state,
            objective=objective,
            baseline_record=baseline_record,
            elapsed_seconds=elapsed_seconds,
            start_step=start_step,
            max_trials=max_trials,
            total_cost=total_cost,
        )

    def _report_optimization_summary(
        self,
        progress: OptimizationProgress,
        state: OptimizationRunState,
        objective: Objective,
        baseline_record: Optional[TrialRecord],
        elapsed_seconds: float,
        start_step: int,
        max_trials: int,
        total_cost: Optional[float] = None,
        **kwargs,
    ) -> None:
        message = self.format_optimization_report(
            state,
            objective,
            baseline_record,
            elapsed_seconds,
            start_step,
            max_trials,
            total_cost=total_cost,
            **kwargs,
        )
        if message is not None:
            progress.write(message)

    def observe(
        self,
        _state: OptimizationRunState,
        _trial_records: List[TrialRecord],
        _objective: Objective,
        **_kwargs,
    ) -> None:
        """
        Hook called after a batch has been evaluated and merged into state.

        Population, archive, Bayesian-search, or gradient-like optimizers can use
        this to update `state.optimizer_state` before the next `propose` call.
        """
        pass

    async def async_observe(
        self,
        state: OptimizationRunState,
        trial_records: List[TrialRecord],
        objective: Objective,
        **kwargs,
    ) -> None:
        """Async variant of `observe`."""
        self.observe(state, trial_records, objective, **kwargs)

    def should_stop(
        self,
        _state: OptimizationRunState,
        _objective: Objective,
        **_kwargs,
    ) -> bool:
        """Return True to stop before exhausting `max_trials`."""
        return False

    def retained_snapshot_ids(
        self,
        _state: OptimizationRunState,
        _objective: Objective,
        **_kwargs,
    ) -> Optional[Set[str]]:
        """Return the set of snapshot_ids to keep after each batch, or None to keep all.

        Default returns None: every snapshot produced during the run is retained (the
        historical behavior). Online-accumulation optimizers, whose program state grows
        each trial and whose product is the *latest* snapshot rather than a search tree,
        can override this to bound memory/disk. For a linear online run, `return set()`
        keeps only the engine floor (baseline, current best, latest snapshot), which is all
        such a run needs. The engine always preserves that floor, so returning a set can
        never strand the best result or the branch head. Pair this with
        `optimize(..., isolate_snapshots=False)` to also skip the per-trial deep copy of
        the (growing) source snapshot.
        """
        return None

    def serialize_optimizer_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of this optimizer's persistent state.

        `state.optimizer_state` is written to disk on every checkpoint, so anything an
        algorithm wants to survive a resume must be JSON-serializable (str, int, float,
        bool, None, list, dict). Many real optimizers keep *live* non-serializable
        objects — an optuna study, a dspy module, a sampler, a vector index, RNG state.
        Keep those as instance attributes on the optimizer and override this hook to
        reduce them to plain data (e.g. the study's trial history, the RNG seed/state)
        right before each checkpoint. Pair it with `load_optimizer_state` to rebuild the
        live objects on resume.

        The returned dict replaces `state.optimizer_state` prior to saving. The default
        persists `state.optimizer_state` unchanged, so optimizers that only ever store
        plain data in `state.optimizer_state` need not override this.

        Args:
            state: The current optimization run state.

        Returns:
            A JSON-serializable dict to persist as `state.optimizer_state`.
        """
        return state.optimizer_state

    def load_optimizer_state(self, state: OptimizationRunState) -> None:
        """
        Rebuild live optimizer state from the persisted `state.optimizer_state`.

        Called once at the start of `optimize` / `async_optimize`, after the run state
        is created or resumed and before baseline evaluation. On a fresh run
        `state.optimizer_state` is empty; on a resumed run it holds whatever
        `serialize_optimizer_state` last wrote. Override to reconstruct non-serializable
        instance attributes (samplers, studies, indices, RNG state) from that dict so the
        algorithm continues exactly where it left off. The default is a no-op.

        Args:
            state: The freshly created or resumed optimization run state.
        """
        return None

    def _checkpoint(self, state: OptimizationRunState) -> None:
        """Serialize optimizer state into the run state and persist it to disk."""
        serialized = self.serialize_optimizer_state(state)
        if serialized is not None:
            state.optimizer_state = serialized
        state.save_state()

    def _prune_retained_snapshots(self, state: OptimizationRunState, objective: Objective, **kwargs) -> None:
        """Apply the optimizer's snapshot-retention policy, if any, after a batch.

        Calls `retained_snapshot_ids`; when it returns a set, prunes `state.snapshots`
        down to that set unioned with an engine-guaranteed floor (baseline, current best,
        and the latest snapshot) so the run can always resolve its best result and branch
        from a valid head. When it returns None (the default) all snapshots are kept.
        """
        keep = self.retained_snapshot_ids(state, objective, **kwargs)
        if keep is None:
            return
        mandatory: Set[str] = set()
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.snapshot_id:
            mandatory.add(baseline_record.snapshot_id)
        if state.best_snapshot_id:
            mandatory.add(state.best_snapshot_id)
        if state.snapshots:
            mandatory.add(state.snapshots[-1].snapshot_id)
        state.prune_snapshots(set(keep) | mandatory)

    def finalize(
        self,
        state: OptimizationRunState,
        objective: Objective,
        best_adapter: Optional[ProgramAdapter],
    ) -> Any:
        """
        Produce the value returned by `optimize` / `async_optimize`.

        The engine drives a "propose candidate -> evaluate -> keep best single snapshot"
        loop, so by default it returns the adapter reconstructed from the best snapshot.
        That single-best view is lossy for algorithms whose real product is something
        else: a quality-diversity archive (MAP-Elites), a Pareto front, an evolved
        population or skill library, or the final *accumulated* state of an online memory
        run (where you want the latest snapshot, not the highest-scoring one). Override
        this hook to assemble and return that product instead — typically from
        `state.optimizer_state`, `state.snapshots`, `state.trial_records`, and/or
        `best_adapter`.

        `best_adapter` is the adapter loaded from `state.best_snapshot_id`, or None when
        no completed trial produced metrics (e.g. every trial failed). The default raises
        in that None case to preserve the historical strict behavior; an overriding
        optimizer may instead return a valid product built from its own state.

        Args:
            state: The final optimization run state.
            objective: The objective used to rank trials.
            best_adapter: Adapter for the best snapshot, or None if there is no best.

        Returns:
            The object to hand back to the caller of `optimize` / `async_optimize`.
        """
        if best_adapter is None:
            raise RuntimeError(
                "Optimization finished but no best snapshot was recorded. "
                "This usually means all trials failed or the run state is corrupted."
            )
        return best_adapter

    async def async_finalize(
        self,
        state: OptimizationRunState,
        objective: Objective,
        best_adapter: Optional[ProgramAdapter],
    ) -> Any:
        """Async variant of `finalize`."""
        return self.finalize(state, objective, best_adapter)

    def _resolve_best_adapter(self, state: OptimizationRunState) -> Optional[ProgramAdapter]:
        """Load the adapter for `state.best_snapshot_id`, or None if there is no best."""
        if state.best_snapshot_id is None:
            return None
        best_snapshot = state.get_snapshot_by_id(state.best_snapshot_id)
        if best_snapshot is None:
            raise RuntimeError(
                f"best_snapshot_id '{state.best_snapshot_id}' not found in run state. "
                "The saved state may be incomplete or corrupted."
            )
        self.adapter._validate_snapshot(best_snapshot, context="best snapshot")
        return self.adapter.load_snapshot(best_snapshot)

    def load_optimized(self, save_dir: str) -> ProgramAdapter:
        """
        Load the best program from a saved run for inference, without re-optimizing.

        Pass the same `save_dir` given to a prior `optimize` / `async_optimize` call, and
        get back the optimized adapter — the best snapshot loaded into an adapter, exactly
        what the original run produced — at zero evaluation cost.

        Args:
            save_dir: The directory a prior run wrote its `optimization_state.json` into.

        Returns:
            A ProgramAdapter reflecting the best configuration found in the saved run.
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"save_dir '{save_dir}' does not exist or is not a directory.")
        state_path = os.path.join(save_dir, "optimization_state.json")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"No 'optimization_state.json' found under save_dir '{save_dir}'; "
                "there is no saved run to load."
            )

        state = OptimizationRunState.load_state(state_path)
        self._check_adapter_fingerprint(state, context="load optimized program")
        best_adapter = self._resolve_best_adapter(state)
        if best_adapter is None:
            raise RuntimeError(
                f"Saved run at '{save_dir}' has no best snapshot recorded; there is no "
                "optimized program to load. The run may have failed or not completed any trial."
            )
        return best_adapter

    @silence_cost_logs
    def optimize(
        self,
        evaluate_fn: Callable[..., EvaluationReturn],
        objective: Objective,
        max_trials: int = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> Any:

        execution_mode, max_workers = self._validate_optimize_args(
            objective,
            max_trials,
            execution_mode,
            max_workers,
            allow_concurrent=False,
        )
        run_started_at = time.perf_counter()
        cost_at_start = cost_manager.get_total_cost()

        # Initialize or load the optimization run state
        state = self._init_run_state(save_dir, resume_from)
        start_step = state.current_step
        self.load_optimizer_state(state)
        trials_root = self._resolve_trials_root(state, save_dir, resume_from)
        progress = OptimizationProgress(total=max_trials, initial=state.current_step)

        # Evaluate baseline once if not yet done; persists so resume skips this.
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is None:
            baseline_snapshot = state.get_snapshot_by_id(baseline_record.snapshot_id)
            baseline_adapter = self.adapter.load_snapshot(baseline_snapshot) if baseline_snapshot is not None else self.adapter
            baseline_trial_dir: Optional[str] = None
            baseline_workspace: Optional[TrialWorkspace] = None
            baseline_validation_results: List[ValidationResult] = []
            try:
                if baseline_snapshot is None:
                    raise RuntimeError("Baseline snapshot is missing from optimization state.")
                baseline_adapter._validate_snapshot(baseline_snapshot, context="baseline snapshot")
                baseline_trial_dir, baseline_workspace = _prepare_trial_dirs(
                    baseline_adapter,
                    baseline_snapshot,
                    BASELINE_TRIAL_ID,
                    trials_root,
                    workspace_root,
                    keep_trial_workspaces,
                    metadata={"baseline": True},
                    **kwargs,
                )
                baseline_validation_results = _normalize_validation_results(
                    baseline_adapter.validate_trial(
                        baseline_snapshot,
                        [],
                        workspace=baseline_workspace,
                        **kwargs,
                    ),
                    BASELINE_TRIAL_ID,
                )
                validation_error = _validation_failure_message(baseline_validation_results)
                if validation_error is not None:
                    raise RuntimeError(validation_error)
                baseline_context = TrialContext(
                    trial_id=BASELINE_TRIAL_ID,
                    trial_dir=baseline_trial_dir,
                    workspace=baseline_workspace,
                    metadata={"baseline": True},
                )
                evaluation: EvaluationResult = _normalize_evaluation_result(
                    _call_evaluate_fn(evaluate_fn, baseline_adapter, baseline_context), BASELINE_TRIAL_ID
                )
                captured_baseline = baseline_adapter.capture_after_eval(
                    snapshot=baseline_snapshot,
                    evaluation=evaluation,
                    changes=[],
                    workspace=baseline_workspace,
                    **kwargs,
                )
                if captured_baseline is not None:
                    baseline_adapter._validate_snapshot(captured_baseline, context="captured baseline snapshot")
                    state.snapshots.append(captured_baseline)
                    baseline_record.snapshot_id = captured_baseline.snapshot_id
                baseline_record.status = "completed"
                baseline_record.metrics = evaluation.metrics
                baseline_record.traces = evaluation.traces
                baseline_record.artifacts = evaluation.artifacts
                baseline_record.metadata = evaluation.metadata
                baseline_record.validation_results = baseline_validation_results
                baseline_record.trial_dir = baseline_trial_dir
                baseline_record.workspace_dir = baseline_workspace.root_dir if baseline_workspace is not None else None
                baseline_record.error = None
                self._checkpoint(state)
            except Exception as exc:
                baseline_record.status = "failed"
                baseline_record.validation_results = baseline_validation_results
                baseline_record.trial_dir = baseline_trial_dir
                baseline_record.workspace_dir = baseline_workspace.root_dir if baseline_workspace is not None else None
                baseline_record.error = str(exc)
                self._checkpoint(state)
                progress.close()
                raise
            finally:
                if baseline_workspace is not None:
                    baseline_workspace.cleanup()

        # Sync best_snapshot_id / best_metrics from objective (covers both fresh and resume).
        _sync_best(state, objective)
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is not None:
            self._report_trial_progress(progress, state, baseline_record, objective, baseline_record, **kwargs)
        self.on_run_start(state, objective, **kwargs)

        while state.current_step < max_trials:
            if self.should_stop(state, objective, **kwargs):
                self._checkpoint(state)
                break
            remaining = max_trials - state.current_step
            proposals = self.batch_propose(state, objective, budget_remaining=remaining, **kwargs)[:remaining]
            if not proposals:
                self._checkpoint(state)
                break
            record_count_before = len(state.trial_records)
            state = self.runtime.run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                execution_mode=execution_mode,
                max_workers=max_workers,
                trials_root=trials_root,
                workspace_root=workspace_root,
                keep_trial_workspaces=keep_trial_workspaces,
                isolate_snapshots=isolate_snapshots,
                **kwargs
            )
            new_records = state.trial_records[record_count_before:]
            for record in new_records:
                self._report_trial_progress(progress, state, record, objective, baseline_record, **kwargs)
                progress.update(1)
            self.observe(state, new_records, objective, **kwargs)
            state.current_step += len(proposals)
            self._prune_retained_snapshots(state, objective, **kwargs)
            self._checkpoint(state)

        progress.close()
        self._report_optimization_summary(
            progress=progress,
            state=state,
            objective=objective,
            baseline_record=baseline_record,
            elapsed_seconds=time.perf_counter() - run_started_at,
            start_step=start_step,
            max_trials=max_trials,
            total_cost=cost_manager.get_total_cost() - cost_at_start,
            **kwargs,
        )
        best_adapter = self._resolve_best_adapter(state)
        return self.finalize(state, objective, best_adapter)

    @silence_cost_logs
    async def async_optimize(
        self,
        evaluate_fn: Callable[..., Awaitable[EvaluationReturn]],
        objective: Objective,
        max_trials: int = 3,
        save_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        execution_mode: Literal["sequential", "concurrent"] = "sequential",
        max_workers: Optional[int] = None,
        workspace_root: Optional[str] = None,
        keep_trial_workspaces: bool = True,
        isolate_snapshots: bool = True,
        **kwargs
    ) -> Any:
        """
        Async variant of `optimize`. Supports both sequential and concurrent trial execution.

        Args:
            evaluate_fn: Async callable returning either metric dicts or EvaluationResult objects.
                         May be declared as `fn(adapter)` or `fn(adapter, context)`; the second
                         parameter must be named `ctx`/`context`/`trial_context` (or annotated
                         `TrialContext`) to receive a `TrialContext` carrying the trial id and
                         the persistent per-trial directory (`<save_dir>/trials/...`) where
                         evaluation results can be archived.
            objective: Optimization objective.
            max_trials: Total number of individual trials to run.
            save_dir: Directory to write per-trial state checkpoints.
            resume_from: Path to a prior checkpoint to resume from.
            execution_mode: "sequential" (default) or "concurrent". Pass "concurrent" when the
                            underlying program is stateless or safe to evaluate in parallel
                            (e.g. pure API calls, read-only inference).
            max_workers: Maximum number of coroutines running concurrently in concurrent mode.
                         Required and must be greater than 1 in concurrent mode.
                         Use None or 1 in sequential mode.
            **kwargs: Forwarded to `batch_propose` and `async_run_batch`.

        Returns:
            A new ProgramAdapter reflecting the best configuration found.
        """
        execution_mode, max_workers = self._validate_optimize_args(
            objective,
            max_trials,
            execution_mode,
            max_workers,
            allow_concurrent=True,
        )
        run_started_at = time.perf_counter()
        cost_at_start = cost_manager.get_total_cost()

        state = self._init_run_state(save_dir, resume_from)
        start_step = state.current_step
        self.load_optimizer_state(state)
        trials_root = self._resolve_trials_root(state, save_dir, resume_from)
        progress = OptimizationProgress(total=max_trials, initial=state.current_step)

        # Evaluate baseline once if not yet done; persists so resume skips this.
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is None:
            baseline_snapshot = state.get_snapshot_by_id(baseline_record.snapshot_id)
            baseline_adapter = self.adapter.load_snapshot(baseline_snapshot) if baseline_snapshot is not None else self.adapter
            baseline_trial_dir: Optional[str] = None
            baseline_workspace: Optional[TrialWorkspace] = None
            baseline_validation_results: List[ValidationResult] = []
            try:
                if baseline_snapshot is None:
                    raise RuntimeError("Baseline snapshot is missing from optimization state.")
                baseline_adapter._validate_snapshot(baseline_snapshot, context="baseline snapshot")
                baseline_trial_dir, baseline_workspace = _prepare_trial_dirs(
                    baseline_adapter,
                    baseline_snapshot,
                    BASELINE_TRIAL_ID,
                    trials_root,
                    workspace_root,
                    keep_trial_workspaces,
                    metadata={"baseline": True},
                    **kwargs,
                )
                baseline_validation_results = _normalize_validation_results(
                    await baseline_adapter.async_validate_trial(
                        baseline_snapshot,
                        [],
                        workspace=baseline_workspace,
                        **kwargs,
                    ),
                    BASELINE_TRIAL_ID,
                )
                validation_error = _validation_failure_message(baseline_validation_results)
                if validation_error is not None:
                    raise RuntimeError(validation_error)
                baseline_context = TrialContext(
                    trial_id=BASELINE_TRIAL_ID,
                    trial_dir=baseline_trial_dir,
                    workspace=baseline_workspace,
                    metadata={"baseline": True},
                )
                evaluation = _normalize_evaluation_result(
                    await _call_evaluate_fn(evaluate_fn, baseline_adapter, baseline_context), BASELINE_TRIAL_ID
                )
                captured_baseline = await baseline_adapter.async_capture_after_eval(
                    snapshot=baseline_snapshot,
                    evaluation=evaluation,
                    changes=[],
                    workspace=baseline_workspace,
                    **kwargs,
                )
                if captured_baseline is not None:
                    baseline_adapter._validate_snapshot(captured_baseline, context="captured baseline snapshot")
                    state.snapshots.append(captured_baseline)
                    baseline_record.snapshot_id = captured_baseline.snapshot_id
                baseline_record.status = "completed"
                baseline_record.metrics = evaluation.metrics
                baseline_record.traces = evaluation.traces
                baseline_record.artifacts = evaluation.artifacts
                baseline_record.metadata = evaluation.metadata
                baseline_record.validation_results = baseline_validation_results
                baseline_record.trial_dir = baseline_trial_dir
                baseline_record.workspace_dir = baseline_workspace.root_dir if baseline_workspace is not None else None
                baseline_record.error = None
                self._checkpoint(state)
            except Exception as exc:
                baseline_record.status = "failed"
                baseline_record.validation_results = baseline_validation_results
                baseline_record.trial_dir = baseline_trial_dir
                baseline_record.workspace_dir = baseline_workspace.root_dir if baseline_workspace is not None else None
                baseline_record.error = str(exc)
                self._checkpoint(state)
                progress.close()
                raise
            finally:
                if baseline_workspace is not None:
                    baseline_workspace.cleanup()

        # Sync best_snapshot_id / best_metrics from objective (covers both fresh and resume).
        _sync_best(state, objective)
        baseline_record = state.get_baseline_record()
        if baseline_record is not None and baseline_record.metrics is not None:
            self._report_trial_progress(progress, state, baseline_record, objective, baseline_record, **kwargs)
        await self.async_on_run_start(state, objective, **kwargs)

        while state.current_step < max_trials:
            if self.should_stop(state, objective, **kwargs):
                self._checkpoint(state)
                break
            remaining = max_trials - state.current_step
            proposals = (await self.async_batch_propose(state, objective, budget_remaining=remaining, **kwargs))[:remaining]
            if not proposals:
                self._checkpoint(state)
                break
            record_count_before = len(state.trial_records)
            state = await self.runtime.async_run_batch(
                state=state,
                proposals=proposals,
                evaluate_fn=evaluate_fn,
                objective=objective,
                execution_mode=execution_mode,
                max_workers=max_workers,
                trials_root=trials_root,
                workspace_root=workspace_root,
                keep_trial_workspaces=keep_trial_workspaces,
                isolate_snapshots=isolate_snapshots,
                **kwargs
            )
            new_records = state.trial_records[record_count_before:]
            for record in new_records:
                self._report_trial_progress(progress, state, record, objective, baseline_record, **kwargs)
                progress.update(1)
            await self.async_observe(state, new_records, objective, **kwargs)
            state.current_step += len(proposals)
            self._prune_retained_snapshots(state, objective, **kwargs)
            self._checkpoint(state)

        progress.close()
        self._report_optimization_summary(
            progress=progress,
            state=state,
            objective=objective,
            baseline_record=baseline_record,
            elapsed_seconds=time.perf_counter() - run_started_at,
            start_step=start_step,
            max_trials=max_trials,
            total_cost=cost_manager.get_total_cost() - cost_at_start,
            **kwargs,
        )
        best_adapter = self._resolve_best_adapter(state)
        return await self.async_finalize(state, objective, best_adapter)

    def propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        **kwargs
    ) -> OptimizationProposal:
        """
        Propose a single changeset for the next trial.

        Select a source snapshot from `state`, propose changes to optimization units,
        and return them as an OptimizationProposal. Called by the default `batch_propose`,
        so sequential algorithms only need to implement this method.

        A subclass must override **either** `propose` or `batch_propose` (or, for async
        proposal generation, `async_batch_propose`); the constructor enforces that at least
        one is provided. Algorithms that natively generate multiple candidates should
        override `batch_propose` and may leave `propose` unimplemented — this default raises
        when called.

        Args:
            state: Current optimization run state (past snapshots, trial records, best id).
            objective: The objective the proposed changes should aim to improve.
            **kwargs: Algorithm-specific arguments forwarded from `optimize`.

        Returns:
            An OptimizationProposal carrying the source snapshot id and the list of changes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement propose(). Override either propose() "
            f"for single-candidate algorithms or batch_propose() for population-based ones."
        )

    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs
    ) -> List[OptimizationProposal]:
        """
        Propose one or more changesets for the next optimization step.

        Default implementation delegates to `propose()` and returns a single-element list,
        which gives sequential algorithms free compatibility with the batch execution path.

        Override to return multiple proposals when the algorithm naturally generates a
        population of candidates (e.g. evolutionary search, beam search, TPE with parallel
        workers). `budget_remaining` is a *ceiling*, not a *target*: it is the number of
        trials left before `max_trials` is reached, not the number you should produce this
        round. The per-round batch size (population size, beam width, ...) is the
        algorithm's own property; an override should pick its natural batch size and cap it
        with `budget_remaining`, e.g. `n = min(self.population_size, budget_remaining)`.
        The caller (`optimize` / `async_optimize`) also trims the returned list to
        `budget_remaining` as a safety net, but relying on that trim wastes the generation
        cost (often LLM calls) of the discarded candidates, so algorithms should self-limit.

        Args:
            state: Current optimization run state.
            objective: The objective the proposed changes should aim to improve.
            budget_remaining: Upper bound on how many proposals may be returned, i.e. the
                              trials left before `max_trials`. None means unbounded.
                              Population-based algorithms should `min()` their batch size
                              against this rather than treating it as the count to generate.
            **kwargs: Forwarded to `propose` (or used directly if overriding).

        Returns:
            A non-empty list of OptimizationProposal objects.
        """
        proposals = [self.propose(state, objective, **kwargs)]
        return proposals[:budget_remaining] if budget_remaining is not None else proposals

    async def async_batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs
    ) -> List[OptimizationProposal]:
        """
        Async variant of `batch_propose`. Default delegates to sync `batch_propose`.

        Override when proposal generation involves async I/O (e.g. LLM calls to generate
        the next candidate). The async optimization loop calls this instead of `batch_propose`.

        Args:
            state: Current optimization run state.
            objective: The objective the proposed changes should aim to improve.
            budget_remaining: Upper bound on how many proposals may be returned, i.e. the
                              trials left before `max_trials`. None means unbounded.
                              See `batch_propose` for how population-based algorithms should
                              use it (a ceiling to `min()` against, not a target count).
            **kwargs: Forwarded to `batch_propose` (or used directly if overriding).

        Returns:
            A non-empty list of OptimizationProposal objects.
        """
        return self.batch_propose(state, objective, budget_remaining=budget_remaining, **kwargs)
