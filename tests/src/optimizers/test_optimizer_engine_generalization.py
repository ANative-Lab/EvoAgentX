from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, FrozenSet, List

import pytest

from evoagentx.optimizers.engine.adapter import ProgramAdapter, SnapShot
from evoagentx.optimizers.engine.base import (
    EvaluationResult,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    TrialRecord,
    UnitChange,
    ValidationResult,
)
from evoagentx.optimizers.engine.objective import ParetoObjective, ScalarObjective
from evoagentx.optimizers.engine.optimizer import OptimizationRunState, Optimizer


class DummyAdapter(ProgramAdapter):
    def __init__(self, prompt: str = "base", memories: List[str] | None = None):
        self.prompt = prompt
        self.memories = list(memories or [])

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        return {"prompt": self.prompt, "memories": self.memories}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name="prompt",
                uid="prompt",
                unit_type=OptimizationUnitType.PROMPT,
                json_schema={"type": "string"},
                metadata={"role": "system"},
            ),
            OptimizationUnit(
                name="memories",
                uid="memories",
                unit_type=OptimizationUnitType.MEMORY,
                json_schema={"type": "array", "items": {"type": "string"}},
                allowed_operations=["replace", "append"],
                operation_schemas={"append": {"type": "string"}},
                metadata={"store": "experience"},
            ),
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"prompt": self.prompt, "memories": list(self.memories)})

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return DummyAdapter(
            prompt=snapshot.unit_values["prompt"],
            memories=list(snapshot.unit_values.get("memories", [])),
        )

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        values = dict(snapshot.unit_values)
        values["memories"] = list(values.get("memories", []))
        for change in changes:
            if change.operation == "replace":
                values[change.uid] = change.value
            elif change.operation == "append":
                values[change.uid].append(change.value)
            else:
                raise ValueError(f"unsupported operation {change.operation}")
        return SnapShot(unit_values=values, program_config=snapshot.program_config)


class PromptOnlyOptimizer(Optimizer):

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["prompt"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "improved prompt", metadata={"reason": "test"})],
        )


class MemoryAppendOptimizer(Optimizer):
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.MEMORY})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["memories"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "learned experience", operation="append")],
        )

    def observe(
        self,
        state: OptimizationRunState,
        trial_records: List[Any],
        objective: ScalarObjective,
        **kwargs,
    ) -> None:
        state.optimizer_state["observed_trials"] = state.optimizer_state.get("observed_trials", 0) + len(trial_records)


class SkillFileAdapter(ProgramAdapter):
    uses_workspace = True

    def __init__(self, content: str = "def base_skill():\n    return 'base'\n"):
        self.content = content

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        workspace_file = self.workspace.path("skills.py") if self.workspace is not None else None
        return {"content": self.content, "workspace_file": workspace_file}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                uid="skills.py",
                name="skills.py",
                unit_type=OptimizationUnitType.CODE,
                json_schema={"type": "string"},
                allowed_operations=["replace", "append"],
                metadata={"path": "skills.py", "language": "python", "entrypoint": "base_skill"},
            )
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"skills.py": self.content})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        values = dict(snapshot.unit_values)
        for change in changes:
            current = values.get(change.uid, "")
            if change.operation == "replace":
                values[change.uid] = change.value
            elif change.operation == "append":
                separator = "" if not current or current.endswith("\n") else "\n"
                values[change.uid] = f"{current}{separator}{change.value}"
            else:
                raise ValueError(f"unsupported operation {change.operation}")
        return SnapShot(unit_values=values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return SkillFileAdapter(content=snapshot.unit_values["skills.py"])

    def prepare_workspace(self, workspace, snapshot: SnapShot, **kwargs) -> None:
        path = workspace.path("skills.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(snapshot.unit_values["skills.py"])

    def validate_trial(self, snapshot: SnapShot, changes: List[UnitChange], workspace=None, **kwargs) -> List[ValidationResult]:
        content = snapshot.unit_values["skills.py"]
        if "def " not in content:
            return [ValidationResult(validator="python-static", status="failed", message="no function definition")]
        return [ValidationResult(validator="python-static", status="passed", message="contains function definition")]


class SkillCodeOptimizer(Optimizer):
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.CODE})

    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["skills.py"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[
                UnitChange.create(
                    unit,
                    "def new_skill():\n    return 'new'\n",
                    operation="append",
                )
            ],
        )


class BrokenSkillCodeOptimizer(SkillCodeOptimizer):
    def propose(self, state: OptimizationRunState, objective: ScalarObjective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["skills.py"]
        source_snapshot_id = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source_snapshot_id,
            changes=[UnitChange.create(unit, "print('not a skill')", operation="replace")],
        )


def test_optimizer_targets_supported_units_by_default_and_persists_evaluation_artifacts(tmp_path):
    optimizer = PromptOnlyOptimizer(DummyAdapter())

    assert optimizer.target_unit_uids == ["prompt"]

    def evaluate(adapter: ProgramAdapter) -> EvaluationResult:
        output = adapter.execute()
        return EvaluationResult(
            metrics={"score": len(output["prompt"])},
            traces=[{"prompt": output["prompt"]}],
            artifacts={"length": len(output["prompt"])},
            metadata={"split": "dev"},
        )

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "improved prompt"

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.metrics == {"score": len("improved prompt")}
    assert trial.traces == [{"prompt": "improved prompt"}]
    assert trial.artifacts == {"length": len("improved prompt")}
    assert trial.metadata == {"split": "dev"}
    assert state.adapter_fingerprint is not None


def test_optimizer_can_target_memory_unit_and_use_append_changes(tmp_path):
    optimizer = MemoryAppendOptimizer(DummyAdapter(), target_unit_types=OptimizationUnitType.MEMORY)

    assert optimizer.target_unit_uids == ["memories"]

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        return {"score": len(adapter.execute()["memories"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["memories"] == ["learned experience"]

    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.optimizer_state["observed_trials"] == 1


def test_explicit_unsupported_target_unit_is_rejected():
    try:
        PromptOnlyOptimizer(DummyAdapter(), target_unit_uids=["memories"])
    except TypeError as exc:
        assert "unsupported type" in str(exc)
    else:
        raise AssertionError("expected unsupported target unit to be rejected")


def test_code_unit_uses_trial_workspace_and_validation_pipeline(tmp_path):
    optimizer = SkillCodeOptimizer(SkillFileAdapter())

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        output = adapter.execute()
        assert output["workspace_file"] is not None
        assert os.path.exists(output["workspace_file"])
        return {"score": int("new_skill" in output["content"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert "new_skill" in best.execute()["content"]

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.status == "completed"
    assert trial.validation_results[0].status == "passed"
    assert trial.workspace_dir is not None
    workspace_file = os.path.join(trial.workspace_dir, "skills.py")
    assert os.path.exists(workspace_file)
    with open(workspace_file, encoding="utf-8") as handle:
        assert "new_skill" in handle.read()


def test_validation_failure_skips_trial_evaluation(tmp_path):
    optimizer = BrokenSkillCodeOptimizer(SkillFileAdapter())
    calls = {"count": 0}

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        calls["count"] += 1
        return {"score": int("def " in adapter.execute()["content"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert calls["count"] == 1
    assert best.execute()["content"] == "def base_skill():\n    return 'base'\n"

    state = OptimizationRunState.load_state(str(tmp_path))
    trial = state.trial_records[-1]
    assert trial.status == "failed"
    assert trial.validation_results[0].status == "failed"
    assert "python-static" in trial.error


# ---------------------------------------------------------------------------
# Multi-trial: minimize direction + current_step tracking
# ---------------------------------------------------------------------------

def test_multi_trial_minimize_objective(tmp_path):
    """Three trials; the one with the lowest score wins under minimize."""
    score_map = {"base": 100, "prompt-0": 50, "prompt-1": 75, "prompt-2": 60}
    call_n = {"n": 0}

    class SeqOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def propose(self, state, objective, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            name = f"prompt-{call_n['n']}"
            call_n["n"] += 1
            return OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, name)])

    optimizer = SeqOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=lambda a: {"score": score_map[a.execute()["prompt"]]},
        objective=ScalarObjective(metric="score", direction="minimize"),
        max_trials=3,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "prompt-0"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.best_metrics["score"] == 50
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# should_stop hook
# ---------------------------------------------------------------------------

def test_should_stop_halts_before_max_trials(tmp_path):
    class EarlyStopOptimizer(PromptOnlyOptimizer):
        def should_stop(self, state, objective, **kwargs):
            return state.current_step >= 1

    optimizer = EarlyStopOptimizer(DummyAdapter())
    eval_count = {"n": 0}

    def evaluate(adapter):
        eval_count["n"] += 1
        return {"score": eval_count["n"]}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=5,
        save_dir=str(tmp_path),
    )

    # baseline + exactly 1 trial before should_stop fires
    assert eval_count["n"] == 2
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 1


# ---------------------------------------------------------------------------
# Population-based batch_propose (multiple proposals per step)
# ---------------------------------------------------------------------------

def test_batch_propose_returns_multiple_proposals(tmp_path):
    """batch_propose returning N proposals evaluates all N in one step."""
    score_map = {"candidate-0": 10, "candidate-1": 30, "candidate-2": 20}
    received_max = []

    class PopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def batch_propose(self, state, objective, budget_remaining=None, **kwargs):
            received_max.append(budget_remaining)
            unit = self.target_units_by_uid["prompt"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            names = ["candidate-0", "candidate-1", "candidate-2"]
            if budget_remaining is not None:
                names = names[:budget_remaining]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = PopOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=lambda a: {"score": score_map.get(a.execute()["prompt"], 0)},
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=5,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "candidate-1"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert received_max == [5, 2]
    assert state.current_step == 5


# ---------------------------------------------------------------------------
# Sync concurrent execution is rejected
# ---------------------------------------------------------------------------

def test_sync_optimize_rejects_concurrent_execution(tmp_path):
    class ConcurrentPopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def batch_propose(self, state, objective, budget_remaining=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            names = ["cp-0", "cp-1", "cp-2"]
            if budget_remaining is not None:
                names = names[:budget_remaining]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = ConcurrentPopOptimizer(DummyAdapter())

    def evaluate(adapter):
        return {"score": 1}

    with pytest.raises(ValueError, match="Synchronous optimize/run_batch does not support"):
        optimizer.optimize(
            evaluate_fn=evaluate,
            objective=ScalarObjective(metric="score"),
            max_trials=3,
            execution_mode="concurrent",
            max_workers=5,
            save_dir=str(tmp_path),
        )


def test_sync_run_batch_rejects_concurrent_execution(tmp_path):
    optimizer = PromptOnlyOptimizer(DummyAdapter())
    objective = ScalarObjective(metric="score")
    state = optimizer._init_run_state(save_dir=str(tmp_path))
    proposal = optimizer.propose(state, objective)

    with pytest.raises(ValueError, match="Synchronous optimize/run_batch does not support"):
        optimizer.runtime.run_batch(
            state=state,
            proposals=[proposal],
            evaluate_fn=lambda adapter: {"score": len(adapter.execute()["prompt"])},
            objective=objective,
            execution_mode="concurrent",
            max_workers=2,
        )


# ---------------------------------------------------------------------------
# evaluate_fn exception records a failed trial and keeps the baseline as best
# ---------------------------------------------------------------------------

def test_evaluate_fn_exception_records_failed_trial_and_keeps_baseline(tmp_path):
    call_n = {"n": 0}

    def evaluate(adapter):
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise RuntimeError("eval crashed")
        return {"score": 1}

    optimizer = PromptOnlyOptimizer(DummyAdapter())
    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["prompt"] == "base"
    state = OptimizationRunState.load_state(str(tmp_path))
    failed = [r for r in state.trial_records if r.status == "failed"]
    assert len(failed) == 1
    assert "eval crashed" in failed[0].error


# ---------------------------------------------------------------------------
# Multiple changes in a single proposal
# ---------------------------------------------------------------------------

def test_proposal_with_multiple_changes_updates_all_units(tmp_path):
    class MultiChangeOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
            OptimizationUnitType.PROMPT, OptimizationUnitType.MEMORY,
        })

        def propose(self, state, objective, **kwargs):
            prompt_unit = self.target_units_by_uid["prompt"]
            memory_unit = self.target_units_by_uid["memories"]
            source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
            return OptimizationProposal(
                source_snapshot_id=source,
                changes=[
                    UnitChange.create(prompt_unit, "updated prompt"),
                    UnitChange.create(memory_unit, "new memory", operation="append"),
                ],
            )

    optimizer = MultiChangeOptimizer(DummyAdapter())

    def evaluate(adapter):
        out = adapter.execute()
        # trial gets score 2 (has memory); baseline gets score 1 (empty memories)
        return {"score": 1 + len(out["memories"])}

    best = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    output = best.execute()
    assert output["prompt"] == "updated prompt"
    assert "new memory" in output["memories"]


# ---------------------------------------------------------------------------
# keep_trial_workspaces=False deletes workspace after evaluation
# ---------------------------------------------------------------------------

def test_keep_trial_workspaces_false_deletes_workspace(tmp_path):
    optimizer = SkillCodeOptimizer(SkillFileAdapter())

    def evaluate(adapter):
        return {"score": int("new_skill" in adapter.execute()["content"])}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
        keep_trial_workspaces=False,
    )

    state = OptimizationRunState.load_state(str(tmp_path))
    non_baseline = [r for r in state.trial_records if r.trial_id != 0]
    for record in non_baseline:
        assert record.workspace_dir is not None
        assert not os.path.exists(record.workspace_dir), "workspace should be deleted when keep=False"


def test_trial_workspace_create_clears_stale_files(tmp_path):
    from evoagentx.optimizers.engine.adapter import TrialWorkspace

    root = str(tmp_path / "trial_0001_abcd")
    first = TrialWorkspace.create(root, trial_id=1)
    stale = first.path("stale.txt")
    with open(stale, "w") as f:
        f.write("old")
    assert os.path.exists(stale)

    # Reusing the same path (baseline retry / resumed same trial_id) must start fresh.
    second = TrialWorkspace.create(root, trial_id=1)
    assert second.root_dir == first.root_dir
    assert not os.path.exists(stale), "stale file from prior attempt must be cleared"

    # Opt-out preserves existing contents.
    with open(stale, "w") as f:
        f.write("keep")
    TrialWorkspace.create(root, trial_id=1, clean=False)
    assert os.path.exists(stale)


# ---------------------------------------------------------------------------
# Resume from checkpoint skips already-evaluated trials
# ---------------------------------------------------------------------------

def test_resume_from_checkpoint_skips_evaluated_trials(tmp_path):
    eval_count = {"n": 0}

    def evaluate(adapter):
        eval_count["n"] += 1
        return {"score": eval_count["n"]}

    optimizer1 = PromptOnlyOptimizer(DummyAdapter())
    optimizer1.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=2,
        save_dir=str(tmp_path),
    )
    count_after_first_run = eval_count["n"]  # baseline + 2 trials = 3

    optimizer2 = PromptOnlyOptimizer(DummyAdapter())
    optimizer2.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        resume_from=str(tmp_path),
        save_dir=str(tmp_path),
    )

    assert eval_count["n"] == count_after_first_run + 1, "resume should only evaluate the remaining 1 trial"
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3


# ---------------------------------------------------------------------------
# ParetoObjective: dominance logic and best-record selection
# ---------------------------------------------------------------------------

def test_pareto_objective_dominance():
    obj = ParetoObjective(metrics=["accuracy", "speed"], directions=["maximize", "maximize"])

    assert obj.is_better({"accuracy": 0.9, "speed": 100}, {"accuracy": 0.8, "speed": 100})
    assert not obj.is_better({"accuracy": 0.9, "speed": 80}, {"accuracy": 0.8, "speed": 100})
    assert not obj.is_better({"accuracy": 0.7, "speed": 90}, {"accuracy": 0.9, "speed": 100})
    assert not obj.is_better({"accuracy": 0.9, "speed": 100}, {"accuracy": 0.9, "speed": 100})


def test_pareto_objective_select_best_by_dominance_count():
    obj = ParetoObjective(metrics=["f1", "latency"], directions=["maximize", "minimize"])

    records = [
        TrialRecord(trial_id=1, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s1", metrics={"f1": 0.9, "latency": 50}),
        TrialRecord(trial_id=2, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s2", metrics={"f1": 0.7, "latency": 100}),
        TrialRecord(trial_id=3, changes=[], source_snapshot_id="s0", status="completed",
                    snapshot_id="s3", metrics={"f1": 0.8, "latency": 60}),
        TrialRecord(trial_id=4, changes=[], source_snapshot_id="s0", status="failed",
                    snapshot_id=None, metrics=None),
    ]

    best = obj.select_best_trial_record(records)
    # s1 dominates s2 and s3; s3 only dominates s2; s2 dominates nobody.
    assert best.snapshot_id == "s1"


# ---------------------------------------------------------------------------
# Async optimize: sequential and concurrent paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_optimize_concurrent(tmp_path):
    class AsyncPopOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def batch_propose(self, state, objective, budget_remaining=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            names = ["ap-0", "ap-1", "ap-2"]
            if budget_remaining is not None:
                names = names[:budget_remaining]
            return [
                OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, n)])
                for n in names
            ]

    optimizer = AsyncPopOptimizer(DummyAdapter())

    async def evaluate(adapter):
        return {"score": 1}

    await optimizer.async_optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=3,
        execution_mode="concurrent",
        max_workers=5,
        save_dir=str(tmp_path),
    )

    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.current_step == 3
    trial_ids = [r.trial_id for r in state.trial_records]
    assert len(set(trial_ids)) == len(trial_ids), "concurrent async trials must have unique IDs"


# ---------------------------------------------------------------------------
# async_batch_propose override is used in async_optimize (not sync batch_propose)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_batch_propose_override_is_used_in_async_optimize(tmp_path):
    sync_calls = {"n": 0}

    class AsyncProposalOptimizer(Optimizer):
        supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

        def batch_propose(self, state, objective, budget_remaining=None, **kwargs):
            sync_calls["n"] += 1
            raise AssertionError("sync batch_propose must not be called from async_optimize")

        async def async_batch_propose(self, state, objective, budget_remaining=None, **kwargs):
            unit = self.target_units_by_uid["prompt"]
            source = state.snapshots[-1].snapshot_id
            return [OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, "async-p")])]

    optimizer = AsyncProposalOptimizer(DummyAdapter())

    async def evaluate(adapter):
        # baseline prompt="base" (len 4), trial prompt="async-p" (len 7), so trial wins
        return {"score": len(adapter.execute()["prompt"])}

    best = await optimizer.async_optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert sync_calls["n"] == 0
    assert best.execute()["prompt"] == "async-p"


# ---------------------------------------------------------------------------
# Hook 1: finalize() lets an optimizer return an arbitrary product
# (archive / population / accumulated state) instead of the single best adapter.
# ---------------------------------------------------------------------------

class ArchiveOptimizer(PromptOnlyOptimizer):
    """Records every completed score into an archive and returns it from finalize()."""

    def observe(self, state, trial_records, objective, **kwargs):
        archive = state.optimizer_state.setdefault("archive", [])
        for record in trial_records:
            if record.metrics is not None:
                archive.append(record.metrics["score"])

    def finalize(self, state, objective, best_adapter):
        return {"best_adapter": best_adapter, "archive": list(state.optimizer_state.get("archive", []))}


def test_finalize_returns_custom_product(tmp_path):
    optimizer = ArchiveOptimizer(DummyAdapter())

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        return {"score": len(adapter.execute()["prompt"])}

    result = optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=2,
        save_dir=str(tmp_path),
    )

    # finalize controls the return value: a dict, not a bare ProgramAdapter.
    assert isinstance(result, dict)
    assert result["best_adapter"].execute()["prompt"] == "improved prompt"
    # PromptOnlyOptimizer proposes the same prompt each trial, so the archive
    # holds one score per completed (non-baseline) trial.
    assert result["archive"] == [len("improved prompt"), len("improved prompt")]


# ---------------------------------------------------------------------------
# Hook 2: capture_after_eval() persists program state that changed *during*
# evaluation (online memory growth), for both the baseline and trial paths.
# ---------------------------------------------------------------------------

class OnlineMemoryAdapter(ProgramAdapter):
    """Writes a memory entry while being evaluated and exposes it via capture_after_eval."""

    def __init__(self, prompt: str = "base", memories: List[str] | None = None):
        self.prompt = prompt
        self.memories = list(memories or [])
        self._grown: List[str] | None = None

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        # Simulate the agent appending an experience to memory during the rollout.
        self._grown = self.memories + [f"experience::{self.prompt}"]
        return {"prompt": self.prompt, "memories": self._grown}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(name="prompt", uid="prompt", unit_type=OptimizationUnitType.PROMPT, json_schema={"type": "string"}),
            OptimizationUnit(
                name="memories",
                uid="memories",
                unit_type=OptimizationUnitType.MEMORY,
                json_schema={"type": "array", "items": {"type": "string"}},
            ),
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"prompt": self.prompt, "memories": list(self.memories)})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        values = dict(snapshot.unit_values)
        values["memories"] = list(values.get("memories", []))
        for change in changes:
            if change.operation == "replace":
                values[change.uid] = change.value
            else:
                raise ValueError(f"unsupported operation {change.operation}")
        return SnapShot(unit_values=values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return OnlineMemoryAdapter(
            prompt=snapshot.unit_values["prompt"],
            memories=list(snapshot.unit_values.get("memories", [])),
        )

    def capture_after_eval(self, snapshot, evaluation, changes, workspace=None, **kwargs):
        if self._grown is None:
            return None
        values = dict(snapshot.unit_values)
        values["memories"] = list(self._grown)
        return SnapShot(unit_values=values, program_config=snapshot.program_config)


class MemoryChainOptimizer(Optimizer):
    """Branches from the best (accumulated) snapshot and nudges the prompt."""

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

    def propose(self, state, objective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["prompt"]
        source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, "p1")])


def test_capture_after_eval_persists_online_memory_growth(tmp_path):
    optimizer = MemoryChainOptimizer(OnlineMemoryAdapter())

    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        # Longer accumulated memory scores higher, so each trial improves on the last.
        return {"score": len(adapter.execute()["memories"])}

    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    state = OptimizationRunState.load_state(str(tmp_path))

    # Baseline memory growth was captured into the baseline's result snapshot.
    baseline_record = state.get_baseline_record()
    baseline_snapshot = state.get_snapshot_by_id(baseline_record.snapshot_id)
    assert baseline_snapshot.unit_values["memories"] == ["experience::base"]

    # The trial branched from that accumulated snapshot and captured its own growth,
    # so the recorded snapshot reflects the cumulative chain, not the pre-eval state.
    trial_record = state.trial_records[-1]
    trial_snapshot = state.get_snapshot_by_id(trial_record.snapshot_id)
    assert trial_snapshot.unit_values["memories"] == ["experience::base", "experience::p1"]
    # best points at the accumulated trial snapshot.
    assert state.best_snapshot_id == trial_record.snapshot_id


# ---------------------------------------------------------------------------
# Hook 3 & 4: serialize_optimizer_state() / load_optimizer_state() let an
# optimizer keep a live (non-serializable) object and survive resume.
# ---------------------------------------------------------------------------

class _LiveCounter:
    """Stand-in for a live, non-serializable object (sampler / study / index)."""

    def __init__(self, n: int = 0):
        self.n = n


class LiveStateOptimizer(PromptOnlyOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = _LiveCounter()  # lives on the instance, never stored raw in optimizer_state

    def observe(self, state, trial_records, objective, **kwargs):
        self.counter.n += len(trial_records)

    def serialize_optimizer_state(self, state):
        data = dict(state.optimizer_state)
        data["counter_n"] = self.counter.n
        return data

    def load_optimizer_state(self, state):
        self.counter = _LiveCounter(n=state.optimizer_state.get("counter_n", 0))


def test_optimizer_state_serialization_survives_resume(tmp_path):
    def evaluate(adapter: ProgramAdapter) -> Dict[str, Any]:
        return {"score": len(adapter.execute()["prompt"])}

    optimizer1 = LiveStateOptimizer(DummyAdapter())
    optimizer1.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )
    # One trial observed -> counter advanced and was serialized to plain data.
    assert optimizer1.counter.n == 1
    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.optimizer_state["counter_n"] == 1

    # Fresh optimizer (counter starts at 0); resume must reconstruct the live counter.
    optimizer2 = LiveStateOptimizer(DummyAdapter())
    assert optimizer2.counter.n == 0
    optimizer2.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=2,
        resume_from=str(tmp_path),
        save_dir=str(tmp_path),
    )
    # load_optimizer_state restored 1, the single remaining trial added 1 -> 2.
    assert optimizer2.counter.n == 2
    reloaded = OptimizationRunState.load_state(str(tmp_path))
    assert reloaded.optimizer_state["counter_n"] == 2


def test_baseline_failure_checkpoint_serializes_optimizer_state(tmp_path):
    def evaluate(_adapter: ProgramAdapter) -> Dict[str, Any]:
        raise RuntimeError("baseline failed")

    with pytest.raises(RuntimeError, match="baseline failed"):
        LiveStateOptimizer(DummyAdapter()).optimize(
            evaluate_fn=evaluate,
            objective=ScalarObjective(metric="score"),
            max_trials=1,
            save_dir=str(tmp_path),
        )

    state = OptimizationRunState.load_state(str(tmp_path))
    assert state.optimizer_state["counter_n"] == 0
    assert state.get_baseline_record().status == "failed"
    assert state.get_baseline_record().error == "baseline failed"


# ---------------------------------------------------------------------------
# Adapter-defined (custom) operations: allow-listing, schema gating, and
# routing through merge_changes. Exercises extensibility of UnitChange.operation
# beyond the built-in ChangeOperation set (e.g. memory 'consolidate', code 'diff').
# ---------------------------------------------------------------------------


class ConsolidatingAdapter(ProgramAdapter):
    """Memory adapter exposing a custom 'consolidate' operation handled in merge_changes."""

    def __init__(self, notes: List[str] | None = None):
        self.notes = list(notes or [])

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        return {"notes": self.notes}

    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name="notes",
                uid="notes",
                unit_type=OptimizationUnitType.MEMORY,
                json_schema={"type": "array", "items": {"type": "string"}},
                allowed_operations=["replace", "consolidate"],
            )
        ]

    def take_snapshot(self) -> SnapShot:
        return SnapShot(unit_values={"notes": list(self.notes)})

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        values = dict(snapshot.unit_values)
        values["notes"] = list(values.get("notes", []))
        for change in changes:
            if change.operation == "consolidate":
                # domain-specific semantics: collapse the note list into one summary note
                values[change.uid] = [f"summary({len(values[change.uid])})"]
            elif change.operation == "replace":
                values[change.uid] = change.value
            else:
                raise ValueError(f"unsupported operation {change.operation}")
        return SnapShot(unit_values=values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> ProgramAdapter:
        return ConsolidatingAdapter(notes=list(snapshot.unit_values.get("notes", [])))


class ConsolidateOptimizer(Optimizer):
    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.MEMORY})

    def propose(self, state, objective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["notes"]
        source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        return OptimizationProposal(
            source_snapshot_id=source,
            changes=[UnitChange.create(unit, None, operation="consolidate")],
        )


def test_custom_operation_routes_through_merge_changes(tmp_path):
    optimizer = ConsolidateOptimizer(ConsolidatingAdapter(notes=["a", "b", "c"]))

    # Fewer notes scores lower; consolidation collapses 3 notes into 1 under minimize.
    best = optimizer.optimize(
        evaluate_fn=lambda a: {"score": len(a.execute()["notes"])},
        objective=ScalarObjective(metric="score", direction="minimize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    assert best.execute()["notes"] == ["summary(3)"]
    trial = OptimizationRunState.load_state(str(tmp_path)).trial_records[-1]
    assert trial.status == "completed"
    assert trial.changes[0].operation == "consolidate"


# ---------------------------------------------------------------------------
# Online-scalability levers (Plan A): per-run isolate_snapshots flag + the
# retained_snapshot_ids retention policy. Both default to historical behavior.
# ---------------------------------------------------------------------------


class _SeqPromptOptimizer(Optimizer):
    """Proposes a distinct, strictly-improving prompt each trial (p0, p1, ...)."""

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({OptimizationUnitType.PROMPT})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = 0

    def propose(self, state, objective, **kwargs) -> OptimizationProposal:
        unit = self.target_units_by_uid["prompt"]
        source = state.best_snapshot_id or state.snapshots[-1].snapshot_id
        name = f"p{self._n}"
        self._n += 1
        return OptimizationProposal(source_snapshot_id=source, changes=[UnitChange.create(unit, name)])


_SEQ_SCORES = {"base": 0, "p0": 1, "p1": 2, "p2": 3}


def test_snapshot_retention_bounds_stored_snapshots_and_skips_copy(tmp_path):
    class RetainingOptimizer(_SeqPromptOptimizer):
        def retained_snapshot_ids(self, state, objective, **kwargs):
            return set()  # keep only the engine floor: baseline + best + latest

    best = RetainingOptimizer(DummyAdapter()).optimize(
        evaluate_fn=lambda a: {"score": _SEQ_SCORES[a.execute()["prompt"]]},
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=3,
        save_dir=str(tmp_path),
        isolate_snapshots=False,  # exercise the no-copy path alongside retention
    )

    assert best.execute()["prompt"] == "p2"
    state = OptimizationRunState.load_state(str(tmp_path))
    # Without pruning there would be baseline + 3 trials = 4 snapshots; the floor caps it.
    assert len(state.snapshots) <= 3
    # the best result must still be resolvable after pruning
    assert state.best_snapshot_id is not None
    assert state.get_snapshot_by_id(state.best_snapshot_id) is not None
    assert state.current_step == 3


def test_isolate_snapshots_false_leaves_source_snapshot_intact(tmp_path):
    best = PromptOnlyOptimizer(DummyAdapter()).optimize(
        evaluate_fn=lambda a: {"score": len(a.execute()["prompt"])},
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
        isolate_snapshots=False,
    )
    assert best.execute()["prompt"] == "improved prompt"
    # the no-copy path must not have mutated the baseline source snapshot
    state = OptimizationRunState.load_state(str(tmp_path))
    baseline = state.get_baseline_record()
    assert state.get_snapshot_by_id(baseline.snapshot_id).unit_values["prompt"] == "base"


# ---------------------------------------------------------------------------
# Trial directories + TrialContext passed to two-argument evaluate_fn
# ---------------------------------------------------------------------------

def test_trial_dirs_created_and_context_passed_to_evaluate_fn(tmp_path):
    from evoagentx.optimizers.engine.adapter import TrialContext

    seen_contexts: List[TrialContext] = []

    def evaluate(adapter, context):
        seen_contexts.append(context)
        assert context.trial_dir is not None and os.path.isdir(context.trial_dir)
        # the evaluator can archive results into the trial dir
        with open(os.path.join(context.trial_dir, "eval.txt"), "w") as f:
            f.write("done")
        return {"score": len(adapter.execute()["prompt"])}

    optimizer = PromptOnlyOptimizer(DummyAdapter())
    optimizer.optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    # baseline + 1 trial, each with its own directory under <save_dir>/trials
    assert len(seen_contexts) == 2
    assert seen_contexts[0].trial_id == 0 and seen_contexts[0].metadata == {"baseline": True}
    trial_dirs = {ctx.trial_dir for ctx in seen_contexts}
    assert len(trial_dirs) == 2
    for trial_dir in trial_dirs:
        assert os.path.dirname(trial_dir) == os.path.join(str(tmp_path), "trials")
        assert os.path.exists(os.path.join(trial_dir, "eval.txt")), "archived results must survive the trial"

    # trial_dir is recorded on the trial records
    state = OptimizationRunState.load_state(str(tmp_path))
    for record in state.trial_records:
        assert record.trial_dir in trial_dirs


def test_second_param_not_named_context_does_not_receive_it(tmp_path):
    """`def evaluate(adapter, split="dev")` must keep its default, not get a TrialContext."""
    seen_splits = []

    def evaluate(adapter, split="dev"):
        seen_splits.append(split)
        return {"score": len(adapter.execute()["prompt"])}

    PromptOnlyOptimizer(DummyAdapter()).optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )
    assert seen_splits == ["dev", "dev"]  # baseline + trial, default untouched


def test_annotated_second_param_receives_context(tmp_path):
    from evoagentx.optimizers.engine.adapter import TrialContext

    seen = []

    def evaluate(adapter, run_info: TrialContext = None):  # non-standard name, typed
        seen.append(run_info)
        return {"score": len(adapter.execute()["prompt"])}

    PromptOnlyOptimizer(DummyAdapter()).optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )
    assert len(seen) == 2
    assert all(isinstance(c, TrialContext) for c in seen)


def test_single_arg_evaluate_fn_still_supported(tmp_path):
    best = PromptOnlyOptimizer(DummyAdapter()).optimize(
        evaluate_fn=lambda a: {"score": len(a.execute()["prompt"])},
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
        save_dir=str(tmp_path),
    )
    assert best.execute()["prompt"] == "improved prompt"


def test_no_save_dir_means_no_trial_dirs():
    def evaluate(adapter, context):
        assert context.trial_dir is None
        return {"score": len(adapter.execute()["prompt"])}

    best = PromptOnlyOptimizer(DummyAdapter()).optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=1,
    )
    assert best.execute()["prompt"] == "improved prompt"


def test_uses_workspace_adapter_gets_workspace_inside_trial_dir(tmp_path):
    seen = {}

    def evaluate(adapter, context):
        seen["context"] = context
        return {"score": int("new_skill" in adapter.execute()["content"])}

    SkillCodeOptimizer(SkillFileAdapter()).optimize(
        evaluate_fn=evaluate,
        objective=ScalarObjective(metric="score"),
        max_trials=1,
        save_dir=str(tmp_path),
    )

    context = seen["context"]
    assert context.workspace is not None
    assert context.workspace.root_dir == os.path.join(context.trial_dir, "workspace")
    assert os.path.exists(context.workspace.path("skills.py"))
