"""Tests for the engine-based MAP-Elites optimizer (evoagentx.optimizers.map_elites_optimizer)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.engine.optimizer import OptimizationRunState
from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer, MapElitesProgramAdapter


class DummyAdapter(MapElitesProgramAdapter):
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        return {"x": self.params["x"], "y": self.params["y"]}


def make_adapter() -> DummyAdapter:
    return DummyAdapter(
        params={"x": 0, "y": 0},
        search_space={"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]},
    )


def evaluate_fn(adapter: DummyAdapter) -> Dict[str, float]:
    output = adapter.execute()
    x, y = output["x"], output["y"]
    return {"score": float(x + y), "complexity": float(x), "diversity": float(y)}


def make_optimizer(adapter: DummyAdapter, **overrides) -> MapElitesOptimizer:
    config = dict(
        adapter=adapter,
        feature_dimensions=["complexity", "diversity"],
        feature_ranges={"complexity": (0.0, 4.0), "diversity": (0.0, 4.0)},
        feature_bins=5,
        exploration_ratio=0.5,
        seed=7,
    )
    config.update(overrides)
    return MapElitesOptimizer(**config)


def test_map_elites_fills_multiple_cells_and_tracks_best(tmp_path):
    adapter = make_adapter()
    optimizer = make_optimizer(adapter)

    best_adapter = optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=80,
        save_dir=str(tmp_path),
    )

    archive = optimizer.get_archive()
    assert len(archive) >= 5
    assert best_adapter.params["x"] + best_adapter.params["y"] == 8

    # Each elite is self-contained (cell, metrics, configuration) and its cell key
    # matches its stored coordinates.
    for cell_key, entry in archive.items():
        assert cell_key == ",".join(str(c) for c in entry["cell"])
        assert set(entry["unit_values"]) == {"x", "y"}
        assert entry["metrics"]["score"] == float(entry["unit_values"]["x"] + entry["unit_values"]["y"])

    # The baseline (x=0, y=0) is seeded into the archive at run start.
    assert "0,0" in archive


def test_archive_persists_and_resumes(tmp_path):
    save_dir = str(tmp_path)
    optimizer = make_optimizer(make_adapter())
    optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score"),
        max_trials=20,
        save_dir=save_dir,
    )
    cells_before = set(optimizer.get_archive())

    prior = OptimizationRunState.load_state(save_dir)
    assert set(prior.optimizer_state["archive"]) == cells_before
    assert prior.current_step == 20

    resumed = make_optimizer(make_adapter(), seed=123)
    resumed.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score"),
        max_trials=40,
        resume_from=save_dir,
        save_dir=save_dir,
    )
    after = OptimizationRunState.load_state(save_dir)
    assert after.current_step == 40
    # Resume keeps all previously discovered elites.
    assert cells_before <= set(resumed.get_archive())


def _comparable_archive(archive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Archive view without run-specific ids (snapshot ids are random per run)."""
    return {
        key: {"cell": e["cell"], "unit_values": e["unit_values"], "metrics": e["metrics"]}
        for key, e in archive.items()
    }


def _trial_change_sequence(state: OptimizationRunState) -> list:
    return [
        (r.trial_id, [(c.uid, c.value) for c in r.changes])
        for r in state.trial_records
    ]


def test_resume_is_equivalent_to_uninterrupted_run(tmp_path):
    objective = ScalarObjective(metric="score")

    # Run A: 40 trials in one go.
    dir_a = str(tmp_path / "one_shot")
    opt_a = make_optimizer(make_adapter(), seed=7)
    opt_a.optimize(evaluate_fn=evaluate_fn, objective=objective, max_trials=40, save_dir=dir_a)

    # Run B: 20 trials, then resume to 40 with a fresh optimizer instance whose
    # constructor seed differs — the checkpointed RNG state must take precedence.
    dir_b = str(tmp_path / "resumed")
    opt_b1 = make_optimizer(make_adapter(), seed=7)
    opt_b1.optimize(evaluate_fn=evaluate_fn, objective=objective, max_trials=20, save_dir=dir_b)
    opt_b2 = make_optimizer(make_adapter(), seed=999)
    opt_b2.optimize(
        evaluate_fn=evaluate_fn,
        objective=objective,
        max_trials=40,
        resume_from=dir_b,
        save_dir=dir_b,
    )

    state_a = OptimizationRunState.load_state(dir_a)
    state_b = OptimizationRunState.load_state(dir_b)
    assert _trial_change_sequence(state_a) == _trial_change_sequence(state_b)
    assert _comparable_archive(opt_a.get_archive()) == _comparable_archive(opt_b2.get_archive())
    assert state_a.best_metrics == state_b.best_metrics


def test_resume_rejects_mismatched_optimizer_config(tmp_path):
    save_dir = str(tmp_path)
    optimizer = make_optimizer(make_adapter())
    optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score"),
        max_trials=5,
        save_dir=save_dir,
    )

    mismatched = make_optimizer(make_adapter(), feature_bins=3)  # saved run used 5 bins
    with pytest.raises(ValueError, match="optimizer configuration"):
        mismatched.optimize(
            evaluate_fn=evaluate_fn,
            objective=ScalarObjective(metric="score"),
            max_trials=10,
            resume_from=save_dir,
            save_dir=save_dir,
        )


def test_resume_rejects_mismatched_objective(tmp_path):
    save_dir = str(tmp_path)
    optimizer = make_optimizer(make_adapter())
    optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=5,
        save_dir=save_dir,
    )

    resumed = make_optimizer(make_adapter())
    with pytest.raises(ValueError, match="different objective"):
        resumed.optimize(
            evaluate_fn=evaluate_fn,
            objective=ScalarObjective(metric="score", direction="minimize"),
            max_trials=10,
            resume_from=save_dir,
            save_dir=save_dir,
        )


def test_get_archive_returns_isolated_copy(tmp_path):
    optimizer = make_optimizer(make_adapter())
    optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score"),
        max_trials=10,
        save_dir=str(tmp_path),
    )
    archive = optimizer.get_archive()
    cell_key = next(iter(archive))
    archive[cell_key]["metrics"]["score"] = -999.0
    archive[cell_key]["unit_values"].clear()
    # Internal state must be unaffected by mutations of the returned copy.
    fresh = optimizer.get_archive()
    assert fresh[cell_key]["metrics"]["score"] != -999.0
    assert fresh[cell_key]["unit_values"]


def test_get_archive_and_load_elite(tmp_path):
    optimizer = make_optimizer(make_adapter())
    optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score"),
        max_trials=40,
        save_dir=str(tmp_path),
    )
    archive = optimizer.get_archive()
    cell_key, entry = next(iter(archive.items()))
    elite = optimizer.load_elite(cell_key)
    assert isinstance(elite, DummyAdapter)
    assert elite.params == entry["unit_values"]

    with pytest.raises(KeyError):
        optimizer.load_elite("999,999")

    # Archive access also works on a state loaded from disk, without a live run.
    fresh = make_optimizer(make_adapter())
    state = OptimizationRunState.load_state(str(tmp_path))
    assert set(fresh.get_archive(state)) == set(archive)
    with pytest.raises(RuntimeError):
        fresh.get_archive()  # no run yet and no state passed


def test_requires_enum_choices_on_target_units():
    class NoEnumAdapter(MapElitesProgramAdapter):
        def register_units(self):
            units = super().register_units()
            for unit in units:
                unit.json_schema = {"type": "integer"}  # drop the enum
            return units

        def execute(self, *args, **kwargs):
            return dict(self.params)

    adapter = NoEnumAdapter(params={"x": 0}, search_space={"x": [0, 1]})
    with pytest.raises(ValueError, match="enum"):
        make_optimizer(adapter)


def test_missing_feature_range_rejected():
    with pytest.raises(ValueError, match="feature_ranges"):
        make_optimizer(make_adapter(), feature_ranges={"complexity": (0.0, 4.0)})


def test_missing_feature_metric_fails_run(tmp_path):
    def bad_evaluate(adapter: DummyAdapter) -> Dict[str, float]:
        return {"score": 1.0}  # no feature-dimension metrics

    optimizer = make_optimizer(make_adapter())
    with pytest.raises(KeyError, match="complexity"):
        optimizer.optimize(
            evaluate_fn=bad_evaluate,
            objective=ScalarObjective(metric="score"),
            max_trials=2,
            save_dir=str(tmp_path),
        )


def test_adapter_validates_params_against_search_space():
    with pytest.raises(ValueError, match="not present"):
        DummyAdapter(params={"x": 9, "y": 0}, search_space={"x": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="missing or empty"):
        DummyAdapter(params={"x": 0, "y": 0}, search_space={"x": [0, 1]})
    with pytest.raises(ValueError, match="unknown parameter"):
        DummyAdapter(params={"x": 0}, search_space={"x": [0, 1], "z": [1]})
