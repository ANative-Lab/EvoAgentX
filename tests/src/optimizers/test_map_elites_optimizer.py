from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from evoagentx.optimizers.engine.decorators import EntryPoint
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer


@dataclass
class DummyProgram:
    x: int = 0
    y: int = 0

    def run(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}


def test_map_elites_fills_multiple_cells_and_tracks_best():
    program = DummyProgram()
    registry = ParamRegistry()
    registry.track(program, "x", name="x")
    registry.track(program, "y", name="y")

    @EntryPoint()
    def entry():
        return program.run()

    def evaluator(output: Dict[str, Any]) -> Dict[str, Any]:
        x = output["x"]
        y = output["y"]
        score = float(x + y)
        return {"score": score, "complexity": float(x), "diversity": float(y)}

    opt = MapElitesOptimizer(
        registry=registry,
        evaluator=evaluator,
        search_space={"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]},
        feature_dimensions=["complexity", "diversity"],
        feature_ranges={"complexity": (0.0, 4.0), "diversity": (0.0, 4.0)},
        feature_bins=5,
        n_iterations=80,
        exploration_ratio=0.5,
        random_seed=7,
    )

    best_cfg, result = opt.optimize()
    archive = result["archive"]
    best = result["best"]

    assert best_cfg is not None
    assert best is not None
    assert len(archive) >= 5
    assert best.fitness == 8.0
    assert best_cfg["x"] + best_cfg["y"] == 8


def make_seeded_optimizer(evaluator):
    """Build an optimizer with no dependency on the global entry point."""
    program = DummyProgram()
    registry = ParamRegistry()
    registry.track(program, "x", name="x")
    registry.track(program, "y", name="y")
    return MapElitesOptimizer(
        registry=registry,
        evaluator=evaluator,
        search_space={"x": list(range(5)), "y": list(range(5))},
        feature_dimensions=["complexity"],
        feature_ranges={"complexity": (0.0, 4.0)},
        n_iterations=20,
        random_seed=7,
        program=program.run,
    )


def test_seeded_search_does_not_change_global_random_state():
    """Seeding and running an optimizer must not reseed unrelated callers."""
    import random

    before = random.getstate()
    opt = make_seeded_optimizer(lambda out: {"score": out["x"] + out["y"], "complexity": out["x"]})
    opt.optimize()

    assert random.getstate() == before


def test_seeded_search_is_independent_of_evaluator_random_draws():
    """Evaluator randomness must not change the sequence of candidate mutations."""
    import random

    def evaluator(output):
        """Score a candidate without drawing random numbers."""
        return {"score": output["x"] + output["y"], "complexity": output["x"]}

    def noisy_evaluator(output):
        """Return the same score after consuming unrelated randomness."""
        random.random()
        return evaluator(output)

    baseline = make_seeded_optimizer(evaluator)
    noisy = make_seeded_optimizer(noisy_evaluator)

    _, baseline_result = baseline.optimize()
    _, noisy_result = noisy.optimize()

    assert [step["cfg"] for step in baseline_result["history"]] == [
        step["cfg"] for step in noisy_result["history"]
    ]
