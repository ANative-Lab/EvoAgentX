"""
Minimal MAP-Elites demo on the new optimization engine (no LLM calls).

The "program" is a dummy with two integer parameters ``x`` and ``y``. The evaluator
returns a fitness (``score = x + y``, ranked by the ScalarObjective) plus two feature
descriptors (``complexity = x``, ``diversity = y``) that place each candidate into a
5x5 archive grid. MAP-Elites fills the grid with the best candidate found per cell while
the engine tracks the single globally best configuration.

Run with::

    PYTHONPATH=. python examples/optimization/map_elites/map_elites_demo.py
"""

from typing import Any, Dict

from evoagentx.optimizers.engine.objective import ScalarObjective
from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer, MapElitesProgramAdapter


class DummyAdapter(MapElitesProgramAdapter):
    """The dummy program: its output is simply its current (x, y) configuration."""

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        return {"x": self.params["x"], "y": self.params["y"]}


def evaluate_fn(adapter: DummyAdapter) -> Dict[str, float]:
    output = adapter.execute()
    x, y = output["x"], output["y"]
    return {
        "score": float(x + y),       # fitness (objective metric)
        "complexity": float(x),      # feature dimension 1
        "diversity": float(y),       # feature dimension 2
    }


def main():
    adapter = DummyAdapter(
        params={"x": 0, "y": 0},
        search_space={"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]},
    )

    optimizer = MapElitesOptimizer(
        adapter=adapter,
        feature_dimensions=["complexity", "diversity"],
        feature_ranges={"complexity": (0.0, 4.0), "diversity": (0.0, 4.0)},
        feature_bins=5,
        exploration_ratio=0.5,
        seed=7,
    )

    best_adapter: DummyAdapter = optimizer.optimize(
        evaluate_fn=evaluate_fn,
        objective=ScalarObjective(metric="score", direction="maximize"),
        max_trials=60,
        save_dir="./debug/optimizers/map_elites_demo_run",
    )

    archive = optimizer.get_archive()
    print("Archive cells:", len(archive))
    print("Best cfg:", best_adapter.params)
    print("Best metrics:", evaluate_fn(best_adapter))

    # Each archive cell holds the best configuration found for that feature combination.
    for cell_key in sorted(archive):
        entry = archive[cell_key]
        print(f"  cell ({cell_key}): cfg={entry['unit_values']} metrics={entry['metrics']}")


if __name__ == "__main__":
    main()
