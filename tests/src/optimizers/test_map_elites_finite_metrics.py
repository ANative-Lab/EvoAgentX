"""Exercise numeric validation through the actual MAP-Elites search loop."""

import math

import pytest

from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.optimizers.map_elites_optimizer import MapElitesOptimizer


def make_optimizer(evaluator, *, scalar_result=False):
    """Construct a deterministic search that requires no model or network."""
    return MapElitesOptimizer(
        registry=ParamRegistry(),
        evaluator=evaluator,
        search_space={},
        feature_dimensions=[] if scalar_result else ["length"],
        feature_ranges={} if scalar_result else {"length": (0.0, 2.0)},
        n_iterations=2,
        program=dict,
    )


@pytest.mark.parametrize("fitness", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("scalar_result", [False, True])
def test_nonfinite_fitness_is_rejected_before_archive_admission(fitness, scalar_result):
    """Invalid fitness must not become an unbeatable elite or best candidate."""
    result = fitness if scalar_result else {"score": fitness, "length": 1.0}
    optimizer = make_optimizer(lambda output: result, scalar_result=scalar_result)

    with pytest.raises(ValueError, match="fitness_key 'score' must be finite"):
        optimizer.optimize()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feature_is_rejected_before_binning(value):
    """Infinite descriptors must not be silently clipped into boundary cells."""
    optimizer = make_optimizer(lambda output: {"score": 1.0, "length": value})

    with pytest.raises(ValueError, match="feature dimension 'length' must be finite"):
        optimizer.optimize()


@pytest.mark.parametrize("fitness", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("length, cell", [(-1.0, (0,)), (0.0, (0,)), (1.0, (5,)), (2.0, (9,)), (3.0, (9,))])
def test_finite_scores_and_clipped_descriptors_keep_their_behavior(fitness, length, cell):
    """Signed fitness and finite out-of-range descriptors remain supported."""
    optimizer = make_optimizer(lambda output: {"score": fitness, "length": length})

    best_cfg, result = optimizer.optimize()

    assert best_cfg == {}
    assert result["best"].fitness == fitness
    assert result["best"].cell == cell
    assert list(result["archive"]) == [cell]
    assert [row["accepted"] for row in result["history"]] == [True, False]
    assert math.isfinite(result["best"].fitness)


def test_later_finite_improvement_replaces_an_elite():
    """Validation must preserve the optimizer's replacement and best tracking."""
    scores = iter([-1.0, 1.0])
    optimizer = make_optimizer(lambda output: {"score": next(scores), "length": 1.0})

    _, result = optimizer.optimize()

    assert [row["accepted"] for row in result["history"]] == [True, True]
    assert result["best"].fitness == 1.0
    assert result["archive"][(5,)].fitness == 1.0
