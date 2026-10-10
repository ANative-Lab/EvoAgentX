import itertools
import random

import pytest

from evoagentx.optimizers.evoprompt_optimizer import GAOptimizer


@pytest.fixture
def optimizer():
    """Create a sampler without initializing the LLM-backed evolution agents."""
    optimizer = GAOptimizer.__new__(GAOptimizer)
    optimizer.population_size = 5
    optimizer.combination_sample_size = None
    return optimizer


@pytest.mark.parametrize("node_count,prompts_per_node", [(16, 16), (19, 10)])
@pytest.mark.parametrize("sample_size", [None, 7])
def test_sample_large_combination_space(optimizer, monkeypatch, node_count, prompts_per_node, sample_size):
    """Large search spaces still yield the requested number of valid candidates."""
    optimizer.combination_sample_size = sample_size
    populations = {
        f"node_{i}": [f"prompt_{j}" for j in range(prompts_per_node)]
        for i in range(node_count)
    }
    monkeypatch.setattr(random, "choice", random.Random(0).choice)

    def reject_full_enumeration(*args):
        """Fail immediately if sampling tries to enumerate the entire space."""
        pytest.fail("Large combination spaces must be sampled, not enumerated")

    monkeypatch.setattr(itertools, "product", reject_full_enumeration)

    combinations = optimizer._generate_combinations(populations)

    assert len(combinations) == (sample_size or optimizer.population_size)
    assert len({tuple(sorted(combo.items())) for combo in combinations}) == len(combinations)
    for combo in combinations:
        assert combo.keys() == populations.keys()
        assert all(prompt in populations[node] for node, prompt in combo.items())


def test_small_combination_space_is_enumerated(optimizer):
    """Return the full Cartesian product when it fits within the sample size."""
    populations = {"planner": ["plan_a", "plan_b"], "solver": ["solve_a", "solve_b"]}

    combinations = optimizer._generate_combinations(populations)

    assert combinations == [
        dict(zip(populations, prompts)) for prompts in itertools.product(*populations.values())
    ]


def test_empty_node_population_has_no_combinations(optimizer):
    """A workflow with an empty node population cannot produce a candidate."""
    assert optimizer._generate_combinations({"planner": ["plan"], "solver": []}) == []
