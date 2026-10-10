from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from .engine.base import BaseOptimizer
from .engine.decorators import EntryPoint
from .engine.registry import ParamRegistry


Metrics = Dict[str, Any]
EvaluatorReturn = Union[float, Metrics]


@dataclass(frozen=True)
class ArchiveEntry:
    cfg: Dict[str, Any]
    fitness: float
    metrics: Metrics
    cell: Tuple[int, ...]


class MapElitesOptimizer(BaseOptimizer):
    def __init__(
        self,
        registry: ParamRegistry,
        evaluator: Callable[[Dict[str, Any]], EvaluatorReturn],
        search_space: Mapping[str, List[Any]],
        *,
        feature_dimensions: List[str],
        feature_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
        feature_bins: Union[int, Mapping[str, int]] = 10,
        fitness_key: str = "score",
        n_iterations: int = 200,
        exploration_ratio: float = 0.2,
        random_seed: Optional[int] = None,
        program: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        super().__init__(registry=registry, program=program, evaluator=evaluator)
        self.search_space = dict(search_space)
        self.feature_dimensions = list(feature_dimensions)
        self.feature_ranges = dict(feature_ranges or {})
        self.feature_bins = feature_bins
        self.fitness_key = fitness_key
        self.n_iterations = n_iterations
        self.exploration_ratio = exploration_ratio
        self.random_seed = random_seed

        self._rng = random.Random(random_seed)

        missing = [d for d in self.feature_dimensions if d not in self.feature_ranges]
        if missing:
            raise ValueError(
                f"Missing feature_ranges for dimensions: {missing}. "
                f"Provide feature_ranges={{dim: (min, max), ...}}"
            )

    def optimize(self):
        if self.program is None:
            self.program = EntryPoint.get_entry()
        if self.program is None:
            raise RuntimeError("No entry function provided or registered.")

        archive: Dict[Tuple[int, ...], ArchiveEntry] = {}
        history: List[Dict[str, Any]] = []

        best_entry: Optional[ArchiveEntry] = None

        for step in range(self.n_iterations):
            if archive and self._rng.random() > self.exploration_ratio:
                parent = self._rng.choice(list(archive.values()))
                cfg = self._mutate_cfg(parent.cfg)
                source = "mutate"
            else:
                cfg = self._random_cfg()
                source = "random"

            self.apply_cfg(cfg)
            output = self.program()
            metrics, fitness = self._evaluate(output)
            cell = self._cell_from_metrics(metrics)

            accepted = False
            existing = archive.get(cell)
            if existing is None or fitness > existing.fitness:
                entry = ArchiveEntry(cfg=cfg, fitness=fitness, metrics=metrics, cell=cell)
                archive[cell] = entry
                accepted = True
                if best_entry is None or entry.fitness > best_entry.fitness:
                    best_entry = entry

            history.append(
                {
                    "step": step,
                    "source": source,
                    "cfg": cfg,
                    "fitness": fitness,
                    "metrics": metrics,
                    "cell": cell,
                    "accepted": accepted,
                }
            )

        best_cfg = best_entry.cfg if best_entry is not None else None
        return best_cfg, {"archive": archive, "history": history, "best": best_entry}

    def _evaluate(self, output: Dict[str, Any]) -> Tuple[Metrics, float]:
        result = self.evaluator(output)
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            metrics = {self.fitness_key: float(result)}
        elif isinstance(result, dict):
            metrics = dict(result)
        else:
            raise TypeError("evaluator must return float or dict metrics")

        if self.fitness_key not in metrics:
            raise KeyError(f"evaluator metrics missing fitness_key '{self.fitness_key}'")
        fitness = metrics[self.fitness_key]
        if not isinstance(fitness, (int, float)) or isinstance(fitness, bool):
            raise TypeError(f"fitness_key '{self.fitness_key}' must be numeric")
        return metrics, float(fitness)

    def _bins_for_dim(self, dim: str) -> int:
        if isinstance(self.feature_bins, int):
            return self.feature_bins
        return int(self.feature_bins.get(dim, 10))

    def _cell_from_metrics(self, metrics: Metrics) -> Tuple[int, ...]:
        coords = []
        for dim in self.feature_dimensions:
            if dim not in metrics:
                raise KeyError(
                    f"evaluator metrics missing feature dimension '{dim}'. "
                    f"feature_dimensions={self.feature_dimensions}"
                )
            value = metrics[dim]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"feature dimension '{dim}' must be numeric")
            lo, hi = self.feature_ranges[dim]
            bins = max(1, self._bins_for_dim(dim))
            coords.append(self._bin_index(float(value), lo, hi, bins))
        return tuple(coords)

    @staticmethod
    def _bin_index(value: float, lo: float, hi: float, bins: int) -> int:
        if hi <= lo:
            return 0
        t = (value - lo) / (hi - lo)
        if t < 0:
            t = 0.0
        elif t > 1:
            t = 1.0
        idx = int(t * bins)
        if idx >= bins:
            idx = bins - 1
        return idx

    def _random_cfg(self) -> Dict[str, Any]:
        return {name: copy.deepcopy(self._rng.choice(values)) for name, values in self.search_space.items()}

    def _mutate_cfg(self, parent_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not self.search_space:
            return copy.deepcopy(parent_cfg)
        cfg = copy.deepcopy(parent_cfg)
        key = self._rng.choice(list(self.search_space.keys()))
        choices = self.search_space[key]
        if len(choices) <= 1:
            return cfg
        current = cfg.get(key)
        alternatives = [v for v in choices if v != current]
        cfg[key] = copy.deepcopy(self._rng.choice(alternatives)) if alternatives else current
        return cfg
