"""Data-source contracts for the optimization engine.

These are *protocols only* — no implementation, no dataset knowledge. They live
in ``optimizers/core`` because they appear in engine-facing signatures
(``DataEvaluator.__init__``), and the optimizer package must stay importable
without the host application. Concrete data sources live outside this package;
``evoagentx.benchmark.Benchmark`` is one satisfier among others.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable

__all__ = ["ExampleId", "DataSource", "MutableDataSource"]


# An example id must be a stable, pure-JSON scalar: it is the primary key of
# every observation (``ObservationKey``) and is written to report.jsonl. Strings
# are the recommended form; other scalars are tolerated but will be rendered as
# strings by the record layer.
ExampleId = Any


@runtime_checkable
class DataSource(Protocol):
    """What the engine needs from a dataset: subsets, ids, and references."""

    def get_split(
        self,
        split: str = "test",
        *,
        example_ids: Optional[List[ExampleId]] = None,
        sample_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Return the examples of ``split``, optionally narrowed to a subset.

        Args:
            split: Which split to read, e.g. ``"train"`` / ``"dev"`` / ``"test"``.
            example_ids: Explicit subset, given by id. Selection follows the
                order of this argument (failure resampling, bootstrap).
            sample_k: Size of a random subset drawn *after* ``example_ids``
                filtering. ``None`` means the whole selection.
            seed: Seed for that random draw. Sampling with the same ``seed``
                must yield the same examples, in the same order.

        Returns:
            The selected examples, in the order they are to be observed.
        """
        ...

    def get_id(self, example: Any) -> ExampleId:
        """Return the stable unique id of ``example``."""
        ...

    def get_reference(self, example_id: ExampleId) -> Any:
        """Return what a prediction for ``example_id`` is scored against.

        A ground-truth answer, a rubric, a checklist — whatever the paired
        ``Scorer`` expects. Raises ``KeyError`` for an unknown id.
        """
        ...


@runtime_checkable
class MutableDataSource(DataSource, Protocol):
    """Append-only data source (contrib layer, optional).

    Used by consumers that grow a regression suite from run events. The engine
    never appends; it calls ``freeze()`` once per run and evaluates against the
    resulting immutable view.
    """

    def append(self, examples: Iterable[Any]) -> None:
        """Append examples. Never mutates or removes existing ones."""
        ...

    def freeze(self) -> DataSource:
        """Return an immutable snapshot view for the duration of a run."""
        ...
