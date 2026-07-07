"""
MAP-Elites optimizer on the generic optimization engine.

MAP-Elites (Mouret & Clune, 2015) is a quality-diversity algorithm: instead of keeping a
single best candidate, it maintains an *archive* of elites, one per cell of a discretized
feature space. Each trial's metrics provide both a fitness signal (ranked by the run's
``Objective``) and the feature-descriptor values that decide which cell the candidate
falls into; a candidate replaces the cell's current elite only when the objective says it
is strictly better.

How this maps onto the engine:

* **Search space** — declared by the adapter: every target ``OptimizationUnit`` must carry
  a ``json_schema`` with a non-empty ``"enum"`` listing its discrete candidate values
  (:class:`MapElitesProgramAdapter` does this for a plain ``{name: value}`` config
  program). The optimizer samples exclusively from those enums, so every proposal is
  schema-valid by construction.
* **Proposals** — each round produces up to ``num_proposals_per_step`` proposals. With
  probability ``exploration_ratio`` (or while the archive is still empty) a proposal is a
  *random* configuration branched from the baseline snapshot; otherwise it *mutates* a
  uniformly chosen archive elite by replacing one unit's value with a different enum
  choice, branched from that elite's snapshot.
* **Archive** — updated in :meth:`observe` from completed trial records and stored as pure
  JSON in ``state.optimizer_state["archive"]``, so it is checkpointed with the run and
  survives ``resume_from``. The baseline is inserted in :meth:`on_run_start`.
* **Result** — ``optimize()`` returns the globally best adapter (engine convention). The
  full archive is available afterwards through :meth:`get_archive` /
  :meth:`load_elite`, or from any saved ``OptimizationRunState``.
"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..workflow.workflow_graph import WorkFlowGraph

from .engine.adapter import ProgramAdapter, SnapShot
from .engine.base import (
    ChangeOperation,
    OptimizationProposal,
    OptimizationUnit,
    OptimizationUnitType,
    TrialRecord,
    UnitChange,
)
from .engine.objective import Objective
from .engine.optimizer import Optimizer, OptimizationRunState
from .engine.utils import get_best_snapshot_id
from .workflow_adapter import WorkFlowAdapterMixin


# ---------------------------------------------------------------------------
# Optimizer: quality-diversity search over discrete, enum-declared units.
# ---------------------------------------------------------------------------
class MapElitesOptimizer(Optimizer):
    """
    MAP-Elites quality-diversity optimizer over discrete unit choices.

    Every target unit must declare its candidate values via ``json_schema["enum"]``;
    the optimizer alternates between sampling random configurations and mutating one
    unit of a randomly chosen archive elite. Trial metrics must contain a value for
    each ``feature_dimensions`` entry — those place the candidate into an archive cell
    — and the ``objective`` passed to ``optimize()`` decides both the global best and
    whether a candidate displaces a cell's current elite.

    Args:
        adapter: The program adapter to optimize. All target units must carry a
            ``json_schema`` with a non-empty ``"enum"`` of candidate values.
        feature_dimensions: Metric keys used as feature descriptors (archive axes).
        feature_ranges: ``{dimension: (min, max)}`` value range per feature dimension.
            Required for every dimension; values outside the range are clamped into
            the boundary bins.
        feature_bins: Number of bins per dimension — a single int for all dimensions,
            or a ``{dimension: bins}`` mapping (missing dimensions default to 10).
        num_proposals_per_step: Proposals generated per engine round. Keep 1 for the
            classic sequential loop; raise it together with
            ``async_optimize(execution_mode="concurrent")`` to evaluate in parallel.
        exploration_ratio: Probability in [0, 1] of sampling a fresh random
            configuration instead of mutating an archive elite (mutation additionally
            requires a non-empty archive).
        seed: Optional RNG seed for reproducible sampling.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset({
        OptimizationUnitType.FIELD,
        OptimizationUnitType.MODEL,
        OptimizationUnitType.PROMPT,
    })

    def __init__(
        self,
        adapter: ProgramAdapter,
        feature_dimensions: List[str],
        feature_ranges: Mapping[str, Tuple[float, float]],
        feature_bins: Union[int, Mapping[str, int]] = 10,
        num_proposals_per_step: int = 1,
        exploration_ratio: float = 0.2,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)

        if not feature_dimensions:
            raise ValueError("`feature_dimensions` must be a non-empty list of metric keys.")
        self.feature_dimensions = list(feature_dimensions)

        self.feature_ranges = {dim: (float(lo), float(hi)) for dim, (lo, hi) in dict(feature_ranges or {}).items()}
        missing = [dim for dim in self.feature_dimensions if dim not in self.feature_ranges]
        if missing:
            raise ValueError(
                f"Missing feature_ranges for dimensions: {missing}. "
                f"Provide feature_ranges={{dim: (min, max), ...}}"
            )

        if isinstance(feature_bins, int):
            if feature_bins < 1:
                raise ValueError(f"feature_bins must be >= 1, got {feature_bins}")
        elif isinstance(feature_bins, Mapping):
            feature_bins = {dim: int(bins) for dim, bins in feature_bins.items()}
            invalid = [dim for dim, bins in feature_bins.items() if bins < 1]
            if invalid:
                raise ValueError(f"feature_bins must be >= 1 for every dimension; invalid: {invalid}")
        else:
            raise TypeError(f"feature_bins must be an int or a mapping, got {type(feature_bins).__name__}")
        self.feature_bins = feature_bins

        if num_proposals_per_step < 1:
            raise ValueError("num_proposals_per_step must be >= 1")
        self.num_proposals_per_step = num_proposals_per_step

        if not 0.0 <= exploration_ratio <= 1.0:
            raise ValueError(f"exploration_ratio must be in [0, 1], got {exploration_ratio}")
        self.exploration_ratio = exploration_ratio

        self._rng = random.Random(seed)
        self._choices = self._collect_unit_choices()
        self._last_state: Optional[OptimizationRunState] = None

    # -- persistence / resume --------------------------------------------------
    def _config_fingerprint(self) -> Dict[str, Any]:
        """JSON view of every setting the persisted archive's semantics depend on.

        Cell keys are only meaningful under the exact binning configuration that
        produced them, and elites only under the same target units/choices, so a
        resumed run must use a matching configuration.
        """
        return {
            "feature_dimensions": list(self.feature_dimensions),
            "feature_ranges": {dim: [lo, hi] for dim, (lo, hi) in self.feature_ranges.items()},
            "feature_bins": (
                self.feature_bins if isinstance(self.feature_bins, int) else dict(self.feature_bins)
            ),
            "target_unit_uids": list(self.target_unit_uids),
            "unit_choices": {uid: list(choices) for uid, choices in self._choices.items()},
        }

    @staticmethod
    def _objective_fingerprint(objective: Objective) -> Dict[str, Any]:
        """Best-effort JSON descriptor of the objective ranking the archive's elites."""
        descriptor: Dict[str, Any] = {"class": type(objective).__qualname__}
        for attr in ("metric", "direction", "metrics", "directions"):
            value = getattr(objective, attr, None)
            if value is not None:
                descriptor[attr] = list(value) if isinstance(value, list) else value
        return descriptor

    def serialize_optimizer_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        # random.getstate() carries tuples, which JSON reloads as lists; store a
        # list-based form and rebuild the tuples in load_optimizer_state.
        version, internal, gauss_next = self._rng.getstate()
        state.optimizer_state["rng_state"] = {
            "version": version,
            "internal": list(internal),
            "gauss_next": gauss_next,
        }
        state.optimizer_state["config_fingerprint"] = self._config_fingerprint()
        return state.optimizer_state

    def load_optimizer_state(self, state: OptimizationRunState) -> None:
        saved_fingerprint = state.optimizer_state.get("config_fingerprint")
        if saved_fingerprint is not None and saved_fingerprint != self._config_fingerprint():
            raise ValueError(
                "Cannot resume MAP-Elites run: the saved run's optimizer configuration "
                "(feature_dimensions/feature_ranges/feature_bins/target units/choices) does "
                "not match this optimizer's. The persisted archive's cell keys and elites "
                f"are only valid under the original configuration.\n"
                f"saved:   {saved_fingerprint}\n"
                f"current: {self._config_fingerprint()}"
            )
        rng_state = state.optimizer_state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(
                (rng_state["version"], tuple(rng_state["internal"]), rng_state["gauss_next"])
            )

    # -- search space ---------------------------------------------------------
    def _collect_unit_choices(self) -> Dict[str, List[Any]]:
        """Read each target unit's candidate values from its json_schema enum."""
        choices: Dict[str, List[Any]] = {}
        for unit in self.target_units:
            schema = unit.json_schema
            enum = schema.get("enum") if isinstance(schema, dict) else None
            if not isinstance(enum, list) or not enum:
                raise ValueError(
                    f"MapElitesOptimizer requires every target unit to declare its discrete "
                    f"candidate values via json_schema['enum'], but unit '{unit.name}' "
                    f"(uid={unit.uid}) does not. Declare e.g. "
                    f"json_schema={{'enum': [value1, value2, ...]}} in register_units()."
                )
            choices[unit.uid] = list(enum)
        return choices

    # -- archive helpers ------------------------------------------------------
    @staticmethod
    def _cell_key(cell: Tuple[int, ...]) -> str:
        return ",".join(str(coord) for coord in cell)

    @staticmethod
    def _get_archive(state: OptimizationRunState) -> Dict[str, Dict[str, Any]]:
        return state.optimizer_state.setdefault("archive", {})

    def _bins_for_dim(self, dim: str) -> int:
        if isinstance(self.feature_bins, int):
            return self.feature_bins
        return int(self.feature_bins.get(dim, 10))

    @staticmethod
    def _bin_index(value: float, lo: float, hi: float, bins: int) -> int:
        if hi <= lo:
            return 0
        t = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        return min(int(t * bins), bins - 1)

    def _cell_from_metrics(self, metrics: Dict[str, Any]) -> Tuple[int, ...]:
        coords = []
        for dim in self.feature_dimensions:
            if dim not in metrics:
                raise KeyError(
                    f"evaluate_fn metrics missing feature dimension '{dim}'. "
                    f"feature_dimensions={self.feature_dimensions}, metrics keys={sorted(metrics)}"
                )
            value = metrics[dim]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"feature dimension '{dim}' must be numeric, got {value!r}")
            lo, hi = self.feature_ranges[dim]
            coords.append(self._bin_index(float(value), lo, hi, self._bins_for_dim(dim)))
        return tuple(coords)

    def _maybe_insert_elite(
        self,
        state: OptimizationRunState,
        record: TrialRecord,
        objective: Objective,
    ) -> bool:
        """Insert a completed trial into the archive if it beats its cell's elite."""
        if record.status != "completed" or record.metrics is None:
            return False
        cell = self._cell_from_metrics(record.metrics)
        key = self._cell_key(cell)
        archive = self._get_archive(state)
        existing = archive.get(key)
        if existing is not None and not objective.is_better(record.metrics, existing["metrics"]):
            return False
        snapshot = state.get_snapshot_by_id(record.snapshot_id) if record.snapshot_id else None
        archive[key] = {
            "cell": list(cell),
            "trial_id": record.trial_id,
            "snapshot_id": record.snapshot_id,
            "metrics": dict(record.metrics),
            # Target-unit values are duplicated here so the archive stays readable and
            # mutable even if the referenced snapshot is later pruned from the state.
            "unit_values": (
                {uid: copy.deepcopy(snapshot.unit_values[uid]) for uid in self.target_unit_uids}
                if snapshot is not None else None
            ),
        }
        return True

    # -- engine hooks ----------------------------------------------------------
    def on_run_start(self, state: OptimizationRunState, objective: Objective, **kwargs) -> None:
        """Check objective compatibility and seed the archive with the evaluated baseline."""
        # The archive's elite-replacement decisions are only consistent under the same
        # objective, so a resumed run must not silently switch metric or direction.
        objective_fingerprint = self._objective_fingerprint(objective)
        saved_objective = state.optimizer_state.get("objective_fingerprint")
        if saved_objective is not None and saved_objective != objective_fingerprint:
            raise ValueError(
                "Cannot resume MAP-Elites run with a different objective: the persisted "
                "archive's elites were selected under the saved objective.\n"
                f"saved:   {saved_objective}\n"
                f"current: {objective_fingerprint}"
            )
        state.optimizer_state["objective_fingerprint"] = objective_fingerprint

        baseline = state.get_baseline_record()
        if baseline is not None and baseline.metrics is not None:
            self._maybe_insert_elite(state, baseline, objective)

    def observe(
        self,
        state: OptimizationRunState,
        trial_records: List[TrialRecord],
        objective: Objective,
        **kwargs,
    ) -> None:
        for record in trial_records:
            self._maybe_insert_elite(state, record, objective)

    def finalize(
        self,
        state: OptimizationRunState,
        objective: Objective,
        best_adapter: Optional[ProgramAdapter],
    ) -> Any:
        # Keep the final state around so get_archive() / load_elite() work after optimize().
        self._last_state = state
        return super().finalize(state, objective, best_adapter)

    # -- proposal generation -----------------------------------------------
    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        n = self.num_proposals_per_step
        if budget_remaining is not None:
            n = max(0, min(n, budget_remaining))
        archive = self._get_archive(state)
        proposals: List[OptimizationProposal] = []
        for _ in range(n):
            # Mutation needs a parent whose snapshot is still resolvable in the state.
            parents = [
                entry for entry in archive.values()
                if entry.get("unit_values") is not None
                and entry.get("snapshot_id")
                and state.get_snapshot_by_id(entry["snapshot_id"]) is not None
            ]
            if parents and self._rng.random() > self.exploration_ratio:
                proposal = self._mutate_proposal(self._rng.choice(parents))
                if proposal is None:  # no unit has an alternative value to mutate to
                    proposal = self._random_proposal(state)
            else:
                proposal = self._random_proposal(state)
            proposals.append(proposal)
        return proposals

    def _mutate_proposal(self, parent: Dict[str, Any]) -> Optional[OptimizationProposal]:
        """Mutate one unit of an archive elite to a different enum choice."""
        values = parent["unit_values"]
        mutable_uids = [
            uid for uid in self.target_unit_uids
            if any(choice != values.get(uid) for choice in self._choices[uid])
        ]
        if not mutable_uids:
            return None
        uid = self._rng.choice(mutable_uids)
        unit = self.target_units_by_uid[uid]
        alternatives = [choice for choice in self._choices[uid] if choice != values.get(uid)]
        change = UnitChange.create(
            unit=unit,
            new_value=copy.deepcopy(self._rng.choice(alternatives)),
            old_value=values.get(uid),
            operation=ChangeOperation.REPLACE,
            metadata={"strategy": "map_elites_mutation"},
        )
        return OptimizationProposal(
            source_snapshot_id=parent["snapshot_id"],
            changes=[change],
            metadata={
                "source": "MapElitesOptimizer",
                "strategy": "mutate",
                "parent_cell": list(parent["cell"]),
                "parent_trial_id": parent["trial_id"],
            },
        )

    def _random_proposal(self, state: OptimizationRunState) -> OptimizationProposal:
        """Sample a fresh random configuration, branched from the baseline snapshot."""
        baseline = state.get_baseline_record()
        source_id = (
            baseline.snapshot_id
            if baseline is not None and baseline.snapshot_id
            else get_best_snapshot_id(state)
        )
        base_snapshot = state.get_snapshot_by_id(source_id)
        changes: List[UnitChange] = []
        for unit in self.target_units:
            new_value = copy.deepcopy(self._rng.choice(self._choices[unit.uid]))
            old_value = base_snapshot.unit_values.get(unit.uid) if base_snapshot is not None else None
            if base_snapshot is not None and new_value == old_value:
                continue  # already the base value; a REPLACE would be a no-op
            changes.append(
                UnitChange.create(
                    unit=unit,
                    new_value=new_value,
                    old_value=old_value,
                    operation=ChangeOperation.REPLACE,
                    metadata={"strategy": "map_elites_random"},
                )
            )
        return OptimizationProposal(
            source_snapshot_id=source_id,
            changes=changes,
            metadata={"source": "MapElitesOptimizer", "strategy": "random"},
        )

    # -- archive access ------------------------------------------------------
    def _resolve_state(self, state: Optional[OptimizationRunState]) -> OptimizationRunState:
        state = state or self._last_state
        if state is None:
            raise RuntimeError(
                "No optimization run state available. Run optimize()/async_optimize() first, "
                "or pass a state loaded via OptimizationRunState.load_state(save_dir)."
            )
        return state

    def get_archive(self, state: Optional[OptimizationRunState] = None) -> Dict[str, Dict[str, Any]]:
        """
        Return the MAP-Elites archive: ``{cell_key: entry}``.

        Each entry carries ``cell`` (bin coordinates), ``trial_id``, ``snapshot_id``,
        ``metrics``, and ``unit_values`` (the elite's target-unit configuration).
        ``cell_key`` is the comma-joined bin coordinates (e.g. ``"2,3"``).

        Args:
            state: Run state to read from. Defaults to the state of the last
                ``optimize()`` call on this instance.
        """
        # Deep copy so callers cannot mutate the archive inside the run state.
        return copy.deepcopy(self._get_archive(self._resolve_state(state)))

    def load_elite(self, cell_key: str, state: Optional[OptimizationRunState] = None) -> ProgramAdapter:
        """
        Reconstruct the adapter for one archive cell's elite.

        Args:
            cell_key: A key from :meth:`get_archive` (comma-joined bin coordinates).
            state: Run state to read from; defaults to the last run's state.
        """
        state = self._resolve_state(state)
        entry = self._get_archive(state).get(cell_key)
        if entry is None:
            raise KeyError(f"No archive elite for cell '{cell_key}'. Known cells: {sorted(self._get_archive(state))}")
        snapshot = state.get_snapshot_by_id(entry["snapshot_id"]) if entry.get("snapshot_id") else None
        if snapshot is None:
            raise RuntimeError(
                f"Snapshot '{entry.get('snapshot_id')}' for cell '{cell_key}' is not present in the "
                f"run state (it may have been pruned)."
            )
        return self.adapter.load_snapshot(snapshot)


# ---------------------------------------------------------------------------
# ProgramAdapter paired with MapElitesOptimizer.
# ---------------------------------------------------------------------------
class MapElitesProgramAdapter(ProgramAdapter):
    """
    Adapter for programs whose optimizable state is a flat ``{name: value}`` configuration
    drawn from discrete candidate sets, paired with :class:`MapElitesOptimizer`.

    The snapshot / merge / reconstruct plumbing is fully generic, so a user only:

    * passes ``params`` -- the current ``{name: value}`` configuration; each entry becomes
      one ``OptimizationUnit`` of type ``FIELD`` (whole-value REPLACE);
    * passes ``search_space`` -- ``{name: [candidate values]}`` declaring each parameter's
      discrete choices, exposed to optimizers as the unit's ``json_schema["enum"]``. Every
      parameter must have a non-empty candidate list containing its current value, and all
      values must be JSON-serializable so snapshots persist and resume losslessly; and
    * subclasses and implements ``execute`` -- how the program actually runs given its
      current ``self.params`` (the single piece of program-specific glue).
    """

    def __init__(self, params: Dict[str, Any], search_space: Mapping[str, List[Any]]) -> None:
        if not params:
            raise ValueError("MapElitesProgramAdapter requires at least one parameter.")
        if not all(isinstance(name, str) for name in params):
            raise TypeError("`params` must map str parameter names to values.")
        search_space = {name: list(choices) for name, choices in dict(search_space or {}).items()}

        unknown = set(search_space) - set(params)
        if unknown:
            raise ValueError(f"`search_space` contains unknown parameter name(s): {sorted(unknown)}")
        missing = [name for name in params if not search_space.get(name)]
        if missing:
            raise ValueError(
                f"`search_space` must provide a non-empty candidate list for every parameter; "
                f"missing or empty for: {missing}"
            )
        not_in_choices = [name for name, value in params.items() if value not in search_space[name]]
        if not_in_choices:
            raise ValueError(
                f"Current value(s) not present in their search_space candidates for: "
                f"{not_in_choices}. Snapshots are validated against the enum, so the current "
                f"value must be one of the declared choices."
            )

        self.params: Dict[str, Any] = dict(params)
        self.search_space: Dict[str, List[Any]] = search_space

    # -- declare optimizable units -----------------------------------------
    def register_units(self) -> List[OptimizationUnit]:
        return [
            OptimizationUnit(
                name=name,
                uid=name,
                unit_type=OptimizationUnitType.FIELD,
                json_schema={"enum": self.search_space[name]},
                allowed_operations=[ChangeOperation.REPLACE],
            )
            for name in self.params
        ]

    # -- snapshot / merge / reconstruct (all generic) ----------------------
    def take_snapshot(self) -> SnapShot:
        # The search space is non-optimizable config; carry it in program_config so it
        # survives snapshot persistence and reconstruction.
        return SnapShot(
            unit_values=copy.deepcopy(self.params),
            program_config={"search_space": copy.deepcopy(self.search_space)},
        )

    def merge_changes(self, snapshot: SnapShot, changes: List[UnitChange], **kwargs) -> SnapShot:
        new_values = copy.deepcopy(snapshot.unit_values)
        for change in changes:
            new_values[change.uid] = change.value  # REPLACE-only
        return SnapShot(unit_values=new_values, program_config=snapshot.program_config)

    def from_snapshot(self, snapshot: SnapShot, **kwargs) -> "MapElitesProgramAdapter":
        # type(self) reconstructs the concrete subclass, provided it keeps this
        # constructor signature. A subclass with a different __init__ must override.
        search_space = (snapshot.program_config or {}).get("search_space") or self.search_space
        return type(self)(params=dict(snapshot.unit_values), search_space=search_space)

    # -- run the program ----------------------------------------------------
    def execute(self, *args, **kwargs) -> Any:
        """Run the program using ``self.params``; subclasses must implement this."""
        raise NotImplementedError(
            "Subclass MapElitesProgramAdapter and implement execute() to run your program "
            "using self.params. See ProgramAdapter.execute for the reentrancy contract."
        )


__all__ = ["MapElitesOptimizer", "MapElitesProgramAdapter"]
