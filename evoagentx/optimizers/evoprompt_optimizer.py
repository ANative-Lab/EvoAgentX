# -----------------------------------------------------------------------------
# This file re-implements algorithms from the EvoPrompt project:
#   Repo: https://github.com/beeevita/EvoPrompt
#   Paper: "Connecting Large Language Models with Evolutionary Algorithms
#           Yields Powerful Prompt Optimizers"
#   Authors: Qingyan et al.
#
# Re-implementation integrated into EvoAgentX with permission from the authors.
# All mistakes or modifications are our own.
# -----------------------------------------------------------------------------
"""
EvoPrompt optimizers on the generic optimization engine.

Both optimizers evolve a *population* of prompt configurations. One population
member is a full assignment of every target prompt unit (a "combination" in the
original EvoPrompt code), which maps 1:1 onto an engine ``SnapShot``; evaluating
one member is one engine trial.

Run structure (both algorithms):

1. **Initialization** — the first ``batch_propose`` call paraphrases each target
   prompt into ``population_size - 1`` variants per unit, samples that many
   distinct combinations, and proposes them. Together with the engine-evaluated
   baseline they form the initial population (kept in ``state.optimizer_state``,
   so runs are resumable). These initial evaluations consume ``max_trials``
   budget like any other trial.
2. **Evolution** — every subsequent ``batch_propose`` call is one generation:
   the subclass plans one LLM-evolved child per slot, the engine evaluates them,
   and ``observe`` applies the algorithm's selection rule to form the next
   population.
3. **Early stopping** — when the population's best has not strictly improved for
   ``early_stopping_patience`` consecutive generations, ``should_stop`` ends the
   run before ``max_trials`` is exhausted.

:class:`EvoPromptGAOptimizer` (genetic algorithm) crossovers two random parents
per child and keeps the top ``population_size`` of parents + children.
:class:`EvoPromptDEOptimizer` (differential evolution) evolves one child per
member from two donors and the population best, and replaces a member only when
its child strictly beats it.
"""

from __future__ import annotations

import asyncio
import random
import re
from functools import cmp_to_key
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple

from ..core.logging import logger
from ..models.base_model import BaseLLM
from ..models.model_utils import create_llm_instance
from .engine.adapter import ProgramAdapter, SnapShot
from .engine.base import (
    ChangeOperation,
    OptimizationProposal,
    TrialRecord,
    OptimizationUnitType,
    UnitChange,
)
from .engine.objective import Objective
from .engine.optimizer import Optimizer, OptimizationRunState
from .engine.utils import get_best_snapshot


# ---------------------------------------------------------------------------
# LLM prompts. ``[[...]]`` markers are substitution sentinels (str.replace, not
# str.format) so literal ``{variable}`` placeholders in evolved prompts survive.
# ---------------------------------------------------------------------------
_PARAPHRASE_SYSTEM_PROMPT = (
    "You are an expert at rephrasing instructions. Generate a semantically equivalent "
    "but differently worded version of the instruction you are given. Preserve the "
    "original intent and any required input/output format — including placeholder "
    "tokens such as {variable} and any XML tag structure. Respond with ONLY the "
    "paraphrased instruction — no explanations, no surrounding quotes or markdown fences."
)

_PARAPHRASE_USER_PROMPT = (
    "Instruction:\n[[INSTRUCTION]]\n\n"
    "Provide the paraphrased instruction."
)

_EVOLUTION_SYSTEM_PROMPT = (
    "You are an expert prompt engineer evolving prompts for an AI system with an "
    "evolutionary algorithm. Follow the numbered steps you are given to derive a new "
    "prompt. Always preserve the prompts' intent and any required input/output format — "
    "including placeholder tokens such as {variable} and any XML tag structure. "
    "Respond with ONLY the final evolved prompt text — no step-by-step reasoning, no "
    "explanations, no surrounding quotes or markdown fences."
)

# Genetic algorithm: crossover two parents, then mutate (cf. EvoPrompt GA).
_GA_USER_PROMPT = (
    "Please follow the instructions step-by-step to generate a better prompt.\n"
    "1. Crossover the following prompts to generate a new prompt:\n"
    "Prompt 1: [[PARENT1]]\n"
    "Prompt 2: [[PARENT2]]\n"
    "2. Mutate the prompt generated in Step 1 and produce the final evolved prompt.\n\n"
    "Now respond with ONLY the final evolved prompt."
)

# Differential evolution: mutate donor difference, combine with best, crossover
# with the current prompt (cf. EvoPrompt DE).
_DE_USER_PROMPT = (
    "Please follow the instructions step-by-step to generate a better prompt using the "
    "Differential Evolution strategy.\n"
    "1. Identify the parts that differ between these two donor prompts:\n"
    "Donor Prompt 1: [[DONOR1]]\n"
    "Donor Prompt 2: [[DONOR2]]\n"
    "2. Randomly mutate the differing parts identified in Step 1.\n"
    "3. Combine the mutated parts with the best prompt, selectively replacing its content:\n"
    "Best Prompt: [[BEST]]\n"
    "4. Crossover the result of Step 3 with the current prompt to generate the final "
    "evolved prompt:\n"
    "Current Prompt: [[CURRENT]]\n\n"
    "Now respond with ONLY the final evolved prompt."
)


class EvoPromptOptimizer(Optimizer):
    """
    Shared machinery for EvoPrompt-style population-based prompt evolution.

    Not usable directly — instantiate :class:`EvoPromptGAOptimizer` or
    :class:`EvoPromptDEOptimizer`. Subclasses implement the per-generation child
    planning (`_plan_children`) and the survivor-selection rule (`_select_survivors`).

    The population is stored in ``state.optimizer_state`` as
    ``[{"snapshot_id": ..., "metrics": ...}]`` — pure JSON, so it is checkpointed
    with the run state and survives ``resume_from``. Member prompt values are
    resolved through the engine's snapshot store.

    Args:
        adapter: The program adapter whose PROMPT units are optimized.
        llm: A ``BaseLLM`` instance driving paraphrasing and evolution. Provide
            this or ``llm_config``; ``llm`` wins when both are given.
        llm_config: An ``LLMConfig`` used to build the evolution LLM when ``llm``
            is not supplied.
        population_size: Number of configurations kept in the population (>= 2).
        concurrency_limit: Maximum concurrent LLM paraphrase/evolution requests in
            the async proposal path (``async_optimize``); None means unlimited.
            This is separate from ``async_optimize(..., max_workers=...)``, which
            only throttles trial evaluations.
        enable_early_stopping: Stop when the population best stops improving.
        early_stopping_patience: Generations without strict improvement tolerated
            before stopping.
        seed: Optional RNG seed for reproducible parent/donor/combination sampling.
    """

    supported_unit_types: ClassVar[FrozenSet[OptimizationUnitType]] = frozenset(
        {OptimizationUnitType.PROMPT}
    )

    def __init__(
        self,
        adapter: ProgramAdapter,
        llm: Optional[BaseLLM] = None,
        llm_config: Optional[Any] = None,
        population_size: int = 6,
        concurrency_limit: Optional[int] = 10,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 3,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if type(self) is EvoPromptOptimizer:
            raise TypeError(
                "EvoPromptOptimizer is a shared base; instantiate EvoPromptGAOptimizer "
                "or EvoPromptDEOptimizer instead."
            )
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 2:
            raise ValueError(f"population_size must be an integer >= 2, got {population_size!r}")
        if not isinstance(early_stopping_patience, int) or early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be an integer >= 1, got {early_stopping_patience!r}")
        if concurrency_limit is not None and (
            not isinstance(concurrency_limit, int) or isinstance(concurrency_limit, bool) or concurrency_limit < 1
        ):
            raise ValueError(f"concurrency_limit must be a positive integer or None, got {concurrency_limit!r}")
        self.population_size = population_size
        self.concurrency_limit = concurrency_limit
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self._rng = random.Random(seed)
        self.llm = self._build_llm(llm, llm_config)

    @staticmethod
    def _build_llm(llm: Optional[BaseLLM], llm_config: Optional[Any]) -> BaseLLM:
        """Return the evolution LLM, building one from `llm_config` when needed."""
        if llm is None and llm_config is None:
            raise ValueError("Provide at least one of `llm` or `llm_config`.")
        if llm is not None:
            if not isinstance(llm, BaseLLM):
                raise TypeError(f"`llm` must be a BaseLLM instance, got {type(llm).__name__}.")
            if llm_config is not None and llm_config is not llm.config:
                logger.warning(
                    "Both `llm` and `llm_config` were provided; using `llm` and ignoring "
                    "the passed `llm_config`."
                )
            return llm
        return create_llm_instance(llm_config)

    # -- LLM helpers ---------------------------------------------------------
    @staticmethod
    def _clean_variant(text: str) -> str:
        """Strip stray markdown fences / wrapping quotes the LLM may add."""
        cleaned = (text or "").strip()
        fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _generate(self, messages: List[dict]) -> str:
        return self._clean_variant(self.llm.single_generate(messages=messages, output_response=False))

    async def _async_generate(self, messages: List[dict]) -> str:
        return self._clean_variant(
            await self.llm.single_generate_async(messages=messages, output_response=False)
        )

    async def _gather_generations(self, messages_list: List[List[dict]]) -> List[str]:
        """Run LLM calls concurrently, throttled by `concurrency_limit`."""
        if self.concurrency_limit is None:
            return await asyncio.gather(*[self._async_generate(m) for m in messages_list])
        sem = asyncio.Semaphore(self.concurrency_limit)

        async def bounded(messages: List[dict]) -> str:
            async with sem:
                return await self._async_generate(messages)

        return await asyncio.gather(*[bounded(m) for m in messages_list])

    @staticmethod
    def _paraphrase_messages(prompt: str) -> List[dict]:
        return [
            {"role": "system", "content": _PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": _PARAPHRASE_USER_PROMPT.replace("[[INSTRUCTION]]", prompt)},
        ]

    # -- optimizer state -----------------------------------------------------
    def _ensure_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        ost = state.optimizer_state
        ost.setdefault("phase", "init")
        ost.setdefault("population", [])
        ost.setdefault("generation", 0)
        ost.setdefault("best_metrics", None)
        ost.setdefault("no_improvement", 0)
        return ost

    def serialize_optimizer_state(self, state: OptimizationRunState) -> Dict[str, Any]:
        # Persist the RNG state alongside the population so a resumed run continues
        # the same parent/donor/combination sampling sequence instead of restarting
        # from the constructor seed.
        ost = dict(state.optimizer_state)
        version, internal_state, gauss_next = self._rng.getstate()
        ost["rng_state"] = [version, list(internal_state), gauss_next]
        return ost

    def load_optimizer_state(self, state: OptimizationRunState) -> None:
        rng_state = state.optimizer_state.get("rng_state")
        if rng_state:
            version, internal_state, gauss_next = rng_state
            self._rng.setstate((version, tuple(internal_state), gauss_next))

    def on_run_start(self, state: OptimizationRunState, objective: Objective, **kwargs) -> None:
        # Seed the population with the engine-evaluated baseline: the original
        # configuration is a population member for free, mirroring the original
        # EvoPrompt behavior of including the initial prompt in the population.
        ost = self._ensure_state(state)
        if ost["phase"] != "init" or ost["population"]:
            return
        baseline = state.get_baseline_record()
        if baseline is not None and baseline.status == "completed" and baseline.metrics is not None:
            ost["population"].append(
                {"snapshot_id": baseline.snapshot_id, "metrics": baseline.metrics}
            )

    def _sort_by_objective(
        self,
        items: List[Any],
        objective: Objective,
        metrics_of: Callable[[Any], Dict[str, Any]] = lambda member: member["metrics"],
    ) -> List[Any]:
        """Sort `items` best-first using only the objective's pairwise `is_better`."""
        def compare(a: Any, b: Any) -> int:
            metrics_a, metrics_b = metrics_of(a), metrics_of(b)
            if objective.is_better(metrics_a, metrics_b):
                return -1
            if objective.is_better(metrics_b, metrics_a):
                return 1
            return 0
        return sorted(items, key=cmp_to_key(compare))

    def _live_population(
        self, state: OptimizationRunState, ost: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Population members whose snapshots are still materializable, as
        ``{"member": ..., "snapshot": SnapShot}`` entries."""
        entries = []
        for member in ost["population"]:
            snapshot = state.get_snapshot_by_id(member["snapshot_id"])
            if snapshot is not None:
                entries.append({"member": member, "snapshot": snapshot})
        return entries

    def _baseline_snapshot(self, state: OptimizationRunState) -> SnapShot:
        record = state.get_baseline_record()
        if record is not None and record.snapshot_id:
            snapshot = state.get_snapshot_by_id(record.snapshot_id)
            if snapshot is not None:
                return snapshot
        return get_best_snapshot(state)

    # -- proposal construction -----------------------------------------------
    def _make_proposal(
        self,
        source_snapshot: SnapShot,
        new_values: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> Optional[OptimizationProposal]:
        """Build one proposal replacing target units; drop empty / no-op changes."""
        changes: List[UnitChange] = []
        for unit in self.target_units:
            current = source_snapshot.unit_values[unit.uid]
            value = new_values.get(unit.uid, "")
            if not value or value == current:
                continue
            changes.append(
                UnitChange.create(
                    unit=unit,
                    new_value=value,
                    old_value=current,
                    operation=ChangeOperation.REPLACE,
                    metadata={"strategy": metadata.get("strategy", "evoprompt")},
                )
            )
        if not changes:
            return None
        return OptimizationProposal(
            source_snapshot_id=source_snapshot.snapshot_id,
            changes=changes,
            metadata=metadata,
        )

    # -- initialization phase --------------------------------------------------
    def _init_budget(self, budget_remaining: Optional[int]) -> int:
        """Initial combinations to propose: the baseline already fills one population
        slot, and the remaining trial budget caps the rest (so no paraphrase tokens
        are spent on combinations that could never be evaluated)."""
        n = self.population_size - 1
        if budget_remaining is not None:
            n = min(n, budget_remaining)
        return n

    def _init_paraphrase_jobs(
        self, baseline_values: Dict[str, Any], num_variants: int
    ) -> List[Tuple[str, List[dict]]]:
        """One (uid, messages) paraphrase job per node-population slot to fill."""
        jobs: List[Tuple[str, List[dict]]] = []
        for unit in self.target_units:
            for _ in range(num_variants):
                jobs.append((unit.uid, self._paraphrase_messages(baseline_values[unit.uid])))
        return jobs

    def _assemble_init_proposals(
        self,
        baseline_snapshot: SnapShot,
        node_populations: Dict[str, List[str]],
        n: int,
    ) -> List[OptimizationProposal]:
        """Sample up to `n` distinct combinations from the node populations and propose them."""
        baseline_key = tuple(sorted(
            (unit.uid, baseline_snapshot.unit_values[unit.uid]) for unit in self.target_units
        ))
        combos: List[Dict[str, str]] = []
        seen = {baseline_key}
        attempts, max_attempts = 0, max(n * 5, 10)
        while len(combos) < n and attempts < max_attempts:
            combo = {uid: self._rng.choice(pool) for uid, pool in node_populations.items()}
            key = tuple(sorted(combo.items()))
            if key not in seen:
                seen.add(key)
                combos.append(combo)
            attempts += 1
        proposals = []
        for index, combo in enumerate(combos):
            proposal = self._make_proposal(
                baseline_snapshot,
                combo,
                metadata={
                    "source": self.__class__.__name__,
                    "phase": "init",
                    "variant_index": index,
                },
            )
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # -- algorithm-specific hooks ----------------------------------------------
    def _plan_children(
        self,
        state: OptimizationRunState,
        objective: Objective,
        ost: Dict[str, Any],
        budget_remaining: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Plan one generation of children.

        Return a list of plans, each
        ``{"source_snapshot": SnapShot, "unit_messages": {uid: chat_messages},
        "metadata": {...}}``. The base class runs the LLM calls (sync or
        concurrently in async mode) and turns each plan into a proposal.
        """
        raise NotImplementedError

    def _select_survivors(
        self,
        population: List[Dict[str, Any]],
        records: List[TrialRecord],
        objective: Objective,
    ) -> List[Dict[str, Any]]:
        """Apply the algorithm's selection rule and return the next population."""
        raise NotImplementedError

    # -- proposal generation ----------------------------------------------------
    def batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        ost = self._ensure_state(state)
        if ost["phase"] == "init":
            baseline_snapshot = self._baseline_snapshot(state)
            n = self._init_budget(budget_remaining)
            node_populations = {
                unit.uid: [baseline_snapshot.unit_values[unit.uid]] for unit in self.target_units
            }
            for uid, messages in self._init_paraphrase_jobs(baseline_snapshot.unit_values, n):
                variant = self._generate(messages)
                if variant:
                    node_populations[uid].append(variant)
            return self._assemble_init_proposals(baseline_snapshot, node_populations, n)

        proposals: List[OptimizationProposal] = []
        for plan in self._plan_children(state, objective, ost, budget_remaining):
            new_values = {
                uid: self._generate(messages) for uid, messages in plan["unit_messages"].items()
            }
            proposal = self._make_proposal(plan["source_snapshot"], new_values, plan["metadata"])
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    async def async_batch_propose(
        self,
        state: OptimizationRunState,
        objective: Objective,
        budget_remaining: Optional[int] = None,
        **kwargs,
    ) -> List[OptimizationProposal]:
        ost = self._ensure_state(state)
        if ost["phase"] == "init":
            baseline_snapshot = self._baseline_snapshot(state)
            n = self._init_budget(budget_remaining)
            node_populations = {
                unit.uid: [baseline_snapshot.unit_values[unit.uid]] for unit in self.target_units
            }
            jobs = self._init_paraphrase_jobs(baseline_snapshot.unit_values, n)
            variants = await self._gather_generations([messages for _, messages in jobs])
            for (uid, _), variant in zip(jobs, variants):
                if variant:
                    node_populations[uid].append(variant)
            return self._assemble_init_proposals(baseline_snapshot, node_populations, n)

        plans = self._plan_children(state, objective, ost, budget_remaining)
        # Evolve every (plan, unit) pair concurrently, then regroup per plan.
        index: List[Tuple[int, str]] = []
        tasks_messages: List[List[dict]] = []
        for plan_idx, plan in enumerate(plans):
            for uid, messages in plan["unit_messages"].items():
                index.append((plan_idx, uid))
                tasks_messages.append(messages)
        results = await self._gather_generations(tasks_messages)

        values_per_plan: Dict[int, Dict[str, str]] = {i: {} for i in range(len(plans))}
        for (plan_idx, uid), value in zip(index, results):
            values_per_plan[plan_idx][uid] = value

        proposals: List[OptimizationProposal] = []
        for plan_idx, plan in enumerate(plans):
            proposal = self._make_proposal(
                plan["source_snapshot"], values_per_plan[plan_idx], plan["metadata"]
            )
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # -- selection & early stopping ---------------------------------------------
    def observe(
        self,
        state: OptimizationRunState,
        trial_records: List[TrialRecord],
        objective: Objective,
        **kwargs,
    ) -> None:
        ost = self._ensure_state(state)
        completed = [
            record for record in trial_records
            if record.status == "completed" and record.metrics is not None and record.snapshot_id
        ]
        is_init_batch = ost["phase"] == "init"
        if is_init_batch:
            members = [
                {"snapshot_id": record.snapshot_id, "metrics": record.metrics}
                for record in completed
            ]
            population = self._sort_by_objective(
                ost["population"] + members, objective
            )[: self.population_size]
            ost["phase"] = "evolve"
        else:
            population = self._select_survivors(list(ost["population"]), completed, objective)
            ost["generation"] += 1
        ost["population"] = population

        if not population:
            return
        best = self._sort_by_objective(population, objective)[0]
        if ost["best_metrics"] is None or objective.is_better(best["metrics"], ost["best_metrics"]):
            ost["best_metrics"] = best["metrics"]
            ost["no_improvement"] = 0
        elif not is_init_batch:
            ost["no_improvement"] += 1
            logger.info(
                f"EvoPrompt: no population improvement for {ost['no_improvement']} generation(s) "
                f"(patience={self.early_stopping_patience})."
            )

    def should_stop(self, state: OptimizationRunState, objective: Objective, **kwargs) -> bool:
        if not self.enable_early_stopping:
            return False
        ost = self._ensure_state(state)
        if ost["no_improvement"] >= self.early_stopping_patience:
            logger.info(
                f"EvoPrompt: early stopping after {ost['generation']} generation(s); no "
                f"improvement for {ost['no_improvement']} consecutive generation(s)."
            )
            return True
        return False


class EvoPromptGAOptimizer(EvoPromptOptimizer):
    """
    Genetic-algorithm EvoPrompt: each generation crossovers two random parent
    configurations per child (one LLM crossover+mutation call per prompt unit)
    and keeps the top ``population_size`` of parents + children.
    """

    def _plan_children(
        self,
        state: OptimizationRunState,
        objective: Objective,
        ost: Dict[str, Any],
        budget_remaining: Optional[int],
    ) -> List[Dict[str, Any]]:
        live = self._live_population(state, ost)
        if not live:
            return []
        n = self.population_size
        if budget_remaining is not None:
            n = min(n, budget_remaining)
        plans: List[Dict[str, Any]] = []
        for child_index in range(n):
            if len(live) >= 2:
                parent1, parent2 = self._rng.sample(live, 2)
            else:
                parent1 = parent2 = live[0]
            unit_messages = {}
            for unit in self.target_units:
                user = (
                    _GA_USER_PROMPT
                    .replace("[[PARENT1]]", parent1["snapshot"].unit_values[unit.uid])
                    .replace("[[PARENT2]]", parent2["snapshot"].unit_values[unit.uid])
                )
                unit_messages[unit.uid] = [
                    {"role": "system", "content": _EVOLUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ]
            plans.append({
                "source_snapshot": parent1["snapshot"],
                "unit_messages": unit_messages,
                "metadata": {
                    "source": self.__class__.__name__,
                    "strategy": "ga",
                    "generation": ost["generation"] + 1,
                    "variant_index": child_index,
                    "parents": [
                        parent1["member"]["snapshot_id"],
                        parent2["member"]["snapshot_id"],
                    ],
                },
            })
        return plans

    def _select_survivors(
        self,
        population: List[Dict[str, Any]],
        records: List[TrialRecord],
        objective: Objective,
    ) -> List[Dict[str, Any]]:
        children = [
            {"snapshot_id": record.snapshot_id, "metrics": record.metrics}
            for record in records
        ]
        return self._sort_by_objective(population + children, objective)[: self.population_size]


class EvoPromptDEOptimizer(EvoPromptOptimizer):
    """
    Differential-evolution EvoPrompt: each generation evolves one trial child per
    population member from two random donors and the population best (one LLM DE
    call per prompt unit), and a member is replaced only when its child strictly
    beats it under the objective.
    """

    def _plan_children(
        self,
        state: OptimizationRunState,
        objective: Objective,
        ost: Dict[str, Any],
        budget_remaining: Optional[int],
    ) -> List[Dict[str, Any]]:
        live = self._live_population(state, ost)
        if not live:
            return []
        n = len(live)
        if budget_remaining is not None:
            n = min(n, budget_remaining)
        best_entry = self._sort_by_objective(
            live, objective, metrics_of=lambda entry: entry["member"]["metrics"]
        )[0]
        plans: List[Dict[str, Any]] = []
        for target in live[:n]:
            others = [entry for entry in live if entry is not target]
            if len(others) >= 2:
                donors = self._rng.sample(others, 2)
            elif others:
                donors = self._rng.choices(others, k=2)
            else:
                donors = [target, target]
            unit_messages = {}
            for unit in self.target_units:
                user = (
                    _DE_USER_PROMPT
                    .replace("[[DONOR1]]", donors[0]["snapshot"].unit_values[unit.uid])
                    .replace("[[DONOR2]]", donors[1]["snapshot"].unit_values[unit.uid])
                    .replace("[[BEST]]", best_entry["snapshot"].unit_values[unit.uid])
                    .replace("[[CURRENT]]", target["snapshot"].unit_values[unit.uid])
                )
                unit_messages[unit.uid] = [
                    {"role": "system", "content": _EVOLUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ]
            plans.append({
                # Sourcing the child from its target member lets `_select_survivors`
                # pair each record back to that member via source_snapshot_id.
                "source_snapshot": target["snapshot"],
                "unit_messages": unit_messages,
                "metadata": {
                    "source": self.__class__.__name__,
                    "strategy": "de",
                    "generation": ost["generation"] + 1,
                    "target_snapshot_id": target["member"]["snapshot_id"],
                    "donors": [
                        donors[0]["member"]["snapshot_id"],
                        donors[1]["member"]["snapshot_id"],
                    ],
                    "best_snapshot_id": best_entry["member"]["snapshot_id"],
                },
            })
        return plans

    def _select_survivors(
        self,
        population: List[Dict[str, Any]],
        records: List[TrialRecord],
        objective: Objective,
    ) -> List[Dict[str, Any]]:
        index_by_snapshot_id = {
            member["snapshot_id"]: idx for idx, member in enumerate(population)
        }
        for record in records:
            target_idx = index_by_snapshot_id.get(record.source_snapshot_id)
            if target_idx is None:
                continue
            if objective.is_better(record.metrics, population[target_idx]["metrics"]):
                population[target_idx] = {
                    "snapshot_id": record.snapshot_id,
                    "metrics": record.metrics,
                }
        return population


__all__ = ["EvoPromptOptimizer", "EvoPromptGAOptimizer", "EvoPromptDEOptimizer"]
