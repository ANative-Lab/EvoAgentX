# Reliable AFlow trials (opt in)

`AFlowOptimizer` can checkpoint an evolution experiment as complete, comparable
validation evidence. This is a new local trial protocol, not a claim that a scalar
score proves better reasoning or statistical superiority. Existing optimizers and
legacy directories retain their existing mode unless configured explicitly.

```python
from evoagentx.optimizers.aflow import ReliableTrialConfig
from evoagentx.optimizers.aflow_optimizer import AFlowOptimizer

optimizer = AFlowOptimizer(
    question_type="qa",
    graph_path="seed_workflow",
    optimized_path="experiment_workflows",
    optimizer_llm=proposal_llm,
    executor_llm=execution_llm,
    max_rounds=10,
    reliable_trials=ReliableTrialConfig(
        experiment_id="hotpotqa_run_1",
        replicates=3,
        concurrency=8,
        attempts_per_example=2,
        max_example_attempts=20000,
        max_proposals=30,
        max_trials=11,  # includes the baseline
        duplicate_limit=5,
        direction="maximize",
        improvement_margin=0.0,
        deadline_seconds=3600,
    ),
)
result = optimizer.optimize(benchmark)
# In an existing event loop: result = await optimizer.optimize_async(benchmark)
print(result.status, result.reason, result.best_round, result.score)
```

In reliable mode `replicates` fixes validation and test replicate coverage;
`validation_rounds`/`eval_rounds` remain legacy options. `initial_round` must be 0:
resume uses the experiment's saved identity/stage instead of a round number.
Calling optimize again repairs exports and resumes incomplete stages without
repeating completed scored coordinates. A finished experiment returns its saved
terminal result; start a new experiment ID for a new plan or additional rounds.

Every example occurrence retains its original ID and ordered occurrence index.
The plan fingerprints exact example contents/order, split, metric implementation,
public model configuration, seed sources and AFlow policy. Changed data, model,
metric or policy rejects resume. Authentication fields and endpoint credentials
are omitted. Custom benchmarks without inspectable source must supply a stable
`metric_id`, which the caller must change whenever scoring behavior changes.

Only a finite numeric result counts as scored evidence. A valid incorrect answer
is scored zero. Explicit configured timeout/connection exception types may be
retried after their asyncio operation finishes; unknown exceptions and invalid
metrics are inconclusive. A summary is eligible only when every requested
replicate/example is scored. Diagnostic partial averages never rank or advance
convergence. Ties and improvements smaller than the configured margin retain the
parent. Proposal-only, failed and test trials do not advance convergence.

## Checkpoints and reports

The authoritative directory is
`experiment_workflows/reliable/hotpotqa_run_1/`. It contains `plan.json`, stable
trial identities, staged graph/prompt generations, immutable observation files,
operation intents, budget state and one committed `summary.json` per finished
trial. Graph source remains trusted code under AFlow's existing execution model.
Artifact hashes establish exact contents, not semantic equivalence or a sandbox.

`reports/<trial_id>.json` includes every requested coordinate, the final status,
score or explicit missing result, typed infrastructure error, attempt history,
known cost, pricing uncertainty and original example ID. `reports/experiment.json`
includes parent selection, accepted/rejected/inconclusive feedback, pending calls,
controlled admission counts and recovery guidance. Raw provider error messages and
raw example contents are not copied into these reports.

Legacy-shaped `results.json`, round artifacts and `experience.json` are repairable
exports of the same committed score. An export failure is reported in
`result.export_error` and `export_status.json`; it does not undo a committed
summary. Test evaluation uses a separate test plan/store and `test_results.json`.
Legacy result readers keep their old fields; reliable provenance is additive.
No old result file is accepted as reliable coverage evidence: imported graphs
must receive a fresh explicit baseline evaluation.

## Unknown calls and recovery

A crash after a provider call but before its outcome checkpoint is ambiguous.
There is no exactly-once provider guarantee. Reliable mode never silently repeats
that call. Cancellation/timeout of an awaiter cannot prove that a remote call or
`asyncio.to_thread` worker has stopped. Owned asyncio tasks are joined, late cost
events still update global accounting, and the experiment remains stopping/unknown
without promotion. Stop the previous process and confirm any remote/thread work
has terminated before recovery. A live previous owner is never overridden.

```python
from evoagentx.optimizers.aflow import TrialStore
from evoagentx.optimizers.aflow.reporting import TrialReporter

with TrialStore.open_existing(path, recover_owner=True) as store:
    print(TrialReporter(store).recovery())
    for attempt in store.unknown_attempts():
        # Operator assertion: confirm termination externally before this call.
        store.resolve_unknown(
            attempt["attempt_id"], "abandon",  # or explicit "rerun"
            "Confirmed previous local and remote operation terminated",
        )
```

Explicit reruns receive new attempt IDs and consume remaining budgets. Abandoning
an unknown example leaves incomplete evidence. Known cost reservations remain
conservative where usage/pricing is unknown. Recovery decisions are audited.
Local `fcntl` locking enforces one owner, and parallel workers submit outcomes
through that owner. Unsupported lock platforms reject reliable mode. Multi-host
and NFS mutation are unsupported.

## Supported limits and accounting

Admission counters bound controlled trial/proposal/example attempts, and the
worker count bounds controlled example concurrency. Internal provider retries
are not separate trial attempts and are reported as unobserved. Token/request
counts inside arbitrary user graphs have no hard-cap claim. The elapsed deadline
stops new admission and requests cooperative cancellation; it does not guarantee
thread termination or shutdown within a wall-clock duration.

`max_known_cost` requires both `example_cost_bound` and `proposal_cost_bound`.
Supplying these bounds is an explicit caller/provider contract that they cover
the entire admitted operation, including nested calls/retries. Without a supported
whole-operation pre-call upper bound, leave the cost limit unset and inspect
observed/unknown usage. Unknown pricing is never assumed free. Reservations are
persisted before work, retries do not reset them, and an observed bound violation
halts further admission. Trusted graph imports/constructor side effects and any
external unobserved accounting paths cannot acquire hard provider limits merely
by setting a number in this configuration.

The optional `CostManager.observe` scope preserves existing process totals while
isolating sibling asyncio contexts. `asyncio.to_thread` propagates the observer;
raw threads, external processes and providers bypassing `CostManager.update_cost`
are unsupported accounting paths. Missing events/prices remain unobserved or
unknown rather than known zero.

## Diagnostic evidence in planning

Reliable graph-proposal prompts include a bounded `validation_diagnostics` JSON
payload. It matches committed parent/candidate validation outcomes by original ID,
occurrence and replicate under the exact same plan. Equal means can therefore
retain distinct observed improvements/regressions/ties. Metric direction controls
these labels; score deltas retain their numeric sign. Missing and infrastructure
outcomes have separate coverage categories. An incomplete trial contributes
execution diagnostics and cannot support task-regression conclusions. Recovered
retry failures are counted as execution diagnostics, without blaming a strategy.

`feedback_max_examples` defaults to 12 across the payload, and
`feedback_max_characters` defaults to 12000 including the JSON tags (minimum 512).
Selection uses recent candidate trials first, then fixed plan order. The payload
states omitted comparison/example counts; if detailed evidence cannot fit, it
explicitly reports unavailable detail and scalar-only summaries. Oversized IDs are
omitted as complete records rather than silently changing their identity.

The payload contains IDs, observed scores, typed status categories and scoped
recorded-attempt cost deltas. It includes no raw labels or exception tracebacks.
Unknown pricing keeps the complete cost delta unknown; an observed known subtotal
is identified separately. Overflowing differences receive an explicit marker and
never produce nonfinite JSON. These are diagnostic associations, not causal or
agent-step attribution, and do not change the committed promotion score.

The current bounded view is exported under `planner_feedback` in the experiment
report. Each proposal intent persists the exact payload sent to that provider;
`proposal_feedback` retains these historical snapshots even after later trials
change the current view. Test outcomes never enter either validation payload.
