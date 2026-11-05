"""Reliable evidence boundaries, using local disk and real cost contexts."""
import asyncio
import json
from dataclasses import replace

import pytest


def api():
    from evoagentx.optimizers.aflow import (
        EvaluationPlan, Observation, ReliableTrialConfig, TrialStore,
        TrialSummary, AdmissionBudget,
    )
    return EvaluationPlan, Observation, ReliableTrialConfig, TrialStore, TrialSummary, AdmissionBudget


def plan(data=None, **kwargs):
    Plan, _, Config, *_ = api()
    return Plan.create('Fixture', data or [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}],
                       {'model': 'local', 'api_key': 'TOP_SECRET', 'temperature': 0},
                       'fixture.metric.v1', Config(**kwargs))


def test_plan_preserves_duplicate_occurrences_order_and_omits_secrets():
    p = plan()
    assert [e.original_id for e in p.examples] == ['same', 'same']
    assert [e.occurrence for e in p.examples] == [0, 1]
    assert 'TOP_SECRET' not in json.dumps(p.to_dict())
    assert p.identity != plan([{'_id': 'same', 'answer': 'b'}, {'_id': 'same', 'answer': 'a'}]).identity
    assert p.identity != plan(replicates=2).identity


def test_summary_requires_exact_finite_coverage():
    _, Obs, _, _, Summary, _ = api()
    p = plan(replicates=2)
    observations = [Obs.scored('trial', p.examples[0], 0, 1, 0.0)]
    partial = Summary.build(p, 'trial', 0, observations)
    assert not partial.eligible and partial.score is None
    assert partial.scored == 1 and partial.requested == 4 and partial.partial_score == 0
    for replicate in range(2):
        for example in p.examples:
            if replicate or example.index:
                observations.append(Obs.scored('trial', example, replicate, 1, 1.0))
    summary = Summary.build(p, 'trial', 0, observations)
    assert summary.eligible and summary.score == .75
    with pytest.raises(ValueError):
        Obs.scored('trial', p.examples[0], 0, 1, float('nan'))


def test_store_atomic_identity_observation_idempotence_and_owner_lock(tmp_path):
    _, Obs, _, Store, Summary, _ = api()
    p = plan()
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        obs = Obs.scored('trial', p.examples[0], 0, 1, 0)
        store.record(obs)
        store.record(obs)
        assert len(store.observations('trial')) == 1
        with pytest.raises(ValueError, match='conflict'):
            store.record(replace(obs, score=1))
        with pytest.raises(RuntimeError, match='writer'):
            with Store(tmp_path, p):
                pass
        store.record(Obs.scored('trial', p.examples[1], 0, 1, 1))
        summary = Summary.build(p, 'trial', 0, store.observations('trial'))
        store.commit_summary(summary)
        assert store.summaries()[0].score == .5
    with pytest.raises(ValueError, match='identity'):
        with Store(tmp_path, plan(replicates=2)):
            pass
    with Store(tmp_path, p) as store:
        assert store.summaries()[0].score == .5


def test_started_attempt_requires_explicit_resolution_before_rerun(tmp_path):
    *_, Store, _, Budget = api()
    p = plan()
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        store.start_attempt('trial', p.examples[0], 0, 1)
    with Store(tmp_path, p) as store:
        assert len(store.unknown_attempts()) == 1
        with pytest.raises(RuntimeError, match='unknown'):
            store.assert_resolved()
        attempt = store.unknown_attempts()[0]
        store.resolve_unknown(attempt['attempt_id'], 'rerun', 'operator confirmed original call has terminated')
        store.assert_resolved()
        assert store.audit()[-1]['action'] == 'resolve_unknown'


def test_artifact_generation_is_immutable_and_digest_verified(tmp_path):
    *_, Store, _, _ = api()
    with Store(tmp_path, plan()) as store:
        store.create_trial('trial', 0, 'baseline')
        artifact = store.stage_artifacts('trial', 'class Workflow: pass\n', 'PROMPT = "x"\n')
        assert store.read_artifacts('trial')[0] == 'class Workflow: pass\n'
        with pytest.raises(ValueError, match='immutable'):
            store.stage_artifacts('trial', 'different', '')
        (store.path / artifact['directory'] / 'graph.py').write_text('tampered')
        with pytest.raises(ValueError, match='digest'):
            store.read_artifacts('trial')


def test_admissions_reserve_known_cost_and_do_not_reset_on_retry(tmp_path):
    *_, Store, _, Budget = api()
    p = plan(max_example_attempts=2, max_known_cost=.5, example_cost_bound=.3, proposal_cost_bound=.1)
    with Store(tmp_path, p) as store:
        budget = Budget(store, p.config)
        first = budget.admit('example', 'a')
        assert first['reserved'] == .3
        with pytest.raises(RuntimeError, match='cost'):
            budget.admit('example', 'b')
        budget.finish('a', .1, True)
        budget.admit('example', 'b')
        budget.finish('b', .1, True)
        with pytest.raises(RuntimeError, match='attempt'):
            budget.admit('example', 'c')


def test_context_cost_observation_isolates_siblings_and_propagates_to_thread():
    from evoagentx.models.model_utils import Cost, CostManager
    manager = CostManager()
    seen = []
    async def owned():
        with manager.observe(seen.append):
            manager.update_cost(Cost(cost=.1), 'owned')
            await asyncio.to_thread(manager.update_cost, Cost(cost=.2), 'thread')
            manager.update_cost(Cost(), 'unknown')
    async def unrelated():
        await asyncio.sleep(0)
        manager.update_cost(Cost(cost=10), 'sibling')
    async def run():
        await asyncio.gather(owned(), unrelated())
    asyncio.run(run())
    assert [event.model for event in seen] == ['owned', 'thread', 'unknown']
    assert sum(e.cost or 0 for e in seen) == pytest.approx(.3)
    assert seen[-1].pricing == 'unknown'
    assert manager.get_total_cost() == pytest.approx(10.3)


def test_finished_reservations_do_not_grow_admission_checkpoint(tmp_path):
    *_, Store, _, Budget = api()
    with Store(tmp_path, plan(max_example_attempts=150)) as store:
        budget = Budget(store, store.plan.config)
        for i in range(100):
            budget.admit('example', str(i))
            budget.finish(str(i), 0, True)
        assert (tmp_path / 'budget.json').stat().st_size < 1000
        with pytest.raises(ValueError, match='already'):
            budget.admit('example', '0')


def test_bound_violation_halts_later_admission(tmp_path):
    *_, Store, _, Budget = api()
    with Store(tmp_path, plan(max_known_cost=1, example_cost_bound=.1, proposal_cost_bound=.1)) as store:
        budget = Budget(store, store.plan.config)
        budget.admit('example', 'a')
        with pytest.raises(RuntimeError, match='violated'):
            budget.finish('a', .2, True)
        with pytest.raises(RuntimeError, match='violated'):
            budget.admit('example', 'b')


def test_detailed_evaluator_preserves_zero_retries_and_exact_partial_resume(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    _, Obs, _, Store, _, Budget = api()
    p = plan(replicates=2, concurrency=2)
    data = [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}]
    calls = []
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            calls.append(example['answer'])
            if example['answer'] == 'b' and calls.count('b') == 1:
                raise TimeoutError('local provider boundary')
            return 0 if example['answer'] == 'a' else 1
    class Graph:
        def __init__(self, name, llm_config, benchmark):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        store.record(Obs.scored('trial', p.examples[0], 0, 1, 0))
        summary = asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph, data, p, 'trial', store, Budget(store, p.config)))
        assert summary.eligible and summary.score == .5
        assert calls.count('a') == 1 and calls.count('b') == 3
        assert len(store.observations('trial')) == 5
        assert store.summary('trial') == summary


def test_unknown_metric_error_is_inconclusive_and_not_retried(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan()
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            return float('inf') if example['answer'] == 'a' else 0
    class Graph:
        def __init__(self, **kwargs):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        summary = asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
            [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
        assert not summary.eligible and summary.score is None and summary.scored == 1
        assert summary.partial_score == 0
        assert summary.outcomes == {'inconclusive': 1, 'scored': 1}
        assert len(store.observations('trial')) == 2


def test_deadline_cancels_owned_work_and_never_promotes_late_completion(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan(attempt_timeout=.01, concurrency=1)
    drained = []
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            try:
                await asyncio.sleep(10)
            finally:
                drained.append(example['answer'])
    class Graph:
        def __init__(self, **kwargs):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        with pytest.raises(RuntimeError, match='unknown'):
            asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
                [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
        assert drained == ['a']
        assert store.summary('trial') is None
        assert store.observations('trial')[0].status == 'unknown'
        assert len(store.unknown_attempts()) == 1
    with pytest.raises(RuntimeError, match='recover_owner'):
        with Store(tmp_path, p):
            pass


def test_outer_cancellation_joins_workers_and_records_unknown(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan(concurrency=1)
    entered = asyncio.Event()
    drained = []
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(.01)
                drained.append(1)
    class Graph:
        def __init__(self, **kwargs):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        async def run():
            task = asyncio.create_task(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
                [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        assert drained == [1]
        assert store.summary('trial') is None
        assert store.unknown_attempts()
