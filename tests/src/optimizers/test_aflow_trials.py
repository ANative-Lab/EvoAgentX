"""Reliable evidence boundaries, using local disk and real cost contexts."""
import asyncio
import json
from dataclasses import replace

import pytest


def api():
    from evoagentx.optimizers.aflow import (
        AdmissionBudget,
        EvaluationPlan,
        Observation,
        ReliableTrialConfig,
        TrialStore,
        TrialSummary,
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
        with pytest.raises(RuntimeError, match='writer'), Store(tmp_path, p):
            pass
        store.record(Obs.scored('trial', p.examples[1], 0, 1, 1))
        summary = Summary.build(p, 'trial', 0, store.observations('trial'))
        store.commit_summary(summary)
        assert store.summaries()[0].score == .5
    with pytest.raises(ValueError, match='identity'), Store(tmp_path, plan(replicates=2)):
        pass
    with Store(tmp_path, p) as store:
        assert store.summaries()[0].score == .5


def test_started_attempt_requires_explicit_resolution_before_rerun(tmp_path):
    *_, Store, _, _Budget = api()
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
    with pytest.raises(RuntimeError, match='recover_owner'), Store(tmp_path, p):
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


def test_existing_store_can_be_opened_for_audited_offline_resolution(tmp_path):
    *_, Store, _, _ = api()
    p = plan()
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        store.start_attempt('trial', p.examples[0], 0, 1)
    with Store.open_existing(tmp_path) as store:
        assert store.plan.identity == p.identity
        unknown = store.unknown_attempts()[0]
        store.resolve_unknown(unknown['attempt_id'], 'abandon', 'confirmed terminated; do not repeat')
        store.assert_resolved()


def test_versioned_plan_reload_rejects_modified_policy_and_schema(tmp_path):
    Plan, *_ = api()
    payload = plan().to_dict()
    assert Plan.from_dict(payload).identity == plan().identity
    payload['config']['replicates'] = 5
    with pytest.raises(ValueError, match='identity'):
        Plan.from_dict(payload)
    payload['schema_version'] = 77
    with pytest.raises(ValueError, match='schema'):
        Plan.from_dict(payload)


def test_per_example_report_includes_missing_coordinates_without_inventing_scores(tmp_path):
    from evoagentx.optimizers.aflow.reporting import TrialReporter
    _, Obs, _, Store, Summary, _ = api()
    p = plan(replicates=2)
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        store.record(Obs.scored('trial', p.examples[0], 0, 1, 0))
        summary = Summary.build(p, 'trial', 0, store.observations('trial'))
        store.commit_summary(summary)
        report = TrialReporter(store).trial('trial')
        assert report['summary']['score'] is None
        assert [row['status'] for row in report['examples']] == ['scored', 'missing', 'missing', 'missing']
        assert report['examples'][0]['score'] == 0
        assert report['examples'][1]['score'] is None
        assert len(report['examples']) == 4
        TrialReporter(store).export()
        assert json.loads((tmp_path / 'reports' / 'trial.json').read_text()) == report


def test_model_endpoint_identity_keeps_port_and_drops_credentials():
    Plan, _, Config, *_ = api()
    def make(endpoint):
        return Plan.create('fixture', [{'_id': 1}], {'base_url': endpoint}, 'm', Config())
    left = make('https://user:password@localhost:8000/v1?api_key=secret')
    right = make('https://localhost:9000/v1')
    assert left.identity != right.identity
    assert left.model_config['base_url'] == 'https://localhost:8000/v1'


def test_summary_publication_failure_resumes_without_repeating_examples(tmp_path, monkeypatch):
    import evoagentx.optimizers.aflow.store as storage
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan()
    calls = []
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            calls.append(1)
            return 1
    class Graph:
        def __init__(self, **kwargs):
            pass
    original = storage.atomic_json
    def fail_summary(path, value):
        if str(path).endswith('/summary.json'):
            raise OSError('scripted summary storage failure')
        return original(path, value)
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        monkeypatch.setattr(storage, 'atomic_json', fail_summary)
        with pytest.raises(OSError):
            asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
                [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
        monkeypatch.setattr(storage, 'atomic_json', original)
    with Store(tmp_path, p) as store:
        summary = asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
            [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
        assert summary.eligible and summary.score == 1 and len(calls) == 2


def test_late_thread_cost_retains_unknown_and_never_changes_committed_outcome(tmp_path):
    import threading

    from evoagentx.models.model_utils import Cost, cost_manager
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan(attempt_timeout=.01, concurrency=1)
    release = threading.Event()
    finished = threading.Event()
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            def blocking():
                release.wait(2)
                cost_manager.update_cost(Cost(cost=.7), 'late-thread')
                finished.set()
                return 1
            return await asyncio.to_thread(blocking)
    class Graph:
        def __init__(self, **kwargs):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        before = cost_manager.get_total_cost()
        async def run():
            try:
                with pytest.raises(RuntimeError, match='unknown'):
                    await DetailedEvaluator(None).evaluate(Benchmark(), Graph,
                        [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config))
                frozen = store.observations('trial')[0]
                release.set()
                await asyncio.to_thread(finished.wait, 2)
                assert store.observations('trial')[0] == frozen
                assert frozen.status == 'unknown' and frozen.score is None
                assert frozen.known_cost == 0
                assert store.summary('trial') is None
            finally:
                release.set()
        asyncio.run(run())
        assert finished.is_set()
        assert cost_manager.get_total_cost() - before == pytest.approx(.7)


def test_finite_extreme_scores_keep_finite_aggregate_and_summary_reload_checks_diagnostics():
    _, Obs, _, _, Summary, _ = api()
    p = plan()
    summary = Summary.build(p, 'trial', 0, [Obs.scored('trial', e, 0, 1, 1e308) for e in p.examples])
    assert summary.score == 1e308
    payload = summary.to_dict()
    payload['partial_score'] = float('nan')
    with pytest.raises(ValueError):
        Summary.from_dict(payload)


def test_completed_observation_recovers_pending_cost_reservation_without_reexecution(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    _, Obs, _, Store, Summary, Budget = api()
    p = plan(max_known_cost=1, example_cost_bound=.2, proposal_cost_bound=.1)
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        budget = Budget(store, p.config)
        for e in p.examples:
            obs = Obs.scored('trial', e, 0, 1, 1, known_cost=.1, pricing='known')
            budget.admit('example', obs.observation_id)
            store.start_attempt('trial', e, 0, 1)
            store.record(obs)  # crash before budget.finish
        store.commit_summary(Summary.build(p, 'trial', 0, store.observations('trial')))
    class NoCalls:
        name = 'Fixture'
        async def async_evaluate(self, *args):
            raise AssertionError('completed external call must not repeat')
    with Store(tmp_path, p) as store:
        budget = Budget(store, p.config)
        summary = asyncio.run(DetailedEvaluator(None).evaluate(NoCalls(), None,
            [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, budget))
        assert summary.score == 1
        assert budget.report()['known_spent'] == pytest.approx(.2)
        assert budget.report()['reserved'] == 0


def test_total_admission_budget_and_concurrency_are_bounded(tmp_path):
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    Plan, _, Config, Store, _, Budget = api()
    data = [{'_id': i} for i in range(10)]
    p = Plan.create('fixture', data, {}, 'v1', Config(concurrency=2, max_example_attempts=3))
    active = 0
    peak = 0
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.001)
            active -= 1
            return 1
    class Graph:
        def __init__(self, **kwargs):
            pass
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        budget = Budget(store, p.config)
        summary = asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph, data, p, 'trial', store, budget))
        assert summary.score is None and summary.scored == 3 and summary.requested == 10
        assert budget.report()['counts']['example'] == 3 and peak == 2


def test_interrupted_outcome_write_leaves_unknown_instead_of_paid_retry(tmp_path, monkeypatch):
    from evoagentx.optimizers.aflow import store as storage
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    *_, Store, _, Budget = api()
    p = plan(concurrency=1)
    class Benchmark:
        name = 'Fixture'
        async def async_evaluate(self, graph, example):
            return 1
    class Graph:
        def __init__(self, **kwargs):
            pass
    original = storage.atomic_json
    def fail_outcome(path, value):
        if '/observations/' in str(path):
            raise OSError('scripted outcome publication failure')
        return original(path, value)
    with Store(tmp_path, p) as store:
        store.create_trial('trial', 0, 'baseline')
        monkeypatch.setattr(storage, 'atomic_json', fail_outcome)
        with pytest.raises(OSError):
            asyncio.run(DetailedEvaluator(None).evaluate(Benchmark(), Graph,
                [{'_id': 'same', 'answer': 'a'}, {'_id': 'same', 'answer': 'b'}], p, 'trial', store, Budget(store, p.config)))
        monkeypatch.setattr(storage, 'atomic_json', original)
    with Store(tmp_path, p) as store:
        assert len(store.unknown_attempts()) == 1
        assert store.observations('trial') == []
        with pytest.raises(RuntimeError, match='unknown'):
            store.assert_resolved()


def test_artifact_publish_interruption_never_mixes_graph_prompt_generations(tmp_path, monkeypatch):
    from evoagentx.optimizers.aflow import store as storage
    *_, Store, _, _ = api()
    with Store(tmp_path, plan()) as store:
        store.create_trial('trial', 0, 'baseline')
        original = storage.os.rename
        def fail_rename(source, destination):
            raise OSError('scripted generation publication failure')
        monkeypatch.setattr(storage.os, 'rename', fail_rename)
        with pytest.raises(OSError):
            store.stage_artifacts('trial', 'graph', 'prompt')
        assert not (tmp_path / 'trials' / 'trial' / 'artifacts').exists()
        monkeypatch.setattr(storage.os, 'rename', original)
        store.stage_artifacts('trial', 'new graph', 'new prompt')
        assert store.read_artifacts('trial')[:2] == ('new graph', 'new prompt')


@pytest.mark.parametrize('provider', ['openrouter', 'novita', 'siliconflow', 'aliyun', 'litellm'])
def test_existing_unpriced_provider_fallbacks_emit_unknown_pricing(provider):
    from types import SimpleNamespace

    from openai.types import CompletionUsage

    from evoagentx.models.model_utils import cost_manager
    usage = CompletionUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
    seen = []
    with cost_manager.observe(seen.append):
        if provider == 'aliyun':
            from evoagentx.models.aliyun_model import AliyunLLM
            cost = AliyunLLM._compute_cost(SimpleNamespace(_has_pricing=False), usage)
            cost_manager.update_cost(cost, 'unpriced')
        elif provider == 'litellm':
            from evoagentx.models.litellm_model import LiteLLM
            from evoagentx.models.model_configs import LiteLLMConfig
            llm = object.__new__(LiteLLM)
            llm.config = LiteLLMConfig(model='unpriced-fixture-no-such-model', is_local=False)
            cost_manager.update_cost(llm._compute_cost(usage), 'unpriced')
        else:
            from evoagentx.models.novita_model import NovitaLLM
            from evoagentx.models.openrouter_model import OpenRouterLLM
            from evoagentx.models.siliconflow_model import SiliconFlowLLM
            cls = {'openrouter': OpenRouterLLM, 'novita': NovitaLLM, 'siliconflow': SiliconFlowLLM}[provider]
            cls._update_cost(SimpleNamespace(config=SimpleNamespace(model='unpriced')),
                             SimpleNamespace(id='local', usage=usage))
    assert len(seen) == 1 and seen[0].pricing == 'unknown' and seen[0].cost is None
    assert seen[0].input_tokens == 2 and seen[0].output_tokens == 1


def test_model_identity_retains_tokenizer_behavior_and_sanitizes_nested_headers_endpoints():
    Plan, _, Config, *_ = api()
    model = {'model': 'local', 'tokenizer': 'first', 'max_tokens': 20,
             'azure_endpoint': 'https://user:secret@host:8000/v1?api_key=hidden',
             'default_headers': {'X-Api-Key': 'hidden-key', 'Authorization': 'Bearer hidden'}}
    left = Plan.create('fixture', [{'_id': 1}], model, 'v1', Config())
    model['tokenizer'] = 'second'
    right = Plan.create('fixture', [{'_id': 1}], model, 'v1', Config())
    assert left.identity != right.identity
    assert left.model_config['azure_endpoint'] == 'https://host:8000/v1'
    assert left.model_config['default_headers'] == {}
    assert 'hidden' not in json.dumps(left.to_dict())


def test_current_live_stopping_writer_cannot_be_overridden_and_dead_recovery_is_audited(tmp_path):
    import os
    from dataclasses import replace

    from evoagentx.optimizers.aflow.store import atomic_json
    *_, Store, _, _ = api()
    p = plan()
    with Store(tmp_path, p):
        pass
    owner = json.loads((tmp_path / 'owner.json').read_text())
    owner.update(status='stopping', pid=os.getpid())
    atomic_json(tmp_path / 'owner.json', owner)
    recovery = replace(p, config=replace(p.config, recover_owner=True))
    with pytest.raises(RuntimeError, match='still be running'), Store(tmp_path, recovery):
        pass
    owner['pid'] = 99999999
    atomic_json(tmp_path / 'owner.json', owner)
    with Store(tmp_path, recovery) as store:
        assert store.audit()[-1]['action'] == 'recover_owner'
