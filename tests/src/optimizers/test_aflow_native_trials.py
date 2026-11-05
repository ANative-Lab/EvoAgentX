"""Native AFlowOptimizer/HotPotQA regressions; only provider I/O is scripted."""
import asyncio
import json
from pathlib import Path

import pytest

from evoagentx.benchmark.hotpotqa import AFlowHotPotQA
from evoagentx.core.registry import register_model
from evoagentx.models.base_model import BaseLLM
from evoagentx.models.model_configs import LLMConfig
from evoagentx.models.model_utils import Cost, cost_manager
from evoagentx.optimizers.aflow import ReliableTrialConfig
from evoagentx.optimizers.aflow_optimizer import AFlowOptimizer

PROVIDER_STATE = {}

GRAPH = '''class Workflow:
    def __init__(self, name, llm_config, benchmark):
        self.llm = create_llm_instance(llm_config)
    async def __call__(self, problem):
        response = await self.llm.async_generate(prompt=problem, parse_mode="str")
        return response.content
'''


@register_model(LLMConfig)
class TrialFixtureProvider(BaseLLM):
    def init_model(self):
        self.state = PROVIDER_STATE
    def formulate_messages(self, prompts, system_messages=None):
        return [[{'role': 'user', 'content': p}] for p in prompts]
    def single_generate(self, messages, **kwargs):
        raise AssertionError('only async provider boundary is supported')
    def batch_generate(self, batch_messages, **kwargs):
        raise AssertionError('only async provider boundary is supported')
    async def single_generate_async(self, messages, **kwargs):
        await asyncio.sleep(0)
        if 'You are building a Graph' in messages[-1]['content']:
            self.state['proposal'] += 1
            cost_manager.update_cost(Cost(cost=.01), 'proposal-fixture')
            if self.state['proposal'] <= self.state['outage']:
                raise TimeoutError('scripted provider interruption')
            self.state['mode'] = 'right'
            graph = GRAPH + '\n# candidate generation\n'
            return '<modification>Use exact answer</modification><graph>' + graph + '</graph><prompt># no prompts</prompt>'
        self.state['execution'] += 1
        cost_manager.update_cost(Cost(cost=.01), 'execution-fixture')
        if self.state['mode'] == 'executor_outage':
            raise TimeoutError('scripted executor outage')
        return 'Paris' if self.state['mode'] == 'right' else 'wrong'


def setup(tmp_path, monkeypatch, reliable=True, **kwargs):
    if reliable:
        assert 'reliable_trials' in AFlowOptimizer.model_fields, 'explicit reliable trial option missing'
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    seed = tmp_path / 'seed'
    seed.mkdir(exist_ok=True)
    (seed / 'graph.py').write_text('from evoagentx.models.model_utils import create_llm_instance\n' + GRAPH)
    (seed / 'prompt.py').write_text('# seed prompts\n')
    for split in ('validate', 'test'):
        (tmp_path / f'hotpotqa_{split}.jsonl').write_text(''.join(json.dumps({'_id': str(i), 'question': q, 'answer': 'Paris', 'context': []}) + '\n' for i, q in enumerate(['first', 'second'])))
    benchmark = AFlowHotPotQA(path=str(tmp_path), mode='all')
    PROVIDER_STATE.clear()
    PROVIDER_STATE.update(mode='wrong', execution=0, proposal=0, outage=0)
    llm = TrialFixtureProvider(LLMConfig(llm_type='TrialFixtureProvider', model='local-fixture'))
    config = ReliableTrialConfig(replicates=2, max_proposals=4, duplicate_limit=2, **kwargs) if reliable else None
    optimizer = AFlowOptimizer(question_type='qa', graph_path='seed', optimized_path='results',
                               optimizer_llm=llm, validation_rounds=2, eval_rounds=1, max_rounds=2,
                               check_convergence=False, reliable_trials=config)
    return optimizer, benchmark, llm


def test_native_proposal_retry_never_repeats_completed_baseline_and_summary_is_shared(tmp_path, monkeypatch):
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    llm.state['outage'] = 1
    result = optimizer.optimize(benchmark)
    assert result.status == 'exhausted'
    assert result.score == 1 and result.best_round == 1
    assert llm.state['execution'] == 8  # 4 baseline, 4 candidate; duplicate candidates not evaluated
    assert llm.state['proposal'] == 4
    rows = json.loads(Path('results/results.json').read_text())
    assert [(r['round'], r['score']) for r in rows] == [(0, 0), (1, 1)]
    experience = json.loads(Path('results/round_1/experience.json').read_text())
    assert experience['before'] == 0 and experience['after'] == 1 and experience['succeed']
    assert optimizer._load_best_round() == 1
    summaries = list(Path('results/reliable/aflow/trials').glob('*/summary.json'))
    assert len(summaries) == 2
    assert all(json.loads(path.read_text())['requested'] == 4 for path in summaries)


def test_native_resume_reuses_all_completed_evidence_and_test_is_isolated(tmp_path, monkeypatch):
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    first = optimizer.optimize(benchmark)
    executions = llm.state['execution']
    before = Path('results/results.json').read_bytes()
    llm.state['mode'] = 'wrong'
    optimizer.test(benchmark, [1])
    assert Path('results/results.json').read_bytes() == before
    assert json.loads(Path('results/test_results.json').read_text())[0]['score'] == 0
    second = optimizer.optimize(benchmark)
    assert second.score == first.score and second.best_trial_id == first.best_trial_id
    assert llm.state['execution'] == executions + 4  # separate reliable test plan has 2 replicates
    assert optimizer._load_best_round() == 1


def test_native_plan_rejects_model_change_without_external_call(tmp_path, monkeypatch):
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    optimizer.optimize(benchmark)
    count = llm.state['execution']
    llm.config.temperature = .9
    with pytest.raises(ValueError, match='identity'):
        optimizer.optimize(benchmark)
    assert llm.state['execution'] == count


def test_legacy_test_results_do_not_change_validation_ranking(tmp_path, monkeypatch):
    optimizer, benchmark, _llm = setup(tmp_path, monkeypatch, reliable=False)
    optimizer.data_utils.save_results('results/results.json', [optimizer.data_utils.create_result_data(0, 1, 0, 0)])
    optimizer.test(benchmark, [0])
    assert optimizer.data_utils._load_scores() == [{'round': 0, 'score': 1}]
    assert Path('results/test_results.json').exists()


def test_native_exports_repair_after_failure_without_repeating_evidence(tmp_path, monkeypatch):
    from evoagentx.optimizers.aflow import evidence
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    original = evidence.atomic_json
    def fail_results(path, value):
        if str(path).endswith('/results.json'):
            raise OSError('scripted export failure')
        return original(path, value)
    monkeypatch.setattr(evidence, 'atomic_json', fail_results)
    first = optimizer.optimize(benchmark)
    assert first.score == 1 and first.export_error == 'builtins.OSError'
    calls = llm.state['execution']
    monkeypatch.setattr(evidence, 'atomic_json', original)
    repaired = optimizer.optimize(benchmark)
    assert repaired.score == 1 and repaired.export_error is None
    assert llm.state['execution'] == calls
    report = json.loads(Path('results/reliable/aflow/reports/candidate_1.json').read_text())
    assert [row['score'] for row in report['examples']] == [1, 1, 1, 1]
    assert report['ranking_eligible']


def test_legacy_tuple_cost_ignores_sibling_context(tmp_path, monkeypatch):
    from evoagentx.evaluators.aflow_evaluator import AFlowEvaluator
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch, reliable=False)
    graph = optimizer.graph_utils.load_graph(0, 'results')
    async def run():
        async def sibling():
            await asyncio.sleep(0)
            cost_manager.update_cost(Cost(cost=10), 'sibling')
        result, _ = await asyncio.gather(AFlowEvaluator(llm).graph_evaluate_async(benchmark, graph), sibling())
        return result
    score, avg_cost, total_cost, all_failed = asyncio.run(run())
    assert score == 0 and not all_failed
    assert avg_cost == pytest.approx(.01) and total_cost == pytest.approx(.02)


def test_native_provider_outage_is_incomplete_evidence_without_failed_strategy(tmp_path, monkeypatch):
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    llm.state['mode'] = 'executor_outage'
    result = optimizer.optimize(benchmark)
    assert result.status == 'incomplete' and result.score is None
    assert llm.state['proposal'] == 0 and llm.state['execution'] == 8
    assert not Path('results/round_1/experience.json').exists()
    report = json.loads(Path('results/reliable/aflow/reports/baseline.json').read_text())
    assert not report['ranking_eligible']
    assert all(row['status'] == 'retryable' and row['score'] is None for row in report['examples'])


@pytest.mark.parametrize('kind', ['humaneval', 'mbpp', 'math'])
def test_native_task_failure_is_scored_zero_across_aflow_benchmarks(tmp_path, kind):
    from evoagentx.benchmark.humaneval import AFlowHumanEval
    from evoagentx.benchmark.math_benchmark import AFlowMATH
    from evoagentx.benchmark.mbpp import AFlowMBPP
    from evoagentx.optimizers.aflow import AdmissionBudget, EvaluationPlan, TrialStore
    from evoagentx.optimizers.aflow.evaluator import DetailedEvaluator
    if kind == 'math':
        data = [{'id': 'local', 'problem': 'What is 1+1?', 'solution': r'\boxed{2}'}]
        benchmark_class = AFlowMATH
        prediction = r'\boxed{3}'
    else:
        example = {'task_id': 'local', 'prompt': 'def add(a, b):\n',
                   'canonical_solution': '    return a + b\n', 'code': 'def add(a, b):\n    return a+b\n',
                   'test': 'def check(candidate):\n    assert candidate(1, 2) == 3\n', 'entry_point': 'add'}
        data = [example]
        benchmark_class = AFlowHumanEval if kind == 'humaneval' else AFlowMBPP
        prediction = '    return 0\n' if kind == 'humaneval' else 'def add(a, b):\n    return 0\n'
        (tmp_path / f'{kind}_public_test.jsonl').write_text(json.dumps({'entry_point': 'add', 'test': example['test']}) + '\n')
    (tmp_path / f'{kind}_validate.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in data))
    benchmark = benchmark_class(path=str(tmp_path), mode='dev')
    class Graph:
        def __init__(self, **kwargs):
            pass
        async def __call__(self, *args):
            return prediction
    config = ReliableTrialConfig(concurrency=1)
    plan = EvaluationPlan.create(benchmark.name, data, {}, f'local.{kind}', config)
    with TrialStore(tmp_path / 'evidence', plan) as store:
        store.create_trial('baseline', 0, 'baseline')
        summary = asyncio.run(DetailedEvaluator(None).evaluate(benchmark, Graph, data, plan, 'baseline', store, AdmissionBudget(store, config)))
        assert summary.eligible and summary.score == 0
        assert store.observations('baseline')[0].status == 'scored'


def test_proposal_cost_evidence_survives_cleared_stage_cursor(tmp_path, monkeypatch):
    optimizer, benchmark, llm = setup(tmp_path, monkeypatch)
    llm.state['outage'] = 1
    optimizer.optimize(benchmark)
    report = json.loads(Path('results/reliable/aflow/reports/experiment.json').read_text())
    operations = report['operation_accounting']
    proposals = [row for row in operations if row['stage'] == 'proposal']
    assert len(proposals) == 4
    assert sum(row['known_cost'] for row in proposals) == pytest.approx(.04)
    assert sum(row['known_cost'] for row in operations) == pytest.approx(.12)
    assert all(row['trial_id'] and row['pricing_known'] for row in proposals)
