"""Planner evidence compares observed validation coordinates, not scalar averages."""
import json
from dataclasses import replace

import pytest

from evoagentx.optimizers.aflow import (
    EvaluationPlan,
    Observation,
    ReliableTrialConfig,
    TrialStore,
    TrialSummary,
)
from evoagentx.optimizers.aflow.evidence import EvidenceView
from evoagentx.optimizers.aflow.reporting import TrialReporter
from evoagentx.optimizers.aflow.types import canonical


def fixture(tmp_path, **kwargs):
    config = ReliableTrialConfig(replicates=2, **kwargs)
    data = [{'_id': 'duplicate', 'value': 'first'}, {'_id': 'duplicate', 'value': 'second'}]
    plan = EvaluationPlan.create('local', data, {}, 'metric.v1', config)
    store = TrialStore(tmp_path, plan)
    return store, plan


def commit(store, plan, trial_id, scores, parent=None, pricing='known', cost=.1):
    number = len(store.trials())
    store.create_trial(trial_id, number, 'candidate' if parent else 'baseline', parent, 'local modification')
    for replicate in range(plan.config.replicates):
        for example in plan.examples:
            score = scores[replicate * len(plan.examples) + example.index]
            if score == 'missing':
                continue
            if score is None:
                observation = Observation.make(trial_id, example, replicate, 1, 'inconclusive', error_type='TimeoutError', pricing=pricing, known_cost=cost)
            else:
                observation = Observation.scored(trial_id, example, replicate, 1, score, pricing=pricing, known_cost=cost)
            store.record(observation)
    summary = TrialSummary.build(plan, trial_id, number, store.observations(trial_id))
    store.commit_summary(summary)
    return summary


def test_duplicate_ids_replicates_and_direction_keep_matched_tradeoffs(tmp_path):
    store, plan = fixture(tmp_path, direction='minimize')
    with store:
        commit(store, plan, 'parent', [0, 1, 0, 1])
        commit(store, plan, 'child', [1, 0, 1, 0], 'parent', cost=.2)
        payload = EvidenceView(store).planner_feedback('parent')
        comparison = payload['comparisons'][0]
        assert payload['metric_direction'] == 'minimize'
        assert comparison['coverage']['matched_scored'] == 4
        assert comparison['counts'] == {'improved': 2, 'regressed': 2, 'tied': 0, 'infrastructure': 0, 'missing': 0}
        assert comparison['score_delta'] == 0
        assert comparison['cost_delta']['value'] == pytest.approx(.4)
        examples = comparison['examples']
        assert [(row['example_id'], row['occurrence'], row['replicate']) for row in examples] == [
            ('duplicate', 0, 0), ('duplicate', 1, 0), ('duplicate', 0, 1), ('duplicate', 1, 1)]
        assert [row['change'] for row in examples] == ['regressed', 'improved', 'regressed', 'improved']
        assert all(row['score_delta'] in (-1, 1) for row in examples)
        prompt = EvidenceView(store).prompt_feedback('parent')
        assert canonical(payload) in prompt
        assert TrialReporter(store).experiment()['planner_feedback']['parent'] == payload


def test_incomplete_and_test_evidence_never_become_task_regressions(tmp_path):
    store, plan = fixture(tmp_path)
    with store:
        commit(store, plan, 'parent', [1, 1, 1, 1])
        commit(store, plan, 'incomplete', [0, None, 0, 'missing'], 'parent', pricing='unknown')
        payload = EvidenceView(store).planner_feedback('parent')
        row = payload['comparisons'][0]
        assert not row['task_conclusions']
        assert row['counts']['regressed'] == 0 and row['counts']['improved'] == 0
        assert row['counts']['infrastructure'] == 1 and row['counts']['missing'] == 1
        assert row['coverage']['matched_scored'] == 2
        assert row['cost_delta']['status'] == 'unknown' and row['cost_delta']['value'] is None
        assert all(example['change'] not in ('improved', 'regressed', 'tied') for example in row['examples'])
        prompt = EvidenceView(store).prompt_feedback('parent')
        assert 'TimeoutError' not in prompt  # categories, never arbitrary exception payloads
    test_plan = replace(plan, split='test')
    with TrialStore(tmp_path / 'test', test_plan) as test_store:
        commit(test_store, test_plan, 'parent', [1, 1, 1, 1])
        commit(test_store, test_plan, 'child', [0, 0, 0, 0], 'parent')
        payload = EvidenceView(test_store).planner_feedback('parent')
        assert payload['availability'] == 'unavailable'
        assert payload['comparisons'] == []


def test_feedback_is_deterministically_bounded_and_reports_omissions(tmp_path):
    store, plan = fixture(tmp_path, feedback_max_examples=2, feedback_max_characters=1800)
    with store:
        commit(store, plan, 'parent', [1, 0, 1, 0])
        for index in range(4):
            commit(store, plan, f'child_{index}', [0, 1, 0, 1], 'parent')
        view = EvidenceView(store)
        first = view.planner_feedback('parent')
        assert first == view.planner_feedback('parent')
        assert len(canonical(first)) <= 1800
        assert first['example_count'] <= 2
        assert first['omitted_examples'] >= 14
        assert first['omitted_comparisons'] > 0
        assert first['comparison_count'] == 4


def test_extreme_finite_deltas_are_explicit_overflow_and_json_stays_finite(tmp_path):
    store, plan = fixture(tmp_path)
    with store:
        commit(store, plan, 'parent', [-1e308, -1e308, -1e308, -1e308], cost=1e307)
        commit(store, plan, 'child', [1e308, 1e308, 1e308, 1e308], 'parent', cost=2e307)
        payload = EvidenceView(store).planner_feedback('parent')
        comparison = payload['comparisons'][0]
        assert comparison['score_delta'] is None and comparison['score_delta_status'] == 'overflow'
        assert comparison['counts']['improved'] == 4
        assert all(row['score_delta_status'] == 'overflow' for row in comparison['examples'])
        json.dumps(payload, allow_nan=False)


def test_minimum_character_budget_and_oversized_original_id_have_explicit_fallback(tmp_path):
    config = ReliableTrialConfig(feedback_max_characters=512)
    plan = EvaluationPlan.create('local', [{'_id': 'long' * 1000}], {}, 'metric.v1', config)
    with TrialStore(tmp_path, plan) as store:
        commit(store, plan, 'parent', [0])
        commit(store, plan, 'child', [1], 'parent')
        payload = EvidenceView(store).planner_feedback('parent')
        assert len(EvidenceView(store).prompt_feedback('parent')) <= 512
        assert payload['availability'] == 'unavailable' and payload['omitted_examples'] == 1
        assert payload['omitted_comparisons'] == 1 and payload['reason']


def test_foreign_plan_is_excluded_and_source_labels_are_never_rendered(tmp_path):
    from evoagentx.optimizers.aflow.store import atomic_json
    store, plan = fixture(tmp_path)
    with store:
        commit(store, plan, 'parent', [1, 1, 1, 1])
        summary = commit(store, plan, 'foreign', [0, 0, 0, 0], 'parent')
        payload = summary.to_dict()
        payload['plan_id'] = 'different-experiment-plan'
        atomic_json(tmp_path / 'trials' / 'foreign' / 'summary.json', payload)
        feedback = EvidenceView(store).planner_feedback('parent')
        assert feedback['comparison_count'] == 0 and feedback['comparisons'] == []
        assert 'first' not in canonical(feedback) and 'second' not in canonical(feedback)


def test_smallest_budget_still_holds_for_maximum_length_trial_id(tmp_path):
    config = ReliableTrialConfig(feedback_max_characters=512)
    plan = EvaluationPlan.create('local', [{'_id': 'x'}], {}, 'metric.v1', config)
    parent = 'p' * 100
    with TrialStore(tmp_path, plan) as store:
        commit(store, plan, parent, [0])
        commit(store, plan, 'child', [1], parent)
        view = EvidenceView(store)
        assert len(view.prompt_feedback(parent)) <= 512
        assert view.planner_feedback(parent)['reason']


def test_observed_cost_sum_overflow_has_explicit_marker_not_unknown_free_cost():
    from evoagentx.optimizers.aflow.feedback import MatchedFeedback
    plan = EvaluationPlan.create('local', [{'_id': 'x'}], {}, 'metric.v1', ReliableTrialConfig())
    example = plan.examples[0]
    parent = [Observation.scored('parent', example, 0, 1, 0, known_cost=0, pricing='known')]
    child = [Observation.scored('child', example, 0, attempt, 1, known_cost=1e308, pricing='known') for attempt in (1, 2)]
    cost = MatchedFeedback._cost_delta(parent, child)
    assert cost['value'] is None and cost['status'] == 'overflow'
    assert cost['known_subtotal_delta_status'] == 'overflow'
    json.dumps(cost, allow_nan=False)
