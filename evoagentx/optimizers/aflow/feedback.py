"""Bounded matched validation diagnostics for AFlow's next graph proposal.

Observed score changes identify examples worth inspecting. They do not establish
causes, step-level provenance, statistical superiority or a better agent policy.
"""

import math
from collections import Counter, defaultdict

from .types import SCHEMA_VERSION, canonical

OPEN_TAG = '<validation_diagnostics>'
CLOSE_TAG = '</validation_diagnostics>'


def finite_difference(after, before):
    """Represent unavailable or overflowing differences without nonfinite JSON."""
    if after is None or before is None:
        return None, 'unavailable'
    value = after - before
    return (value, 'finite') if math.isfinite(value) else (None, 'overflow')


def finite_total(values):
    """Sum observed nonnegative costs without emitting infinity on overflow."""
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


class MatchedFeedback:
    """Compare exact committed plans, then apply deterministic output budgets."""

    def __init__(self, store):
        self.store = store
        self.plan = store.plan
        self.config = self.plan.config
        self._expected = {
            (canonical(example.original_id), example.occurrence, replicate): example
            for replicate in range(self.config.replicates) for example in self.plan.examples
        }

    def _history(self, trial_id):
        """Match IDs plus occurrence/replicate, validating exact plan membership."""
        histories = defaultdict(list)
        for observation in self.store.observations(trial_id):
            key = (canonical(observation.example_id), observation.occurrence, observation.replicate)
            example = self._expected.get(key)
            if example is None or observation.example_key != example.key:
                raise ValueError('matched feedback observation identity mismatch')
            histories[key].append(observation)
        for values in histories.values():
            values.sort(key=lambda observation: observation.attempt)
        return histories

    @staticmethod
    def _cost_delta(parent, child):
        """Separate known observed subtotals from a complete priced cost delta."""
        before = finite_total(observation.known_cost for observation in parent)
        after = finite_total(observation.known_cost for observation in child)
        delta, status = finite_difference(after, before) if before is not None and after is not None else (None, 'overflow')
        known = bool(parent and child) and all(observation.pricing == 'known' for observation in parent + child)
        return {'value': delta if known else None,
                'status': status if known else 'unknown',
                'known_subtotal_delta': delta, 'known_subtotal_delta_status': status,
                'parent_known_cost': before, 'candidate_known_cost': after,
                'scope': 'recorded_example_attempts'}

    def _change(self, before, after):
        """Use ordering, avoiding overflowing subtraction when deciding direction."""
        if before == after:
            return 'tied'
        improved = after > before if self.config.direction == 'maximize' else after < before
        return 'improved' if improved else 'regressed'

    def _comparison(self, parent, child, remaining_examples):
        """Keep task conclusions separate from complete and incomplete execution data."""
        before = self._history(parent.trial_id)
        after = self._history(child.trial_id)
        task_conclusions = parent.eligible and child.eligible
        matched = 0
        recovered = 0
        counts = {'improved': 0, 'regressed': 0, 'tied': 0, 'infrastructure': 0, 'missing': 0}
        parent_statuses = Counter()
        child_statuses = Counter()
        examples = []
        for key, example in self._expected.items():
            parent_history = before.get(key, [])
            child_history = after.get(key, [])
            original = parent_history[-1] if parent_history else None
            candidate = child_history[-1] if child_history else None
            parent_status = original.status if original is not None else 'missing'
            child_status = candidate.status if candidate is not None else 'missing'
            parent_statuses[parent_status] += 1
            child_statuses[child_status] += 1
            paired = parent_status == child_status == 'scored'
            matched += paired
            for history in (parent_history, child_history):
                if history and history[-1].status == 'scored':
                    recovered += sum(observation.status == 'retryable' for observation in history[:-1])
            if original is None or candidate is None:
                change = 'missing'
            elif not paired:
                change = 'infrastructure'
            elif task_conclusions:
                change = self._change(original.score, candidate.score)
            else:
                change = 'unavailable'
            if change in counts:
                counts[change] += 1
            if len(examples) >= remaining_examples:
                continue
            score_delta, delta_status = finite_difference(
                candidate.score if candidate is not None and task_conclusions else None,
                original.score if original is not None and task_conclusions else None)
            examples.append({'example_id': example.original_id, 'occurrence': example.occurrence,
                             'replicate': key[2], 'parent_status': parent_status,
                             'candidate_status': child_status, 'change': change,
                             'parent_score': original.score if paired and task_conclusions else None,
                             'candidate_score': candidate.score if paired and task_conclusions else None,
                             'score_delta': score_delta, 'score_delta_status': delta_status,
                             'cost_delta': self._cost_delta(parent_history, child_history)})
        delta, delta_status = finite_difference(child.score, parent.score)
        all_before = [observation for values in before.values() for observation in values]
        all_after = [observation for values in after.values() for observation in values]
        cost_delta = self._cost_delta(all_before, all_after)
        if matched != self.plan.requested:
            # Missing/unscored coordinates cannot establish complete trial cost,
            # even if every recorded event happened to have known pricing.
            cost_delta['value'] = None
            cost_delta['status'] = 'unknown'
        return {'trial_id': child.trial_id, 'parent_id': parent.trial_id,
                'task_conclusions': task_conclusions,
                'evidence': ('accepted' if child.improves(parent, self.config) else 'rejected') if task_conclusions else 'inconclusive',
                'parent_score': parent.score, 'candidate_score': child.score,
                'score_delta': delta, 'score_delta_status': delta_status,
                'counts': counts,
                'coverage': {'requested': self.plan.requested, 'parent_scored': parent.scored,
                             'candidate_scored': child.scored, 'matched_scored': matched,
                             'parent_statuses': dict(parent_statuses), 'candidate_statuses': dict(child_statuses),
                             'recovered_retry_attempts': recovered},
                'cost_delta': cost_delta, 'examples': examples,
                'omitted_examples': self.plan.requested - len(examples)}

    def _envelope(self, parent_id, comparison_count):
        """Reserve enough space for explicit coverage/omission and fallback metadata."""
        return {'schema_version': SCHEMA_VERSION, 'parent_id': parent_id, 'split': 'validation',
                'metric_direction': self.config.direction,
                'availability': 'unavailable', 'reason': 'No comparable committed validation candidates.',
                'interpretation': 'Observed validation associations only; no causal attribution.',
                'comparison_count': comparison_count, 'omitted_comparisons': comparison_count,
                'example_count': 0, 'omitted_examples': self.plan.requested * comparison_count,
                'comparisons': []}

    def _fits(self, payload):
        """Bound actual tagged prompt characters, not only the inner JSON length."""
        return len(OPEN_TAG) + len(canonical(payload)) + len(CLOSE_TAG) <= self.config.feedback_max_characters

    def payload(self, parent_id):
        """Select recent candidates then fixed plan order under shared hard budgets."""
        parent = self.store.summary(parent_id)
        candidates = []
        if parent is not None and parent.plan_id == self.plan.identity and parent.split == self.plan.split == 'validation':
            for trial in reversed(self.store.trials()):
                if trial['kind'] != 'candidate' or trial['parent_id'] != parent_id:
                    continue
                child = self.store.summary(trial['trial_id'])
                if child is not None and child.plan_id == parent.plan_id and child.split == 'validation':
                    candidates.append(child)
        payload = self._envelope(parent_id, len(candidates))
        if not candidates:
            return payload
        payload['reason'] = None
        for child in candidates:
            remaining = self.config.feedback_max_examples - payload['example_count']
            comparison = self._comparison(parent, child, remaining)
            payload['comparisons'].append(comparison)
            payload['omitted_comparisons'] -= 1
            payload['example_count'] += len(comparison['examples'])
            payload['omitted_examples'] -= len(comparison['examples'])
            payload['availability'] = 'available'
            while comparison['examples'] and not self._fits(payload):
                comparison['examples'].pop()
                comparison['omitted_examples'] += 1
                payload['example_count'] -= 1
                payload['omitted_examples'] += 1
            if not self._fits(payload):
                payload['comparisons'].pop()
                payload['omitted_comparisons'] += 1
                break
        if not payload['comparisons']:
            payload['availability'] = 'unavailable'
            payload['reason'] = 'Detail omitted by budget; scalar summaries only.'
            if not self._fits(payload):
                payload['interpretation'] = 'Observed only; no causal attribution.'
        return payload

    @staticmethod
    def render(payload):
        """Render exactly the JSON payload persisted for this proposal operation."""
        return OPEN_TAG + canonical(payload) + CLOSE_TAG
