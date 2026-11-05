"""Shared consumers of committed AFlow summaries and context-local costs."""

import threading
from dataclasses import dataclass

from .store import atomic_json


class CostLedger:
    """Accumulate bounded context-local accounting, including supported threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cost = 0.0
        self._events = 0
        self._unknown = False
        self._input = 0
        self._output = 0
        self._closed = False
        self._late_events = 0

    def observe(self, event):
        """Accept provider events without performing disk writes from workers."""
        with self._lock:
            if self._closed:
                self._late_events += 1
                return
            self._events += 1
            self._unknown |= event.pricing != 'known'
            self._cost += event.cost or 0.0
            self._input += event.input_tokens
            self._output += event.output_tokens

    def close(self):
        """Freeze a snapshot; late accounting cannot mutate committed evidence."""
        with self._lock:
            self._closed = True
            return {'known_cost': self._cost,
                    'pricing': 'unknown' if self._unknown else ('known' if self._events else 'unobserved'),
                    'input_tokens': self._input, 'output_tokens': self._output}

    @property
    def late_events(self):
        """Expose ignored late events for diagnostics; global totals still include them."""
        with self._lock:
            return self._late_events


@dataclass(frozen=True)
class EvolutionResult:
    """Truthful terminal status, selected summary and controlled budget report."""

    status: str
    reason: str
    best_trial_id: str | None
    best_round: int | None
    score: float | None
    committed_trials: int
    budget: dict
    export_error: str | None = None


class EvidenceView:
    """Use exactly one committed score for selection, feedback and convergence."""

    def __init__(self, store):
        self.store = store
        self.config = store.plan.config

    def comparable(self):
        """Select complete validation trials from the exact fixed experiment plan."""
        return [s for s in self.store.summaries() if s.eligible and s.split == 'validation'
                and s.plan_id == self.store.plan.identity]

    def ranked(self):
        """Rank eligible summaries stably, retaining earlier trials on ties."""
        reverse = self.config.direction == 'maximize'
        return sorted(self.comparable(), key=lambda s: s.score, reverse=reverse)

    def best(self):
        """Apply the explicit improvement margin in actual trial order."""
        parent = None
        for summary in self.comparable():
            if parent is None or summary.improves(parent, self.config):
                parent = summary
        return parent

    def feedback(self):
        """Distinguish task evidence from infrastructure without strategy prohibitions."""
        feedback = []
        for summary in self.store.summaries():
            trial = self.store.trial(summary.trial_id)
            if trial['kind'] != 'candidate' or summary.split != 'validation':
                continue
            parent = self.store.summary(trial['parent_id'])
            accepted = summary.improves(parent, self.config) if parent is not None else False
            feedback.append({'trial_id': summary.trial_id, 'parent_id': trial['parent_id'],
                             'modification': trial['modification'],
                             'evidence': ('accepted' if accepted else 'rejected') if summary.eligible else 'inconclusive',
                             'before': parent.score if parent is not None else None,
                             'after': summary.score, 'requested': summary.requested,
                             'scored': summary.scored, 'outcomes': summary.outcomes})
        return feedback

    def planner_feedback(self, parent_id):
        """Return bounded matched diagnostics shared by planner and report consumers."""
        from .feedback import MatchedFeedback
        return MatchedFeedback(self.store).payload(parent_id)

    def prompt_feedback(self, parent_id, payload=None):
        """Render the exact persisted proposal evidence without task-cause claims."""
        from .feedback import MatchedFeedback
        return MatchedFeedback.render(self.planner_feedback(parent_id) if payload is None else payload)

    def converged(self):
        """Count only comparable committed trials since the last strict improvement."""
        parent = None
        unchanged = 0
        for summary in self.comparable():
            if parent is None or summary.improves(parent, self.config):
                parent = summary
                unchanged = 0
            else:
                unchanged += 1
        return unchanged >= self.config.convergence_patience

    def export(self, root_path):
        """Repair legacy-shaped exports from authoritative evidence, idempotently.

        Export failure is diagnostic only: a committed summary remains committed.
        Inconclusive candidates are excluded from legacy failure experience.
        """
        from pathlib import Path
        root = Path(root_path)
        rows = []
        for summary in self.comparable():
            rows.append({'round': summary.round, 'score': summary.score,
                         'avg_cost': summary.known_cost / summary.requested,
                         'total_cost': summary.known_cost, 'time': '',
                         'trial_id': summary.trial_id, 'split': summary.split,
                         'pricing': summary.pricing, 'plan_id': summary.plan_id})
        atomic_json(root / 'results.json', rows)
        for row in self.feedback():
            summary = self.store.summary(row['trial_id'])
            atomic_json(root / f'round_{summary.round}' / 'trial_feedback.json', row)
            if row['evidence'] != 'inconclusive':
                parent = self.store.summary(row['parent_id'])
                atomic_json(root / f'round_{summary.round}' / 'experience.json',
                            {'father node': parent.round, 'modification': row['modification'],
                             'before': row['before'], 'after': row['after'],
                             'succeed': row['evidence'] == 'accepted'})
        atomic_json(self.store.path / 'feedback.json', self.feedback())
