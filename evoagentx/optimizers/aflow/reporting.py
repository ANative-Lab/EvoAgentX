"""Inspectable per-example evidence and recovery guidance for local trials."""

from collections import Counter, defaultdict

from .evidence import EvidenceView
from .store import atomic_json, read_json
from .types import SCHEMA_VERSION, TrialSummary


class TrialReporter:
    """Build reports from immutable outcomes, never infer missing task scores.

    Reports are repairable exports. Only each trial's summary.json authorizes
    ranking; a partially written or absent report cannot change promotion.
    """

    def __init__(self, store):
        self.store = store
        self.plan = store.plan

    @classmethod
    def inspect(cls, path, recover_owner=False):
        """Take an owner-locked snapshot without executing any user graph/provider."""
        from .store import TrialStore
        with TrialStore.open_existing(path, recover_owner=recover_owner) as store:
            return cls(store).experiment()

    def _histories(self, trial_id):
        """Group attempt evidence by the exact fixed replicate/example coordinate."""
        histories = defaultdict(list)
        for observation in self.store.observations(trial_id):
            histories[(observation.example_key, observation.replicate)].append(observation)
        for values in histories.values():
            values.sort(key=lambda observation: observation.attempt)
        return histories

    def _pending(self, trial_id):
        """Index unresolved operations without exposing provider exception messages."""
        return [operation for operation in self.store.unknown_attempts()
                if operation['trial_id'] == trial_id]

    def _example_row(self, example, replicate, observations, pending):
        """Represent missing, scored and infrastructure evidence distinctly."""
        latest = observations[-1] if observations else None
        unresolved = [operation['attempt_id'] for operation in pending
                      if operation.get('example_key') == example.key and operation.get('replicate') == replicate]
        if latest is not None:
            status = latest.status
        else:
            status = 'unknown' if unresolved else 'missing'
        return {'example_index': example.index, 'example_id': example.original_id,
                'occurrence': example.occurrence, 'content_digest': example.content_digest,
                'replicate': replicate, 'status': status,
                'score': latest.score if latest is not None and latest.status == 'scored' else None,
                'latest_attempt': latest.attempt if latest is not None else None,
                'error_type': latest.error_type if latest is not None else None,
                'known_cost': sum(observation.known_cost for observation in observations),
                'pricing': 'known' if observations and all(o.pricing == 'known' for o in observations) else 'unknown',
                'input_tokens': sum(o.input_tokens for o in observations),
                'output_tokens': sum(o.output_tokens for o in observations),
                'unresolved_attempt_ids': unresolved,
                'attempts': [observation.to_dict() for observation in observations]}

    def trial(self, trial_id):
        """Report every requested coordinate, including those never admitted."""
        trial = self.store.trial(trial_id)
        summary = self.store.summary(trial_id)
        observations = self.store.observations(trial_id)
        diagnostic = TrialSummary.build(self.plan, trial_id, trial['round'], observations)
        if summary is not None and summary != diagnostic:
            raise ValueError('committed summary does not match immutable evidence')
        histories = self._histories(trial_id)
        pending = self._pending(trial_id)
        examples = []
        for replicate in range(self.plan.config.replicates):
            for example in self.plan.examples:
                values = histories.get((example.key, replicate), [])
                examples.append(self._example_row(example, replicate, values, pending))
        artifact = None
        directory = self.store._trial_path(trial_id) / 'artifacts'
        if directory.exists():
            _, _, artifact = self.store.read_artifacts(trial_id)
        return {'schema_version': SCHEMA_VERSION, 'trial': trial,
                'plan_id': self.plan.identity, 'split': self.plan.split,
                'summary_committed': summary is not None,
                'summary': (summary or diagnostic).to_dict(),
                'artifact': artifact, 'examples': examples,
                'pending_operations': pending,
                'coverage': {'requested': self.plan.requested,
                             'scored': diagnostic.scored,
                             'missing_or_unscored': self.plan.requested - diagnostic.scored,
                             'attempts_recorded': len(observations)},
                'ranking_eligible': bool(summary and summary.eligible and summary.split == 'validation')}

    def _operation_totals(self):
        """Expose controlled attempts by stage/status without provider-internal claims."""
        stages = defaultdict(Counter)
        for path in (self.store.path / 'attempts').glob('*.json'):
            operation = read_json(path)
            stages[operation['stage']][operation['status']] += 1
        return {stage: dict(counts) for stage, counts in stages.items()}

    def recovery(self):
        """Explain exactly which unknown calls block further automatic execution."""
        unknown = self.store.unknown_attempts()
        owner = self.store.read_state('owner', {})
        return {'blocked': bool(unknown), 'owner_status': owner.get('status'),
                'unknown_attempts': unknown,
                'required_action': ('Confirm previous local/remote work terminated, then explicitly resolve each unknown '
                                    'attempt as rerun or abandon with an operator note.' if unknown else None),
                'replay_policy': 'Completed observations are never repeated. Unknown calls are never automatically repeated.',
                'termination_limit': 'Cancelling an async awaiter does not terminate arbitrary threads or remote requests.'}

    def operation_accounting(self):
        """Retain proposal/retry costs after the stage cursor has advanced."""
        rows = []
        for path in sorted((self.store.path / 'attempts').glob('*.json')):
            operation = read_json(path)
            admission = self.store.path / 'admissions' / path.name
            costs = read_json(admission) if admission.exists() else {}
            rows.append({'attempt_id': operation['attempt_id'], 'trial_id': operation['trial_id'],
                         'stage': operation['stage'], 'status': operation['status'],
                         'known_cost': costs.get('known_cost', 0.0),
                         'pricing_known': costs.get('pricing_known', False),
                         'reserved_upper_bound': costs.get('reserved', 0.0)})
        return rows

    def experiment(self):
        """Take a stable owner-held inventory of summaries, outcomes and recovery."""
        view = EvidenceView(self.store)
        best = view.best()
        reports = [self.trial(trial['trial_id']) for trial in self.store.trials()]
        summaries = self.store.summaries()
        budget = self.store.read_state('budget', {})
        parents = {trial['parent_id'] for trial in self.store.trials()
                   if trial['kind'] == 'candidate' and trial['parent_id'] is not None}
        if best is not None:
            parents.add(best.trial_id)
        proposal_feedback = {}
        for path in sorted((self.store.path / 'attempts').glob('*.json')):
            operation = read_json(path)
            if operation['stage'] == 'proposal' and 'planner_feedback' in operation:
                proposal_feedback[operation['attempt_id']] = operation['planner_feedback']
        return {'schema_version': SCHEMA_VERSION, 'experiment_id': self.plan.config.experiment_id,
                'plan_id': self.plan.identity, 'split': self.plan.split,
                'direction': self.plan.config.direction, 'margin': self.plan.config.improvement_margin,
                'requested_per_trial': self.plan.requested,
                'best_trial_id': best.trial_id if best else None,
                'best_round': best.round if best else None,
                'best_score': best.score if best else None,
                'ranked_trials': [{'trial_id': s.trial_id, 'round': s.round, 'score': s.score}
                                  for s in view.ranked()],
                'converged': view.converged(),
                'committed_trials': len(summaries),
                'incomplete_trials': [s.trial_id for s in summaries if not s.eligible],
                'evolution': self.store.read_state('evolution', None),
                'feedback': view.feedback(), 'operation_totals': self._operation_totals(),
                'planner_feedback': {parent: view.planner_feedback(parent) for parent in sorted(parents)},
                'proposal_feedback': proposal_feedback,
                'operation_accounting': self.operation_accounting(),
                'cost': {'known_spent': budget.get('known_spent', 0),
                         'uncertain': budget.get('uncertain', True),
                         'reserved_upper_bounds': sum(r['reserved'] for r in budget.get('reservations', {}).values()),
                         'proposal_cost_included': True,
                         'provider_internal_requests': 'unobserved',
                         'unsupported_accounting_paths': ['raw threads', 'external processes', 'providers bypassing CostManager']},
                'deadline_semantics': 'Stops new admission and requests cooperative cancellation; shutdown may exceed deadline.',
                'recovery': self.recovery(), 'audit': self.store.audit(),
                'export': self.store.read_state('export_status', None), 'trials': reports}

    def export(self):
        """Publish repairable per-example and experiment reports atomically."""
        report = self.experiment()
        directory = self.store.path / 'reports'
        for trial in report['trials']:
            atomic_json(directory / f"{trial['trial']['trial_id']}.json", trial)
        atomic_json(directory / 'experiment.json', report)
        return report
