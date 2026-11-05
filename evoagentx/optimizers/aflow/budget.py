"""Persistent admission counters and conservative whole-operation reservations."""

import time
from .types import SCHEMA_VERSION, finite_score
from .store import atomic_json, read_json


class AdmissionBudget:
    """Bound only admissions controlled by the reliable AFlow owner.

    Nested provider retries/tokens are unobserved unless the caller's upper bound
    covers them. Reservations survive crashes and ambiguous completion. An elapsed
    deadline never establishes that noncooperative work has terminated.
    """

    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.state = store.read_state('budget', None)
        if self.state is None:
            self.state = {'schema_version': SCHEMA_VERSION, 'started_at': time.time(),
                          'counts': {'example': 0, 'proposal': 0, 'trial': 0},
                          'known_spent': 0.0, 'uncertain': False, 'halted': False, 'reservations': {}}
            self._save()
        if self.state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported budget schema')
        # Wall time makes the admission deadline survive process restarts; a
        # backwards clock is treated conservatively as an expired experiment.
        elapsed = time.time() - self.state['started_at']
        self._deadline = time.monotonic() + max(0, config.deadline_seconds - elapsed) if config.deadline_seconds is not None else None
        if elapsed < 0:
            self._deadline = time.monotonic()

    def _save(self):
        """Checkpoint counters before an admitted operation can start."""
        self.store.write_state('budget', self.state)

    def remaining_time(self):
        """Return remaining admission time, not a shutdown wall-clock guarantee."""
        return None if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    def check_deadline(self):
        """Stop further admission when the fixed experiment deadline elapses."""
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise RuntimeError('experiment admission deadline exhausted')

    def admit(self, stage, operation_id):
        """Persist one attempt and cost reservation before executing user code."""
        self.check_deadline()
        self.store.assert_resolved()
        if self.state['halted']:
            raise RuntimeError('supported pre-call cost bound violated; admission halted')
        limits = {'example': self.config.max_example_attempts, 'proposal': self.config.max_proposals,
                  'trial': self.config.max_trials}
        if stage not in limits:
            raise ValueError('invalid admission stage')
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', operation_id):
            raise ValueError('invalid admission ID')
        admission_path = self.store.path / 'admissions' / f'{operation_id}.json'
        if admission_path.exists() or operation_id in self.state['reservations']:
            raise ValueError('operation admission ID already used')
        if self.state['counts'][stage] >= limits[stage]:
            raise RuntimeError(f'{stage} attempt admission budget exhausted')
        bound = getattr(self.config, f'{stage}_cost_bound', None) or 0.0
        reserved = sum(value['reserved'] for value in self.state['reservations'].values())
        if self.config.max_known_cost is not None and self.state['known_spent'] + reserved + bound > self.config.max_known_cost + 1e-12:
            raise RuntimeError('known cost admission budget exhausted')
        reservation = {'stage': stage, 'reserved': bound, 'status': 'admitted'}
        self.state['counts'][stage] += 1
        self.state['reservations'][operation_id] = reservation
        self._save()
        atomic_json(admission_path, reservation)
        return dict(reservation)

    def finish(self, operation_id, known_cost, pricing_known, terminated=True):
        """Release only a terminated reservation; retain conservative unknown cost."""
        known_cost = finite_score(known_cost)
        if known_cost < 0:
            raise ValueError('negative observed cost')
        reservation = self.state['reservations'].get(operation_id)
        if reservation is None:
            raise ValueError('operation has no admission reservation')
        if reservation['status'] != 'admitted':
            raise ValueError('operation reservation already finalized')
        bound = reservation['reserved']
        if self.config.max_known_cost is not None and known_cost > bound + 1e-12:
            self.state['uncertain'] = True
            self.state['halted'] = True
            reservation['status'] = 'bound_violation'
            self._save()
            raise RuntimeError('supported pre-call cost bound violated; admission halted')
        self.state['known_spent'] += known_cost
        if not pricing_known:
            self.state['uncertain'] = True
        # A supplied upper bound remains reserved for unknown pricing, even if
        # execution terminated; absent observer events never prove zero dollars.
        reservation['reserved'] = 0.0 if terminated and pricing_known else max(0, bound - known_cost)
        reservation['status'] = 'finished' if terminated else 'unknown'
        atomic_json(self.store.path / 'admissions' / f'{operation_id}.json', reservation)
        if reservation['reserved'] == 0 and terminated:
            self.state['reservations'].pop(operation_id)
        self._save()

    def report(self):
        """Expose actual supported bounds and unobserved provider-internal usage."""
        return {'counts': dict(self.state['counts']), 'known_spent': self.state['known_spent'],
                'reserved': sum(r['reserved'] for r in self.state['reservations'].values()),
                'pricing_uncertain': self.state['uncertain'],
                'deadline_is_admission_only': True, 'provider_internal_attempts': 'unobserved',
                'hard_cost_bound': self.config.max_known_cost is not None,
                'tokens_hard_bounded': False}
