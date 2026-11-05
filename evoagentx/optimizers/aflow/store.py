"""Single-writer local checkpoints with atomic, versioned evidence publication.

The lock and rename protocol targets a local filesystem. It does not provide
multi-host/NFS coordination, a security sandbox, or exactly-once provider calls.
"""

import contextlib
import json
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .types import (
    SCHEMA_VERSION,
    EvaluationPlan,
    Observation,
    TrialSummary,
    canonical,
    digest,
)


def atomic_json(path, value):
    """Publish complete JSON and fsync the parent directory on local filesystems."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def sync_directory(path):
    """Persist a renamed directory entry where the platform supports it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path):
    """Read strict versioned JSON; corrupted checkpoints fail closed."""
    def reject_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    with open(path, encoding='utf-8') as stream:
        return json.load(stream, parse_constant=reject_constant)


class TrialStore:
    """Own one reliable experiment and its immutable evidence files.

    Args:
        path: Dedicated local reliable experiment directory.
        plan: Exact immutable evaluation plan used for every validation trial.
    """

    def __init__(self, path, plan):
        self.path = Path(path)
        self.plan = plan
        self._lock_file = None
        self._owner_token = uuid.uuid4().hex
        self._unclean = False
        self._active_operations = set()
        self._pending_operations = {}
        self._examples = {example.key: example for example in plan.examples}

    @classmethod
    def open_existing(cls, path, recover_owner=False):
        """Open an existing saved plan for explicit offline recovery/inspection.

        Args:
            path: Reliable experiment directory containing plan.json.
            recover_owner: Explicit acknowledgement of an unclean dead owner.

        This never automatically resolves unknown calls or admits paid work.
        A previous running/stopping process must have terminated before recovery.
        """
        plan = EvaluationPlan.from_dict(read_json(Path(path) / 'plan.json'))
        if recover_owner:
            plan = replace(plan, config=replace(plan.config, recover_owner=True))
        return cls(path, plan)

    def __enter__(self):
        """Acquire an OS-held writer lock before reading or mutating checkpoints."""
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError('reliable mode requires a supported local fcntl writer lock') from exc
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.path / 'writer.lock', 'a+', encoding='utf-8')
        try:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another writer owns this reliable experiment') from exc
            owner_file = self.path / 'owner.json'
            old_owner = read_json(owner_file) if owner_file.exists() else None
            if old_owner and old_owner.get('status') in ('running', 'stopping'):
                if not self.plan.config.recover_owner:
                    raise RuntimeError('unclean writer requires explicit recover_owner acknowledgement')
                # Even after flock is released, a live previous process may own
                # noncooperative thread/provider work. Never infer its termination.
                if old_owner.get('host') != socket.gethostname() or self._pid_alive(old_owner.get('pid')):
                    raise RuntimeError('previous writer may still be running; explicit termination is required')
                self._append_audit('recover_owner', previous=old_owner)
            plan_file = self.path / 'plan.json'
            if plan_file.exists():
                existing = read_json(plan_file)
                if existing.get('schema_version') != SCHEMA_VERSION:
                    raise ValueError('unsupported plan schema')
                if canonical(existing) != canonical(self.plan.to_dict()):
                    raise ValueError('experiment plan identity mismatch; use a new experiment directory')
            else:
                # A legacy result file supplies no coverage/split provenance.
                if (self.path / 'results.json').exists():
                    raise ValueError('legacy results cannot be imported as reliable evidence')
                atomic_json(plan_file, self.plan.to_dict())
            atomic_json(owner_file, self._owner('running'))
            for path in (self.path / 'attempts').glob('*.json'):
                operation = read_json(path)
                if operation['status'] in ('started', 'unknown', 'cancelled'):
                    self._pending_operations[operation['attempt_id']] = operation
            return self
        except BaseException:
            self._release()
            raise

    @staticmethod
    def _pid_alive(pid):
        """Conservatively reject recovery while a previous local process exists."""
        if not isinstance(pid, int) or pid < 1:
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _owner(self, status):
        """Build owner metadata without credentials or provider messages."""
        return {'schema_version': SCHEMA_VERSION, 'pid': os.getpid(),
                'host': socket.gethostname(), 'token': self._owner_token,
                'status': status, 'updated_at': time.time()}

    def _require_owner(self):
        """Reject accidental writes outside the owning context."""
        if self._lock_file is None:
            raise RuntimeError('writer context is not active')

    def mark_stopping(self):
        """Retain an unclean owner when cancellation cannot prove termination."""
        self._require_owner()
        self._unclean = True
        atomic_json(self.path / 'owner.json', self._owner('stopping'))

    def __exit__(self, exc_type, exc, tb):
        """Release the lock without erasing unknown external execution evidence."""
        try:
            if self._lock_file is not None:
                atomic_json(self.path / 'owner.json', self._owner('stopping' if self._unclean else 'closed'))
        finally:
            self._release()

    def _release(self):
        """Close the OS lock descriptor, including on failed initialization."""
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def _trial_path(self, trial_id):
        """Constrain all durable IDs to a single local path component."""
        if not isinstance(trial_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', trial_id):
            raise ValueError('invalid trial ID')
        return self.path / 'trials' / trial_id

    def create_trial(self, trial_id, round_number, kind, parent_id=None, modification=''):
        """Create an idempotent stable trial record before staging or execution."""
        self._require_owner()
        if kind not in ('baseline', 'candidate', 'test') or round_number < 0:
            raise ValueError('invalid trial kind or round')
        path = self._trial_path(trial_id) / 'trial.json'
        value = {'schema_version': SCHEMA_VERSION, 'trial_id': trial_id, 'round': round_number,
                 'kind': kind, 'parent_id': parent_id, 'modification': modification,
                 'plan_id': self.plan.identity}
        if path.exists():
            if canonical(read_json(path)) != canonical(value):
                raise ValueError('trial identity conflict')
        else:
            atomic_json(path, value)
        return value

    def trials(self):
        """Read trials in actual monotonic round/creation order."""
        return sorted((read_json(p) for p in (self.path / 'trials').glob('*/trial.json')),
                      key=lambda t: (t['round'], t['trial_id']))

    def trial(self, trial_id):
        """Read one stable trial identity."""
        return read_json(self._trial_path(trial_id) / 'trial.json')

    def stage_artifacts(self, trial_id, graph, prompt):
        """Publish graph and prompt together as one immutable artifact generation."""
        self._require_owner()
        trial_path = self._trial_path(trial_id)
        self.trial(trial_id)
        artifact_digest = digest({'graph': graph, 'prompt': prompt})
        directory = trial_path / 'artifacts'
        manifest = {'schema_version': SCHEMA_VERSION, 'digest': artifact_digest,
                    'directory': str(directory.relative_to(self.path)),
                    'graph_digest': digest(graph), 'prompt_digest': digest(prompt)}
        if directory.exists():
            if read_json(directory / 'manifest.json') != manifest:
                raise ValueError('artifact generation is immutable')
            self.read_artifacts(trial_id)
            return manifest
        temporary = Path(tempfile.mkdtemp(prefix='.generation-', dir=trial_path))
        try:
            for name, content in (('graph.py', graph), ('prompt.py', prompt)):
                with open(temporary / name, 'w', encoding='utf-8') as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            atomic_json(temporary / 'manifest.json', manifest)
            sync_directory(temporary)
            os.rename(temporary, directory)
            sync_directory(trial_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return manifest

    def read_artifacts(self, trial_id):
        """Verify exact generation content before loading trusted user code."""
        directory = self._trial_path(trial_id) / 'artifacts'
        manifest = read_json(directory / 'manifest.json')
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported artifact schema')
        graph = (directory / 'graph.py').read_text(encoding='utf-8')
        prompt = (directory / 'prompt.py').read_text(encoding='utf-8')
        if manifest['digest'] != digest({'graph': graph, 'prompt': prompt}):
            raise ValueError('artifact digest mismatch')
        return graph, prompt, manifest

    def start_attempt(self, trial_id, example, replicate, attempt):
        """Checkpoint intent before a possibly effectful external example call."""
        observation_id = digest([trial_id, example.key, replicate, attempt])
        return self.begin_operation(observation_id, 'example', trial_id,
                                    example_key=example.key, replicate=replicate, attempt=attempt)

    def begin_operation(self, attempt_id, stage, trial_id, **coordinates):
        """Persist a proposal/example intent; missing completion means unknown."""
        self._require_owner()
        self._trial_path(trial_id)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', attempt_id):
            raise ValueError('invalid attempt ID')
        value = {'schema_version': SCHEMA_VERSION, 'attempt_id': attempt_id, 'stage': stage,
                 'trial_id': trial_id, 'status': 'started', **coordinates}
        path = self.path / 'attempts' / f'{attempt_id}.json'
        if path.exists():
            raise ValueError('attempt already exists; never replay an external call ID')
        atomic_json(path, value)
        self._pending_operations[attempt_id] = value
        self._active_operations.add(attempt_id)
        return value

    def finish_operation(self, attempt_id, status, **metadata):
        """Record operation completion without overwriting immutable observations."""
        self._require_owner()
        path = self.path / 'attempts' / f'{attempt_id}.json'
        operation = read_json(path)
        if operation['status'] != 'started':
            if operation['status'] == status and all(operation.get(k) == v for k, v in metadata.items()):
                return
            raise ValueError('operation completion conflict')
        operation.update(status=status, **metadata)
        atomic_json(path, operation)
        self._active_operations.discard(attempt_id)
        if status in ('unknown', 'cancelled'):
            self._pending_operations[attempt_id] = operation
        else:
            self._pending_operations.pop(attempt_id, None)

    def record(self, observation):
        """Commit an immutable result before it can be consumed by selection."""
        self._require_owner()
        self.trial(observation.trial_id)
        example = self._examples.get(observation.example_key)
        if example is None or observation.replicate >= self.plan.config.replicates:
            raise ValueError('observation outside fixed plan')
        expected = digest([observation.trial_id, example.key, observation.replicate, observation.attempt])
        if observation.observation_id != expected or observation.example_id != example.original_id or observation.occurrence != example.occurrence:
            raise ValueError('observation identity conflict')
        path = self._trial_path(observation.trial_id) / 'observations' / f'{observation.observation_id}.json'
        value = observation.to_dict()
        if path.exists():
            if canonical(read_json(path)) != canonical(value):
                raise ValueError('observation payload conflict')
            return
        if (self._trial_path(observation.trial_id) / 'summary.json').exists():
            raise ValueError('committed trial evidence is immutable')
        atomic_json(path, value)
        operation = self.path / 'attempts' / f'{observation.observation_id}.json'
        if operation.exists():
            self.finish_operation(observation.observation_id, observation.status)

    def observations(self, trial_id):
        """Read immutable outcomes in deterministic replicate/example order."""
        values = [Observation(**read_json(p)) for p in (self._trial_path(trial_id) / 'observations').glob('*.json')]
        return sorted(values, key=lambda o: (o.replicate, o.example_key, o.attempt))

    def unknown_attempts(self):
        """Find ambiguous calls, including a crash between result and intent updates."""
        result = []
        for operation in self._pending_operations.values():
            if operation['attempt_id'] in self._active_operations:
                continue
            if operation['status'] not in ('started', 'unknown', 'cancelled'):
                continue
            observation_path = self._trial_path(operation['trial_id']) / 'observations' / f"{operation['attempt_id']}.json"
            if observation_path.exists() and read_json(observation_path)['status'] not in ('unknown', 'cancelled'):
                continue
            result.append(operation)
        return result

    def assert_resolved(self):
        """Block admission whenever an earlier external call may still have effects."""
        if self.unknown_attempts():
            raise RuntimeError('unknown external attempts require explicit resolution; no automatic retry')

    def resolve_unknown(self, attempt_id, action, note):
        """Audit an operator's confirmation of termination and replay decision.

        This API is an explicit operator assertion that the previous operation has
        stopped. It cannot terminate remote requests or Python threads itself.
        A rerun receives a new ID; completed scored results are never discarded.
        """
        self._require_owner()
        if action not in ('rerun', 'abandon') or not note or not note.strip():
            raise ValueError('resolution needs rerun/abandon and an operator note')
        operations = {o['attempt_id']: o for o in self.unknown_attempts()}
        if attempt_id not in operations:
            raise ValueError('attempt is not unknown')
        value = operations[attempt_id]
        value.update(status='resolved', resolution=action)
        self._append_audit('resolve_unknown', attempt_id=attempt_id, resolution=action, note=note)
        atomic_json(self.path / 'attempts' / f'{attempt_id}.json', value)
        self._pending_operations.pop(attempt_id, None)

    def operation(self, attempt_id):
        """Read an attempt's explicit recovery decision."""
        path = self.path / 'attempts' / f'{attempt_id}.json'
        return read_json(path) if path.exists() else None

    def commit_summary(self, summary):
        """Atomically publish authoritative evidence after verifying durable coverage."""
        self._require_owner()
        if summary.plan_id != self.plan.identity:
            raise ValueError('summary plan identity mismatch')
        actual = TrialSummary.build(self.plan, summary.trial_id, summary.round,
                                    self.observations(summary.trial_id))
        if actual != summary:
            raise ValueError('summary does not match durable observations')
        path = self._trial_path(summary.trial_id) / 'summary.json'
        if path.exists() and canonical(read_json(path)) != canonical(summary.to_dict()):
            raise ValueError('summary conflict')
        atomic_json(path, summary.to_dict())

    def summary(self, trial_id):
        """Read one published summary without using diagnostic export rows."""
        path = self._trial_path(trial_id) / 'summary.json'
        return TrialSummary.from_dict(read_json(path)) if path.exists() else None

    def summaries(self):
        """Return only committed summaries in actual trial order."""
        return [summary for trial in self.trials() if (summary := self.summary(trial['trial_id'])) is not None]

    def read_state(self, name, default=None):
        """Read owner-maintained stage/budget state independently of exports."""
        if not re.fullmatch(r'[a-z_]+', name):
            raise ValueError('invalid state name')
        path = self.path / f'{name}.json'
        return read_json(path) if path.exists() else default

    def write_state(self, name, value):
        """Publish owner-maintained state with the same local atomic protocol."""
        self._require_owner()
        if not re.fullmatch(r'[a-z_]+', name):
            raise ValueError('invalid state name')
        atomic_json(self.path / f'{name}.json', value)

    def _append_audit(self, action, **details):
        """Append a rare recovery event atomically; normal observations are separate."""
        path = self.path / 'audit.json'
        values = read_json(path) if path.exists() else []
        values.append({'action': action, 'at': time.time(), **details})
        atomic_json(path, values)

    def audit(self):
        """Read explicit recovery history for operator inspection."""
        return self.read_state('audit', [])
