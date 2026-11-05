"""Immutable identities and evidence for opt-in AFlow experiments."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SCHEMA_VERSION = 1
_SECRET = re.compile(r'api_?key|secret|password|authorization|credential|(?:^|_)(?:key|token)$', re.IGNORECASE)


def canonical(value: Any) -> str:
    """Encode exact JSON data without accepting nonfinite or opaque values."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    """Hash exact canonical content for identity, not semantic equivalence."""
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def public_model_config(value: Any) -> Any:
    """Remove credentials and endpoint credentials before identity persistence."""
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            # Generation limits are behavior, whereas authentication tokens are secrets.
            normalized_key = str(key).lower().replace('-', '_')
            if _SECRET.search(normalized_key) and normalized_key not in ('max_tokens', 'max_completion_tokens'):
                continue
            if normalized_key.endswith(('base_url', 'api_base', 'endpoint')) and isinstance(item, str):
                url = urlsplit(item)
                host = url.hostname or ''
                if ':' in host:
                    host = f'[{host}]'
                authority = f'{host}:{url.port}' if url.port is not None else host
                item = urlunsplit((url.scheme, authority, url.path, '', ''))
            result[str(key)] = public_model_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [public_model_config(item) for item in value]
    canonical(value)
    return value


def finite_score(value: Any) -> float:
    """Require a numeric finite scalar rather than coercing error values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Numpy scalar metrics expose item(), but arrays are deliberately rejected.
        if hasattr(value, 'shape') and value.shape == () and hasattr(value, 'item'):
            value = value.item()
        else:
            raise ValueError('metric must be a finite numeric scalar')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('metric must be a finite numeric scalar')
    return float(value)


@dataclass(frozen=True)
class ReliableTrialConfig:
    """Fixed experiment policy; limits cover owned admissions, not provider internals.

    Cost bounds are a caller-supplied pre-call upper-bound contract covering the
    entire graph/proposal operation, including any internal retries. Without that
    contract max_known_cost is rejected. A deadline requests cancellation but
    cannot establish termination of arbitrary code or threads.
    """

    experiment_id: str = 'aflow'
    direction: str = 'maximize'
    improvement_margin: float = 0.0
    replicates: int = 1
    concurrency: int = 20
    attempts_per_example: int = 2
    max_example_attempts: int = 1000000
    max_proposals: int = 60
    max_trials: int = 21
    duplicate_limit: int = 10
    deadline_seconds: float | None = None
    attempt_timeout: float | None = None
    proposal_timeout: float | None = None
    max_known_cost: float | None = None
    example_cost_bound: float | None = None
    proposal_cost_bound: float | None = None
    retryable_errors: tuple = ('builtins.TimeoutError', 'builtins.ConnectionError')
    convergence_patience: int = 5
    metric_id: str | None = None
    recover_owner: bool = False
    feedback_max_examples: int = 12
    feedback_max_characters: int = 12000

    def __post_init__(self):
        """Reject ambiguous budgets before any external work can be admitted."""
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', self.experiment_id):
            raise ValueError('experiment_id must be a local filename-safe identifier')
        if self.direction not in ('maximize', 'minimize'):
            raise ValueError('direction must be maximize or minimize')
        if finite_score(self.improvement_margin) < 0:
            raise ValueError('improvement_margin must be nonnegative')
        for name in ('replicates', 'concurrency', 'attempts_per_example', 'max_example_attempts',
                     'max_proposals', 'max_trials', 'duplicate_limit', 'convergence_patience',
                     'feedback_max_examples', 'feedback_max_characters'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.feedback_max_characters < 512:
            raise ValueError('feedback_max_characters must be at least 512 for an explicit diagnostic fallback')
        for name in ('deadline_seconds', 'attempt_timeout', 'proposal_timeout', 'max_known_cost',
                     'example_cost_bound', 'proposal_cost_bound'):
            value = getattr(self, name)
            if value is not None and finite_score(value) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.max_known_cost is not None and (self.example_cost_bound is None or self.proposal_cost_bound is None):
            raise ValueError('cost admission requires example and proposal pre-call upper bounds')
        if not all(isinstance(name, str) and '.' in name for name in self.retryable_errors):
            raise ValueError('retryable_errors must contain qualified exception names')
        object.__setattr__(self, 'retryable_errors', tuple(self.retryable_errors))

    def identity_dict(self) -> dict:
        """Exclude operational owner recovery from the immutable execution plan."""
        value = asdict(self)
        value.pop('recover_owner')
        return value


@dataclass(frozen=True)
class ExampleIdentity:
    """An ordered example occurrence, retaining its benchmark identifier."""

    index: int
    original_id: Any
    occurrence: int
    content_digest: str

    @property
    def key(self) -> str:
        """Return an unambiguous position key even for duplicate dataset IDs."""
        return f'{self.index}:{self.occurrence}:{self.content_digest}'


@dataclass(frozen=True)
class EvaluationPlan:
    """Complete immutable split coverage and behavior identity."""

    dataset: str
    split: str
    examples: tuple
    model_config: dict
    metric_id: str
    config: ReliableTrialConfig
    identity: str
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, dataset: str, data: list, model_config: Any, metric_id: str,
               config: ReliableTrialConfig, split: str = 'validation'):
        """Freeze exact data order and occurrence identity before evaluation."""
        if split not in ('validation', 'test'):
            raise ValueError('split must be validation or test')
        if not data:
            raise ValueError('an evaluation plan requires nonempty data')
        if not metric_id or not dataset:
            raise ValueError('dataset and metric identity are required')
        seen = Counter()
        examples = []
        for index, example in enumerate(data):
            original_id = example.get('_id', example.get('id', example.get('task_id', index))) if isinstance(example, dict) else index
            key = canonical(original_id)
            examples.append(ExampleIdentity(index, original_id, seen[key], digest(example)))
            seen[key] += 1
        model = public_model_config(model_config)
        identity = digest({'schema': SCHEMA_VERSION, 'dataset': dataset, 'split': split,
                           'examples': [asdict(e) for e in examples], 'model': model,
                           'metric': metric_id, 'policy': config.identity_dict()})
        return cls(dataset, split, tuple(examples), model, metric_id, config, identity)

    def to_dict(self) -> dict:
        """Return the versioned durable plan without raw example contents."""
        value = asdict(self)
        value['config'] = self.config.identity_dict()
        return value

    @classmethod
    def from_dict(cls, value):
        """Reconstruct a versioned plan for offline inspection and explicit recovery.

        Resume still compares the live benchmark/model configuration with this
        plan. Loading a saved plan alone never proves current data provenance.
        """
        value = dict(value)
        if value.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported plan schema')
        config = ReliableTrialConfig(**value['config'])
        examples = tuple(ExampleIdentity(**e) for e in value['examples'])
        if not examples or [e.index for e in examples] != list(range(len(examples))):
            raise ValueError('invalid ordered example coverage')
        occurrences = Counter()
        for example in examples:
            key = canonical(example.original_id)
            if example.occurrence != occurrences[key]:
                raise ValueError('invalid example occurrence identity')
            if not re.fullmatch(r'[0-9a-f]{64}', example.content_digest):
                raise ValueError('invalid example content digest')
            occurrences[key] += 1
        if value['split'] not in ('validation', 'test') or not value['dataset'] or not value['metric_id']:
            raise ValueError('invalid dataset/split/metric identity')
        model = public_model_config(value['model_config'])
        if canonical(model) != canonical(value['model_config']):
            raise ValueError('saved model configuration contains credentials')
        identity = digest({'schema': SCHEMA_VERSION, 'dataset': value['dataset'], 'split': value['split'],
                           'examples': [asdict(e) for e in examples], 'model': model,
                           'metric': value['metric_id'], 'policy': config.identity_dict()})
        if value['identity'] != identity:
            raise ValueError('saved plan identity mismatch')
        result = cls(value['dataset'], value['split'], examples, model, value['metric_id'], config, identity)
        if canonical(result.to_dict()) != canonical(value):
            raise ValueError('unsupported or inconsistent plan fields')
        return result

    def verify_data(self, data: list):
        """Reject mutated/reordered examples before any new attempt is admitted."""
        if len(data) != len(self.examples) or any(digest(d) != e.content_digest for d, e in zip(data, self.examples)):
            raise ValueError('evaluation data identity changed')

    @property
    def requested(self) -> int:
        """Return the exact fixed replicate/example denominator."""
        return len(self.examples) * self.config.replicates


@dataclass(frozen=True)
class Observation:
    """Immutable outcome of one owned example attempt; never an inferred zero."""

    observation_id: str
    trial_id: str
    example_key: str
    example_id: Any
    occurrence: int
    replicate: int
    attempt: int
    status: str
    score: float | None = None
    error_type: str | None = None
    known_cost: float = 0.0
    pricing: str = 'unobserved'
    input_tokens: int = 0
    output_tokens: int = 0
    duration: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        """Validate durable evidence, including reloaded observations."""
        if self.status not in ('scored', 'retryable', 'inconclusive', 'unknown', 'cancelled'):
            raise ValueError('invalid observation status')
        if self.status == 'scored':
            object.__setattr__(self, 'score', finite_score(self.score))
        elif self.score is not None:
            raise ValueError('unscored observations cannot contain a task score')
        if self.replicate < 0 or self.attempt < 1:
            raise ValueError('invalid observation coordinates')
        if finite_score(self.known_cost) < 0 or finite_score(self.duration) < 0:
            raise ValueError('negative cost or duration')
        if self.pricing not in ('known', 'unknown', 'unobserved'):
            raise ValueError('invalid pricing status')
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError('unsupported observation schema')

    @classmethod
    def make(cls, trial_id, example, replicate, attempt, status, **kwargs):
        """Build the deterministic idempotency key from attempt coordinates."""
        key = digest([trial_id, example.key, replicate, attempt])
        return cls(key, trial_id, example.key, example.original_id, example.occurrence,
                   replicate, attempt, status, **kwargs)

    @classmethod
    def scored(cls, trial_id, example, replicate, attempt, score, **kwargs):
        """Construct explicit task evidence, including valid zero scores."""
        return cls.make(trial_id, example, replicate, attempt, 'scored', score=score, **kwargs)

    def to_dict(self):
        """Return the versioned observation payload."""
        return asdict(self)


@dataclass(frozen=True)
class TrialSummary:
    """One authoritative score and diagnostic denominator for an entire trial."""

    trial_id: str
    round: int
    plan_id: str
    split: str
    requested: int
    scored: int
    eligible: bool
    score: float | None
    partial_score: float | None
    observation_ids: tuple
    known_cost: float
    pricing: str
    outcomes: dict
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def build(cls, plan, trial_id, round_number, observations):
        """Require every requested coordinate and reject foreign/conflicting evidence."""
        expected = {(e.key, r) for e in plan.examples for r in range(plan.config.replicates)}
        latest = {}
        ids = set()
        cost = 0.0
        pricing = 'known'
        for obs in observations:
            coordinate = (obs.example_key, obs.replicate)
            if obs.trial_id != trial_id or coordinate not in expected:
                raise ValueError('observation does not belong to trial plan')
            if obs.observation_id in ids:
                raise ValueError('duplicate observation in summary input')
            ids.add(obs.observation_id)
            cost += obs.known_cost
            if obs.pricing != 'known':
                pricing = 'unknown'
            if coordinate not in latest or latest[coordinate].attempt < obs.attempt:
                latest[coordinate] = obs
            elif latest[coordinate].attempt == obs.attempt:
                raise ValueError('conflicting attempt coordinates')
        scored = [o.score for o in latest.values() if o.status == 'scored']
        complete = len(scored) == plan.requested
        mean = math.fsum(score / len(scored) for score in scored) if scored else None
        return cls(trial_id, round_number, plan.identity, plan.split, plan.requested, len(scored),
                   complete, mean if complete else None, mean, tuple(sorted(ids)), cost,
                   pricing, dict(Counter(o.status for o in latest.values())))

    @classmethod
    def from_dict(cls, value):
        """Validate the schema and finite authoritative score when reading disk."""
        value = dict(value)
        if value.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported summary schema')
        value['observation_ids'] = tuple(value['observation_ids'])
        result = cls(**value)
        if not isinstance(result.eligible, bool) or not isinstance(result.requested, int) or not isinstance(result.scored, int):
            raise TypeError('invalid summary coverage types')
        if result.round < 0 or result.scored < 0 or result.scored > result.requested:
            raise ValueError('invalid summary coverage counts')
        if result.eligible != (result.scored == result.requested and result.requested > 0):
            raise ValueError('inconsistent summary coverage')
        if result.eligible:
            finite_score(result.score)
        elif result.score is not None:
            raise ValueError('incomplete summary has a ranking score')
        if result.partial_score is not None:
            finite_score(result.partial_score)
        if result.eligible and result.partial_score != result.score:
            raise ValueError('summary diagnostic score disagrees with committed score')
        if result.pricing not in ('known', 'unknown', 'unobserved') or finite_score(result.known_cost) < 0:
            raise ValueError('invalid summary accounting')
        if tuple(sorted(set(result.observation_ids))) != result.observation_ids:
            raise ValueError('duplicate or unordered summary observations')
        return result

    def to_dict(self):
        """Return the committed summary payload."""
        value = asdict(self)
        value['observation_ids'] = list(self.observation_ids)
        return value

    def improves(self, parent, config):
        """Compare only complete validation evidence under the exact same plan."""
        if not self.eligible or not parent.eligible or self.split != 'validation' or parent.split != 'validation':
            return False
        if self.plan_id != parent.plan_id:
            raise ValueError('cannot compare different experiment plans')
        difference = self.score - parent.score if config.direction == 'maximize' else parent.score - self.score
        return difference > config.improvement_margin
