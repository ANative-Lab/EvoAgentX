"""AFlow-only staged evolution over committed, comparable trial evidence."""

import asyncio
import inspect
from dataclasses import replace
from pathlib import Path

from ...models.model_utils import cost_manager
from .artifacts import load_staged_graph, prepare_graph, publish_round
from .budget import AdmissionBudget
from .evaluator import DetailedEvaluator, owned_call
from .evidence import CostLedger, EvidenceView, EvolutionResult
from .store import TrialStore, atomic_json
from .types import SCHEMA_VERSION, EvaluationPlan, digest


def metric_identity(benchmark):
    """Fingerprint the native scorer and inherited benchmark implementation."""
    sources = []
    for cls in type(benchmark).__mro__:
        if cls is object:
            continue
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            raise ValueError('custom benchmark requires an explicit stable metric_id')
        sources.append((f'{cls.__module__}.{cls.__qualname__}', source))
    return digest(sources)


class ReliableAFlowRunner:
    """Own one reliable experiment; provider/model APIs remain unchanged."""

    def __init__(self, optimizer, benchmark):
        self.optimizer = optimizer
        self.benchmark = benchmark
        self.config = optimizer.reliable_trials
        self.root = Path(optimizer.root_path)
        self.path = self.root / 'reliable' / self.config.experiment_id
        self.data = list(benchmark.get_dev_data() or [])
        self.plan = self._plan(self.data, self.config, 'validation')
        self.export_error = None

    def _plan(self, data, config, split):
        """Include model roles and AFlow behavior alongside exact dataset identity."""
        optimizer = self.optimizer
        models = {'executor': optimizer.executor_llm.config, 'optimizer': optimizer.optimizer_llm.config}
        models = {role: config.model_dump(mode='json') for role, config in models.items()}
        models['aflow_policy'] = {'question_type': optimizer.question_type, 'operators': optimizer.operators,
                                  'sample': optimizer.sample, 'max_rounds': optimizer.max_rounds,
                                  'check_convergence': optimizer.check_convergence}
        # Input seed changes must invalidate resume even after baseline completion.
        seed_graph = (Path(optimizer.graph_path) / 'graph.py').read_text(encoding='utf-8')
        seed_prompt = (Path(optimizer.graph_path) / 'prompt.py').read_text(encoding='utf-8')
        models['seed_digest'] = digest({'graph': seed_graph, 'prompt': seed_prompt})
        return EvaluationPlan.create(self.benchmark.name, data, models,
                                     config.metric_id or metric_identity(self.benchmark), config, split)

    def _save(self):
        """Persist the stage cursor before further external work."""
        self.store.write_state('evolution', self.state)

    def _initial_state(self):
        """Build a stable baseline/proposal cursor independent of retry count."""
        return {'schema_version': SCHEMA_VERSION, 'phase': 'baseline', 'round': 0,
                'candidate': None, 'parent': None, 'duplicates': 0,
                'modifications': [], 'artifact_digests': [], 'pending_proposal': None,
                'terminal_status': None, 'terminal_reason': None}

    def _result(self, status, reason):
        """Report explicit terminal state using only a committed eligible parent."""
        best = self.view.best()
        return EvolutionResult(status, reason, best.trial_id if best else None,
                               best.round if best else None, best.score if best else None,
                               len(self.store.summaries()), self.budget.report(), self.export_error)

    def _terminal(self, status, reason):
        """Checkpoint terminal state so a repeated optimize call does no paid work."""
        self.state.update(phase='finished', terminal_status=status, terminal_reason=reason)
        self._save()
        self._export()
        return self._result(status, reason)

    def _export(self):
        """Repair compatibility files without undoing a committed summary on failure."""
        try:
            for summary in self.store.summaries():
                if summary.eligible:
                    publish_round(self.store, summary.trial_id, self.root)
            self.view.export(self.root)
            from .reporting import TrialReporter
            TrialReporter(self.store).export()
        except Exception as exc:  # noqa: BLE001 - user/provider/export boundary is typed as evidence
            self.export_error = f'{type(exc).__module__}.{type(exc).__qualname__}'
            self.store.write_state('export_status', {'status': 'failed', 'error_type': self.export_error})
        else:
            self.export_error = None
            self.store.write_state('export_status', {'status': 'complete'})

    async def _baseline(self):
        """Import seed artifacts but evaluate fresh coverage rather than old results."""
        trial_id = 'baseline'
        if not any(t['trial_id'] == trial_id for t in self.store.trials()):
            self.budget.admit('trial', 'trial_baseline')
            self.store.create_trial(trial_id, 0, 'baseline')
            self.budget.finish('trial_baseline', 0, True)
        graph = (Path(self.optimizer.graph_path) / 'graph.py').read_text(encoding='utf-8')
        prompt = (Path(self.optimizer.graph_path) / 'prompt.py').read_text(encoding='utf-8')
        graph, prompt = prepare_graph(graph, prompt)
        manifest = self.store.stage_artifacts(trial_id, graph, prompt)
        if manifest['digest'] not in self.state['artifact_digests']:
            self.state['artifact_digests'].append(manifest['digest'])
            self._save()
        summary = await self._evaluate(trial_id)
        if not summary.eligible:
            return self._terminal('incomplete', 'baseline does not cover the complete validation plan')
        self.state.update(phase='proposal', parent=trial_id, round=1)
        self._save()
        self._export()
        return None

    async def _evaluate(self, trial_id):
        """Load exact staged code and use native detailed benchmark evaluation."""
        existing = self.store.summary(trial_id)
        if existing is not None:
            return existing
        graph = load_staged_graph(self.store, trial_id)
        return await DetailedEvaluator(self.optimizer.executor_llm).evaluate(
            self.benchmark, graph, self.data, self.plan, trial_id, self.store, self.budget)

    def _proposal_prompt(self, parent):
        """Reuse the existing AFlow proposal prompt with measured feedback."""
        graph, prompt, _ = self.store.read_artifacts(parent.trial_id)
        extracted = self.optimizer.graph_utils.extract_solve_graph(graph)
        if not extracted:
            raise ValueError('parent graph must define class Workflow')
        descriptions = self.optimizer.graph_utils.load_operators_description(self.optimizer.operators,
                                                                            self.optimizer.optimizer_llm)
        return self.optimizer.graph_utils.create_graph_optimize_prompt(
            self.view.prompt_feedback(parent.trial_id), parent.score, extracted[0], prompt,
            descriptions, self.optimizer.question_type, '')

    def _parse_proposal(self, content, parent):
        """Preserve native XML parsing/fallback while requiring executable source."""
        from ..aflow_optimizer import GraphOptimizeOutput
        graph, prompt, _ = self.store.read_artifacts(parent.trial_id)
        extracted = self.optimizer.graph_utils.extract_solve_graph(graph)
        try:
            parsed = GraphOptimizeOutput.parse(content, parse_mode='xml').get_structured_data()
        except Exception:  # noqa: BLE001 - retain the existing XML parser fallback
            parsed = self.optimizer._parse_optimizer_llm_output(content, extracted[0], prompt)
        if not isinstance(parsed.get('modification'), str) or not parsed['modification'].strip():
            raise ValueError('proposal requires a modification description')
        graph, prompt = prepare_graph(parsed['graph'], parsed['prompt'], generated=True)
        return {'graph': graph, 'prompt': prompt, 'modification': parsed['modification']}

    async def _request_proposal(self, parent):
        """Count proposal calls separately; an outage never restarts the baseline."""
        number = self.budget.state['counts']['proposal'] + 1
        operation_id = f'proposal_{number}'
        prompt = self._proposal_prompt(parent)
        self.budget.admit('proposal', operation_id)
        self.store.begin_operation(operation_id, 'proposal', parent.trial_id)
        ledger = CostLedger()
        timeouts = [v for v in (self.config.proposal_timeout, self.budget.remaining_time()) if v is not None]
        try:
            with cost_manager.observe(ledger.observe):
                state, value = await owned_call(lambda: self.optimizer.optimizer_llm.async_generate(
                    prompt=prompt, parse_mode='str'), min(timeouts) if timeouts else None)
        except asyncio.CancelledError:
            costs = ledger.close()
            self.store.finish_operation(operation_id, 'cancelled')
            self.store.mark_stopping()
            self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known', False)
            raise
        costs = ledger.close()
        if state == 'unknown':
            self.store.finish_operation(operation_id, 'unknown')
            self.store.mark_stopping()
            self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known', False)
            raise RuntimeError('unknown proposal completion requires explicit resolution')
        if state == 'error':
            retryable = DetailedEvaluator._retryable(value, self.config)
            self.store.finish_operation(operation_id, 'retryable' if retryable else 'inconclusive',
                                        error_type=f'{type(value).__module__}.{type(value).__qualname__}')
            self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known')
            if retryable:
                return None
            raise ValueError('proposal infrastructure failure is inconclusive')
        try:
            response = self._parse_proposal(value.content, parent)
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            self.store.finish_operation(operation_id, 'inconclusive', error_type=type(exc).__name__)
            self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known')
            return None
        # Durable parsed source proves response completion if a crash precedes
        # operation/budget finalization. It is never reconstructed from old rows.
        self.state['pending_proposal'] = {'operation_id': operation_id, 'response': response,
                                          'parent_id': parent.trial_id, 'costs': costs}
        self._save()
        self.store.finish_operation(operation_id, 'completed')
        self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known')
        return response

    def _recover_pending_proposal(self):
        """Finalize checkpointed provider output without repeating its external call."""
        pending = self.state['pending_proposal']
        if pending is None:
            return None
        operation_id = pending['operation_id']
        operation = self.store.operation(operation_id)
        if operation is None:
            raise ValueError('pending proposal has no durable attempt intent')
        if operation['status'] == 'started':
            self.store.finish_operation(operation_id, 'completed')
        costs = pending['costs']
        if operation_id in self.budget.state['reservations']:
            self.budget.finish(operation_id, costs['known_cost'], costs['pricing'] == 'known')
        return pending['response']

    async def _proposal(self):
        """Bound both paid requests and duplicate candidates under a stable parent."""
        parent = self.view.best()
        if parent is None:
            return self._terminal('incomplete', 'no eligible validation parent')
        response = self._recover_pending_proposal()
        self.store.assert_resolved()
        if response is None:
            response = await self._request_proposal(parent)
            if response is None:
                return None
        normalized = ' '.join(response['modification'].casefold().split())
        artifact_digest = digest({'graph': response['graph'], 'prompt': response['prompt']})
        if normalized in self.state['modifications'] or artifact_digest in self.state['artifact_digests']:
            self.state['duplicates'] += 1
            self.state['pending_proposal'] = None
            self._save()
            if self.state['duplicates'] >= self.config.duplicate_limit:
                return self._terminal('exhausted', 'duplicate candidate budget exhausted')
            return None
        trial_id = f"candidate_{self.state['round']}"
        # If a crash occurs after trial admission but before creating the trial,
        # that reservation remains conservative; automatic duplicate admission
        # is rejected rather than silently resetting an attempt budget.
        if not any(t['trial_id'] == trial_id for t in self.store.trials()):
            self.budget.admit('trial', f'trial_{trial_id}')
            self.store.create_trial(trial_id, self.state['round'], 'candidate', parent.trial_id,
                                    response['modification'])
            self.budget.finish(f'trial_{trial_id}', 0, True)
        self.store.stage_artifacts(trial_id, response['graph'], response['prompt'])
        self.state['modifications'].append(normalized)
        self.state['artifact_digests'].append(artifact_digest)
        self.state.update(phase='evaluation', candidate=trial_id, parent=parent.trial_id,
                          pending_proposal=None, duplicates=0)
        self._save()
        return None

    async def run(self):
        """Execute or resume stable stages and return a truthful terminal result."""
        with TrialStore(self.path, self.plan) as self.store:
            self.budget = AdmissionBudget(self.store, self.config)
            self.view = EvidenceView(self.store)
            self.state = self.store.read_state('evolution', self._initial_state())
            if self.state.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('unsupported evolution state schema')
            self._recover_pending_proposal()
            self.store.assert_resolved()
            if self.state['phase'] == 'finished':
                self._export()
                return self._result(self.state['terminal_status'], self.state['terminal_reason'])
            try:
                while True:
                    self.budget.check_deadline()
                    if self.state['phase'] == 'baseline':
                        result = await self._baseline()
                        if result is not None:
                            return result
                    elif self.state['phase'] == 'proposal':
                        if self.state['round'] > self.optimizer.max_rounds:
                            return self._terminal('completed', 'requested optimization rounds completed')
                        result = await self._proposal()
                        if result is not None:
                            return result
                    elif self.state['phase'] == 'evaluation':
                        summary = await self._evaluate(self.state['candidate'])
                        self._export()
                        if not summary.eligible:
                            return self._terminal('incomplete', 'candidate has incomplete validation evidence')
                        self.optimizer.round = summary.round
                        self.optimizer.avg_score = summary.score
                        self.state.update(phase='proposal', candidate=None, round=self.state['round'] + 1)
                        self._save()
                        if self.optimizer.check_convergence and self.view.converged():
                            return self._terminal('completed', 'committed validation trials converged')
                    else:
                        raise ValueError('unknown evolution stage')
            except RuntimeError as exc:
                if self.store.unknown_attempts():
                    # Preserve stage for explicit operator resolution. No late
                    # result callback is allowed to mutate this owner state.
                    return self._result('incomplete', 'unknown external execution requires explicit resolution')
                return self._terminal('exhausted', str(exc))
            except (ValueError, ImportError, SyntaxError, AttributeError, TypeError) as exc:
                return self._terminal('incomplete', f'workflow/proposal infrastructure: {type(exc).__name__}')

    async def test(self, rounds):
        """Use a distinct split/store; test labels never enter evolutionary evidence."""
        validation_plan = self.plan
        data = list(self.benchmark.get_test_data() or [])
        test_config = replace(self.config, experiment_id=self.config.experiment_id + '_test')
        plan = self._plan(data, test_config, 'test')
        results = []
        with TrialStore(self.path, validation_plan) as validation:
            validation.assert_resolved()
            best = EvidenceView(validation).best()
            if rounds is None:
                if best is None:
                    raise ValueError('no committed validation trial to test')
                rounds = [best.round]
            selected = {trial['round']: trial['trial_id'] for trial in validation.trials()
                        if validation.summary(trial['trial_id']) is not None}
            with TrialStore(self.root / 'reliable' / test_config.experiment_id, plan) as store:
                budget = AdmissionBudget(store, test_config)
                store.assert_resolved()
                for round_number in rounds:
                    if round_number not in selected:
                        raise ValueError('test round has no reliable validation provenance')
                    source_id = selected[round_number]
                    graph, prompt, _ = validation.read_artifacts(source_id)
                    trial_id = f'test_{round_number}'
                    if not any(t['trial_id'] == trial_id for t in store.trials()):
                        budget.admit('trial', f'trial_{trial_id}')
                        store.create_trial(trial_id, round_number, 'test', source_id)
                        budget.finish(f'trial_{trial_id}', 0, True)
                    store.stage_artifacts(trial_id, graph, prompt)
                    summary = await DetailedEvaluator(self.optimizer.executor_llm).evaluate(
                        self.benchmark, load_staged_graph(store, trial_id), data, plan, trial_id, store, budget)
                    results.append(summary)
                atomic_json(self.root / 'test_results.json',
                            [{'round': s.round, 'score': s.score, 'eligible': s.eligible,
                              'trial_id': s.trial_id, 'split': 'test', 'requested': s.requested,
                              'scored': s.scored, 'known_cost': s.known_cost} for s in results])
        return results
