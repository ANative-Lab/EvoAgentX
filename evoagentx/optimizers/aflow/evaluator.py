"""Bounded detailed evaluation through the existing AFlow benchmark contract."""

import asyncio
import time

from ...models.model_utils import cost_manager
from .evidence import CostLedger
from .types import Observation, TrialSummary, finite_score


async def join_owned(tasks):
    """Drain owned async work despite repeated cancellation of its owner.

    This joins asyncio tasks only. Cancellation of a to_thread awaiter does not
    prove that its thread or a remote operation stopped; that evidence stays unknown.
    """
    waiter = asyncio.gather(*tasks, return_exceptions=True)
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            continue
    return waiter.result()


async def owned_call(operation, timeout):
    """Run one operation and distinguish completed errors from ambiguous timeout."""
    task = asyncio.create_task(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            task.cancel()
            await join_owned([task])
            return 'unknown', None
        try:
            return 'completed', task.result()
        except asyncio.CancelledError:
            return 'unknown', None
        except Exception as exc:  # noqa: BLE001 - user/provider/export boundary is typed as evidence
            return 'error', exc
    except asyncio.CancelledError:
        task.cancel()
        await join_owned([task])
        raise


class DetailedEvaluator:
    """Capture exact typed per-example evidence without changing Benchmark APIs."""

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _timeout(plan, budget):
        """Use the shorter cooperative attempt/deadline cancellation request."""
        values = [v for v in (plan.config.attempt_timeout, budget.remaining_time()) if v is not None]
        return min(values) if values else None

    @staticmethod
    def _retryable(error, config):
        """Classify explicit configured exception types after the task has terminated."""
        names = {f'{cls.__module__}.{cls.__qualname__}' for cls in type(error).__mro__}
        return bool(names.intersection(config.retryable_errors))

    async def _attempt(self, benchmark, graph, example, identity, replicate, attempt,
                       plan, trial_id, store, budget):
        """Own admission, intent checkpoint, call, cost snapshot and result commit."""
        obs_id = Observation.make(trial_id, identity, replicate, attempt, 'unknown').observation_id
        budget.admit('example', obs_id)
        store.start_attempt(trial_id, identity, replicate, attempt)
        ledger = CostLedger()
        started = time.monotonic()
        try:
            with cost_manager.observe(ledger.observe):
                state, value = await owned_call(lambda: benchmark.async_evaluate(graph, example), self._timeout(plan, budget))
        except asyncio.CancelledError:
            costs = ledger.close()
            obs = Observation.make(trial_id, identity, replicate, attempt, 'cancelled',
                                   duration=time.monotonic() - started, **costs)
            store.record(obs)
            store.mark_stopping()
            budget.finish(obs_id, costs['known_cost'], costs['pricing'] == 'known', terminated=False)
            raise
        costs = ledger.close()
        duration = time.monotonic() - started
        if state == 'unknown':
            obs = Observation.make(trial_id, identity, replicate, attempt, 'unknown',
                                   error_type='AmbiguousTimeoutOrCancellation', duration=duration, **costs)
            store.record(obs)
            store.mark_stopping()
            budget.finish(obs_id, costs['known_cost'], costs['pricing'] == 'known', terminated=False)
            return obs
        if state == 'error':
            status = 'retryable' if self._retryable(value, plan.config) else 'inconclusive'
            error_type = f'{type(value).__module__}.{type(value).__qualname__}'
            obs = Observation.make(trial_id, identity, replicate, attempt, status,
                                   error_type=error_type, duration=duration, **costs)
        else:
            try:
                score = finite_score(value)
            except ValueError:
                obs = Observation.make(trial_id, identity, replicate, attempt, 'inconclusive',
                                       error_type='InvalidMetric', duration=duration, **costs)
            else:
                obs = Observation.scored(trial_id, identity, replicate, attempt, score,
                                         duration=duration, **costs)
        store.record(obs)
        budget.finish(obs_id, costs['known_cost'], costs['pricing'] == 'known')
        return obs

    async def evaluate(self, benchmark, graph, data, plan, trial_id, store, budget):
        """Resume completed coordinates and commit one complete or incomplete summary."""
        plan.verify_data(data)
        # A durable completed outcome establishes completion even if the process
        # died before releasing its conservative cost reservation.
        for observation in store.observations(trial_id):
            if observation.status not in ('unknown', 'cancelled'):
                budget.reconcile(observation.observation_id, observation.known_cost, observation.pricing == 'known')
        existing = store.summary(trial_id)
        if existing is not None:
            return existing
        store.assert_resolved()
        configured = graph(name=benchmark.name, llm_config=self.llm.config if self.llm is not None else None,
                           benchmark=benchmark)
        latest = {}
        for obs in store.observations(trial_id):
            key = (obs.example_key, obs.replicate)
            if key not in latest or obs.attempt > latest[key].attempt:
                latest[key] = obs
        intents = {}
        for operation in store.example_attempts(trial_id):
            key = (operation['example_key'], operation['replicate'])
            if key not in intents or operation['attempt'] > intents[key]['attempt']:
                intents[key] = operation
            if operation.get('resolution') in ('rerun', 'abandon'):
                # Resolution confirms termination, not actual usage. An outcome
                # absent after a crash cannot manufacture a known zero price.
                budget.reconcile(operation['attempt_id'], 0.0, False)
        coordinates = iter((identity, replicate) for replicate in range(plan.config.replicates)
                           for identity in plan.examples)
        stop_reason = []

        async def worker():
            for identity, replicate in coordinates:
                if stop_reason:
                    return
                previous = latest.get((identity.key, replicate))
                intent = intents.get((identity.key, replicate))
                if intent is not None and (previous is None or intent['attempt'] >= previous.attempt):
                    if intent.get('resolution') == 'abandon':
                        continue
                    if (previous is None or intent['attempt'] > previous.attempt) and intent.get('resolution') != 'rerun':
                        raise RuntimeError('durable example intent has no completed outcome or explicit rerun')
                if previous is not None:
                    if previous.status == 'scored' or previous.status == 'inconclusive':
                        continue
                    if previous.status in ('unknown', 'cancelled'):
                        resolution = store.operation(previous.observation_id)
                        if not resolution or resolution.get('resolution') != 'rerun':
                            continue
                    start = previous.attempt + 1
                else:
                    start = 1
                if intent is not None:
                    start = max(start, intent['attempt'] + 1)
                for attempt in range(start, plan.config.attempts_per_example + 1):
                    try:
                        observation = await self._attempt(benchmark, configured, data[identity.index], identity,
                                                          replicate, attempt, plan, trial_id, store, budget)
                    except RuntimeError as exc:
                        stop_reason.append(str(exc))
                        return
                    if observation.status in ('unknown', 'cancelled'):
                        stop_reason.append('unknown external execution requires explicit resolution')
                        return
                    if observation.status != 'retryable':
                        break

        tasks = [asyncio.create_task(worker()) for _ in range(min(plan.config.concurrency, plan.requested))]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await join_owned(tasks)
            raise
        store.assert_resolved()
        trial = store.trial(trial_id)
        summary = TrialSummary.build(plan, trial_id, trial['round'], store.observations(trial_id))
        store.commit_summary(summary)
        if stop_reason:
            store.write_state('last_evaluation_stop', {'trial_id': trial_id, 'reason': stop_reason[0]})
        return summary
