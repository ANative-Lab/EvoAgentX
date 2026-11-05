"""Independent review regressions for native accounting and durable recovery."""
import asyncio
from types import SimpleNamespace

import pytest

from evoagentx.models.model_utils import cost_manager
from evoagentx.models.openrouter_model import OpenRouterLLM
from evoagentx.optimizers.aflow import (
    AdmissionBudget,
    EvaluationPlan,
    ReliableTrialConfig,
    TrialStore,
)
from evoagentx.optimizers.aflow.evidence import CostLedger


def budget_store(tmp_path):
    config = ReliableTrialConfig(max_known_cost=1, example_cost_bound=.8, proposal_cost_bound=.1)
    return TrialStore(tmp_path, EvaluationPlan.create('local', [{'id': 'x'}], {}, 'v1', config))


def priced_response(receiver):
    OpenRouterLLM._update_cost(receiver, SimpleNamespace(id='priced', usage=SimpleNamespace(
        prompt_tokens=1, completion_tokens=1, cost=.1)))


@pytest.mark.parametrize('provider', ['openrouter', 'openai', 'litellm', 'aliyun', 'novita', 'siliconflow'])
def test_mixed_native_missing_usage_retains_whole_operation_bound(tmp_path, provider):
    from evoagentx.models.aliyun_model import AliyunLLM
    from evoagentx.models.litellm_model import LiteLLM
    from evoagentx.models.novita_model import NovitaLLM
    from evoagentx.models.openai_model import OpenAILLM
    from evoagentx.models.siliconflow_model import SiliconFlowLLM
    cls = {'openrouter': OpenRouterLLM, 'openai': OpenAILLM, 'litellm': LiteLLM,
           'aliyun': AliyunLLM, 'novita': NovitaLLM, 'siliconflow': SiliconFlowLLM}[provider]
    receiver = SimpleNamespace(config=SimpleNamespace(model='native-mixed-accounting'))
    with budget_store(tmp_path) as store:
        budget = AdmissionBudget(store, store.plan.config)
        budget.admit('example', 'first')
        ledger = CostLedger()
        before = cost_manager.get_total_cost()
        with cost_manager.observe(ledger.observe):
            priced_response(receiver)
            cls._update_cost(receiver, SimpleNamespace(id='missing', usage=None))
        snapshot = ledger.close()
        assert snapshot['pricing'] == 'unknown'
        assert snapshot['known_cost'] == pytest.approx(.1)
        assert cost_manager.get_total_cost() - before == pytest.approx(.1)
        budget.finish('first', snapshot['known_cost'], snapshot['pricing'] == 'known')
        assert budget.report()['reserved'] == pytest.approx(.7)
        assert budget.report()['pricing_uncertain']
        with pytest.raises(RuntimeError, match='cost'):
            budget.admit('example', 'second')


@pytest.mark.parametrize('provider', ['openrouter', 'openai'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_mixed_native_stream_without_usage_is_unknown(tmp_path, provider, asynchronous):
    from evoagentx.models.openai_model import OpenAILLM
    cls = OpenRouterLLM if provider == 'openrouter' else OpenAILLM
    receiver = object.__new__(cls)
    receiver.config = SimpleNamespace(model='native-stream-missing')
    chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content='answer', tool_calls=None))])]
    ledger = CostLedger()
    async def stream():
        for chunk in chunks:
            yield chunk
    async def consume():
        return await receiver.get_stream_output_async(stream(), output_response=False)
    with cost_manager.observe(ledger.observe):
        priced_response(receiver)
        if asynchronous:
            assert asyncio.run(consume()) == 'answer'
        else:
            assert receiver.get_stream_output(iter(chunks), output_response=False) == 'answer'
    assert ledger.close()['pricing'] == 'unknown'


@pytest.mark.parametrize('provider', ['openrouter', 'openai', 'litellm'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_mixed_native_failed_call_keeps_prior_price_incomplete(provider, asynchronous, monkeypatch):
    from evoagentx.models import litellm_model
    from evoagentx.models.litellm_model import LiteLLM
    from evoagentx.models.openai_model import OpenAILLM
    cls = {'openrouter': OpenRouterLLM, 'openai': OpenAILLM, 'litellm': LiteLLM}[provider]
    def failed(**kwargs):
        raise ConnectionError('scripted provider boundary; completion usage unavailable')
    async def failed_async(**kwargs):
        raise ConnectionError('scripted provider boundary; completion usage unavailable')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failed)))
    async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failed_async)))
    receiver = SimpleNamespace(config=SimpleNamespace(model='native-failed-call', stream=False, output_response=False),
        ensure_client=lambda: client, ensure_async_client=lambda: async_client,
        get_completion_params=lambda **kwargs: {}, prepare_request=lambda messages, params: (messages, params),
        _apply_provider_params=lambda params: params)
    monkeypatch.setattr(litellm_model, 'completion', failed)
    monkeypatch.setattr(litellm_model, 'acompletion', failed_async)
    method = cls.single_generate_async if asynchronous else cls.single_generate
    # Avoid transport retries/waits; this tests the native accounting boundary of
    # one failed provider attempt, not Tenacity's implementation.
    method = getattr(method, '__wrapped__', method)
    ledger = CostLedger()
    with cost_manager.observe(ledger.observe):
        priced_response(receiver)
        with pytest.raises(RuntimeError, match='scripted provider boundary'):
            if asynchronous:
                asyncio.run(method(receiver, [{'role': 'user', 'content': 'local'}]))
            else:
                method(receiver, [{'role': 'user', 'content': 'local'}])
    snapshot = ledger.close()
    assert snapshot['known_cost'] == pytest.approx(.1) and snapshot['pricing'] == 'unknown'


@pytest.mark.parametrize('provider', ['openrouter', 'openai', 'litellm'])
def test_native_cancelled_call_marks_unknown_and_preserves_cancellation(provider, monkeypatch):
    from evoagentx.models import litellm_model
    from evoagentx.models.litellm_model import LiteLLM
    from evoagentx.models.openai_model import OpenAILLM
    cls = {'openrouter': OpenRouterLLM, 'openai': OpenAILLM, 'litellm': LiteLLM}[provider]
    async def cancelled(**kwargs):
        raise asyncio.CancelledError
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=cancelled)))
    receiver = SimpleNamespace(config=SimpleNamespace(model='native-cancelled', stream=False, output_response=False),
        ensure_async_client=lambda: client, get_completion_params=lambda **kwargs: {},
        prepare_request=lambda messages, params: (messages, params), _apply_provider_params=lambda params: params)
    monkeypatch.setattr(litellm_model, 'acompletion', cancelled)
    method = getattr(cls.single_generate_async, '__wrapped__', cls.single_generate_async)
    ledger = CostLedger()
    with cost_manager.observe(ledger.observe):
        priced_response(receiver)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(method(receiver, [{'role': 'user', 'content': 'local'}]))
    assert ledger.close()['pricing'] == 'unknown'
