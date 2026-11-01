import pytest

from evoagentx.core.registry import MODEL_REGISTRY
from evoagentx.models import LLMOutputParser
from evoagentx.models.model_configs import RuncrateConfig
from evoagentx.models.model_utils import cost_manager
from evoagentx.models.runcrate_model import RUNCRATE_BASE_URL, RuncrateLLM

from tests.src.models.mock_response import (
    mock_openai_stream_completions_create,
    mock_openai_tool_call_create,
    mock_async_openai_create,
)

RUNCRATE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SYNC_PATCH = "openai.resources.chat.completions.Completions.create"
ASYNC_PATCH = "openai.resources.chat.completions.AsyncCompletions.create"

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(autouse=True)
def reset_cost_manager():
    cost_manager.input_tokens.clear()
    cost_manager.output_tokens.clear()
    cost_manager.total_tokens.clear()
    cost_manager.cost_per_model.clear()
    yield


def _make_llm(**kwargs) -> RuncrateLLM:
    kwargs.setdefault("runcrate_key", "mock_runcrate_key")
    config = RuncrateConfig(model=RUNCRATE_MODEL, output_response=False, **kwargs)
    return RuncrateLLM(config=config)


def _assert_tokens_recorded(model: str = RUNCRATE_MODEL):
    assert cost_manager.total_tokens[model] > 0, "No tokens recorded"
    assert cost_manager.cost_per_model[model] == 0.0


def test_registry_alias():
    assert MODEL_REGISTRY.get_model("RuncrateLLM") is RuncrateLLM
    assert MODEL_REGISTRY.get_model("runcrate") is RuncrateLLM


def test_client_uses_runcrate_base_url():
    client = _make_llm().ensure_client()
    assert str(client.base_url).rstrip("/") == RUNCRATE_BASE_URL
    assert client.api_key == "mock_runcrate_key"


def test_runcrate_key_not_sent_as_completion_param():
    params = _make_llm(temperature=0.2).get_completion_params(messages=[])
    assert "runcrate_key" not in params
    assert params["temperature"] == 0.2


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("RUNCRATE_API_KEY", "env_runcrate_key")
    llm = _make_llm(runcrate_key=None)
    assert llm.ensure_client().api_key == "env_runcrate_key"
    assert llm.ensure_async_client().api_key == "env_runcrate_key"


def test_sync_non_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_stream_completions_create)
    out = _make_llm(stream=False).generate(prompt="What is the capital of China?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Beijing"
    _assert_tokens_recorded()


def test_sync_stream(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_stream_completions_create)
    out = _make_llm(stream=True).generate(prompt="What is the capital of China?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_tokens_recorded()


def test_sync_tool_call(mocker):
    mocker.patch(SYNC_PATCH, mock_openai_tool_call_create)
    llm = _make_llm(stream=False, tools=[GET_WEATHER_TOOL], tool_choice="auto")
    out = llm.generate(prompt="What is the weather in Tokyo?")
    assert "<tool_call>" in out.content
    assert "get_weather" in out.content
    _assert_tokens_recorded()


async def test_async_stream(mocker):
    mocker.patch(ASYNC_PATCH, mock_async_openai_create)
    out = await _make_llm(stream=True).async_generate(prompt="What is the capital of China?")
    assert isinstance(out, LLMOutputParser)
    assert out.content == "Paris"
    _assert_tokens_recorded()
