from unittest.mock import MagicMock

from openai.types.completion_usage import CompletionUsage

from evoagentx.core.registry import MODEL_REGISTRY
from evoagentx.models.atlascloud_model import ATLASCLOUD_BASE_URL, AtlasCloudLLM
from evoagentx.models.model_configs import AtlasCloudConfig
from evoagentx.models.model_utils import cost_manager


def _config() -> AtlasCloudConfig:
    return AtlasCloudConfig(
        model="openai/gpt-5.6-luna",
        atlascloud_key="test-key",
        output_response=False,
    )


def test_atlascloud_model_is_registered():
    assert MODEL_REGISTRY.get_model("AtlasCloudLLM") is AtlasCloudLLM
    assert MODEL_REGISTRY.get_model("atlascloud") is AtlasCloudLLM
    assert MODEL_REGISTRY.get_model_config("atlascloud") is AtlasCloudConfig


def test_atlascloud_clients_use_the_provider_endpoint(mocker):
    sync_client = mocker.patch("evoagentx.models.atlascloud_model.OpenAI")
    async_client = mocker.patch("evoagentx.models.atlascloud_model.AsyncOpenAI")
    llm = AtlasCloudLLM(config=_config())

    llm._init_client(llm.config)
    llm._init_async_client(llm.config)

    sync_client.assert_called_once_with(
        api_key="test-key", base_url=ATLASCLOUD_BASE_URL
    )
    async_client.assert_called_once_with(
        api_key="test-key", base_url=ATLASCLOUD_BASE_URL
    )


def test_atlascloud_credentials_are_not_sent_as_completion_params():
    llm = AtlasCloudLLM(config=_config())

    assert llm.get_completion_params() == {"model": "openai/gpt-5.6-luna"}


def test_atlascloud_records_tokens_without_guessing_cost():
    model = "openai/gpt-5.6-luna"
    cost_manager.input_tokens.clear()
    cost_manager.output_tokens.clear()
    cost_manager.total_tokens.clear()
    cost_manager.cost_per_model.clear()
    response = MagicMock(
        id="atlas-test",
        usage=CompletionUsage(prompt_tokens=12, completion_tokens=5, total_tokens=17),
    )

    AtlasCloudLLM(config=_config())._update_cost(response)

    assert cost_manager.input_tokens[model] == 12
    assert cost_manager.output_tokens[model] == 5
    assert cost_manager.total_tokens[model] == 17
    assert cost_manager.cost_per_model[model] == 0.0
