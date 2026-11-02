from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..core.logging import logger
from ..core.registry import register_model
from .model_configs import AtlasCloudConfig
from .model_utils import Cost, cost_manager
from .openai_model import OpenAILLM

ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"


@register_model(config_cls=AtlasCloudConfig, alias=["atlascloud"])
class AtlasCloudLLM(OpenAILLM):
    """Atlas Cloud client for its OpenAI-compatible chat-completions API."""

    def init_model(self):
        self._client = None
        self._async_client = None
        self._default_ignore_fields = [
            "llm_type",
            "atlascloud_key",
            "output_response",
        ]

    def _init_client(self, config: AtlasCloudConfig):
        return OpenAI(api_key=config.atlascloud_key, base_url=ATLASCLOUD_BASE_URL)

    def _init_async_client(self, config: AtlasCloudConfig):
        return AsyncOpenAI(api_key=config.atlascloud_key, base_url=ATLASCLOUD_BASE_URL)

    def _update_cost(self, response: ChatCompletion | ChatCompletionChunk):
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(
                f"[AtlasCloudLLM] usage is missing from response "
                f"(id={getattr(response, 'id', '?')}); tokens will not be recorded."
            )
            return
        cost_manager.update_cost(
            cost=Cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost=0.0,
            ),
            model=self.config.model,
        )
