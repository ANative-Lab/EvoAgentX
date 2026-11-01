import os
from typing import Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .openai_model import OpenAILLM
from .model_configs import RuncrateConfig
from ..core.logging import logger
from ..core.registry import register_model
from .model_utils import Cost, cost_manager

# Runcrate exposes an OpenAI-compatible API at this base URL.
RUNCRATE_BASE_URL = "https://api.runcrate.ai/v1"


@register_model(config_cls=RuncrateConfig, alias=["runcrate"])
class RuncrateLLM(OpenAILLM):
    """Runcrate LLM client.

    Runcrate serves open-weight models through the OpenAI chat-completions
    protocol, so this reuses all of ``OpenAILLM``'s generation/streaming/tool-call
    logic and only overrides client construction and cost handling.

    If ``runcrate_key`` is not set in the config, the ``RUNCRATE_API_KEY``
    environment variable is used.

    LiteLLM has no pricing data for Runcrate-hosted models, so the dollar cost
    cannot be computed or approximated. Token counts are still tracked; the
    per-model cost is recorded as 0. A warning is emitted once at init time.
    """

    def init_model(self):
        self._client = None
        self._async_client = None
        # parameters in RuncrateConfig that are not Runcrate models' input parameters
        self._default_ignore_fields = ["llm_type", "runcrate_key", "output_response"]
        logger.warning(
            "[RuncrateLLM] LiteLLM has no pricing data for Runcrate models, so dollar "
            "cost cannot be computed. Token usage will be tracked, but cost will "
            "be recorded as 0."
        )

    @staticmethod
    def _api_key(config: RuncrateConfig) -> Optional[str]:
        return config.runcrate_key or os.getenv("RUNCRATE_API_KEY")

    def _init_client(self, config: RuncrateConfig):
        return OpenAI(api_key=self._api_key(config), base_url=RUNCRATE_BASE_URL)

    def _init_async_client(self, config: RuncrateConfig):
        return AsyncOpenAI(api_key=self._api_key(config), base_url=RUNCRATE_BASE_URL)

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        # Override OpenAILLM's LiteLLM-based cost computation: only record token
        # counts and leave cost at 0 (see class docstring).
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(
                f"[RuncrateLLM] usage is None in response (id={getattr(response, 'id', '?')}); "
                "tokens will not be recorded."
            )
            return
        cost = Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=0.0,
        )
        cost_manager.update_cost(cost=cost, model=self.config.model)
