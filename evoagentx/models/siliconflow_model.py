from typing import Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .openai_model import OpenAILLM
from .model_configs import SiliconFlowConfig
from ..core.logging import logger
from ..core.registry import register_model
from .model_utils import Cost, cost_manager

# SiliconFlow exposes an OpenAI-compatible API. Use the international endpoint
# (api.siliconflow.com), not the mainland one (api.siliconflow.cn).
SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"


@register_model(config_cls=SiliconFlowConfig, alias=["siliconflow"])
class SiliconFlowLLM(OpenAILLM):
    """SiliconFlow LLM client.

    SiliconFlow speaks the OpenAI chat-completions protocol, so this reuses all of
    ``OpenAILLM``'s generation/streaming/tool-call logic and only overrides client
    construction and cost handling.

    Unlike OpenRouter, SiliconFlow does not return ``usage.cost`` in its responses,
    and LiteLLM has no pricing data for SiliconFlow-hosted models, so the dollar
    cost cannot be computed or approximated. Token counts are still tracked; the
    per-model cost is recorded as 0. A warning is emitted once at init time.
    """

    def init_model(self):
        self._client = None
        self._async_client = None
        # parameters in SiliconFlowConfig that are not SiliconFlow models' input parameters
        self._default_ignore_fields = ["llm_type", "siliconflow_key", "output_response"]
        logger.warning(
            "[SiliconFlowLLM] SiliconFlow does not report usage.cost and LiteLLM has no "
            "pricing data for SiliconFlow models, so dollar cost cannot be computed. "
            "Token usage will be tracked, but cost will be recorded as 0."
        )

    def _init_client(self, config: SiliconFlowConfig):
        return OpenAI(api_key=config.siliconflow_key, base_url=SILICONFLOW_BASE_URL)

    def _init_async_client(self, config: SiliconFlowConfig):
        return AsyncOpenAI(api_key=config.siliconflow_key, base_url=SILICONFLOW_BASE_URL)

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        # Override OpenAILLM's LiteLLM-based cost computation: only record token
        # counts and leave cost at 0 (see class docstring).
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(
                f"[SiliconFlowLLM] usage is None in response (id={getattr(response, 'id', '?')}); "
                "tokens will not be recorded."
            )
            return
        cost = Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=0.0,
            pricing_known=False,
        )
        cost_manager.update_cost(cost=cost, model=self.config.model)
