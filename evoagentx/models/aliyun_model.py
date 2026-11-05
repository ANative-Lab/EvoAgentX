from openai import AsyncOpenAI, OpenAI
from openai.types.completion_usage import CompletionUsage
from litellm import cost_per_token
from litellm.types.utils import Usage

from ..core.logging import logger
from ..core.registry import register_model
from .openai_model import OpenAILLM
from .model_configs import AliyunLLMConfig
from .model_utils import Cost, get_openai_model_cost


@register_model(config_cls=AliyunLLMConfig, alias=["aliyun_llm"])
class AliyunLLM(OpenAILLM):
    """Aliyun Bailian (DashScope) LLM client.

    Bailian exposes an OpenAI-compatible endpoint (``.../compatible-mode/v1``), so this
    reuses all of ``OpenAILLM``'s generation/streaming/tool-call logic and only overrides
    client construction and cost handling.

    Two things differ from plain OpenAI:

    1. Besides the API key (``aliyun_api_key``, i.e. the ``DASHSCOPE_API_KEY``), a
       ``aliyun_base_url`` is required because the endpoint is workspace-specific (the
       URL embeds the user's WorkspaceId).
    2. The compatible-mode responses carry standard ``usage`` token counts but no
       ``usage.cost``. Dollar cost is recovered through LiteLLM, whose pricing table
       keys DashScope models under the ``dashscope/`` prefix (e.g. ``dashscope/qwen-plus``).
       If LiteLLM has no pricing for the configured model, tokens are still tracked and
       cost is recorded as 0 (a warning is emitted once at init time).
    """

    def init_model(self):
        config: AliyunLLMConfig = self.config
        if not config.aliyun_api_key:
            raise ValueError("Aliyun API key is required. You should set `aliyun_api_key` in AliyunLLMConfig")
        if not config.aliyun_base_url:
            raise ValueError(
                "Aliyun base URL is required. You should set `aliyun_base_url` in AliyunLLMConfig "
                "(it is workspace-specific, e.g. "
                "'https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')."
            )
        self._client = None
        self._async_client = None
        # parameters in AliyunLLMConfig that are not OpenAI-compatible request params
        self._default_ignore_fields = [
            "llm_type", "output_response", "aliyun_api_key", "aliyun_base_url",
        ]
        # LiteLLM keys DashScope models under the "dashscope/" prefix for pricing lookups.
        self._litellm_model = self._to_litellm_model(config.model)
        self._has_pricing = self._litellm_model in get_openai_model_cost()
        if self._has_pricing:
            logger.warning(
                f"[AliyunLLM] Aliyun does not report usage.cost, so dollar cost for '{config.model}' "
                f"is estimated from LiteLLM's public pricing for '{self._litellm_model}'. The figure "
                "may differ from Aliyun's official billing."
            )
        else:
            logger.warning(
                f"[AliyunLLM] LiteLLM has no pricing for '{self._litellm_model}', so dollar cost "
                "cannot be computed. Token usage will be tracked, but cost will be recorded as 0."
            )

    @staticmethod
    def _to_litellm_model(model: str) -> str:
        return model if model.startswith("dashscope/") else f"dashscope/{model}"

    def _init_client(self, config: AliyunLLMConfig):
        return OpenAI(api_key=config.aliyun_api_key, base_url=config.aliyun_base_url)

    def _init_async_client(self, config: AliyunLLMConfig):
        return AsyncOpenAI(api_key=config.aliyun_api_key, base_url=config.aliyun_base_url)

    def _compute_cost(self, usage: CompletionUsage) -> Cost:
        # Aliyun's compatible-mode usage has no `cost` field; price it via LiteLLM using
        # the "dashscope/"-prefixed model name. Fall back to token-only when unpriced.
        if not self._has_pricing:
            return Cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost=0.0,
                pricing_known=False,
            )
        usage_object = usage if isinstance(usage, Usage) else Usage(**usage.model_dump())
        input_cost, output_cost = cost_per_token(model=self._litellm_model, usage_object=usage_object)
        return Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
        )
