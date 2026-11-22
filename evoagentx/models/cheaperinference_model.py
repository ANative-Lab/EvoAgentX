from typing import Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .openai_model import OpenAILLM
from .model_configs import CheaperInferenceConfig
from ..core.logging import logger
from ..core.registry import register_model
from .model_utils import Cost, cost_manager

# Cheaper Inference exposes an OpenAI-compatible API at this base URL.
CHEAPERINFERENCE_BASE_URL = "https://api.cheaperinference.com/v1"


@register_model(config_cls=CheaperInferenceConfig, alias=["cheaperinference"])
class CheaperInferenceLLM(OpenAILLM):
    """Cheaper Inference LLM client.

    Cheaper Inference speaks the OpenAI chat-completions protocol, so this reuses all of
    ``OpenAILLM``'s generation/streaming/tool-call logic and only overrides client
    construction and cost handling.

    LiteLLM has no pricing data for Cheaper Inference models, so the dollar cost
    cannot be computed or approximated. Token counts are still tracked; the
    per-model cost is recorded as 0. A warning is emitted once at init time.
    """

    def init_model(self):
        self._client = None
        self._async_client = None
        # parameters in CheaperInferenceConfig that are not model input parameters
        self._default_ignore_fields = ["llm_type", "cheaperinference_key", "output_response"]
        logger.warning(
            "[CheaperInferenceLLM] LiteLLM has no pricing data for Cheaper Inference models, so dollar "
            "cost cannot be computed. Token usage will be tracked, but cost will "
            "be recorded as 0."
        )

    def _init_client(self, config: CheaperInferenceConfig):
        return OpenAI(api_key=config.cheaperinference_key, base_url=CHEAPERINFERENCE_BASE_URL)

    def _init_async_client(self, config: CheaperInferenceConfig):
        return AsyncOpenAI(api_key=config.cheaperinference_key, base_url=CHEAPERINFERENCE_BASE_URL)

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        # Override OpenAILLM's LiteLLM-based cost computation: only record token
        # counts and leave cost at 0 (see class docstring).
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(
                f"[CheaperInferenceLLM] usage is None in response (id={getattr(response, 'id', '?')}); "
                "tokens will not be recorded."
            )
            return
        cost = Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=0.0,
        )
        cost_manager.update_cost(cost=cost, model=self.config.model)
