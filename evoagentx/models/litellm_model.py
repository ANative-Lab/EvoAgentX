import os
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from litellm import completion, acompletion
from typing import List
from ..core.logging import logger
from ..core.registry import register_model
from .model_configs import LiteLLMConfig
from .openai_model import OpenAILLM
from .model_utils import infer_litellm_company_from_model, Cost

@register_model(config_cls=LiteLLMConfig, alias=["litellm"])
class LiteLLM(OpenAILLM):

    def init_model(self):
        """
        Initialize the model based on the configuration.
        """
        # Check if llm_type is correct
        if self.config.llm_type != "LiteLLM":
            raise ValueError("llm_type must be 'LiteLLM'")

        # Set model and extract the company name
        self.model = self.config.model
        self.api_base = self.config.api_base  # save api_base
        self.api_key = self.config.api_key
        # company = self.model.split("/")[0] if "/" in self.model else "openai"
        company = infer_litellm_company_from_model(self.model)

        if self.config.is_local or company == "local":  # update support local model
            if not self.api_base:
                raise ValueError("api_base is required for local models in LiteLLMConfig")
            # local llm doesn't need API key
            litellm.api_base = self.api_base  # set litellm global api_base
            litellm.api_key = self.api_key
        else:
            # Set environment variables based on the company
            if company == "openai":
                if not self.config.openai_key:
                    raise ValueError("OpenAI API key is required for OpenAI models. You should set `openai_key` in LiteLLMConfig")
                os.environ["OPENAI_API_KEY"] = self.config.openai_key
            elif company == "azure":
                if not self.config.azure_key or not self.config.azure_endpoint:
                    raise ValueError("Azure OpenAI key and endpoint are required for Azure models. You should set `azure_key` and `azure_endpoint` in LiteLLMConfig")
                os.environ["AZURE_API_KEY"] = self.config.azure_key
                os.environ["AZURE_API_BASE"] = self.config.azure_endpoint
                if self.config.api_version:
                    os.environ["AZURE_API_VERSION"] = self.config.api_version
            elif company == "deepseek":
                if not self.config.deepseek_key:
                    raise ValueError("DeepSeek API key is required for DeepSeek models. You should set `deepseek_key` in LiteLLMConfig")
                os.environ["DEEPSEEK_API_KEY"] = self.config.deepseek_key
            elif company == "anthropic":
                if not self.config.anthropic_key:
                    raise ValueError("Anthropic API key is required for Anthropic models. You should set `anthropic_key` in LiteLLMConfig")
                os.environ["ANTHROPIC_API_KEY"] = self.config.anthropic_key
            elif company == "gemini":
                if not self.config.gemini_key:
                    raise ValueError("Gemini API key is required for Gemini models. You should set `gemini_key` in LiteLLMConfig")
                os.environ["GEMINI_API_KEY"] = self.config.gemini_key 
            elif company == "meta_llama":
                if not self.config.meta_llama_key:
                    raise ValueError("Meta Llama API key is required for Meta Llama models. You should set `meta_llama_key` in LiteLLMConfig")
                os.environ["LLAMA_API_KEY"] = self.config.meta_llama_key
            elif company == "openrouter":
                if not self.config.openrouter_key:
                    raise ValueError("OpenRouter API key is required for OpenRouter models. You should set `openrouter_key` in LiteLLMConfig. You can also set `openrouter_base` in LiteLLMConfig to use a custom base URL [optional]")
                os.environ["OPENROUTER_API_KEY"] = self.config.openrouter_key
                os.environ["OPENROUTER_API_BASE"] = self.config.openrouter_base # [optional]
            elif company == "perplexity":
                if not self.config.perplexity_key:
                    raise ValueError("Perplexity API key is required for Perplexity models. You should set `perplexity_key` in LiteLLMConfig")
                os.environ["PERPLEXITYAI_API_KEY"] = self.config.perplexity_key
            elif company == "groq":
                if not self.config.groq_key:
                    raise ValueError("Groq API key is required for Groq models. You should set `groq_key` in LiteLLMConfig")
                os.environ["GROQ_API_KEY"] = self.config.groq_key
            else:
                raise ValueError(f"Unsupported company: {company}")

        self._default_ignore_fields = [
            "llm_type", "output_response", "openai_key", "deepseek_key", "anthropic_key", 
            "gemini_key", "meta_llama_key", "openrouter_key", "openrouter_base", "perplexity_key", 
            "groq_key", "api_base", "is_local", "azure_endpoint", "azure_key", "api_version", "api_key"
        ] # parameters in LiteLLMConfig that are not LiteLLM models' input parameters 
    
    def supports_native_tool_calling(self) -> bool:
        # LiteLLM is a meta-provider. For cloud providers it translates the native
        # tool-calling round-trip (tool_calls + role:tool) per backend, which works
        # for the OpenAI/Anthropic/Gemini/DeepSeek families. Local models (Ollama,
        # LM Studio, etc.) have unreliable tool support, so keep the prompt-based
        # fallback for them. (`litellm.supports_function_calling` is not used here:
        # it wrongly reports False for many tool-capable hosted models.)
        return not bool(self.config.is_local)

    def _apply_provider_params(self, completion_params: dict) -> dict:
        """Inject provider-specific routing parameters (local / Azure) into the
        LiteLLM completion params. OpenAI and the remaining providers are routed
        purely through the environment variables set in ``init_model``."""
        company = infer_litellm_company_from_model(self.model)
        if self.config.is_local or company == "local":  # route local model through its api_base
            completion_params["api_base"] = self.api_base
            completion_params["api_key"] = self.api_key
        elif company == "azure":  # Add Azure OpenAI specific parameters
            completion_params["api_base"] = self.config.azure_endpoint
            completion_params["api_version"] = self.config.api_version
            completion_params["api_key"] = self.config.azure_key
        return completion_params

    def _compute_cost(self, usage) -> Cost:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # Local models are free; LiteLLM has no pricing for them.
        if self.config.is_local:
            return Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=0.0, output_cost=0.0)
        try:
            return super()._compute_cost(usage)
        except Exception as e:
            # Unlike OpenAILLM (which validates the model against the price map at
            # init), LiteLLM accepts arbitrary provider/model names, and
            # ``litellm.cost_per_token`` raises for any model missing from LiteLLM's
            # price map. Since cost is computed inside the generation path, a pricing
            # gap must not abort an otherwise-successful (and possibly retried) call —
            # record the tokens and fall back to zero cost.
            logger.warning(
                f"[LiteLLM] Could not compute cost for model '{self.config.model}': {e}. "
                "Recording tokens with zero cost."
            )
            return Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=0.0, output_cost=0.0)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        """
        Generate a single response using the LiteLLM completion function.

        Args:
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.

        Returns:
            str: A string containing the model's response.
        """
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            completion_params = self.get_completion_params(**kwargs)
            self._apply_provider_params(completion_params)
            response = completion(messages=messages, **completion_params)
            # get_stream_output / get_completion_output record cost internally via _update_cost.
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of LiteLLM: {str(e)}")

        return output

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a single response using the async LiteLLM completion function.

        Args:
            messages (List[dict]): A list of dictionaries representing the conversation history.
            **kwargs (Any): Additional parameters to be passed to the `completion` function.

        Returns:
            str: A string containing the model's response.
        """
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            completion_params = self.get_completion_params(**kwargs)
            self._apply_provider_params(completion_params)
            response = await acompletion(messages=messages, **completion_params)
            if stream:
                if hasattr(response, "__aiter__"):
                    output = await self.get_stream_output_async(response, output_response=output_response)
                else:
                    output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of LiteLLM: {str(e)}")

        return output
