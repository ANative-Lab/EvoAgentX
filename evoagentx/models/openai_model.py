import json
from typing import Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from litellm import cost_per_token
from litellm.types.utils import Usage

from ..core.logging import logger
from ..core.registry import register_model
from ..prompts.tool_calling import TOOL_CALL_FORMAT
from ..utils.utils import format_tool_calls
from .model_configs import OpenAILLMConfig
from .base_model import BaseLLM
from .model_utils import Cost, cost_manager, get_openai_model_cost


@register_model(config_cls=OpenAILLMConfig, alias=["openai_llm"])
class OpenAILLM(BaseLLM):

    def init_model(self):
        self._client = None
        self._async_client = None
        self._default_ignore_fields = [
            "llm_type", "output_response", "openai_key", "deepseek_key", "anthropic_key",
            "gemini_key", "meta_llama_key", "openrouter_key", "openrouter_base", "perplexity_key",
            "groq_key"
        ] # parameters in OpenAILLMConfig that are not OpenAI models' input parameters
        if self.config.model not in get_openai_model_cost():
            raise KeyError(f"'{self.config.model}' is not a valid OpenAI model name!")

    def supports_native_tool_calling(self) -> bool:
        # OpenAI's chat-completions API is the reference implementation of native
        # function calling. SiliconFlow and Aliyun (DashScope compatible-mode) speak
        # the same protocol and inherit this; all three were verified end-to-end
        # (native tool_calls + role:tool round-trip) against the real APIs.
        return True

    def _init_client(self, config: OpenAILLMConfig):
        return OpenAI(api_key=config.openai_key)

    def _init_async_client(self, config: OpenAILLMConfig):
        return AsyncOpenAI(api_key=config.openai_key)

    def ensure_client(self):
        if self._client is None or self._client.is_closed():
            self._client = self._init_client(self.config)
        return self._client

    def close_client(self):
        if self._client is not None and not self._client.is_closed():
            self._client.close()

    def ensure_async_client(self):
        if self._async_client is None or self._async_client.is_closed():
            self._async_client = self._init_async_client(self.config)
        return self._async_client

    async def close_async_client(self):
        if self._async_client is not None and not self._async_client.is_closed():
            await self._async_client.close()

    def formulate_messages(self, prompts: List[str], system_messages: Optional[List[str]] = None) -> List[List[dict]]:

        if system_messages:
            assert len(prompts) == len(system_messages), f"the number of prompts ({len(prompts)}) is different from the number of system_messages ({len(system_messages)})"
        else:
            system_messages = [None] * len(prompts)

        messages_list = []
        for prompt, system_message in zip(prompts, system_messages):
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        return messages_list

    def update_completion_params(self, params1: dict, params2: dict) -> dict:
        config_params: list = self.config.get_config_params()
        for key, value in params2.items():
            if key in self._default_ignore_fields:
                continue
            if key not in config_params:
                continue
            params1[key] = value
        return params1

    def get_completion_params(self, **kwargs):
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        completion_params = self.update_completion_params(completion_params, kwargs)
        # automatically set stream_options to include usage if stream is True,
        # as OpenAI's streaming response does not include usage by default, 
        # which is needed for cost tracking.
        if completion_params.get("stream"):
            stream_options = dict(completion_params.get("stream_options") or {})
            stream_options.setdefault("include_usage", True)
            completion_params["stream_options"] = stream_options
        return completion_params

    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        """
        Process stream response and return the complete output.

        Args:
            response: The stream response from OpenAI
            output_response: Whether to print the response in real-time

        Returns:
            str: The complete output text
        """
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                usage_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if output_response:
                    print(delta.content, end="", flush=True)
                output += delta.content
            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, tool_calls_accum)
        if output_response:
            print("")
        if tool_calls_accum:
            formatted = self._format_streamed_tool_calls(tool_calls_accum)
            if formatted:
                tool_call_str = TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
                output += tool_call_str
                if output_response:
                    print(tool_call_str)
        if usage_chunk is not None:
            self._update_cost(usage_chunk)
        else:
            logger.warning("[OpenAILLM] No usage data in stream response; cost will not be recorded. Set stream_options={'include_usage': True} to enable cost tracking.")
        return output

    async def get_stream_output_async(self, response, output_response: bool = False) -> str:
        """
        Process async stream response and return the complete output.

        Args:
            response (AsyncIterator[ChatCompletionChunk]): The async stream response from OpenAI
            output_response (bool): Whether to print the response in real-time

        Returns:
            str: The complete output text
        """
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                usage_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if output_response:
                    print(delta.content, end="", flush=True)
                output += delta.content
            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, tool_calls_accum)
        if output_response:
            print("")
        if tool_calls_accum:
            formatted = self._format_streamed_tool_calls(tool_calls_accum)
            if formatted:
                tool_call_str = TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
                output += tool_call_str
                if output_response:
                    print(tool_call_str)
        if usage_chunk is not None:
            self._update_cost(usage_chunk)
        else:
            logger.warning("[OpenAILLM] No usage data in stream response; cost will not be recorded. Set stream_options={'include_usage': True} to enable cost tracking.")
        return output

    def get_completion_output(self, response: ChatCompletion, output_response: bool=True) -> str:
        output = response.choices[0].message.content or ""
        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            formatted = format_tool_calls(tool_calls)
            output += TOOL_CALL_FORMAT.format(tool_calls=json.dumps(formatted, indent=4, ensure_ascii=False))
        if output_response:
            print(output)
        self._update_cost(response)
        return output

    @staticmethod
    def _accumulate_tool_calls(delta_tool_calls, accum: Dict[int, dict]):
        for tc in delta_tool_calls:
            idx = tc.index
            if idx not in accum:
                accum[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
            if tc.id:
                accum[idx]["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    accum[idx]["function"]["name"] += tc.function.name
                if tc.function.arguments:
                    accum[idx]["function"]["arguments"] += tc.function.arguments

    @staticmethod
    def _format_streamed_tool_calls(accum: Dict[int, dict]) -> List[dict]:
        formatted = []
        for idx in sorted(accum.keys()):
            tc = accum[idx]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                logger.error(f"Failed to parse streaming tool call arguments for `{tc['function']['name']}`:\n{tc['function']['arguments']}")
                continue
            formatted.append({"id": tc["id"], "function_name": tc["function"]["name"], "function_args": args})
        return formatted

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        try:
            client = self.ensure_client()
            completion_params = self.get_completion_params(**kwargs)
            response = client.chat.completions.create(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")

        return output

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:

        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            async_client = self.ensure_async_client()
            completion_params = self.get_completion_params(**kwargs)
            response = await async_client.chat.completions.create(messages=messages, **completion_params)
            if stream:
                output = await self.get_stream_output_async(response, output_response=output_response)
            else:
                # The network I/O is already awaited above; the response is fully in memory here,
                # so this synchronous parsing/cost call does not block the event loop.
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of OpenAILLM: {str(e)}")

        return output

    def _compute_cost(self, usage: CompletionUsage) -> Cost:
        # Pass the full usage object to LiteLLM so it can apply the correct rates for
        # cached / reasoning tokens (cached input tokens are billed at a lower rate).
        # LiteLLM expects its own Usage type, so convert from OpenAI's CompletionUsage.
        usage_object = usage if isinstance(usage, Usage) else Usage(**usage.model_dump())
        input_cost, output_cost = cost_per_token(model=self.config.model, usage_object=usage_object)
        return Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
        )

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.warning(f"[OpenAILLM] usage is None in response (id={getattr(response, 'id', '?')}); cost will not be recorded.")
            return
        cost_manager.update_cost(cost=self._compute_cost(usage), model=self.config.model)
