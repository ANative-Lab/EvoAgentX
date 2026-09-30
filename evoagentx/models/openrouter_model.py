import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.logging import logger
from ..core.registry import register_model
from ..prompts.tool_calling import TOOL_CALL_FORMAT
from ..utils.utils import format_tool_calls
from .base_model import BaseLLM
from .model_configs import OpenRouterConfig
from .model_utils import Cost, cost_manager


# Models already warned about paid cache writes. Process-wide so the warning
# fires at most once per model regardless of how many LLM instances are created.
_PROMPT_CACHING_COST_WARNED: set = set()


@register_model(config_cls=OpenRouterConfig, alias=["openrouter"])
class OpenRouterLLM(BaseLLM):

    # Model-family prefixes that require an explicit `cache_control` breakpoint AND
    # bill cache writes at a premium (verified against OpenRouter's pricing docs):
    #   anthropic/*    -> writes at ~1.25x input (5-min) / 2x (1-hour)
    #   google/gemini* -> writes at input price plus a cache-storage fee
    #   qwen/*         -> Alibaba explicit-cache write multiplier
    # Automatic-cache providers (openai/*, deepseek/*, x-ai/*, moonshot/*) are
    # intentionally absent: their writes are free and need no breakpoint, so we
    # leave their requests untouched and let OpenRouter cache them server-side.
    _PAID_CACHE_WRITE_PREFIXES: Tuple[str, ...] = ("anthropic/", "google/gemini", "qwen/")

    def init_model(self):
        self._client = None
        self._async_client = None
        self._default_ignore_fields = ["llm_type", "openrouter_key", "openrouter_base", "openrouter_model_base", "output_response"]
    
    def supports_native_tool_calling(self) -> bool:
        # OpenRouter proxies the OpenAI tool-calling protocol for the models that
        # support it; native tool calling was the original behavior here.
        return True

    def prepare_request(self, messages: List[dict], params: dict) -> Tuple[List[dict], dict]:
        """Inject OpenRouter prompt-caching breakpoints when opted in.

        Caching is gated behind `enable_prompt_caching` (per-call kwarg, else the
        config default of False) because cache writes are billed at a premium on
        the providers that need explicit breakpoints (see
        `_PAID_CACHE_WRITE_PREFIXES`). The flag is popped here so it never leaks
        into the OpenAI-compatible request body.
        """
        # `enable_prompt_caching` is a config field, so a per-call kwarg flows into
        # `params` via get_completion_params; pop it regardless to keep it off the wire.
        enabled = params.pop("enable_prompt_caching", None)
        if enabled is None:
            enabled = bool(getattr(self.config, "enable_prompt_caching", False))
        if not enabled:
            return messages, params

        model = (getattr(self.config, "model", "") or "").lower()
        if not model.startswith(self._PAID_CACHE_WRITE_PREFIXES):
            # Automatic-cache providers need no breakpoint and bill no write premium.
            return messages, params

        self._warn_prompt_caching_cost_once(model)
        return self._add_cache_control_to_last_text_block(messages), params

    @staticmethod
    def _warn_prompt_caching_cost_once(model: str) -> None:
        if model in _PROMPT_CACHING_COST_WARNED:
            return
        _PROMPT_CACHING_COST_WARNED.add(model)
        logger.warning(
            f"[OpenRouterLLM] Prompt caching is enabled for '{model}'. On this provider "
            "OpenRouter bills cache WRITES at a premium (e.g. Anthropic ~1.25x input "
            "price; Gemini adds a cache-storage fee; Qwen applies a write multiplier), "
            "so a single non-repeated call costs MORE than without caching — it only "
            "pays off across repeated calls that share a prompt prefix. To disable, set "
            "`enable_prompt_caching=False` on the OpenRouterConfig."
        )

    @staticmethod
    def _add_cache_control_to_last_text_block(messages: List[dict]) -> List[dict]:
        """Return a copy with OpenRouter block-level prompt caching enabled.

        OpenRouter routes Anthropic/Gemini/Qwen prompt caching through
        Anthropic-style content-block metadata. Use a single breakpoint on the
        latest cacheable text block so multi-turn agent loops cache the growing
        shared prefix without accumulating multiple paid cache writes. The input
        `messages` is never mutated.
        """
        cached_messages = deepcopy(messages)
        target_message: Optional[dict] = None
        target_block: Optional[dict] = None

        for message in cached_messages:
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    target_message = message
                    target_block = None
                continue

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block.pop("cache_control", None)
                    if (
                        block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                        and block.get("text")
                    ):
                        target_message = None
                        target_block = block

        if target_block is not None:
            target_block["cache_control"] = {"type": "ephemeral"}
            return cached_messages

        if target_message is not None:
            target_message["content"] = [
                {
                    "type": "text",
                    "text": target_message["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return cached_messages

    def _init_client(self, config: OpenRouterConfig):
        return OpenAI(api_key=config.openrouter_key, base_url=config.openrouter_base)

    def _init_async_client(self, config: OpenRouterConfig):
        return AsyncOpenAI(api_key=config.openrouter_key, base_url=config.openrouter_base)

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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close_async_client()

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
        return completion_params
    
    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        for chunk in response:
            if chunk.usage is not None:
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
            logger.warning("[OpenRouterLLM] No usage data in stream response; cost will not be recorded.")
        return output

    async def get_stream_output_async(self, response, output_response: bool = False) -> str:
        output = ""
        tool_calls_accum: Dict[int, dict] = {}
        usage_chunk = None
        async for chunk in response:
            if chunk.usage is not None:
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
            logger.warning("[OpenRouterLLM] No usage data in stream response; cost will not be recorded.")
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

    def _update_cost(self, response: Union[ChatCompletion, ChatCompletionChunk]):
        usage = response.usage
        if usage is None:
            logger.warning(f"[OpenRouterLLM] usage is None in response (id={response.id}); cost will not be recorded.")
            return
        cost_value = getattr(usage, "cost", None)
        if cost_value is None:
            logger.warning(
                f"[OpenRouterLLM] usage.cost not present in response (id={response.id}); "
                "cost will be recorded as 0. Check OpenRouter dashboard for actual spend."
            )
            cost_value = 0.0
        cost = Cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=cost_value,
        )
        cost_manager.update_cost(cost, model=self.config.model)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            client = self.ensure_client()
            completion_params = self.get_completion_params(**kwargs)
            messages, completion_params = self.prepare_request(messages, completion_params)
            response = client.chat.completions.create(messages=messages, **completion_params)
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
            else:
                output: str = self.get_completion_output(response=response, output_response=output_response)
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of OpenRouterLLM: {str(e)}")

        return output

    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]

    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        stream = kwargs.get("stream", self.config.stream)
        output_response = kwargs.get("output_response", self.config.output_response)

        try:
            async_client = self.ensure_async_client()
            completion_params = self.get_completion_params(**kwargs)
            messages, completion_params = self.prepare_request(messages, completion_params)
            response = await async_client.chat.completions.create(
                messages=messages, **completion_params
            )
            if stream:
                output = await self.get_stream_output_async(response, output_response=output_response)
            else:
                # The network I/O is already awaited above; the response is fully in memory here,
                # so this synchronous parsing/cost call does not block the event loop.
                output: str = self.get_completion_output(response=response, output_response=output_response)

        except Exception as e:
            raise RuntimeError(f"Error during single_generate_async of OpenRouterLLM: {str(e)}")

        return output
