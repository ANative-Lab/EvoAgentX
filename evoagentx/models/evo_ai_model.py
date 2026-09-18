"""
EVO-AI LLM Integration for EvoAgentX

Adds EVO-AI as a custom model backend for EvoAgentX. EVO-AI is a distributed
self-evolving language model network - any EvoAgentX agent can call our API
as a regular LLM backend.

Usage:
    from evoagentx.models import EvoAILLM, EvoAILLMConfig
    
    config = EvoAILLMConfig(
        model="evo-ai-distributed",
        evo_network="https://kkkix-47-253-174-153.run.pinggy-free.link",
        evo_node_id="my-bot",
    )
    llm = EvoAILLM(config)
    
    response = llm.single_generate([
        {"role": "user", "content": "Explain self-evolution"}
    ])
    
    # Async
    response = await llm.single_generate_async([...])
"""

from typing import Dict, List, Optional, Union, Any
import asyncio
import json
import os
import time

from .base_model import BaseLLM
from .model_configs import LLMConfig
from ..core.logging import logger
from ..core.registry import register_model
from ..prompts.tool_calling import TOOL_CALL_FORMAT
from ..utils.utils import format_tool_calls


# Default network endpoints (public HTTPS bridges)
DEFAULT_NETWORKS = [
    "https://kkkix-47-253-174-153.run.pinggy-free.link",
    "https://59b15f36b60415b9-47-253-174-153.serveousercontent.com",
    "http://47.253.174.153:80",
]


class EvoAILLMConfig(LLMConfig):
    """Configuration for the EVO-AI distributed LLM backend.
    
    Attributes:
        llm_type: Always "EvoAILLM" (auto-set).
        model: The model identifier within EVO-AI. Default "evo-ai-distributed".
        evo_network: Public URL of the EVO-AI network. Optional - uses first
                     reachable from DEFAULT_NETWORKS if unset.
        evo_node_id: Node identifier for credit attribution. Earns EVO
                     tokens (51K welcome + 100/day heartbeat).
        timeout: Request timeout in seconds (default 60).
    """
    llm_type: str = "EvoAILLM"
    model: str = "evo-ai-distributed"
    
    evo_network: Optional[str] = None
    evo_node_id: Optional[str] = None
    timeout: Optional[float] = 60.0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256


@register_model(config_cls=EvoAILLMConfig, alias=["evo_ai", "evo-ai", "evoai"])
class EvoAILLM(BaseLLM):
    """EVO-AI distributed LLM client.
    
    EVO-AI exposes a JSON HTTP API at /api/generate that returns text completions
    from a self-evolving 813K-parameter GPT model trained across volunteer nodes.
    Earnings are paid in EVO tokens (12,750 EVO/USD swap rate).
    
    Quick start for users:
        pip install requests  # only stdlib + requests required
        
        from EvoAgentX.models import EvoAILLM, EvoAILLMConfig
        config = EvoAILLMConfig(model="evo-ai-distributed")
        llm = EvoAILLM(config)
        print(llm.single_generate([{"role": "user", "content": "Hello"}]))
    """
    
    def init_model(self):
        """Lazy client - we don't open a socket until generate() is called."""
        self._client = None
        self._async_client = None
        self._network = self._resolve_network()
        self._node_id = self.config.evo_node_id or f"evoagentx-{os.getpid()}"
        self._default_ignore_fields = [
            "llm_type", "output_response", "evo_network", "evo_node_id",
            "timeout", "evo_ai_key",
        ]
    
    def _resolve_network(self) -> str:
        """Pick the first reachable EVO-AI network."""
        candidates = []
        if self.config.evo_network:
            candidates.append(self.config.evo_network)
        candidates += DEFAULT_NETWORKS
        
        import requests
        for url in candidates:
            try:
                r = requests.get(f"{url}/api/info", timeout=5)
                if r.status_code == 200:
                    return url
            except Exception:
                continue
        # Fall back to the first candidate even if it might be down - upstream
        # generate calls will surface a clear error message.
        return candidates[0]
    
    def supports_native_tool_calling(self) -> bool:
        """EVO-AI's distributed GPT does not yet support native tool-calling.
        
        The agent loop will fall back to the prompt-based <tool_call> protocol,
        which works but requires the model to be instructed to emit JSON.
        Future EvoAgentX PRs may upgrade this to True once EVO-AI supports it.
        """
        return False
    
    def prepare_request(self, messages: List[dict], params: dict) -> tuple:
        """No request-shaping needed - we wrap OpenAI-style messages into a prompt."""
        return messages, params
    
    def formulate_messages(
        self, prompts: List[str], system_messages: Optional[List[str]] = None
    ) -> List[List[dict]]:
        """Wrap raw prompts in chat-format. Same as OpenAILLM.formulate_messages."""
        if system_messages:
            assert len(prompts) == len(system_messages), (
                f"the number of prompts ({len(prompts)}) is different from "
                f"the number of system_messages ({len(system_messages)})"
            )
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
    
    def get_completion_params(self, **kwargs):
        """Return params for get_completion - mostly inherited from config."""
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        # Per-call overrides
        for key, value in kwargs.items():
            if key in self._default_ignore_fields:
                continue
            if key in self.config.get_config_params():
                completion_params[key] = value
        # Drop OpenAI-only keys
        for k in ["stream", "stream_options", "tools", "tool_choice", "modalities",
                  "response_format", "logprobs", "top_logprobs", "prediction",
                  "parallel_tool_calls", "reasoning_effort", "n"]:
            completion_params.pop(k, None)
        return completion_params
    
    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """Flatten chat messages into a single prompt string."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            parts.append(f"[{role}] {content}")
        parts.append("[assistant]")
        return "\n".join(parts)
    
    def _call_api(self, messages: List[dict], **kwargs) -> str:
        """Call EVO-AI /api/generate synchronously."""
        import requests
        prompt = self._messages_to_prompt(messages)
        params = self.get_completion_params(**kwargs)
        try:
            r = requests.post(
                f"{self._network}/api/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": params.get("max_tokens", 256),
                    "temperature": params.get("temperature", 0.7),
                    "node_id": self._node_id,
                },
                timeout=self.config.timeout or 60,
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"EVO-AI /api/generate returned {r.status_code}: {r.text[:200]}"
                )
            data = r.json()
            return data.get("text") or data.get("output") or json.dumps(data)
        except requests.RequestException as e:
            raise RuntimeError(f"EVO-AI network error: {str(e)}")
    
    async def _call_api_async(self, messages: List[dict], **kwargs) -> str:
        """Call EVO-AI /api/generate asynchronously."""
        import aiohttp
        prompt = self._messages_to_prompt(messages)
        params = self.get_completion_params(**kwargs)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._network}/api/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": params.get("max_tokens", 256),
                        "temperature": params.get("temperature", 0.7),
                        "node_id": self._node_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout or 60),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"EVO-AI /api/generate returned {resp.status}: {text[:200]}"
                        )
                    data = await resp.json()
                    return data.get("text") or data.get("output") or json.dumps(data)
        except Exception as e:
            raise RuntimeError(f"EVO-AI network error (async): {str(e)}")
    
    def get_completion_output(
        self, response: Any, output_response: bool = True
    ) -> str:
        """No-op - response is already a string from _call_api."""
        if output_response:
            print(response, end="", flush=True)
            print("")
        return response if isinstance(response, str) else json.dumps(response)
    
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        """Generate one completion. Streams if requested (via prints)."""
        output_response = kwargs.get("output_response", self.config.output_response)
        try:
            output = self._call_api(messages, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error during EvoAILLM.single_generate: {str(e)}")
        if output_response:
            print(output)
        return output
    
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """Sequential batch - EVO-AI is rate-limited to one call per node at a time."""
        return [self.single_generate(messages=m, **kwargs) for m in batch_messages]
    
    async def single_generate_async(self, messages: List[dict], **kwargs) -> str:
        """Async generate - uses aiohttp if available, otherwise falls back to thread."""
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            # Fall back to running sync call in thread
            return await asyncio.to_thread(self.single_generate, messages, **kwargs)
        
        output_response = kwargs.get("output_response", self.config.output_response)
        try:
            output = await self._call_api_async(messages, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error during EvoAILLM.single_generate_async: {str(e)}")
        if output_response:
            print(output)
        return output
    
    def supports_function_calling(self) -> bool:
        """Legacy alias - newer code uses supports_native_tool_calling."""
        return False
    
    def get_stream_output(self, response: Any, output_response: bool = True) -> str:
        """Streaming isn't supported in EVO-AI - return single-shot response."""
        return self.get_completion_output(response, output_response)
    
    async def get_stream_output_async(
        self, response: Any, output_response: bool = False
    ) -> str:
        """Async streaming shim - EVO-AI is request/response only."""
        if asyncio.iscoroutine(response):
            response = await response
        return self.get_completion_output(response, output_response)
    
    def close_client(self):
        """No persistent client - requests is stateless."""
        pass
    
    async def close_async_client(self):
        """No persistent async client."""
        pass
    
    def ensure_client(self):
        """No persistent client - always returns self for API compatibility."""
        return self
    
    def ensure_async_client(self):
        """No persistent async client."""
        return self


__all__ = ["EvoAILLM", "EvoAILLMConfig"]
