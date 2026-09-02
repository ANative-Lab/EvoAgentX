from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tenacity import wait_none

from evoagentx.models.litellm_model import LiteLLM
from evoagentx.models.model_configs import (
    LiteLLMConfig,
    OpenAILLMConfig,
    OpenRouterConfig,
)
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.openrouter_model import OpenRouterLLM
from tests.src.models.mock_response import (
    get_openai_chat_completion,
    get_openrouter_chat_completion,
)


def _make_openai_llm() -> OpenAILLM:
    return OpenAILLM(
        config=OpenAILLMConfig(
            model="gpt-4o-mini",
            openai_key="mock_openai_key",
            output_response=False,
        )
    )


def _make_litellm() -> LiteLLM:
    return LiteLLM(
        config=LiteLLMConfig(
            model="gpt-4o-mini",
            openai_key="mock_openai_key",
            output_response=False,
        )
    )


def _make_openrouter_llm() -> OpenRouterLLM:
    return OpenRouterLLM(
        config=OpenRouterConfig(
            model="openai/gpt-4o-mini",
            openrouter_key="mock_openrouter_key",
            output_response=False,
        )
    )


@pytest.mark.parametrize(
    ("llm_factory", "create_target", "response", "expected"),
    [
        (
            _make_openai_llm,
            "openai.resources.chat.completions.AsyncCompletions.create",
            get_openai_chat_completion(),
            "Beijing",
        ),
        (
            _make_litellm,
            "evoagentx.models.litellm_model.acompletion",
            get_openai_chat_completion(),
            "Beijing",
        ),
        (
            _make_openrouter_llm,
            "openai.resources.chat.completions.AsyncCompletions.create",
            get_openrouter_chat_completion(),
            "Paris",
        ),
    ],
)
async def test_single_generate_async_retries_transient_failure(
    mocker,
    monkeypatch,
    llm_factory: Callable[[], Any],
    create_target: str,
    response: Any,
    expected: str,
) -> None:
    llm = llm_factory()
    create = mocker.patch(
        create_target,
        new_callable=AsyncMock,
        side_effect=[RuntimeError("transient error"), response],
    )
    retry_controller = getattr(type(llm).single_generate_async, "retry", None)
    assert retry_controller is not None
    monkeypatch.setattr(retry_controller, "wait", wait_none())

    result = await llm.single_generate_async(
        messages=[{"role": "user", "content": "hello"}]
    )

    assert result == expected
    assert create.await_count == 2
