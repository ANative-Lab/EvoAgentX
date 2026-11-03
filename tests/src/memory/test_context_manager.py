"""Regression coverage for consecutive text and multimodal user prompts."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from evoagentx.memory.context_manager import ContextManager
from evoagentx.models import BaseLLM

IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
TEXT = {"type": "text", "text": "Describe this image."}


@pytest.mark.parametrize("native_tool_calling", [False, True])
@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("First", "Second", "First\n\nSecond"),
        ("First", [IMAGE], [{"type": "text", "text": "First"}, IMAGE]),
        ([IMAGE], [TEXT], [IMAGE, TEXT]),
        ([IMAGE], TEXT["text"], [IMAGE, TEXT]),
    ],
)
def test_consecutive_user_prompts_preserve_content_parts(
    native_tool_calling, first, second, expected
):
    """Text following an image remains one text part, never individual characters."""
    llm = Mock(spec=BaseLLM)
    llm.supports_native_tool_calling.return_value = native_tool_calling
    context = ContextManager(llm=llm)
    first, second = deepcopy(first), deepcopy(second)
    original_inputs = deepcopy((first, second))

    context.add_user_prompt(first)
    context.add_user_prompt(second)

    assert context.context == [{"role": "user", "content": expected}]
    assert (first, second) == original_inputs
