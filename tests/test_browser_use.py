import asyncio
import sys
from types import ModuleType

import pytest

from evoagentx.tools.browser_use import BrowserUseBase


def install_fake_browser_use(monkeypatch, captured):
    browser_use = ModuleType("browser_use")
    browser_use_llm = ModuleType("browser_use.llm")

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return "done"

    class FakeBrowserProfile:
        def __init__(self, **kwargs):
            self.headless = kwargs["headless"]

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    browser_use.Agent = FakeAgent
    browser_use.BrowserProfile = FakeBrowserProfile
    browser_use_llm.ChatOpenAI = FakeChatOpenAI
    browser_use_llm.ChatAnthropic = FakeChatOpenAI

    monkeypatch.setitem(sys.modules, "browser_use", browser_use)
    monkeypatch.setitem(sys.modules, "browser_use.llm", browser_use_llm)


def test_current_browser_use_receives_headless_profile(monkeypatch):
    captured = {}
    install_fake_browser_use(monkeypatch, captured)

    browser = BrowserUseBase(api_key="test-key", headless=True)
    result = asyncio.run(browser.execute_task("Open example.com"))

    assert result == {"success": True, "result": "done"}
    assert captured["browser_profile"].headless is True
    assert "headless" not in captured
    assert "browser_type" not in captured


def test_current_browser_use_rejects_unsupported_browser(monkeypatch):
    install_fake_browser_use(monkeypatch, {})

    with pytest.raises(ValueError, match="supports Chromium only"):
        BrowserUseBase(api_key="test-key", browser_type="firefox")
