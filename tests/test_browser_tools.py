"""
Tests for browser tools
"""

import pytest
from browser_agent.src.tools import browser_tools


@pytest.mark.asyncio
async def test_extract_content_tool_filters_visible_text():
    class DummyPage:
        async def evaluate(self, script, args=None):
            return "\n".join(["Header", "foo item one", "bar item two", "foo item three"])

    # Inject dummy page into the module (no Playwright required).
    browser_tools._current_page = DummyPage()

    res = await browser_tools.extract_content_tool("foo")
    assert res["success"] is True
    assert "foo item one" in res["content"]
    assert "foo item three" in res["content"]
    assert "bar item two" not in res["content"]


@pytest.mark.asyncio
async def test_extract_content_tool_truncates_output():
    class DummyPage:
        async def evaluate(self, script, args=None):
            return ("line with keyword\n" * 10000).strip()

    browser_tools._current_page = DummyPage()

    res = await browser_tools.extract_content_tool("keyword")
    assert res["success"] is True
    assert len(res["content"]) <= 6000  # includes "[Truncated]" marker
