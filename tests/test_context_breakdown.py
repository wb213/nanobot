"""Tests for context breakdown calculation and formatting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.utils.helpers import calculate_context_breakdown, format_context_breakdown


class _FakeMemory:
    """Mock MemoryStore for testing."""

    def __init__(
        self,
        memory_content: str = "",
        history_entries: list[dict[str, Any]] | None = None,
        last_dream_cursor: int = 0,
    ):
        self._memory_content = memory_content
        self._history_entries = history_entries or []
        self._last_dream_cursor = last_dream_cursor

    def get_memory_context(self) -> str:
        return self._memory_content

    def read_unprocessed_history(self, since_cursor: int = 0) -> list[dict[str, Any]]:
        return self._history_entries

    def get_last_dream_cursor(self) -> int:
        return self._last_dream_cursor


class _FakeSkills:
    """Mock SkillsLoader for testing."""

    def __init__(
        self,
        always_skills: list[str] | None = None,
        always_content: str = "",
        skills_summary: str = "",
    ):
        self._always_skills = always_skills or []
        self._always_content = always_content
        self._skills_summary = skills_summary

    def get_always_skills(self) -> list[str]:
        return self._always_skills

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        return self._always_content

    def build_skills_summary(self) -> str:
        return self._skills_summary


class _FakeContextBuilder:
    """Mock ContextBuilder for testing."""

    _MAX_RECENT_HISTORY = 50

    def __init__(
        self,
        identity: str = "Test Identity",
        bootstrap: str = "Bootstrap Files",
        memory: _FakeMemory | None = None,
        skills: _FakeSkills | None = None,
        timezone: str | None = None,
    ):
        self._identity = identity
        self._bootstrap = bootstrap
        self.memory = memory or _FakeMemory()
        self.skills = skills or _FakeSkills()
        self.timezone = timezone

    def _get_identity(self, channel: str | None = None) -> str:
        return self._identity

    def _load_bootstrap_files(self) -> str:
        return self._bootstrap

    @staticmethod
    def _build_runtime_context(
        channel: str | None,
        chat_id: str | None,
        timezone: str | None = None,
        session_summary: str | None = None,
    ) -> str:
        return "[Runtime Context]\nCurrent Time: 2026-04-17\n[/Runtime Context]"


class _FakeSession:
    """Mock Session for testing."""

    def __init__(self, messages: list[dict[str, Any]] | None = None):
        self.messages = messages or []

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        return self.messages if max_messages == 0 else self.messages[-max_messages:]


class _FakeTools:
    """Mock ToolRegistry for testing."""

    def __init__(self, tools: list[dict[str, Any]] | None = None):
        self._tools = tools or []

    def get_definitions(self) -> list[dict[str, Any]]:
        return self._tools


class _FakeLoop:
    """Mock AgentLoop for testing."""

    def __init__(self, tools: _FakeTools | None = None):
        self.tools = tools or _FakeTools()


def test_calculate_context_breakdown_basic():
    """Test basic context breakdown calculation."""
    context_builder = _FakeContextBuilder(
        identity="Identity: 100 chars",
        bootstrap="Bootstrap: 200 chars",
        memory=_FakeMemory(memory_content="Memory: 300 chars"),
        skills=_FakeSkills(
            always_skills=["skill1"],
            always_content="Always Skills: 150 chars",
            skills_summary="Skills Summary: 250 chars",
        ),
    )

    session = _FakeSession(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )

    loop = _FakeLoop(
        tools=_FakeTools(
            tools=[
                {"name": "tool1", "description": "A test tool"},
                {"name": "tool2", "description": "Another tool"},
            ]
        )
    )

    breakdown = calculate_context_breakdown(context_builder, session, loop)

    # Verify structure
    assert "parts" in breakdown
    assert "total_chars" in breakdown
    assert "history_stats" in breakdown
    assert "tools_stats" in breakdown

    # Verify parts exist
    parts = breakdown["parts"]
    assert "identity" in parts
    assert "bootstrap" in parts
    assert "memory" in parts
    assert "always_skills" in parts
    assert "skills_summary" in parts
    assert "recent_history" in parts
    assert "system_prompt_total" in parts
    assert "history_messages" in parts
    assert "tools_definitions" in parts
    assert "runtime_context" in parts

    # Verify identity and bootstrap match
    assert parts["identity"] == len("Identity: 100 chars")
    assert parts["bootstrap"] == len("Bootstrap: 200 chars")
    assert parts["memory"] == len("Memory: 300 chars")

    # Verify history stats
    assert breakdown["history_stats"]["total_messages"] == 2
    assert breakdown["history_stats"]["user_messages"] == 1
    assert breakdown["history_stats"]["assistant_messages"] == 1

    # Verify tools stats
    assert breakdown["tools_stats"]["total_tools"] == 2


def test_calculate_context_breakdown_empty_memory():
    """Test with no memory content."""
    context_builder = _FakeContextBuilder(
        memory=_FakeMemory(memory_content=""),
    )
    session = _FakeSession()
    loop = _FakeLoop()

    breakdown = calculate_context_breakdown(context_builder, session, loop)

    assert breakdown["parts"]["memory"] == 0


def test_calculate_context_breakdown_no_skills():
    """Test with no skills."""
    context_builder = _FakeContextBuilder(
        skills=_FakeSkills(always_skills=[], always_content="", skills_summary=""),
    )
    session = _FakeSession()
    loop = _FakeLoop()

    breakdown = calculate_context_breakdown(context_builder, session, loop)

    assert breakdown["parts"]["always_skills"] == 0


def test_calculate_context_breakdown_with_history_entries():
    """Test with recent history entries."""
    history_entries = [
        {"timestamp": "2026-04-17 10:00", "content": "Entry 1"},
        {"timestamp": "2026-04-17 10:05", "content": "Entry 2"},
        {"timestamp": "2026-04-17 10:10", "content": "Entry 3"},
    ]

    context_builder = _FakeContextBuilder(
        memory=_FakeMemory(history_entries=history_entries),
    )
    session = _FakeSession()
    loop = _FakeLoop()

    breakdown = calculate_context_breakdown(context_builder, session, loop)

    # Should have non-zero recent history
    assert breakdown["parts"]["recent_history"] > 0


def test_format_context_breakdown_basic():
    """Test formatting context breakdown."""
    breakdown = {
        "parts": {
            "identity": 500,
            "bootstrap": 3200,
            "memory": 4500,
            "always_skills": 2000,
            "skills_summary": 4500,
            "recent_history": 500,
            "system_prompt_total": 15200,
            "history_messages": 18300,
            "tools_definitions": 5800,
            "runtime_context": 300,
        },
        "total_chars": 39600,
        "history_stats": {
            "total_messages": 42,
            "user_messages": 25,
            "assistant_messages": 17,
        },
        "tools_stats": {
            "total_tools": 12,
        },
    }

    last_usage = {
        "prompt_tokens": 45231,
        "completion_tokens": 1234,
        "cached_tokens": 35280,
    }

    output = format_context_breakdown(breakdown, last_usage)

    # Verify key sections are present
    assert "Context Breakdown" in output
    assert "System Prompt:" in output
    assert "History Messages:" in output
    assert "Tools Definitions:" in output
    assert "Runtime Context:" in output
    assert "Token Usage" in output

    # Verify character sizes are formatted (k suffix)
    assert "15.2k" in output  # system_prompt_total
    assert "18.3k" in output  # history_messages
    assert "5.8k" in output   # tools_definitions

    # Verify percentages are shown
    assert "38%" in output  # system_prompt percentage
    assert "46%" in output  # history_messages percentage

    # Verify message stats
    assert "42 messages" in output
    assert "25 user" in output
    assert "17 assistant" in output

    # Verify tools stats
    assert "12 tools" in output  # May show "12 tools:" or "12 tools loaded" depending on availability

    # Verify token usage
    assert "45,231 tokens" in output
    assert "1,234 tokens" in output
    assert "35,280 tokens" in output
    assert "77% of input" in output  # cache percentage (35280/45231 = 77%)

    # Verify char/token ratio
    assert "Estimated Ratio:" in output


def test_format_context_breakdown_no_token_usage():
    """Test formatting when no token usage data is available."""
    breakdown = {
        "parts": {
            "identity": 500,
            "bootstrap": 0,
            "memory": 0,
            "always_skills": 0,
            "skills_summary": 0,
            "recent_history": 0,
            "system_prompt_total": 500,
            "history_messages": 0,
            "tools_definitions": 100,
            "runtime_context": 50,
        },
        "total_chars": 650,
        "history_stats": {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
        },
        "tools_stats": {
            "total_tools": 2,
        },
    }

    last_usage = {}  # Empty usage data

    output = format_context_breakdown(breakdown, last_usage)

    assert "No token usage data yet" in output
    assert "Estimated Ratio" not in output


def test_format_context_breakdown_zero_cached():
    """Test formatting when cached tokens is zero."""
    breakdown = {
        "parts": {
            "identity": 1000,
            "bootstrap": 0,
            "memory": 0,
            "always_skills": 0,
            "skills_summary": 0,
            "recent_history": 0,
            "system_prompt_total": 1000,
            "history_messages": 500,
            "tools_definitions": 200,
            "runtime_context": 50,
        },
        "total_chars": 1750,
        "history_stats": {
            "total_messages": 2,
            "user_messages": 1,
            "assistant_messages": 1,
        },
        "tools_stats": {
            "total_tools": 3,
        },
    }

    last_usage = {
        "prompt_tokens": 2000,
        "completion_tokens": 300,
        "cached_tokens": 0,
    }

    output = format_context_breakdown(breakdown, last_usage)

    assert "0% of input" in output  # Should show 0% cache


def test_format_context_breakdown_sizes():
    """Test size formatting for different magnitudes."""
    breakdown = {
        "parts": {
            "identity": 123,       # < 1k, should be "123"
            "bootstrap": 1234,     # >= 1k, should be "1.2k"
            "memory": 12345,       # >= 10k, should be "12.3k"
            "always_skills": 0,
            "skills_summary": 0,
            "recent_history": 0,
            "system_prompt_total": 13702,
            "history_messages": 0,
            "tools_definitions": 0,
            "runtime_context": 0,
        },
        "total_chars": 13702,
        "history_stats": {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
        },
        "tools_stats": {
            "total_tools": 0,
        },
    }

    last_usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0}

    output = format_context_breakdown(breakdown, last_usage)

    # Small number should not have k suffix
    assert "123 (" in output or "123(" in output

    # Large numbers should have k suffix
    assert "1.2k" in output
    assert "12.3k" in output
