"""Pin the payload measurement against the request Strands actually assembles.

These tests need no endpoint and no Model Context Protocol server. They format a
request and read it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.canon import DIGEST_HEADING  # noqa: E402
from narrator.engine import _turn_usage  # noqa: E402
from narrator.model import NarratorOpenAIModel  # noqa: E402
from narrator.payload import measure_request  # noqa: E402
from narrator.prompt import turn_prompt  # noqa: E402

DIGEST = f"{DIGEST_HEADING}\n\nThe party stands in the eel market."

TOOL_SPECS = [
    {
        "name": "dice_roll",
        "description": "Roll dice.",
        "inputSchema": {"json": {"type": "object", "properties": {}}},
    }
]


def _model(recorder=None) -> NarratorOpenAIModel:
    return NarratorOpenAIModel(
        client_args={"base_url": "http://localhost:8000/v1", "api_key": "not-required"},
        model_id="google/gemma-4-26B-A4B-it",
        params={"temperature": 0.4, "max_tokens": 8192},
        recorder=recorder,
    )


def _messages(count: int) -> list[dict]:
    """``count`` turn messages, each carrying its own canon digest."""
    return [
        {
            "role": "user",
            "content": [{"text": turn_prompt(f"Rill: line {index}", canon=DIGEST)}],
        }
        for index in range(count)
    ]


def test_the_formatter_carries_the_canon_prefix_into_the_request_unchanged():
    """The measurement reads the span ``turn_prompt`` wrote, through the real formatter."""
    rows: list[dict] = []
    request = _model(rows.append).format_request(
        _messages(3), TOOL_SPECS, system_prompt="SYSTEM"
    )

    assert len(rows) == 1
    assert rows[0]["canon_blocks"] == 3
    assert rows[0]["canon_current_chars"] == len(DIGEST) + 2
    assert rows[0]["canon_superseded_chars"] == 2 * (len(DIGEST) + 2)
    assert rows[0]["system_chars"] == len("SYSTEM")
    assert rows[0]["message_count"] == 3
    assert rows[0]["tools_chars"] > 0
    assert measure_request(request) == rows[0]


def test_recording_a_request_changes_none_of_its_bytes():
    """A measurement that moved a byte would invalidate every run it measured."""
    rows: list[dict] = []
    recorded = _model(rows.append).format_request(
        _messages(2), TOOL_SPECS, system_prompt="SYSTEM"
    )
    plain = _model().format_request(_messages(2), TOOL_SPECS, system_prompt="SYSTEM")

    assert recorded == plain
    assert len(rows) == 1


def test_a_recorder_that_raises_never_costs_the_turn_its_request():
    """The measurement is instrumentation; a broken one must not stop a table."""

    def explode(_row: dict) -> None:
        raise RuntimeError("instrument fault")

    request = _model(explode).format_request(
        _messages(1), TOOL_SPECS, system_prompt="SYSTEM"
    )

    assert request["messages"][0]["role"] == "system"
    assert len(request["messages"]) == 2


def test_the_usage_reader_takes_this_turns_invocation_not_the_session_total():
    """The engine caches one agent per channel, so the lifetime accumulator is wrong.

    ``EventLoopMetrics.accumulated_usage`` is never reset, and a soak that summed it
    once per turn reported a triangular total. ``agent_invocations`` opens one entry per
    ``invoke_async`` call, and its last entry covers that call alone. This runs the real
    framework class against both fields rather than a double of this reading.
    """
    from strands.telemetry.metrics import EventLoopMetrics

    metrics = EventLoopMetrics()
    metrics.reset_usage_metrics()
    metrics.update_usage({"inputTokens": 100, "outputTokens": 10, "totalTokens": 110})
    metrics.reset_usage_metrics()
    metrics.update_usage({"inputTokens": 200, "outputTokens": 20, "totalTokens": 220})

    assert metrics.accumulated_usage["inputTokens"] == 300
    assert _turn_usage(SimpleNamespace(metrics=metrics))["inputTokens"] == 200
    assert _turn_usage(SimpleNamespace(metrics=metrics))["totalTokens"] == 220


def test_the_usage_reader_reports_nothing_rather_than_raising():
    """A framework rename must cost the measurement, never the turn."""
    assert _turn_usage(None) == {}
    assert _turn_usage(SimpleNamespace(metrics=None)) == {}
    assert _turn_usage(SimpleNamespace(metrics=SimpleNamespace(agent_invocations=[]))) == {}


def test_the_empty_tools_workaround_survives_the_recorder():
    """vLLM rejects ``tools: []``; the settler sends exactly that shape."""
    rows: list[dict] = []
    request = _model(rows.append).format_request([], [], system_prompt="SYSTEM")

    assert "tools" not in request
    assert rows[0]["tools_chars"] == 0
