"""The last test in this file proves the mechanism against the real library code rather
than a double of it: it shows the warning firing on the unstripped input first, so a
regression that quietly dropped the strip would fail this file rather than pass it
vacuously.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from strands.models.openai import OpenAIModel  # noqa: E402

from narrator.model import NarratorOpenAIModel, _without_reasoning_content  # noqa: E402

#: One assistant turn shaped the way Strands appends a streamed reasoning delta: a
#: ``reasoningContent`` block ahead of the spoken text, retained verbatim in history.
_HISTORY_WITH_REASONING = [
    {
        "role": "assistant",
        "content": [
            {"reasoningContent": {"reasoningText": {"text": "the player wants to flee"}}},
            {"text": "You pivot and bolt for the door."},
        ],
    },
    {"role": "user", "content": [{"text": "roll dex"}]},
]


def _model(**kwargs) -> NarratorOpenAIModel:
    return NarratorOpenAIModel(
        client_args={"base_url": "http://localhost:8000/v1", "api_key": "not-required"},
        model_id="google/gemma-4-26B-A4B-it",
        **kwargs,
    )


def test_without_reasoning_content_drops_only_the_reasoning_block():
    stripped = _without_reasoning_content(_HISTORY_WITH_REASONING)

    assert stripped[0]["content"] == [{"text": "You pivot and bolt for the door."}]
    # The untouched message is unaffected -- same object, not just equal content.
    assert stripped[1] is _HISTORY_WITH_REASONING[1]
    # The source list is never mutated in place.
    assert len(_HISTORY_WITH_REASONING[0]["content"]) == 2


def test_without_reasoning_content_leaves_a_reasoning_free_history_untouched():
    plain_history = [{"role": "user", "content": [{"text": "hello"}]}]

    assert _without_reasoning_content(plain_history) == plain_history


def test_without_reasoning_content_tolerates_a_message_with_no_content():
    messages = [{"role": "user", "content": []}, {"role": "assistant"}]

    assert _without_reasoning_content(messages) == messages


async def test_stream_forwards_the_stripped_history_to_the_base_model(monkeypatch):
    """``NarratorOpenAIModel.stream`` strips before ``super().stream`` ever runs.

    This drives the real production entry point: the event loop calls
    ``model.stream(messages, tool_specs, system_prompt, ...)`` positionally
    (``strands/event_loop/streaming.py``), so this test calls it the same way rather
    than through a keyword-only double.
    """
    captured: dict = {}

    async def fake_base_stream(self, messages, *args, **kwargs):
        captured["messages"] = messages
        yield {"chunk_type": "message_start"}
        yield {"metadata": {"usage": {"totalTokens": 12}}}

    monkeypatch.setattr(OpenAIModel, "stream", fake_base_stream)

    usage_rows: list[dict] = []
    model = _model(usage_recorder=usage_rows.append)

    events = [
        event
        async for event in model.stream(_HISTORY_WITH_REASONING, None, "system prompt")
    ]

    forwarded = captured["messages"]
    assert all("reasoningContent" not in block for block in forwarded[0]["content"])
    assert forwarded[1] is _HISTORY_WITH_REASONING[1]
    # Every event still passes through unchanged, and the usage recorder still fires --
    # the strip must cost this method none of its existing behaviour.
    assert events == [
        {"chunk_type": "message_start"},
        {"metadata": {"usage": {"totalTokens": 12}}},
    ]
    # The usage record carries the client-side generated-reasoning count -- zero
    # here, because this stream yielded no reasoning deltas.
    assert usage_rows == [{"totalTokens": 12, "reasoningOutputChars": 0}]


async def test_stream_counts_generated_reasoning_into_the_usage_record(monkeypatch):
    """The client-side reasoning count: streamed deltas summed, paired to the usage.

    The served vLLM reports no completion token detail, so this count is the only
    observer of how much a request thought -- the step-level reading the replay
    A/B compares between arms.
    """

    async def fake_base_stream(self, messages, *args, **kwargs):
        yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "abc"}}}}
        yield {"contentBlockDelta": {"delta": {"text": "spoken"}}}
        yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "de"}}}}
        yield {"metadata": {"usage": {"totalTokens": 9}}}

    monkeypatch.setattr(OpenAIModel, "stream", fake_base_stream)

    usage_rows: list[dict] = []
    model = _model(usage_recorder=usage_rows.append)
    async for _ in model.stream(_HISTORY_WITH_REASONING, None, "gm"):
        pass

    assert usage_rows == [{"totalTokens": 9, "reasoningOutputChars": 5}]


async def test_stream_strips_a_history_passed_by_keyword_too(monkeypatch):
    captured: dict = {}

    async def fake_base_stream(self, **kwargs):
        captured["messages"] = kwargs["messages"]
        yield {"chunk_type": "message_start"}

    monkeypatch.setattr(OpenAIModel, "stream", fake_base_stream)

    model = _model()
    events = [
        event
        async for event in model.stream(messages=_HISTORY_WITH_REASONING, tool_specs=None)
    ]

    assert all("reasoningContent" not in block for block in captured["messages"][0]["content"])
    assert events == [{"chunk_type": "message_start"}]


def test_the_unstripped_history_actually_triggers_the_librarys_own_warning(caplog):
    """Sabotage check: prove the failure mode exists before proving the fix removes it.
    """
    with caplog.at_level(logging.WARNING, logger="strands.models.openai"):
        OpenAIModel._format_regular_messages(_HISTORY_WITH_REASONING)

    assert any(
        "reasoningContent is not supported" in record.message for record in caplog.records
    )

    caplog.clear()
    stripped = _without_reasoning_content(_HISTORY_WITH_REASONING)
    with caplog.at_level(logging.WARNING, logger="strands.models.openai"):
        OpenAIModel._format_regular_messages(stripped)

    assert not any(
        "reasoningContent is not supported" in record.message for record in caplog.records
    )
