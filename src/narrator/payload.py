"""What one assembled request carries, and what it stops carrying.

This module owns both readings of one span: the measurement that prices every canon
digest in a request, and the removal that leaves exactly one. Keeping them together
makes them agree by construction, because a drift between the two would report a
reclaimed payload the request still carries.

This module answers that by measuring the request the model layer is about to send,
after ``NarratorOpenAIModel.format_request`` has assembled it. Measuring there rather
than over ``agent.messages`` records what the endpoint receives, including the tool
schemas and the system prompt, so the digest's share is a share of the real payload
rather than of the history alone.

The canon bytes are locatable because the two writers publish their own markers.
``canon.DIGEST_HEADING`` opens every digest and ``prompt.TURN_BODY_MARKER`` opens the
channel body that follows it, so a digest's span inside a stored user message is the
exact span ``turn_prompt`` created. The last such span is the current turn's digest;
every earlier one is superseded by construction, because the current digest renders the
same six blocks (session-zero cue, location, NPCs, exits, scene, party resources —
``canon.render_digest``) at their latest state.

``strip_superseded_canon`` removes those earlier spans from the agent's own message
history. The engine calls it before it appends the current turn, so every digest the
history holds at that moment is superseded and no comparison is needed. Removing them
loses no canon: the current digest renders the same six blocks at their latest
state. Retention costs more than characters, because a superseded digest states clock
fills, a summary, and a location file that later commits changed, no label marks any of
it historical, and the model therefore reads stale canon beside current canon with
nothing distinguishing the two.

The module imports nothing outside the standard library and the two marker constants,
so the main test environment pins it without Strands.
"""

from __future__ import annotations

import json
from typing import Any

from narrator.canon import DIGEST_HEADING
from narrator.prompt import TURN_BODY_MARKER


def _content_text(content: Any) -> str:
    """Every character of model-visible text in one formatted message's content.

    The OpenAI shape carries three forms: a plain string for system and tool messages,
    a list of typed blocks for user messages, and ``None`` for an assistant message
    that only calls tools. Each form contributes its text and nothing else.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _tool_call_chars(message: dict) -> int:
    """The characters an assistant message spends on tool calls.

    A tool call carries a name and a JSON argument string, and both ride in the request
    body. Counting them keeps ``messages_chars`` a true sum over the message list rather
    than a sum over prose only.
    """
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return 0
    total = 0
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            for key in ("name", "arguments"):
                value = function.get(key)
                if isinstance(value, str):
                    total += len(value)
    return total


def _canon_spans(text: str) -> list[int]:
    """The character length of every canon digest this text carries, in order.

    A digest runs from its heading to the channel body that ``turn_prompt`` writes
    after it, so the span includes the blank line separating the two. A heading with no
    body marker after it runs to the end of the text, which is the shape a truncated or
    hand-built message would take.
    """
    spans: list[int] = []
    start = text.find(DIGEST_HEADING)
    while start != -1:
        body = text.find(TURN_BODY_MARKER, start)
        end = len(text) if body == -1 else body
        spans.append(end - start)
        start = text.find(DIGEST_HEADING, end)
    return spans


def _canon_prefix_end(text: str) -> int:
    """Where a stored turn message's canon prefix ends, or 0 when it carries none.

    The cut point is conservative in one direction on purpose. The text must open with
    the digest heading, so a message the engine did not build with a digest is never
    touched. The end is the first channel-body marker, which lies at or before the
    boundary ``turn_prompt`` created, because the canon block precedes the body. A
    scene record that somehow contained the marker therefore leaves a digest tail in
    place; it can never take a character of player text.
    """
    if not text.startswith(DIGEST_HEADING):
        return 0
    body = text.find(TURN_BODY_MARKER)
    return 0 if body == -1 else body


def strip_superseded_canon(messages: list) -> int:
    """Drop the canon prefix from every message in the history, and count the drops.

    The caller runs this before appending the current turn, so every digest present is
    superseded. A stripped message equals the prompt ``turn_prompt`` builds with no
    canon at all, which is the shape this project shipped before the digest existed.

    Three properties hold by construction. Only a user message's text blocks change, so
    a tool-use and tool-result pair keeps both halves and the conversation manager's
    trim points stay valid. A message carrying no digest keeps every byte. Running this
    twice changes nothing the first run left, because a stripped message no longer
    opens with the heading.
    """
    stripped = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            end = _canon_prefix_end(text)
            if end:
                block["text"] = text[end:]
                stripped += 1
    return stripped


def measure_request(request: dict) -> dict:
    """One row of payload composition for one assembled request.

    ``total_chars`` is the serialised body, so it is the number the endpoint reads and
    the number a cache miss recomputes. The three components below it — system prompt,
    tool schemas, message list — do not sum to it, because JSON syntax, role labels,
    tool-call identifiers, and the sampling parameters ride between them. The report
    carries both, so a reader can price the fixed prefix against the variable history
    without inferring either.
    """
    messages = request.get("messages")
    messages = messages if isinstance(messages, list) else []

    system_chars = 0
    messages_chars = 0
    message_count = 0
    canon_lengths: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = _content_text(message.get("content"))
        if message.get("role") == "system":
            system_chars += len(text)
            continue
        message_count += 1
        messages_chars += len(text) + _tool_call_chars(message)
        canon_lengths.extend(_canon_spans(text))

    tools = request.get("tools")
    tools_chars = (
        len(json.dumps(tools, ensure_ascii=False)) if isinstance(tools, list) else 0
    )

    return {
        "total_chars": len(json.dumps(request, ensure_ascii=False, default=str)),
        "system_chars": system_chars,
        "tools_chars": tools_chars,
        "messages_chars": messages_chars,
        "message_count": message_count,
        "canon_blocks": len(canon_lengths),
        "canon_current_chars": canon_lengths[-1] if canon_lengths else 0,
        "canon_superseded_chars": sum(canon_lengths[:-1]),
    }
