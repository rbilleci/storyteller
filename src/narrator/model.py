"""The one Strands model subclass this project needs, and why it exists.

vLLM v0.26.0 rejects a chat completion whose ``tools`` field is an empty array, with
HTTP 400: \"`tools` must not be an empty array. Either provide at least one tool or omit
the field entirely.\" The real OpenAI API tolerates the empty array, so Strands 1.50.2
never noticed: its ``OpenAIModel.structured_output`` builds the request without tool
specs and ships ``tools: []`` verbatim. Every structured-output call from a tool-less
agent therefore fails against vLLM and succeeds against OpenAI.

The channel agent carries the model-facing tools and never triggers the bug; it uses this class
anyway so the project holds exactly one model construction path.

The class also carries one optional request hook. ``request_overrides`` is a callable
the engine supplies for the channel agent alone; whatever mapping it returns is written
onto the assembled request just before it leaves, and it is consulted on every request
rather than once at construction, so a setting a player changes mid-session -- the
thinking level ``NarratorEngine.set_turn_thinking_level`` holds -- reaches the next
request without rebuilding the agent and losing its conversation. Strands merges
``params`` into the request the same way, but ``params`` is read into ``self.config`` at
construction; this hook is the late-bound counterpart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from strands.models.openai import OpenAIModel

from narrator.payload import measure_request


def _without_reasoning_content(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Return ``messages`` with every ``reasoningContent`` content block dropped.

    A message with no ``reasoningContent`` block returns unchanged rather than
    rebuilt, so a caller comparing by identity sees no difference for the common case.
    """
    filtered_messages = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not content:
            filtered_messages.append(message)
            continue
        filtered_content = [block for block in content if "reasoningContent" not in block]
        if len(filtered_content) == len(content):
            filtered_messages.append(message)
        else:
            filtered_messages.append({**message, "content": filtered_content})
    return filtered_messages


class NarratorOpenAIModel(OpenAIModel):
    """``OpenAIModel`` that omits an empty ``tools`` array instead of sending it."""

    def __init__(
        self,
        *args,
        recorder: Callable[[dict], None] | None = None,
        usage_recorder: Callable[[dict], None] | None = None,
        request_overrides: Callable[[], Mapping[str, Any]] | None = None,
        **kwargs,
    ):
        """``recorder`` takes one payload row per request; ``usage_recorder`` its usage.

        ``request_overrides`` returns the keys to write onto each outgoing request
        (see the module docstring); ``None`` or an empty mapping leaves it untouched.

        All keywords are consumed here rather than passed on. ``OpenAIModel`` collects
        every unknown keyword into ``self.config`` and warns about it, and
        ``format_request`` reads only the keys it knows, so an unconsumed keyword would
        sit in the configuration as a warned-about stray rather than reach the endpoint.
        Consuming them keeps the configuration equal to the model's own.
        """
        super().__init__(*args, **kwargs)
        self._recorder = recorder
        self._usage_recorder = usage_recorder
        self._request_overrides = request_overrides

    def format_request(self, *args, **kwargs):
        request = super().format_request(*args, **kwargs)
        if not request.get("tools"):
            request.pop("tools", None)
            request.pop("tool_choice", None)
        if self._request_overrides is not None:
            for key, value in dict(self._request_overrides() or {}).items():
                request[key] = value
        if self._recorder is not None:
            try:
                self._recorder(measure_request(request))
            except Exception:  # noqa: BLE001 - a measurement never fails a turn
                pass
        return request

    async def stream(self, *args, **kwargs):
        """Strip retained reasoning fields, pass every event through, and record usage.

        ``OpenAIModel.stream`` ends each request by yielding one ``metadata`` event
        carrying that request's usage, and ``format_chunk`` fills
        ``cacheReadInputTokens`` from vLLM's ``prompt_tokens_details.cached_tokens``.
        Reading it here pairs one usage record with one assembled request, which the
        agent-level accumulator cannot do: it sums a turn's model calls into one figure.

        Pairing rests on one ordering fact rather than on a correlation identifier.
        ``format_request`` runs inside this call, before the request leaves, and the
        usage event arrives before this call returns, so a caller that never runs two
        requests concurrently on one model instance sees strict alternation. The engine
        builds one model per channel agent and one per settle or sweep call, and drives
        each sequentially.

        ``cacheReadInputTokens`` is absent when the endpoint reports zero cached tokens
        and also when it reports no detail at all, because Strands writes the key only
        for a truthy count. The consumer therefore counts requests carrying the key
        rather than reading its absence as zero reuse.
        """
        if args:
            messages, *rest = args
            args = (_without_reasoning_content(messages), *rest)
        elif "messages" in kwargs:
            kwargs = {**kwargs, "messages": _without_reasoning_content(kwargs["messages"])}
        # Generated-reasoning accounting, in characters, client-side by necessity: the
        # served vLLM reports no completion token detail, so the only observer of how
        # much this request actually thought is the reasoning delta stream itself.
        # The count rides the usage record as ``reasoningOutputChars`` -- an integer,
        # so the engine's usage filter keeps it -- pairing it to the same request row
        # the usage already lands on. It measures the step-level replay claim: a later
        # step that continues a replayed thought should generate fewer of these than
        # one re-deriving from scratch.
        reasoning_output_chars = 0
        async for event in super().stream(*args, **kwargs):
            if isinstance(event, dict):
                try:
                    delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
                    text = (delta.get("reasoningContent") or {}).get("text")
                    if isinstance(text, str):
                        reasoning_output_chars += len(text)
                except Exception:  # noqa: BLE001 - a measurement never fails a turn
                    pass
            if self._usage_recorder is not None and isinstance(event, dict):
                try:
                    usage = (event.get("metadata") or {}).get("usage")
                    if isinstance(usage, dict):
                        self._usage_recorder(
                            {**usage, "reasoningOutputChars": reasoning_output_chars}
                        )
                except Exception:  # noqa: BLE001 - a measurement never fails a turn
                    pass
            yield event
