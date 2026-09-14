"""The single result envelope every mechanics tool returns.

The narrator reads `narration_facts` and may state those facts in prose. It may
not state any mechanical fact that does not appear in a tool result.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ToolEnvelopeSuccess(TypedDict):
    """The base shape every ``results.success()`` call produces.

    ``event_id``, ``roll``, and ``outcome`` are set only when the caller passes
    ``sequence``, ``roll``, or ``outcome`` (respectively) to ``success()``, so they
    are ``NotRequired`` here. Every tool-specific field a given MCP tool adds
    through ``success()``'s ``**extra`` belongs on that tool's own subclass in
    ``service.py``, not here: ``**extra`` is open-ended, so it cannot be captured
    by one shared TypedDict without losing the per-tool precision this envelope
    exists to add.
    """

    ok: bool
    summary: str
    state_changes: list[str]
    narration_facts: list[str]
    warnings: list[str]
    event_id: NotRequired[str]
    roll: NotRequired[dict]
    outcome: NotRequired[str]


class ToolEnvelopeFailure(TypedDict):
    """The complete shape every ``results.failure()`` call produces."""

    ok: bool
    error: str
    message: str
    allowed_next_steps: list[str]


def event_id(sequence: int | None) -> str | None:
    if sequence is None:
        return None
    return f"evt-{sequence:06d}"


def success(
    summary: str,
    *,
    sequence: int | None = None,
    roll: dict | None = None,
    outcome: str | None = None,
    state_changes: list[str] | None = None,
    narration_facts: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "summary": summary,
        "state_changes": state_changes or [],
        "narration_facts": narration_facts or [],
        "warnings": warnings or [],
    }
    identifier = event_id(sequence)
    if identifier:
        payload["event_id"] = identifier
    if roll is not None:
        payload["roll"] = roll
    if outcome is not None:
        payload["outcome"] = outcome
    payload.update(extra)
    return payload


def failure(code: str, message: str, allowed_next_steps: list[str] | None = None) -> ToolEnvelopeFailure:
    return {
        "ok": False,
        "error": code,
        "message": message,
        "allowed_next_steps": allowed_next_steps or [],
    }
