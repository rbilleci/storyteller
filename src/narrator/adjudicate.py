"""The adjudicate step: resolve a pending mechanical ruling from the narration.

When a tool leaves a ``pending_ruling`` in campaign state — a mechanical obligation
whose target is a fictional choice, such as which possession a demon steals — the
engine runs one structured model request that picks the option the narration already
implied, then applies it through the engine-only ``ability_apply_ruling`` tool.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ADJUDICATOR_SYSTEM_PROMPT = (
    "You resolve one game-master ruling. The narration below has already happened. "
    "Choose the single option from the list that the narration states or most fits it. "
    "Answer with exactly one option, copied verbatim from the list."
)


class AdjudicateOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: str = Field(
        description="Exactly one option from the provided list, copied verbatim.",
    )


def adjudicate_prompt(ruling: dict, narration: str) -> str:
    """Build the transcribe-first adjudication prompt from one ruling and the turn."""
    options = "\n".join(f"- {option}" for option in ruling.get("options", []))
    return (
        f"Ruling: {ruling.get('question', 'Choose one option.')}\n\n"
        f"Options:\n{options}\n\n"
        f"The turn's narration:\n{narration.strip()}\n\n"
        "Reply with the one option the narration names, or the most fitting option."
    )
