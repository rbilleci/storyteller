"""The settle contract: the schema the settler must answer in, and its prompt.

The settle step closes the contradiction by making both answers executable. A commit
becomes a validated object the engine itself sends to ``scene_commit``; a waive becomes
an audited ``ledger_settle`` call that clears the debt with the model's reason. Either
way the turn ends settled, which is what makes cascades structurally impossible.

The schema is a discriminated union because a live probe measured what looser shapes
do: with two independent optional fields, the model set ``nothing_durable`` true and
filled a commit anyway. A union with a required discriminator makes that state
unrepresentable. The same probe measured the residual risk this file's prompt exists
to control: offered a forced schema with no ledger context, the model committed
confidently about pure banter. The prompt therefore transcribes the ledger's own debt
entries — each with the stake the model itself declared and which branch the dice
realized — so committing is copying, not inventing.

One residual is named rather than solved. The narration entering ``settle_prompt`` is
player-influenced, so a table could try to steer the settler through it — the model
holds no path to ``ledger_settle``, but the settler reads text the model wrote. The
blast radius is bounded on both branches: a coerced waive skips one scene entry while
the mechanics stand committed and every waived stake persists in the audit event, and
a coerced commit writes nothing the model could not already write through its own
``scene_commit``. The settle step hands the model zero new capability; it narrows what
silence can do, not what words can.

Validation constraints mirror the server's own guards (blank summaries and negative
time are rejected there too), so a schema-valid commit cannot bounce off the server
for a reason the schema could have caught.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from narrator.shape import example_json

SETTLER_SYSTEM_PROMPT = (
    "You settle a game master's ledger after a turn. You transcribe realized stakes "
    "into one durable scene entry, or you waive them on the record with a reason. "
    "You never invent events that the ledger and narration do not contain."
)


class SettleCommit(BaseModel):
    """Ratify the outstanding outcomes with one scene entry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    kind: Literal["commit"]


    public_summary: str = Field(min_length=1, max_length=600, description="One or two sentences of durable change, written from the realized stakes below.")
    #: The 128 visible-change items in the same corpus run 27 to 177 characters with
    #: a median of 66.5, so 240 is 1.36 times the longest observed; an earlier bound
    #: of 160 sat inside the observed range and would have grammar-truncated the two
    #: recorded items of 161 and 177 characters.
    visible_changes: list[Annotated[str, StringConstraints(max_length=240)]] = Field(
        default_factory=list, max_length=4, description="Player-visible changes, one clause each."
    )
    #: The integer is bounded too, because an audit proved an unbounded one defeats
    #: the whole paragraph: the grammar admitted a schema-legal 3,000-digit run, a
    #: digit-loop sibling of the whitespace loop. 10,080 minutes is 7 in-game days,
    #: 168 times the largest of the 15 nonzero deltas in the 121-commit corpus (60),
    #: and caps the field at 5 digits. With every field at its bound — both strings
    #: full, 4 full items, the 5-digit delta — the object serializes at 1,676
    #: characters by ``model_dump_json`` (compact separators), so a schema-legal
    #: object cannot exhaust the settler's 900-token budget; only a degenerate
    #: generation can, and the budget bounds what that costs.
    in_game_time_delta_minutes: int = Field(default=0, ge=0, le=10080)


class SettleWaive(BaseModel):
    """Clear the outstanding outcomes without a scene entry, on the record."""

    model_config = ConfigDict(str_strip_whitespace=True)

    kind: Literal["waive"]
    reason: str = Field(min_length=1, max_length=600, description="Why none of the outcomes below needs a scene entry.")


class SettleOutcome(BaseModel):
    """Exactly one of: a commit, or an audited waive."""

    outcome: SettleCommit | SettleWaive = Field(discriminator="kind")


def _realized_stake_text(debt: dict) -> str:
    """Derive the realized stake from a raw ledger entry.

    ``narrator.ledger.read_fiction_debt`` returns the dictionaries persisted in
    ``campaign/state.json``, so the computed ``realized_public_text`` property of the
    server-side model is not present; this reproduces it from the stored fields. The
    stake text is what the model itself declared when it called the tool, which is
    what makes the settle step transcription rather than recall.
    """
    stakes = debt.get("stakes") or {}
    realized = debt.get("realized", "none")
    if realized in ("success", "failure"):
        return str(stakes.get(realized, "") or "").strip()
    return ""


def settle_prompt(debts: list[dict], narration: str) -> str:
    """Render the ledger's outstanding entries into the settler's user message.

    Every line is transcription material the model already authored: the reason it
    gave when calling the tool, and the stake text it declared for the branch the
    dice then realized. The stake text is empty when no stake was declared for the
    realized branch, in which case the entry shows its bare outcome.
    """
    lines = ["The fiction-debt ledger holds these unratified outcomes:", ""]
    for debt in debts:
        realized = _realized_stake_text(debt)
        detail = realized if realized else f"outcome: {debt.get('outcome', 'unknown')}"
        lines.append(
            f"- seq {debt.get('seq')}: {debt.get('tool')} — {debt.get('reason', '')} → {detail}"
        )
    lines += [
        "",
        "The narration this turn ended on:",
        "",
        narration.strip() or "(the turn produced no deliverable narration)",
        "",
        "Settle the ledger. Choose commit when any outcome above should enter the "
        "durable scene record; write the summary from the realized stakes, not from "
        "imagination. Choose waive only when every outcome above is already covered "
        "by an earlier scene entry or genuinely changes nothing durable, and say why.",
        "",


        "Answer with one JSON object, every field present, in this exact order. "
        f'For a commit: {{"outcome": {example_json(SettleCommit)}}} — '
        "visible_changes may be an empty list and the minutes 0, but both fields "
        f'must appear. For a waive: {{"outcome": {example_json(SettleWaive)}}}.',
    ]
    return "\n".join(lines)
