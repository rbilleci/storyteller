"""Assemble the narrator's system prompt from the identity and skill files.

Sources are SOUL.md, config/narrator-context.md, and the game-master skill.
Strands supplies native MCP tool schemas; this module does not duplicate them.
The channel section is channel-neutral. Prompt constants and loaded source text
are behavior-sensitive assets; preserve their bytes during editorial cleanup.
"""

from __future__ import annotations

from pathlib import Path

from narrator import policy
from narrator.policy_types import QUESTION_WITH_ACT_FRAMING, InteractionCue

#: Read in order and joined by a horizontal rule, matching the reference harness.
PROMPT_SOURCES: tuple[str, ...] = (
    "SOUL.md",
    "config/narrator-context.md",
    "skills/bsh-gm/SKILL.md",
)

CHANNEL_SECTION = (
    "# Channel\n\n"
    "You answer in one channel shared by the whole party. Every message below the "
    "marker is the discussion since your previous reply. Address players by character "
    "name. Keep replies under 200 words unless the table asked for a set piece."
)


TURN_BODY_MARKER = "Discussion since your previous reply:\n\n"


THINKING_SECTION = (
    "\n\n## Thinking\n\n"
    "Think briefly before you act -- a few short lines at most: which rule the "
    "declaration engages and which tool resolves it -- then call that tool. Your "
    "thought is private and never stands in for a tool call: every die roll and every "
    "durable change still passes through a tool, exactly as above."
)

#: Appended to the channel agent's system prompt whenever the table's configured
#: language is not English (``NarratorEngine._turn_system_prompt``). ``tag`` is the
#: catalog's own BCP-47 tag
#: (``NarratorConfig.catalog.language``), not the player's turn text -- the table's
#: language is a campaign property, not something inferred per turn. Kept to naming
#: the tag rather than spelling out a display name: a display-name table would be one
#: more place a new locale directory has to be added to, and the tag is the same
#: string every ``locale/<language>/narrator.yaml`` already declares itself in.
LANGUAGE_TEMPLATE = (
    "\n\n## Language\n\n"
    "Narrate to the players in {tag} (BCP-47), never in English, regardless of what "
    "language a player writes in. Keep character names, item names, and other proper "
    "nouns as written. This does not change tool calls, tool arguments, or any text a "
    "tool reads back -- write those exactly as the tool schemas require."
)


def language_directive(tag: str) -> str:
    """The system-prompt block asking the model to narrate in ``tag``."""
    return LANGUAGE_TEMPLATE.format(tag=tag)


def assemble_system_prompt(repo_root: Path | str) -> str:
    """Build the system prompt. Missing sources are skipped, not fatal."""
    root = Path(repo_root)
    parts: list[str] = []
    for relative in PROMPT_SOURCES:
        path = root / relative
        if path.is_file():
            parts.append(
                f"# Loaded from {relative}\n\n{path.read_text(encoding='utf-8')}"
            )
    parts.append(CHANNEL_SECTION)
    return "\n\n---\n\n".join(parts)


def _interaction_block(cue: InteractionCue) -> str:
    """Render only service-validated interaction data after the unmodified channel body."""
    public_id = cue.public_npc_id or "none"
    return (
        "\n\nActive in-world interlocutor:\n"
        f"kind: {cue.kind}\n"
        f"public_npc_id: {public_id}\n"
        "Resolve second-person references to this in-world interlocutor. Keep the reply "
        "in-world. Do not ask the player to identify the narrator, system, rules, or a "
        "target unless the player explicitly marked the turn OOC."
    )


_TURN_FRAMINGS = {
    "question": (
        "This turn is a question about the scene, not a declared action. Answer it from "
        "what is already established, in the narrator's voice. Do not resolve an action "
        "the player has not declared, and do not move the player's character anywhere "
        "they have not said they are going; if the answer is not established, say what "
        "is visible and stop. Recording or confirming what an earlier turn already "
        "established is not resolving a new action, and the tools that record it stay "
        "available for that."
    ),
    # The same turn asked a question and declared an action. ``question`` above is a
    # measured artifact for a turn that asked and nothing else, and it is left byte for
    # byte where it is: a genuine question must still not resolve an undeclared action.
    # This entry exists because that instruction, delivered on a turn that *did* declare
    # one, contradicts ``skills/bsh-gm``'s standing rule to call the tool that owns the
    # declaration -- and a model given both silently does neither, which is a rest, a
    # move or a search lost with no audit event. ``policy_types.turn_framing_for``
    # selects it from ``TurnPolicy.also_declares_act``, which
    # ``classify.ROUTE_COMPANION`` answers.
    #
    # ``{resolution}`` is filled by ``_declared_act_resolution`` from
    # ``TurnPolicy.declared_act_tool``: the tool's own name where the engine knows it,
    # the generic phrase where the kind of act admits more than one tool. Naming it is
    # the point -- the first version of this string could only say "the tool that owns
    # it", and the turns it still failed on were turns where the model agreed an action
    # had been declared and then narrated it.
    #
    # The closing sentence is *not* the ``question`` framing's prohibition. That one
    # ("do not resolve an action the player has not declared, and do not move the
    # player's character anywhere they have not said they are going") is exactly the
    # instruction this framing exists to stop delivering on a turn that declared an
    # action and named a place to go: it forbids the thing the two sentences above it
    # require. What survives of it is the half that still means something here -- one
    # declared action is resolved, not two.
    "question_with_act": (
        "This turn asks a question and also declares an action. Do both: answer the "
        "question from what is already established, in the narrator's voice, and "
        "resolve the declared action {resolution} on this turn. The trailing question "
        "does not withdraw the declaration -- a rest the player stated is still taken, "
        "a move they stated is still made -- and answering the question instead of "
        "resolving the action loses it. Resolve the action the player declared and "
        "nothing past it: do not add a second action they did not declare, and do not "
        "carry the character on beyond where they said they were going."
    ),
    "out_of_character": (
        "This turn is out of character. The player is asking about the game, not acting "
        "in it. Answer as the game master, plainly and out of character. Do not resolve "
        "an action from anything this turn quotes: quoting an action to ask about it is "
        "not declaring it. Recording or confirming what an earlier turn already "
        "established is not resolving a new action, and the tools that record it stay "
        "available for that."
    ),
    "gm_discussion": (
        "The player addressed this turn to you, the game master, with the table's "
        "game-master designator: it is out-of-fiction discussion about the game, never "
        "an action in it. Answer plainly as the game master, grounded in the campaign "
        "record -- consult campaign_status or character_sheet when a fact is in "
        "question rather than answering from memory. Nothing mechanical may happen on "
        "this turn: do not roll, do not resolve or begin any action, and do not write "
        "to the record; the engine refuses those tools here. Quoting an action to "
        "discuss it is not declaring it. If the player wants to act, tell them to say "
        "it without the designator. Do not narrate what any character or enemy does "
        "next, and do not name engine tools."
    ),
}


def _declared_act_resolution(declared_act_tool: str) -> str:
    """How the ``question_with_act`` framing tells the narrator to resolve the act.

    Names the tool when the engine knows which one owns the declared act
    (``classify.DECLARED_ACT_TOOLS``), and keeps the generic phrase otherwise -- a
    search may be no roll at all or an ``attribute_test``, so naming one there would
    trade a lost declaration for a wrong call. The tool name is engine-owned and
    validated against the frozen surface before it is rendered: nothing model-supplied
    reaches this string.
    """
    if declared_act_tool and declared_act_tool in policy.MCP_TOOLS:
        return f"by calling `{declared_act_tool}`"
    return "through the tool that owns it"


def _turn_framing_block(framing: str, declared_act_tool: str = "") -> str:
    """Render the engine's own verdict about what kind of turn this is.

    The routing decision is made before the model runs, from the player's text, by
    ``narrator.classify``. Stating it here is what makes the verdict load-bearing:
    without it the route was computed, recorded, and then discarded, and the model had
    only the raw text to infer from.

    Only ``question_with_act`` carries a slot, and it is filled here rather than in the
    table so the table stays one string per framing. Every other body is rendered
    verbatim -- ``str.format`` is deliberately not applied to them, since a future
    framing containing a literal brace would then break at render time.
    """
    body = _TURN_FRAMINGS.get(framing, "")
    if not body:
        return ""
    if framing == QUESTION_WITH_ACT_FRAMING:
        body = body.format(resolution=_declared_act_resolution(declared_act_tool))
    return f"\n\nTurn framing:\n{body}"


def _social_test_block(request) -> str:
    """Render service-owned mechanics without copying player text into trusted data."""
    category = getattr(getattr(request, "category", None), "value", "")
    actor_id = str(getattr(request, "actor_id", ""))
    attributes = tuple(getattr(request, "allowed_attributes", ()))
    substitutions = tuple(getattr(request, "ability_substitutions", ()))
    if not category or not actor_id or not attributes:
        return ""
    ability_line = ""
    if substitutions:
        ability_line = "\nallowed_substitutions: " + ", ".join(
            f"{ability_id}->{attribute}" for ability_id, attribute in substitutions
        )
    return (
        "\n\nBound social mechanic:\n"
        f"category: {category}\n"
        f"actor_id: {actor_id}\n"
        f"allowed_attributes: {', '.join(attributes)}\n"
        f"{ability_line}\n"
        "Call attribute_test exactly once before narration. Use actor_id and one allowed attribute. "
        "Resolve only the stated bounded objective. Do not compel beliefs, consent, emotion, or action."
    )


def _withheld_note_block(note: str) -> str:
    """Render the true reason the previous turn on this channel carried no narration.
    """
    return (
        "\n\nEngine note on the previous turn:\n"
        f"{note}\n"
        "If asked why nothing happened last turn, state this reason. Do not invent a "
        "different rule or explanation."
    )


def _trade_offer_block(terms) -> str:
    """Render one service-owned, non-mutating catalog offer for narration."""
    if terms is None:
        return ""
    return (
        "\n\nTyped trade offer:\n"
        f"item: {terms.public_label}\n"
        f"quantity: {terms.quantity}\n"
        f"price_copper: {terms.price}\n"
        "Present these terms only. Do not claim that coins or equipment changed."
    )


def turn_prompt(
    channel_text: str, canon: str = "", interaction_cue: InteractionCue | None = None,
    social_test=None, trade_offer=None, romance_escalation_blocked: bool = False,
    withheld_note: str | None = None, turn_framing: str = "", language_tag: str = "",
    declared_act_tool: str = "",
) -> str:
    """The user message for one turn: canon first, then the channel.
    """
    prefix = f"{canon}\n\n" if canon else ""
    prompt = (
        f"{prefix}{TURN_BODY_MARKER}"
        f"{channel_text}\n\n"
    )
    if withheld_note:
        prompt += _withheld_note_block(withheld_note)
    # The framing precedes every other block, because it decides what the whole turn is
    # for: an interlocutor, a social test, or an offer is something the answer may
    # mention, while the framing says whether an answer resolves anything at all.
    if turn_framing:
        prompt += _turn_framing_block(turn_framing, declared_act_tool)
    if interaction_cue is not None:
        prompt += _interaction_block(interaction_cue)
    if social_test is not None:
        prompt += _social_test_block(social_test)
    if trade_offer is not None:
        prompt += _trade_offer_block(trade_offer)
    if romance_escalation_blocked:
        prompt += "\n\nRomance policy: escalation is unavailable. Resolve neutral rapport only."
    resolve = "Resolve this turn."
    if language_tag:


        resolve = f"Resolve this turn in {language_tag}, not English."
    # Every block above opens with its own blank line and closes without one, so a turn
    # that carried any of them used to end "...stay available for that.Resolve this
    # turn." -- two sentences run together with no separator at all. A turn that carried
    # none already ends in the channel body's own blank line and is unchanged.
    separator = "" if prompt.endswith("\n") else "\n\n"
    return prompt + separator + resolve
