"""Pin ``turn_prompt``'s rendering of the withheld-turn note.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.prompt import turn_prompt  # noqa: E402


def test_no_withheld_note_reproduces_the_prompt_byte_for_byte():
    """An absent note must change nothing, matching every prior recorded prompt."""
    with_default = turn_prompt("Rill: I look around.")
    with_explicit_none = turn_prompt("Rill: I look around.", withheld_note=None)
    with_empty = turn_prompt("Rill: I look around.", withheld_note="")
    assert with_default == with_explicit_none == with_empty
    assert "Engine note" not in with_default


def test_a_withheld_note_appears_verbatim_and_names_the_reason():
    reason = "No action occurred because this hazard requires an authenticated confirmation."
    prompt = turn_prompt("Rill: GM, why was I refused?", withheld_note=reason)
    assert reason in prompt
    assert "Engine note on the previous turn:" in prompt
    assert "Do not invent a different rule" in prompt


def test_the_withheld_note_precedes_the_resolve_instruction():
    """The note must land inside the assembled prompt, not appended after the sign-off."""
    reason = "No action occurred because the proposed action was declined."
    prompt = turn_prompt("Rill: why?", withheld_note=reason)
    assert prompt.index(reason) < prompt.rindex("Resolve this turn.")


def test_the_declared_act_framing_asks_for_both_the_answer_and_the_tool_call():
    """The ``read`` route's framing, for a turn that also declared an action.

    ``question`` tells the model not to resolve an action the player has not declared,
    which is right for a question and wrong for "we take a short rest ... does anything
    find us?" -- the player declared one. This framing must ask for both halves, and it
    must not be reachable from a route alone: ``turn_framing_for`` selects it only when
    the route companion reported the declaration.
    """
    from narrator.policy_types import turn_framing_for

    assert turn_framing_for("read") == "question"
    assert turn_framing_for("read", True) == "question_with_act"
    # The routes that answer in the game master's voice keep their own framings: those
    # strings are measured artifacts and a declared act does not rewrite them.
    assert turn_framing_for("out_of_character", True) == "out_of_character"
    assert turn_framing_for("gm", True) == "gm_discussion"
    assert turn_framing_for("planner", True) == ""

    framed = turn_prompt(
        "Rill: we take a short rest and watch the water. Does anything find us?",
        turn_framing=turn_framing_for("read", True),
    )
    assert "Turn framing:" in framed
    assert "asks a question and also declares an action" in framed
    assert "resolve the declared action through the tool that owns it" in framed
    assert "does not withdraw the declaration" in framed
    # What survives of the ``question`` framing's prohibition: one declared action is
    # resolved, not two. The clause it replaced ("do not resolve an action the player
    # has not declared, and do not move the player's character anywhere they have not
    # said they are going") forbade, on this framing, the very thing the two sentences
    # above it require -- it is the last echo of the defect this framing exists to fix.
    assert "do not add a second action they did not declare" in framed
    assert "Do not resolve any further action the player has not declared" not in framed
    assert "anywhere they have not said they are going" not in framed

    # The measured ``question`` string is untouched and still selected for a turn that
    # only asks -- including that prohibition, which is right for a turn that only asks.
    question = turn_prompt('Rill: Can you see a doorway?', turn_framing=turn_framing_for("read"))
    assert "This turn is a question about the scene, not a declared action." in question
    assert "anywhere they have not said they are going" in question
    assert framed != question


def test_the_declared_act_framing_names_the_tool_the_engine_knows_and_no_other():
    """``declared_act_tool`` fills one blank in the framing above, and only that one.
    """
    from narrator.policy_types import turn_framing_for

    framing = turn_framing_for("read", True)
    rest = turn_prompt("Rill: we take a short rest. Anything out there?",
                       turn_framing=framing, declared_act_tool="rest")
    assert "resolve the declared action by calling `rest` on this turn" in rest
    assert "the tool that owns it" not in rest

    moving = turn_prompt("Rill: we head for the quay. How long does it take?",
                         turn_framing=framing, declared_act_tool="scene_commit")
    assert "by calling `scene_commit`" in moving

    # No tool, or a name outside the frozen surface: the generic wording stands. The
    # second case is the guard, not a scenario -- the name is engine-owned, and this
    # keeps it structurally impossible for anything else to be rendered as a tool.
    generic = turn_prompt("Rill: I keep watch. Anything out there?", turn_framing=framing)
    assert "resolve the declared action through the tool that owns it" in generic
    assert turn_prompt("Rill: I keep watch.", turn_framing=framing,
                       declared_act_tool="rm -rf /") == turn_prompt(
        "Rill: I keep watch.", turn_framing=framing)

    # Every other framing ignores the argument entirely, so a stray value cannot move
    # a measured string.
    for other in ("question", "out_of_character", "gm_discussion"):
        assert turn_prompt("Rill: x", turn_framing=other, declared_act_tool="rest") == (
            turn_prompt("Rill: x", turn_framing=other)
        )


def test_the_resolve_instruction_is_separated_from_the_block_above_it():
    """Every block opens with a blank line and closes without one, so the sign-off used
    to run straight into the last sentence: "...stay available for that.Resolve this
    turn." A turn carrying no block ends in the channel body's own blank line and is
    unchanged, which is what keeps every recorded plain-turn prompt byte-identical."""
    plain = turn_prompt("Rill: I force the door")
    assert plain.endswith("I force the door\n\nResolve this turn.")

    framed = turn_prompt('Rill: Can you see a doorway?', turn_framing="question")
    assert ".Resolve this turn." not in framed
    assert framed.endswith("\n\nResolve this turn.")


def test_a_withheld_note_coexists_with_every_other_optional_block():
    """The note must not crowd out or corrupt the other rendered blocks."""
    from narrator.policy_types import InteractionCue

    reason = "No action occurred because this hazard requires an authenticated confirmation."
    prompt = turn_prompt(
        "Rill: GM, why was I refused?",
        canon="# Canon\n\nSome facts.",
        interaction_cue=InteractionCue("scene_interlocutor"),
        withheld_note=reason,
        romance_escalation_blocked=True,
    )
    assert reason in prompt
    assert "Active in-world interlocutor:" in prompt
    assert "Romance policy: escalation is unavailable." in prompt
    assert "Some facts." in prompt
