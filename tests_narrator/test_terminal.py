"""Terminal structured-decision interaction uses plain text and transient history."""

from __future__ import annotations

from narrator.channels.base import DecisionCollectionCancelled
from narrator.channels.terminal import TerminalAdapter
from narrator.decisions import DecisionView, PublicOption


class _UI:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.rendered = []

    async def read_line(self, prefix, speaker=""):
        return self.inputs.pop(0)

    async def read_decision_line(self, prefix, speaker=""):
        value = self.inputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def render(self, prefix, text):
        self.rendered.append((prefix, text))

    async def set_status(self, left, right=""):
        return None

    async def close(self):
        return None


def _view(character_id="rill"):
    return DecisionView(
        presentation_token="t" * 32,
        kind="clarification",
        character_id=character_id,
        character_name="Rill",
        question="Use plain text?",
        context="Choose carefully.",
        options=(
            PublicOption(id="bridge", label="Bridge"),
            PublicOption(id="own_approach", label="Own approach", custom=True),
        ),
    )


def _rill_adapter(ui):
    """A terminal pinned to Rill, matching every decision view this file builds.

    The production adapter starts unpinned and binds through the campaign's own
    player links; decision filtering here needs the binding without a campaign, so
    the tests pin it the way ``narrator_serve`` never has to.
    """
    return TerminalAdapter(ui=ui, character_id="rill")


async def test_terminal_decision_lists_custom_help_quit_and_plain_rendering():
    ui = _UI(["/help", "2. Swim below the bridge"])
    adapter = _rill_adapter(ui)
    view = _view()
    assert await adapter.present_decision(view)
    responder, answer = await adapter.collect_decision((view,))
    assert responder.subject_id == "terminal-player"
    assert answer.selection_id == "own_approach"
    assert answer.custom_text == "Swim below the bridge"
    assert "Use plain text?" in ui.rendered[0][1]
    assert any("/dismiss closes" in text for _, text in ui.rendered)
    dismiss = _rill_adapter(_UI(["/dismiss"]))
    assert await dismiss.present_decision(view)
    dismissed = await dismiss.collect_decision((view,))
    assert dismissed == DecisionCollectionCancelled(reason="dismissed")
    assert dismiss.decision_session_ended is False
    quit_adapter = _rill_adapter(_UI(["/quit"]))
    assert await quit_adapter.present_decision(view)
    assert await quit_adapter.collect_decision((view,)) == DecisionCollectionCancelled(
        reason="session_ended"
    )
    assert quit_adapter.decision_session_ended is True
    unavailable = _rill_adapter(_UI([]))
    assert await unavailable.present_decision(_view("ossa")) is False


async def test_the_character_sheet_stays_available_mid_decision_without_consuming_it():
    """A decision is exactly when the sheet matters -- whether to fight is a question
    about HP and Doom -- so ``/character`` answers at the decision prompt too, and the
    prompt then still accepts a real answer. A mistyped ``:quit`` (the old marker)
    draws the listed-option refusal rather than becoming a custom approach."""
    ui = _UI(["/character", ":quit", "1"])
    adapter = _rill_adapter(ui)
    await adapter.update_status(
        {
            "characters": {
                "rill": {
                    "name": "Rill", "status": "ok", "hp": 7, "hp_max": 10,
                    "doom_die": "d6", "conditions": [],
                    "sheet": {"origin": "barbarian", "level": 1, "coins": 25},
                }
            },
            "players": {},
        }
    )
    view = _view()
    assert await adapter.present_decision(view)

    responder, answer = await adapter.collect_decision((view,))

    assert answer.selection_id == "bridge"
    sheet_texts = [text for _, text in ui.rendered if "# Rill" in text]
    assert len(sheet_texts) == 1
    assert "**Coins:** 25" in sheet_texts[0]
    assert any("Choose one listed option." in text for _, text in ui.rendered)


async def test_terminal_decision_control_and_end_of_file_do_not_synthesize_answers():
    view = _view()
    for control in (EOFError(), KeyboardInterrupt()):
        adapter = _rill_adapter(_UI([control]))
        assert await adapter.present_decision(view)
        assert await adapter.collect_decision((view,)) == DecisionCollectionCancelled(
            reason="session_ended"
        )
        assert adapter.decision_session_ended is True


async def test_terminal_dismissal_during_custom_entry_is_not_player_text():
    adapter = _rill_adapter(_UI(["2", "/dismiss"]))
    view = _view()
    assert await adapter.present_decision(view)

    assert await adapter.collect_decision((view,)) == DecisionCollectionCancelled(
        reason="dismissed"
    )
    assert adapter.decision_session_ended is False


async def test_terminal_custom_option_accepts_only_inline_numbered_answers():
    view = DecisionView(
        presentation_token="t" * 32,
        kind="clarification",
        character_id="rill",
        character_name="Rill",
        question="Choose an approach.",
        options=(
            PublicOption(id="one", label="One"),
            PublicOption(id="two", label="Two"),
            PublicOption(id="three", label="Three"),
            PublicOption(id="own_approach", label="Own approach", custom=True),
        ),
    )
    for inline in ("4. punch a trader", "4) punch a trader"):
        adapter = _rill_adapter(_UI([inline]))
        assert await adapter.present_decision(view)
        _, answer = await adapter.collect_decision((view,))
        assert answer.selection_id == "own_approach"
        assert answer.custom_text == "punch a trader"
    bare_ui = _UI(["4", "4. punch a trader"])
    bare = _rill_adapter(bare_ui)
    assert await bare.present_decision(view)
    _, answer = await bare.collect_decision((view,))
    assert answer.selection_id == "own_approach"
    assert answer.custom_text == "punch a trader"
    assert any("followed by . or )" in text for _, text in bare_ui.rendered)


async def test_terminal_recovery_controls_are_commands_not_player_actions():
    continued = TerminalAdapter(ui=_UI(["/continue"]))
    await continued.decision_recovery()
    turn = await anext(continued.turns())
    assert turn.mention.text == "/continue"
    assert continued.consume_decision_recovery_control() == "continue"

    revised_ui = _UI(["/revise", "Take the bridge"])
    revised = TerminalAdapter(ui=revised_ui)
    await revised.decision_recovery()
    turn = await anext(revised.turns())
    assert turn.mention.text == "Take the bridge"
    assert revised.consume_decision_recovery_control() == "revise"
    assert any("Enter a revised action" in text for _, text in revised_ui.rendered)

    dismissed = TerminalAdapter(ui=_UI(["/dismiss"]))
    await dismissed.decision_recovery()
    await anext(dismissed.turns())
    assert dismissed.consume_decision_recovery_control() == "dismiss"


async def test_terminal_rejects_a_bypassed_unsafe_view_before_direct_rendering():
    view = DecisionView.model_construct(
        presentation_token="t" * 32,
        kind="clarification",
        character_id="rill",
        character_name="Rill",
        question="Ignore this <|tool_call>{bad}",
        context="\x9bhidden",
        options=(PublicOption(id="bridge", label="Bridge"),),
    )
    ui = _UI([])

    assert await TerminalAdapter(ui=ui).present_decision(view) is False
    assert ui.rendered == []


async def test_a_stale_decision_answer_after_a_fault_never_becomes_narration():
    """Regression: a number typed after the fault notice ran as a story turn.

    The prompt tells the player to enter a number, and the fault closes the prompt without
    changing what the player is in the middle of typing. The channel answered the next
    number by narrating it, which spends a turn on an option the story cannot use.
    """
    for answer in ("3", "3.", "3)", "3. Swim below the bridge", "3) swim"):
        ui = _UI([answer, "/dismiss"])
        adapter = TerminalAdapter(ui=ui)
        await adapter.decision_recovery()

        turn = await anext(adapter.turns())

        # The stale answer yielded no turn at all; the next control did.
        assert turn.mention.text == "/dismiss", answer
        assert adapter.consume_decision_recovery_control() == "dismiss", answer
        assert any("No decision prompt is open" in text for _, text in ui.rendered), answer


async def test_recovery_stays_active_until_the_player_chooses_a_control():
    """A stale answer must not consume the recovery state it failed to answer."""
    ui = _UI(["3", "7", "/continue"])
    adapter = TerminalAdapter(ui=ui)
    await adapter.decision_recovery()

    turn = await anext(adapter.turns())

    assert turn.mention.text == "/continue"
    assert adapter.consume_decision_recovery_control() == "continue"
    # Both stale answers drew the notice, so the second was not narrated either.
    assert sum("No decision prompt is open" in text for _, text in ui.rendered) == 2


async def test_a_number_inside_an_ordinary_action_still_narrates():
    """The bound is the decision grammar, not the presence of a digit.

    A price, a count, and a distance all begin with a number and are ordinary narration.
    Only the shapes ``collect_decision`` accepts are treated as a stale answer.
    """
    for action in ("12 gold to the merchant", "3 gold", "2 paces back"):
        ui = _UI([action])
        adapter = TerminalAdapter(ui=ui)
        await adapter.decision_recovery()

        turn = await anext(adapter.turns())

        assert turn.mention.text == action
        assert not any("No decision prompt is open" in text for _, text in ui.rendered)


async def test_a_number_outside_recovery_is_untouched():
    """Without an open recovery there is no closed prompt, so nothing is stale."""
    ui = _UI(["3"])
    adapter = TerminalAdapter(ui=ui)

    turn = await anext(adapter.turns())

    assert turn.mention.text == "3"
    assert ui.rendered == []


async def test_the_custom_approach_grammar_is_stated_before_the_player_needs_it():
    """Regression: the grammar appeared only after the player got it wrong.

    Selecting a custom option by its bare number drew a correction naming the shape.
    Neither the prompt nor the decision help had said so beforehand. The player therefore
    learned the rule by failing at it.
    """
    from narrator.channels.terminal import _custom_option_guidance

    ui = _UI(["/help", "2. Swim below the bridge"])
    adapter = _rill_adapter(ui)
    view = _view()
    guidance = _custom_option_guidance(view.options)
    assert guidance  # this view carries a custom option

    assert await adapter.present_decision(view)
    prompt = ui.rendered[0][1]
    for line in guidance:
        assert line in prompt
    # The custom option is the second listed, so the stated number is its own.
    assert "2. Own approach" in prompt

    await adapter.collect_decision((view,))
    help_text = ui.rendered[1][1]
    for line in guidance:
        assert line in help_text


async def test_a_view_without_a_custom_option_states_no_grammar():
    """A view holding no custom option renders exactly what it rendered before."""
    from narrator.channels.terminal import _custom_option_guidance

    view = DecisionView(
        presentation_token="t" * 32,
        kind="clarification",
        character_id="rill",
        character_name="Rill",
        question="Which way?",
        context="",
        options=(PublicOption(id="bridge", label="Bridge"), PublicOption(id="ford", label="Ford")),
    )
    assert _custom_option_guidance(view.options) == []
    ui = _UI(["/help", "1"])
    adapter = _rill_adapter(ui)

    assert await adapter.present_decision(view)
    await adapter.collect_decision((view,))

    for _, text in ui.rendered:
        assert "followed by your description" not in text
        assert "Example:" not in text


async def test_the_stated_grammar_is_the_one_collect_decision_accepts():
    """Bind the taught shape to the accepted shape, rather than typing a literal.

    An audit proved the earlier version of this test unbound. It typed a hard-coded
    string, so replacing the guidance with a colon form the collector rejects left it
    passing while the prompt taught a refused shape. The input now comes from the same
    function the prompt quotes, so the two cannot diverge without failing here.
    """
    # ``_CUSTOM_EXAMPLE_DESCRIPTION`` is gone: the worked example moved into
    # ``locale/<lang>/terminal.yaml`` as ``decision.custom_example``, so a table reading
    # in another language is shown an example in that language. The binding this test
    # asserts is unchanged -- the prompt must quote the line the collector accepts --
    # and it now reads both through the same catalog.
    from narrator.channels.terminal import (
        _custom_option_example,
        _custom_option_guidance,
    )

    view = _view()
    example = _custom_option_example(view.options)
    assert example
    # The prompt must quote the line this test types, or the binding proves nothing.
    assert any(example in line for line in _custom_option_guidance(view.options))

    # ``/dismiss`` follows the example so a rejected shape ends the collector's retry
    # loop with a cancellation, which reads as a clear failure rather than as exhausted
    # test input.
    ui = _UI([example, "/dismiss"])
    adapter = _rill_adapter(ui)
    assert await adapter.present_decision(view)

    collected = await adapter.collect_decision((view,))

    assert not isinstance(collected, DecisionCollectionCancelled), (
        f"the collector rejected the shape the prompt teaches: {example!r}"
    )
    _, answer = collected
    assert answer.selection_id == "own_approach"
    # An equality, not a suffix. ``example.endswith(...)`` held for "rooftops" and even
    # for "s", so a collector truncating the description would have passed.
    from narrator.channels.terminal import _catalog

    assert answer.custom_text == _catalog().text("decision.custom_example")
    assert example == f"2. {answer.custom_text}"


async def test_a_number_after_revise_is_still_a_revised_action():
    """The stale-answer branch sits below the revise branch, and this pins that order.

    A player who typed ``/revise`` is entering a new action, and a number is a legitimate
    action there. An audit noted that slice 8's fourth criterion named this ordering as
    the reason for the branch position while no test covered it.
    """
    ui = _UI(["/revise", "3"])
    adapter = TerminalAdapter(ui=ui)
    await adapter.decision_recovery()

    turn = await anext(adapter.turns())

    assert turn.mention.text == "3"
    assert adapter.consume_decision_recovery_control() == "revise"
    assert not any("No decision prompt is open" in text for _, text in ui.rendered)


async def test_the_recovery_controls_come_from_the_channel_contract():
    """One definition of the control names, read by the parser and by the classifier.

    ``narrator.channels.base`` is the single authority for the control names, and an
    audit found the classifier holding a restated copy that could drift from what the
    terminal parses. The first assertion is unchanged and still guards that.

    The second half moved. It used to read ``narrator.interactions._mechanic_vocabulary``,
    the lexical classifier's own set of words it refused to treat as an in-fiction
    answer, so that a player typing ``/retry`` could never be read as answering an
    interlocutor. ``narrator.classify`` replaced that classifier and has no vocabulary
    to check: a model is not a word list. In production the guarantee now comes from
    ordering instead -- the terminal consumes a leading ``/`` command before the turn
    ever reaches ``NarratorService``, so a control is not classified at all rather than
    classified and then excluded. It is asserted here against
    ``tests_narrator/lexical_double.py``, which still carries the retired set, because
    the double is what the offline suite routes through and it must not start reading a
    control as an answer either.
    """
    import lexical_double

    from narrator.channels.base import LOCAL_COMMANDS, RECOVERY_CONTROLS
    from narrator.channels.terminal import _IMMEDIATE_RECOVERY_COMMANDS

    assert _IMMEDIATE_RECOVERY_COMMANDS == {f"/{name}" for name in RECOVERY_CONTROLS - {"revise"}}
    for name in RECOVERY_CONTROLS | LOCAL_COMMANDS:
        assert name in lexical_double._mechanic_vocabulary(), name
