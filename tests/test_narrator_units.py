"""Test narrator units.
"""

from narrator_units_shared import *  # noqa: F401,F403 -- the split's shared header

# -- security policy ---------------------------------------------------------


def test_the_manifest_matches_the_servers_registered_tools():
    """The narrator manifest and the server's own tool list must not drift apart."""
    from test_mcp_wiring import EXPECTED_TOOLS

    assert policy.MCP_SERVED_TOOLS == EXPECTED_TOOLS
    assert policy.MCP_TOOLS == EXPECTED_TOOLS - policy.ENGINE_ONLY_TOOLS
    assert policy.PLAYER_FACING_TOOLS == (EXPECTED_TOOLS - policy.ENGINE_ONLY_TOOLS) | {"skills"}


def test_the_surface_check_demands_equality_not_a_subset():
    """A subset check would let a future release auto-register a tool and still pass."""
    policy.assert_tool_surface(policy.PLAYER_FACING_TOOLS)

    with pytest.raises(policy.ToolSurfaceError, match="missing"):
        policy.assert_tool_surface(policy.PLAYER_FACING_TOOLS - {"scene_commit"})

    with pytest.raises(policy.ToolSurfaceError, match="unexpected"):
        policy.assert_tool_surface(policy.PLAYER_FACING_TOOLS | {"summarise"})


def test_every_forbidden_tool_is_rejected_by_name():
    """The canary reports which dangerous tool arrived, not merely that one did."""
    for forbidden in sorted(policy.FORBIDDEN_TOOLS):
        with pytest.raises(policy.ToolSurfaceError, match="forbidden"):
            policy.assert_tool_surface(policy.PLAYER_FACING_TOOLS | {forbidden})


# -- leak filter -------------------------------------------------------------


def test_the_filter_strips_all_three_recorded_spellings():
    """Test the filter strips all three recorded spellings.
    """
    xml = "Before <tool_call>{\"a\":1}</tool_call> after"
    token = "Before <|tool_call>call:rest{}<tool_call|> after"
    bare = 'Before scene_commit{public_summary:<|"|>text<|"|>} after'

    for text in (xml, token, bare):
        cleaned, removed = delivery.scrub_markup(text)
        assert "<|" not in cleaned and "<tool_call" not in cleaned, text
        assert cleaned == "Before after"
        assert len(removed) == 1


def test_the_filter_strips_a_synthetic_nested_tool_call():
    """The filter strips a synthetic nested tool call."""
    recorded = (
        'scene_commit{public_summary:<|"|>A scout found a blue tile.<|"|>,visible_changes:[<|"|>The tile lies beside the crate.<|"|>]}'
    )
    cleaned, removed = delivery.scrub_markup(recorded)
    assert cleaned == ""
    assert removed


def test_the_filter_strips_every_spelling_of_every_control_token():
    """Derived from ``delivery._CONTROL_TOKEN_NAMES`` rather than hand-listed. An audit
    found the two lists drifting apart: this test named ``start_of_turn`` while the
    filter's tuple held two names, so a truncated ``<start_of_turn`` leaked while this
    test passed. Deriving both from one tuple means a new name cannot be covered here
    and missed there.
    """
    assert {"tool_call", "turn", "start_of_turn", "end_of_turn"} <= set(
        delivery._CONTROL_TOKEN_NAMES
    )
    for name in delivery._CONTROL_TOKEN_NAMES:
        for token in (f"<{name}|>", f"<|{name}>", f"<|{name}|>"):
            cleaned, removed = delivery.scrub_markup(f"Narration. {token}")
            assert cleaned == "Narration.", token
            assert removed, token


def test_the_filter_leaves_ordinary_markup_alone():
    """Angle brackets without pipes are prose or markup, not control tokens."""
    for safe in ("<b>bold</b>", "a < b and c > d", "<https://example.com>"):
        cleaned, _ = delivery.scrub_markup(safe)
        assert cleaned == safe


def test_the_filter_keeps_a_trailing_less_than_that_prose_supplies():
    """An audit found the truncated-opener rule deleting narration to end of string.

    Every string here ends in ``<`` followed only by word characters, with no ``>`` and
    no newline after it. The predecessor rule accepted any such run as a cut control
    token and returned a span to ``len(text)``. This game rolls under target numbers,
    so these are the sentences a narrator writes, not markup.
    """
    for safe in (
        "Ossa grins. <3",
        "Roll under <13",
        "compare a<b",
        "Rill's hit points drop below <5",
        "Damage is 2<4 so the blow lands",
    ):
        cleaned, removed = delivery.scrub_markup(safe)
        assert cleaned == safe
        assert removed == []


def test_the_filter_leaves_ordinary_narration_untouched():
    """A filter that mangles prose is worse than the leak it prevents."""
    prose = "Rill steps onto the plank walk. The tide is rising.\n\nOssa hesitates."
    cleaned, removed = delivery.scrub_markup(prose)
    assert cleaned == prose
    assert removed == []


# -- ratification gate -------------------------------------------------------


def test_the_gate_reads_the_servers_own_state_file(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state = campaign / "state.json"

    state.write_text(json.dumps({"fiction_debt": []}), encoding="utf-8")
    assert ledger.is_ratified(tmp_path) is True

    state.write_text(json.dumps({"fiction_debt": [{"seq": 1}]}), encoding="utf-8")
    assert ledger.is_ratified(tmp_path) is False
    assert len(ledger.read_fiction_debt(tmp_path)) == 1

    state.write_text(json.dumps({"fiction_debt": []}), encoding="utf-8")
    assert ledger.is_ratified(tmp_path) is True


def test_the_gate_fails_closed_on_an_unreadable_file(tmp_path):
    """Absence must not read as settled.

    Returning an empty list here made a missing state file indistinguishable from a
    ratified one, which delivered every turn ungated for a whole run.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ledger.LedgerUnreadable):
        ledger.read_fiction_debt(tmp_path)
    assert ledger.is_ratified(tmp_path) is False

    # A wholly absent campaign directory behaves the same way.
    assert ledger.is_ratified(tmp_path / "nowhere") is False


def test_the_gate_rejects_a_non_object_and_bad_bytes(tmp_path):
    """A JSON array and invalid UTF-8 both aborted the service instead of withholding."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state = campaign / "state.json"

    state.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ledger.LedgerUnreadable):
        ledger.read_fiction_debt(tmp_path)
    assert ledger.is_ratified(tmp_path) is False

    state.write_bytes(b'\xff\xfe not utf-8')
    with pytest.raises(ledger.LedgerUnreadable):
        ledger.read_fiction_debt(tmp_path)
    assert ledger.is_ratified(tmp_path) is False


def test_the_filter_strips_nested_tool_call_bodies():
    """Three model-facing tools declare nested parameters, so this shape is reachable."""
    for call in (
        'scene_commit{new_clocks:[{name:<|"|>Tide<|"|>,segments:6}]}',
        'combat_start{initial_ranges:{ossa:<|"|>near<|"|>}}',
        'scene_commit{a:{b:{c:1}},public_summary:<|"|>text<|"|>}',
    ):
        cleaned, removed = delivery.scrub_markup(call)
        assert cleaned == "", call
        assert removed


def test_the_filter_strips_openers_a_token_cut_left_unterminated():
    """max_tokens is 8192, so a reply cut mid-markup is reachable."""
    for text in ("Narration.<|tool_call", "Narration.<|tool_", "Narration. <tool_call>{"):
        cleaned, removed = delivery.scrub_markup(text)
        assert cleaned == "Narration.", text
        assert removed


def test_the_filter_preserves_formatting_on_a_leaking_turn():
    """The recorded leak appends a token to clean prose; the prose must survive intact."""
    text = "Findings:\n    - water-stained ledger\n\n| a | b |\n|---|---|\n| 1 | 2 |<turn|>"
    cleaned, removed = delivery.scrub_markup(text)
    assert removed
    assert "    - water-stained ledger" in cleaned
    assert "|---|---|" in cleaned


def test_strip_leaked_notices_removes_a_verbatim_notice_at_the_start():
    text = f"{_ROMANCE_NOTICE}\n\nRade is currently acting in combat, and his turn is open."
    cleaned, removed = delivery.strip_leaked_notices(text, (_ROMANCE_NOTICE,))
    assert cleaned == "Rade is currently acting in combat, and his turn is open."
    assert removed == [_ROMANCE_NOTICE]


def test_strip_leaked_notices_removes_a_notice_mid_reply_and_repairs_the_seam():
    text = f"Vessa hesitates. {_ROMANCE_NOTICE}\n\nShe steps back instead."
    cleaned, removed = delivery.strip_leaked_notices(text, (_ROMANCE_NOTICE,))
    assert cleaned == "Vessa hesitates.\n\nShe steps back instead."
    assert removed == [_ROMANCE_NOTICE]


def test_strip_leaked_notices_is_a_no_op_when_no_notice_appears():
    text = "Rade lashes out with a heavy, salt-crusted knife."
    cleaned, removed = delivery.strip_leaked_notices(text, (_ROMANCE_NOTICE,))
    assert cleaned == text
    assert removed == []


def test_strip_leaked_notices_never_matches_a_templated_notice():
    """``decision_refused``'s ``$refusal`` placeholder never appears in model text
    verbatim by construction; checking for the raw template would only ever find
    nothing, so it is skipped rather than pointlessly compared."""
    templated = "No action resolved: the rules engine refused $refusal, and nothing changed."
    text = f"{templated}\n\nMore fiction."
    cleaned, removed = delivery.strip_leaked_notices(text, (templated,))
    assert cleaned == text
    assert removed == []


def test_strip_leaked_notices_strips_every_distinct_notice_present():
    fault = locale.load("en", REPO_ROOT / "locale").notice("classifier_fault")
    text = f"{_ROMANCE_NOTICE} {fault}\n\nThe fight continues."
    cleaned, removed = delivery.strip_leaked_notices(text, (_ROMANCE_NOTICE, fault))
    assert cleaned == "The fight continues."
    assert sorted(removed) == sorted([_ROMANCE_NOTICE, fault])


async def test_a_withheld_turn_posts_the_notice_and_never_model_text():
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))
    outcome = delivery.TurnOutcome(
        narration="unratified fiction", ratified=False, withheld=True
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == config.withheld_notice
    assert adapter.posted == [("c1", config.withheld_notice)]
    assert "unratified fiction" not in posted


async def test_a_turn_that_scrubs_empty_posts_the_fault_notice():
    """A scrubbed-to-nothing reply is a lost story, not an unsettled ledger.

    An audit proved the first notice split routed this state to the unsettled
    notice, which asserted "outcomes are not yet settled" on a turn whose ledger
    had closed cleanly. The fault notice's copy — dice stand, story lost, do not
    redo — is true here.
    """
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))
    outcome = delivery.TurnOutcome(narration="   ", ratified=True, withheld=False)

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text
    assert posted == config.fault_notice


async def test_a_faulted_turn_posts_the_fault_notice_and_never_model_text():
    """The two notices split what one string used to blur: fault against unsettled."""
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))
    outcome = delivery.TurnOutcome(
        narration="half-written fiction", ratified=False, withheld=True,
        error="TimeoutError: settle attempt exceeded 30.0 seconds",
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == config.fault_notice
    assert "half-written fiction" not in posted
    assert config.fault_notice != config.withheld_notice


async def test_a_ratified_turn_posts_exactly_once():
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))
    outcome = delivery.TurnOutcome(narration="The bell is silent.", ratified=True, withheld=False)

    await delivery.deliver(adapter, turn, outcome, config)
    assert adapter.posted == [("c1", "The bell is silent.")]


async def test_a_decision_recovery_turn_with_no_tool_calls_still_posts_no_action_occurred():
    """The common case, unchanged: nothing this turn's tools did, so the claim holds."""
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        error="the bound social test was not used",
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == config.decision_fault_notice


async def test_a_decision_recovery_turn_whose_mechanic_was_refused_names_the_refusal():
    """Test a decision recovery turn whose mechanic was refused names the refusal.
    """
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
        tool_events=(
            {"tool": "combat_attack", "ok": False, "error": "target_out_of_reach", "event_id": None},
        ),
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert "No action resolved" in posted
    assert "combat_attack" in posted
    assert "target_out_of_reach" in posted
    assert "/retry" in posted


async def test_a_decision_recovery_turn_with_no_refused_mechanic_keeps_the_fault_notice():
    """With no mechanic refused and no state changed, the general fault notice is the
    only true claim left."""
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == config.decision_fault_notice


async def test_a_decision_recovery_turn_whose_tools_changed_state_never_claims_no_action():
    """Test a decision recovery turn whose tools changed state never claims no action.
    """
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="",
        ratified=True,
        withheld=True,
        decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
        tool_events=(
            {"tool": "npc_create", "ok": True, "error": "", "event_id": 1},
            {"tool": "combat_start", "ok": True, "error": "", "event_id": 2},
            {"tool": "combat_begin_turn", "ok": True, "error": "", "event_id": 3},
        ),
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted != config.decision_fault_notice
    assert "No action occurred" not in posted
    assert posted == delivery.DECISION_INCOMPLETE_NOTICE
    assert adapter.posted == [("c1", delivery.DECISION_INCOMPLETE_NOTICE)]


async def test_a_decision_recovery_turn_whose_only_success_is_read_only_still_posts_no_action_occurred():
    """AUD-1 regression (independent audit, result-1.json, blocker AUD-1).

    ``campaign_status`` and ``character_sheet`` are registered read-only tools
    (``src/bsh_mcp/server.py``'s ``# -- read-only tools --`` section) that call
    ``results.success()`` with no ``sequence=``, so ``redact_tool_event``'s
    ``event_id`` stays ``None`` for both -- confirmed against the real
    ``src/bsh_mcp/service.py`` bodies of both, neither of which ever opens a
    ``store.transaction()``. ``campaign_status``'s own server-side instructions tell
    the model to call it "at the start of a consequential turn" and "after any tool
    error," which is exactly the circumstance surrounding a decision-recovery turn,
    so this is the realistic shape the auditor reproduced against the pre-fix code:
    a lone successful ``campaign_status`` call made ``deliver()`` claim setup
    "already reached the record" when nothing was recorded. ``ok: true`` alone must
    never be enough; ``event_id`` must be non-``None`` too.
    """
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="",
        ratified=True,
        withheld=True,
        decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
        tool_events=(
            {"tool": "campaign_status", "ok": True, "error": "", "event_id": None},
        ),
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == config.decision_fault_notice
    assert posted != delivery.DECISION_INCOMPLETE_NOTICE


async def test_a_decision_recovery_turn_still_claims_progress_beside_a_read_only_lookup():
    """The positive case survives the fix: a genuine mutating success anywhere in
    the turn's tool events still triggers the truthful notice, even alongside a
    read-only lookup that alone would not -- the mixed shape a real turn that opens
    with ``campaign_status`` before attempting the confirmed action would produce.
    """
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=Path("/nonexistent"))
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = delivery.TurnOutcome(
        narration="",
        ratified=True,
        withheld=True,
        decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
        tool_events=(
            {"tool": "campaign_status", "ok": True, "error": "", "event_id": None},
            {"tool": "npc_create", "ok": True, "error": "", "event_id": 4},
        ),
    )

    posted = (await delivery.deliver(adapter, turn, outcome, config)).text

    assert posted == delivery.DECISION_INCOMPLETE_NOTICE


async def test_a_withheld_turns_narration_never_reaches_the_adapter(tmp_path):
    """The whole point: model text must not pass the delivery path when withheld."""
    _seed_ledger(tmp_path, [{"seq": 1}])
    stub = _StubAgent("Unratified narration.")
    engine = _engine_with_stub(tmp_path, stub)

    async def failing_settle(narration):
        return ("failed", 2, "settler stub")

    engine._settle = failing_settle  # noqa: SLF001 - test seam
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=tmp_path)
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)
    await delivery.deliver(adapter, turn, outcome, config)

    # A failed settle sets outcome.error, so the fault notice posts — the operator
    # distinction between a broken engine and a stuck ledger. Model text still never.
    assert adapter.posted == [("c1", config.fault_notice)]
    assert all("Unratified narration." not in text for _, text in adapter.posted)


async def test_run_turn_delivers_once_the_ledger_is_clean(tmp_path):
    _seed_ledger(tmp_path, [])
    stub = _StubAgent("The bell is silent.")
    engine = _engine_with_stub(tmp_path, stub)
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)

    assert outcome.ratified is True and outcome.withheld is False
    assert outcome.narration == "The bell is silent."
    assert outcome.settle == "none"
    assert outcome.settle_attempts == 0


def test_a_whitespace_commit_summary_fails_the_schema_not_the_server():
    import pytest as _pytest
    from pydantic import ValidationError

    from narrator.settle import SettleCommit

    with _pytest.raises(ValidationError):
        SettleCommit(kind="commit", public_summary="   ")


def test_narration_claims_mechanical_fact_pins_each_category_and_the_negative(tmp_path):
    """Offline fixture text for the detector ``guard_ratification`` uses."""
    from narrator.sweep import (
        narration_claims_mechanical_fact,
        stated_coin_totals,
        stated_hp_totals,
    )

    assert narration_claims_mechanical_fact("You now have 40 coins.") == "coins"
    assert narration_claims_mechanical_fact("The trader takes 3 copper from your palm.") == "coins"
    assert narration_claims_mechanical_fact("You are down to 7 hit points.") == "hp"
    assert narration_claims_mechanical_fact("You are at 9 HP after the fall.") == "hp"
    assert narration_claims_mechanical_fact("You now have the rope in your pack.") == "item"
    assert narration_claims_mechanical_fact("She picks up the lantern from the stall.") == "item"
    assert narration_claims_mechanical_fact("The tide rolls in past the pier.") == ""
    assert narration_claims_mechanical_fact("Banter only, nothing durable here.") == ""
    assert stated_coin_totals("You now have 40 coins, not the 35 coins you had before.") == [40, 35]
    assert stated_coin_totals("1,200 coins spill across the deck.") == [1200]
    assert stated_coin_totals("The tide rolls in.") == []
    assert stated_hp_totals("You are down to 7 hit points, then to 3 hp.") == [7, 3]
    assert stated_hp_totals("The tide rolls in.") == []


def test_read_trusted_scope_strips_the_life_status_annotation(tmp_path):
    """The trusted-scope parser reads the same render: an annotated dead NPC must
    stay in scope under its bare id, not vanish on the identifier check."""
    from narrator.interactions import read_trusted_scope

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "---\nsession: 1\nlocation_id: market\n---\n\n"
        "## Present NPCs\n\n- rade (dead)\n- sera-vane\n",
        encoding="utf-8",
    )
    scope = read_trusted_scope(tmp_path)
    assert scope.present_npc_ids == ("rade", "sera-vane")


def test_match_character_resolves_by_name_id_or_the_lone_party_member():
    """The resolution rules ``guard_ratification`` relies on, pinned directly."""
    from narrator.sweep import _match_character

    party = {
        "rill": _character_entry("rill", "Rill", coins=3),
        "ossa": _character_entry("ossa", "Ossa", coins=100),
    }
    assert _match_character("Rill now has 3 coins.", party)["id"] == "rill"
    assert _match_character("rill now has 3 coins.", party)["id"] == "rill"
    # Zero characters named, more than one in the party: cannot be resolved.
    assert _match_character("You now have 3 coins.", party) is None
    # Both characters named: still ambiguous, not "pick the first".
    assert _match_character("Rill and Ossa split the coins.", party) is None
    # Exactly one character in the party: "you" has no other referent.
    solo = {"rill": _character_entry("rill", "Rill", coins=3)}
    assert _match_character("You now have 3 coins.", solo)["id"] == "rill"
    # No characters at all: nothing to resolve against.
    assert _match_character("You now have 3 coins.", {}) is None


def test_a_characterless_campaign_renders_the_session_zero_cue(tmp_path):
    """A campaign that begins at session zero renders every other block empty, so
    without this cue the model gets no signal that the disclosed procedure applies
    and no pointer to the tools that ground it. The cue disappears with the first
    character sheet, and it fails ABSENT: a missing characters directory renders
    nothing, because misdirecting an established campaign on a read error is worse
    than a missing cue."""
    from narrator.canon import render_digest

    campaign = tmp_path / "campaign"
    characters = campaign / "characters"
    characters.mkdir(parents=True)
    (characters / ".gitkeep").write_text("", encoding="utf-8")

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)
    assert digest.stats["session_zero"] is True
    assert "## Session zero" in digest.text
    assert "bsh-session-zero" in digest.text
    assert "character_options" in digest.text
    assert "character_create" in digest.text

    (characters / "kara.json").write_text('{"id": "kara"}', encoding="utf-8")
    populated = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)
    assert populated.stats["session_zero"] is False
    assert "Session zero" not in populated.text

    absent = render_digest(tmp_path / "missing", tmp_path / "missing", 2000, 2600, 1200)
    assert absent.stats["session_zero"] is False
    assert absent.text == ""


def test_every_emitted_section_is_verbatim_and_the_block_never_exceeds_the_bound():
    """Composition reorders nothing and invents nothing: it selects whole sections.

    A composed block that carried edited text would put words in the server's mouth,
    so each piece must appear in the render exactly as written.
    """
    from narrator.canon import TRUNCATION_MARKER, _compose_scene_block, _section_spans

    text = _scene_render(60).strip()
    block, cut, rows, policy = _compose_scene_block(text, 2000)
    body = block[: -len(TRUNCATION_MARKER)] if cut else block

    assert policy == "priority"
    assert len(block) <= 2000
    for span in _section_spans(text):
        row = next(r for r in rows if r["name"] == span["name"])
        if row["state"] == "full" and span["name"] != "frontmatter":
            assert text[span["start"]:span["end"]].strip() in body, span["name"]
    # Document order survives: every kept heading appears in the render's own order.
    kept = [r["name"] for r in rows if r["state"] != "absent" and r["name"] != "frontmatter"]
    positions = [body.index(f"## {name}") for name in kept if f"## {name}" in body]
    assert positions == sorted(positions)


def test_a_fully_retained_fact_section_stays_verbatim(tmp_path):
    """The branch a partial-facts fixture never reaches, and where a rejoin would drift.
    """
    from narrator.canon import _compose_scene_block, _section_spans

    text = _scene_render(6, hidden_count=40).strip()
    block, cut, rows, policy = _compose_scene_block(text, 2000)
    span = next(s for s in _section_spans(text) if s["name"] == "Visible facts")
    facts = next(r for r in rows if r["name"] == "Visible facts")

    assert policy == "priority" and cut is True
    assert facts["state"] == "full"
    assert facts["items_retained"] == facts["items"] == 6
    assert text[span["start"]:span["end"]] in block
    assert "\n\n\n" not in block
    assert len(block) <= 2000


def test_the_measurement_separates_the_fixed_prefix_from_the_history():
    """System prompt and tool schemas are the fixed cost; the message list is not.

    A reader pricing a cache miss needs the split, because the fixed prefix is the part
    a prefix cache can serve and the history is the part front eviction moves.
    """
    from narrator.payload import measure_request

    tools = [{"type": "function", "function": {"name": "dice_roll", "parameters": {}}}]
    row = measure_request(_request("Resolve this turn.", system="S" * 9617, tools=tools))

    assert row["system_chars"] == 9617
    assert row["tools_chars"] == len(json.dumps(tools, ensure_ascii=False))
    assert row["messages_chars"] == len("Resolve this turn.")
    assert row["message_count"] == 1
    assert row["total_chars"] > row["system_chars"] + row["tools_chars"]


def test_a_tool_call_and_its_result_both_count_against_the_payload():
    """Tool traffic rides in the request body, so a payload measurement counts it.

    The narrator's own turns average several tool calls, and a measurement that read
    prose only would understate what a cache miss recomputes.
    """
    from narrator.payload import measure_request

    request = {
        "messages": [
            {"role": "system", "content": "S"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {"name": "dice_roll", "arguments": '{"dice":"1d20"}'},
                        "id": "call-1",
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"ok": true}'},
        ],
        "model": "test",
    }

    row = measure_request(request)

    assert row["message_count"] == 2
    assert row["messages_chars"] == len("dice_roll") + len('{"dice":"1d20"}') + len(
        '{"ok": true}'
    )
    assert row["canon_blocks"] == 0


def test_stripping_never_touches_a_tool_result_or_an_assistant_message():
    """A tool-use and tool-result pair must survive whole.

    ``SlidingWindowConversationManager`` refuses to trim at a point that orphans a
    tool result, so editing one would break the eviction path rather than the prompt.
    """
    from narrator.payload import strip_superseded_canon

    text, _ = _digest_turn("block")
    tool_result = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": text}]}}],
    }
    assistant = {"role": "assistant", "content": [{"text": text}]}
    messages = [tool_result, assistant, *_history(text)]

    assert strip_superseded_canon(messages) == 1
    assert messages[0] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": text}]}}],
    }
    assert messages[1]["content"][0]["text"] == text
    assert messages[2]["content"][0]["text"] == prompt.turn_prompt("Rill: hello")


def test_stripping_twice_removes_nothing_the_first_pass_left():
    """The engine runs this every turn over a history it already stripped."""
    from narrator.payload import strip_superseded_canon

    messages = _history(_digest_turn("block")[0])
    assert strip_superseded_canon(messages) == 1
    before = messages[0]["content"][0]["text"]

    assert strip_superseded_canon(messages) == 0
    assert messages[0]["content"][0]["text"] == before


def test_the_served_and_model_facing_surfaces_differ_by_exactly_the_engine_tools():
    """The fix for an unreliable model must not hand the model a new capability."""
    assert policy.MCP_SERVED_TOOLS - policy.MCP_TOOLS == policy.ENGINE_ONLY_TOOLS
    assert policy.ENGINE_ONLY_TOOLS == frozenset({"ledger_settle", "ability_apply_ruling"})
    assert not (policy.ENGINE_ONLY_TOOLS & policy.PLAYER_FACING_TOOLS)
    assert len(policy.MCP_SERVED_TOOLS) == len(policy.MCP_TOOLS) + 2  # plus the engine-only pair
    assert len(policy.PLAYER_FACING_TOOLS) == len(policy.MCP_TOOLS) + 1  # plus the skills tool


def test_has_open_rulings_reads_state_and_fails_closed(tmp_path):
    """The adjudication gate withholds while a ruling is open, and on an unreadable state."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text(
        json.dumps({"fiction_debt": [], "pending_rulings": []}), encoding="utf-8"
    )
    assert ledger.has_open_rulings(tmp_path) is False
    (campaign / "state.json").write_text(
        json.dumps({"pending_rulings": [{"id": "r1", "question": "?"}]}), encoding="utf-8"
    )
    assert ledger.has_open_rulings(tmp_path) is True
    (campaign / "state.json").write_text("{ not json", encoding="utf-8")
    assert ledger.has_open_rulings(tmp_path) is True  # fail closed


async def test_adjudicate_resolves_a_pending_ruling_from_the_narration(tmp_path):
    """The engine reads the open ruling, picks the model's choice, and applies it."""
    from narrator.adjudicate import AdjudicateOutcome

    _seed_ruling(tmp_path, {
        "id": "r1", "actor_id": "grim", "kind": "demon_revenge_steal",
        "question": "Which possession?", "options": ["long knife", "amulet"],
        "default_strategy": "random",
    })
    engine = _engine_with_stub(tmp_path, _StubAgent())

    async def stub_once(prompt):
        assert "amulet" in prompt  # the option space and narration reach the prompt
        return AdjudicateOutcome(choice="amulet")

    engine._adjudicate_once = stub_once  # noqa: SLF001
    calls: list = []
    engine._call_tool = lambda name, arguments, origin="settle": (  # noqa: SLF001
        calls.append((name, arguments)) or "success"
    )
    kind = await engine._adjudicate("The demon rips the amulet from Grim's throat.")
    assert kind == "resolved"
    assert calls == [
        ("ability_apply_ruling", {"ruling_id": "r1", "choice": "amulet", "source": "model"})
    ]


async def test_adjudicate_defaults_when_the_model_faults(tmp_path):
    """A model fault hands ability_apply_ruling an empty choice and the default source."""
    _seed_ruling(tmp_path, {
        "id": "r1", "actor_id": "grim", "kind": "demon_revenge_steal",
        "question": "Which?", "options": ["long knife"], "default_strategy": "random",
    })
    engine = _engine_with_stub(tmp_path, _StubAgent())

    async def boom(prompt):
        raise RuntimeError("model unreachable")

    engine._adjudicate_once = boom  # noqa: SLF001
    calls: list = []
    engine._call_tool = lambda name, arguments, origin="settle": (  # noqa: SLF001
        calls.append(arguments) or "success"
    )
    kind = await engine._adjudicate("The demon takes what it will.")
    assert kind == "resolved"  # the tool still applies the default
    assert calls == [{"ruling_id": "r1", "choice": "", "source": "default"}]


async def test_run_turn_withholds_while_a_ruling_is_open_then_delivers(tmp_path):
    """The delivery gate withholds a turn while a ruling is open, delivers once resolved."""
    ruling = {
        "id": "r1", "actor_id": "grim", "kind": "demon_revenge_steal",
        "question": "Which?", "options": ["long knife"], "default_strategy": "random",
    }
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Grim", "invoke"))

    _seed_ruling(tmp_path, ruling)
    engine = _engine_with_stub(tmp_path, _StubAgent("The blade is torn away."))

    async def leave_open(narration):
        return "partial"  # the ruling stays open

    engine._adjudicate = leave_open  # noqa: SLF001
    assert (await engine.run_turn(turn)).withheld is True

    _seed_ruling(tmp_path, ruling)

    async def resolve(narration):
        path = tmp_path / "campaign" / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["pending_rulings"] = []
        path.write_text(json.dumps(state), encoding="utf-8")
        return "resolved"

    engine._adjudicate = resolve  # noqa: SLF001
    assert (await engine.run_turn(turn)).withheld is False


async def test_an_unreadable_ledger_withholds_rather_than_delivering(tmp_path):
    """A campaign root with no state file must not deliver every turn ungated."""
    stub = _StubAgent()
    engine = _engine_with_stub(tmp_path, stub)
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)
    assert outcome.withheld is True


async def test_a_framework_fault_withholds_and_records_the_error(tmp_path):
    _seed_ledger(tmp_path, [])

    class _Exploding(_StubAgent):
        async def invoke_async(self, text: str):
            raise RuntimeError("model unreachable")

    engine = _engine_with_stub(tmp_path, _Exploding())
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)
    assert outcome.withheld is True
    assert "model unreachable" in outcome.error


# -- prompt and channel shape ------------------------------------------------


def test_the_prompt_carries_identity_context_and_skill_but_no_tool_catalogue():
    assembled = prompt.assemble_system_prompt(REPO_ROOT)
    assert "# Loaded from SOUL.md" in assembled
    assert "# Loaded from skills/bsh-gm/SKILL.md" in assembled
    assert "# Channel" in assembled
    # Strands transmits the 17 schemas natively; a hand-built catalogue duplicates them.
    assert "# Available tools" not in assembled


def test_no_prompt_bound_file_names_a_transport():
    """NR-CHANNEL-NEUTRAL, over every file the model actually reads.

    An audit found the predecessor of this test asserting ``\"Discord\" not in assembled``.
    That is case-sensitive, and the surviving occurrence was the lowercase ``discord``
    tag in ``skills/bsh-gm/SKILL.md`` front matter, which ``prompt.py`` reads whole. The
    check passed while the criterion failed. It also inspected only the assembled
    prompt, never the skills ``engine.DISCLOSED_SKILLS`` hands to the plugin, one of
    which told the narrator to post a Discord recap.
    """
    sources = [prompt.assemble_system_prompt(REPO_ROOT)]
    for relative in engine.DISCLOSED_SKILLS:
        skill = REPO_ROOT / relative / "SKILL.md"
        assert skill.is_file(), f"disclosed skill missing: {skill}"
        sources.append(skill.read_text(encoding="utf-8"))

    for text in sources:
        residue = text.lower().replace("discord_user_id", "")
        assert "discord" not in residue
        assert "slack" not in residue


def test_the_prompt_skips_a_missing_source(tmp_path):
    """Assembly must survive the Hermes removal renaming files underneath it."""
    (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
    assembled = prompt.assemble_system_prompt(tmp_path)
    assert "soul" in assembled and "# Channel" in assembled


def test_the_turn_text_matches_the_reference_harness_shape():
    """Test the turn text matches the reference harness shape.
    """
    turn = InboundTurn(
        channel_id="c1",
        mention=ChannelMessage("Rill", 'We search different rooms.'),
        backfill=(ChannelMessage("Ossa", "Let us inspect the workshop."),),
    )
    assert turn.channel_text() == (
        'Ossa: Let us inspect the workshop.\n@GM We search different rooms.'
    )


# -- configuration -----------------------------------------------------------


def test_the_pinned_hermes_values_survive():
    config = NarratorConfig(campaign_root=Path("/tmp/x"))
    assert config.max_tokens == 8192
    assert config.backfill_limit == 50
    assert config.base_url == "http://localhost:8000/v1"
    assert config.model_id == "google/gemma-4-26B-A4B-it"


def test_the_server_launch_is_portable(tmp_path, monkeypatch):
    """The server launches from the checkout; its state lives in the campaign root."""
    monkeypatch.delenv("BSH_SERVER_PYTHON", raising=False)
    config = NarratorConfig(campaign_root=tmp_path, repo_root=tmp_path / "checkout")
    command = resolved_server_command(config)
    assert command[0] == "uv"
    assert str(tmp_path / "checkout") in command
    assert server_env(config)["BSH_CAMPAIGN_ROOT"] == str(tmp_path)


def test_the_campaign_root_and_the_repo_root_stay_distinct(tmp_path):
    """A sandbox campaign must not make the harness look for server code inside it."""
    config = NarratorConfig(campaign_root=tmp_path, repo_root=tmp_path / "checkout")
    assert config.campaign_root != config.repo_root


def test_the_interpreter_override_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("BSH_SERVER_PYTHON", "/usr/bin/python3")
    config = NarratorConfig(campaign_root=tmp_path, repo_root=tmp_path / "checkout")
    command = resolved_server_command(config)
    assert command[0] == "/usr/bin/python3"
    assert command[1].endswith("src/bsh_mcp/server.py")


# -- combat visibility --------------------------------------------------------


def test_the_cut_never_evicts_the_combat_section(tmp_path):
    """A dropped Combat section returns the planner to deciding a fight blind.
    """
    from narrator.canon import _PROTECTED_SECTIONS, render_digest

    assert "Combat" in _PROTECTED_SECTIONS
    digest = render_digest(_canon_root(tmp_path, 60, combat=True), tmp_path, 2000, 2600, 1200)

    states = _states(digest)
    assert digest.stats["truncated"]["scene"] is True
    assert states["Visible facts"] == "partial"
    assert states["Combat"] == "full"
    assert "- round: 3" in digest.text
    assert "- combatant: sera-vane side=npc range=close" in digest.text


def test_the_combat_snapshot_reads_the_rendered_section(tmp_path):
    from narrator.interactions import read_combat_snapshot

    root = _canon_root(tmp_path, 4, combat=True)
    snapshot = read_combat_snapshot(root)

    assert snapshot.active is True
    assert snapshot.round == 3
    assert snapshot.active_actor == "sera-vane"
    assert snapshot.order == ("rill", "sera-vane")
    assert snapshot.side_of("rill") == "pc"
    assert snapshot.side_of("sera-vane") == "npc"
    assert snapshot.side_of("nobody") == ""
    assert snapshot.npc_combatants == ("sera-vane",)


def test_the_combat_snapshot_reads_a_closed_fight_and_a_missing_file_as_inactive(tmp_path):
    """Every failure mode must land on the value that preserves pre-combat behaviour."""
    from narrator.interactions import read_combat_snapshot

    assert read_combat_snapshot(_canon_root(tmp_path / "quiet", 4)).active is False
    assert read_combat_snapshot(tmp_path / "absent").active is False
    unreadable = tmp_path / "broken"
    (unreadable / "campaign").mkdir(parents=True)
    (unreadable / "campaign" / "scene.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    assert read_combat_snapshot(unreadable).active is False


def test_the_combat_snapshot_rejects_a_malformed_identifier(tmp_path):
    """The parser validates every identifier, matching ``read_trusted_scope``."""
    from narrator.interactions import read_combat_snapshot

    root = tmp_path / "hostile"
    (root / "campaign").mkdir(parents=True)
    (root / "campaign" / "scene.md").write_text(
        "## Combat\n\n"
        "- round: not-a-number\n"
        "- active actor: ../escape\n"
        "- order: rill, ../escape, sera-vane\n"
        "- combatant: rill side=pc range=close actions=2/2\n"
        "- combatant: ../escape side=npc range=close actions=2/2\n",
        encoding="utf-8",
    )
    snapshot = read_combat_snapshot(root)

    assert snapshot.round == 0
    assert snapshot.active_actor == ""
    assert snapshot.order == ("rill", "sera-vane")
    assert snapshot.sides == (("rill", "pc"),)


# -- redacted tool-event retention -------------------------------------------


def test_redact_tool_event_keeps_only_disposition_fields():
    from narrator.engine import redact_tool_event

    record = redact_tool_event(
        "combat_attack",
        exception=False,
        payload={
            "ok": True,
            "event_id": "evt-000007",
            "roll": {"die": 20, "result": 4},
            "narration_facts": ["Rill hits for 5"],
            "target_id": "orso-pell",
        },
    )
    assert record == {
        "tool": "combat_attack",
        "ok": True,
        "error": "",
        "event_id": "evt-000007",
    }
    assert "orso-pell" not in repr(record)
    assert "roll" not in record


def test_redact_tool_event_reads_a_failure_envelope():
    from narrator.engine import redact_tool_event

    record = redact_tool_event(
        "combat_attack",
        exception=False,
        payload={"ok": False, "error": "target_out_of_reach", "message": "too far"},
    )
    assert record == {
        "tool": "combat_attack",
        "ok": False,
        "error": "target_out_of_reach",
        "event_id": None,
    }
    assert "too far" not in repr(record)


def test_redact_tool_event_records_a_framework_exception_without_inventing_state():
    from narrator.engine import redact_tool_event

    record = redact_tool_event("scene_commit", exception=True, payload={})
    assert record == {
        "tool": "scene_commit",
        "ok": False,
        "error": "exception",
        "event_id": None,
    }


def test_public_helpers_reduce_each_source_to_player_scope(tmp_path):
    from narrator.canon import _npc_entries, _public_location, _public_scene

    location = (
        "---\nid: x\nhidden_entities:\n  - ghost\n---\n\n"
        "## Public description\n\nseen.\n\n## Hidden truths\n\nsecret.\n"
    )
    kept = _public_location(location)
    assert "seen." in kept and "secret." not in kept and "ghost" not in kept

    scene = "# Scene\n\n## Visible facts\n\n- open\n\n## Game-master-only facts\n\n- closed\n"
    trimmed = _public_scene(scene)
    assert "open" in trimmed and "closed" not in trimmed and "Game-master-only" not in trimmed

    index = "npcs:\n  - id: a\n    name: A\n    role: spy\n    faction: guild\n"
    reduced = _npc_entries(index, ["a"], public=True)
    assert "id: a" in reduced and "name: A" in reduced
    assert "spy" not in reduced and "guild" not in reduced


def test_exit_ids_and_entries_resolve_a_scene_slug_to_its_display_name():
    """Exit ids and entries resolve a scene slug to its display name. Synthetic fixtures exercise this contract."""
    from narrator.canon import _exit_entries, _scene_exit_ids

    scene = (
        "# Scene\n\n## Exits\n\n- the-drowned-customs-house\n- the-road-shrine\n\n"
        "## Objects\n\n- None recorded.\n"
    )
    assert _scene_exit_ids(scene) == ["the-drowned-customs-house", "the-road-shrine"]
    assert _scene_exit_ids("# Scene\n\n## Exits\n\n- None recorded.\n") == []

    index = (
        "locations:\n"
        "  - id: the-drowned-customs-house\n    name: The Drowned Customs House\n"
        "    exits: [the-eel-market]\n"
        "  - id: the-road-shrine\n    name: The Road Shrine\n"
        "    exits: [the-eel-market]\n"
        "  - id: the-eel-market\n    name: The Eel Market\n"
        "    exits: [the-drowned-customs-house, the-road-shrine]\n"
    )
    entries = _exit_entries(index, ["the-drowned-customs-house", "the-road-shrine"])
    assert "id: the-drowned-customs-house" in entries
    assert "name: The Drowned Customs House" in entries
    assert "id: the-road-shrine" in entries
    assert "name: The Road Shrine" in entries
    # The exit not present in this scene, and each entry's own exits list, are both
    # excluded -- only the id/name pair for a scene-listed exit belongs here.
    assert "the-eel-market" not in entries
    assert "exits:" not in entries

    assert _exit_entries(index, []) == ""


def test_a_decision_answer_is_accepted_when_it_names_a_label_or_carries_a_separator():
    """Test a decision answer is accepted when it names a label or carries a separator.
    """
    from narrator.channels.terminal import _select_option

    options = _decision_options()

    # An index carrying the separator the prompt's own lines print.
    assert _select_option(options, "1.") == ("scramble", "")
    assert _select_option(options, "3)") == ("flee", "")
    # The shapes that already worked keep working, byte for byte.
    assert _select_option(options, "2") == ("cautious", "")
    assert _select_option(options, '4. move quietly across the bridge') == (
        "own_approach", "move quietly across the bridge",
    )
    assert _select_option(options, "scramble") == ("scramble", "")

    # An exact option label, which is the text the prompt actually shows the player.
    assert _select_option(options, "Cautious Search") == ("cautious", "")
    assert _select_option(options, "ignore and flee") == ("flee", "")
    assert _select_option(options, "  Desperate Scramble  ") == ("scramble", "")


    assert _select_option(options, 'I climb slowly toward the upper opening') == (
        "own_approach", 'I climb slowly toward the upper opening',
    )


def test_an_answer_matching_nothing_still_refuses_without_consuming_the_view():
    """The lenience has edges, and each of them refuses rather than guessing.

    An out-of-range index is the sharpest: with a custom option present, free text
    always lands, so "9" at a four-option list would otherwise be silently recorded as
    an approach literally reading "9" -- worse than the refusal it replaced. A refusal
    returns no selection at all, so ``collect_decision`` re-prompts and the view stays
    open, exactly as it did before.
    """
    from narrator.channels.terminal import _select_option

    options = _decision_options()

    # A number naming no option is a mis-typed index, never a description.
    assert _select_option(options, "9") == ("", "")
    assert _select_option(options, "9.") == ("", "")
    assert _select_option(options, "0") == ("", "")
    assert _select_option(options, "9. do the thing") == ("", "")
    # A mistyped command draws a refusal rather than becoming an approach.
    assert _select_option(options, ":retyr") == ("", "")
    assert _select_option(options, "") == ("", "")

    # A view with no custom option -- a confirmation -- refuses free text. "I confirm
    # nothing" must never confirm, and against a fixed yes/no gate the only honest
    # reading of prose is that it is neither answer.
    from narrator.decisions import PublicOption

    confirmation = (
        PublicOption(id="confirm", label="Confirm"),
        PublicOption(id="decline", label="Decline"),
    )
    assert _select_option(confirmation, "I confirm") == ("", "")
    assert _select_option(confirmation, "whatever you think best") == ("", "")
    # Its own two shapes still resolve.
    assert _select_option(confirmation, "Confirm") == ("confirm", "")
    assert _select_option(confirmation, "1") == ("confirm", "")
    assert _select_option(confirmation, "2.") == ("decline", "")

    # An approach list carries no custom option either, so it refuses prose the same way.
    assert _select_option(_decision_options(custom=False), "I try something else") == ("", "")


def test_the_skill_file_states_no_concrete_roll_exemplar_the_model_could_copy():
    """The correctness check the requirement was really protecting -- a printed
    target matching the real sheet, a printed verdict matching ``classify`` --
    still runs on whatever ``engine.announced_rolls`` finds, unreachable today
    but exercised against known-bad text by
    ``test_the_skill_exemplars_would_have_failed_before_the_correction`` below, so
    a future edit that reintroduces a concrete exemplar still gets it checked
    rather than silently trusted.
    """
    from bsh_mcp.rules import classify

    skill_text = (REPO_ROOT / "skills" / "bsh-gm" / "SKILL.md").read_text(encoding="utf-8")
    sheets = _campaign_attributes()
    exemplars = engine.announced_rolls(skill_text)

    assert exemplars == (), exemplars

    for line in exemplars:
        sheet = sheets[line["character"].casefold()]
        real = sheet["attributes"][line["attribute"]]
        assert line["target"] == real, (
            f"{line['character']} rolls {line['attribute']} vs {line['target']}, but "
            f"the real sheet says {real}"
        )
        computed = classify(selected=line["total"], total=line["total"], target=real)
        assert computed == line["outcome"], (
            f"{line['character']} rolls {line['attribute']}: {line['total']} vs "
            f"{line['target']} is printed {line['outcome']} but classify says {computed}"
        )


def test_the_skill_exemplars_would_have_failed_before_the_correction():
    """The same test, run against the exact pre-fix text, must fail on both defects.
    """
    from bsh_mcp.rules import classify

    sheets = _campaign_attributes()
    pre_fix = (
        "Example: `Rill rolls DEX: 9 vs 14, success.`\n"
        "a failed group listen check still opens `Ossa rolls WIS: 3 vs 9, failure.` "
        "and `Rill rolls WIS: 11 vs 14, success.` before\n"
    )
    failures = []
    for line in engine.announced_rolls(pre_fix):
        real = sheets[line["character"].casefold()]["attributes"][line["attribute"]]
        if line["target"] != real:
            failures.append((line["character"], line["attribute"], "target"))
        elif classify(
            selected=line["total"], total=line["total"], target=real
        ) != line["outcome"]:
            failures.append((line["character"], line["attribute"], "verdict"))
    # Rill's DEX and WIS exemplars both printed "vs 14", a number on neither sheet;
    # Ossa's WIS exemplar printed the right target and the inverted verdict.
    assert failures == [
        ("Rill", "DEX", "target"),
        ("Ossa", "WIS", "verdict"),
        ("Rill", "WIS", "target"),
    ], failures


def test_the_verdict_guard_catches_inverted_and_invented_announcements():
    """The verdict guard catches inverted and invented announcements."""
    facts = (
        {"tool": "group_test", "character": "Ossa", "attribute": "WIS", "total": 3,
         "target": 9, "outcome": "success"},
        {"tool": "group_test", "character": "Rill", "attribute": "WIS", "total": 11,
         "target": 10, "outcome": "failure"},
    )
    narration = (
        "Ossa rolls WIS: 3 vs 9, failure.\n"
        "Rill rolls WIS: 11 vs 14, success.\n"
        "An empty cart rattles past."
    )
    mismatches = engine.verdict_mismatches(narration, facts)
    assert [entry["kind"] for entry in mismatches] == ["verdict", "target"], mismatches
    assert mismatches[0]["announced_outcome"] == "failure"
    assert mismatches[1]["announced_target"] == 14

    # The truthful rendering of the very same two rolls reports nothing.
    truthful = (
        "Ossa rolls WIS: 3 vs 9, success.\n"
        "Rill rolls WIS: 11 vs 10, failure.\n"
        "An empty cart rattles past."
    )
    assert engine.verdict_mismatches(truthful, facts) == ()

    # A roll no tool returned at all is the third shape: dice that never fell.
    invented = "Ossa rolls STR: 4 vs 11, success."
    assert [e["kind"] for e in engine.verdict_mismatches(invented, facts)] == ["unbacked"]

    # A turn that rolled nothing and announced nothing is the common case.
    assert engine.verdict_mismatches("The tide comes in.", ()) == ()
    assert engine.verdict_mismatches("The tide comes in.", facts) == ()


def test_the_verdict_guard_reads_the_real_envelopes_the_tools_return():
    """``roll_facts`` must read the three envelope shapes the mechanics actually emit.

    A guard fed by a parser that misses an envelope shape is a guard that stays silent
    on exactly the tool it cannot read, so this pins each shape against the payload the
    service composes: a single test, a group test's per-participant entries, and a
    ``combat_attack`` whose target number lives only inside its ``roll``.
    """
    single = engine.roll_facts("attribute_test", {
        "ok": True, "attribute": "INT", "target": 12, "outcome": "success",
        "roll": {"notation": "1d20", "dice": [8], "selected": 8, "total": 8, "target": 12},
    })
    assert single == ({"tool": "attribute_test", "character": "", "attribute": "INT",
                       "total": 8, "target": 12, "outcome": "success", "name": "",
                       "kills_helpless": None},)

    group = engine.roll_facts("group_test", {
        "ok": True, "attribute": "CHA", "outcome": "success", "individual": [
            {"character_id": "ossa", "name": "Ossa", "outcome": "failure", "target": 12,
             "roll": {"selected": 13, "total": 13, "target": 12}},
            {"character_id": "rill", "name": "Rill", "outcome": "success", "target": 13,
             "roll": {"selected": 4, "total": 4, "target": 13}},
        ],
    })
    assert [(f["character"], f["total"], f["target"], f["outcome"]) for f in group] == [
        ("Ossa", 13, 12, "failure"), ("Rill", 4, 13, "success")
    ]


    attack = engine.roll_facts("combat_attack", {
        "ok": True, "attribute": "STR", "outcome": "success", "attacker_id": "rill",
        "target_id": "reed-thug", "target_npc": {"id": "reed-thug", "hp": 1},
        "roll": {"selected": 5, "total": 5, "target": 12},
    })
    assert attack == ({"tool": "combat_attack", "character": "rill", "attribute": "STR",
                       "total": 5, "target": 12, "outcome": "success", "name": "",
                       "kills_helpless": None},)

    # A refused call returns no facts, so a mechanic that raised can never back an
    # announcement -- the same completed-versus-attempted discipline the combat probe
    # keeps.
    assert engine.roll_facts("combat_attack", {"ok": False, "error": "out_of_reach"}) == ()
    assert engine.roll_facts("campaign_status", {"ok": True, "summary": "quiet"}) == ()


def test_a_survived_helpless_roll_must_reach_the_table():
    """``helpless_roll`` is the one rolling tool whose result is a table face rather than
    a roll-under verdict, so it carries no attribute and no target and cannot be caught
    by the announcement comparison. It is still a ``roll`` field the skill's own rule
    covers, and a character who came back from zero hit points is the least skippable
    fact a turn can hold.
    """
    recovered = engine.roll_facts("helpless_roll", {
        "ok": True, "character_id": "ossa", "outcome": "scratched", "hp": 3,
        "roll": {"notation": "1d6", "dice": [1], "selected": 1, "total": 1},
    })
    assert recovered[0]["attribute"] == "" and recovered[0]["target"] is None

    silent = "The fight ends. The tide goes out, and the market stalls close for the night."
    assert [e["kind"] for e in engine.verdict_mismatches(silent, recovered)] == ["helpless"]

    # Either signal satisfies it: the table result's own word, or the hit points back.
    named = "Ossa is Scratched -- a new scar, nothing worse. She stands."
    assert engine.verdict_mismatches(named, recovered) == ()
    numbered = "Ossa drags herself upright, back to 3 hit points."
    assert engine.verdict_mismatches(numbered, recovered) == ()

    # A killed character is not a recovery, so an unmentioned death is not this defect
    # (the ledger's own debt covers it) and must not be reported here.
    killed = engine.roll_facts("helpless_roll", {
        "ok": True, "character_id": "ossa", "outcome": "killed",
        "roll": {"notation": "1d6", "dice": [6], "selected": 6, "total": 6},
    })
    assert engine.verdict_mismatches(silent, killed) == ()


def test_the_announcement_parser_reads_live_shapes_and_ignores_the_format_spec():
    """The parser must survive the emphasis a live model adds, and must not read the
    skill file's own ``<Name> rolls <ATTRIBUTE>:`` template as an announcement."""
    assert engine.announced_rolls("`<Name> rolls <ATTRIBUTE>: <total> vs <target>, <outcome>.`") == ()
    bolded = engine.announced_rolls("**Rill rolls DEX: 9 vs 12, success.**")
    assert bolded[0]["attribute"] == "DEX" and bolded[0]["outcome"] == "success"
    bulleted = engine.announced_rolls("- Ossa rolls WIS: 14 vs. 9, critical failure.")
    assert bulleted[0]["outcome"] == "critical_failure"
    assert bulleted[0]["target"] == 9
    # Prose that merely mentions a roll is not an announcement.
    assert engine.announced_rolls("She rolls her shoulders, then vaults the rail.") == ()


async def test_a_contradicted_verdict_is_corrected_once_before_the_turn_delivers():
    """Test a contradicted verdict is corrected once before the turn delivers.
    """
    facts = (
        {"tool": "attribute_test", "character": "Ossa", "attribute": "WIS", "total": 3,
         "target": 9, "outcome": "success"},
    )
    corrected_text = "Ossa rolls WIS: 3 vs 9, success.\nThe latch gives under her fingers."
    agent = _CorrectionAgent(corrected_text)
    narration = "Ossa rolls WIS: 3 vs 9, failure.\nThe latch will not give."

    delivered, record = await _engine_holding(facts)._correct_verdicts(narration, agent)

    assert delivered == corrected_text
    assert record["triggered"] is True and record["retried"] is True
    assert record["resolved"] is True
    assert record["mismatches"] == 1 and record["kinds"] == ["verdict"]
    # Exactly one re-invoke, and it never re-requests tools: the dice already fell.
    assert len(agent.prompts) == 1
    assert agent.prompts[0] == engine.VERDICT_CORRECTION_PROMPT
    assert "Do not call any tool" in agent.prompts[0]


async def test_a_truthful_turn_is_never_re_invoked_and_a_failed_correction_still_delivers():
    """The common case costs nothing, and no failure mode can lose a resolved turn."""
    facts = (
        {"tool": "attribute_test", "character": "Ossa", "attribute": "WIS", "total": 3,
         "target": 9, "outcome": "success"},
    )
    quiet = _CorrectionAgent("unused")
    narration = "Ossa rolls WIS: 3 vs 9, success.\nThe latch gives."
    delivered, record = await _engine_holding(facts)._correct_verdicts(narration, quiet)
    assert delivered == narration
    assert record["triggered"] is False and quiet.prompts == []

    # A turn that rolled nothing is not checked at all.
    unrolled, record = await _engine_holding(())._correct_verdicts("The tide comes in.", quiet)
    assert unrolled == "The tide comes in." and record["triggered"] is False
    assert quiet.prompts == []

    # A blank replacement falls back to the original rather than delivering
    # nothing -- with the contradicted line itself scrubbed out of it (see
    # ``test_an_unresolved_contradicted_announcement_is_scrubbed_not_delivered``).
    blank = _CorrectionAgent("   ")
    mismatched = "Ossa rolls WIS: 3 vs 9, failure.\nThe latch will not give."
    delivered, record = await _engine_holding(facts)._correct_verdicts(mismatched, blank)
    assert delivered == "The latch will not give."
    assert record["retried"] is True and record["resolved"] is False
    assert record["announcements_scrubbed"] == 1

    # So does a faulted one, and the turn is still delivered rather than withheld.
    class _Faulting:
        prompts: list = []

        async def invoke_async(self, prompt):
            raise RuntimeError("the endpoint dropped the connection")

    delivered, record = await _engine_holding(facts)._correct_verdicts(mismatched, _Faulting())
    assert delivered == "The latch will not give."
    assert record["retried"] is True and record["resolved"] is False

    # A replacement that is still wrong is delivered too -- one retry, never a loop --
    # and records resolved False, which is the number a soak run watches; the
    # still-wrong line is scrubbed and the prose around it kept.
    stubborn = _CorrectionAgent("Ossa rolls WIS: 3 vs 9, failure. Still stuck.")
    delivered, record = await _engine_holding(facts)._correct_verdicts(mismatched, stubborn)
    assert delivered == "Still stuck."
    assert record["resolved"] is False and len(stubborn.prompts) == 1
    assert record["announcements_scrubbed"] == 1


async def test_an_unresolved_contradicted_announcement_is_scrubbed_not_delivered():
    """Test an unresolved contradicted announcement is scrubbed not delivered.
    """
    facts = (
        {"tool": "grant_runic_weapon", "character": "rill", "attribute": "INT", "total": 15,
         "target": 10, "outcome": "failure", "name": "Sorrow", "kills_helpless": False},
    )
    narration = (
        "Rill grips the black blade. This is Sorrow.\n\n"
        "Sorrow rolls INT: rolled 7 vs target 10, success. The blade's hunger is a "
        "silent, brooding presence in her hand."
    )
    stubborn = _CorrectionAgent(narration)
    delivered, record = await _engine_holding(facts)._correct_verdicts(narration, stubborn)
    assert delivered == (
        "Rill grips the black blade. This is Sorrow.\n\n"
        "The blade's hunger is a silent, brooding presence in her hand."
    )
    assert record["resolved"] is False and record["announcements_scrubbed"] == 1
    # The scrub is a pure function too, and leaves a matching line alone.
    kept = "Sorrow rolls INT: rolled 15 vs target 10, failure.\nIt is cold."
    assert engine.scrub_contradicted_announcements(kept, facts) == (kept, 0)
    # Nothing to cut for a helpless-recovery mismatch: it is an omission, not a line.
    helpless = ({"tool": "helpless_roll", "character": "rill", "attribute": "", "total": 3,
                 "target": None, "outcome": "scarred"},)
    assert engine.scrub_contradicted_announcements("She stirs.", helpless) == ("She stirs.", 0)


# -- an empty-facts turn is not exempt from the verdict guard ----------------


async def test_an_empty_roll_facts_turn_still_catches_a_fabricated_verdict():
    """Pre-fix, ``_correct_verdicts`` opened with
    ``if not narration.strip() or not self._roll_facts_this_turn: return narration,
    record`` -- when a turn made zero real dice-rolling tool calls,
    ``_roll_facts_this_turn`` stayed empty and the function returned before
    ``verdict_mismatches`` was ever called. A turn with zero executed rolls is exactly
    the turn most likely to carry a fabricated verdict: nothing real exists locally to
    contradict it, and nothing for the model to have gotten wrong-but-close.

    Confirmed to fail against the pre-fix short-circuit (``git stash`` of
    ``src/narrator/engine.py`` reproduces it: the guard returns the narration
    unchanged and ``record[\"triggered\"]`` reads False) and to pass against the fix.
    """
    agent = _CorrectionAgent("The group listens beside an empty courtyard.")
    narration = (
        'A watch begins beside the warehouse.\n\nOssa rolls WIS: 14 vs 10, success.\n\nThe watch ends without incident.'
    )

    delivered, record = await _engine_holding(())._correct_verdicts(narration, agent)

    assert delivered == "The group listens beside an empty courtyard."
    assert record["triggered"] is True and record["retried"] is True
    assert record["mismatches"] == 1 and record["kinds"] == ["unbacked"]
    assert len(agent.prompts) == 1
    assert agent.prompts[0] == engine.VERDICT_CORRECTION_PROMPT


def test_combat_starts_initiative_rolls_populate_roll_facts():
    """``roll_facts`` recognized only a top-level ``roll`` field and a ``group_test``-
    style ``individual`` list; ``combat_start``'s own envelope
    (``src/bsh_mcp/service.py``) carries its per-character initiative rolls under a
    third shape, ``initiative``, so a ``combat_start`` call contributed zero facts even
    though it rolled and audited a real WIS test per character -- an initiative
    announcement was never verified, independent of the empty-facts short-circuit
    fixed above. The payload below is ``combat_start``'s own construction, field names
    verbatim: each ``initiative`` entry carries ``character_id``, ``name``, ``roll``
    (``bsh_mcp.dice.Roll.as_dict``), ``outcome``, ``bucket``, and
    ``first_turn_actions``, with no ``attribute`` key of its own because every
    initiative roll is a WIS test, hard-coded in ``combat_start`` itself.
    """
    payload = {
        "ok": True,
        "outcome": "combat_started",
        "order": ["ossa", "reed-thug", "rill"],
        "initiative": [
            {
                "character_id": "ossa", "name": "Ossa",
                "roll": {"notation": "1d20", "dice": [6], "selected": 6, "total": 6,
                          "modifier": 0, "edge": "single", "target": 9},
                "outcome": "success", "bucket": "before", "first_turn_actions": 2,
            },
            {
                "character_id": "rill", "name": "Rill",
                "roll": {"notation": "1d20", "dice": [15], "selected": 15, "total": 15,
                          "modifier": 0, "edge": "single", "target": 10},
                "outcome": "failure", "bucket": "after", "first_turn_actions": 2,
            },
        ],
    }
    facts = engine.roll_facts("combat_start", payload)
    assert [
        (f["tool"], f["character"], f["attribute"], f["total"], f["target"], f["outcome"])
        for f in facts
    ] == [
        ("combat_start", "Ossa", "WIS", 6, 9, "success"),
        ("combat_start", "Rill", "WIS", 15, 10, "failure"),
    ]

    # An invented initiative verdict is caught exactly like any other roll-under test.
    inverted = "Ossa rolls WIS: 6 vs 9, failure.\nCombat begins."
    assert [e["kind"] for e in engine.verdict_mismatches(inverted, facts)] == ["verdict"]

    # The truthful rendering of both rolls reports nothing.
    truthful = (
        "Ossa rolls WIS: 6 vs 9, success.\nRill rolls WIS: 15 vs 10, failure.\n"
        "Combat begins."
    )
    assert engine.verdict_mismatches(truthful, facts) == ()

    # Confirmed to fail against the pre-fix ``roll_facts`` (``git stash`` of
    # ``src/narrator/engine.py`` reproduces it): with no ``initiative`` branch, this
    # exact call returned zero facts, so the first assertion above is the pin.
    assert facts != ()


def test_grant_runic_weapons_own_session_test_populates_roll_facts_and_classifies_it():
    """``grant_runic_weapon``'s own envelope (``src/bsh_mcp/service.py``) carries its INT
    session test under ``details.session_test`` -- a bare roll
    (``Roll.as_dict()``), with no ``attribute`` or already-classified ``outcome``
    beside it the way every other roll-carrying tool's envelope carries one.
    ``skills/bsh-gm/SKILL.md`` instructs the model to announce any tool result
    carrying a ``roll`` field, wherever it sits in the envelope, so a live turn is
    expected to state this roll truthfully. Before this repair, widening
    ``_correct_verdicts``'s guard to check any turn whose narration announces a roll
    -- the fix for the empty-facts short-circuit -- meant a truthful
    ``grant_runic_weapon`` announcement had zero backing facts (``roll_facts`` did
    not read this shape at all) and was wrongly classified ``\"unbacked\"``, triggering
    a spurious correction on a turn that was already correct: the exact regression
    class this fix exists to prevent, reintroduced by the fix itself for one
    narrower shape. Confirmed to fail against the pre-repair candidate (``git
    stash`` of ``src/narrator/engine.py``): with no ``details.session_test`` branch,
    ``roll_facts`` returned zero facts and the truthful announcement below scored
    as ``\"unbacked\"``.
    """
    payload = {
        "ok": True,
        "details": {
            "name": "Whisper",
            "personality": "vengeful",
            "damage_attribute": "STR",
            "weapon_int": 9,
            "weapon_int_roll": {"notation": "2d6", "dice": [4, 3], "total": 7, "score": 9},
            "kills_helpless": True,
            "session_test": {
                "notation": "1d20", "dice": [5], "selected": 5, "total": 5,
                "modifier": 0, "edge": "single", "target": 9,
            },
        },
    }
    facts = engine.roll_facts("grant_runic_weapon", payload)
    assert [
        (f["tool"], f["attribute"], f["total"], f["target"], f["outcome"]) for f in facts
    ] == [("grant_runic_weapon", "INT", 5, 9, "success")]

    truthful = "Whisper rolls INT: 5 vs 9, success.\nThe blade hums against her palm."
    assert engine.verdict_mismatches(truthful, facts) == ()

    # An invented verdict on the same roll is still caught, exactly like any other
    # roll-under test -- this fix is not a blanket exemption for this tool.
    inverted = "Whisper rolls INT: 5 vs 9, failure.\nThe blade hums against her palm."
    assert [e["kind"] for e in engine.verdict_mismatches(inverted, facts)] == ["verdict"]


def test_a_runic_weapons_session_test_is_announced_under_the_weapons_name(
    service, roller
):
    """The weapon's start-of-session INT test must reach the table, from both tools
    that roll it, named for the roller -- which is the weapon, never the wielder.

    Confirmed to fail on the pre-fix tree: zero lines injected for both tools.
    """
    from conftest import make_character

    make_character(service, roller, name="Mara")
    roller.queue(4, 1, 1)  # 2d6 -> weapon INT 9; session test: natural 1
    granted = service.grant_runic_weapon("mara", name="the Widow's Tooth", personality="brutal")
    assert granted["ok"], granted
    facts = engine.roll_facts("grant_runic_weapon", granted)
    names = {"mara": "Mara"}
    amended, injected = engine.inject_missing_roll_announcements(
        "The blade hums in Mara's grip.", facts, names
    )
    assert injected == 1
    assert amended == (
        "the Widow's Tooth rolls INT: rolled 1 vs target 9, critical success.\n"
        "the Widow's Tooth has taken the measure of Mara: if Mara falls Helpless this "
        "session, the blade kills Mara outright.\n\n"
        "The blade hums in Mara's grip."
    )
    # The fact still attributes the roll to the wielder for every other consumer,
    # and carries the verdict the stake line is built from.
    assert facts[0]["character"] == "mara" and facts[0]["kills_helpless"] is True

    roller.queue(20)  # next session's test: a critical failure disarms the blade
    closed = service.session_close(
        session_title="The Ashen Bell", public_summary="The bell-keeper is dead."
    )
    assert closed["ok"], closed
    facts = engine.roll_facts("session_close", closed)
    amended, injected = engine.inject_missing_roll_announcements("", facts, names)
    assert injected == 1
    assert amended == (
        "the Widow's Tooth rolls INT: rolled 20 vs target 9, critical failure.\n"
        "the Widow's Tooth has not taken the measure of Mara for the next session: a "
        "Helpless fall in it is not the blade's kill."
    )
    assert facts[0]["character"] == "mara" and facts[0]["tool"] == "session_close"
    assert facts[0]["kills_helpless"] is False
    # A disobedient model that writes the line itself still earns no duplicate, and
    # an invented verdict on it is still caught, exactly as for every other roll.
    written = "the Widow's Tooth rolls INT: rolled 20 vs target 9, critical failure.\nIt sulks."
    assert engine.inject_missing_roll_announcements(written, facts, names)[1] == 0
    inverted = "the Widow's Tooth rolls INT: rolled 20 vs target 9, success.\nIt sulks."
    assert [e["kind"] for e in engine.verdict_mismatches(inverted, facts)] == ["verdict"]


def test_a_withheld_turn_still_carries_its_real_roll_lines(service, roller):
    """`withheld_mechanical_lines` builds, from a real grant envelope, the same two
    lines a delivered turn would inject -- the announcement and the stake -- so a
    ResolutionGuard withhold posts them above its notice instead of losing the
    audited d20 forever (`TurnOutcome.mechanical_lines`, `narrator.delivery`).
    Fails on the pre-fix tree: the helper did not exist and the withheld branch
    carried nothing."""
    from conftest import make_character

    make_character(service, roller, name="Mara")
    roller.queue(4, 1, 1)
    granted = service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    facts = engine.roll_facts("grant_runic_weapon", granted)
    lines = engine.withheld_mechanical_lines(facts, {"mara": "Mara"})
    assert lines == (
        "Sorrow rolls INT: rolled 1 vs target 9, critical success.",
        "Sorrow has taken the measure of Mara: if Mara falls Helpless this session, "
        "the blade kills Mara outright.",
    )
    assert engine.withheld_mechanical_lines((), {}) == ()


def test_every_language_renders_the_runic_stake_lines():
    """The four stake strings are player-facing text and therefore live in every
    ``locale/<language>/narrator.yaml``; the loader refuses a catalog that drops one.
    Each must render with both placeholders and differ between armed and disarmed."""
    for language in ("en", "fr", "de", "ja", "ru"):
        catalog = locale.load(language, REPO_ROOT / "locale")
        lines = {
            key: catalog.text(f"runic.{key}", name="Sorrow", wielder="Mara")
            for key in ("armed", "disarmed", "armed_next", "disarmed_next")
        }
        assert len(set(lines.values())) == 4, language
        for key, line in lines.items():
            assert "Sorrow" in line and "Mara" in line and "$" not in line, (language, key)


def test_classify_roll_under_agrees_with_the_real_bsh_mcp_classify():
    """``engine._classify_roll_under`` is a byte-for-byte duplicate of
    ``bsh_mcp.rules.classify`` (kept local rather than imported for the same
    ``mcp`` version-conflict reason ``TRAVERSAL_CLOCK_PREFIX`` states), so a future
    change to the real function's own logic must fail this test rather than let the
    two silently drift apart. This test file runs in the project environment, not
    the narrator's own ``mcp``-1.29 environment, so importing ``bsh_mcp`` here is
    safe.
    """
    from bsh_mcp.rules import classify

    # Every selected/total/target combination that matters: the two critical faces,
    # a success and a failure on the ordinary d20 range, and the roll-under boundary
    # itself (total exactly equal to target is a failure, never a success).
    for selected in (1, 2, 10, 19, 20):
        for total in (1, 5, 9, 10, 15, 20):
            for target in (1, 9, 10, 15, 20):
                real = classify(selected=selected, total=total, target=target)
                dup = engine._classify_roll_under(selected, total, target)
                assert real == dup, (selected, total, target, real, dup)


def test_the_skill_carries_no_rest_rule_the_tool_description_now_owns():
    """M18's rest-declaration rule (below) moved house entirely.
    """
    skill_text = (REPO_ROOT / "skills" / "bsh-gm" / "SKILL.md").read_text(encoding="utf-8")

    assert "is a mechanical action the `rest` tool resolves" not in skill_text
    assert "no safe-environment or authenticated-confirmation step" not in skill_text
    assert 'Never open a rest declaration with "I cannot resolve...\"' not in skill_text
    assert "Never state a specific hit-point figure for what the rest restored" not in skill_text
    assert "never narrate the rest atmospherically with no tool call at all" not in skill_text
    assert "Call `rest` for every short-rest declaration the party makes" not in skill_text
    assert "not your memory of an earlier call, is what settles" not in skill_text
    assert "the tool trusts what you pass rather than verifying the fiction itself" not in skill_text
    assert "Do not declare a wilderness camp safe without a specific fictional reason" not in skill_text
    assert _REST_EXAMPLE_PATTERN.search(skill_text) is None, (
        "a worked rest example belongs beside the tool's own docstring, not the skill"
    )

    server_text = (REPO_ROOT / "src" / "bsh_mcp" / "server.py").read_text(encoding="utf-8")
    raw_docstring = server_text.split("def rest(", 1)[1].split('"""', 2)[1]
    # The source docstring wraps at line length; normalize before substring checks
    # so a check does not depend on exactly where a line happens to break.
    rest_docstring = " ".join(raw_docstring.split())
    assert "Call this for every rest declaration the party makes" in rest_docstring
    assert "including a repeat later in the same session you expect to already be" in rest_docstring
    assert "the tool's own response, never your memory of a previous answer" in rest_docstring
    assert "do not refuse it yourself or invent a confirmation step first" in rest_docstring
    assert "trusted, never" in rest_docstring
    assert "wilderness camp is not" in rest_docstring
    assert "half CON" not in rest_docstring
    assert "once per in-game day" not in rest_docstring


def test_the_rest_declaration_rule_would_have_failed_before_the_fix():
    """The same assertions, run against the exact pre-fix text, must fail.
    """
    pre_fix = (
        "Select the correct tool. Use `attribute_test` for one character under "
        "risk. Use `group_test` for a coordinated party action. Use `usage_roll` "
        "when a tracked resource sees use. Use the combat tools inside a fight. "
        "Use `doom_roll` only for a trigger the other tools do not already "
        "cover.\n\n"
        "Narrate only the facts the tool returned, plus fiction that contradicts "
        "none of them. Read `narration_facts` and `warnings` in every result."
    )
    assert "is a mechanical action the `rest` tool resolves" not in pre_fix
    assert 'Never open a rest declaration with "I cannot resolve...\"' not in pre_fix

    pre_fix_calls = (
        "A tracked resource sees use:\n\n"
        "```json\n"
        '{"name": "usage_roll", "arguments": {"owner_id": "mara", "resource_id": '
        '"arrows", "reason": "two shafts loosed at the fleeing scout"}}\n'
        "```\n\n"
        "One melee attack inside combat:"
    )
    assert _REST_EXAMPLE_PATTERN.search(pre_fix_calls) is None


def test_the_one_invariant_denies_the_party_resources_block_as_a_source():
    """Test the one invariant denies the party resources block as a source.
    """
    skill_text = (REPO_ROOT / "skills" / "bsh-gm" / "SKILL.md").read_text(encoding="utf-8")

    assert "## The one invariant" in skill_text
    invariant_section = skill_text.split("## The one invariant", 1)[1].split("## Turn procedure", 1)[0]

    assert "Party resources block is not an exception to this" in invariant_section
    assert (
        "it is background, never a source you may quote a hit-point total from"
        in invariant_section
    )
    assert "call `campaign_status` or `character_sheet`" in invariant_section

    # The pre-existing invariant sentences are untouched by this addition.
    assert (
        "Never state a hit-point total, a Doom step, a Usage Die grade, an "
        "initiative outcome, applied damage, a death, or a Helpless result that did "
        "not arrive in a tool result. If you want to know a number, call the tool."
    ) in invariant_section


def test_the_one_invariant_party_resources_clause_would_have_failed_before_the_fix():
    """Test the one invariant party resources clause would have failed before the fix.
    """
    pre_fix_invariant_section = (
        "\n\nYou may improvise fiction freely. You may never state a mechanical "
        "fact that a tool did not return.\n\n"
        "Never write \"you roll a 7 and succeed\" unless `attribute_test` returned "
        "that result. Never state a hit-point total, a Doom step, a Usage Die "
        "grade, an initiative outcome, applied damage, a death, or a Helpless "
        "result that did not arrive in a tool result. If you want to know a "
        "number, call the tool.\n\n"
        "Withholding a number a tool did return is the same violation, run in "
        "reverse: a player who cannot see the roll cannot tell it from prose."
    )
    assert "Party resources block is not an exception to this" not in pre_fix_invariant_section
    assert "call `campaign_status` or `character_sheet`" not in pre_fix_invariant_section


def test_the_one_invariant_no_longer_excepts_a_pending_rest_declaration():
    """The repair-round-1 exception this test used to pin is gone, on purpose.

    That exception existed only because a rest-declaration rule lived in this file
    and needed a forward reference from the general HP-sourcing clause to it. Once
    the rest-declaration rule itself moved to the ``rest`` tool's own MCP
    description (see `test_the_skill_carries_no_rest_rule_the_tool_description_now_owns`),
    there is nothing left for the invariant to except -- a rest declaration is now
    just another pending action with no tool result yet, covered by the same
    unconditional fallback every other mechanic already uses. This pins the
    invariant's return to that simpler, unexcepted state.
    """
    skill_text = (REPO_ROOT / "skills" / "bsh-gm" / "SKILL.md").read_text(encoding="utf-8")
    invariant_section = skill_text.split("## The one invariant", 1)[1].split("## Turn procedure", 1)[0]

    assert "except when the pending action is a short- or long-rest declaration" not in invariant_section
    assert (
        "campaign_status` and `character_sheet` only ever return a character's "
        "current total, which before `rest` resolves is still the pre-rest figure, "
        "not what the rest restored"
    ) not in invariant_section
    assert "see the rest-declaration rule below" not in invariant_section

    # The unconditional fallback stands alone now, exactly as it did before repair
    # round 1 ever carved an exception out of it.
    assert (
        "call `campaign_status` or `character_sheet` and state what it returns."
    ) in invariant_section
    assert "is a mechanical action the `rest` tool resolves" not in skill_text


def test_the_one_invariant_rest_exception_would_have_failed_before_the_repair():
    """Test the one invariant rest exception would have failed before the repair.
    """
    pre_repair_invariant_section = (
        "\n\nYou may improvise fiction freely. You may never state a mechanical "
        "fact that a tool did not return.\n\n"
        "Never write \"you roll a 7 and succeed\" unless `attribute_test` returned "
        "that result. Never state a hit-point total, a Doom step, a Usage Die "
        "grade, an initiative outcome, applied damage, a death, or a Helpless "
        "result that did not arrive in a tool result. If you want to know a "
        "number, call the tool.\n\n"
        "The canon digest's Party resources block is not an exception to this. It "
        "carries each character's real current numbers so your fiction stays "
        "consistent with them and so you never estimate or recompute one; it is "
        "background, never a source you may quote a hit-point total from. When you "
        "want to state a hit-point figure and no tool result this turn carried one, "
        "call `campaign_status` or `character_sheet` and state what it returns.\n\n"
        "Withholding a number a tool did return is the same violation, run in "
        "reverse: a player who cannot see the roll cannot tell it from prose."
    )
    assert (
        "except when the pending action is a short- or long-rest declaration"
        not in pre_repair_invariant_section
    )
    assert "see the rest-declaration rule below" not in pre_repair_invariant_section
    # The unconditional fallback the finding named is present, unexcepted, pre-repair.
    assert (
        "call `campaign_status` or `character_sheet` and state what it returns.\n\n"
    ) in pre_repair_invariant_section


# -- the game-master address lane: routing by designator, never vocabulary -----


def test_gm_discussion_remainder_is_syntax_only_and_multilingual():
    """The designator is configured syntax: any token, any script, no keywords.
    """
    from narrator.interactions import gm_discussion_remainder as remainder

    assert remainder("@GM how do doom dice work?", "GM") == "how do doom dice work?"
    assert remainder(
        "@gm i had attacked and killed him. look at the transcript", "GM"
    ) == "i had attacked and killed him. look at the transcript"
    assert remainder("  @Gm: was that a critical?", "GM") == "was that a critical?"
    assert remainder("@GM", "GM") == ""
    assert remainder("@MJ j'avais déjà tué Rade", "MJ") == "j'avais déjà tué Rade"
    assert remainder("@ВЕД он уже мёртв", "ВЕД") == "он уже мёртв"
    assert remainder("@進行 彼はもう死んでいる", "進行") == "彼はもう死んでいる"
    # A designator never claims a longer word, only the configured token opens the
    # lane, and a mid-turn address does not: the act comes first, the floor holds.
    assert remainder("@GMX attack", "GM") is None
    assert remainder("@GM attack", "MJ") is None
    assert remainder("attack @GM", "GM") is None
    assert remainder("I attack Rade", "GM") is None
    assert remainder("@GM attack", "") is None


def test_the_gm_route_owes_the_gm_discussion_framing():
    from narrator.interactions import gm_discussion_policy
    from narrator.policy_types import turn_framing_for

    verdict = gm_discussion_policy()
    assert verdict.route == "gm"
    assert verdict.risk_category == "none"
    assert verdict.social_test_required is False
    assert verdict.trade_phase == ""
    assert turn_framing_for("gm") == "gm_discussion"


def test_turn_prompt_renders_the_gm_discussion_framing():
    rendered = prompt.turn_prompt(
        "why did the attack miss?", turn_framing="gm_discussion"
    )
    assert "out-of-fiction discussion" in rendered
    assert "do not roll" in rendered


def test_gm_discussion_tools_are_pure_reads_within_the_player_surface():
    assert policy.GM_DISCUSSION_TOOLS < policy.PLAYER_FACING_TOOLS
    assert set(policy.GM_DISCUSSION_TOOLS) == {
        "campaign_status", "character_options", "character_sheet", "skills",
    }


def test_the_meta_tool_refusal_is_an_allowlist(tmp_path):
    """Only the read tools pass a game-master-discussion turn; everything else --
    including a tool that does not exist yet -- is refused by default."""
    eng = _engine_with_stub(tmp_path, _StubAgent())
    assert eng._meta_tool_refusal("combat_attack") == ""  # noqa: SLF001 - off outside a meta turn
    eng._meta_turn_active = True  # noqa: SLF001
    for allowed in ("campaign_status", "character_options", "character_sheet", "skills"):
        assert eng._meta_tool_refusal(allowed) == ""  # noqa: SLF001
    for refused in (
        "combat_attack", "attribute_test", "doom_roll", "scene_commit",
        "npc_create", "inventory_update", "some_future_tool",
    ):
        assert eng._meta_tool_refusal(refused)  # noqa: SLF001


def test_the_focus_store_keeps_a_person_focus_while_the_person_is_present(tmp_path):
    """Phase 3: a ``scene_person`` focus survives across turns exactly as a
    ``canonical_npc`` focus does, and drops when the person leaves the render."""
    from narrator.interactions import InteractionTracker, read_trusted_scope
    from narrator.policy_types import InteractionCue, TurnPolicy

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    render = (
        "---\nlocation_id: the-eel-market\nsession: 1\n---\n\n## Present NPCs\n\n- rade\n\n"
        "## Other persons present\n\n- salt-magistrate-clerk: Salt Magistrate clerk\n"
    )
    (campaign / "scene.md").write_text(render, encoding="utf-8")
    (campaign / "state.json").write_text('{"npcs": {"rade": {"id": "rade", "name": "Rade", "status": "alive"}}}', encoding="utf-8")
    scope = read_trusted_scope(tmp_path)
    assert scope.present_person_ids == ("salt-magistrate-clerk",)
    tracker = InteractionTracker()
    cue = InteractionCue("scene_person", "salt-magistrate-clerk")
    policy = TurnPolicy(
        route="social", risk_category="none", interaction_cue=cue, focus_candidate=cue,
        scope=scope, retains_focus=True, clears_focus=False,
    )
    tracker.complete_delivery("market", policy, tmp_path, "What do you want with the bell?")
    assert tracker._channels["market"].focus.cue == cue  # noqa: SLF001
    assert tracker._channels["market"].focus.awaiting_reply is True  # noqa: SLF001

    # The person leaves the render: the focus goes with them on the next read.
    (campaign / "scene.md").write_text(render.replace("- salt-magistrate-clerk: Salt Magistrate clerk", "- None recorded."), encoding="utf-8")

    async def classify(text, *, scope, combat):
        return None

    import asyncio
    asyncio.run(tracker.policy("market", "hello", tmp_path, classify=classify))
    assert tracker._channels["market"].focus is None  # noqa: SLF001


def test_similarity_reads_every_script_and_never_calls_two_empties_identical():
    """§3.3. The old ``[a-z0-9]`` class normalized Cyrillic and CJK to the empty
    string, and ``SequenceMatcher("", "")`` is 1.0 -- every consecutive Japanese or
    Russian turn read as an exact repeat and fired the corrective re-invoke. This
    fails against the pre-fix tree at 1.0 == 1.0."""
    from narrator.engine import narration_similarity, normalize_for_similarity

    ja_a = "市場の橋の下で、鐘が静かに揺れている。潮が引いていく。"
    ja_b = "衛兵は無言で書類に判を押し、二人を通した。"
    ru_a = "Колокол молчит уже два дня, и торговцы начинают шептаться."
    ru_b = "Рилл поскальзывается на мокрых досках и падает в грязь."
    assert narration_similarity(ja_a, ja_b) < 0.5
    assert narration_similarity(ru_a, ru_b) < 0.5
    assert narration_similarity(ja_a, ja_a) == 1.0
    assert narration_similarity("", "") == 0.0
    assert narration_similarity("...", "The tide rolls in.") == 0.0
    # ASCII English output is byte-identical to the old normalization.
    sample = "Rill's rope-ladder, at LOW water -- 2 hours!"
    old = " ".join(part for part in __import__("re").split(r"[^a-z0-9]+", sample.lower()) if part)
    assert normalize_for_similarity(sample) == old
    assert normalize_for_similarity("maître") == "maître"  # accents survive now


def test_the_announcement_union_detects_verifies_and_never_duplicates_in_every_locale():
    """§3.6. The detector is derived from the same catalog string that renders the
    announcement, so for each shipped locale: the rendered line is detected with
    canonical fields, a wrong verdict in that language is flagged, and injection
    adds nothing beside a line already present -- and adds the localized line when
    it is missing."""
    from narrator.engine import (
        announced_rolls,
        format_roll_announcement,
        inject_missing_roll_announcements,
        verdict_mismatches,
    )

    fact = {"tool": "attribute_test", "character": "rill", "attribute": "STR",
            "total": 4, "target": 13, "outcome": "success"}
    for language in ("en", "fr", "de", "ja", "ru"):
        catalog = _load_catalog(language)
        line = format_roll_announcement(fact, "Rill", catalog)
        parsed = announced_rolls(line, catalog)
        assert len(parsed) == 1, (language, line)
        assert (parsed[0]["attribute"], parsed[0]["total"], parsed[0]["target"], parsed[0]["outcome"]) == (
            "STR", 4, 13, "success"
        ), (language, parsed)
        wrong = format_roll_announcement({**fact, "outcome": "failure"}, "Rill", catalog)
        kinds = [m["kind"] for m in verdict_mismatches(wrong, (fact,), catalog)]
        assert "verdict" in kinds, (language, wrong)
        amended, injected = inject_missing_roll_announcements(line, (fact,), {"rill": "Rill"}, catalog)
        assert injected == 0 and amended == line, language  # already announced: no duplicate
        amended, injected = inject_missing_roll_announcements("The mud shifts.", (fact,), {"rill": "Rill"}, catalog)
        assert injected == 1 and line in amended, language  # missing: the localized line


def test_helpless_recovery_reads_a_figure_in_any_script():
    """§1.4, the widened figure signal: any decimal digit counts as a stated total,
    which is this guard's own documented safe direction; the outcome-word arm is
    unchanged."""
    from narrator.engine import _states_helpless_recovery

    fact = {"outcome": "Scratched"}
    assert _states_helpless_recovery("Rill est de nouveau debout, avec 3 points.", fact)
    assert _states_helpless_recovery("リルは立ち上がった。残りは3。", fact)
    assert _states_helpless_recovery("Barely scratched, Rill stands.", fact)
    assert not _states_helpless_recovery("Rill lies still in the mud.", fact)
