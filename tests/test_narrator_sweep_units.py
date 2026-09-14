"""Test narrator sweep units.
"""

import asyncio
import contextlib

from narrator_units_shared import *  # noqa: F401,F403 -- the split's shared header


async def test_a_withheld_turn_without_error_posts_the_unsettled_notice_via_the_engine(tmp_path):
    """The middle branch, reached through the real engine rather than a constructed
    outcome: settle reports no error but the ledger stays unratified."""
    _seed_ledger(tmp_path, [{"seq": 1, "tool": "attribute_test"}])
    stub = _StubAgent("Unratified narration.")
    engine = _engine_with_stub(tmp_path, stub)

    async def settle_without_error(narration):
        return ("none", 0, "")

    engine._settle = settle_without_error  # noqa: SLF001 - test seam
    adapter = _RecordingAdapter()
    config = NarratorConfig(campaign_root=tmp_path)
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)
    await delivery.deliver(adapter, turn, outcome, config)

    assert outcome.error == ""
    assert adapter.posted == [("c1", config.withheld_notice)]


async def test_run_turn_withholds_when_the_settle_step_fails(tmp_path):
    """The barrier's engine half: a ledger the settler cannot close withholds the turn."""
    _seed_ledger(tmp_path, [{"seq": 1, "tool": "attribute_test"}])
    stub = _StubAgent("Unratified narration.")
    engine = _engine_with_stub(tmp_path, stub)

    async def failing_settle(narration):
        return ("failed", 2, "settler stub broke twice")

    engine._settle = failing_settle  # noqa: SLF001 - test seam
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    outcome = await engine.run_turn(turn)

    assert outcome.withheld is True
    assert outcome.ratified is False
    assert outcome.settle == "failed"
    assert outcome.settle_attempts == 2
    assert outcome.error == "settler stub broke twice"


async def test_the_settle_commit_path_runs_through_the_engines_own_client(tmp_path):
    """A commit outcome becomes an engine-authored scene_commit with the schema's args."""
    from narrator.settle import SettleCommit, SettleOutcome

    _seed_ledger(tmp_path, _full_debt())
    engine = _engine_with_stub(tmp_path, _StubAgent())
    settled = SettleOutcome(
        outcome=SettleCommit(
            kind="commit",
            public_summary="The crypt door yielded nothing; the keeper stays hidden.",
            visible_changes=["The party still lacks the keeper's location."],
            in_game_time_delta_minutes=10,
        )
    )
    async def stub_settle_once(prompt):
        return settled

    engine._settle_once = stub_settle_once  # noqa: SLF001
    calls: list[tuple[str, dict]] = []

    def fake_call_tool(name, arguments):
        calls.append((name, arguments))
        _seed_ledger(tmp_path, [])  # the server clears the ledger on success
        return "success"

    engine._call_tool = fake_call_tool  # noqa: SLF001

    kind, attempts, error = await engine._settle("Narration.")

    assert (kind, attempts, error) == ("commit", 1, "")
    assert calls[0][0] == "scene_commit"
    assert calls[0][1]["public_summary"].startswith("The crypt door")
    assert calls[0][1]["in_game_time_delta_minutes"] == 10


async def test_the_settle_waive_path_clears_the_ledger_on_the_record(tmp_path):
    from narrator.settle import SettleOutcome, SettleWaive

    _seed_ledger(tmp_path, _full_debt())
    engine = _engine_with_stub(tmp_path, _StubAgent())
    settled = SettleOutcome(
        outcome=SettleWaive(kind="waive", reason="The failed listen changed nothing.")
    )
    async def stub_settle_once(prompt):
        return settled

    engine._settle_once = stub_settle_once  # noqa: SLF001
    calls: list[tuple[str, dict]] = []

    def fake_call_tool(name, arguments):
        calls.append((name, arguments))
        _seed_ledger(tmp_path, [])
        return "success"

    engine._call_tool = fake_call_tool  # noqa: SLF001

    kind, attempts, error = await engine._settle("Narration.")

    assert (kind, attempts, error) == ("waive", 1, "")
    assert calls == [("ledger_settle", {"reason": "The failed listen changed nothing."})]


async def test_a_settler_returning_nothing_fails_closed_after_the_attempt_ceiling(tmp_path):
    _seed_ledger(tmp_path, _full_debt())
    engine = _engine_with_stub(tmp_path, _StubAgent())
    async def stub_settle_once(prompt):
        return None

    engine._settle_once = stub_settle_once  # noqa: SLF001
    engine._call_tool = lambda name, arguments: "success"  # noqa: SLF001

    kind, attempts, error = await engine._settle("Narration.")

    assert kind == "failed"
    assert attempts == 2
    assert error == "structured output returned no object"


def test_the_settle_prompt_transcribes_the_realized_stake():
    from narrator.settle import settle_prompt

    prompt = settle_prompt(_full_debt(), "The door stays shut.")
    assert "The party hears nothing and stays unaware." in prompt
    assert "The party hears the keeper moving below." not in prompt
    assert "listen at the crypt door" in prompt
    assert "The door stays shut." in prompt


def test_example_json_reproduces_the_settle_shape_paragraph_byte_for_byte():
    """Test example json reproduces the settle shape paragraph byte for byte.
    """
    from narrator.settle import SettleCommit, SettleWaive, settle_prompt
    from narrator.shape import example_json

    commit = '{"outcome": ' + example_json(SettleCommit) + "}"
    waive = '{"outcome": ' + example_json(SettleWaive) + "}"
    assert commit == (
        '{"outcome": {"kind": "commit", "public_summary": "...", '
        '"visible_changes": ["..."], "in_game_time_delta_minutes": 0}}'
    )
    assert waive == '{"outcome": {"kind": "waive", "reason": "..."}}'
    prompt = settle_prompt(_full_debt(), "The door stays shut.")
    assert commit in prompt
    assert waive in prompt


def test_the_settle_schema_bounds_every_string_the_grammar_can_generate():
    """A live probe recorded a settle request burning 8,192 tokens inside an unbounded
    string. maxLength is a grammar bound under guided decoding, so the schema itself
    must carry it on every free-text field."""
    import pytest as _pytest
    from pydantic import ValidationError

    from narrator.settle import SettleCommit, SettleWaive

    with _pytest.raises(ValidationError):
        SettleCommit(kind="commit", public_summary="x" * 601)
    with _pytest.raises(ValidationError):
        SettleCommit(kind="commit", public_summary="ok", visible_changes=["y" * 241])
    with _pytest.raises(ValidationError):
        SettleCommit(kind="commit", public_summary="ok", visible_changes=["z"] * 5)
    with _pytest.raises(ValidationError):
        SettleWaive(kind="waive", reason="r" * 601)
    # Boundary probes above pin rejection one past each bound; these pin the bounds
    # themselves, so a silent schema regression cannot pass on looser values an
    # earlier abandoned schema would also have rejected.
    with _pytest.raises(ValidationError):
        SettleCommit(kind="commit", public_summary="ok", in_game_time_delta_minutes=10081)
    schema = SettleCommit.model_json_schema()
    assert schema["properties"]["public_summary"]["maxLength"] == 600
    assert schema["properties"]["visible_changes"]["maxItems"] == 4
    assert schema["properties"]["visible_changes"]["items"]["maxLength"] == 240
    assert schema["properties"]["in_game_time_delta_minutes"]["maximum"] == 10080
    assert SettleWaive.model_json_schema()["properties"]["reason"]["maxLength"] == 600
    assert SettleCommit(kind="commit", public_summary="ok",
                        visible_changes=["y" * 240] * 4).visible_changes[0]


# -- the sweep: recovering facts the model no longer volunteers ----------------


async def test_the_sweep_record_path_writes_one_engine_authored_scene_commit(tmp_path):
    """A record outcome becomes a scene_commit with time 0 and a sweep-labeled id."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(
            kind="record",
            public_summary="The party marked the hiding place under the customs floor.",
            visible_changes=["A marked hiding place under the customs house floor."],
        )
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001 - test seam
    calls: list[tuple[str, dict, str]] = []

    def fake_call_tool(name, arguments, origin="settle"):
        calls.append((name, arguments, origin))
        return "success"

    engine._call_tool = fake_call_tool  # noqa: SLF001 - test seam

    kind = await engine._sweep("She marks the spot under the customs house floor.")

    assert kind == "record"
    assert calls[0][0] == "scene_commit"
    assert calls[0][1]["in_game_time_delta_minutes"] == 0
    assert calls[0][2] == "sweep"


async def test_the_sweep_declines_none_and_fails_open(tmp_path):
    """none is the common case; every failure shape costs one log entry, nothing more."""
    from narrator.sweep import SweepNone, SweepOutcome

    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[str] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append(name) or "success"
    )

    async def declining(prompt):
        return SweepOutcome(outcome=SweepNone(kind="none"))

    engine._sweep_once = declining  # noqa: SLF001
    assert await engine._sweep("Banter only.") == "none"

    async def empty(prompt):
        return None

    engine._sweep_once = empty  # noqa: SLF001
    assert await engine._sweep("Banter only.") == "failed"

    async def broken(prompt):
        raise RuntimeError("endpoint unreachable")

    engine._sweep_once = broken  # noqa: SLF001
    assert await engine._sweep("Banter only.") == "failed"
    assert calls == []


async def test_the_sweep_declines_an_unbacked_coin_claim(tmp_path):
    """Test the sweep declines an unbacked coin claim.
    """
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(kind="record", public_summary="You now have 40 coins.")
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    # No character file exists at all, so no real coin total can back this claim.

    kind = await engine._sweep("The trader nods. You now have 40 coins.")

    assert kind == "none"
    assert calls == []


async def test_the_sweep_ratifies_a_coin_claim_matching_the_real_character_sheet(tmp_path):
    """The same claim, true against the actual character sheet, ratifies."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    _write_character(tmp_path, "rill", name="Rill", coins=40, hp=9, hp_max=9)
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(kind="record", public_summary="You now have 40 coins.")
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    # Zero tool calls ran this turn -- the claim ratifies because it is true right
    # now, not because some tool happened to run. A purchase confirmed on an earlier
    # turn, restated accurately here, must ratify exactly the same way.
    engine._tool_events_this_turn = []  # noqa: SLF001

    kind = await engine._sweep("You now have 40 coins.")

    assert kind == "record"
    assert calls[0][0] == "scene_commit"


async def test_the_sweep_declines_a_fabricated_coin_claim_despite_an_unrelated_successful_tool_call(
    tmp_path,
):
    """BLOCKER-1 regression (independent audit, result-1.json).

    The prior guard treated "some tool call succeeded this turn" as proof a claimed
    figure was real. Reproduced end to end here exactly as the auditor found it: one
    unrelated, read-only, non-mutating ``character_sheet`` call succeeds, alongside a
    fabricated coin total, and the fabricated figure must still decline because
    nothing about that unrelated call backs *this* claim.
    """
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    _write_character(tmp_path, "rill", name="Rill", coins=3, hp=9, hp_max=9)
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(kind="record", public_summary="You now have 9001 coins.")
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    engine._tool_events_this_turn = [  # noqa: SLF001 - unrelated, read-only, successful
        {"tool": "character_sheet", "ok": True, "error": "", "event_id": 3}
    ]

    kind = await engine._sweep("You now have 9001 coins.")

    assert kind == "none"
    assert calls == []


async def test_the_sweep_ratifies_an_unrelated_record_despite_a_coin_mention_elsewhere_in_the_turn(
    tmp_path,
):
    """BLOCKER-2 regression (independent audit, result-1.json).

    The prior guard scanned the whole turn's narration for a coin/HP/item-shaped
    substring, so an NPC price quote elsewhere in the same turn could sink a
    genuinely unrelated record. Reproduced end to end: the delivered narration
    mentions "40 coins" as the merchant's price, but the record actually being
    ratified is about a hidden cellar and states no mechanical fact at all -- it
    must ratify regardless of what the rest of the turn's narration said.
    """
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(
            kind="record", public_summary="The party found the hidden cellar."
        )
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    engine._tool_events_this_turn = []  # noqa: SLF001

    narration = (
        "The merchant wants 40 coins for the blade. The party found the hidden "
        "cellar behind the shelf."
    )
    kind = await engine._sweep(narration)

    assert kind == "record"
    assert calls[0][0] == "scene_commit"


async def test_the_sweep_still_ratifies_ordinary_narration_with_no_backing_state(tmp_path):
    """The guard's absence-of-harm case: a plain durable fact still ratifies."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(
            kind="record",
            public_summary="The party marked the hiding place under the customs floor.",
        )
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    engine._tool_events_this_turn = []  # noqa: SLF001 - no tool call, no mechanical claim either

    kind = await engine._sweep("She marks the spot under the customs house floor.")

    assert kind == "record"
    assert calls[0][0] == "scene_commit"


async def test_the_sweep_declines_an_item_claim_with_no_equipment_change(tmp_path):
    """An item claim needs the equipment snapshot to actually differ."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    _write_character(tmp_path, "rill", name="Rill", coins=3, hp=9, hp_max=9, equipment=["rope"])
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(kind="record", public_summary="Rill picks up a lantern.")
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    engine._tool_events_this_turn = []  # noqa: SLF001
    # The "before" snapshot run_turn would have taken equals the current state --
    # no lantern was ever actually added to any character's equipment.
    engine._resources_before_turn = {  # noqa: SLF001
        "rill": {"id": "rill", "name": "Rill", "coins": 3, "hp": 9, "equipment": ("rope",)},
    }

    kind = await engine._sweep("Rill picks up a lantern.")

    assert kind == "none"
    assert calls == []


async def test_the_sweep_ratifies_an_item_claim_when_equipment_actually_changed(tmp_path):
    """The same claim, with equipment that actually differs from turn start, ratifies."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord

    _seed_ledger(tmp_path, [])
    _write_character(
        tmp_path, "rill", name="Rill", coins=3, hp=9, hp_max=9, equipment=["rope", "lantern"]
    )
    engine = _engine_with_stub(tmp_path, _StubAgent())
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = (  # noqa: SLF001 - test seam
        lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"
    )
    swept = SweepIntroducingOutcome(
        outcome=SweepIntroducingRecord(kind="record", public_summary="Rill picks up a lantern.")
    )

    async def stub_sweep_once(prompt):
        return swept

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    engine._tool_events_this_turn = []  # noqa: SLF001
    engine._resources_before_turn = {  # noqa: SLF001
        "rill": {"id": "rill", "name": "Rill", "coins": 3, "hp": 9, "equipment": ("rope",)},
    }

    kind = await engine._sweep("Rill picks up a lantern.")

    assert kind == "record"
    assert calls[0][0] == "scene_commit"


def test_guard_ratification_pins_both_fixture_cases_from_the_milestone():
    """The two cases the milestone names directly, against the guard alone."""
    from narrator.sweep import SweepNone, SweepOutcome, SweepRecord, guard_ratification

    rill = {"rill": _character_entry("rill", "Rill", coins=3)}

    unbacked = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="You now have 40 coins.")
    )
    guarded = guard_ratification(unbacked, characters_now=rill)
    assert guarded.outcome.kind == "none"

    backed = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="You now have 40 coins.")
    )
    ratified = guard_ratification(
        backed, characters_now={"rill": _character_entry("rill", "Rill", coins=40)}
    )
    assert ratified.outcome.kind == "record"

    already_none = SweepOutcome(outcome=SweepNone(kind="none"))
    assert guard_ratification(already_none, characters_now={}).outcome.kind == "none"


def test_guard_ratification_blocker_1_and_blocker_2_at_the_unit_level():
    """Both audit-reproduced defects, pinned directly against ``guard_ratification``.

    BLOCKER-1: a coin figure with no real backing value declines regardless of
    what else the caller reports -- the guard checks the *claimed character's own*
    real value, never a coarser "something happened" signal.
    BLOCKER-2: ``guard_ratification`` no longer accepts a raw turn narration at
    all -- only the outcome itself -- so an unrelated coin mention elsewhere in a
    turn has no path to reach the detector in the first place.
    """
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    rill = {"rill": _character_entry("rill", "Rill", coins=3, hp=9)}

    fabricated = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="You now have 9001 coins.")
    )
    guarded = guard_ratification(fabricated, characters_now=rill)
    assert guarded.outcome.kind == "none"

    unrelated = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="The party found the hidden cellar.")
    )
    ratified = guard_ratification(unrelated, characters_now=rill)
    assert ratified.outcome.kind == "record"


def test_guard_ratification_verifies_hp_and_item_claims():
    """HP claims verify like coins; item claims need that character's own equipment
    snapshot to actually differ."""
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    wrong_hp = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="You are down to 2 hit points.")
    )
    assert guard_ratification(
        wrong_hp, characters_now={"rill": _character_entry("rill", "Rill", hp=9)}
    ).outcome.kind == "none"

    true_hp = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="You are down to 2 hit points.")
    )
    assert guard_ratification(
        true_hp, characters_now={"rill": _character_entry("rill", "Rill", hp=2)}
    ).outcome.kind == "record"

    now = {"rill": _character_entry("rill", "Rill", equipment=("rope",))}
    before_unchanged = {"rill": _character_entry("rill", "Rill", equipment=("rope",))}
    before_changed = {"rill": _character_entry("rill", "Rill", equipment=())}

    item_unchanged = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rill picks up a lantern.")
    )
    assert guard_ratification(
        item_unchanged, characters_now=now, characters_before=before_unchanged
    ).outcome.kind == "none"

    item_changed = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rill picks up a lantern.")
    )
    assert guard_ratification(
        item_changed, characters_now=now, characters_before=before_changed
    ).outcome.kind == "record"


def test_guard_ratification_declines_blocker_6_pooled_state_misattribution():
    """BLOCKER-6 regression (independent audit, result-2.json), all three categories.

    An earlier repair checked a claimed figure against the party's pooled state --
    every character's values flattened into one tuple, or one whole-party boolean
    for equipment -- so a real figure attributed to the WRONG character still
    ratified as long as *some* character in the party had it. Reproduced with the
    auditor's own two-character shape: Rill and Ossa hold different real values,
    and a record attributing Rill's true figure to Ossa (or vice versa) must
    decline, because it is false for the character it actually names.
    """
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    characters_now = {
        "rill": _character_entry("rill", "Rill", coins=3, hp=9, equipment=("rope",)),
        "ossa": _character_entry("ossa", "Ossa", coins=100, hp=2, equipment=("dagger",)),
    }
    characters_before = {
        "rill": _character_entry("rill", "Rill", coins=3, hp=9, equipment=("rope",)),
        "ossa": _character_entry("ossa", "Ossa", coins=100, hp=2, equipment=("dagger",)),
    }

    # Coins: 3 is really Rill's total, falsely attributed to Ossa.
    misattributed_coins = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Ossa now has 3 coins.")
    )
    assert guard_ratification(
        misattributed_coins, characters_now=characters_now, characters_before=characters_before
    ).outcome.kind == "none"
    # The same figure, correctly attributed, still ratifies.
    correct_coins = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rill now has 3 coins.")
    )
    assert guard_ratification(
        correct_coins, characters_now=characters_now, characters_before=characters_before
    ).outcome.kind == "record"

    # HP: 2 is really Ossa's total, falsely attributed to Rill.
    misattributed_hp = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rill is down to 2 hit points.")
    )
    assert guard_ratification(
        misattributed_hp, characters_now=characters_now, characters_before=characters_before
    ).outcome.kind == "none"
    correct_hp = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Ossa is down to 2 hit points.")
    )
    assert guard_ratification(
        correct_hp, characters_now=characters_now, characters_before=characters_before
    ).outcome.kind == "record"

    # Equipment: only Ossa's equipment actually changed this turn; a claim naming
    # Rill must not ratify off the party-wide "something changed" signal.
    equipment_before = {
        "rill": _character_entry("rill", "Rill", equipment=("rope",)),
        "ossa": _character_entry("ossa", "Ossa", equipment=()),
    }
    equipment_now = {
        "rill": _character_entry("rill", "Rill", equipment=("rope",)),
        "ossa": _character_entry("ossa", "Ossa", equipment=("dagger",)),
    }
    misattributed_item = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rill picks up a dagger.")
    )
    assert guard_ratification(
        misattributed_item, characters_now=equipment_now, characters_before=equipment_before
    ).outcome.kind == "none"
    correct_item = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Ossa picks up a dagger.")
    )
    assert guard_ratification(
        correct_item, characters_now=equipment_now, characters_before=equipment_before
    ).outcome.kind == "record"


def test_stated_life_statuses_pins_polarity_negation_and_the_negative():
    """Test stated life statuses pins polarity negation and the negative.
    """
    from narrator.sweep import stated_life_statuses

    assert stated_life_statuses("Rade is alive beside a red cart.") == ["alive"]
    assert stated_life_statuses("Rade is dead") == ["dead"]
    assert stated_life_statuses("Rade is not dead.") == ["alive"]
    assert stated_life_statuses("Rade is no longer alive.") == ["dead"]
    assert stated_life_statuses("Rade is still alive.") == ["alive"]
    assert stated_life_statuses("Vessa killed Rade.") == ["dead"]
    assert stated_life_statuses("Rade was slain at his stall.") == ["dead"]
    assert stated_life_statuses("The tide rolls in.") == []
    assert stated_life_statuses("A dead end blocks the alley.") == []
    assert stated_life_statuses("They walk in the dead of night.") == []


def test_guard_ratification_declines_a_life_status_claim_state_contradicts():
    """Test guard ratification declines a life status claim state contradicts.
    """
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    dead_rade = {"rade": _npc_entry("rade", "Rade", status="dead")}
    resurrection = SweepOutcome(
        outcome=SweepRecord(
            kind="record",
            public_summary="Rade is alive beside a red cart.",
        )
    )
    assert guard_ratification(resurrection, npcs_now=dead_rade).outcome.kind == "none"

    # The mirror hallucination -- an unaudited death claim -- declines the same way.
    live_rade = {"rade": _npc_entry("rade", "Rade", status="alive")}
    unaudited_kill = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rade is dead.")
    )
    assert guard_ratification(unaudited_kill, npcs_now=live_rade).outcome.kind == "none"

    kill_verb = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Vessa killed Rade at his stall.")
    )
    assert guard_ratification(kill_verb, npcs_now=live_rade).outcome.kind == "none"


def test_guard_ratification_ratifies_a_life_status_claim_state_backs():
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    dead_rade = {"rade": _npc_entry("rade", "Rade", status="dead")}
    true_death = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rade is dead behind his stall.")
    )
    assert guard_ratification(true_death, npcs_now=dead_rade).outcome.kind == "record"

    # Fled and captured NPCs are alive: an alive claim about them still ratifies.
    fled_rade = {"rade": _npc_entry("rade", "Rade", status="fled")}
    alive_claim = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rade is alive, somewhere in the marshes.")
    )
    assert guard_ratification(alive_claim, npcs_now=fled_rade).outcome.kind == "record"


def test_guard_ratification_leaves_life_wording_about_no_recorded_npc_alone():
    """A life-status phrase naming no recorded NPC stays the model's call: it may
    describe a player character or pure fiction, and declining it would cost real
    records for a claim this guard cannot check."""
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    npcs = {"rade": _npc_entry("rade", "Rade", status="dead")}
    unrelated = SweepOutcome(
        outcome=SweepRecord(
            kind="record", public_summary="The old watchman is dead, the traders say."
        )
    )
    assert guard_ratification(unrelated, npcs_now=npcs).outcome.kind == "record"


def test_states_new_arrival_pins_the_live_sessions_own_wording():
    from narrator.sweep import states_new_arrival

    assert states_new_arrival("The players arrive at the Eel Market.")
    assert states_new_arrival("Vessa enters the drowned customs house.")
    assert states_new_arrival("The party reaches the road shrine at dusk.")
    assert not states_new_arrival("Rade lashes out with his knife.")
    assert not states_new_arrival("The tide rolls in.")


def test_guard_ratification_declines_a_claimed_arrival_unconditionally():
    """Test guard ratification declines a claimed arrival unconditionally.
    """
    from narrator.sweep import SweepOutcome, SweepRecord, guard_ratification

    arrival = SweepOutcome(
        outcome=SweepRecord(
            kind="record",
            public_summary="The players arrive at the Eel Market.",
            visible_changes=['Several covered booths border the square.'],
        )
    )
    assert guard_ratification(arrival).outcome.kind == "none"

    genuine = SweepOutcome(
        outcome=SweepRecord(kind="record", public_summary="Rade grumbles about the market tax.")
    )
    assert guard_ratification(genuine).outcome.kind == "record"


async def test_run_turn_sweeps_only_delivered_turns_that_left_no_mark(tmp_path):
    """The gate: ratified, narration present, and an existing record it did not touch."""
    _seed_ledger(tmp_path, [])
    scene = tmp_path / "campaign" / "scene.md"
    scene.write_text("---\nlocation_id: here\n---\nThe record.", encoding="utf-8")
    stub = _StubAgent("The bell tolls once.")
    engine = _engine_with_stub(tmp_path, stub)
    swept: list[str] = []

    async def stub_sweep(narration, resources_before=None):
        swept.append(narration)
        return "record"

    engine._sweep = stub_sweep  # noqa: SLF001 - test seam
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))

    # C2: the gate firing now dispatches rather than resolves synchronously, so
    # this turn's own outcome reports "pending", not the eventual "record", and
    # ``asyncio.ensure_future`` only schedules ``stub_sweep`` -- it has not
    # necessarily run even its first line yet. ``flush_pending_sweep`` awaits it
    # to completion, which is what makes ``swept`` and the patched ``_sweep_log``
    # entry both observable below.
    outcome = await engine.run_turn(turn)
    assert outcome.sweep == "pending"
    assert engine._sweep_log == ["pending"]  # noqa: SLF001
    await engine.flush_pending_sweep()
    assert swept == ["The bell tolls once."]
    assert engine._sweep_log == ["record"]  # noqa: SLF001 - patched in place, same index

    class _CommittingAgent(_StubAgent):
        async def invoke_async(self, text):
            scene.write_text("---\nlocation_id: here\n---\nChanged.", encoding="utf-8")
            return await super().invoke_async(text)

    engine_changed = _engine_with_stub(tmp_path, _CommittingAgent("A new entry lands."))
    engine_changed._sweep = stub_sweep  # noqa: SLF001
    outcome = await engine_changed.run_turn(turn)
    assert outcome.sweep == "skipped"

    scene.unlink()
    engine_absent = _engine_with_stub(tmp_path, _StubAgent("No record exists."))
    engine_absent._sweep = stub_sweep  # noqa: SLF001
    outcome = await engine_absent.run_turn(turn)
    assert outcome.sweep == "skipped"

    _seed_ledger(tmp_path, [{"seq": 1, "tool": "attribute_test"}])
    scene.write_text("---\nlocation_id: here\n---\nThe record.", encoding="utf-8")
    engine_withheld = _engine_with_stub(tmp_path, _StubAgent("Unratified."))

    async def failing_settle(narration):
        return ("failed", 2, "settler stub")

    engine_withheld._settle = failing_settle  # noqa: SLF001
    engine_withheld._sweep = stub_sweep  # noqa: SLF001
    outcome = await engine_withheld.run_turn(turn)
    assert outcome.sweep == "skipped"
    assert swept == ["The bell tolls once."]
    assert engine_withheld._sweep_log == ["skipped"]  # noqa: SLF001

    class _RaisingAgent(_StubAgent):
        async def invoke_async(self, text):
            raise RuntimeError("framework fault")

    engine_faulted = _engine_with_stub(tmp_path, _RaisingAgent())
    engine_faulted._sweep = stub_sweep  # noqa: SLF001
    outcome = await engine_faulted.run_turn(turn)
    assert outcome.sweep == "skipped" and outcome.withheld is True
    # One entry per turn holds on the fault path too, so per-turn sweep
    # dispositions never misalign in exactly the runs an operator diagnoses.
    assert engine_faulted._sweep_log == ["skipped"]  # noqa: SLF001
    # The tool log keeps the same contract on both paths. A probe reading it by
    # turn index would attribute one turn's tool calls to another if it did not.
    assert engine_faulted._tool_log == [[]]  # noqa: SLF001
    assert engine_withheld._tool_log == [[]]  # noqa: SLF001
    assert len(engine_changed._tool_log) == len(engine_changed._sweep_log) == 1  # noqa: SLF001


def test_the_sweep_outcome_union_makes_mixed_states_unrepresentable():
    """The settle probe's lesson holds: optional fields invite mixed states; a
    discriminated union forbids them, and every bound is a grammar bound."""
    import pytest as _pytest
    from pydantic import ValidationError

    from narrator.sweep import SweepNone, SweepOutcome, SweepRecord

    with _pytest.raises(ValidationError):
        SweepRecord(kind="record", public_summary="   ")
    with _pytest.raises(ValidationError):
        SweepRecord(kind="record", public_summary="x" * 601)
    with _pytest.raises(ValidationError):
        SweepRecord(kind="record", public_summary="ok", visible_changes=["y" * 241])
    with _pytest.raises(ValidationError):
        SweepRecord(kind="record", public_summary="ok", visible_changes=["z"] * 5)
    # Pydantic discards undeclared input rather than raising, matching the settle
    # models: a summary smuggled onto a decline never reaches the engine, because
    # the parsed object simply does not carry it.
    none_outcome = SweepNone(kind="none", public_summary="smuggled")
    assert not hasattr(none_outcome, "public_summary")
    with _pytest.raises(ValidationError):
        SweepOutcome(outcome={"kind": "record"})
    schema = SweepRecord.model_json_schema()
    assert schema["properties"]["public_summary"]["maxLength"] == 600
    assert schema["properties"]["visible_changes"]["maxItems"] == 4
    assert schema["properties"]["visible_changes"]["items"]["maxLength"] == 240
    # Swept facts come from delivered narration, so they are party knowledge by
    # construction: the schema must hold no hidden channel at all.
    assert "hidden_changes" not in SweepRecord.model_fields
    assert "in_game_time_delta_minutes" not in SweepRecord.model_fields


def test_example_json_reproduces_the_sweep_shape_paragraph_byte_for_byte():
    """Test example json reproduces the sweep shape paragraph byte for byte.
    """
    from narrator.shape import example_json
    from narrator.sweep import SweepIntroducingRecord, SweepNone, empty_roster, sweep_prompt

    record = '{"outcome": ' + example_json(SweepIntroducingRecord) + "}"
    decline = '{"outcome": ' + example_json(SweepNone) + "}"
    assert record == (
        '{"outcome": {"kind": "record", "public_summary": "...", '
        '"visible_changes": ["..."], "mentions": [{"entity_id": "...", '
        '"kind": "npc|character|object|exit", "claim": "none|alive|dead|absent|'
        'locked|unlocked|barred|open|closed"}], "claimed_coins": 40|null, '
        '"claimed_hp": 9|null, "introduced_persons": ["..."]}}'
    )
    assert decline == '{"outcome": {"kind": "none"}}'
    prompt = sweep_prompt("Ossa pockets the harbor-master's ring.", empty_roster())
    assert record in prompt
    assert decline in prompt


def test_example_json_raises_rather_than_render_an_unrecognized_field_type():
    """A field type this module has no rule for must fail loudly, not emit prompt
    text nobody verified -- the whole point of deriving the shape from the schema
    is that an unhandled shape is a build-time error, never a silent wrong prompt."""
    import pytest as _pytest
    from pydantic import BaseModel as _BaseModel

    from narrator.shape import example_json

    class _Unsupported(_BaseModel):
        flag: bool

    with _pytest.raises(TypeError):
        example_json(_Unsupported)


def test_the_sweep_prompt_states_both_shapes_and_copies_narration():
    from narrator.sweep import empty_roster, sweep_prompt

    prompt = sweep_prompt("Ossa pockets the harbor-master's ring.", empty_roster())
    assert "Ossa pockets the harbor-master's ring." in prompt
    assert '{"outcome": {"kind": "record", "public_summary": ' in prompt
    assert '{"outcome": {"kind": "none"}}' in prompt
    assert "record nothing the narration does not state" in prompt
    assert "that is the common case" in prompt


def test_the_sweep_prompt_names_fixed_physical_detail_as_a_durable_category():
    """Test the sweep prompt names fixed physical detail as a durable category.
    """
    from narrator.sweep import empty_roster, sweep_prompt

    prompt = sweep_prompt("A copper weather vane is mounted on an oak post.", empty_roster())
    assert "a fixed physical detail newly described" in prompt
    assert "an object's mounting, structure, or placement in the scene" in prompt
    # The anti-invention floor is untouched: still present, still declines banter.
    assert "Banter, atmosphere, and unresolved intentions are not durable." in prompt


def test_the_canon_digest_resources_block_is_absent_with_no_characters(tmp_path):
    """No character files yet (a fresh campaign) means no block, not an empty heading."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert "Party resources" not in digest.text
    assert digest.stats["resources_chars"] == 0
    assert digest.stats["truncated"]["resources"] is False


def test_a_bound_below_the_truncation_marker_reports_every_section_absent(tmp_path):
    """The bounds are configuration, so the measurement must survive a hostile one.

    ``_bounded`` retains ``bound`` minus the marker's length, which is negative below
    72 characters and clamps to zero. Zero retained characters means zero surviving
    sections, and the measurement must say so rather than divide by that zero.
    """
    from narrator.canon import render_digest

    digest = render_digest(_canon_root(tmp_path, 19), tmp_path, 10, 2600, 1200)

    assert set(_states(digest).values()) == {"absent"}
    facts = next(
        section
        for section in digest.stats["scene_sections"]
        if section["name"] == "Visible facts"
    )
    assert facts["items"] == 19 and facts["items_retained"] == 0


def test_the_skill_states_a_first_mention_item_is_established_not_verified():
    """This is prose policy, not a worked numeric exemplar like
    ``test_every_skill_roll_exemplar_agrees_with_classify_and_the_real_sheets`` above --
    there is no computed value to check the paragraph against -- so the pin is
    structural: the rule names the specific anti-pattern this defect produced, states
    the opposite instruction in words a reviewer can find, and a worked
    ``scene_commit`` call demonstrates the corrected shape as schema-valid JSON.
    """
    skill_text = (REPO_ROOT / "skills" / "bsh-gm" / "SKILL.md").read_text(encoding="utf-8")

    assert "is not a precondition to verify" in skill_text
    assert "Never refuse a first-mention item" in skill_text
    assert "establishes it" in skill_text

    match = _FIRST_MENTION_EXAMPLE_PATTERN.search(skill_text)
    assert match, "no worked scene_commit example follows the first-mention-item rule"
    call = json.loads(match.group(1))
    assert call["name"] == "scene_commit"
    assert "bone-clasp case" in call["arguments"]["public_summary"]
    assert call["arguments"]["visible_changes"], call


def test_the_first_mention_item_rule_would_have_failed_before_the_fix():
    """The same assertions, run against the exact pre-fix text, must fail.
    """
    pre_fix = (
        "A movement declaration or a position change can presuppose a fact the scene "
        "record already fixed: whether a door is locked, barred, or open, or whether a "
        "character already stands past a barrier nobody has recorded crossing. Before "
        "narrating passage through a barrier or any change in position, re-read the "
        "scene record's `## Objects` section for that barrier's own typed state -- "
        "never the declaration's assumption. When the declaration presupposes a state "
        "the record contradicts -- picking a lock the record already calls barred, "
        "closing a door nobody opened, a bolt thrown from one side swinging free from "
        "the other -- stop and let the structured decision planner clarify the actual "
        "state rather than adopt the presupposition and narrate past it.\n\n"
        "A barrier's state, changed durably: give the object a stable id, one of "
        "`locked`, `unlocked`, `barred`, `open`, or `closed`, and a note carrying the "
        "physical detail. Naming only `state` for an id already on record keeps that "
        "object's existing note; the record, not this turn's prose, is what the next "
        "turn reads back:\n\n"
        "```json\n"
        '{"name": "scene_commit", "arguments": {"public_summary": "The party finds the '
        'tower door barred from within.", "object_updates": {"tower-door": {"state": '
        '"barred", "note": "a heavy oak beam, not a lock"}}}}\n'
        "```\n\n"
        "A movement that will not finish this turn: open the clock and log this turn's "
        "own progress in the same call, id sorted by location:"
    )
    assert "is not a precondition to verify" not in pre_fix
    assert "Never refuse a first-mention item" not in pre_fix
    assert _FIRST_MENTION_EXAMPLE_PATTERN.search(pre_fix) is None


async def test_a_meta_turn_never_sweeps_and_never_becomes_the_comparison_point(tmp_path):
    """Out-of-fiction talk must not move the fiction: no sweep entry, no narration
    tail, no repetition comparison point -- and the tool freeze holds exactly for
    the main invoke."""
    from narrator.service import PreparedNarrationTurn

    _seed_ledger(tmp_path, [])
    scene = tmp_path / "campaign" / "scene.md"
    scene.write_text("---\nlocation_id: here\n---\nThe record.", encoding="utf-8")
    stub = _StubAgent("Out of fiction: the attack resolved and Rade died.")
    eng = _engine_with_stub(tmp_path, stub)
    freeze_during_invoke: list[bool] = []

    original_invoke = stub.invoke_async

    async def probing_invoke(text):
        freeze_during_invoke.append(eng._meta_turn_active)  # noqa: SLF001
        return await original_invoke(text)

    stub.invoke_async = probing_invoke
    swept: list[str] = []

    async def stub_sweep(narration, resources_before=None):
        swept.append(narration)
        return "record"

    eng._sweep = stub_sweep  # noqa: SLF001 - test seam
    turn = PreparedNarrationTurn(
        InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "why did he die?")),
        None,
        turn_framing="gm_discussion",
        meta=True,
    )

    outcome = await eng.run_turn(turn)
    assert outcome.withheld is False
    assert outcome.sweep == "skipped"
    assert swept == []
    assert freeze_during_invoke == [True]
    assert eng._meta_turn_active is False  # noqa: SLF001 - cleared with the guard
    # Not a blanket ``eng._narration == {}``: ``last_declaration`` records every turn's
    # inbound text unconditionally, meta or not, so the channel entry exists -- only
    # the two fields a meta turn must leave untouched are asserted here.
    channel = eng._narration.get("c1")  # noqa: SLF001
    assert channel is None or channel.last_delivered == ""
    assert channel is None or channel.uncommitted == []

    # The identical turn without the flag keeps every fiction-shaped afterchannel:
    # this is the arm that proves the assertions above bite.
    fiction = _engine_with_stub(tmp_path, _StubAgent("The bell tolls once."))
    fiction._sweep = stub_sweep  # noqa: SLF001
    plain = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "look"))
    outcome = await fiction.run_turn(plain)
    assert outcome.sweep == "pending"  # C2: dispatched, not resolved synchronously
    # asyncio.ensure_future only schedules stub_sweep; flush it to completion
    # before checking that it ran.
    await fiction.flush_pending_sweep()
    assert swept == ["The bell tolls once."]
    assert fiction._narration["c1"].last_delivered == "The bell tolls once."  # noqa: SLF001


def test_the_sweep_prompt_places_the_roster_block_before_the_shape_paragraph():
    from narrator.sweep import sweep_prompt

    prompt = sweep_prompt(_MENTION_NARRATION, _roster(persons={"clerk": "Clerk"}))
    instruction = prompt.index("Decide whether that narration")
    roster_at = prompt.index("Recorded in the scene, by identifier:")
    shape_at = prompt.index("Answer with one JSON object")
    assert instruction < roster_at < shape_at
    assert "  people: rade (npc, dead), orso-pell (npc, alive), clerk (person)" in prompt
    assert "  party: ossa, rill" in prompt
    assert "  objects: tower-door (barred)" in prompt
    assert "  exits: the-road-shrine" in prompt
    assert '"mentions": [{"entity_id"' in prompt
    assert "introduced_persons: a person the narration names or describes" in prompt
    assert "one individual per entry; never a group, a crowd, or a kind of people" in prompt
    assert "a person newly named or described in the scene, " in prompt
    assert '"introduced_persons": ["..."]' in prompt
    # An empty list is rendered as such, never omitted.
    from narrator.sweep import empty_roster

    assert "  people: (none)" in sweep_prompt(_MENTION_NARRATION, empty_roster())


def test_the_sweep_schema_answers_in_prompt_order():
    from narrator.sweep import SweepIntroducingRecord

    # Guided decoding answers in schema order: the record text before the new fields.
    assert list(SweepIntroducingRecord.model_json_schema()["properties"]) == [
        "kind", "public_summary", "visible_changes", "mentions", "claimed_coins", "claimed_hp",
        "introduced_persons",
    ]


def test_validated_mentions_drops_off_roster_identifiers_and_duplicates():
    from narrator.sweep import SweepMention, validated_mentions

    kept = validated_mentions(
        [
            SweepMention(entity_id="rade", kind="npc", claim="dead"),
            SweepMention(entity_id="rade", kind="npc", claim="alive"),  # duplicate key
            SweepMention(entity_id="ghost", kind="npc", claim="alive"),  # hallucinated
            SweepMention(entity_id="ossa", kind="npc"),  # wrong kind for a character
            SweepMention(entity_id="ossa", kind="character"),
            SweepMention(entity_id="clerk", kind="npc"),  # a person counts as a person-kind npc
            SweepMention(entity_id="tower-door", kind="object", claim="open"),
            SweepMention(entity_id="the-road-shrine", kind="exit"),
        ],
        _roster(persons={"clerk": "Clerk"}),
    )
    assert [(m.kind, m.entity_id, m.claim) for m in kept] == [
        ("npc", "rade", "dead"),
        ("character", "ossa", "none"),
        ("npc", "clerk", "none"),
        ("object", "tower-door", "open"),
        ("exit", "the-road-shrine", "none"),
    ]


def test_introduced_person_names_drops_people_the_roster_already_records():
    """An NPC or party member named in a label, by slug or as a whole word inside an
    epithet, is not an introduction. A recorded person or an authored name passes
    through to the server, which converges it (``resolve_person_label``)."""
    from narrator.sweep import introduced_person_names

    roster = _roster(persons={"clerk": "Clerk"}, npc_names={"rade": "Rade", "orso-pell": "Orso Pell"})
    names = introduced_person_names(
        ["Salt Magistrate clerk", "Rade", "ossa", "  ", "salt magistrate clerk", "Orso Pell",
         "Rade, the fishmonger", "Ossa the barbarian", "the clerk", "Sera Vane, the bell-keeper"],
        roster,
    )
    assert names == ["Salt Magistrate clerk", "the clerk", "Sera Vane, the bell-keeper"]


def test_guard_ratification_declines_a_mention_resurrection_in_five_languages():
    """Test guard ratification declines a mention resurrection in five languages.
    """
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
    )

    dead_rade = {"rade": _npc_entry("rade", "Rade", status="dead")}
    alive_mention = [SweepMention(entity_id="rade", kind="npc", claim="alive")]
    for language, text in _RESURRECTIONS.items():
        with_mention = SweepEntityOutcome(
            outcome=SweepEntityRecord(kind="record", public_summary=text, mentions=alive_mention)
        )
        assert guard_ratification(with_mention, npcs_now=dead_rade).outcome.kind == "none", language

        regex_only = SweepOutcome(outcome=SweepRecord(kind="record", public_summary=text))
        expected = "none" if language == "en" else "record"
        assert guard_ratification(regex_only, npcs_now=dead_rade).outcome.kind == expected, language

    # The mention path verifies in both directions and leaves an unknown id alone.
    true_death = SweepEntityOutcome(
        outcome=SweepEntityRecord(
            kind="record", public_summary="Раде мёртв.",
            mentions=[SweepMention(entity_id="rade", kind="npc", claim="dead")],
        )
    )
    assert guard_ratification(true_death, npcs_now=dead_rade).outcome.kind == "record"
    unknown = SweepEntityOutcome(
        outcome=SweepEntityRecord(
            kind="record", public_summary="Le gardien est vivant.",
            mentions=[SweepMention(entity_id="watchman", kind="npc", claim="alive")],
        )
    )
    assert guard_ratification(unknown, npcs_now=dead_rade).outcome.kind == "record"


def test_guard_ratification_declines_an_object_state_mention_the_record_contradicts():
    """The M6 contradiction class, checked at the sweep for the first time: a door
    the record holds barred narrated open declines; the matching state ratifies; an
    object the record does not know, or a mention asserting no state, is left alone."""
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        guard_ratification,
    )

    objects = {"tower-door": "barred"}

    def record(claim: str, object_id: str = "tower-door"):
        return SweepEntityOutcome(
            outcome=SweepEntityRecord(
                kind="record", public_summary="La porte de la tour.",
                mentions=[SweepMention(entity_id=object_id, kind="object", claim=claim)],
            )
        )

    assert guard_ratification(record("open"), objects_now=objects).outcome.kind == "none"
    assert guard_ratification(record("barred"), objects_now=objects).outcome.kind == "record"
    assert guard_ratification(record("none"), objects_now=objects).outcome.kind == "record"
    assert guard_ratification(record("open", "cellar-hatch"), objects_now=objects).outcome.kind == "record"
    assert guard_ratification(record("open"), objects_now=None).outcome.kind == "record"


def test_guard_ratification_resolves_a_claim_subject_from_a_single_character_mention():
    """A coin claim whose text names nobody, in a two-character party: the regex path
    cannot resolve the subject and declines (the pooled-state defect's fix), while a
    single ``character`` mention resolves it and the figure is verified against that
    character alone. Two character mentions stay ambiguous and decline."""
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
    )

    party = {
        "rill": {"id": "rill", "name": "Rill", "coins": 14, "hp": 9, "equipment": []},
        "ossa": {"id": "ossa", "name": "Ossa", "coins": 3, "hp": 11, "equipment": []},
    }
    text = "He now has 14 coins after the trade."
    regex_only = SweepOutcome(outcome=SweepRecord(kind="record", public_summary=text))
    assert guard_ratification(regex_only, characters_now=party).outcome.kind == "none"

    def with_mentions(*ids: str):
        return SweepEntityOutcome(
            outcome=SweepEntityRecord(
                kind="record", public_summary=text,
                mentions=[SweepMention(entity_id=i, kind="character") for i in ids],
            )
        )

    assert guard_ratification(with_mentions("rill"), characters_now=party).outcome.kind == "record"
    assert guard_ratification(with_mentions("ossa"), characters_now=party).outcome.kind == "none"
    assert guard_ratification(with_mentions("rill", "ossa"), characters_now=party).outcome.kind == "none"


async def test_the_sweep_passes_the_roster_and_commits_introduced_persons(tmp_path):
    """Slice A and B through the engine: the roster reaches the prompt, an off-roster
    mention is dropped before the guard, and a person the narration introduced rides
    the one ratified commit through the same ``persons`` argument the model has --
    minus anyone the roster already records."""
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord, SweepMention

    _entity_campaign(tmp_path)
    engine = _engine_with_stub(tmp_path, _StubAgent())
    engine._resources_before_turn = {}  # noqa: SLF001
    prompts: list[str] = []

    async def stub_sweep_once(prompt):
        prompts.append(prompt)
        return SweepIntroducingOutcome(
            outcome=SweepIntroducingRecord(
                kind="record",
                public_summary="A Salt Magistrate clerk walks toward Rill past Rade's body.",
                visible_changes=['A city records officer approaches Rill.'],
                mentions=[
                    SweepMention(entity_id="rade", kind="npc", claim="dead"),
                    SweepMention(entity_id="ghost", kind="npc", claim="alive"),  # hallucinated
                    SweepMention(entity_id="rill", kind="character"),
                ],
                introduced_persons=["Salt Magistrate clerk", "Ossa"],
            )
        )

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"  # noqa: SLF001

    assert await engine._sweep("A clerk approaches.") == "record"
    assert "Recorded in the scene, by identifier:" in prompts[0]
    assert "  people: rade (npc, dead)" in prompts[0]
    assert "  party: ossa, rill" in prompts[0]
    assert "  objects: tower-door (barred)" in prompts[0]
    assert calls[0][1]["persons"] == [{"name": "Salt Magistrate clerk"}]
    assert calls[0][2] == "sweep"


async def test_the_sweep_declines_a_mention_resurrection_end_to_end(tmp_path):
    from narrator.sweep import SweepEntityOutcome, SweepEntityRecord, SweepMention

    _entity_campaign(tmp_path)
    engine = _engine_with_stub(tmp_path, _StubAgent())
    engine._resources_before_turn = {}  # noqa: SLF001

    async def stub_sweep_once(prompt):
        return SweepEntityOutcome(
            outcome=SweepEntityRecord(
                kind="record",
                public_summary="Раде жив рядом с красной телегой.",
                mentions=[SweepMention(entity_id="rade", kind="npc", claim="alive")],
            )
        )

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    calls: list[str] = []
    engine._call_tool = lambda name, arguments, origin="settle": calls.append(name) or "success"  # noqa: SLF001
    assert await engine._sweep("Раде жив.") == "none"
    assert calls == []


def test_the_entity_roster_reads_recorded_state_only(tmp_path):
    from narrator.engine import _entity_roster

    _entity_campaign(tmp_path)
    roster = _entity_roster(tmp_path)
    assert roster["npcs"] == {"rade": "dead"}
    assert roster["characters"] == {"ossa": "Ossa", "rill": "Rill"}
    assert roster["objects"] == {"tower-door": "barred"}
    assert roster["exits"] == ["the-road-shrine"]
    assert roster["persons"] == {}
    assert roster["npc_names"] == {"rade": "Rade"}
    assert roster["summary"] == "The party stands in the eel market at low water."
    # Fail-open: an absent campaign yields an empty roster, never an exception.
    assert _entity_roster(tmp_path / "nowhere")["npcs"] == {}


def test_an_entity_record_declines_the_partys_arrival_but_not_a_strangers():
    """Test an entity record declines the partys arrival but not a strangers.
    """
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepIntroducingOutcome,
        SweepIntroducingRecord,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
        party_arrival,
    )

    party = {
        "vessa": {"id": "vessa", "name": "Vessa", "coins": 1, "hp": 5, "equipment": []},
        "rill": {"id": "rill", "name": "Rill", "coins": 1, "hp": 5, "equipment": []},
    }

    def entity(text, **fields):
        return SweepIntroducingOutcome(
            outcome=SweepIntroducingRecord(kind="record", public_summary=text, **fields)
        )

    stranger = entity(
        "A Salt Magistrate clerk arrives and observes Rill.",
        introduced_persons=["a Salt Magistrate clerk in a salt-stained coat"],
    )
    assert guard_ratification(stranger, characters_now=party).outcome.kind == "record"
    assert guard_ratification(entity("The party reaches the bell tower."), characters_now=party).outcome.kind == "none"
    assert guard_ratification(entity("The players arrive at the Eel Market."), characters_now=party).outcome.kind == "none"
    assert guard_ratification(entity("Vessa enters the drowned customs house."), characters_now=party).outcome.kind == "none"
    assert guard_ratification(entity("You enter the road shrine at dusk."), characters_now=party).outcome.kind == "none"


    assert guard_ratification(entity("You reach the road shrine at dusk."), characters_now=party).outcome.kind == "record"
    assert guard_ratification(entity("Rade enters, grumbling about the tax."), characters_now=party).outcome.kind == "record"
    # A plain entity record (Slice A schema) gets the same subject reading.
    plain = SweepEntityOutcome(outcome=SweepEntityRecord(kind="record", public_summary="The clerk arrives."))
    assert guard_ratification(plain, characters_now=party).outcome.kind == "record"
    # The legacy record stays unconditional: yesterday's pin, unchanged.
    legacy = SweepOutcome(outcome=SweepRecord(kind="record", public_summary="The clerk arrives."))
    assert guard_ratification(legacy, characters_now=party).outcome.kind == "none"

    assert party_arrival("Orso Pell arrives. The party reaches the tower.", ["Rill"]) is True
    assert party_arrival("Orso Pell arrives and greets Rill.", ["Rill"]) is False
    assert party_arrival("Rill reaches the tower.", ["Rill"]) is True


def test_an_entity_record_keeps_its_facts_when_the_partys_arrival_is_its_headline():
    """Test an entity record keeps its facts when the partys arrival is its headline.
    """
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
    )

    party = {"rill": {"id": "rill", "name": "Rill", "coins": 1, "hp": 5, "equipment": []}}
    current = "The party stands in the eel market at low water."

    def entity(summary, changes):
        return SweepEntityOutcome(
            outcome=SweepEntityRecord(kind="record", public_summary=summary, visible_changes=changes)
        )

    salvaged = guard_ratification(
        entity("The party reaches the bell tower.", [
            "A copper weather vane is mounted on an oak post.",
            "The party reaches the top of the tower.",
        ]),
        characters_now=party, scene_summary=current,
    ).outcome
    assert salvaged.kind == "record"
    assert salvaged.public_summary == current
    assert salvaged.visible_changes == ["A copper weather vane is mounted on an oak post."]

    all_arrival = entity("The party reaches the bell tower.", ["Rill enters the gallery."])
    assert guard_ratification(all_arrival, characters_now=party, scene_summary=current).outcome.kind == "none"
    no_summary = entity("The party reaches the bell tower.", ["The bell hangs from a frame."])
    assert guard_ratification(no_summary, characters_now=party, scene_summary="").outcome.kind == "none"
    no_changes = entity("The party reaches the bell tower.", [])
    assert guard_ratification(no_changes, characters_now=party, scene_summary=current).outcome.kind == "none"
    # The salvaged record still passes the remaining checks: an unbacked figure in a
    # surviving change declines it as any other record.
    fabricated = entity("The party reaches the bell tower.", ["Rill now has 40 coins."])
    assert guard_ratification(fabricated, characters_now=party, scene_summary=current).outcome.kind == "none"
    legacy = SweepOutcome(outcome=SweepRecord(kind="record", public_summary="The party reaches the bell tower.",
                                              visible_changes=["The bell hangs from a frame."]))
    assert guard_ratification(legacy, characters_now=party, scene_summary=current).outcome.kind == "none"


def test_guard_ratification_verifies_a_typed_coin_or_hp_claim_in_any_language():
    """spec-language-independent-guards.md §3.5, the numeric half. A French record
    stating a coin total reads nothing through ``_COIN_PATTERN``; the typed
    ``claimed_coins`` the model extracted verifies against the real sheet, with the
    subject resolved by the character mention -- union with the regex, never
    replacement. The pre-fix state is the ``SweepRecord`` branch: the same French text
    ratifies unverified."""
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
    )

    party = {"rill": {"id": "rill", "name": "Rill", "coins": 5, "hp": 9, "equipment": []}}
    text = "Rill a maintenant 40 pièces de cuivre après l'échange."
    rill = [SweepMention(entity_id="rill", kind="character")]

    def entity(**fields):
        return SweepEntityOutcome(outcome=SweepEntityRecord(kind="record", public_summary=text, mentions=rill, **fields))

    assert guard_ratification(SweepOutcome(outcome=SweepRecord(kind="record", public_summary=text)), characters_now=party).outcome.kind == "record"
    assert guard_ratification(entity(claimed_coins=40), characters_now=party).outcome.kind == "none"
    assert guard_ratification(entity(claimed_coins=5), characters_now=party).outcome.kind == "record"
    hp = "Il ne reste à Rill que 3 points de vie."
    wounded = SweepEntityOutcome(outcome=SweepEntityRecord(kind="record", public_summary=hp, mentions=rill, claimed_hp=3))
    assert guard_ratification(wounded, characters_now=party).outcome.kind == "none"
    # A typed figure with no resolvable subject declines rather than guessing. (A
    # Latin name inside French text still resolves through the regex -- "Rill" is
    # "Rill" in any Latin-script language -- so the subject-less case names nobody.)
    nobody = SweepEntityOutcome(outcome=SweepEntityRecord(kind="record", public_summary="Il a maintenant 40 pièces de cuivre.", claimed_coins=5))
    two = {**party, "ossa": {"id": "ossa", "name": "Ossa", "coins": 5, "hp": 9, "equipment": []}}
    assert guard_ratification(nobody, characters_now=two).outcome.kind == "none"
    # Regex and field disagreeing: every stated figure must match.
    mixed = SweepEntityOutcome(outcome=SweepEntityRecord(kind="record", public_summary="Rill now has 5 coins.", mentions=rill, claimed_coins=40))
    assert guard_ratification(mixed, characters_now=party).outcome.kind == "none"


def test_the_digit_tripwire_declines_an_unextracted_figure_about_a_person_only():
    """spec §3.2, narrowed: a digit in a record naming a party member, with no figure
    extracted by regex or field, is unverified and declines; a digit about the world
    -- the authored tide-table clue -- ratifies, because the unnarrowed rule would
    decline exactly the fact the continuity probe plants."""
    from narrator.sweep import (
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        SweepOutcome,
        SweepRecord,
        guard_ratification,
    )

    party = {"rill": {"id": "rill", "name": "Rill", "coins": 5, "hp": 9, "equipment": []}}
    rill = [SweepMention(entity_id="rill", kind="character")]
    hidden_figure = SweepEntityOutcome(outcome=SweepEntityRecord(
        kind="record", public_summary="Rill compte 40 pièces dans sa bourse.", mentions=rill))
    assert guard_ratification(hidden_figure, characters_now=party).outcome.kind == "none"
    tide = SweepEntityOutcome(outcome=SweepEntityRecord(
        kind="record", public_summary="The tide table says high water comes 2 hours after sunset."))
    assert guard_ratification(tide, characters_now=party).outcome.kind == "record"
    tide_with_rill = SweepEntityOutcome(outcome=SweepEntityRecord(
        kind="record", public_summary="Rill reads the tide table: high water 2 hours after sunset.", mentions=rill))
    # Names a party member and carries a digit the model did not extract: declines.
    # The cost is one fail-open sweep; the spec's error budget (C5) prefers it.
    assert guard_ratification(tide_with_rill, characters_now=party).outcome.kind == "none"
    legacy = SweepOutcome(outcome=SweepRecord(kind="record", public_summary="Rill compte 40 pièces dans sa bourse."))
    assert guard_ratification(legacy, characters_now=party).outcome.kind == "record"  # legacy path unchanged


def test_an_absent_claim_departs_a_person_and_never_contradicts_an_object():
    """Open question 3, the persons half: ``absent`` on a recorded person is a
    departure the sweep may drive; on a recorded NPC it is ignored; and it is not a
    barrier state, so the object check never reads it."""
    from narrator.sweep import (
        OBJECT_CLAIMS,
        SweepEntityOutcome,
        SweepEntityRecord,
        SweepMention,
        departed_person_ids,
        guard_ratification,
    )

    roster = _roster(persons={"vell-the-eel-seller": "Vell the eel-seller"})
    mentions = [
        SweepMention(entity_id="vell-the-eel-seller", kind="npc", claim="absent"),
        SweepMention(entity_id="rade", kind="npc", claim="absent"),  # an NPC: tool-driven presence, ignored
        SweepMention(entity_id="vell-the-eel-seller", kind="npc", claim="absent"),  # duplicate
    ]
    assert departed_person_ids(mentions, roster) == ["vell-the-eel-seller"]
    assert "absent" not in OBJECT_CLAIMS
    door_absent = SweepEntityOutcome(outcome=SweepEntityRecord(
        kind="record", public_summary="The door is gone from the frame.",
        mentions=[SweepMention(entity_id="tower-door", kind="object", claim="absent")],
    ))
    assert guard_ratification(door_absent, objects_now={"tower-door": "barred"}).outcome.kind == "record"


async def test_the_sweep_passes_departed_persons_into_the_scene_commit(tmp_path):
    from narrator.sweep import SweepIntroducingOutcome, SweepIntroducingRecord, SweepMention

    _entity_campaign(tmp_path)
    state = json.loads((tmp_path / "campaign" / "state.json").read_text(encoding="utf-8"))
    state["scene"]["persons"] = {"vell-the-eel-seller": {"id": "vell-the-eel-seller", "name": "Vell the eel-seller"}}
    (tmp_path / "campaign" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    engine = _engine_with_stub(tmp_path, _StubAgent())
    engine._resources_before_turn = {}  # noqa: SLF001

    async def stub_sweep_once(prompt):
        return SweepIntroducingOutcome(outcome=SweepIntroducingRecord(
            kind="record", public_summary="Vell shutters her stall and walks off toward the customs house.",
            mentions=[SweepMention(entity_id="vell-the-eel-seller", kind="npc", claim="absent")],
        ))

    engine._sweep_once = stub_sweep_once  # noqa: SLF001
    calls: list[tuple[str, dict, str]] = []
    engine._call_tool = lambda name, arguments, origin="settle": calls.append((name, arguments, origin)) or "success"  # noqa: SLF001
    assert await engine._sweep("Vell walks off.") == "record"
    assert calls[0][1]["departed_persons"] == ["vell-the-eel-seller"]


# -- C2: the sweep backgrounded with a next-turn barrier ----------------------


async def test_a_dispatched_sweeps_resources_before_snapshot_survives_the_next_turns_reset(tmp_path):
    """The subtlety ``_sweep``'s own docstring names: a backgrounded sweep must
    verify against the snapshot from *its own* turn, not whatever
    ``self._resources_before_turn`` holds by the time it actually runs -- which, for
    a sweep still pending when the next turn starts, is already that later turn's
    own snapshot. ``_dispatch_sweep`` captures its turn's snapshot by value at
    dispatch time for exactly this reason."""
    engine = _engine_with_stub(tmp_path, _StubAgent())
    received_before: list[dict] = []

    async def stub_sweep(narration, resources_before=None):
        received_before.append(resources_before)
        return "record"

    engine._sweep = stub_sweep  # noqa: SLF001 - test seam
    turn_one_snapshot = {"rill": {"coins": 5, "hp": 9, "equipment": ()}}
    engine._resources_before_turn = turn_one_snapshot  # noqa: SLF001

    engine._dispatch_sweep("Turn one's narration.")  # noqa: SLF001

    # Simulate the next turn's own top-of-run_turn reset, before the pending sweep
    # has resolved -- the exact race this test exists to prove is handled.
    engine._resources_before_turn = {"rill": {"coins": 40, "hp": 9, "equipment": ()}}  # noqa: SLF001

    await engine.flush_pending_sweep()

    assert received_before == [turn_one_snapshot]


async def test_sweep_log_index_alignment_holds_across_consecutive_dispatching_turns(tmp_path):
    """Every ``_sweep_log`` entry must still mean "this turn's own sweep" after C2,
    even when several turns in a row each dispatch one before the previous
    resolves is checked -- ``run_turn``'s own top-of-turn ``_resolve_pending_sweep``
    call is what keeps at most one ever pending, per this module's own
    single-turn-at-a-time guarantee (``engine.py``'s module docstring)."""
    _seed_ledger(tmp_path, [])
    scene = tmp_path / "campaign" / "scene.md"
    scene.write_text("---\nlocation_id: here\n---\nThe record.", encoding="utf-8")
    engine = _engine_with_stub(tmp_path, _StubAgent("Narration."))

    calls: list[str] = []

    async def stub_sweep(narration, resources_before=None):
        calls.append(narration)
        return "record"

    engine._sweep = stub_sweep  # noqa: SLF001 - test seam

    for i in range(3):
        turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", f"look {i}"))
        outcome = await engine.run_turn(turn)
        assert outcome.sweep == "pending"

    # Three turns dispatched, none flushed yet but for the two ``run_turn`` itself
    # resolved at each later turn's own top -- so exactly the first two have
    # already patched from "pending" to "record", and the third (still in flight)
    # has not.
    assert engine._sweep_log[:2] == ["record", "record"]  # noqa: SLF001
    assert engine._sweep_log[2] == "pending"  # noqa: SLF001

    await engine.flush_pending_sweep()
    assert engine._sweep_log == ["record", "record", "record"]  # noqa: SLF001
    assert len(calls) == 3


async def test_flush_pending_sweep_applies_the_result(tmp_path):
    """The reason ``flush_pending_sweep`` exists: a caller that awaits it before
    shutdown gets the sweep's real disposition instead of losing it."""
    engine = _engine_with_stub(tmp_path, _StubAgent())

    async def stub_sweep(narration, resources_before=None):
        return "record"

    engine._sweep = stub_sweep  # noqa: SLF001 - test seam
    engine._dispatch_sweep("Narration.")  # noqa: SLF001
    assert engine._sweep_log == ["pending"]  # noqa: SLF001

    await engine.flush_pending_sweep()

    assert engine._sweep_log == ["record"]  # noqa: SLF001
    assert engine._pending_sweep is None  # noqa: SLF001


async def test_stop_alone_cancels_a_still_pending_sweep_instead_of_losing_track_of_it(tmp_path):
    """The documented, accepted trade-off for a caller that does not flush first
    (``stop()``'s own docstring, and ``docs/context-and-model-call-architecture-
    review.md``'s C2 entry): the pending task is cancelled cleanly, not abandoned
    to a "Task was destroyed but it is pending" warning, and the fact it would
    have committed is lost -- ``NarratorService.run()``'s own shutdown flushes
    first for exactly this reason."""
    started = asyncio.Event()

    async def stub_sweep(narration, resources_before=None):
        started.set()
        await asyncio.Event().wait()  # blocks until cancelled; never set
        return "record"  # pragma: no cover - stop() cancels before this is reached

    engine = _engine_with_stub(tmp_path, _StubAgent())
    engine._sweep = stub_sweep  # noqa: SLF001 - test seam
    engine._dispatch_sweep("Narration.")  # noqa: SLF001
    task, _index = engine._pending_sweep  # noqa: SLF001
    await started.wait()  # the task has genuinely started, not merely been scheduled

    engine.stop()

    assert engine._pending_sweep is None  # noqa: SLF001 - cleared, not left dangling
    assert task.cancelling()  # cancel() requested synchronously; let it finish below
    with contextlib.suppress(asyncio.CancelledError):
        await task
