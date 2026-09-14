"""Decision delivery gates remain separate from narration delivery."""

from __future__ import annotations

from pathlib import Path

from narrator.channels.base import ChannelCapabilities, DecisionDeliveryReceipt
from narrator.decisions import DecisionView, PublicOption
from narrator.delivery import present_decisions


class _Adapter:
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    async def present_decision(self, view):
        self.views.append(view)
        return self.result

    def __init__(self, result=True):
        self.result = result
        self.views = []

    async def deliver_decision_views(self, views):
        self.views.extend(views)
        return DecisionDeliveryReceipt(
            status="delivered" if self.result else "unavailable",
            decision_count=len(tuple(views)) if self.result else 0,
        )


def _view() -> DecisionView:
    return DecisionView(
        presentation_token="t" * 32,
        kind="confirmation",
        character_id="rill",
        character_name="Rill",
        question="Open the door?",
        options=(PublicOption(id="confirm", label="Confirm"),),
    )


async def test_decision_delivery_rejects_open_ledger_ruling_and_failed_presentation(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state = campaign / "state.json"
    state.write_text('{"fiction_debt": [{"seq": 1}], "pending_rulings": []}', encoding="utf-8")
    adapter = _Adapter()
    assert (await present_decisions(adapter, (_view(),), tmp_path)).failure_category == "ledger_gate"
    state.write_text('{"fiction_debt": [], "pending_rulings": [{"id": "r"}]}', encoding="utf-8")
    assert (await present_decisions(adapter, (_view(),), tmp_path)).failure_category == "ruling_gate"
    state.write_text('{"fiction_debt": [], "pending_rulings": []}', encoding="utf-8")
    adapter = _Adapter(False)
    assert (await present_decisions(adapter, (_view(),), tmp_path)).failure_category == "audience"


async def test_unknown_channel_capability_version_fails_closed(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text(
        '{"fiction_debt": [], "pending_rulings": []}', encoding="utf-8"
    )
    adapter = _Adapter()
    adapter.decision_capabilities = ChannelCapabilities(
        version=2, structured_decisions=True, atomic_decision_delivery=True
    )

    result = await present_decisions(adapter, (_view(),), tmp_path)

    assert result.failure_category == "unsupported_channel"


class _RecoveryAdapter:
    """Record every posted string and every time recovery was armed."""

    def __init__(self) -> None:
        self.posted: list[str] = []
        self.recovery_calls = 0

    async def post(self, channel_id, text):
        self.posted.append(text)

    async def decision_recovery(self) -> None:
        self.recovery_calls += 1


class _StatusLineAdapter(_RecoveryAdapter):
    """A ``_RecoveryAdapter`` that also renders a status line, for refresh tests."""

    def __init__(self) -> None:
        super().__init__()
        self.statuses: list[dict] = []

    async def update_status(self, snapshot: dict) -> None:
        self.statuses.append(snapshot)


class _Turn:
    channel_id = "terminal"


async def test_a_withheld_turn_arms_the_recovery_controls_its_notice_advertises():
    """``deliver_decision_fault`` and ``deliver_decision_recovery`` armed recovery.
    ``deliver`` -- the path a ``ResolutionGuard`` withhold takes -- did not, even though
    the two notices it selects for that path both end \"Use /retry, /revise, or
    /dismiss.\" The terminal therefore kept ``_decision_recovery_active`` false and
    answered \"Unknown command\" to a control its own game master had just told the
    player to type. Against the code before this slice both assertions on
    ``recovery_calls`` below read 0.
    """
    from narrator.config import NarratorConfig
    from narrator.delivery import DECISION_INCOMPLETE_NOTICE, TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))

    # A guard withhold whose own tool calls changed nothing: the fault notice posts.
    adapter = _RecoveryAdapter()
    outcome = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
    )
    assert (await deliver(adapter, _Turn(), outcome, config)).text == config.decision_fault_notice
    assert adapter.recovery_calls == 1

    # A guard withhold whose setup did reach the record: the incomplete notice posts,
    # and it names the same controls, so it arms them too.
    armed = _RecoveryAdapter()
    changed = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        tool_events=({"tool": "npc_create", "ok": True, "error": "", "event_id": 4},),
    )
    assert (await deliver(armed, _Turn(), changed, config)).text == DECISION_INCOMPLETE_NOTICE
    assert armed.recovery_calls == 1


async def test_a_withheld_turns_real_rolls_reach_the_table_on_its_notice():
    """Test a withheld turns real rolls reach the table on its notice.
    """
    from narrator.config import NarratorConfig
    from narrator.delivery import DECISION_INCOMPLETE_NOTICE, TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))
    lines = (
        "Sorrow rolls INT: rolled 5 vs target 9, success.",
        "Sorrow has taken the measure of Rill: if Rill falls Helpless this session, "
        "the blade kills Rill outright.",
    )

    adapter = _RecoveryAdapter()
    outcome = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        tool_events=({"tool": "grant_runic_weapon", "ok": True, "error": "", "event_id": 4},),
        mechanical_lines=lines,
    )
    posted = (await deliver(adapter, _Turn(), outcome, config)).text
    assert posted == "\n".join(lines) + "\n\n" + DECISION_INCOMPLETE_NOTICE
    assert adapter.posted == [posted]
    assert adapter.recovery_calls == 1  # the notice's controls still arm

    # The plain withheld path carries them the same way.
    unsettled = TurnOutcome(
        narration="model text that must not post", ratified=False, withheld=True,
        mechanical_lines=lines,
    )
    quiet = _RecoveryAdapter()
    posted = (await deliver(quiet, _Turn(), unsettled, config)).text
    assert posted == "\n".join(lines) + "\n\n" + config.withheld_notice
    assert "model text" not in posted

    # No lines: byte-identical to the previous behaviour.
    bare = _RecoveryAdapter()
    outcome = TurnOutcome(narration="", ratified=True, withheld=True, decision_recovery=True)
    assert (await deliver(bare, _Turn(), outcome, config)).text == config.decision_fault_notice


async def test_a_refused_mechanic_names_itself_in_the_recovery_notice():
    """Test a refused mechanic names itself in the recovery notice.
    """
    from narrator.config import NarratorConfig
    from narrator.delivery import TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))
    adapter = _RecoveryAdapter()
    outcome = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        error="the confirmed action rolled no resolving mechanic",
        tool_events=(
            {"tool": "combat_attack", "ok": False, "error": "no_actions_remaining", "event_id": None},
        ),
    )
    posted = (await deliver(adapter, _Turn(), outcome, config)).text
    assert "combat_attack" in posted
    assert "no_actions_remaining" in posted
    assert "/retry" in posted
    assert adapter.recovery_calls == 1


async def test_a_refused_read_only_tool_never_reaches_the_refusal_notice():
    """Only a refused *mechanic* explains a withheld decision turn. A failed lookup
    beside it (here ``campaign_status``, the call the server tells the model to make
    after any error) keeps the general fault notice."""
    from narrator.config import NarratorConfig
    from narrator.delivery import TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))
    adapter = _RecoveryAdapter()
    outcome = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        tool_events=(
            {"tool": "campaign_status", "ok": False, "error": "campaign_locked", "event_id": None},
        ),
    )
    assert (await deliver(adapter, _Turn(), outcome, config)).text == config.decision_fault_notice


async def test_a_state_change_still_outranks_the_refusal_notice():
    """Setup that reached the record is the more important truth: the incomplete
    notice still posts even when a mechanic was also refused."""
    from narrator.config import NarratorConfig
    from narrator.delivery import DECISION_INCOMPLETE_NOTICE, TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))
    adapter = _RecoveryAdapter()
    outcome = TurnOutcome(
        narration="", ratified=True, withheld=True, decision_recovery=True,
        tool_events=(
            {"tool": "npc_create", "ok": True, "error": "", "event_id": 4},
            {"tool": "combat_attack", "ok": False, "error": "turn_not_open", "event_id": None},
        ),
    )
    assert (await deliver(adapter, _Turn(), outcome, config)).text == DECISION_INCOMPLETE_NOTICE


async def test_only_a_notice_that_names_a_control_arms_one_and_narration_never_does():
    """The wording decides, not the call site -- and model prose is never a notice.

    Every notice-posting path in ``narrator.delivery`` routes through one function, so
    a future notice that names a control arms it without a second edit, and a notice
    that names none arms nothing. Narration posts on its own path, so a story that
    happens to contain "/retry" can never arm a control.
    """
    from narrator.config import NarratorConfig
    from narrator.delivery import (
        DECISION_INCOMPLETE_NOTICE,
        TurnOutcome,
        advertises_recovery_controls,
        deliver,
        deliver_decision_decline,
        deliver_no_action,
    )

    config = NarratorConfig(campaign_root=Path("."))
    notices = {
        "decision_fault_notice": config.decision_fault_notice,
        "decision_segment_notice": config.decision_segment_notice,
        "decision_declined_notice": config.decision_declined_notice,
        "risk_confirmation_notice": config.risk_confirmation_notice,
        "romance_boundary_notice": config.romance_boundary_notice,
        "trade_confirmation_notice": config.trade_confirmation_notice,
        "trade_completed_notice": config.trade_completed_notice,
        "withheld_notice": config.withheld_notice,
        "fault_notice": config.fault_notice,
        "decision_incomplete_notice": DECISION_INCOMPLETE_NOTICE,
    }
    named = set()
    for name, notice in notices.items():
        adapter = _RecoveryAdapter()
        await deliver_no_action(adapter, _Turn(), notice)
        expected = 1 if advertises_recovery_controls(notice) else 0
        assert adapter.recovery_calls == expected, name
        if expected:
            named.add(name)

    # The three the players are actually told to use. Naming them keeps this test a
    # claim about the notices rather than a tautology over whatever they happen to say.
    assert named == {
        "decision_fault_notice", "decision_segment_notice", "decision_incomplete_notice",
    }

    # A decline is terminal and names no control, so it arms nothing.
    declined = _RecoveryAdapter()
    await deliver_decision_decline(declined, _Turn(), config)
    assert declined.recovery_calls == 0

    # Narration mentioning a control is still narration.
    prose = _RecoveryAdapter()
    story = "The scribe mutters something about a /retry and turns the page."
    delivered = await deliver(
        prose, _Turn(), TurnOutcome(narration=story, ratified=True, withheld=False), config
    )
    assert delivered.kind == "narration"
    assert delivered.text == story
    assert prose.recovery_calls == 0


async def test_a_narration_naming_a_served_tool_now_delivers_normally():
    """Test a narration naming a served tool now delivers normally.
    """
    from narrator.config import NarratorConfig
    from narrator.delivery import TurnOutcome, deliver

    config = NarratorConfig(campaign_root=Path("."))

    formerly_leaking = _RecoveryAdapter()
    story = (
        'I will force the engine to bind the actor with combat_begin_turn.'
    )
    delivered = await deliver(
        formerly_leaking,
        _Turn(),
        TurnOutcome(narration=story, ratified=True, withheld=False),
        config,
    )
    assert delivered.text == story
    assert formerly_leaking.posted == [story]

    clean = _RecoveryAdapter()
    ordinary_story = 'Rade brings his cleaver down toward you.'
    delivered_clean = await deliver(
        clean, _Turn(), TurnOutcome(narration=ordinary_story, ratified=True, withheld=False), config
    )
    assert delivered_clean.text == ordinary_story
    assert clean.posted == [ordinary_story]


async def test_delivery_refreshes_the_status_line_on_every_posting_path(tmp_path: Path):
    """Every path that puts text on the wire also refreshes a channel's status line.

    An adapter without ``update_status`` (``_RecoveryAdapter``, used throughout this
    file) is untouched by any of this -- ``getattr`` finds nothing callable and every
    other assertion in this module already proves that adapter still works. This test
    is only about the additive ``StatusLineAdapter`` contract.
    """
    from narrator.channels.base import StatusLineAdapter
    from narrator.config import NarratorConfig
    from narrator.delivery import (
        TurnOutcome,
        deliver,
        deliver_decision_decline,
        deliver_decision_fault,
        deliver_decision_recovery,
        deliver_no_action,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text(
        '{"scene": {"title": "The Ashen Bell"}, "fiction_debt": [], "pending_rulings": []}',
        encoding="utf-8",
    )
    config = NarratorConfig(campaign_root=tmp_path)

    narrated = _StatusLineAdapter()
    assert isinstance(narrated, StatusLineAdapter)
    delivered = await deliver(
        narrated, _Turn(), TurnOutcome(narration="The bell tolls.", ratified=True, withheld=False), config
    )
    assert delivered.text == "The bell tolls."
    assert narrated.statuses == [{
        "scene_title": "The Ashen Bell", "location_id": "", "day": None,
        "combat_active": False, "combat_round": None, "combat_active_actor": None,
        "characters": {},
        "players": {},
    }]

    withheld = _StatusLineAdapter()
    await deliver(withheld, _Turn(), TurnOutcome(narration="", ratified=True, withheld=True), config)
    assert len(withheld.statuses) == 1

    no_action = _StatusLineAdapter()
    await deliver_no_action(no_action, _Turn(), config.decision_segment_notice, config.campaign_root)
    assert len(no_action.statuses) == 1

    fault = _StatusLineAdapter()
    await deliver_decision_fault(fault, _Turn(), config)
    assert len(fault.statuses) == 1

    recovery = _StatusLineAdapter()
    await deliver_decision_recovery(recovery, _Turn(), config)
    assert len(recovery.statuses) == 1

    declined = _StatusLineAdapter()
    await deliver_decision_decline(declined, _Turn(), config)
    assert len(declined.statuses) == 1

    # ``deliver_no_action``'s ``campaign_root`` default is only a safety net for a
    # caller that omits it; every production caller (``NarratorService._post_notice``)
    # always passes ``config.campaign_root`` explicitly. This only proves the default
    # does not raise -- ``narrator.status.snapshot`` fails open regardless of what it
    # reads at that path, so its exact content here is not this test's concern.
    defaulted = _StatusLineAdapter()
    await deliver_no_action(defaulted, _Turn(), "a notice")
    assert len(defaulted.statuses) == 1


def test_turn_post_text_is_byte_identical_to_the_retired_composition():
    """``TurnPost.text`` must reproduce exactly what the pre-``TurnPost`` module posted:
    narration verbatim, a bare notice verbatim, and a withheld turn's mechanical
    lines joined above its notice with the same separators ``deliver`` used.
    """
    from narrator.delivery import TurnPost

    story = "The bell tolls twice."
    assert TurnPost(kind="narration", origin="turn", model_text=story).text == story

    notice = "NOTICE:withheld"
    assert TurnPost(kind="notice", origin="turn", notice_text=notice).text == notice

    lines = ("Rill rolls DEX: 9 vs 14, success.", "Doom holds steady.")
    composed = TurnPost(
        kind="notice", origin="turn", engine_lines=lines,
        notice_key="withheld", notice_text=notice,
    )
    assert composed.text == "\n".join(lines) + "\n\n" + notice
