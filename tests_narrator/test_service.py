"""Service sequencing proves decisions stop normal narrator invocation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fake_classifier import ClassifyingEngine, fake_classify_intent, policy_for

from narrator import service as service_module
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionCollectionCancelled,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import (
    ConfirmationDraft,
    DecisionSubmission,
    DeclinedPlan,
    PlanOutcome,
    ProceedPlan,
    RequestDecisionPlan,
)
from narrator.delivery import TurnOutcome
from narrator.engine import PLANNER_FAILURE_REASONS
from narrator.interactions import open_npc_turn, read_trusted_scope
from narrator.policy_types import CombatSnapshot, InteractionCue
from narrator.service import NarratorService
from narrator.social import TradeTerms


class _Adapter:
    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self):
        self.posted = []
        self.presented = []
        self.closed = False
        self.records = []
        self.recorder = self

    def record(self, event, **fields):
        self.records.append((event, fields))

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", "Open it", ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )

    async def post(self, channel_id, text):
        self.posted.append((channel_id, text))

    async def close(self):
        self.closed = True

    async def present_decision(self, view):
        self.presented.append(view)
        return True

    async def deliver_decision_views(self, views):
        self.presented.extend(views)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(tuple(views)))

    async def collect_decision(self, views):
        assert len(views) == 1
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="confirm"),
        )

    async def acknowledge_decision(self, result):
        raise AssertionError(result)


class _Engine(ClassifyingEngine):
    """The base fake engine, classifier included.

    ``NarratorService`` reaches the turn classifier through a duck-typed
    ``classify_intent`` on whatever engine the channel handed it, and an engine without
    one routes nothing at all -- every turn is withheld with
    ``config.classifier_fault_notice``. That is the production contract (there is no
    lexical fallback behind the model), so a fake engine needs a classifier of its own
    or this suite would only ever measure the fault notice. ``ClassifyingEngine`` supplies
    the offline double; see ``tests_narrator/fake_classifier.py`` for what it can and
    cannot establish. Every subclass below that defines ``__init__`` chains through
    ``super().__init__()``, which is what keeps the seam inherited rather than copied.
    """

    def __init__(self):
        super().__init__()
        self.plan_calls = 0
        self.run_calls = 0
        self.resolutions = ()

    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        if self.plan_calls == 1:
            return PlanOutcome(
                plan=RequestDecisionPlan(
                    kind="request",
                    decision=ConfirmationDraft(
                        kind="confirmation", question="Open the sealed door?",
                        audience={"kind": "current"},
                    ),
                )
            )
        assert resolutions[0].answer_kind == "confirm"
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.resolutions = decision_resolutions
        return TurnOutcome("The seal opens.", ratified=True, withheld=False)


class _ActionAdapter(_Adapter):
    def __init__(self, action):
        super().__init__()
        self.action = action

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill",
                self.action,
                ChannelPrincipal("terminal", "terminal-player", "Rill"),
            ),
        )


class _PlannerSentinelEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        raise AssertionError("a read-only turn must not invoke the planner")


class _ProceedPlannerEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions, *, session=None):
        self.plan_calls += 1
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))


#: One open fight, in the grammar ``CampaignStore._combat_section`` writes and
#: ``narrator.interactions.read_combat_snapshot`` parses. The enemy holds the active
#: turn, which is when Black Sword Hack asks the target to parry or dodge.
_FIGHT = """
## Combat

- round: 2
- active actor: reed-thug
- order: rill, reed-thug
- combatant: rill side=pc actions=1/1
- combatant: reed-thug side=npc range=close actions=1/1
"""


def _campaign(tmp_path: Path, players: str | None = None, combat: str = "", present: str = ""):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text("# Low water\n" + present + combat, encoding="utf-8")
    (campaign / "state.json").write_text('{"event_seq": 1, "fiction_debt": [], "pending_rulings": []}', encoding="utf-8")
    (campaign / "players.yaml").write_text(players or """players:
- discord_user_id: terminal-player
  character_id: rill
  display_name: Rill
""", encoding="utf-8")


async def test_pre_action_decision_collects_before_the_normal_agent_runs(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _Adapter()
    engine = _Engine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    report = await NarratorService(config, adapter, engine).run()
    assert engine.plan_calls == 2
    assert engine.run_calls == 1
    assert engine.resolutions[0].answer_kind == "confirm"
    assert report.decision_faults == 0
    assert adapter.posted == [("terminal", "The seal opens.")]
    assert all("Open the sealed door?" not in repr(fields) for _, fields in adapter.records)
    assert {event for event, _ in adapter.records} == {"decision_lifecycle"}


async def test_an_unlinked_principal_skips_planning_and_reaches_the_narrator(tmp_path: Path):
    """Session zero: the campaign holds no player links yet, so a planner-route
    declaration has no character a structured decision could bind to. The turn must
    proceed straight to the narrator -- the conversation IS the game (boundaries,
    origins, ``character_create``) -- rather than plan against an empty directory
    and fault at audience authorization, which is what happened before this check:
    the planner's requested confirmation could never authorize and every committed
    declaration drew a fault notice instead of an answer.
    """
    _campaign(tmp_path, players="players: []\n")
    adapter = _Adapter()
    engine = _Engine()
    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert report.decision_faults == 0
    assert adapter.posted == [("terminal", "The seal opens.")]


async def test_a_socially_phrased_session_zero_line_reaches_the_narrator(tmp_path: Path):
    """A live TUI ceremony recorded "I want to play a decadent character. What are
    my background options?" drawing the decision-fault notice: the line routed
    social with a required test, no test could derive for a player who owns no
    character, and the run loop's fail-closed gate ate the turn. An unlinked
    principal's social phrasing is conversation, not an influence roll, so it must
    proceed; the same gate still withholds for a linked player whose test cannot
    derive."""

    class _SocialLineAdapter(_Adapter):
        async def turns(self):
            yield InboundTurn(
                "terminal",
                ChannelMessage(
                    "Player",
                    "I want to play a decadent character. What are my background options?",
                    ChannelPrincipal("terminal", "terminal-player", "Player"),
                ),
            )

    _campaign(tmp_path, players="players: []\n")
    adapter = _SocialLineAdapter()
    engine = _ProceedPlannerEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    report = await NarratorService(config, adapter, engine).run()

    assert engine.run_calls == 1
    assert report.decision_faults == 0
    assert adapter.posted == [("terminal", "The seal opens.")]


async def test_an_unreadable_directory_still_fails_closed_for_planner_turns(tmp_path: Path):
    """The session-zero skip must never widen into fail-open: an *unreadable*
    directory is a fault, exactly as the planner path has always treated it, not an
    invitation to narrate without authorization.
    """
    _campaign(tmp_path, players="players: {broken\n")
    adapter = _Adapter()
    engine = _Engine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    report = await NarratorService(config, adapter, engine).run()

    assert engine.run_calls == 0
    assert report.decision_faults == 1
    assert adapter.posted == [("terminal", config.decision_fault_notice)]
    assert [
        fields.get("failure_category") for event, fields in adapter.records
        if fields.get("category") == "fault"
    ] == ["directory"]


async def test_fault_notice_for_selects_the_matching_bucket(tmp_path: Path):
    """Test fault notice for selects the matching bucket.
    """
    config = NarratorConfig(campaign_root=tmp_path)
    service = NarratorService(config, _Adapter(), _Engine())

    assert service._fault_notice_for("") == ("decision_fault", config.decision_fault_notice)
    for category in (*PLANNER_FAILURE_REASONS, "directory"):
        assert service._fault_notice_for(category) == (
            "decision_fault", config.decision_fault_notice
        )
    assert service._fault_notice_for("context") == (
        "decision_stale_context", config.decision_stale_context_notice
    )
    assert service._fault_notice_for("confirmation") == (
        "decision_stale_confirmation", config.decision_stale_confirmation_notice
    )

    service._record_decision(category="fault", failure="context")
    assert service._last_fault_category == "context"
    service._record_decision(category="fault", failure="confirmation")
    assert service._last_fault_category == "confirmation"
    # A non-fault category must not overwrite the stash with an unrelated failure name
    # -- ``budget_boundary`` passes ``failure="view_limit"``/``"view_capacity"`` under
    # ``category="segment"``, which must never leak into the next genuine fault's notice.
    service._record_decision(category="segment", failure="view_limit")
    assert service._last_fault_category == "confirmation"


async def test_a_stale_directory_context_gets_its_own_notice(tmp_path: Path):
    """The table's own state changing between a decision's presentation and its answer
    used to render the same generic "could not prepare a safe decision" sentence as
    every other fault. It now says plainly that the state moved and a retry needs no
    different input, not that anything was malformed.
    """

    class _CorruptingAdapter(_Adapter):
        def __init__(self, state_path: Path):
            super().__init__()
            self._state_path = state_path

        async def collect_decision(self, views):
            principal, submission = await super().collect_decision(views)
            # Simulate the table's own state changing while this decision was open.
            self._state_path.write_text("not valid json", encoding="utf-8")
            return principal, submission

    _campaign(tmp_path)
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    adapter = _CorruptingAdapter(tmp_path / "campaign" / "state.json")
    report = await NarratorService(config, adapter, _Engine()).run()

    assert report.decision_faults == 1
    assert adapter.posted[-1] == ("terminal", config.decision_stale_context_notice)
    assert "could not prepare a safe decision" not in adapter.posted[-1][1]


async def test_read_only_turns_bypass_planner_at_the_service_boundary(tmp_path: Path):
    for index, action in enumerate(
        (
            "What are the traders selling?",
            "Look at the traders",
            "I inspect the room",
            "I listen at the door",
            "I read the trader notice",
            "How do I use this door?",
        )
    ):
        root = tmp_path / str(index)
        root.mkdir()
        _campaign(root)
        engine = _PlannerSentinelEngine()

        report = await NarratorService(
            NarratorConfig(campaign_root=root, max_decision_rounds=2),
            _ActionAdapter(action),
            engine,
        ).run()

        assert report.decision_faults == 0
        assert engine.plan_calls == 0
        assert engine.run_calls == 1


async def test_compound_committed_turn_remains_planner_eligible(tmp_path: Path):
    _campaign(tmp_path)
    engine = _ProceedPlannerEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("I inspect the room and open the door"),
        engine,
    ).run()

    assert engine.plan_calls == 1
    assert engine.run_calls == 1
    assert engine.resolutions == ()


async def test_non_hostile_violence_confirms_before_any_planner_invocation(tmp_path: Path):
    for index, action in enumerate(("I stab the barkeep", "I assault the librarian")):
        root = tmp_path / str(index)
        root.mkdir()
        _campaign(root)
        adapter = _ActionAdapter(action)
        engine = _PlannerSentinelEngine()

        report = await NarratorService(
            NarratorConfig(campaign_root=root, max_decision_rounds=2), adapter, engine
        ).run()

        assert report.decision_faults == 0
        assert engine.plan_calls == 0
        assert engine.run_calls == 1
        assert len(adapter.presented) == 1


async def test_engine_risk_floor_confirms_once_then_proceeds_without_repeating(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _ActionAdapter("Hurl a trader into the canal")
    engine = _ProceedPlannerEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert len(adapter.presented) == 1


class _PartyAdapter(_Adapter):
    name = "discord"

    def __init__(self):
        super().__init__()
        self.collections = []

    async def turns(self):
        yield InboundTurn(
            "discord",
            ChannelMessage("Rill", "Cross the canal", ChannelPrincipal("discord", "discord-rill", "Rill")),
        )

    async def collect_decision(self, views):
        self.collections.append(views)
        by_character = {view.character_id: view for view in views}
        attempt = len(self.collections)
        if attempt <= 2:
            character_id = "rill"
        else:
            character_id = "ossa"
        view = by_character[character_id]
        return (
            ChannelPrincipal("discord", f"discord-{character_id}", character_id.title()),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="bridge"),
        )


class _PartyEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        if self.plan_calls == 1:
            return PlanOutcome(
                plan=RequestDecisionPlan(
                    kind="request",
                    decision={
                        "kind": "clarification",
                        "question": "How do you cross the canal?",
                        "audience": {"kind": "party"},
                        "paths": ({"id": "bridge", "label": "Use the bridge"},),
                    },
                )
            )
        assert {item.character_id for item in resolutions} == {"rill", "ossa"}
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))


async def test_request_collection_waits_for_all_targets_and_ignores_a_partial_retry(tmp_path: Path):
    _campaign(
        tmp_path,
        """players:
- discord_user_id: discord-rill
  character_id: rill
  display_name: Rill
- discord_user_id: discord-ossa
  character_id: ossa
  display_name: Ossa
""",
    )
    adapter = _PartyAdapter()
    engine = _PartyEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert report.decision_faults == 0
    assert engine.run_calls == 1
    assert [len(views) for views in adapter.collections] == [2, 2, 2]
    rill_tokens = [views[0].presentation_token for views in adapter.collections]
    assert rill_tokens == [rill_tokens[0]] * 3
    assert {resolution.character_id for resolution in engine.resolutions} == {"rill", "ossa"}


class _BlockingDecisionAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.collection_started = asyncio.Event()
        self.collection_cancelled = asyncio.Event()
        self.view = None

    async def collect_decision(self, views):
        self.view = views[0]
        self.collection_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.collection_cancelled.set()
            raise


async def test_shutdown_cancels_a_live_collection_and_tombstones_its_request(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _BlockingDecisionAdapter()
    service = NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, _Engine()
    )

    running = asyncio.create_task(service.run())
    await asyncio.wait_for(adapter.collection_started.wait(), timeout=1)
    service.request_stop()
    report = await asyncio.wait_for(running, timeout=1)

    assert report.turns == 1
    assert adapter.collection_cancelled.is_set()
    assert await service._decisions.pending_count() == 0  # noqa: SLF001 - lifecycle seam
    result = await service._decisions.submit(  # noqa: SLF001 - tombstone seam
        principal=ChannelPrincipal("terminal", "terminal-player", "Rill"),
        submission=DecisionSubmission(
            presentation_token=adapter.view.presentation_token,
            selection_id="confirm",
        ),
        context_fingerprint="irrelevant after cancellation",
    )
    assert result.status == "cancelled"


class _DismissingDecisionAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.view = None
        self.turn_source_resumed = False

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", "Open it", ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )
        self.turn_source_resumed = True

    async def collect_decision(self, views):
        self.view = views[0]
        return DecisionCollectionCancelled(reason="dismissed")


async def test_explicit_terminal_dismissal_tombstones_and_resumes_the_turn_source(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _DismissingDecisionAdapter()
    engine = _Engine()
    service = NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    )

    report = await service.run()

    assert report.turns == 1
    assert report.decision_pending == 1
    assert engine.run_calls == 0
    assert adapter.turn_source_resumed is True
    assert await service._decisions.pending_count() == 0  # noqa: SLF001 - lifecycle seam
    result = await service._decisions.submit(  # noqa: SLF001 - tombstone seam
        principal=ChannelPrincipal("terminal", "terminal-player", "Rill"),
        submission=DecisionSubmission(
            presentation_token=adapter.view.presentation_token,
            selection_id="confirm",
        ),
        context_fingerprint="irrelevant after dismissal",
    )
    assert result.status == "cancelled"


async def test_final_allowed_view_runs_the_post_answer_planner_and_can_proceed(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _Adapter()
    engine = _Engine()

    service = NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1), adapter, engine
    )
    report = await service.run()

    assert report.decision_faults == 0
    assert engine.plan_calls == 2
    assert engine.run_calls == 1


class _LimitEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        return PlanOutcome(
            plan=RequestDecisionPlan(
                kind="request",
                decision=ConfirmationDraft(
                    kind="confirmation", question="Confirm?", audience={"kind": "current"}
                ),
            )
        )


async def test_another_request_after_the_view_limit_posts_no_action_recovery(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _Adapter()
    engine = _LimitEngine()

    service = NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1), adapter, engine
    )
    report = await service.run()

    assert report.decision_pending == 1
    assert engine.plan_calls == 2
    assert engine.run_calls == 0
    assert "No action occurred" in adapter.posted[0][1]


class _OversizedPartyAdapter(_PartyAdapter):
    async def collect_decision(self, views):
        raise AssertionError("an oversized audience must not be presented or collected")


class _OversizedPartyEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        return PlanOutcome(
            plan=RequestDecisionPlan(
                kind="request",
                decision={
                    "kind": "clarification",
                    "question": "Which route does the party take?",
                    "audience": {"kind": "party"},
                    "paths": ({"id": "bridge", "label": "Use the bridge"},),
                },
            )
        )


async def test_multi_view_request_exceeding_remaining_capacity_is_cancelled_before_delivery(
    tmp_path: Path,
):
    _campaign(
        tmp_path,
        """players:
- discord_user_id: discord-rill
  character_id: rill
  display_name: Rill
- discord_user_id: discord-ossa
  character_id: ossa
  display_name: Ossa
""",
    )
    adapter = _OversizedPartyAdapter()
    engine = _OversizedPartyEngine()

    service = NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1), adapter, engine
    )
    report = await service.run()

    assert report.decision_pending == 1
    assert adapter.presented == []
    assert adapter.collections == []
    assert await service._decisions.pending_count() == 0  # noqa: SLF001 - cancellation seam


class _DecliningRiskAdapter(_Adapter):
    def __init__(self, text: str = "Punch a trader"):
        super().__init__()
        self._text = text

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", self._text, ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )

    async def collect_decision(self, views):
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="decline"),
        )


class _DecliningRiskEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions, *, session=None):
        self.plan_calls += 1
        if session is not None and session.action_is_declined:
            return DeclinedPlan(kind="declined")
        return PlanOutcome(
            plan=RequestDecisionPlan(
                kind="request",
                decision=ConfirmationDraft(
                    kind="confirmation", question="Confirm?", audience={"kind": "current"}
                ),
            )
        )


async def test_declined_risk_confirmation_posts_no_action_notice_without_narration(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _DecliningRiskAdapter()
    engine = _DecliningRiskEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert engine.run_calls == 0
    assert adapter.posted == [("terminal", config.decision_declined_notice)]


async def test_a_pre_engine_notice_is_logged_with_service_origin_and_its_key(tmp_path: Path):
    """Test a pre engine notice is logged with service origin and its key.
    """
    _campaign(tmp_path)
    adapter = _DecliningRiskAdapter()
    engine = _DecliningRiskEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    service = NarratorService(config, adapter, engine)
    await service.run()

    assert [post.text for post in service.post_log] == [text for _, text in adapter.posted]
    assert [
        (post.kind, post.origin, post.notice_key) for post in service.post_log
    ] == [("notice", "service", "decision_declined")]


class _BareAttackDecliningAdapter(_Adapter):
    """Declares a bare "attack" and declines the confirmation it expects, exactly
    ``_DecliningRiskAdapter`` above -- with "attack" in place of "Punch a trader" so a
    sanctioning fight (a sole recorded NPC combatant) makes ``combat_sanctions_violence``
    agree with it, which the bystander-named "trader" above never would.
    """

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", "attack", ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )

    async def collect_decision(self, views):
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="decline"),
        )


async def test_aud_4_combat_state_is_re_read_fresh_each_decision_round_not_cached(
    tmp_path: Path, monkeypatch
):
    """AUD-4 repair: each decision round inside ``_decision_phase``'s loop must read
    combat state fresh, not reuse a snapshot cached before the round's ``await`` for
    real player input -- the window an independent audit found a cached snapshot could
    outlive, since combat state can genuinely change on another channel or through
    another tool call while that await is in flight.

    This monkeypatches ``read_combat_snapshot`` (imported into ``narrator.service``) to
    return no fight for its first two calls -- the early sanction check ahead of the
    decision loop, then the first loop iteration's ``verify_plan`` call, both of which
    must see no fight yet, or the confirmation would never be presented at all -- and a
    fight sanctioning the declared "attack" (its sole recorded NPC combatant) on every
    call after, simulating the fight opening while the first round's collection await is
    in flight. The player declines that round's confirmation regardless. A cached
    snapshot (the first repair round's rejected behavior) would still see no fight on
    the second ``verify_plan`` call and honor the decline, matching
    ``test_declined_risk_confirmation_posts_no_action_notice_without_narration`` above;
    a fresh read picks up the now-sanctioning fight and proceeds instead, matching this
    test.
    """
    _campaign(tmp_path)
    adapter = _BareAttackDecliningAdapter()
    engine = _DecliningRiskEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    sanctioning = CombatSnapshot(
        active=True,
        round=1,
        active_actor="reed-thug",
        order=("rill", "reed-thug"),
        sides=(("rill", "pc"), ("reed-thug", "npc")),
    )
    calls = {"count": 0}

    def fake_read_combat_snapshot(campaign_root):
        del campaign_root
        calls["count"] += 1
        return CombatSnapshot() if calls["count"] <= 2 else sanctioning

    monkeypatch.setattr(service_module, "read_combat_snapshot", fake_read_combat_snapshot)

    await NarratorService(config, adapter, engine).run()

    # At least the early check, the first (unsanctioned) loop iteration, and a second
    # (now-sanctioned) loop iteration after the decline.
    assert calls["count"] >= 3
    # The fight sanctions the attack the instant it is read fresh, so the turn
    # proceeds to narration -- the decline is moot -- rather than posting the
    # declined-action notice a stale snapshot would still have produced.
    assert engine.run_calls == 1
    assert adapter.posted == [("terminal", "The seal opens.")]


def _enrolling_fight() -> CombatSnapshot:
    return CombatSnapshot(
        active=True,
        round=1,
        active_actor="rill",
        order=("rill", "reed-thug"),
        sides=(("rill", "pc"), ("reed-thug", "npc")),
    )


async def test_a_committed_declaration_during_the_players_fight_skips_the_planner(
    tmp_path: Path, monkeypatch
):
    """Test a committed declaration during the players fight skips the planner.
    """
    _campaign(tmp_path)
    monkeypatch.setattr(
        service_module, "read_combat_snapshot", lambda root: _enrolling_fight()
    )
    adapter = _Adapter()
    engine = _Engine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert adapter.posted == [("terminal", "The seal opens.")]


async def test_a_fight_not_enrolling_the_player_leaves_the_planner_in_charge(
    tmp_path: Path, monkeypatch
):
    """Someone else's fight is not this player's pace problem: the planner runs."""
    _campaign(tmp_path)
    other_fight = CombatSnapshot(
        active=True,
        round=1,
        active_actor="ossa",
        order=("ossa", "reed-thug"),
        sides=(("ossa", "pc"), ("reed-thug", "npc")),
    )
    monkeypatch.setattr(
        service_module, "read_combat_snapshot", lambda root: other_fight
    )
    adapter = _Adapter()
    engine = _Engine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert engine.plan_calls == 2
    assert engine.run_calls == 1


async def test_a_bystander_hazard_mid_fight_still_confirms(tmp_path: Path, monkeypatch):
    """The bypass covers the planner route only. A violence declaration naming a
    recorded person outside the fight's roster routes risk, where the sanction
    rightly withholds, so the confirmation is still presented -- and this player's
    decline still terminates the action."""
    _campaign(tmp_path, present="\n## Present NPCs\n\n- orso-pell\n")
    monkeypatch.setattr(
        service_module, "read_combat_snapshot", lambda root: _enrolling_fight()
    )
    adapter = _DecliningRiskAdapter("Punch orso pell")
    engine = _DecliningRiskEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert engine.run_calls == 0
    assert adapter.posted == [("terminal", config.decision_declined_notice)]


class _ContinuationAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self._control = ""

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", "Open it", ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", "/continue", ChannelPrincipal("terminal", "terminal-player", "Rill")),
        )

    async def decision_recovery(self):
        self._control = "continue"

    def consume_decision_recovery_control(self):
        control = self._control
        self._control = ""
        return control


class _ContinuationEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.executed_text = ""

    async def plan_turn(self, turn, eligible, resolutions):
        self.plan_calls += 1
        if self.plan_calls < 3:
            return PlanOutcome(
                plan=RequestDecisionPlan(
                    kind="request",
                    decision=ConfirmationDraft(
                        kind="confirmation", question="Confirm?", audience={"kind": "current"}
                    ),
                )
            )
        assert turn.mention.text == "Open it"
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_text = turn.mention.text
        return TurnOutcome("The seal opens.", ratified=True, withheld=False)


async def test_continue_replans_retained_action_without_narrating_the_control_text(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _ContinuationAdapter()
    engine = _ContinuationEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1), adapter, engine
    ).run()

    assert report.decision_pending == 1
    assert engine.run_calls == 1
    assert engine.executed_text == "Open it"


class _FailingPlannerEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions):
        from narrator.engine import DecisionPlanningError

        raise DecisionPlanningError("planner action and identity must stay private")


class _FaultRecoveryAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.recovery_calls = 0

    async def decision_recovery(self):
        self.recovery_calls += 1


async def test_planner_failure_posts_only_engine_authored_recovery(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _FaultRecoveryAdapter()
    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1),
        adapter,
        _FailingPlannerEngine(),
    ).run()

    assert report.decision_faults == 1
    assert "/retry" in adapter.posted[0][1]
    assert "identity" not in adapter.posted[0][1]
    assert adapter.recovery_calls == 1


class _TimingOutPlannerEngine(_Engine):
    """A planner failure that names which failure it was."""

    async def plan_turn(self, turn, eligible, resolutions, **kwargs):
        from narrator.engine import DecisionPlanningError

        raise DecisionPlanningError("planner could not produce a safe decision", "planner_timeout")


async def test_the_planner_s_own_failure_reason_reaches_the_diagnostic_transcript(
    tmp_path: Path,
):
    """Test the planner s own failure reason reaches the diagnostic transcript.
    """
    _campaign(tmp_path)
    adapter = _FaultRecoveryAdapter()
    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1),
        adapter,
        _TimingOutPlannerEngine(),
    ).run()

    assert report.decision_faults == 1
    assert [
        fields.get("failure_category") for event, fields in adapter.records
        if fields.get("category") == "fault"
    ] == ["planner_timeout"]


class _RetryFlagEngine(_Engine):
    """Record the ``retry`` keyword every planner call carried."""

    def __init__(self):
        super().__init__()
        self.retry_flags: list[bool] = []

    async def plan_turn(
        self, turn, eligible, resolutions, *, session=None, interaction_cue=None,
        policy=None, retry=False,
    ):
        self.plan_calls += 1
        self.retry_flags.append(retry)
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        if self.run_calls == 1:
            return TurnOutcome(
                "", ratified=True, withheld=True, decision_recovery=True,
                error="the confirmed action rolled no resolving mechanic",
            )
        return TurnOutcome("The door gives.", ratified=True, withheld=False)


async def test_only_a_retry_turn_arms_the_engine_s_bounded_planner_fallback(tmp_path: Path):
    """The wiring half of the bounded fallback. ``NarratorEngine.plan_turn`` gives up its
    hard fault for its own floor only when the player has already read a no-action notice
    for this declaration and typed ``/retry``, so the service has to say which turn that
    is. A first, freshly typed declaration must carry ``retry=False`` or ordinary play
    silently downgrades to the floor on every planner outage.
    """
    _campaign(tmp_path)
    adapter = _EngineWithholdAdapter()
    engine = _RetryFlagEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.retry_flags == [False, True]


class _ToolEventEngine(_Engine):
    """Return one delivered turn carrying redacted tool events."""

    def __init__(self, tool_events):
        super().__init__()
        self._tool_events = tuple(tool_events)

    async def plan_turn(self, turn, eligible, resolutions, *, session=None):
        self.plan_calls += 1
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        return TurnOutcome(
            "You strike.", ratified=True, withheld=False, tool_events=self._tool_events
        )


async def test_the_service_records_each_tool_event_to_the_diagnostic_recorder(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _ActionAdapter("stab the thug")
    engine = _ToolEventEngine(
        (
            {"tool": "combat_attack", "ok": True, "error": "", "event_id": "evt-000007"},
            {"tool": "combat_attack", "ok": False, "error": "target_out_of_reach", "event_id": None},
        )
    )
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    tool_records = [fields for event, fields in adapter.records if event == "tool_call"]
    assert tool_records == [
        {"tool": "combat_attack", "ok": True, "error": "", "event_id": "evt-000007"},
        {"tool": "combat_attack", "ok": False, "error": "target_out_of_reach", "event_id": None},
    ]


async def test_the_service_records_no_tool_call_when_the_turn_ran_no_tool(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _ActionAdapter("just looking around")
    engine = _ToolEventEngine(())
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert not [event for event, _ in adapter.records if event == "tool_call"]


async def test_a_recorded_tool_event_never_carries_arguments_or_result_content(tmp_path: Path):
    """Test a recorded tool event never carries arguments or result content.
    """
    _campaign(tmp_path)
    adapter = _ActionAdapter("stab the thug")
    # An event dict that smuggles argument and content keys must not reach the recorder.
    engine = _ToolEventEngine(
        (
            {
                "tool": "combat_attack",
                "ok": True,
                "error": "",
                "event_id": "evt-000009",
                "arguments": {"target_id": "orso-pell"},
                "content": "You drive the knife home.",
            },
        )
    )
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    tool_records = [fields for event, fields in adapter.records if event == "tool_call"]
    assert tool_records == [
        {"tool": "combat_attack", "ok": True, "error": "", "event_id": "evt-000009"}
    ]
    assert "orso-pell" not in repr(adapter.records)
    assert "knife" not in repr(adapter.records)


async def test_plan_turn_requests_the_public_scope_digest(tmp_path: Path, monkeypatch):
    """The planner must read the public digest; its output becomes player-visible text."""
    from narrator import canon
    from narrator.canon import CanonDigest
    from narrator.engine import NarratorEngine

    _campaign(tmp_path)
    captured: dict[str, object] = {}

    def fake_render(*args, public: bool = False, **kwargs):
        captured["public"] = public
        return CanonDigest(text="canon", stats={})

    monkeypatch.setattr(canon, "render_digest", fake_render)
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2))

    async def fake_plan_once(prompt, *, repair=False):
        return {"plan": {"kind": "proceed"}}

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)
    # An unstarted engine's own ``classify_intent`` answers ``None``, and every hazard
    # decision reads that as "confirm", which would stop the turn above the planner and
    # never reach the digest this test is about. The offline double stands in for the
    # model call so the planner branch runs.
    monkeypatch.setattr(engine, "classify_intent", fake_classify_intent)
    turn = InboundTurn(
        "terminal",
        ChannelMessage(
            "Rill", "I open the sealed door", ChannelPrincipal("terminal", "terminal-player", "Rill")
        ),
    )

    result = await engine.plan_turn(turn, (), ())

    assert isinstance(result.plan, ProceedPlan)
    assert captured["public"] is True


class _DefenceSentinelEngine(_Engine):
    async def plan_turn(self, turn, eligible, resolutions, *, session=None):
        self.plan_calls += 1
        raise AssertionError("a bound defence must not invoke the planner")


async def test_a_lone_dodge_in_an_open_fight_binds_the_defence_without_a_planner(tmp_path: Path):
    """Test a lone dodge in an open fight binds the defence without a planner.
    """
    _campaign(tmp_path, combat=_FIGHT)
    engine = _DefenceSentinelEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("dodge"),
        engine,
    ).run()

    assert report.decision_faults == 0
    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert len(engine.resolutions) == 1
    directive = engine.resolutions[0].directive
    assert directive.kind == "combat_defend"
    assert directive.character_id == "rill"  # the authenticated principal's character
    assert directive.method == "dodge"


async def test_a_defence_word_never_carries_a_violent_action_past_the_risk_floor(tmp_path: Path):
    """An open fight must not turn a defence word into a bypass of the violence gate.

    The compound's violent half names a recorded person outside the fight's roster
    (orso-pell is present and alive but never rostered), which is what withholds the
    sanction and keeps the confirmation."""
    for index, action in enumerate(
        ("I dodge his swing and stab orso pell", "parry then kill orso pell")
    ):
        root = tmp_path / str(index)
        root.mkdir()
        _campaign(root, combat=_FIGHT, present="\n## Present NPCs\n\n- orso-pell\n")
        adapter = _ActionAdapter(action)
        engine = _PlannerSentinelEngine()

        report = await NarratorService(
            NarratorConfig(campaign_root=root, max_decision_rounds=2), adapter, engine
        ).run()

        assert report.decision_faults == 0
        assert engine.plan_calls == 0
        assert len(adapter.presented) == 1  # the risk floor confirmed the violence
        assert all(
            getattr(item.directive, "kind", "") != "combat_defend"
            for item in engine.resolutions
        )


async def test_a_defence_word_outside_an_open_fight_still_reaches_the_planner(tmp_path: Path):
    """Nothing changes for a campaign with no fight running."""
    _campaign(tmp_path)  # the scene render carries no Combat section
    engine = _ProceedPlannerEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("dodge"),
        engine,
    ).run()

    assert engine.plan_calls == 1
    assert engine.resolutions == ()


async def test_a_defence_binds_only_a_character_the_fight_records_on_the_player_side(
    tmp_path: Path,
):
    """An onlooker's dodge binds nothing, because the fight does not record them."""
    _campaign(
        tmp_path,
        players="""players:
- discord_user_id: terminal-player
  character_id: sella
  display_name: Sella
""",
        combat=_FIGHT,
    )
    engine = _ProceedPlannerEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("dodge"),
        engine,
    ).run()

    assert engine.plan_calls == 1  # sella is absent from the combatant list
    assert engine.resolutions == ()


def _write_open_npc_turn_state(tmp_path: Path, *, active_id: str, npc_name: str, pc_id: str) -> None:
    """The ``campaign/state.json`` shape ``narrator.interactions.open_npc_turn`` reads:
    an active fight whose open actor sits on the NPC side with an action unspent and
    the campaign still recording it alive. Written directly, the same way
    ``NarratorEngine._combat_recovery_view`` and this function's own production
    counterpart both bypass the scene-render's Combat section -- that section
    (``_FIGHT`` above) carries no ``turn_open`` flag at all.
    """
    (tmp_path / "campaign" / "state.json").write_text(
        json.dumps(
            {
                "event_seq": 1,
                "fiction_debt": [],
                "pending_rulings": [],
                "combat": {
                    "active": True,
                    "active_actor": active_id,
                    "actors": {
                        active_id: {"side": "npc", "turn_open": True, "actions_used": 0, "actions_max": 1},
                        pc_id: {"side": "pc", "turn_open": False, "actions_used": 0, "actions_max": 1},
                    },
                },
                "npcs": {active_id: {"name": npc_name, "status": "alive"}},
            }
        ),
        encoding="utf-8",
    )


def test_open_npc_turn_reads_the_live_open_actor(tmp_path: Path):
    (tmp_path / "campaign").mkdir()
    _write_open_npc_turn_state(tmp_path, active_id="reed-thug", npc_name="Reed Thug", pc_id="rill")
    result = open_npc_turn(tmp_path)
    assert result.actor_id == "reed-thug"
    assert result.name == "Reed Thug"
    assert result.actions_remaining == 1


def test_open_npc_turn_is_none_without_any_of_its_required_conditions(tmp_path: Path):
    """Every shape short of a live, alive, unspent NPC turn returns ``None`` -- the
    same fail-closed discipline ``ledger.py`` documents for the fiction-debt ledger."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    def _state(**combat_overrides):
        combat = {
            "active": True, "active_actor": "reed-thug",
            "actors": {"reed-thug": {"side": "npc", "turn_open": True, "actions_used": 0, "actions_max": 1}},
        }
        combat.update(combat_overrides)
        (campaign / "state.json").write_text(
            json.dumps({"combat": combat, "npcs": {"reed-thug": {"name": "Reed Thug", "status": "alive"}}}),
            encoding="utf-8",
        )

    # No file at all.
    assert open_npc_turn(tmp_path) is None
    # No fight running.
    (campaign / "state.json").write_text(json.dumps({"combat": {"active": False}}), encoding="utf-8")
    assert open_npc_turn(tmp_path) is None
    # The open actor is the player's own side, not an NPC's.
    _state(actors={"reed-thug": {"side": "pc", "turn_open": True, "actions_used": 0, "actions_max": 1}})
    assert open_npc_turn(tmp_path) is None
    # The NPC's turn is not open.
    _state(actors={"reed-thug": {"side": "npc", "turn_open": False, "actions_used": 0, "actions_max": 1}})
    assert open_npc_turn(tmp_path) is None
    # Every action already spent.
    _state(actors={"reed-thug": {"side": "npc", "turn_open": True, "actions_used": 1, "actions_max": 1}})
    assert open_npc_turn(tmp_path) is None
    # The campaign already records this NPC dead.
    _state()
    (campaign / "state.json").write_text(
        json.dumps({
            "combat": {
                "active": True, "active_actor": "reed-thug",
                "actors": {"reed-thug": {"side": "npc", "turn_open": True, "actions_used": 0, "actions_max": 1}},
            },
            "npcs": {"reed-thug": {"name": "Reed Thug", "status": "dead"}},
        }),
        encoding="utf-8",
    )
    assert open_npc_turn(tmp_path) is None


class _DodgeChoiceAdapter(_Adapter):
    """Declares an unrelated action, then answers a presented decision with "dodge"."""

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "I check my pack",
                ChannelPrincipal("terminal", "terminal-player", "Rill"),
            ),
        )

    async def collect_decision(self, views):
        view = views[0]
        dodge = next(option for option in view.options if option.id == "dodge")
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id=dodge.id),
        )


async def test_an_unrelated_declaration_during_an_enemy_turn_presents_the_defence_choice(
    tmp_path: Path,
):
    """Test an unrelated declaration during an enemy turn presents the defence choice.
    """
    _campaign(tmp_path, combat=_FIGHT)
    _write_open_npc_turn_state(tmp_path, active_id="reed-thug", npc_name="Reed Thug", pc_id="rill")
    engine = _Engine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _DodgeChoiceAdapter(),
        engine,
    ).run()

    assert engine.plan_calls == 0  # never reaches the model-driven planner loop
    assert engine.run_calls == 1
    assert len(engine.resolutions) == 1
    resolution = engine.resolutions[0]
    assert resolution.character_id == "rill"
    assert resolution.directive.kind == "combat_defend"
    assert resolution.directive.method == "dodge"
    assert report.decision_faults == 0


async def test_a_typed_defence_word_skips_the_presented_choice_entirely(tmp_path: Path):
    """A one-word "dodge"/"parry" answer already binds through
    ``_combat_defence_resolution``'s faster, planner-free path -- this decision must
    never also fire for it, or the table would see the choice presented twice."""
    _campaign(tmp_path, combat=_FIGHT)
    _write_open_npc_turn_state(tmp_path, active_id="reed-thug", npc_name="Reed Thug", pc_id="rill")
    engine = _Engine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("dodge"),
        engine,
    ).run()

    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert len(engine.resolutions) == 1
    assert engine.resolutions[0].directive.method == "dodge"


async def test_the_players_own_open_turn_never_presents_the_defence_choice(tmp_path: Path):
    """``open_npc_turn`` names an NPC's own turn specifically; the player's own open
    turn must never route through this at all -- it falls through to
    ``_combat_turn_pace_bypass``'s own, unrelated handling of an ordinary
    declaration made during the player's own combat turn, exactly as it did before
    this check existed."""
    _campaign(tmp_path, combat=_FIGHT)
    # rill's own turn is open here, not reed-thug's -- ``open_npc_turn`` refuses any
    # open actor whose side is not "npc", so this must fall through untouched.
    (tmp_path / "campaign" / "state.json").write_text(
        json.dumps(
            {
                "event_seq": 1, "fiction_debt": [], "pending_rulings": [],
                "combat": {
                    "active": True, "active_actor": "rill",
                    "actors": {
                        "rill": {"side": "pc", "turn_open": True, "actions_used": 0, "actions_max": 1},
                        "reed-thug": {"side": "npc", "turn_open": False, "actions_used": 0, "actions_max": 1},
                    },
                },
                "npcs": {"reed-thug": {"name": "Reed Thug", "status": "alive"}},
            }
        ),
        encoding="utf-8",
    )
    engine = _ProceedPlannerEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("I check my pack"),
        engine,
    ).run()

    # rill's own open combat turn already bypasses the planner (a pre-existing,
    # unrelated mechanism); the claim this test pins is that no *decision* over a
    # defence choice was presented or bound instead.
    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert engine.resolutions == ()


class _WithheldNoteEngine(_Engine):
    """Records the ``withheld_note`` every executed turn carried, never its text elsewhere."""

    def __init__(self):
        super().__init__()
        self.seen_notes: list[str | None] = []

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.seen_notes.append(getattr(turn, "withheld_note", "missing"))
        return TurnOutcome("Understood.", ratified=True, withheld=False)


class _RiskThenAskAdapter(_Adapter):
    """A hazard declaration the risk floor withholds, then an out-of-character question."""

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "I attack the shadow", ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "GM, why was I refused?", ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )


async def test_a_withheld_risk_turn_carries_its_true_reason_into_the_next_prompt(tmp_path: Path):
    """The first turn never reaches the narrator: ``max_decision_rounds=0`` reproduces the
    adapter/replay arm ``NarratorConfig.max_decision_rounds`` documents, so
    ``_decision_phase`` returns ``risk_confirmation_required`` before any planner call.
    The second turn is plain out-of-character routing and always reaches the narrator.
    The engine-authored notice posted for the first turn must be the exact text the
    second turn's execution wrapper carries as ``withheld_note``, so
    ``narrator.prompt.turn_prompt`` can render the true reason rather than nothing.
    """
    _campaign(tmp_path)
    adapter = _RiskThenAskAdapter()
    engine = _WithheldNoteEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=0)

    report = await NarratorService(config, adapter, engine).run()

    assert report.turns == 2
    assert adapter.posted == [
        ("terminal", config.risk_confirmation_notice),
        ("terminal", "Understood."),
    ]
    assert engine.plan_calls == 0
    assert engine.run_calls == 1
    assert engine.seen_notes == [config.risk_confirmation_notice]


async def test_a_turn_with_nothing_withheld_before_it_carries_no_note(tmp_path: Path):
    """The common case: no prior withhold, so the field stays ``None`` rather than stale."""
    _campaign(tmp_path)
    engine = _WithheldNoteEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _ActionAdapter("dodge"),
        engine,
    ).run()

    assert engine.seen_notes == [None]


class _RiskThenDeclareThenAskAdapter(_Adapter):
    """A withheld hazard, an unrelated declaration, then a game-master question.
    """

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "I attack the shadow", ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "I tie my boots", ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "GM, why was I refused?", ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )


async def test_an_unrelated_declaration_between_a_withhold_and_a_question_never_sees_the_note(
    tmp_path: Path,
):
    """The note answers a question, not merely 'the turn after a withhold.'
    """
    _campaign(tmp_path)
    adapter = _RiskThenDeclareThenAskAdapter()
    engine = _WithheldNoteEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=0)

    report = await NarratorService(config, adapter, engine).run()

    assert report.turns == 3
    assert engine.run_calls == 2  # the risk turn never reaches run_turn at all
    assert engine.seen_notes == [None, config.risk_confirmation_notice]


#: Three narrowing questions no pair of which is a near-duplicate of another. The
#: chain test needs that: ``repeats_answered_question`` would otherwise short-circuit
#: the loop before the view budget ever ran, and the test would pass for the wrong
#: reason. ``test_the_chain_questions_are_genuinely_distinct`` proves the property
#: rather than assuming it.
_CHAIN_QUESTIONS = (
    "Which shutter do you mean, the one on the water side or the one facing the alley?",
    "Do you lever it, cut the hinge, or lift the whole frame clear?",
    "How much noise are you willing to make getting through?",
)


def test_the_chain_questions_are_genuinely_distinct():
    """Guard the guard: the chain fixture must not accidentally be three repeats."""
    from narrator.service import REPETITION_SIMILARITY_THRESHOLD, narration_similarity

    for index, first in enumerate(_CHAIN_QUESTIONS):
        for second in _CHAIN_QUESTIONS[index + 1:]:
            assert narration_similarity(first, second) < REPETITION_SIMILARITY_THRESHOLD


def _clarification(question: str, marker: str) -> dict:
    return {
        "kind": "clarification",
        "question": question,
        "audience": {"kind": "current"},
        "paths": (
            {"id": f"{marker}_near", "label": f"Take the near way ({marker})"},
            {"id": f"{marker}_far", "label": f"Take the far way ({marker})"},
        ),
    }


class _FirstOptionAdapter(_Adapter):
    """Answer every presented view with its first listed option, every round."""

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "I carefully open the shutter",
                ChannelPrincipal("terminal", "terminal-player", "Rill"),
            ),
        )

    async def collect_decision(self, views):
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(
                presentation_token=view.presentation_token,
                selection_id=view.options[0].id,
            ),
        )


class _ClarificationChainEngine(_Engine):
    """A planner that always wants one more narrowing round, as the live one did."""

    def __init__(self):
        super().__init__()
        self.executed_resolutions = ()
        self.seen_prior_questions: list[tuple[str, ...]] = []

    async def plan_turn(self, turn, eligible, resolutions, session=None):
        self.seen_prior_questions.append(
            tuple(item.question for item in session.asked_questions) if session else ()
        )
        self.plan_calls += 1
        question = _CHAIN_QUESTIONS[min(self.plan_calls - 1, len(_CHAIN_QUESTIONS) - 1)]
        return PlanOutcome(
            plan=RequestDecisionPlan(
                kind="request", decision=_clarification(question, f"r{self.plan_calls}")
            )
        )

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_resolutions = tuple(decision_resolutions)
        return TurnOutcome("Rill eases the shutter open.", ratified=True, withheld=False)


async def test_a_two_clarification_chain_never_discards_the_answers_it_collected(
    tmp_path: Path,
):
    """Test a two clarification chain never discards the answers it collected.
    """
    _campaign(tmp_path)
    adapter = _FirstOptionAdapter()
    engine = _ClarificationChainEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    # Two views were presented and answered; the third request hit the budget.
    assert engine.plan_calls == 3
    assert len(adapter.presented) == 2

    # The claim: the turn ran, and it ran carrying both answers.
    assert report.decision_pending == 0
    assert engine.run_calls == 1
    assert len(engine.executed_resolutions) == 2
    assert [item.selection_id for item in engine.executed_resolutions] == [
        "r1_near", "r2_near",
    ]

    # Nothing was withheld and no boundary notice was posted.
    assert report.withheld == 0
    assert all("No action occurred" not in text for _, text in adapter.posted)
    assert all(
        fields.get("category") != "segment"
        for event, fields in adapter.records
        if event == "decision_lifecycle"
    )


class _RepeatingQuestionEngine(_Engine):
    """Ask one question, take its answer, then ask it again with permuted options.

    This is the live planner's recorded behaviour, not a hypothetical. Because
    ``planning_prompt`` sent only ``selection_id`` values the planner had authored
    itself one call earlier, it could not recognize its own question, and a transcript
    recorded the same clarification put to the player eight times.
    """

    _QUESTION = 'The opponent is badly injured. Which method will you use for the finishing strike?'

    def __init__(self):
        super().__init__()
        self.executed_resolutions = ()
        self.seen_prior_questions: list[tuple[str, ...]] = []

    async def plan_turn(self, turn, eligible, resolutions, session=None):
        self.seen_prior_questions.append(
            tuple(item.question for item in session.asked_questions) if session else ()
        )
        self.plan_calls += 1
        paths = [
            {"id": "thrust", "label": "Drive the knife in"},
            {"id": "throw", "label": "Throw the knife"},
        ]
        if self.plan_calls % 2 == 0:


            paths.reverse()
        return PlanOutcome(
            plan=RequestDecisionPlan(
                kind="request",
                decision={
                    "kind": "clarification",
                    "question": self._QUESTION,
                    "audience": {"kind": "current"},
                    "paths": tuple(paths),
                },
            )
        )

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_resolutions = tuple(decision_resolutions)
        return TurnOutcome("The knife goes home.", ratified=True, withheld=False)


class _FinishItAdapter(_FirstOptionAdapter):
    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", "finish it off",
                ChannelPrincipal("terminal", "terminal-player", "Rill"),
            ),
        )


async def test_the_planner_never_re_asks_a_question_the_chain_already_answered(
    tmp_path: Path,
):
    """The prompt half: the exact question text of every answered narrowing view reaches
    the next planner call, so a compliant planner can tell a new question from one it
    has already had answered. The mechanical half: a draft whose question near-repeats
    an answered one is refused before it is ever presented, so a planner that ignores
    the prompt cannot spend a second round of the player's attention on it.

    Against the code before this slice, the second call carries no question history at
    all and the repeat is presented: the player answers the identical question twice,
    the budget then closes the segment, and the engine never runs.
    """
    _campaign(tmp_path)
    adapter = _FinishItAdapter()
    engine = _RepeatingQuestionEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    # The prompt half: the first call saw no history, the second saw the answered one.
    assert engine.seen_prior_questions[0] == ()
    assert engine.seen_prior_questions[1] == (_RepeatingQuestionEngine._QUESTION,)

    # The mechanical half: the repeat was never presented, and the turn ran on the
    # answer the player had already given.
    assert engine.plan_calls == 2
    assert len(adapter.presented) == 1
    assert engine.run_calls == 1
    assert [item.selection_id for item in engine.executed_resolutions] == ["thrust"]
    assert report.decision_pending == 0
    assert all("No action occurred" not in text for _, text in adapter.posted)


class _SegmentThenNewDeclarationAdapter(_Adapter):
    """A withheld turn, then a materially different declaration and no control typed.
    """

    _FIRST = 'I walk toward the harbor'
    _SECOND = "I sit down by the fire and clean my knife"

    async def turns(self):
        principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        yield InboundTurn("terminal", ChannelMessage("Rill", self._FIRST, principal))
        yield InboundTurn("terminal", ChannelMessage("Rill", self._SECOND, principal))


class _RetainThenPlanEngine(_Engine):
    """Fault the first turn, which retains a session, then record what turn two plans."""

    def __init__(self):
        super().__init__()
        self.planned_actions: list[str] = []
        self.executed_text = ""

    async def plan_turn(self, turn, eligible, resolutions, session=None):
        self.plan_calls += 1
        self.planned_actions.append(
            session.effective_action if session is not None else turn.mention.text
        )
        if self.plan_calls == 1:
            from narrator.engine import DecisionPlanningError

            raise DecisionPlanningError("planner inputs are unavailable")
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_text = turn.mention.text
        return TurnOutcome("Rill settles by the fire.", ratified=True, withheld=False)


async def test_a_new_declaration_after_a_boundary_notice_is_planned_as_itself(
    tmp_path: Path,
):
    """Test a new declaration after a boundary notice is planned as itself.
    """
    _campaign(tmp_path)
    adapter = _SegmentThenNewDeclarationAdapter()
    engine = _RetainThenPlanEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert report.decision_faults == 1
    assert engine.plan_calls == 2
    # The claim: turn two was planned and narrated as the text the player just typed.
    assert engine.planned_actions[0] == _SegmentThenNewDeclarationAdapter._FIRST
    assert engine.planned_actions[1] == _SegmentThenNewDeclarationAdapter._SECOND
    assert engine.executed_text == _SegmentThenNewDeclarationAdapter._SECOND


class _ResumeAdapter(_ContinuationAdapter):
    """Log the order of planner calls and presentations across a segment boundary."""

    def __init__(self, events: list[str]):
        super().__init__()
        self._events = events

    async def deliver_decision_views(self, views):
        self._events.append("presented")
        return await super().deliver_decision_views(views)


class _ResumeEngine(_Engine):
    """Two confirmation requests, then proceed -- with every planner call logged."""

    _QUESTION = "The stair is rotten through. Commit to the climb?"

    def __init__(self, events: list[str]):
        super().__init__()
        self._events = events
        self.executed_resolutions = ()

    async def plan_turn(self, turn, eligible, resolutions, session=None):
        self._events.append("planned")
        self.plan_calls += 1
        if self.plan_calls <= 2:
            return PlanOutcome(
                plan=RequestDecisionPlan(
                    kind="request",
                    decision=ConfirmationDraft(
                        kind="confirmation", question=self._QUESTION,
                        audience={"kind": "current"},
                    ),
                )
            )
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_resolutions = tuple(decision_resolutions)
        return TurnOutcome("Rill starts up the stair.", ratified=True, withheld=False)


async def test_continue_resumes_the_pending_view_instead_of_issuing_a_planner_call(
    tmp_path: Path,
):
    """``/continue`` and ``/retry`` were byte-identical: both rebuilt the turn from the
    retained action and both issued a fresh planner call, because the retained state was
    only ``(session, resolutions)`` and held nothing to resume. The notice a segment
    boundary posts says the segment \"needs another player view\", so ``/continue`` now
    presents exactly the view the boundary refused, with no planner round trip -- while
    ``/retry`` keeps re-planning, which is what makes the two controls different.

    The discriminating assertion is the event order: the first thing the resumed
    segment does is present, not plan. Before this slice the fourth event was another
    ``planned``.
    """
    _campaign(tmp_path)
    events: list[str] = []
    adapter = _ResumeAdapter(events)
    engine = _ResumeEngine(events)

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1), adapter, engine
    ).run()

    assert events == ["planned", "presented", "planned", "presented", "planned"]
    # Turn one hit the view budget holding an unanswered confirmation; a confirmation is
    # a gate rather than a narrowing question, so the segment closed rather than
    # proceeding past it (see ``budget_boundary``).
    assert report.decision_pending == 1
    # The resumed view is the very draft the boundary refused, not a re-planned one.
    assert [view.question for view in adapter.presented] == [_ResumeEngine._QUESTION] * 2
    assert engine.run_calls == 1
    assert len(engine.executed_resolutions) == 2


class _EngineWithholdAdapter(_Adapter):
    """A turn the engine withholds, then the /retry the notice told the player to type.

    Nothing before this milestone armed a control on this path, so nothing could type
    one: ``deliver`` posted a notice ending "Use /retry, /revise, or /dismiss." and left
    ``decision_recovery`` uncalled. This adapter arms only when told to, so the second
    turn's control exists exactly when production says it does.
    """

    _DECLARATION = "I force the tower door"

    def __init__(self):
        super().__init__()
        self._control = ""
        self.recovery_calls = 0

    async def turns(self):
        principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        yield InboundTurn(
            "terminal",
            ChannelMessage("Rill", self._DECLARATION, principal),
            backfill=(ChannelMessage("Ossa", "The hinges look rusted through.", None),),
        )
        yield InboundTurn("terminal", ChannelMessage("Rill", "/retry", principal))

    async def decision_recovery(self):
        self.recovery_calls += 1
        self._control = "retry"

    def consume_decision_recovery_control(self):
        control = self._control
        self._control = ""
        return control


class _WithholdThenReplanEngine(_Engine):
    """Withhold turn one through the guard path, then record what /retry replans."""

    def __init__(self):
        super().__init__()
        self.planned_actions: list[str] = []
        self.executed_texts: list[str] = []

    async def plan_turn(self, turn, eligible, resolutions, session=None):
        self.plan_calls += 1
        self.planned_actions.append(
            session.effective_action if session is not None else turn.mention.text
        )
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_texts.append(turn.mention.text)
        if self.run_calls == 1:
            # The ``ResolutionGuard`` withhold shape: no narration, recovery advertised.
            return TurnOutcome(
                "", ratified=True, withheld=True, decision_recovery=True,
                error="the confirmed action rolled no resolving mechanic",
            )
        return TurnOutcome("The door gives.", ratified=True, withheld=False)


async def test_a_retry_after_an_engine_withhold_replans_the_declaration_as_typed(
    tmp_path: Path,
):
    """Three things had to hold at once for the advertised control to work, and none of
    them did. ``deliver`` had to arm the controls its own notice names. The channel had
    to hold a session for the armed control to act on, or ``_decision_phase``'s orphan
    branch returned ``\"pending\"``, which posts nothing at all -- the player types the
    control the game master just recommended and the game answers with silence. And the
    retained session had to carry the declaration the player typed rather than the
    rendering the narrator reads: ``channel_text`` puts backfill first and an ``@GM``
    marker before the mention, so a session started from it replans and narrates
    ``\"@GM I force the tower door\"``, with ``run_turn`` prefixing it a second time. The
    backfill line in this adapter's first turn exists to catch exactly that.
    """
    _campaign(tmp_path)
    adapter = _EngineWithholdAdapter()
    engine = _WithholdThenReplanEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    # Turn one withheld and its notice armed the controls it advertised.
    assert report.withheld == 1
    assert "/retry" in adapter.posted[0][1]
    assert adapter.recovery_calls == 1

    # Turn two is the control landing on something: it replanned and ran, rather than
    # returning "pending" and posting nothing.
    assert report.decision_pending == 0
    assert report.delivered == 1
    assert engine.plan_calls == 2
    assert engine.run_calls == 2

    # And what it replanned is the declaration as the player typed it.
    assert engine.planned_actions == [
        _EngineWithholdAdapter._DECLARATION, _EngineWithholdAdapter._DECLARATION,
    ]
    assert engine.executed_texts == [
        _EngineWithholdAdapter._DECLARATION, _EngineWithholdAdapter._DECLARATION,
    ]
    assert all("@GM" not in action for action in engine.planned_actions)
    assert all("hinges" not in text for text in engine.executed_texts)


class _RetryAfterWithholdAdapter(_Adapter):
    """A violent declaration the engine withholds, then the ``/retry`` it advertised."""

    _DECLARATION = "I attack rade"

    def __init__(self):
        super().__init__()
        self._control = ""
        self.recovery_calls = 0

    async def turns(self):
        principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        yield InboundTurn("terminal", ChannelMessage("Rill", self._DECLARATION, principal))
        yield InboundTurn("terminal", ChannelMessage("Rill", "/retry", principal))

    async def decision_recovery(self):
        self.recovery_calls += 1
        self._control = "retry"

    def consume_decision_recovery_control(self):
        control = self._control
        self._control = ""
        return control


class _WithholdOnceEngine(_Engine):
    """Withhold turn one through the guard path; record every planned and run text.

    ``plan_turn`` raises rather than returning: a risk-routed declaration is answered by
    the engine's own floor inside ``_routed_decision_phase``, never by the planner, so
    reaching this at all means the turn was routed on something other than the
    declaration's verdict.
    """

    def __init__(self):
        super().__init__()
        self.executed_texts: list[str] = []

    async def plan_turn(self, turn, eligible, resolutions, *, session=None, **kwargs):
        self.plan_calls += 1
        raise AssertionError("a risk-routed declaration must never reach the planner")

    async def run_turn(self, turn, decision_resolutions=()):
        self.run_calls += 1
        self.executed_texts.append(turn.mention.text)
        if self.run_calls == 1:
            # The ``ResolutionGuard`` withhold shape: no narration, recovery advertised.
            return TurnOutcome(
                "", ratified=True, withheld=True, decision_recovery=True,
                error="the confirmed action rolled no resolving mechanic",
            )
        return TurnOutcome("The knife goes in.", ratified=True, withheld=False)


async def test_retry_reclassifies_the_retained_declaration_not_the_control_token(
    tmp_path: Path,
):
    """The claim is the third ``classify_calls`` entry. Turn two's text is ``/retry``; the
    verdict the turn is routed on must be the retained declaration's, which is why the
    service classifies a third time. Before the fix the list ended at ``/retry`` and the
    hazard floor never ran on turn two -- ``/retry`` routes ``planner`` through
    ``lexical_double``, so the engine's planner answered a violence declaration instead
    of the floor. ``_WithholdOnceEngine.plan_turn`` raises for exactly that reason.
    """
    _campaign(tmp_path)
    adapter = _RetryAfterWithholdAdapter()
    engine = _WithholdOnceEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.classify_calls == [
        _RetryAfterWithholdAdapter._DECLARATION,
        "/retry",
        _RetryAfterWithholdAdapter._DECLARATION,
    ]
    assert engine.plan_calls == 0
    # Both turns asked the engine-authored hazard confirmation, so the retry did not
    # slip past the floor, and neither turn narrated the control token.
    assert len(adapter.presented) == 2
    assert engine.executed_texts == [_RetryAfterWithholdAdapter._DECLARATION] * 2
    assert report.withheld == 1
    assert report.delivered == 1


class _ReadingControlEngine(_WithholdOnceEngine):
    """The defect's own condition: the control token classifies as a pure read.

    ``lexical_double`` happens to route ``/retry`` to ``planner``, which is the benign
    end of the bug. The dangerous end is a verdict of ``read``: ``_routed_decision_phase``
    returns ``continue`` for that route before any hazard gate, before the turn is
    rebuilt around the retained action, and before the control is consumed -- so the
    declaration narrates with no confirmation, the model is handed the literal
    ``/retry`` as the player's line, and the control stays armed for the next turn.
    Production cannot be made to answer ``read`` on demand, so the double is steered
    here: any verdict for the token that is not the declaration's will do.
    """

    async def classify_intent(self, declaration, *, scope=None, combat=None, offer_open=False):
        self.classify_calls.append(declaration)
        return await fake_classify_intent(
            "I look around" if declaration.startswith("/") else declaration,
            scope=scope, combat=combat, offer_open=offer_open,
        )


async def test_a_control_token_that_classifies_read_cannot_bypass_the_hazard_floor(
    tmp_path: Path,
):
    """This is the assertion the diagnosis was written for. With the token classified
    ``read``, the pre-fix service returned ``continue`` at the read gate holding the
    *original* turn, so ``run_turn`` was handed ``/retry`` verbatim and the violence was
    resolved with no confirmation at all -- and because consumption sat below that gate,
    the control was still armed afterwards.
    """
    _campaign(tmp_path)
    adapter = _RetryAfterWithholdAdapter()
    engine = _ReadingControlEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    # The floor ran on the retry: a second confirmation was asked.
    assert len(adapter.presented) == 2
    # The model never saw transport syntax, and never resolved the violence unasked.
    assert engine.executed_texts == [_RetryAfterWithholdAdapter._DECLARATION] * 2
    assert "/retry" not in engine.executed_texts
    # The control was consumed by the turn that carried it.
    assert adapter.consume_decision_recovery_control() == ""


class _ReviseAdapter(_Adapter):
    """A withheld declaration, then ``/revise <new text>``.

    The terminal channel yields the revised text as the mention (``terminal.py``'s
    ``/revise `` branch), never the control token, which is why ``/revise`` was never
    part of this defect and must not grow a second classification now.
    """

    _DECLARATION = "I attack rade"
    _REVISED = "I walk to the quay"

    def __init__(self):
        super().__init__()
        self._control = ""

    async def turns(self):
        principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        yield InboundTurn("terminal", ChannelMessage("Rill", self._DECLARATION, principal))
        yield InboundTurn("terminal", ChannelMessage("Rill", self._REVISED, principal))

    async def decision_recovery(self):
        self._control = "revise"

    def consume_decision_recovery_control(self):
        control = self._control
        self._control = ""
        return control


class _ProceedingWithholdEngine(_WithholdOnceEngine):
    """As above, but the planner proceeds: the revised action routes ``planner``."""

    async def plan_turn(self, turn, eligible, resolutions, *, session=None, **kwargs):
        self.plan_calls += 1
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))


async def test_revise_is_classified_once_because_the_channel_yields_its_own_text(
    tmp_path: Path,
):
    """The scope boundary of the fix. ``/revise`` already arrived correctly classified,
    so it must still spend exactly one classification per turn -- adding a second would
    be a real cost (one model call) bought for nothing.
    """
    _campaign(tmp_path)
    adapter = _ReviseAdapter()
    engine = _ProceedingWithholdEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.classify_calls == [_ReviseAdapter._DECLARATION, _ReviseAdapter._REVISED]
    assert engine.executed_texts[-1] == _ReviseAdapter._REVISED


class _UnreadableRetryEngine(_WithholdOnceEngine):
    """The classifier answers for the declaration, then cannot answer for the retry."""

    async def classify_intent(self, declaration, *, scope=None, combat=None, offer_open=False):
        self.classify_calls.append(declaration)
        if len(self.classify_calls) > 2:
            raise RuntimeError("endpoint hiccup")
        return await fake_classify_intent(
            declaration, scope=scope, combat=combat, offer_open=offer_open
        )


async def test_a_retry_whose_reclassification_fails_withholds_instead_of_proceeding(
    tmp_path: Path,
):
    """Fail closed, and closed here means withheld outright.

    ``requires_risk_confirmation(None)`` is True, so the floor deeper in would have
    confirmed rather than bypassed; this stops short of even that. The engine does not
    know what the retained declaration does, so it resolves nothing, posts the same
    notice a freshly typed unclassifiable turn gets, and holds the session so the
    advertised control still has something to act on.
    """
    _campaign(tmp_path)
    adapter = _RetryAfterWithholdAdapter()
    engine = _UnreadableRetryEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert adapter.posted[-1] == ("terminal", config.classifier_fault_notice)
    # Turn one ran and withheld; the retry resolved nothing at all.
    assert engine.run_calls == 1
    assert engine.executed_texts == [_RetryAfterWithholdAdapter._DECLARATION]


_ORDINARY_TRADE_WORD_TURNS = (
    "take a break",
    "buy nothing",
    "take the satchel",
    'how many turns would it take to reach the harbor?',
    '@GM how many turns would it take to reach the harbor?',
)


def _market(tmp_path: Path) -> None:
    """One campaign whose scene holds the merchant this section trades with."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "---\nsession: 1\nlocation_id: market\n---\n\n# Market\n\n## Present NPCs\n\n- rade\n",
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(
        '{"event_seq": 1, "fiction_debt": [], "pending_rulings": []}', encoding="utf-8"
    )
    (campaign / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n",
        encoding="utf-8",
    )


class _ScriptedAdapter(_Adapter):
    """One channel, several turns, answering every confirmation with confirm."""

    def __init__(self, *texts: str):
        super().__init__()
        self.texts = texts

    async def turns(self):
        for text in self.texts:
            yield InboundTurn(
                "terminal",
                ChannelMessage("Rill", text, ChannelPrincipal("terminal", "terminal-player", "Rill")),
            )


class _RingMerchant:
    """The engine-only purchase port, quoting one item and recording every debit.
    """

    def __init__(self):
        self.confirmed = []

    def quote_for_text(self, seller_id: str, text: str) -> dict:
        del text
        return {
            "ok": True, "seller_id": seller_id, "item_id": "ring",
            "public_label": "silver ring", "quantity": 1, "currency": "copper",
            "price_copper": 2, "stock_version": 4,
        }

    def confirm(self, request) -> dict:
        self.confirmed.append(request)
        return {"ok": True, "outcome": "confirmed"}

    def reserve_social_ability(self, actor_id: str, ability_id: str) -> dict:
        del actor_id, ability_id
        return {"ok": False}


async def test_a_completed_purchase_never_answers_the_next_declaration(tmp_path: Path):
    """A completed purchase end to end, then two ordinary declarations after it.
    """
    _market(tmp_path)
    merchant = _RingMerchant()
    adapter = _ScriptedAdapter(
        "Rade, what is the price for the ring?",
        "Deal.",
        "I buy the ring now.",
        "take the satchel",
        'how many turns would it take to reach the harbor?',
    )
    engine = _ProceedPlannerEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    report = await NarratorService(config, adapter, engine, purchase_executor=merchant).run()

    assert report.turns == 5
    assert report.errors == []
    # The legitimate purchase ran once, through the authenticated confirmation.
    assert len(merchant.confirmed) == 1
    assert merchant.confirmed[0].item_id == "ring"
    assert len(adapter.presented) == 1
    assert adapter.presented[0].kind == "confirmation"
    posted = [text for _, text in adapter.posted]
    assert config.trade_completed_notice in posted
    # Neither declaration after the purchase was intercepted.
    assert config.trade_confirmation_notice not in posted
    assert [text for _, text in adapter.posted][-2:] == ["The seal opens."] * 2
    assert engine.run_calls == 4  # every turn but the confirming one reached narration


async def test_an_ordinary_trade_word_is_narrated_on_a_channel_with_no_trade(tmp_path: Path):
    """Test an ordinary trade word is narrated on a channel with no trade.
    """
    for index, declaration in enumerate(_ORDINARY_TRADE_WORD_TURNS):
        root = tmp_path / str(index)
        root.mkdir()
        _market(root)
        adapter = _ScriptedAdapter(declaration)
        engine = _ProceedPlannerEngine()
        config = NarratorConfig(campaign_root=root, max_decision_rounds=2)

        report = await NarratorService(
            config, adapter, engine, purchase_executor=_RingMerchant()
        ).run()

        assert report.errors == []
        assert engine.run_calls == 1, declaration
        assert adapter.posted == [("terminal", "The seal opens.")], declaration
        assert adapter.presented == [], declaration


async def test_a_negotiation_stalled_at_confirmation_still_reports_itself(tmp_path: Path):
    """The notice keeps the one case it is true of, so the two tests above are not vacuous.

    A frame whose confirmation was opened and then abandoned is a live negotiation the
    player cannot complete by repeating themselves, and
    ``_trade_confirmation_phase``'s early return still says so.
    """
    _market(tmp_path)
    adapter = _ScriptedAdapter("I buy the ring now.")
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
    service = NarratorService(
        config, adapter, _ProceedPlannerEngine(), purchase_executor=_RingMerchant()
    )
    scope = read_trusted_scope(tmp_path)
    state = service._social  # noqa: SLF001 - trade-frame seam
    state.accept_offer(
        "terminal",
        scope=scope,
        seller=InteractionCue("canonical_npc", "rade"),
        terms=TradeTerms("rade", "ring", "silver ring", 1, "copper", 2, 4),
        delivered=True,
    )
    state.advance_trade("terminal", scope=scope, policy=policy_for("Deal.", scope=scope))
    state.advance_trade("terminal", scope=scope, policy=policy_for("I buy the ring now.", scope=scope))
    assert state.begin_confirmation("terminal", scope=scope, decision_key="stranded") is not None

    await service.run()

    assert adapter.posted == [("terminal", config.trade_confirmation_notice)]


class _AssessingRiskEngine(_DecliningRiskEngine):
    """A risk-confirming engine whose typed assessor answers with one fixed verdict."""

    def __init__(self, verdict):
        super().__init__()
        self._verdict = verdict
        self.assess_calls = 0

    async def assess_hazard(self, declaration, *, scope=None, combat=None):
        self.assess_calls += 1
        if isinstance(self._verdict, Exception):
            raise self._verdict
        return self._verdict


async def test_a_report_shaped_risk_turn_proceeds_when_the_assessor_downgrades(tmp_path: Path):
    """Test a report shaped risk turn proceeds when the assessor downgrades.
    """
    from narrator.assess import HazardAssessment

    _campaign(tmp_path)
    adapter = _DecliningRiskAdapter("i attacked him")
    engine = _AssessingRiskEngine(HazardAssessment(kind="report", reason="disputes past events"))

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.assess_calls == 1
    assert engine.run_calls == 1  # the turn reached the narrator instead of a prompt
    assert adapter.presented == []
    assert ("decision_lifecycle", {"category": "assessed", "decision_kind": "report"}) in adapter.records


async def test_a_declaration_verdict_keeps_the_risk_floor(tmp_path: Path):
    from narrator.assess import HazardAssessment

    _campaign(tmp_path)
    adapter = _DecliningRiskAdapter()
    engine = _AssessingRiskEngine(HazardAssessment(kind="declaration", reason="acts now"))
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)

    await NarratorService(config, adapter, engine).run()

    assert engine.assess_calls == 1
    assert engine.run_calls == 0
    assert adapter.posted == [("terminal", config.decision_declined_notice)]


async def test_an_assessor_fault_keeps_the_risk_floor(tmp_path: Path):
    """Every assessor failure -- an exception here, a None verdict, or an engine
    without the capability at all -- leaves the lexical floor exactly as it shipped."""
    _campaign(tmp_path)
    for verdict in (RuntimeError("endpoint down"), None):
        adapter = _DecliningRiskAdapter()
        engine = _AssessingRiskEngine(verdict)
        config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2)
        await NarratorService(config, adapter, engine).run()
        assert engine.run_calls == 0
        assert adapter.posted == [("terminal", config.decision_declined_notice)]
