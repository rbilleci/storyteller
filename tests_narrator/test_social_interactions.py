"""Deterministic service contracts for in-world social interaction routing.

Routing is a model call now (``narrator.classify``), so the classifier this suite feeds
the engine is ``fake_classifier``: the retired English lexicon, kept verbatim as
``lexical_double``, rendered back into the schema the model answers in. That keeps every
assertion below deterministic and offline while still driving the real seams --
``InteractionTracker.policy``, ``classify.policy_from``, the service's fail-closed
branch. What it cannot establish is classification quality; only
``tests_narrator/test_probe_classifier.py``, against the live endpoint, speaks to that.

Where a test calls ``classify_turn`` or ``reply_shaped`` directly it is measuring the
double rather than production routing, and says so.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import lexical_double
import pytest
from fake_classifier import ClassifyingEngine, fake_classify_intent, policy_for
from lexical_double import classify_social_input, classify_turn

from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.channels.replay import TranscriptReplayAdapter
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission
from narrator.delivery import TurnOutcome
from narrator.engine import NarratorEngine
from narrator.interactions import InteractionTracker
from narrator.policy_types import InteractionCue, TrustedScope
from narrator.prompt import TURN_BODY_MARKER, turn_prompt
from narrator.service import NarratorService, PreparedNarrationTurn
from narrator.social import (
    SocialState,
    TradePhase,
    TradeTerms,
)


def _campaign(root: Path, present: str = "") -> None:
    campaign = root / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "---\nsession: 1\nlocation_id: market\n---\n\n# Market\n\n## Present NPCs\n\n"
        + (f"- {present}\n" if present else "- None recorded.\n"),
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(
        '{"event_seq": 1, "fiction_debt": [], "pending_rulings": []}', encoding="utf-8"
    )
    (campaign / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n",
        encoding="utf-8",
    )


class _StructuredSocialAdapter:
    name = "terminal"
    decision_capabilities = ChannelCapabilities(structured_decisions=True, atomic_decision_delivery=True)

    def __init__(self, messages: tuple[tuple[str, str], ...]) -> None:
        self.messages = messages
        self.posted: list[tuple[str, str]] = []
        self.presented: list[object] = []
        self.recoveries = 0

    async def turns(self):
        for channel_id, text in self.messages:
            yield InboundTurn(
                channel_id,
                ChannelMessage(
                    "Rill", text, ChannelPrincipal("terminal", "terminal-player", "Rill")
                ),
            )

    async def post(self, channel_id, text):
        self.posted.append((channel_id, text))

    async def close(self):
        return None

    async def deliver_decision_views(self, views):
        self.presented.extend(views)
        raise AssertionError("a direct social turn must not render a decision view")

    async def collect_decision(self, views):
        raise AssertionError("a direct social turn must not collect a decision")

    async def acknowledge_decision(self, result):
        raise AssertionError(result)

    async def decision_recovery(self):
        self.recoveries += 1


class _SocialEngine(ClassifyingEngine):
    """A narrator that never plans, carrying the classifier the service requires.

    Without ``classify_intent`` -- which ``ClassifyingEngine`` supplies from the lexical
    double -- ``NarratorService`` reads every turn as a classifier fault and withholds
    it, so a fake engine that omits it measures nothing but the fault notice.
    """

    def __init__(self, outcomes: tuple[TurnOutcome, ...] = ()) -> None:
        super().__init__()
        self.plan_calls = 0
        self.turns: list[PreparedNarrationTurn] = []
        self._outcomes = list(outcomes)

    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    async def plan_turn(self, turn, eligible, resolutions, *, session=None):
        self.plan_calls += 1
        raise AssertionError("social routing must force ProceedPlan before planner invocation")

    async def run_turn(self, turn, decision_resolutions=(), decision_action_fingerprint=""):
        assert isinstance(turn, PreparedNarrationTurn)
        self.turns.append(turn)
        if self._outcomes:
            return self._outcomes.pop(0)
        return TurnOutcome("The trader answers in character.", ratified=True, withheld=False)


class _ChoiceSocialAdapter(_StructuredSocialAdapter):
    async def deliver_decision_views(self, views):
        self.presented.extend(views)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(tuple(views)))

    async def collect_decision(self, views):
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="attribute_int"),
        )


async def test_social_greeting_then_name_bypasses_every_decision_surface(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _StructuredSocialAdapter(
        (
            ("market", "approach the traders selling rope and say hello"),
            ("market", "whats your name?"),
        )
    )
    engine = _SocialEngine()

    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert engine.plan_calls == 0
    assert adapter.presented == []
    assert report.decision_pending == 0
    assert report.decision_faults == 0
    assert adapter.recoveries == 0
    assert report.delivered == 2
    assert len(adapter.posted) == 2
    assert engine.turns[1].mention.text == "whats your name?"
    assert engine.turns[1].channel_text() == "@GM whats your name?"
    assert engine.turns[1].interaction_cue is not None
    assert engine.turns[1].interaction_cue.kind == "scene_interlocutor"
    prompt = turn_prompt(engine.turns[1].channel_text(), interaction_cue=engine.turns[1].interaction_cue)
    _, _, trusted_block = prompt.partition("Active in-world interlocutor:")
    assert "whats your name?" not in trusted_block
    assert "Do not ask the player to identify the narrator" in trusted_block


async def test_named_and_anonymous_focus_use_only_validated_scene_npc_identifiers(tmp_path: Path):
    _campaign(tmp_path, "sera-vane")
    tracker = InteractionTracker()

    named = await tracker.policy(
        "market", "approach sera-vane and say hello", tmp_path, classify=fake_classify_intent
    )
    assert named.route == "social"
    assert named.focus_candidate is not None
    assert named.focus_candidate.kind == "canonical_npc"
    assert named.focus_candidate.public_npc_id == "sera-vane"
    tracker.complete_delivery("market", named, tmp_path)
    follow_up = await tracker.policy(
        "market", "what’s your name?", tmp_path, classify=fake_classify_intent
    )
    # The cue carries forward, which is this test's subject and is the tracker's own
    # behavior. The follow-up's *route* is deliberately not asserted here any more.
    # The retired classifier decided it from ``active_focus`` -- a question asked with a
    # live interlocutor routed ``social`` rather than ``read`` -- and the model
    # classifier is never given the focus: it reads the message and the scene, and
    # ``policy_from`` applies the focus afterwards. The offline double therefore cannot
    # reproduce a focus-dependent route, and asserting one here would measure the
    # double's blind spot rather than production.
    #
    # Production does route it ``social``; that is measured against the endpoint by
    # ``test_probe_classifier.test_the_live_classifier_routes_a_follow_up_to_the_interlocutor``,
    # which also pins the complement -- a question addressed to nobody routes ``read``
    # and still keeps the cue.
    assert follow_up.interaction_cue == named.focus_candidate

    anonymous = await tracker.policy(
        "other", "approach the traders and say hello", tmp_path, classify=fake_classify_intent
    )
    assert anonymous.focus_candidate is not None
    assert anonymous.focus_candidate.kind == "scene_interlocutor"
    assert anonymous.focus_candidate.public_npc_id is None


def test_social_result_reader_uses_mcp_structured_content_before_text():
    payload = NarratorEngine._tool_result_payload(
        {"structuredContent": {"ok": True, "outcome": "failure"}, "content": [{"text": "ignored"}]}
    )
    assert payload == {"ok": True, "outcome": "failure"}


def test_social_result_reader_uses_a_model_result_structured_content():
    class _ModelResult:
        structuredContent = {"ok": True, "outcome": "success"}  # noqa: N815 -- the real MCP SDK's own field name

    assert NarratorEngine._tool_result_payload(_ModelResult()) == {
        "ok": True,
        "outcome": "success",
    }


def test_name_apostrophe_variants_and_precedence_are_deterministic():
    scope = TrustedScope("1", "market", ())
    active = classify_turn("approach the traders and say hello", scope=scope).focus_candidate
    assert active is not None
    for text in ("whats your name?", "what's your name?", "what’s your name?"):
        policy = classify_turn(text, scope=scope, active_focus=active)
        assert policy.route == "social"
        assert policy.interaction_cue == active
    assert classify_turn("approach the traders selling rope and say hello", scope=scope).route == "social"
    compound = classify_turn("say hello, then steal their rope", scope=scope, active_focus=active)
    assert compound.route == "risk"
    assert compound.risk_category == "theft"
    question = classify_turn("What happens if I stab the trader?", scope=scope)
    assert question.route == "read"
    assert question.risk_category == "none"
    committed_question = classify_turn("I stab the trader?", scope=scope)
    assert committed_question.route == "risk"
    assert committed_question.risk_category == "violence"
    ooc = classify_turn("OOC: say hello to the trader", scope=scope, active_focus=active)
    assert ooc.route == "out_of_character"
    assert ooc.interaction_cue is None
    assert classify_turn("bribe the trader", scope=scope, active_focus=active).route == "social"


async def test_focus_is_channel_scoped_invalidated_by_scope_and_unchanged_after_failed_delivery(tmp_path: Path):
    _campaign(tmp_path, "sera-vane")
    tracker = InteractionTracker()
    first = await tracker.policy(
        "one", "approach sera-vane and say hello", tmp_path, classify=fake_classify_intent
    )
    tracker.complete_delivery("one", first, tmp_path)
    assert (
        await tracker.policy("two", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is None
    assert (
        await tracker.policy("one", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue == first.focus_candidate

    _campaign_scene = tmp_path / "campaign" / "scene.md"
    _campaign_scene.write_text(
        "---\nsession: 2\nlocation_id: elsewhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    assert (
        await tracker.policy("one", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is None

    failed = InteractionTracker()
    failed_policy = await failed.policy(
        "one", "approach the traders and say hello", tmp_path, classify=fake_classify_intent
    )
    assert failed_policy.focus_candidate is not None
    assert (
        await failed.policy("one", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is None


async def test_social_departure_clears_focus_only_after_successful_delivery(tmp_path: Path):
    _campaign(tmp_path)
    tracker = InteractionTracker()
    greeting = await tracker.policy(
        "market", "approach the traders and say hello", tmp_path, classify=fake_classify_intent
    )
    tracker.complete_delivery("market", greeting, tmp_path)

    departure = await tracker.policy(
        "market", "say goodbye to the traders and leave", tmp_path, classify=fake_classify_intent
    )
    assert departure.route == "social"
    assert departure.clears_focus is True
    assert departure.interaction_cue == greeting.focus_candidate

    tracker.complete_delivery("market", departure, tmp_path)
    assert (
        await tracker.policy("market", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is None

    failed = InteractionTracker()
    failed.complete_delivery(
        "market",
        await failed.policy(
            "market", "say hello to the traders", tmp_path, classify=fake_classify_intent
        ),
        tmp_path,
    )
    failed_departure = await failed.policy(
        "market", "say goodbye to the traders and leave", tmp_path, classify=fake_classify_intent
    )
    assert failed_departure.clears_focus is True
    assert (
        await failed.policy("market", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is not None


async def test_an_ordinary_first_person_reply_no_longer_drops_the_focus(tmp_path: Path):
    """An ordinary first person reply no longer drops the focus. Synthetic fixtures exercise this contract."""
    _campaign(tmp_path)
    tracker = InteractionTracker()

    greet = await tracker.policy(
        "market",
        'I greet a trader and ask what she has for sale',
        tmp_path,
        classify=fake_classify_intent,
    )
    assert greet.route == "social"
    tracker.complete_delivery("market", greet, tmp_path)
    assert (
        await tracker.policy("market", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is not None

    reply = await tracker.policy(
        "market", 'I need a coil of good rope.', tmp_path, classify=fake_classify_intent
    )
    assert reply.clears_focus is False  # the fix: no departure signal, so no clearing
    tracker.complete_delivery("market", reply, tmp_path)

    negotiate = await tracker.policy(
        "market",
        'would you accept three coins instead?',
        tmp_path,
        classify=fake_classify_intent,
    )
    assert negotiate.interaction_cue is not None
    assert negotiate.interaction_cue == greet.focus_candidate


async def test_a_genuine_departure_still_clears_the_focus_through_an_unrelated_reply(tmp_path: Path):
    """The fix above narrows ``clears_focus`` to ``_departure`` alone; this pins that a
    real departure declaration still clears it, so the terminal fallback did not simply
    stop clearing focus altogether."""
    _campaign(tmp_path)
    tracker = InteractionTracker()
    greet = await tracker.policy(
        "market",
        'I greet a trader and ask what she has for sale',
        tmp_path,
        classify=fake_classify_intent,
    )
    tracker.complete_delivery("market", greet, tmp_path)

    leave = await tracker.policy(
        "market", "I leave the stall and walk away", tmp_path, classify=fake_classify_intent
    )
    assert leave.clears_focus is True
    tracker.complete_delivery("market", leave, tmp_path)
    assert (
        await tracker.policy("market", "whats your name?", tmp_path, classify=fake_classify_intent)
    ).interaction_cue is None


async def test_failed_narration_does_not_commit_focus_and_capability_free_replay_still_runs(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _StructuredSocialAdapter(
        (("market", "approach the traders and say hello"), ("market", "whats your name?"))
    )
    engine = _SocialEngine(
        (TurnOutcome("", ratified=False, withheld=True), TurnOutcome("The trader replies.", True, False))
    )
    await NarratorService(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine).run()
    assert engine.turns[1].interaction_cue is None

    transcript = tmp_path / "social.txt"
    transcript.write_text("@GM approach the traders and say hello\n@GM whats your name?\n", encoding="utf-8")
    replay = TranscriptReplayAdapter(transcript, echo=False)
    replay_engine = _SocialEngine()
    report = await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), replay, replay_engine
    ).run()
    assert report.delivered == 2
    assert replay_engine.plan_calls == 0


async def test_successful_departure_clears_service_focus_but_failed_delivery_does_not(tmp_path: Path):
    _campaign(tmp_path)
    messages = (
        ("market", "approach the traders and say hello"),
        ("market", "say goodbye to the traders and leave"),
        ("market", "whats your name?"),
    )
    successful_engine = _SocialEngine()
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _StructuredSocialAdapter(messages),
        successful_engine,
    ).run()
    assert successful_engine.turns[2].interaction_cue is None

    failed_engine = _SocialEngine(
        (
            TurnOutcome("The traders greet Rill.", True, False),
            TurnOutcome("", False, True),
            TurnOutcome("The traders answer.", True, False),
        )
    )
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2),
        _StructuredSocialAdapter(messages),
        failed_engine,
    ).run()
    assert failed_engine.turns[2].interaction_cue is not None


async def test_explicit_bargain_binds_charisma_without_calling_the_planner(tmp_path: Path):
    _campaign(tmp_path, "rade")
    adapter = _StructuredSocialAdapter((("market", "I bargain for a better price."),))
    engine = _SocialEngine()
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()
    assert engine.plan_calls == 0
    request = engine.turns[0].social_test
    assert request is not None
    assert request.actor_id == "rill"
    assert request.allowed_attributes == ("CHA",)


async def test_social_attribute_choice_uses_authenticated_approach_ui(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _ChoiceSocialAdapter((("market", "I debate the facts."),))
    engine = _SocialEngine()

    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()

    assert len(adapter.presented) == 1
    assert engine.turns[0].social_test is not None
    assert engine.turns[0].social_test.allowed_attributes == ("INT",)


async def test_ooc_rules_inquiry_has_no_social_frame_or_bound_test(tmp_path: Path):
    _campaign(tmp_path, "rade")
    adapter = _StructuredSocialAdapter((("market", "OOC: how does bargaining work?"),))
    engine = _SocialEngine()
    await NarratorService(
        NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2), adapter, engine
    ).run()
    assert engine.turns[0].interaction_cue is None
    assert engine.turns[0].social_test is None


async def test_committed_hazard_never_bypasses_confirmation_when_decisions_are_disabled(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _StructuredSocialAdapter((("market", "I stab the trader?"),))
    engine = _SocialEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=0)

    report = await NarratorService(config, adapter, engine).run()

    assert report.delivered == 0
    assert engine.turns == []
    assert adapter.posted == [("market", config.risk_confirmation_notice)]


async def test_romance_escalation_blocks_before_narration_even_when_enabled_without_consent(tmp_path: Path):
    _campaign(tmp_path)
    adapter = _StructuredSocialAdapter((("market", "I kiss the stranger."),))
    engine = _SocialEngine()
    config = NarratorConfig(
        campaign_root=tmp_path,
        max_decision_rounds=2,
        romance_escalation_enabled=True,
    )

    await NarratorService(config, adapter, engine).run()

    assert engine.turns == []
    assert adapter.posted == [("market", config.romance_boundary_notice)]


async def test_romance_escalation_requires_every_resolved_player_principal(tmp_path: Path):
    _campaign(tmp_path, "rade")
    (tmp_path / "campaign" / "players.yaml").write_text(
        "players:\n"
        "- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n"
        "- discord_user_id: mara-player\n  character_id: mara\n  display_name: Mara\n",
        encoding="utf-8",
    )
    adapter = _StructuredSocialAdapter(
        (("market", "Rade, hello."), ("market", "I kiss Mara."))
    )
    engine = _SocialEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2, romance_escalation_enabled=True)
    service = NarratorService(config, adapter, engine)
    assert service.record_romance_consent("market", ChannelPrincipal("terminal", "terminal-player", "Rill"))
    await service.run()
    assert len(engine.turns) == 1
    assert engine.turns[0].interaction_cue is not None

    consenting_adapter = _StructuredSocialAdapter(
        (("market", "Rade, hello."), ("market", "I kiss Mara."))
    )
    consenting_engine = _SocialEngine()
    consenting = NarratorService(config, consenting_adapter, consenting_engine)
    assert consenting.record_romance_consent("market", ChannelPrincipal("terminal", "terminal-player", "Rill"))
    assert consenting.record_romance_consent("market", ChannelPrincipal("terminal", "mara-player", "Mara"))
    await consenting.run()
    assert len(consenting_engine.turns) == 2
    assert consenting_engine.turns[-1].interaction_cue is None


async def test_romance_escalation_rejects_a_player_and_npc_target_ambiguity(tmp_path: Path):
    _campaign(tmp_path, "mara")
    (tmp_path / "campaign" / "players.yaml").write_text(
        "players:\n"
        "- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n"
        "- discord_user_id: mara-player\n  character_id: mara\n  display_name: Mara\n",
        encoding="utf-8",
    )
    adapter = _StructuredSocialAdapter(
        (("market", "Mara, hello."), ("market", "I kiss Mara."))
    )
    engine = _SocialEngine()
    config = NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2, romance_escalation_enabled=True)
    service = NarratorService(config, adapter, engine)
    assert service.record_romance_consent("market", ChannelPrincipal("terminal", "terminal-player", "Rill"))
    assert service.record_romance_consent("market", ChannelPrincipal("terminal", "mara-player", "Mara"))

    await service.run()

    assert len(engine.turns) == 1
    assert adapter.posted[-1] == ("market", config.romance_boundary_notice)


def test_no_cue_prompt_remains_identical_and_prepared_turn_preserves_channel_text():
    raw = "Rill: hello\n@GM whats your name?"
    expected = f"{TURN_BODY_MARKER}{raw}\n\nResolve this turn."
    assert turn_prompt(raw) == expected
    source = InboundTurn("market", ChannelMessage("Rill", "whats your name?"))
    prepared = PreparedNarrationTurn(source, None)
    assert prepared.channel_text() == source.channel_text()


def test_narration_invites_reply_reads_only_the_closing_line():
    """The signal is the narration's last line, so a mid-paragraph question does not arm it."""
    from narrator.interactions import narration_invites_reply

    assert narration_invites_reply('He turns to you. "What is your name?"') is True
    assert narration_invites_reply("Orso studies you.\n\nWho sent you?") is True
    assert narration_invites_reply("**And your name?**") is True  # emphasis closes after the mark
    assert narration_invites_reply("The lamp gutters.") is False
    assert narration_invites_reply("She asks your name? Then she turns away.") is False
    assert narration_invites_reply("") is False
    assert narration_invites_reply("   \n\n  ") is False


def test_is_bare_yes_or_no_admits_only_a_committed_answer_word():
    """Is bare yes or no admits only a committed answer word. Synthetic fixtures exercise this contract."""
    from lexical_double import is_bare_yes_or_no

    for reply in ("Yes", "yes.", "Yeah", "Sure", "No", "Nope"):
        assert is_bare_yes_or_no(reply), reply
    for reply in ("Maybe", "yes, strike him", "yes and take the cleaver", "I attack Rade", ""):
        assert not is_bare_yes_or_no(reply), reply


def test_a_reply_to_a_direct_question_routes_social_without_a_speech_verb():
    """Regression: the same answer routed two ways depending on a speech verb.
    """
    from narrator.policy_types import InteractionCue

    scope = TrustedScope("1", "market", ("orso-pell",))
    focus = InteractionCue("canonical_npc", "orso-pell")
    scope = TrustedScope("1", "market", ("orso-pell",), ("rill",))
    for reply in ("Rill", "My name is Rill", "yes", "no", "nope"):
        invited = classify_turn(reply, scope=scope, active_focus=focus, awaiting_reply=True)
        assert invited.route == "social", reply
        assert invited.interaction_cue == focus  # the reply stays with its interlocutor
        assert invited.retains_focus is True
        # Without the invitation the same words keep the route that shipped before.
        assert classify_turn(reply, scope=scope, active_focus=focus).route == "planner", reply
    # "Rill of the low water" leaves three unrecognized words, so the bound declines it and
    # it keeps the planner route. The bound fails toward the behaviour that shipped before
    # the invitation, never toward capturing a turn it cannot read.
    assert classify_turn(
        "Rill of the low water", scope=scope, active_focus=focus, awaiting_reply=True
    ).route == "planner"
    # The speech-verb form already routed social, and still does.
    assert classify_turn(
        "I say my name is Rill", scope=scope, active_focus=focus
    ).route == "social"


def test_the_reply_invitation_changes_no_earlier_route():
    """It sits at the fallback, so every route above it must classify identically."""
    from narrator.policy_types import InteractionCue

    scope = TrustedScope("1", "market", ("orso-pell",))
    focus = InteractionCue("canonical_npc", "orso-pell")
    for text in (
        "I stab the barkeep",          # risk
        "I steal the rope",            # risk
        "open the door",               # committed action, planner
        "Look at the traders",         # read
        "What are they selling?",      # question
        "ooc how does initiative work",  # out of character
    ):
        invited = classify_turn(text, scope=scope, active_focus=focus, awaiting_reply=True)
        baseline = classify_turn(text, scope=scope, active_focus=focus)
        assert invited == baseline, text
    # An invitation with no interlocutor binds nothing, so a bare reply stays planner.
    assert classify_turn("Rill", scope=scope, awaiting_reply=True).route == "planner"


async def test_the_tracker_carries_the_invitation_with_the_focus(tmp_path: Path):
    """The invitation rides with the focus, so every path that drops one drops both."""

    _campaign(tmp_path, present="orso-pell")
    tracker = InteractionTracker()
    policy = await tracker.policy(
        "market", "hello orso pell", tmp_path, classify=fake_classify_intent
    )
    assert policy.focus_candidate is not None

    tracker.complete_delivery("market", policy, tmp_path, 'Orso Pell asks, "Who sent you?"')
    assert (
        await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "social"

    # A narration that asks nothing leaves the next bare turn on its planner route.
    tracker_two = InteractionTracker()
    tracker_two.complete_delivery("market", policy, tmp_path, "Orso Pell turns back to his ledger.")
    assert (
        await tracker_two.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "planner"

    # An omitted narration records no invitation, which preserves the older call site.
    tracker_three = InteractionTracker()
    tracker_three.complete_delivery("market", policy, tmp_path)
    assert (
        await tracker_three.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "planner"


#: Declarations an audit collected when the first invitation captured the whole fallback
#: class. They are retained as a regression floor, never as the sample: a test that
#: asserts only over the examples which produced it certifies the bound against itself.
_AUDIT_FALLBACK_ACTIONS = (
    "I search the crates", "I pick the lock", "I sneak past him", "I draw my knife",
    "search the body", "I hide behind the barrel", "follow him", "I pull the lever",
    "I push the door", "I count the coins", "I put on the ring", "I wait", "I rest",
)

#: Declarations chosen independently of that audit, covering shapes its list omits:
#: an imperative with no object, a two-verb chain, a possessive object, and a number.
_INDEPENDENT_FALLBACK_ACTIONS = (
    "kneel", "swim across", "untie the mooring rope", "empty his purse onto the table",
    "trade places with the boy", "whistle twice", "cut my palm", "12",
)


def test_the_invitation_never_captures_an_action_declaration():
    """Sample the fallback class from the module's own lexicons, not from the audit.

    The first version of this branch tested the invitation and the focus and nothing about
    the declaration, so an action declaration routed social with the interlocutor cue
    attached. The planner never saw it and the prompt framed a physical act as speech.

    The sample is generated from every verb the module's action sets name, in three
    shapes. A verb added to any of those sets therefore extends this test with no edit here.

    "The module" is ``lexical_double`` now: the lexicons moved there with the classifier
    they belong to, and the bound is still what the offline invitation path obeys.
    """
    scope = TrustedScope("1", "market", ("orso-pell",), ("rill",))
    focus = InteractionCue("canonical_npc", "orso-pell")
    verbs = sorted(lexical_double._mechanic_vocabulary())
    # Four shapes. The filler-led one is the shape an audit found admitted, where a
    # recognized word stands ahead of the verb: "then run", "back away", "into the water".
    fillers = sorted(lexical_double._TARGETLESS_WORDS | lexical_double._ANSWER_WORDS)
    shapes = ("{verb}", "{verb} him", "I {verb} the crate")
    generated = [shape.format(verb=verb) for verb in verbs for shape in shapes]
    generated += [f"{filler} {verb}" for verb in verbs for filler in fillers]
    for declaration in (*generated, *_AUDIT_FALLBACK_ACTIONS, *_INDEPENDENT_FALLBACK_ACTIONS):
        invited = classify_turn(declaration, scope=scope, active_focus=focus, awaiting_reply=True)
        baseline = classify_turn(declaration, scope=scope, active_focus=focus)
        assert invited == baseline, declaration


def test_the_defence_literals_and_recovery_controls_never_read_as_answers():
    """A later slice must not defeat an accepted one, and an audit found this doing so.

    The earlier bound exempted every one-word declaration, so "dodge" and "parry" routed
    social. That path returns from `_decision_phase` before the gate binding
    `CombatDefendDirective`, which is the mechanic the accepted predecessor slice added.
    The same exemption swallowed the decision-recovery controls the service consumes.
    """
    party = ("rill",)
    for literal in sorted(lexical_double._DEFENCE_METHODS):
        assert lexical_double.reply_shaped(literal, party) is False, literal
    for control in ("continue", "retry", "revise", "dismiss", "help", "quit"):
        assert lexical_double.reply_shaped(control, party) is False, control


def test_reply_shaped_admits_only_the_vocabulary_it_recognizes():
    """Bound which declarations may be admitted, rather than asserting a residual holds.

    An assertion that a residual still exists cannot fail when the residual widens. This
    enumerates the admitting side instead: a one-word declaration passes only when the
    recognized vocabulary already holds it.

    ``reply_shaped`` is ``lexical_double``'s now; production reads the classifier's
    ``bare_answer`` instead. The bound is kept because the double still answers every
    offline invitation case, and because it is the record of what the audit closed.
    """
    party = ("rill", "sella")
    recognized = (
        lexical_double._TARGETLESS_WORDS | lexical_double._ANSWER_WORDS | frozenset(party)
    )
    sample = sorted(recognized) + [
        "dodge", "run", "kneel", "sprint", "twelve", "orso", "ledger", "whistle", "xyzzy",
    ]
    for word in sample:
        # A first-person subject is recognized vocabulary and still rejected, because it
        # marks a declared act. "we" is the case that separates the two conditions.
        expected = word in recognized and word not in lexical_double._FIRST_PERSON_SUBJECTS
        assert lexical_double.reply_shaped(word, party) is expected, word

    # Names are recognized rather than exempted, so an unlisted character is not an answer.
    assert lexical_double.reply_shaped("Rill", party) is True
    assert lexical_double.reply_shaped("Sella", party) is True
    assert lexical_double.reply_shaped("Rill", ()) is False

    # Multi-word answers, and the shapes the three conditions exclude.
    assert lexical_double.reply_shaped("My name is Rill", party) is True
    assert lexical_double.reply_shaped("I am Rill", party) is False       # first-person subject
    assert lexical_double.reply_shaped("follow him", party) is False      # leading unknown word
    assert lexical_double.reply_shaped("search the body", party) is False  # two unknown words
    assert lexical_double.reply_shaped("", party) is False
    # Declines an answer it cannot recognize, which is the safe direction.
    assert lexical_double.reply_shaped("Rill of the low water", party) is False

    # A recognized word ahead of an unrecognized one does not make an answer. An audit
    # found the earlier bound admitting every one of these.
    for declaration in ("then run", "back away", "up the ladder", "into the water",
                        "back to the ship", "no run", "not run"):
        assert lexical_double.reply_shaped(declaration, party) is False, declaration

    # The admitting residual, stated as an assertion rather than as prose: an
    # unrecognized word directly behind a copula is read as the fact the answer supplies.
    assert lexical_double.reply_shaped("my name is Torvald", party) is True
    assert lexical_double.reply_shaped("it is broken", party) is True


async def test_the_invitation_never_outlives_the_narration_that_armed_it(tmp_path: Path):
    """A turn that keeps a focus without replacing it must still rewrite the invitation.

    `complete_delivery` returns early when a policy carries no focus candidate and clears
    nothing, which is the shape of the out-of-character and read policies. An audit found
    that path leaving a stale invitation, so an answer-shaped turn two narrations later
    still routed social.
    """
    _campaign(tmp_path, present="orso-pell")
    tracker = InteractionTracker()
    greeting = await tracker.policy(
        "market", "hello orso pell", tmp_path, classify=fake_classify_intent
    )
    tracker.complete_delivery("market", greeting, tmp_path, 'Orso Pell asks, "Who sent you?"')
    assert (
        await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "social"

    # An out-of-character turn keeps the focus and carries no candidate. Its narration
    # asks nothing, so the invitation must not survive it.
    ooc = await tracker.policy(
        "market", "ooc how does initiative work", tmp_path, classify=fake_classify_intent
    )
    assert ooc.route == "out_of_character"
    assert ooc.focus_candidate is None and ooc.clears_focus is False
    tracker.complete_delivery("market", ooc, tmp_path, "Initiative runs on a DEX test.")
    assert (
        await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "planner"


async def test_a_players_file_change_invalidates_the_channel_focus(tmp_path: Path):
    """The invalidation edge `TrustedScope.party_name_tokens` introduced had no test.

    The field participates in scope equality, so editing `campaign/players.yaml` changes
    the scope and drops the focus. An audit recorded the gap and the fail-safe direction.
    """
    _campaign(tmp_path, present="orso-pell")
    tracker = InteractionTracker()
    greeting = await tracker.policy(
        "market", "hello orso pell", tmp_path, classify=fake_classify_intent
    )
    tracker.complete_delivery("market", greeting, tmp_path, 'Orso Pell asks, "Who sent you?"')
    assert (
        await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "social"

    (tmp_path / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: sella\n"
        "  display_name: Sella\n",
        encoding="utf-8",
    )

    # The scope changed, so the focus and its invitation are gone.
    assert (
        await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    ).route == "planner"


async def test_an_unreadable_players_file_drops_the_focus_rather_than_failing(tmp_path: Path):
    """`_read_party_name_tokens` returns nothing on any error, and the turn still routes."""
    _campaign(tmp_path, present="orso-pell")
    tracker = InteractionTracker()
    greeting = await tracker.policy(
        "market", "hello orso pell", tmp_path, classify=fake_classify_intent
    )
    tracker.complete_delivery("market", greeting, tmp_path, 'Orso Pell asks, "Who sent you?"')

    (tmp_path / "campaign" / "players.yaml").write_text("players: [oh dear\n", encoding="utf-8")

    policy = await tracker.policy("market", "Rill", tmp_path, classify=fake_classify_intent)
    assert policy.route == "planner"
    assert policy.scope.party_name_tokens == ()


def test_read_character_display_names_maps_id_to_name(tmp_path: Path):
    """The pairing `_read_party_name_tokens` flattens away is what constructing a
    roll announcement needs: a specific name for a specific id, not a token set."""
    from narrator.interactions import read_character_display_names

    _campaign(tmp_path)
    (tmp_path / "campaign" / "players.yaml").write_text(
        "players:\n"
        "- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n"
        "- discord_user_id: other-player\n  character_id: ossa\n  display_name: Ossa\n",
        encoding="utf-8",
    )
    assert read_character_display_names(tmp_path) == {"rill": "Rill", "ossa": "Ossa"}


def test_read_character_display_names_returns_empty_on_any_read_failure(tmp_path: Path):
    from narrator.interactions import read_character_display_names

    assert read_character_display_names(tmp_path) == {}
    _campaign(tmp_path)
    (tmp_path / "campaign" / "players.yaml").write_text("players: [oh dear\n", encoding="utf-8")
    assert read_character_display_names(tmp_path) == {}


_ZERO_CONTEXT_TRADE_WORDS = (
    ("take a break", "planner"),
    ("buy nothing", "social"),
    ("take the satchel", "planner"),
    ('how many turns would it take to reach the harbor?', "planner"),
    ('@GM how many turns would it take to reach the harbor?', "out_of_character"),
)

_MARKET_SCOPE = TrustedScope("1", "market", ("rade",))


def _ring_terms(price: int = 2) -> TradeTerms:
    """Ring terms.
    """
    return TradeTerms("rade", "ring", "silver ring", 1, "copper", price, 4)


def _agreed_ring_frame(state: SocialState, channel_id: str = "market"):
    """Drive one channel to an agreed price through the public trade API only."""
    state.accept_offer(
        channel_id,
        scope=_MARKET_SCOPE,
        seller=InteractionCue("canonical_npc", "rade"),
        terms=_ring_terms(),
        delivered=True,
    )
    return state.advance_trade(channel_id, scope=_MARKET_SCOPE, policy=policy_for("Deal.", scope=_MARKET_SCOPE))


@pytest.mark.parametrize(
    ("declaration", "route"),
    tuple(pair for pair in _ZERO_CONTEXT_TRADE_WORDS if "take" in pair[0]),
)
def test_a_bare_take_is_no_purchase_intent_without_a_negotiation(declaration, route):
    """No scope, no seller, no trade history: none of these states a purchase.

    Every one routed to ``social`` with ``trade_phase == "purchase_intent"`` before this
    milestone, purely because it contained the word "take", and that is what sent them to
    the service's trade gate. Each now takes the route its own shape earns: a planner
    declaration, or an out-of-character question where the player addressed the narrator.
    """
    assert classify_social_input(declaration).trade_phase is None
    policy = classify_turn(declaration, scope=_MARKET_SCOPE)
    assert policy.trade_phase == ""
    assert policy.route == route


def test_a_trade_word_that_survives_classification_is_stopped_by_the_missing_frame():
    """"buy nothing" still reads as trade vocabulary, and still buys nothing.

    "Buy" is a purchase word wherever it appears, so narrowing the classifier is the
    wrong place to answer this one -- the classifier is right and the missing fact is
    the frame. ``advance_trade`` answers ``None`` for a channel holding no negotiation,
    and ``NarratorService._advancing_purchase`` (pinned end to end in
    tests_narrator/test_service.py) reads the same absence and routes the turn to
    ordinary narration rather than to the purchase-confirmation notice.
    """
    assert classify_social_input("buy nothing").trade_phase is TradePhase.PURCHASE_INTENT
    assert classify_turn("buy nothing", scope=_MARKET_SCOPE).route == "social"
    assert SocialState().advance_trade(
        "market", scope=_MARKET_SCOPE, policy=policy_for("buy nothing", scope=_MARKET_SCOPE)
    ) is None


def test_a_bare_take_states_a_purchase_only_inside_an_advancing_negotiation():
    """The live frame is the missing fact, and it is the only thing that supplies it."""
    assert classify_social_input("I take it").trade_phase is None
    assert (
        classify_social_input("I take it", negotiating=True).trade_phase
        is TradePhase.PURCHASE_INTENT
    )
    # "buy" is trade vocabulary on its own and needs no frame to be read as one.
    assert classify_social_input("I buy the rope now.").trade_phase is TradePhase.PURCHASE_INTENT
    # The flag reaches nothing else: every other stage classifies identically.
    for text in ("Deal.", "Rade, what is the price for rope?", "I would pay 3 copper"):
        assert (
            classify_social_input(text).trade_phase
            == classify_social_input(text, negotiating=True).trade_phase
        )


_SYNTHETIC_BUCKLE_DECLARATIONS = (
    'I tell the trader: "The buckle is scratched. I can pay 7 coins for it."',
    "Collect the buckle",
    'I pay 7 coins and collect the buckle',
    '@GM Why is collecting the agreed buckle blocked?',
    "pay for the buckle",
)


def test_buckle_declarations_without_a_frame_do_not_authorize_purchase():
    """Synthetic trade vocabulary must not authorize an unconfirmed purchase."""
    for declaration in _SYNTHETIC_BUCKLE_DECLARATIONS:
        phase = classify_social_input(declaration).trade_phase
        assert phase is not TradePhase.PURCHASE_INTENT, (declaration, phase)
    assert classify_social_input(_SYNTHETIC_BUCKLE_DECLARATIONS[0]).trade_phase is None
    assert classify_social_input(_SYNTHETIC_BUCKLE_DECLARATIONS[-1]).trade_phase is None


def test_a_channel_with_no_trade_frame_advances_nothing():
    state = SocialState()
    for declaration, _route in _ZERO_CONTEXT_TRADE_WORDS:
        assert state.advance_trade("market", scope=_MARKET_SCOPE, policy=policy_for(declaration, scope=_MARKET_SCOPE)) is None


def test_an_agreed_negotiation_still_reaches_purchase_intent():
    """Test an agreed negotiation still reaches purchase intent.
    """
    state = SocialState()
    agreed = _agreed_ring_frame(state)
    assert agreed is not None and agreed.phase is TradePhase.AGREEMENT
    assert agreed.agreed_price == 2
    bought = state.advance_trade(
        "market", scope=_MARKET_SCOPE, policy=policy_for("I buy the ring now.", scope=_MARKET_SCOPE)
    )
    assert bought is not None and bought.phase is TradePhase.PURCHASE_INTENT
    # A bare "take" accepts the agreed terms exactly as it did before this milestone,
    # because the frame it is read against is advancing.
    state_two = SocialState()
    _agreed_ring_frame(state_two)
    taken = state_two.advance_trade("market", scope=_MARKET_SCOPE, policy=policy_for("I take it", scope=_MARKET_SCOPE, offer_open=True))
    assert taken is not None and taken.phase is TradePhase.PURCHASE_INTENT


def test_a_completed_purchase_frame_stops_being_the_channels_live_frame():
    """``confirm_purchase`` wrote ``COMPLETED`` back into the channel's trade map and
    nothing removed it, so the finished frame answered every later turn for the life of
    the process.
    """
    state = SocialState()
    _agreed_ring_frame(state)
    state.advance_trade("market", scope=_MARKET_SCOPE, policy=policy_for("I buy the ring now.", scope=_MARKET_SCOPE))
    assert state.begin_confirmation("market", scope=_MARKET_SCOPE, decision_key="key-1") is not None
    assert (
        state.confirm_purchase(
            "market",
            scope=_MARKET_SCOPE,
            decision_key="key-1",
            actor_id="rill",
            action_fingerprint="f" * 64,
            purchase=lambda *_: "confirmed",
        )
        == "confirmed"
    )
    assert state.trade_frame("market", _MARKET_SCOPE) is None
    for declaration in ("take the satchel", 'how many turns would it take to reach the harbor?'):
        assert state.advance_trade("market", scope=_MARKET_SCOPE, policy=policy_for(declaration, scope=_MARKET_SCOPE)) is None


def test_a_cancelled_negotiation_stops_being_the_channels_live_frame():
    """The same terminal rule, reached through the other terminal phase."""
    state = SocialState()
    _agreed_ring_frame(state)
    state.cancel("market")
    assert state.trade_frame("market", _MARKET_SCOPE) is None
    assert state.advance_trade("market", scope=_MARKET_SCOPE, policy=policy_for("I take it", scope=_MARKET_SCOPE, offer_open=True)) is None


_PROBE_DIRECTORY = str(Path(__file__).resolve().parents[1] / "scripts")


def _probe_module():
    if _PROBE_DIRECTORY not in sys.path:
        sys.path.insert(0, _PROBE_DIRECTORY)
    import probe_social_interactions

    return probe_social_interactions


def test_the_post_purchase_declaration_holds_its_own_scripted_turn():
    """The scripted flow really does complete a purchase first, then declare.

    This probe drives one scripted sequence rather than a ``SCENARIOS`` tuple, so the only
    thing binding ``_ORDINARY_DECLARATION_TURN`` to the declaration it names is the turn
    list's own order. Reordering the turns, or rewording the sixth one into something that
    states a purchase, would leave the live check passing while measuring a different claim.
    """
    probe = _probe_module()

    async def _scripted_texts():
        return [turn.mention.text async for turn in probe._RuntimeProbeAdapter().turns()]

    texts = asyncio.run(_scripted_texts())
    assert len(texts) == probe._SCRIPTED_TURNS
    assert len(probe._EXPECTED_OUTPUT_CATEGORIES) == probe._SCRIPTED_TURNS
    # The purchase completes, and is answered by its own completion notice, before the
    # declaration this milestone measures.
    assert texts[probe._PURCHASE_TURN - 1] == "I buy the rope now."  # the probe campaign stocks rope
    assert probe._EXPECTED_OUTPUT_CATEGORIES[probe._PURCHASE_TURN - 1] == "trade_completed_notice"
    assert probe._ORDINARY_DECLARATION_TURN > probe._COIN_COUNT_TURN > probe._PURCHASE_TURN
    declaration = texts[probe._ORDINARY_DECLARATION_TURN - 1]

    assert declaration == "take the satchel"
    assert not {"buy", "purchase", "pay", "deal", "price"} & set(declaration.split())
    assert probe._EXPECTED_OUTPUT_CATEGORIES[probe._ORDINARY_DECLARATION_TURN - 1] == "narration"


def test_the_probe_separates_a_service_notice_from_narration():
    """The typed classifier the M12 live check reads, pinned to its inputs.
    """
    from types import SimpleNamespace

    from narrator.delivery import TurnPost

    probe = _probe_module()
    adapter = probe._RuntimeProbeAdapter()
    adapter.output_categories = [None, None, None, None]
    service = SimpleNamespace(post_log=[
        TurnPost(kind="notice", origin="service", notice_key="trade_confirmation",
                 notice_text="NOTICE:trade-confirmation"),
        TurnPost(kind="narration", origin="turn",
                 model_text="Rade counts out the coil of rope and nods."),
        TurnPost(kind="notice", origin="service", notice_key="trade_completed",
                 notice_text="NOTICE:trade-completed"),
        TurnPost(kind="notice", origin="turn", notice_text="NOTICE:keyless"),
    ])
    probe._classify_output_categories(adapter, service)
    assert adapter.output_categories == [
        "trade_confirmation_notice", "narration", "trade_completed_notice", "notice",
    ]

    short = probe._RuntimeProbeAdapter()
    short.output_categories = [None]
    with pytest.raises(RuntimeError):
        probe._classify_output_categories(short, service)


def test_the_live_check_splits_the_stale_frame_from_the_planners_own_variance():
    """The M12 discriminator, and the sampled check it is deliberately kept apart from.

    ``_ordinary_declaration_narrated`` is the sampled claim, and the two must not
    collapse into one: a live attempt recorded the planner spending the round budget on
    that declaration and posting ``decision_fault_notice``. That turn was released by the
    trade path, which is what this milestone closes, and scoring it as a stale-frame
    failure would misattribute another milestone's behaviour to this one.
    """
    probe = _probe_module()
    no_stale = probe._no_stale_trade_notice
    narrated_check = probe._ordinary_declaration_narrated

    narrated = ["narration", "narration", "narration", "trade_completed_notice",
                "narration", "narration"]
    assert no_stale(narrated) and narrated_check(narrated)

    swallowed = list(narrated)
    swallowed[5] = "trade_confirmation_notice"
    assert not no_stale(swallowed)
    assert not narrated_check(swallowed)

    elsewhere = list(narrated)
    elsewhere[4] = "trade_confirmation_notice"
    assert not no_stale(elsewhere)

    # The planner's own round budget: released by the trade path, not narrated.
    planner_fault = list(narrated)
    planner_fault[5] = "decision_fault_notice"
    assert no_stale(planner_fault)
    assert not narrated_check(planner_fault)

    for short in (narrated[:5], []):
        assert not no_stale(short)
        assert not narrated_check(short)


def test_the_probe_names_every_category_it_can_record():
    """The known set derives from the catalog's own notice keys, so a new notice
    added to ``locale/en/narrator.yaml`` extends this probe's vocabulary here or
    fails loud -- never a silent narration misfile."""
    probe = _probe_module()
    config = NarratorConfig(campaign_root=Path("."))
    derived = {f"{key}_notice" for key in config.catalog.notices} | {"narration", "notice"}
    assert derived == set(probe._KNOWN_OUTPUT_CATEGORIES)


def test_the_purchase_confirmation_check_reads_the_purchase_turn():
    """The negotiate-then-buy flow's original bar, restated against the turn that asked.
    """
    probe = _probe_module()
    purchase_turn = probe._PURCHASE_TURN
    bound = probe._purchase_confirmation_bound

    assert bound([{"turn": purchase_turn, "kind": "confirmation"}])
    # A later turn's own decision leaves the purchase's verdict alone.
    assert bound(
        [{"turn": purchase_turn, "kind": "confirmation"}, {"turn": 6, "kind": "clarification"}]
    )
    # Everything the original assertion refused, it still refuses.
    assert not bound([])
    assert not bound([{"turn": purchase_turn, "kind": "clarification"}])
    assert not bound([{"turn": 2, "kind": "approach"}, {"turn": purchase_turn, "kind": "confirmation"}])
    assert not bound([{"turn": purchase_turn, "kind": "confirmation"}] * 2)
    assert not bound([{"turn": 6, "kind": "confirmation"}])
