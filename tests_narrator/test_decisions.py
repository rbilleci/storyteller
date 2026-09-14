"""Focused contracts for tool-less structured player decisions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fake_classifier import classification_for, fake_classify_intent, policy_for

# The retired English classifier, kept as a deterministic offline double so these
# assertions keep measuring the behavior they were written for. See
# ``tests_narrator/lexical_double.py`` for what it does and does not establish.
from lexical_double import classify_turn, defence_method
from pydantic import ValidationError

from narrator.channels.base import ChannelMessage, ChannelPrincipal, InboundTurn
from narrator.classify import policy_from
from narrator.decision_coordinator import DecisionCoordinator
from narrator.decisions import (
    ApproachDraft,
    ApproachOption,
    AttributeTestDirective,
    AttributeTestProposal,
    CharacterAudience,
    ChoiceOption,
    ClarificationDraft,
    CombatDefendDirective,
    CombatDefendProposal,
    ConfirmationDraft,
    CurrentAudience,
    DecisionResolution,
    DecisionSubmission,
    DeclinedPlan,
    HazardResolutionDirective,
    NoTestDirective,
    NoTestProposal,
    PlannerPolicyError,
    PlanOutcome,
    ProceedPlan,
    ProgressiveDecisionSession,
    RequestDecisionPlan,
    defence_obligation_text,
    hazard_obligation_text,
    planning_bypass,
    planning_prompt,
    requires_risk_confirmation,
    resolution_from_submission,
    risk_confirmation,
    verify_plan,
)
from narrator.interactions import combat_sanctions_violence, hazard_scope
from narrator.player_directory import PlayerDirectory
from narrator.policy_types import (
    CombatSnapshot,
    InteractionCue,
    TrustedScope,
    TurnPolicy,
    turn_framing_for,
)
from narrator.prompt import turn_prompt
from narrator.resolution_guard import ResolutionGuard


def _policy(
    declaration: str, *, scope: TrustedScope | None = None, combat=None
) -> TurnPolicy:
    """The turn's routing verdict, derived the way ``NarratorService.run`` derives one.

    ``planning_bypass`` and ``requires_risk_confirmation`` no longer classify anything:
    they take the single verdict the turn already computed. This builds that verdict
    offline -- the lexical double's answer rendered into the classifier's schema, then
    put through the real ``policy_from`` -- so the engine-owned policy under test is
    production's, and only the model call is stood in for.
    """
    scene = hazard_scope(scope, combat)
    return policy_from(classification_for(declaration, scope=scene, combat=combat), scope=scene)


def _fight(*npc_ids: str, active: bool = True) -> CombatSnapshot:
    return CombatSnapshot(
        active=active,
        round=1,
        active_actor="rill",
        order=("rill", *npc_ids),
        sides=(("rill", "pc"), *((npc, "npc") for npc in npc_ids)),
    )


def _campaign(tmp_path: Path, players: str | None = None) -> PlayerDirectory:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text("# Low water\n", encoding="utf-8")
    (campaign / "state.json").write_text('{"event_seq": 7, "fiction_debt": []}', encoding="utf-8")
    (campaign / "players.yaml").write_text(
        players
        or """players:
- discord_user_id: discord-rill
  character_id: rill
  display_name: Rill
- discord_user_id: discord-ossa
  character_id: ossa
  display_name: Ossa
""",
        encoding="utf-8",
    )
    return PlayerDirectory(tmp_path)


def test_planner_envelope_is_extra_forbid_and_discriminated():
    assert PlanOutcome(plan=ProceedPlan(kind="proceed")).plan.kind == "proceed"
    requested = PlanOutcome.model_validate(
        {
            "plan": {
                "kind": "request",
                "decision": {
                    "kind": "confirmation",
                    "question": "Enter the cursed gate?",
                    "audience": {"kind": "current"},
                },
            }
        }
    )
    assert isinstance(requested.plan, RequestDecisionPlan)
    with pytest.raises(ValidationError):
        PlanOutcome.model_validate({"plan": {"kind": "proceed", "tool": "rest"}})
    with pytest.raises(ValidationError):
        PlanOutcome.model_validate({"plan": {"kind": "declined"}})


@pytest.mark.parametrize("unsafe", ["A\x9bcontrol", "Call <|tool_call>{}", "Close <turn|>"])
def test_public_decision_text_rejects_controls_and_delivery_markup(unsafe):
    with pytest.raises(ValidationError):
        ConfirmationDraft(
            kind="confirmation",
            question=unsafe,
            audience={"kind": "current"},
        )


async def test_planner_uses_the_tool_less_structured_output_path(tmp_path: Path):
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    observed = {}

    class _Model:
        async def structured_output(self, schema, messages, *, system_prompt):
            observed.update(schema=schema, messages=messages, system_prompt=system_prompt)
            yield {"output": PlanOutcome(plan=ProceedPlan(kind="proceed"))}

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))
    engine._model = lambda **kwargs: _Model()  # noqa: SLF001 - the model boundary is the unit
    planned = await engine._plan_once("declare")  # noqa: SLF001 - focused planner seam
    assert planned.plan.kind == "proceed"
    assert observed["schema"] is PlanOutcome
    assert engine._agents == {}
    assert engine._client is None


async def test_repair_planner_uses_zero_temperature_without_constructing_an_agent(tmp_path: Path):
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    observed = {}

    class _Model:
        async def structured_output(self, schema, messages, *, system_prompt):
            yield {"output": PlanOutcome(plan=ProceedPlan(kind="proceed"))}

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))

    def _model(**kwargs):
        observed.update(kwargs)
        return _Model()

    engine._model = _model  # noqa: SLF001 - focused guided-output seam
    result = await engine._plan_once("repair", repair=True)  # noqa: SLF001

    assert result.plan.kind == "proceed"
    assert observed["temperature"] == 0.0
    assert observed["origin"] == "decision_repair"
    assert engine._agents == {}


async def test_engine_bypasses_facts_and_observation_before_the_planner(tmp_path: Path):
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))
    calls = []

    async def _unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("planner should not run")

    engine._plan_once = _unexpected  # noqa: SLF001 - deterministic bypass seam
    # An unstarted engine's own ``classify_intent`` answers ``None``, which withholds
    # the turn rather than routing it. The offline double stands in for the model call
    # so this measures the branch it names.
    engine.classify_intent = fake_classify_intent
    for declaration in (
        "What are the traders selling?",
        "I inspect the room",
        "How do I use this door?",
    ):
        outcome = await engine.plan_turn(
            InboundTurn("terminal", ChannelMessage("Rill", declaration)), (), ()
        )
        assert outcome.plan.kind == "proceed"
    assert calls == []


async def test_engine_forces_the_risk_floor_before_the_planner(tmp_path: Path):
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))

    async def _unexpected(*args, **kwargs):
        raise AssertionError("risk floor should decide first")

    engine._plan_once = _unexpected  # noqa: SLF001 - deterministic policy seam
    # An unstarted engine's own ``classify_intent`` answers ``None``, which withholds
    # the turn rather than routing it. The offline double stands in for the model call
    # so this measures the branch it names.
    engine.classify_intent = fake_classify_intent
    outcome = await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "Punch a trader")), (), ()
    )
    assert isinstance(outcome.plan, RequestDecisionPlan)
    assert outcome.plan.decision.kind == "confirmation"


async def test_engine_bypasses_a_bare_yes_or_no_answering_its_own_question(tmp_path: Path):
    """Test engine bypasses a bare yes or no answering its own question.
    """
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))

    async def _unexpected(*args, **kwargs):
        raise AssertionError("planner should not run")

    engine._plan_once = _unexpected  # noqa: SLF001 - deterministic bypass seam
    # An unstarted engine's own ``classify_intent`` answers ``None``, which withholds
    # the turn rather than routing it. The offline double stands in for the model call
    # so this measures the branch it names.
    engine.classify_intent = fake_classify_intent
    question = 'Will you spend an action attacking Rade with the knife?'
    for declaration in ("Yes", "yeah", "No", "nope"):
        outcome = await engine.plan_turn(
            InboundTurn("terminal", ChannelMessage("Rill", declaration)), (), (),
            recent_narration=question,
        )
        assert outcome.plan.kind == "proceed", declaration

    # The same bare word still reaches the planner without an antecedent question --
    # this is not a general shortcut for "yes"/"no", only for answering one.
    reached_planner = False

    async def _mark(*args, **kwargs):
        nonlocal reached_planner
        reached_planner = True
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))

    engine._plan_once = _mark  # noqa: SLF001
    await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "Yes")), (), (),
        recent_narration="Rade swings the cleaver and misses.",
    )
    assert reached_planner


async def test_party_answers_collect_independently_out_of_order(tmp_path: Path):
    directory = _campaign(tmp_path)
    snapshot = directory.snapshot()
    coordinator = DecisionCoordinator()
    draft = ClarificationDraft(
        kind="clarification",
        question="How do you cross the canal?",
        audience={"kind": "party"},
        paths=(ChoiceOption(id="bridge", label="Use the bridge"),),
    )
    rill = ChannelPrincipal("test", "discord-rill", "not-authority")
    ossa = ChannelPrincipal("test", "discord-ossa", "spoofable-name")
    prepared, views = await coordinator.prepare(
        draft=draft, snapshot=snapshot, principal=rill, context_fingerprint="fresh"
    )
    assert {view.character_id for view in views} == {"rill", "ossa"}
    assert "discord" not in " ".join(view.model_dump_json() for view in views)
    assert await coordinator.activate(prepared.request_id)
    by_character = {view.character_id: view for view in views}
    results = await asyncio.gather(
        coordinator.submit(
            principal=ossa,
            submission=DecisionSubmission(
                presentation_token=by_character["ossa"].presentation_token,
                selection_id="own_approach",
                custom_text="I swim under the reeds.",
            ),
            context_fingerprint="fresh",
        ),
        coordinator.submit(
            principal=rill,
            submission=DecisionSubmission(
                presentation_token=by_character["rill"].presentation_token,
                selection_id="bridge",
            ),
            context_fingerprint="fresh",
        ),
    )
    assert {result.status for result in results} == {"accepted", "completed"}
    resolutions = await coordinator.resolutions(prepared.request_id)
    assert [item.character_id for item in resolutions] == ["rill", "ossa"]
    assert resolutions[1].answer_kind == "custom"


async def test_submission_rejects_spoof_duplicate_conflict_stale_and_unknown(tmp_path: Path):
    directory = _campaign(tmp_path)
    snapshot = directory.snapshot()
    coordinator = DecisionCoordinator()
    draft = ConfirmationDraft(
        kind="confirmation", question="Break the seal?", audience={"kind": "current"}
    )
    rill = ChannelPrincipal("test", "discord-rill", "Rill")
    prepared, (view,) = await coordinator.prepare(
        draft=draft, snapshot=snapshot, principal=rill, context_fingerprint="old"
    )
    assert await coordinator.activate(prepared.request_id)
    spoof = ChannelPrincipal("test", "discord-ossa", "Rill")
    unauthorized = await coordinator.submit(
        principal=spoof,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="confirm"),
        context_fingerprint="old",
    )
    assert unauthorized.status == "unauthorized"
    stale = await coordinator.submit(
        principal=rill,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="confirm"),
        context_fingerprint="new",
    )
    assert stale.status == "stale"
    replay = await coordinator.submit(
        principal=rill,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="confirm"),
        context_fingerprint="old",
    )
    assert replay.status == "stale"
    unknown = await coordinator.submit(
        principal=rill,
        submission=DecisionSubmission(presentation_token="x" * 32, selection_id="confirm"),
        context_fingerprint="old",
    )
    assert unknown.status == "unknown"


async def test_current_subset_decline_and_authorized_cancellation(tmp_path: Path):
    directory = _campaign(tmp_path)
    snapshot = directory.snapshot()
    coordinator = DecisionCoordinator()
    rill = ChannelPrincipal("test", "discord-rill")
    subset = ConfirmationDraft(
        kind="confirmation",
        question="Let Ossa light the fuse?",
        audience=CharacterAudience(kind="characters", character_ids=("ossa",)),
    )
    prepared, (view,) = await coordinator.prepare(
        draft=subset, snapshot=snapshot, principal=rill, context_fingerprint="fresh"
    )
    assert view.character_id == "ossa"
    assert await coordinator.activate(prepared.request_id)
    ossa = ChannelPrincipal("test", "discord-ossa")
    answer = DecisionSubmission(presentation_token=view.presentation_token, selection_id="decline")
    assert (await coordinator.submit(principal=ossa, submission=answer, context_fingerprint="fresh")).status == "completed"
    resolution = (await coordinator.resolutions(prepared.request_id))[0]
    assert isinstance(resolution.directive, NoTestDirective)
    assert resolution.directive.character_id == "ossa"
    current = ConfirmationDraft(
        kind="confirmation", question="Open it?", audience=CurrentAudience(kind="current")
    )
    active, _ = await coordinator.prepare(
        draft=current, snapshot=snapshot, principal=rill, context_fingerprint="fresh"
    )
    assert await coordinator.activate(active.request_id)
    assert await coordinator.cancel(active.request_id, reason="facilitator")
    assert await coordinator.pending_count() == 0


async def test_approach_directives_rebind_to_current_subset_and_party_assignments(tmp_path: Path):
    directory = _campaign(tmp_path)
    snapshot = directory.snapshot()
    rill = ChannelPrincipal("test", "discord-rill")
    ossa = ChannelPrincipal("test", "discord-ossa")

    def draft(audience):
        return ApproachDraft(
            kind="approach",
            question="How do you cross?",
            audience=audience,
            approaches=(
                ApproachOption(
                    id="climb",
                    label="Climb",
                    directive=AttributeTestProposal(kind="attribute_test", attribute="DEX"),
                ),
            ),
        )

    current = DecisionCoordinator()
    prepared, (view,) = await current.prepare(
        draft=draft(CurrentAudience(kind="current")),
        snapshot=snapshot,
        principal=rill,
        context_fingerprint="fresh",
    )
    assert await current.activate(prepared.request_id)
    assert (await current.submit(
        principal=rill,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="climb"),
        context_fingerprint="fresh",
    )).status == "completed"
    current_directive = (await current.resolutions(prepared.request_id))[0].directive
    assert isinstance(current_directive, AttributeTestDirective)
    assert current_directive.character_id == "rill"

    subset = DecisionCoordinator()
    prepared, (view,) = await subset.prepare(
        draft=draft(CharacterAudience(kind="characters", character_ids=("ossa",))),
        snapshot=snapshot,
        principal=rill,
        context_fingerprint="fresh",
    )
    assert await subset.activate(prepared.request_id)
    assert (await subset.submit(
        principal=ossa,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="climb"),
        context_fingerprint="fresh",
    )).status == "completed"
    subset_directive = (await subset.resolutions(prepared.request_id))[0].directive
    assert isinstance(subset_directive, AttributeTestDirective)
    assert subset_directive.character_id == "ossa"

    party = DecisionCoordinator()
    prepared, views = await party.prepare(
        draft=draft({"kind": "party"}),
        snapshot=snapshot,
        principal=rill,
        context_fingerprint="fresh",
    )
    assert await party.activate(prepared.request_id)
    by_character = {view.character_id: view for view in views}
    results = await asyncio.gather(
        party.submit(
            principal=rill,
            submission=DecisionSubmission(
                presentation_token=by_character["rill"].presentation_token,
                selection_id="climb",
            ),
            context_fingerprint="fresh",
        ),
        party.submit(
            principal=ossa,
            submission=DecisionSubmission(
                presentation_token=by_character["ossa"].presentation_token,
                selection_id="climb",
            ),
            context_fingerprint="fresh",
        ),
    )
    assert {result.status for result in results} == {"accepted", "completed"}
    assert {
        resolution.directive.character_id
        for resolution in await party.resolutions(prepared.request_id)
        if isinstance(resolution.directive, AttributeTestDirective)
    } == {"rill", "ossa"}


async def test_custom_validation_stays_pending_and_completed_answer_retries_are_idempotent(tmp_path: Path):
    directory = _campaign(tmp_path)
    snapshot = directory.snapshot()
    coordinator = DecisionCoordinator()
    draft = ClarificationDraft(
        kind="clarification",
        question="What do you do?",
        audience=CurrentAudience(kind="current"),
        paths=(ChoiceOption(id="wait", label="Wait"),),
    )
    rill = ChannelPrincipal("test", "discord-rill")
    prepared, (view,) = await coordinator.prepare(
        draft=draft, snapshot=snapshot, principal=rill, context_fingerprint="fresh"
    )
    assert await coordinator.activate(prepared.request_id)
    for custom in ("", "x" * 601):
        result = await coordinator.submit(
            principal=rill,
            submission=DecisionSubmission(
                presentation_token=view.presentation_token,
                selection_id="own_approach",
                custom_text=custom,
            ),
            context_fingerprint="fresh",
        )
        assert result.status == "invalid"
        assert await coordinator.pending_count() == 1
    answer = DecisionSubmission(presentation_token=view.presentation_token, selection_id="wait")
    assert (await coordinator.submit(principal=rill, submission=answer, context_fingerprint="fresh")).status == "completed"
    assert (await coordinator.submit(principal=rill, submission=answer, context_fingerprint="fresh")).status == "completed"
    conflict = await coordinator.submit(
        principal=rill,
        submission=DecisionSubmission(presentation_token=view.presentation_token, selection_id="own_approach", custom_text="Run"),
        context_fingerprint="fresh",
    )
    assert conflict.status == "conflict"


def test_approach_directives_bind_actor_and_mechanic_and_decline_blocks_tests():
    guard = ResolutionGuard(
        (
            # The engine consumes an authenticated, post-submission binding only.
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option",
                selection_id="climb",
                directive=AttributeTestDirective(
                    kind="attribute_test", character_id="rill", attribute="DEX"
                ),
            ),
        )
    )
    assert guard.validate("attribute_test", {"character_id": "ossa", "attribute": "DEX"})
    assert guard.validate("attribute_test", {"character_id": "rill", "attribute": "STR"})
    assert guard.validate("attribute_test", {"character_id": "rill", "attribute": "DEX"}) is None
    # Passing validate is permission to roll, never the resolution itself: only the
    # recorded ok result satisfies the directive.
    assert guard.unused_error()
    guard.record_tool_result(
        "attribute_test", {"character_id": "rill", "attribute": "DEX"}, {"ok": True}
    )
    assert guard.unused_error() == ""
    decline = ResolutionGuard(
        (
            DecisionResolution(
                character_id="rill", kind="confirmation", answer_kind="decline",
                selection_id="decline",
                directive=NoTestDirective(kind="no_test", character_id="rill"),
            ),
        )
    )
    assert decline.validate("campaign_status", {}) is None
    assert decline.validate("attribute_test", {"character_id": "ossa", "attribute": "DEX"}) is None
    assert decline.validate("group_test", {"character_ids": ["ossa"]}) is None
    assert decline.validate("group_test", {"character_ids": ["rill", "ossa"]})
    assert decline.validate("combat_defend", {"defender_id": "rill", "method": "dodge"})
    defence = ResolutionGuard(
        (
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option",
                selection_id="parry",
                directive=CombatDefendDirective(
                    kind="combat_defend", character_id="rill", method="parry"
                ),
            ),
        )
    )
    assert defence.validate("combat_defend", {"defender_id": "rill", "method": "dodge"})
    assert defence.unused_error()
    assert defence.validate("combat_defend", {"defender_id": "rill", "method": "parry"}) is None
    # The bound parry stays bound while nothing refuses it: the other method is
    # still rejected, and the guard still withholds until a roll actually lands.
    assert defence.validate("combat_defend", {"defender_id": "rill", "method": "dodge"})
    assert defence.unused_error()
    defence.record_tool_result(
        "combat_defend", {"defender_id": "rill", "method": "parry"}, {"ok": True}
    )
    assert defence.unused_error() == ""


def test_a_bound_test_is_satisfied_by_the_same_actors_real_attack():
    """A live turn bound a STR test for "I punch the trader in the face"; the model
    resolved it through combat_attack instead -- audited dice, damage applied, an
    action spent -- and the guard withheld the turn, so the player never saw a roll
    that genuinely happened. The stricter mechanic satisfies the bound one; another
    actor's attack, or a refused one, still does not."""
    def _bound_test() -> ResolutionGuard:
        return ResolutionGuard((
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option",
                selection_id="punch",
                directive=AttributeTestDirective(
                    kind="attribute_test", character_id="rill", attribute="STR"
                ),
            ),
        ))

    guard = _bound_test()
    assert guard.validate("combat_attack", {"attacker_id": "rill"}) is None
    guard.record_tool_result(
        "combat_attack", {"attacker_id": "rill", "target_id": "angry-trader"}, {"ok": True}
    )
    assert guard.unused_error() == ""

    other = _bound_test()
    other.record_tool_result("combat_attack", {"attacker_id": "ossa"}, {"ok": True})
    assert other.unused_error()

    refused = _bound_test()
    refused.record_tool_result(
        "combat_attack", {"attacker_id": "rill"}, {"ok": False, "error": "turn_not_open"}
    )
    assert refused.unused_error()


def test_a_bound_test_is_satisfied_by_the_same_actors_runic_grant():
    """Test a bound test is satisfied by the same actors runic grant.
    """
    def _bound_test() -> ResolutionGuard:
        return ResolutionGuard((
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option",
                selection_id="take_up_the_blade",
                directive=AttributeTestDirective(
                    kind="attribute_test", character_id="rill", attribute="STR"
                ),
            ),
        ))

    guard = _bound_test()
    assert guard.validate("grant_runic_weapon", {"character_id": "rill"}) is None
    guard.record_tool_result(
        "grant_runic_weapon",
        {"character_id": "rill", "name": "Sorrow", "personality": "brutal"},
        {"ok": True},
    )
    assert guard.unused_error() == ""

    other = _bound_test()
    other.record_tool_result(
        "grant_runic_weapon", {"character_id": "ossa", "name": "Sorrow", "personality": "brutal"},
        {"ok": True},
    )
    assert other.unused_error()

    refused = _bound_test()
    refused.record_tool_result(
        "grant_runic_weapon", {"character_id": "rill", "name": "Grief", "personality": "cunning"},
        {"ok": False, "error": "runic_weapon_held"},
    )
    assert refused.unused_error()


def test_a_confirmed_hazard_is_cleared_by_the_same_actors_runic_grant():
    """The confirmation-shaped half of the same live finding: "Rill takes it up"
    confirmed as violence, resolved by the grant's own audited dice."""
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result(
        "grant_runic_weapon",
        {"character_id": "rill", "name": "Sorrow", "personality": "brutal"},
        {"ok": True},
    )
    assert guard.unused_error() == ""

    other = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    other.record_tool_result(
        "grant_runic_weapon", {"character_id": "ossa", "name": "Sorrow", "personality": "brutal"},
        {"ok": True},
    )
    assert other.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_a_refused_bound_defence_never_satisfies_the_guard():
    """Test a refused bound defence never satisfies the guard.
    """
    resolution = DecisionResolution(
        character_id="rill", kind="approach", answer_kind="option",
        selection_id="defend",
        directive=CombatDefendDirective(
            kind="combat_defend", character_id="rill", method="dodge"
        ),
    )
    guard = ResolutionGuard((resolution,))
    arguments = {"defender_id": "rill", "method": "dodge"}
    for _ in range(7):
        assert guard.validate("combat_defend", arguments) is None
        guard.record_tool_result(
            "combat_defend", arguments,
            {"ok": False, "error": "incoming_damage_required"},
        )
    assert guard.unused_error() == "the selected decision's required mechanic was not used"
    # The defence is still resolvable: a later call that actually rolls satisfies it.
    assert guard.validate("combat_defend", arguments) is None
    guard.record_tool_result("combat_defend", arguments, {"ok": True})
    assert guard.unused_error() == ""


def test_an_engine_refused_bound_method_unlocks_the_other_defence():
    """What the old pre-execution marking actually bought, preserved explicitly: a
    bound parry the engine refuses against a ranged attack must not lock out the
    dodge that can resolve -- but the dodge must still ROLL before the turn holds."""
    resolution = DecisionResolution(
        character_id="rill", kind="approach", answer_kind="option",
        selection_id="defend",
        directive=CombatDefendDirective(
            kind="combat_defend", character_id="rill", method="parry"
        ),
    )
    guard = ResolutionGuard((resolution,))
    parry = {"defender_id": "rill", "method": "parry"}
    dodge = {"defender_id": "rill", "method": "dodge"}
    assert guard.validate("combat_defend", dodge)  # not the bound method yet
    assert guard.validate("combat_defend", parry) is None
    guard.record_tool_result(
        "combat_defend", parry, {"ok": False, "error": "parry_against_ranged"}
    )
    assert guard.unused_error()  # the refusal is not the resolution
    assert guard.validate("combat_defend", dodge) is None  # the refusal unlocked it
    guard.record_tool_result("combat_defend", dodge, {"ok": True})
    assert guard.unused_error() == ""


def test_read_only_and_passive_observation_bypass_without_hiding_commitments():
    assert planning_bypass(_policy("What are the traders selling?")) == "factual"
    assert planning_bypass(_policy("Look at the traders")) == "observation"
    assert planning_bypass(_policy("I inspect the room")) == "observation"
    assert planning_bypass(_policy("I listen at the door")) == "observation"
    assert planning_bypass(_policy("I read the trader notice")) == "observation"
    assert planning_bypass(_policy("How do I use this door?")) == "factual"
    assert planning_bypass(_policy("Look at the traders and steal their goods")) == ""
    assert planning_bypass(_policy("I inspect the room, then punch the trader")) == ""
    assert planning_bypass(_policy("I read the trader notice, then stab the barkeep")) == ""


def test_an_unreadable_classification_bypasses_nothing_and_still_confirms():
    """The fail-closed direction, now that both gates read one verdict rather than
    classifying for themselves: a ``None`` policy is the classifier declining to answer,
    and the engine does not get to assume the turn was harmless."""
    assert planning_bypass(None) == ""
    assert requires_risk_confirmation(None) is True


async def test_risk_floor_creates_confirmation_before_a_non_hostile_harmful_action():
    session = ProgressiveDecisionSession.start("Punch a trader")
    proposal = PlanOutcome(plan=ProceedPlan(kind="proceed"))

    assert requires_risk_confirmation(_policy("Punch a trader"))
    result = await verify_plan(
        proposal, "Punch a trader", session, classify=fake_classify_intent
    )
    assert isinstance(result.plan, RequestDecisionPlan)
    assert result.plan.decision.kind == "confirmation"


@pytest.mark.parametrize(
    "action",
    (
        "Hurl a trader into the canal",
        "Pilfer the merchant's purse",
        "Pulverize the town gate",
        "Seal the door permanently",
        "I stab the barkeep",
        "I assault the librarian",
    ),
)
def test_risk_floor_covers_custom_wording_and_irreversible_actions(action):
    assert requires_risk_confirmation(_policy(action))


async def test_risk_floor_uses_the_custom_effective_action_and_respects_its_answer():
    clarification = ClarificationDraft(
        kind="clarification",
        question="What do you do?",
        audience=CurrentAudience(kind="current"),
        paths=(ChoiceOption(id="wait", label="Wait"),),
    )
    session = ProgressiveDecisionSession.start("I consider my options")
    session = session.preparing(clarification).accept(
        clarification,
        (
            DecisionResolution(
                character_id="rill",
                kind="clarification",
                answer_kind="custom",
                selection_id="own_approach",
                custom_text="Steal from a merchant",
            ),
        ),
    )
    # The turn's own verdict is threaded in, and is deliberately the harmless one for
    # "ignored": the effective action differs from the declaration, so ``verify_plan``
    # must classify "Steal from a merchant" itself rather than reuse what it was given.
    forced = await verify_plan(
        PlanOutcome(plan=ProceedPlan(kind="proceed")),
        "ignored",
        session,
        classify=fake_classify_intent,
        policy=_policy("ignored"),
    )
    assert isinstance(forced.plan, RequestDecisionPlan)
    assert forced.plan.decision.kind == "confirmation"

    confirmation = risk_confirmation()
    confirmed = ProgressiveDecisionSession.start("Punch a trader").preparing(confirmation).accept(
        confirmation,
        (
            DecisionResolution(
                character_id="rill",
                kind="confirmation",
                answer_kind="confirm",
                selection_id="confirm",
            ),
        ),
    )
    assert isinstance(
        (
            await verify_plan(
                PlanOutcome(plan=ProceedPlan(kind="proceed")),
                "Punch a trader",
                confirmed,
                classify=fake_classify_intent,
            )
        ).plan,
        ProceedPlan,
    )
    declined = ProgressiveDecisionSession.start("Punch a trader").preparing(confirmation).accept(
        confirmation,
        (
            DecisionResolution(
                character_id="rill",
                kind="confirmation",
                answer_kind="decline",
                selection_id="decline",
            ),
        ),
    )
    assert isinstance(
        await verify_plan(
            PlanOutcome(plan=ProceedPlan(kind="proceed")),
            "Punch a trader",
            declined,
            classify=fake_classify_intent,
        ),
        DeclinedPlan,
    )


async def test_progressive_session_closes_target_and_invalidates_dependents_after_a_custom_action():
    target = ClarificationDraft(
        kind="clarification",
        dimension="target",
        question="Which target?",
        audience=CurrentAudience(kind="current"),
        paths=(ChoiceOption(id="random", label="Choose at random"),),
    )
    custom = ClarificationDraft(
        kind="clarification",
        question="What is the approach?",
        audience=CurrentAudience(kind="current"),
        paths=(ChoiceOption(id="wait", label="Wait"),),
    )
    session = ProgressiveDecisionSession.start("Act")
    session = session.preparing(target).accept(
        target,
        (DecisionResolution(character_id="rill", kind="clarification", answer_kind="option", selection_id="random"),),
    )
    assert "target" in session.closed_dimensions


    repeated = PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=target))
    degraded = await verify_plan(repeated, "Act", session, classify=fake_classify_intent)
    assert isinstance(degraded, PlanOutcome)
    assert isinstance(degraded.plan, ProceedPlan)
    session = session.preparing(custom).accept(
        custom,
        (DecisionResolution(character_id="rill", kind="clarification", answer_kind="custom", selection_id="own_approach", custom_text="Try a different route"),),
    )
    assert "target" not in session.closed_dimensions
    assert session.effective_action == "Try a different route"


def test_confirmation_requires_the_exact_current_action_fingerprint():
    confirmation = ConfirmationDraft(
        kind="confirmation", question="Confirm?", audience=CurrentAudience(kind="current")
    )
    target = ClarificationDraft(
        kind="clarification",
        dimension="target",
        question="Which target?",
        audience=CurrentAudience(kind="current"),
        paths=(ChoiceOption(id="one", label="One"),),
    )
    session = ProgressiveDecisionSession.start("Act").preparing(confirmation)
    session = session.accept(
        target,
        (DecisionResolution(character_id="rill", kind="clarification", answer_kind="option", selection_id="one"),),
    )
    with pytest.raises(PlannerPolicyError):
        session.accept(
            confirmation,
            (DecisionResolution(character_id="rill", kind="confirmation", answer_kind="confirm", selection_id="confirm"),),
        )


def test_planner_prompt_and_proposals_exclude_authenticated_actor_bindings():
    resolution = DecisionResolution(
        character_id="rill",
        kind="approach",
        answer_kind="option",
        selection_id="climb",
        action_fingerprint="a" * 64,
        directive=AttributeTestDirective(kind="attribute_test", character_id="rill", attribute="DEX"),
    )
    prompt = planning_prompt(
        "Act", "Canon", (("rill", "Rill"),), (resolution,), closed_dimensions=("target",)
    )
    assert "rill" not in prompt
    assert "a" * 64 not in prompt
    with pytest.raises(ValidationError):
        AttributeTestProposal.model_validate(
            {"kind": "attribute_test", "attribute": "DEX", "character_id": "rill"}
        )


def test_proposed_directive_rejects_the_other_kinds_own_field_structurally():
    """This pins the structural half directly: the exact shape the live probe
    produced now fails for a different, correct reason (an unexpected field on
    the wrong shape, caught by ``extra=\"forbid\"``), not the old cross-field
    validator -- and each proposal type cannot even be constructed with the
    other kind's field, in Python, not just rejected after the fact.
    """
    with pytest.raises(ValidationError):
        AttributeTestProposal.model_validate({"kind": "attribute_test", "attribute": "WIS", "method": "dodge"})
    with pytest.raises(ValidationError):
        AttributeTestProposal(kind="attribute_test", attribute="WIS", method="dodge")
    with pytest.raises(ValidationError):
        CombatDefendProposal.model_validate({"kind": "combat_defend", "method": "dodge", "attribute": "STR"})
    with pytest.raises(ValidationError):
        NoTestProposal.model_validate({"kind": "no_test", "attribute": "STR"})

    # Each shape still constructs cleanly with only its own field.
    assert AttributeTestProposal(kind="attribute_test", attribute="WIS").attribute == "WIS"
    assert CombatDefendProposal(kind="combat_defend", method="dodge").method == "dodge"
    assert NoTestProposal(kind="no_test").kind == "no_test"

    # The discriminated union resolves each shape correctly through the field
    # ApproachOption actually declares (ProposedDirective), the same path a
    # live structured-output call populates.
    option = ApproachOption.model_validate(
        {"id": "climb", "label": "Climb", "directive": {"kind": "attribute_test", "attribute": "DEX"}}
    )
    assert isinstance(option.directive, AttributeTestProposal)


# -- slice 5: the combat-aware risk floor -----------------------------------


@pytest.mark.parametrize("action", ("kick orso", "I knee the guard", "elbow the sailor", "headbutt him", "stomp the acolyte"))
def test_the_risk_floor_catches_violence_verbs_absent_from_the_old_lexicon(action):
    """The old list certified itself; these verbs prove the floor against novel phrasing."""
    assert requires_risk_confirmation(_policy(action))


def test_combat_sanctions_violence_only_for_the_declared_enemy():
    fight = _fight("orso-pell")
    assert combat_sanctions_violence(policy_for("kick orso pell", combat=fight), fight) is True  # named combatant
    assert combat_sanctions_violence(policy_for("stab orso", combat=fight), fight) is True  # partial name, sole enemy
    assert combat_sanctions_violence(policy_for("stab him", combat=fight), fight) is True  # pronoun, sole enemy
    assert combat_sanctions_violence(policy_for("attack", combat=fight), fight) is True  # bare, sole enemy
    two = _fight("orso-pell", "reed-thug")
    assert combat_sanctions_violence(policy_for("stab reed thug", combat=two), two) is True  # named among several
    # Which enemy an unspecific declaration strikes is target selection inside a
    # fight the table already consented to, not a fresh consent question; the combat
    # tools resolve it against a rostered combatant or refuse.
    assert combat_sanctions_violence(policy_for("stab someone", combat=two), two) is True
    assert combat_sanctions_violence(policy_for("attack", combat=two), two) is True
    assert combat_sanctions_violence(policy_for("kick orso pell", combat=None), None) is False  # no combat
    assert combat_sanctions_violence(policy_for("kick orso", combat=_fight("orso-pell", active=False)), _fight("orso-pell", active=False)) is False


def test_combat_never_sanctions_violence_against_a_party_member():
    """The bypass must never drop the confirmation for attacking an ally."""
    fight = _fight("orso-pell")  # order is (rill pc, orso-pell npc)
    assert combat_sanctions_violence(policy_for("attack rill", combat=fight), fight) is False
    assert combat_sanctions_violence(policy_for("stab rill the healer", combat=fight), fight) is False


def test_combat_withholds_the_sanction_only_on_recorded_persons_outside_the_fight():
    """Consent lives in typed combat state, and so does its withholding evidence.
    """
    from narrator.policy_types import TrustedScope

    fight = _fight("orso-pell")
    scope = TrustedScope(
        "session-1",
        "market",
        ("orso-pell", "fishwife"),
        ("rill",),
        (("orso-pell", "alive"), ("fishwife", "alive")),
    )
    assert combat_sanctions_violence(policy_for("kill the fishwife next to orso pell", scope=scope, combat=fight), fight) is False
    assert combat_sanctions_violence(policy_for("stab the fishwife", scope=scope, combat=fight), fight) is False
    assert combat_sanctions_violence(policy_for("kick orso pell", scope=scope, combat=fight), fight) is True
    # Unrecorded role nouns sanction; the tool boundary owns their protection.
    assert combat_sanctions_violence(policy_for("stab the bystander", scope=scope, combat=fight), fight) is True
    assert combat_sanctions_violence(policy_for("stab the innocent hostage", scope=scope, combat=fight), fight) is True


async def test_the_risk_floor_bypasses_confirmation_for_an_open_fight_combatant():
    session = ProgressiveDecisionSession.start("stab orso pell")
    proceed = PlanOutcome(plan=ProceedPlan(kind="proceed"))
    # With no combat, the floor demands confirmation.
    assert isinstance(
        (
            await verify_plan(
                proceed, "stab orso pell", session, classify=fake_classify_intent
            )
        ).plan,
        RequestDecisionPlan,
    )
    # With Orso a recorded combatant, the floor proceeds without re-confirming.
    result = await verify_plan(
        proceed,
        "stab orso pell",
        session,
        combat=_fight("orso-pell"),
        classify=fake_classify_intent,
    )
    assert isinstance(result.plan, ProceedPlan)


async def test_the_risk_floor_still_confirms_violence_against_a_non_combatant():
    """An open fight does not sanction attacking a recorded person outside its roster."""
    from narrator.policy_types import TrustedScope

    session = ProgressiveDecisionSession.start("stab the quay warden")
    result = await verify_plan(
        PlanOutcome(plan=ProceedPlan(kind="proceed")),
        "stab the quay warden",
        session,
        combat=_fight("orso-pell", "reed-thug"),
        classify=fake_classify_intent,
        scope=TrustedScope(
            "session-1",
            "market",
            ("orso-pell", "reed-thug", "quay-warden"),
            ("rill",),
            (("quay-warden", "alive"),),
        ),
    )
    assert isinstance(result.plan, RequestDecisionPlan)


# -- slice 5: confirmed-hazard binding ---------------------------------------


def _confirm(draft, character_id="rill"):
    return resolution_from_submission(
        draft, character_id, DecisionSubmission(presentation_token="x" * 16, selection_id="confirm")
    )


def test_a_confirmed_violence_action_yields_a_hazard_directive():
    directive = _confirm(risk_confirmation("violence")).directive
    assert isinstance(directive, HazardResolutionDirective)
    assert directive.character_id == "rill" and directive.category == "violence"


def test_a_confirmed_non_violence_hazard_binds_nothing():
    assert risk_confirmation("theft").binds_hazard == ""
    assert _confirm(risk_confirmation("theft")).directive is None
    assert _confirm(risk_confirmation()).directive is None  # a trade/planner confirmation


def test_a_declined_confirmation_still_forbids_a_test():
    resolution = resolution_from_submission(
        risk_confirmation("violence"),
        "rill",
        DecisionSubmission(presentation_token="x" * 16, selection_id="decline"),
    )
    assert resolution.directive.kind == "no_test"


def test_the_guard_withholds_a_confirmed_attack_that_never_rolled():
    resolution = _confirm(risk_confirmation("violence"))
    guard = ResolutionGuard((resolution,))
    # Setup tools on the way to the attack are never refused.
    for tool in ("combat_start", "combat_begin_turn", "npc_create", "combat_move"):
        assert guard.validate(tool, {"attacker_id": "rill"}) is None
    # No resolving roll ran, so the turn is withheld.
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_the_guard_clears_a_confirmed_attack_once_the_actor_rolls():
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result("combat_attack", {"attacker_id": "rill"}, {"ok": True})
    assert guard.unused_error() == ""


def test_a_refused_attack_does_not_clear_the_confirmed_hazard():
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result(
        "combat_attack", {"attacker_id": "rill"}, {"ok": False, "error": "target_out_of_reach"}
    )
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_a_combat_start_that_gives_the_enemy_the_first_turn_clears_the_hazard():
    """Test a combat start that gives the enemy the first turn clears the hazard.
    """
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result(
        "combat_start",
        {"pc_ids": ["rill"], "npc_ids": ["rade"]},
        {"ok": True, "active_actor": "rade"},
    )
    assert guard.unused_error() == ""


def test_a_combat_start_where_the_actor_acts_first_does_not_clear_the_hazard():
    """When the actor holds the first turn the attack CAN roll now, so the guard
    still demands it."""
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result(
        "combat_start",
        {"pc_ids": ["rill"], "npc_ids": ["rade"]},
        {"ok": True, "active_actor": "rill"},
    )
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_a_failed_combat_start_does_not_clear_the_hazard():
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result(
        "combat_start",
        {"pc_ids": ["rill"], "npc_ids": ["rade"]},
        {"ok": False, "error": "combat_already_active"},
    )
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_a_group_test_or_attribute_test_for_the_actor_also_clears_the_hazard():
    guard = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    guard.record_tool_result("attribute_test", {"character_id": "rill"}, {"ok": True})
    assert guard.unused_error() == ""
    other = ResolutionGuard((_confirm(risk_confirmation("violence")),))
    other.record_tool_result("group_test", {"character_ids": ["rill", "ossa"]}, {"ok": True})
    assert other.unused_error() == ""


def test_hazard_obligation_text_states_the_roll_for_a_confirmed_attack():
    text = hazard_obligation_text((_confirm(risk_confirmation("violence")),))
    assert "rill" in text and "combat_attack" in text
    assert hazard_obligation_text((_confirm(risk_confirmation("theft")),)) == ""


# -- slice 5: clarification refinement ---------------------------------------


def _category_clarification():
    # Labels route to the planner (no read, social, or risk verb) and stay ambiguous, so
    # the refined action still reaches the closed-dimension check in verify_plan.
    return ClarificationDraft(
        kind="clarification",
        question="How do you proceed?",
        audience=CurrentAudience(kind="current"),
        paths=(
            ChoiceOption(id="rummage", label="Rummage through the merchant's crates"),
            ChoiceOption(id="signal", label="Signal the ferryman across the water"),
        ),
    )


def _refine(session, draft, selection_id="rummage"):
    return session.preparing(draft).accept(
        draft,
        (
            DecisionResolution(
                character_id="rill",
                kind="clarification",
                answer_kind="option",
                selection_id=selection_id,
            ),
        ),
    )


def test_a_clarification_option_refines_the_effective_action_and_reopens_its_dimension():
    draft = _category_clarification()
    session = ProgressiveDecisionSession.start("just mucking around")
    before = session.action_fingerprint
    session = _refine(session, draft)
    # The action refined to the chosen path's label, and its fingerprint changed.
    assert session.effective_action == "Rummage through the merchant's crates"
    assert session.action_fingerprint != before
    # The intent dimension reopened, so a narrowing follow-up is legal rather than a fault.
    assert "intent" not in session.closed_dimensions


async def test_a_refined_action_permits_a_narrowing_follow_up_without_a_fault():
    draft = _category_clarification()
    session = _refine(ProgressiveDecisionSession.start("do something at the market"), draft)
    follow_up = _category_clarification()
    # The refined action routes to the planner and its intent dimension is open, so the
    # closed-dimension rule permits the narrowing follow-up instead of raising.
    result = await verify_plan(
        PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=follow_up)),
        "ignored",
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(result.plan, RequestDecisionPlan)


async def test_a_planner_repeating_a_closed_dimension_on_an_unchanged_action_degrades_to_proceed():
    """A planner repeating a closed dimension on an unchanged action degrades to proceed. Synthetic fixtures exercise this contract."""
    def _approach():
        return ApproachDraft(
            kind="approach",
            question="Which approach?",
            audience=CurrentAudience(kind="current"),
            approaches=(
                ApproachOption(
                    id="quiet",
                    label="Slip past unseen",
                    directive=NoTestProposal(kind="no_test"),
                ),
            ),
        )

    session = ProgressiveDecisionSession.start("get past the sentry").preparing(_approach()).accept(
        _approach(),
        (
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option", selection_id="quiet"
            ),
        ),
    )
    assert "approach" in session.closed_dimensions  # an approach answer does not refine
    outcome = await verify_plan(
        PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=_approach())),
        "get past the sentry",
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, ProceedPlan)


def test_read_combat_snapshot_reads_the_none_sentinel_as_absence(tmp_path: Path):
    from narrator.interactions import read_combat_snapshot

    root = tmp_path / "camp"
    (root / "campaign").mkdir(parents=True)
    (root / "campaign" / "scene.md").write_text(
        "## Combat\n\n- round: 2\n- active actor: none\n- order: none\n"
        "- combatant: reed-thug side=npc range=close actions=1/1\n",
        encoding="utf-8",
    )
    snapshot = read_combat_snapshot(root)
    assert snapshot.active is True
    assert snapshot.active_actor == ""  # "none" is absence, not an actor
    assert snapshot.order == ()  # "none" is not a combatant id
    assert snapshot.npc_combatants == ("reed-thug",)


def test_defence_method_binds_only_a_lone_dodge_or_parry():
    """The two SRD literals bind a defence; every ambiguity falls through to routing.
    """
    fight = _fight("reed-thug")
    assert defence_method("dodge", fight) == "dodge"
    assert defence_method("Parry!", fight) == "parry"
    assert defence_method("I'll try to dodge", fight) == "dodge"
    assert defence_method("i parry it", fight) == "parry"
    assert defence_method("dodge or parry", fight) == ""  # both named is no selection
    assert defence_method("I hold my ground", fight) == ""  # neither named
    assert defence_method("dodge", None) == ""  # no fight to defend in
    assert defence_method("dodge", _fight("reed-thug", active=False)) == ""


def test_a_defence_word_inside_a_larger_declaration_binds_nothing():
    """The residual test is what keeps a violent second action on the risk floor's path."""
    fight = _fight("reed-thug")
    scope = TrustedScope("session-one", "vey-docks", ())
    for declaration in ("I dodge his swing and stab the barkeep", "parry then kill the priest"):
        assert defence_method(declaration, fight) == ""
        # The declaration the binding refuses is one the classifier reads as violence, so
        # the risk floor keeps it. Neither half of that pairing is incidental.
        assert classify_turn(declaration, scope=scope).route == "risk"
    # A non-violent residual withholds the binding too, and keeps its planner route.
    assert defence_method("dodge the reed thug", fight) == ""
    assert classify_turn("dodge the reed thug", scope=scope).route == "planner"


def test_the_route_gate_is_redundant_against_the_current_lexicon():
    """Name which mechanism holds the violence gate, and keep the claim from going stale.

    ``NarratorService._decision_phase`` runs the defence binding only on the planner
    route. That placement reads like the thing keeping a violent compound out, and it is
    not: ``defence_method``'s residual subtraction rejects every such declaration first.
    This enumeration asserts the redundancy rather than describing it, so a later change
    to either word set fails here instead of quietly promoting the route test to load
    bearing. The failure is the signal, not a defect in itself.
    """
    from itertools import combinations

    from lexical_double import _DEFENCE_INTENT_WORDS, _TARGETLESS_WORDS

    fight = _fight("reed-thug")
    scope = TrustedScope("session-one", "vey-docks", ())
    filler = sorted(_TARGETLESS_WORDS | _DEFENCE_INTENT_WORDS)
    admitted_then_rerouted = [
        declaration
        for method in sorted(_DEFENCE_METHODS_UNDER_TEST)
        for size in (0, 1, 2)
        for words in combinations(filler, size)
        if defence_method(declaration := " ".join((*words, method)), fight)
        and classify_turn(declaration, scope=scope).route != "planner"
    ]
    assert admitted_then_rerouted == []


#: The two defence literals, restated here so the enumeration above does not import a
#: private set whose name may change without changing the Standard Reference Document.
_DEFENCE_METHODS_UNDER_TEST = ("dodge", "parry")


def test_a_bound_defence_states_its_obligation_and_the_guard_holds_it():
    """The narrator reads the owed roll in words, so withholding stays the backstop."""
    resolution = DecisionResolution(
        character_id="rill",
        kind="approach",
        answer_kind="option",
        selection_id="defend",
        directive=CombatDefendDirective(
            kind="combat_defend", character_id="rill", method="dodge"
        ),
    )
    obligation = defence_obligation_text((resolution,))
    assert "rill: dodge" in obligation
    assert "combat_defend" in obligation
    assert defence_obligation_text(()) == ""
    assert defence_obligation_text(
        (
            DecisionResolution(
                character_id="rill", kind="approach", answer_kind="option", selection_id="quiet"
            ),
        )
    ) == ""  # a resolution carrying no defence directive owes no defence

    guard = ResolutionGuard((resolution,))
    assert guard.unused_error()  # nothing rolled yet, so the turn would withhold
    assert guard.validate("combat_defend", {"defender_id": "rill", "method": "dodge"}) is None
    assert guard.unused_error()  # allowed to roll is not yet rolled
    guard.record_tool_result(
        "combat_defend", {"defender_id": "rill", "method": "dodge"}, {"ok": True}
    )
    assert guard.unused_error() == ""
    mismatched = ResolutionGuard((resolution,))
    assert mismatched.validate("combat_defend", {"defender_id": "rill", "method": "parry"})
    assert mismatched.validate("combat_defend", {"defender_id": "orso-pell", "method": "dodge"})


def _asked_labels(session):
    return [(item.question, list(item.labels)) for item in session.asked_questions]


def test_an_answered_clarification_is_remembered_with_its_question_and_its_labels():
    """The state the planner prompt and the duplicate guard both read.

    A question is recorded when it is answered, never when it is merely planned: a draft
    the view budget refused, or one a fault closed, was never put to the player and must
    stay askable. ``own_approach`` is not a label the planner offered, so it is excluded.
    """
    draft = _category_clarification()
    session = ProgressiveDecisionSession.start("do something at the market")
    assert session.asked_questions == ()

    session = _refine(session, draft)

    assert len(session.asked_questions) == 1
    remembered = session.asked_questions[0]
    assert remembered.question == draft.question
    assert list(remembered.labels) == [option.label for option in draft.paths]
    assert "Describe your own approach" not in remembered.labels


def test_the_question_history_survives_a_segment_and_clears_on_a_revision():
    """The two halves that together bound a clarification chain.

    ``next_segment`` resets the view budget, because a segment boundary is the player
    asking for another round. Before this slice it reset everything the loop could have
    used to notice a repeat as well, so the same question could be re-asked for as many
    segments as the player was willing to type into -- eight times, in the live
    transcript. ``revised`` still clears the history, because a revised declaration is a
    different action and its questions are fair to ask again.
    """
    session = _refine(
        ProgressiveDecisionSession.start("do something at the market"),
        _category_clarification(),
    )
    asked = _asked_labels(session)
    assert asked

    resumed = session.next_segment()
    assert resumed.completed_views == 0
    assert _asked_labels(resumed) == asked

    revised = resumed.revised("I go and find the bell-keeper instead")
    assert revised.asked_questions == ()


def test_a_confirmation_is_never_recorded_as_an_asked_question():
    """Recording one would let the duplicate guard wave a later hazard through.

    ``risk_confirmation`` authors one fixed wording for every unconfirmed hazardous
    action, so a confirmation recorded here would make the *next* genuine hazard read as
    a repeat -- and ``narrator.service.repeats_answered_question`` would then refuse the
    view and proceed with no confirmation at all. ``action_is_confirmed`` and
    ``action_is_declined`` already bound how often one confirmation can be asked.
    """
    from narrator.decisions import remember_question, risk_confirmation

    assert remember_question((), risk_confirmation("violence")) == ()

    draft = risk_confirmation("violence")
    session = ProgressiveDecisionSession.start('I stab the choir listener now')
    session = session.preparing(draft)
    session = session.accept(
        draft,
        (
            DecisionResolution(
                character_id="rill", kind="confirmation", answer_kind="confirm",
                selection_id="confirm",
            ),
        ),
    )
    assert session.asked_questions == ()


def test_a_decline_does_not_survive_a_segment_boundary():
    """Test a decline does not survive a segment boundary.
    """
    from narrator.decisions import risk_confirmation

    draft = risk_confirmation("violence")
    session = ProgressiveDecisionSession.start('I stab the choir listener now')
    session = session.preparing(draft)
    session = session.accept(
        draft,
        (
            DecisionResolution(
                character_id="rill", kind="confirmation", answer_kind="decline",
                selection_id="decline",
            ),
        ),
    )
    assert session.action_is_declined

    session = session.next_segment()
    assert session.action_is_declined is False

    session = session.preparing(draft)
    session = session.accept(
        draft,
        (
            DecisionResolution(
                character_id="rill", kind="confirmation", answer_kind="confirm",
                selection_id="confirm",
            ),
        ),
    )
    assert session.action_is_confirmed
    assert session.action_is_declined is False


def test_the_planner_prompt_carries_the_questions_and_labels_already_answered():
    """``prior_resolutions`` excludes ``directive``, ``character_id``, and
    ``action_fingerprint``, so what a planner saw of an answered round was a bare
    ``selection_id`` it had authored itself one call earlier. It never saw its own
    question or its own labels, which is why it could re-ask them.
    """
    from narrator.decisions import AskedQuestion, planning_prompt

    asked = (
        AskedQuestion(
            question='Which method will you use for the finishing strike?',
            labels=("Drive the knife in", "Throw the knife"),
        ),
    )
    prompt = planning_prompt(
        "finish it off", "canon", (), (), effective_action="finish it off",
        prior_questions=asked,
    )

    assert 'Which method will you use for the finishing strike?' in prompt
    assert "Drive the knife in" in prompt
    assert "Throw the knife" in prompt
    assert "Never ask any of those questions again" in prompt


def test_the_planner_prompt_tells_the_planner_a_reported_roll_does_not_repeat():
    """Test the planner prompt tells the planner a reported roll does not repeat.
    """
    from narrator.decisions import planning_prompt

    prompt = planning_prompt(
        "I take the cleaver",
        "canon",
        (),
        (),
        recent_narration=(
            "Rill rolls STR: 13 vs 12, failure. The metal screeches against the wood."
        ),
    )
    assert "does not repeat" in prompt


def test_the_planner_prompt_tells_the_planner_to_skip_flavour_only_approach_paths():
    """Test the planner prompt tells the planner to skip flavour only approach paths.
    """
    from narrator.decisions import planning_prompt

    prompt = planning_prompt("I dodge", "canon", (), ())
    assert "themselves differ in what they bind" in prompt
    assert "flavour, not a decision" in prompt

    without_narration = planning_prompt("I take the cleaver", "canon", (), ())
    assert "does not repeat" not in without_narration

    # A call with nothing answered yet renders none of it: no heading, no empty list,
    # no instruction, not even a blank line. This is a guard, not a formatting
    # preference. An earlier version emitted the block unconditionally, and a prompt
    # capture against the disengage scenario measured the consequence: the planner
    # prompt grew by 231 characters on the first call of every turn in every scenario,
    # closing with "Return proceed unless a genuinely new decision is required" -- a
    # live nudge that applied whether or not the list above it held anything. Composing
    # both arms' prompts for the same turn now yields byte-identical text, so a planner
    # call that has no answered question to report cannot be perturbed by this feature
    # at all.
    first = planning_prompt("finish it off", "canon", (), ())
    assert "Questions already asked and answered in this chain" not in first
    assert "Never ask any of those questions again" not in first
    # The section is the only difference between the two, so dropping it must leave the
    # prompt exactly as it is with an explicitly empty history.
    assert first == planning_prompt("finish it off", "canon", (), (), prior_questions=())


def test_the_planner_prompt_carries_the_tracked_interlocutor():
    """The planner prompt carries the tracked interlocutor. Synthetic fixtures exercise this contract."""
    from narrator.decisions import planning_prompt

    generic = planning_prompt(
        'would you accept three coins instead?', "canon", (), (),
        interaction_cue=InteractionCue("scene_interlocutor"),
    )
    assert "Active in-world interlocutor" in generic
    assert "kind: scene_interlocutor" in generic
    assert "public_npc_id: none" in generic
    assert "already engaged with this interlocutor" in generic

    named = planning_prompt(
        'would you accept three coins instead?', "canon", (), (),
        interaction_cue=InteractionCue("canonical_npc", "rade"),
    )
    assert "kind: canonical_npc" in named
    assert "public_npc_id: rade" in named

    # No tracked focus renders no block at all -- byte-identical to the prompt that
    # shipped before this fix, the same discipline the prior-questions block above
    # already documents.
    unset = planning_prompt('would you accept three coins instead?', "canon", (), ())
    assert "Active in-world interlocutor" not in unset
    assert unset == planning_prompt(
        'would you accept three coins instead?', "canon", (), (), interaction_cue=None,
    )


def test_the_question_history_stays_bounded_for_the_prompt():
    """This list enters a model prompt, so an unbounded chain must not grow it."""
    from narrator.decisions import MAX_ASKED_QUESTIONS, AskedQuestion, remember_question

    asked: tuple[AskedQuestion, ...] = ()
    for index in range(MAX_ASKED_QUESTIONS + 5):
        asked = remember_question(
            asked,
            ClarificationDraft(
                kind="clarification",
                question=f"Narrowing question number {index}?",
                audience=CurrentAudience(kind="current"),
                paths=(ChoiceOption(id=f"p{index}", label=f"Path {index}"),),
            ),
        )

    assert len(asked) == MAX_ASKED_QUESTIONS
    # The most recent survive: a planner that just re-asked its own previous question is
    # the shape this exists to catch.
    assert asked[-1].question == f"Narrowing question number {MAX_ASKED_QUESTIONS + 4}?"


def test_the_duplicate_guard_refuses_a_re_ask_and_never_a_confirmation():
    """Test the duplicate guard refuses a re ask and never a confirmation.
    """
    from narrator.decisions import risk_confirmation
    from narrator.service import repeats_answered_question

    session = _refine(
        ProgressiveDecisionSession.start("do something at the market"),
        _category_clarification(),
    )
    answered = session.asked_questions[0].question


    repeat = ClarificationDraft(
        kind="clarification",
        question=answered,
        audience=CurrentAudience(kind="current"),
        paths=(
            ChoiceOption(id="second", label="A different second label"),
            ChoiceOption(id="first", label="A different first label"),
        ),
    )
    assert repeats_answered_question(session, repeat) is True

    # Punctuation and case drift is still the same question.
    drifted = repeat.model_copy(update={"question": answered.upper().replace(",", " --")})
    assert repeats_answered_question(session, drifted) is True

    # A genuinely different narrowing question is not a repeat.
    fresh = repeat.model_copy(
        update={"question": "Do you want to do that quietly or quickly?"}
    )
    assert repeats_answered_question(session, fresh) is False

    # A confirmation is never a repeat, whatever the history holds.
    assert repeats_answered_question(session, risk_confirmation("violence")) is False


def test_m13_the_confirmation_names_the_hazard_it_actually_detected():
    """One sentence covered all three categories before, and it asserted \"a non-hostile
    person\" for every one -- so forcing a lock and burning a rope both asked the table to
    approve harming a person who was never involved.
    """
    violence = risk_confirmation("violence").question
    theft = risk_confirmation("theft").question
    destructive = risk_confirmation("destructive").question
    assert len({violence, theft, destructive}) == 3
    assert "wound or kill" in violence
    assert "property" in theft
    assert "destroy" in destructive
    # Only violence asserts a person, and only violence binds the resolving mechanic.
    assert "non-hostile person" not in theft
    assert "non-hostile person" not in destructive
    assert risk_confirmation("violence").binds_hazard == "violence"
    assert risk_confirmation("theft").binds_hazard == ""
    assert risk_confirmation("destructive").binds_hazard == ""
    # An uncategorized hazard keeps a sentence that names every consequence.
    assert risk_confirmation().question != violence


async def test_m13_the_floor_raises_no_confirmation_over_a_recorded_corpse():
    """Test m13 the floor raises no confirmation over a recorded corpse.
    """
    corpse = TrustedScope(
        "session-1", "market", ("sera-vane",), (), (("sera-vane", "dead"),)
    )
    living = TrustedScope(
        "session-1", "market", ("sera-vane",), (), (("sera-vane", "alive"),)
    )
    proceed = PlanOutcome(plan=ProceedPlan(kind="proceed"))

    session = ProgressiveDecisionSession.start("loot sera vane")
    assert isinstance(
        (
            await verify_plan(
                proceed,
                "loot sera vane",
                session,
                scope=corpse,
                classify=fake_classify_intent,
            )
        ).plan,
        ProceedPlan,
    )
    # The same declaration against the same NPC while she is alive still confirms.
    living_session = ProgressiveDecisionSession.start("loot sera vane")
    assert isinstance(
        (
            await verify_plan(
                proceed,
                "loot sera vane",
                living_session,
                scope=living,
                classify=fake_classify_intent,
            )
        ).plan,
        RequestDecisionPlan,
    )
    # A living bystander named alongside the corpse still confirms: the corpse accounts
    # for itself and for nobody else.
    both = TrustedScope(
        "session-1", "market", ("sera-vane", "orso-pell"), (),
        (("sera-vane", "dead"), ("orso-pell", "alive")),
    )
    pair_session = ProgressiveDecisionSession.start("stab sera vane and orso pell")
    assert isinstance(
        (
            await verify_plan(
                proceed,
                "stab sera vane and orso pell",
                pair_session,
                scope=both,
                classify=fake_classify_intent,
            )
        ).plan,
        RequestDecisionPlan,
    )


def test_m13_the_hazard_call_sites_accept_and_use_a_real_scope():
    """The scope reaches these decisions one step earlier than it used to. ``planning_bypass``
    and ``requires_risk_confirmation`` no longer classify anything themselves, so the scene
    is now what the *classification* is made against, and the verdict they read carries it.
    The claim under test is unchanged: a hazard decision is made against the real scene,
    never the empty placeholder, and a populated scope changes the answer.

    This still does not prove production supplies one -- an audit (``AUD-2``) rejected a
    first candidate where these same assertions passed while both engine call sites still
    passed nothing, so ``hazard_scope(None, None)`` handed them back the very placeholder
    the criterion names. ``test_m13_the_engine_supplies_a_real_scope_at_every_hazard_call_site``
    is the guard that covers the production call shape.
    """
    scope = TrustedScope("session-1", "market", ("orso-pell",), (), (("orso-pell", "alive"),))
    # A present NPC named as a theft target is a person signal the empty scope never saw.
    assert requires_risk_confirmation(
        _policy("take the ledger from orso pell", scope=scope)
    ) is True
    assert requires_risk_confirmation(_policy("take the ledger from orso pell")) is False
    assert planning_bypass(_policy("look at orso pell", scope=scope)) == "observation"
    # An open fight alone supplies a populated scope for a caller holding no scope, so a
    # hazard decision is never made against the empty placeholder.
    assert hazard_scope(None, _fight("orso-pell")).present_npc_ids == ("orso-pell",)
    assert hazard_scope(scope, _fight("reed-thug")) is scope
    assert hazard_scope(None, None).present_npc_ids == ()


async def test_m13_verify_plan_threads_the_scope_into_the_fight_sanction():
    """Test m13 verify plan threads the scope into the fight sanction.
    """
    from narrator.policy_types import TrustedScope

    proceed = PlanOutcome(plan=ProceedPlan(kind="proceed"))
    declaration = "stab orso pell with my dagger"
    session = ProgressiveDecisionSession.start(declaration)
    result = await verify_plan(
        proceed,
        declaration,
        session,
        combat=_fight("orso-pell"),
        classify=fake_classify_intent,
    )
    assert isinstance(result.plan, ProceedPlan)
    # The same weapon phrasing against a recorded person the fight never rostered
    # still confirms: the scope's present-NPC record is the withholding evidence.
    scope = TrustedScope(
        "session-1",
        "market",
        ("orso-pell", "dock-warden"),
        ("rill",),
        (("dock-warden", "alive"),),
    )
    other = ProgressiveDecisionSession.start("stab the dock warden with my dagger")
    assert isinstance(
        (
            await verify_plan(
                proceed,
                "stab the dock warden with my dagger",
                other,
                combat=_fight("orso-pell"),
                scope=scope,
                classify=fake_classify_intent,
            )
        ).plan,
        RequestDecisionPlan,
    )


def test_m13_the_superseded_hostility_lexicons_are_gone():
    """``_NON_HOSTILE_WORDS`` and ``_HOSTILE_WORDS`` were written for exactly this problem
    and never wired in: a repository-wide search found no reader of either. The
    scope-aware floor supersedes them -- hostility now comes from the fight's recorded
    sides and lifecycle from ``TrustedScope.npc_states``, neither of which a word list can
    supply -- so they are removed rather than left claiming to do work nothing reads.
    """
    import narrator.decisions as decisions

    assert not hasattr(decisions, "_NON_HOSTILE_WORDS")
    assert not hasattr(decisions, "_HOSTILE_WORDS")


def test_m13_an_asking_route_carries_framing_into_the_narrator_prompt():
    """The prompt states the engine's own routing verdict rather than leaving the model to
    re-derive it from raw text, which is how \"Is there a window?\" resolved as a physical
    event. An acting route adds nothing, so its prompt stays byte-identical.
    """
    plain = turn_prompt("Rill: I force the door")
    assert turn_prompt("Rill: I force the door", turn_framing="") == plain
    assert turn_prompt("Rill: I force the door", turn_framing=turn_framing_for("planner")) == plain

    question = turn_prompt('Rill: Can you see a doorway?', turn_framing=turn_framing_for("read"))
    assert "Turn framing:" in question
    assert "not a declared action" in question
    assert "Do not resolve an action the player has not declared" in question

    ooc = turn_prompt("Rill: @GM why was I refused?", turn_framing=turn_framing_for("out_of_character"))
    assert "out of character" in ooc
    assert ooc != question

    # Neither framing may forbid recording settled fiction: a question asked after a
    # declared action must not cost the player that action. An earlier wording ended "do
    # not advance the fiction on the player's behalf", which reads as forbidding exactly
    # that. This pins the allowance both framings now carry.
    for framed in (question, ooc):
        assert "already established is not resolving a new action" in framed
        assert "advance the fiction on the player" not in framed


class _FramingStubResult:
    """The shape ``Agent.invoke_async`` returns: an object carrying ``.message``."""

    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _FramingStubAgent:
    """Stands in for a Strands agent so this test needs no model, capturing each prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.prompts.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": "Narration."}]})
        return _FramingStubResult("Narration.")


async def _prompt_for(tmp_path: Path, turn_framing: str) -> str:
    """The prompt the real engine composes for one prepared turn, through the real path."""
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine
    from narrator.service import PreparedNarrationTurn

    (tmp_path / "campaign").mkdir(parents=True, exist_ok=True)
    (tmp_path / "campaign" / "state.json").write_text(
        json.dumps({"fiction_debt": []}), encoding="utf-8"
    )
    agent = _FramingStubAgent()
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._agent_for = lambda channel_id: agent  # noqa: SLF001 - test seam

    async def _settled(narration):
        return ("none", 0, "")

    engine._settle = _settled  # noqa: SLF001 - test seam
    source = InboundTurn("c1", ChannelMessage("Rill", "Is there a window?"))
    await engine.run_turn(
        PreparedNarrationTurn(source, None, None, None, False, None, turn_framing)
    )
    return agent.prompts[0]


async def test_m13_the_engine_carries_the_route_framing_into_the_real_prompt(tmp_path: Path):
    """The two halves of this plumbing each had their own guard already, and both passed
    while the prompt the model actually received still said nothing: ``turn_prompt`` grew
    the parameter and the service set the field, but ``NarratorEngine.run_turn`` -- the
    sole production caller -- never forwarded it. This is the guard that fails if that
    forwarding is dropped again.
    """
    framed = await _prompt_for(tmp_path, "question")
    assert "Turn framing:" in framed
    assert "not a declared action" in framed
    assert "Do not resolve an action the player has not declared" in framed

    ooc = await _prompt_for(tmp_path, "out_of_character")
    assert "out of character" in ooc

    # An acting route adds nothing at all, so its prompt is byte-identical.
    declaring = await _prompt_for(tmp_path, "")
    assert "Turn framing:" not in declaring


async def test_m13_the_engine_supplies_a_real_scope_at_every_hazard_call_site(tmp_path: Path):
    """Requirement ``AUD-2``'s required correction. ``NarratorEngine.plan_turn`` is the only
    production caller of ``planning_bypass`` and ``requires_risk_confirmation``, and it
    passed neither a scope nor a combat snapshot, so both resolved to
    ``TrustedScope(\"\", \"\", ())`` however many parameters they had gained. The audit found
    that the criterion's own offline evidence -- \"take the ledger from orso pell\" requiring
    confirmation -- returned the opposite answer through the real call path, because the
    theft branch reads the scope to decide whether a person is involved at all.

    The scene records a present NPC and opens no fight, which is the case a combat-only
    fallback cannot cover: ``hazard_scope`` falls back to the empty placeholder when
    ``combat.active`` is false, so passing the snapshot alone would leave this path
    scope-blind exactly as before.

    Where the scope arrives moved with the model classifier. ``planning_bypass`` and
    ``requires_risk_confirmation`` no longer take one -- they read the single verdict the
    turn already computed -- so the scene now enters through ``classify_intent``, and the
    verdict carries it forward. The spies follow it to both places: ``classify_intent``
    must be handed the real scene, and each hazard call site must be handed a verdict
    derived from that same scene. A ``None`` scope at either point is the placeholder
    defect the criterion names, restated in the shape the code now has.

    Two things are asserted, because either alone is passable while the wiring is broken.
    The spies pin that each call site actually receives a populated scope -- a first
    version asserting only the outcome passed against the reverted wiring, because the
    confirmation was arriving from the planner branch's own ``verify_plan`` further down.
    ``_plan_once`` is stubbed to fail loudly for the same reason: it isolates this
    assertion to the risk floor above the planner, and it stops the \"offline\" suite from
    reaching a served endpoint, which that first version silently did. ``classify_intent``
    is stubbed onto the engine for that second reason as well: an unstarted engine answers
    ``None``, which would withhold the turn instead of measuring the threading.
    """
    from narrator import engine as engine_module
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    campaign = tmp_path / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps({"fiction_debt": [], "npcs": {"orso-pell": {"status": "alive"}}}),
        encoding="utf-8",
    )
    # No "## Combat" section: no fight is open.
    (campaign / "scene.md").write_text(
        "---\nlocation_id: the-eel-market\nsession: 1\n---\n\n"
        "# Probe\n\n## Present NPCs\n\n- orso-pell\n",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}
    classified: list[object] = []
    real_bypass = engine_module.planning_bypass
    real_floor = engine_module.requires_risk_confirmation

    def _spy_bypass(policy):
        seen["planning_bypass"] = policy
        return real_bypass(policy)

    def _spy_floor(policy):
        seen["requires_risk_confirmation"] = policy
        return real_floor(policy)

    async def _spy_classify(declaration, *, scope=None, combat=None):
        classified.append(scope)
        return await fake_classify_intent(declaration, scope=scope, combat=combat)

    async def _no_planner(*args, **kwargs):
        raise AssertionError("the risk floor must author this turn above the planner")

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=2))
    engine._plan_once = _no_planner  # noqa: SLF001 - test seam
    engine.classify_intent = _spy_classify  # the model call, stood in for offline
    monkeyed = {"planning_bypass": _spy_bypass, "requires_risk_confirmation": _spy_floor}
    originals = {name: getattr(engine_module, name) for name in monkeyed}
    for name, replacement in monkeyed.items():
        setattr(engine_module, name, replacement)
    try:
        turn = InboundTurn("c1", ChannelMessage("Rill", "take the ledger from orso pell"))
        outcome = await engine.plan_turn(turn, ())
    finally:
        for name, original in originals.items():
            setattr(engine_module, name, original)

    # The classifier -- the one reader of the scene on this path now -- received it.
    assert classified, "the engine classified nothing"
    for scope in classified:
        assert scope is not None, "classify_intent received no scope"
        assert scope.present_npc_ids == ("orso-pell",), scope
        assert scope.status_of("orso-pell") == "alive", scope

    # And both call sites received a verdict derived from that same scene, not one
    # derived from the placeholder the criterion names.
    for call_site in ("planning_bypass", "requires_risk_confirmation"):
        policy = seen.get(call_site)
        assert policy is not None, f"{call_site} received no verdict"
        assert policy.scope.present_npc_ids == ("orso-pell",), (call_site, policy.scope)
        assert policy.scope.status_of("orso-pell") == "alive", (call_site, policy.scope)

    # And the scene changed the answer: with the placeholder this is not a hazard at all.
    assert isinstance(outcome, PlanOutcome), outcome
    assert isinstance(outcome.plan, RequestDecisionPlan), outcome.plan
    assert outcome.plan.decision.kind == "confirmation"
    # The theft wording, not the violence wording: the category reached the prompt too.
    assert "property" in outcome.plan.decision.question


def _answered_clarification(action: str, label: str, selection_id: str = "listed_path"):
    """One session that declared ``action`` and selected the listed path ``label``."""
    session = ProgressiveDecisionSession.start(action)
    draft = ClarificationDraft(
        kind="clarification",
        question='Which resolution should we use?',
        audience=CurrentAudience(kind="current"),
        paths=(
            ChoiceOption(id=selection_id, label=label),
            ChoiceOption(id="other_path", label="Handle it a different way."),
        ),
    )
    session = session.preparing(draft)
    answer = DecisionResolution(
        character_id="rill",
        kind="clarification",
        answer_kind="option",
        selection_id=selection_id,
    )
    return session.accept(draft, (answer,))


async def test_a_planner_label_never_reclassifies_the_players_action_as_a_hazard():
    """Test a planner label never reclassifies the players action as a hazard.
    """
    session = _answered_clarification(
        'I hold her gently and try to comfort her',
        'Narrative only: Give a short roleplay response without using a mechanical test.',
    )
    outcome = await verify_plan(
        PlanOutcome(plan=ProceedPlan(kind="proceed")),
        session.effective_action,
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, ProceedPlan)


async def test_a_players_own_words_keep_the_hazard_floor_through_a_label():
    """The inverse boundary: when the player's declaration itself carries the
    hazard, selecting a listed path does not launder the confirmation away."""
    session = _answered_clarification(
        "I stab the sleeping merchant",
        "Strike immediately, before anyone can react.",
    )
    outcome = await verify_plan(
        PlanOutcome(plan=ProceedPlan(kind="proceed")),
        session.effective_action,
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, RequestDecisionPlan)
    assert isinstance(outcome.plan.decision, ConfirmationDraft)


async def test_a_confirmed_action_is_never_re_asked_and_never_faults():
    """Test a confirmed action is never re asked and never faults.
    """
    session = ProgressiveDecisionSession.start("I open the door")
    confirmation = ConfirmationDraft(
        kind="confirmation",
        question="Force the heavy door open?",
        audience=CurrentAudience(kind="current"),
    )
    session = session.preparing(confirmation)
    session = session.accept(
        confirmation,
        (
            DecisionResolution(
                character_id="rill",
                kind="confirmation",
                answer_kind="confirm",
                selection_id="confirm",
            ),
        ),
    )
    assert session.action_is_confirmed
    outcome = await verify_plan(
        PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=confirmation)),
        "I open the door",
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, ProceedPlan)


async def test_a_repeated_closed_dimension_proceeds_instead_of_faulting():
    """Same fault family as the confirmed-action repeat: the planner re-opening a
    dimension the session already closed must degrade to proceed, because the player
    already supplied that answer and a fault would discard their whole turn."""
    session = ProgressiveDecisionSession.start("I sort through the cargo")
    approach = ApproachDraft(
        kind="approach",
        question="How do you sort it?",
        audience=CurrentAudience(kind="current"),
        approaches=(
            ApproachOption(
                id="care", label="Slow and careful", directive={"kind": "no_test"}
            ),
            ApproachOption(
                id="fast",
                label="Fast and rough",
                directive={"kind": "attribute_test", "attribute": "DEX"},
            ),
        ),
    )
    session = session.preparing(approach)
    session = session.accept(
        approach,
        (
            DecisionResolution(
                character_id="rill",
                kind="approach",
                answer_kind="option",
                selection_id="care",
            ),
        ),
    )
    assert "approach" in session.closed_dimensions
    outcome = await verify_plan(
        PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=approach)),
        "I sort through the cargo",
        session,
        classify=fake_classify_intent,
    )
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, ProceedPlan)
    # The security boundary is untouched: a planner-authored actor identity still
    # raises rather than degrades.
    named = ClarificationDraft(
        kind="clarification",
        question="Who acts?",
        audience={"kind": "characters", "character_ids": ("rill",)},
        paths=(ChoiceOption(id="one", label="One path"),),
    )
    with pytest.raises(PlannerPolicyError):
        await verify_plan(
            PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=named)),
            "I sort through the cargo",
            session,
            classify=fake_classify_intent,
        )
