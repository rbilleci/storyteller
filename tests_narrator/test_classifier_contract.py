"""The classifier's composed contract: its bytes, its seams, and its coherence rules.

Three things are pinned here, all offline.

The bytes. ``narrator.classify`` composes its prompt and schema from facets and
providers (``narrator.facets``), and the composition must reproduce, byte for byte,
the fused prompt it replaced -- because that prompt is a measured artifact
(``tests_narrator/test_probe_classifier.py`` accepted it at ten repeats per case).
The golden files under ``tests_narrator/golden/`` hold those bytes: one prompt per
scene configuration, the strict JSON schema the OpenAI client sends (property order
included, since vLLM's guided decoding emits properties in that order), and the
system prompts. The assessor's prompt is pinned the same way, because it shares the
providers. **A failure here is not a test to update; it is a prompt change.** To
change a golden, re-run the live probes at ten repeats, record the measurement in
``tests_narrator/golden/ACCEPTANCE.md``, and regenerate with
``scripts/capture_classifier_goldens.py``.

The seams. A context can compose a classifier with fewer facets and the schema,
prompt and derivation all follow; a provider that declares a trust class outside the
two the design allows is refused; a verdict from a custom composition derives with
that composition's own rules through the unchanged ``policy_from``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from openai.lib._pydantic import to_strict_json_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.assess import (  # noqa: E402
    ASSESSOR_SYSTEM_PROMPT,
    HazardAssessment,
    assessment_prompt,
)
from narrator.classify import (  # noqa: E402
    CLASSIFIER_SYSTEM_PROMPT,
    DECLARED_ACT_KINDS,
    DEFAULT_TURN_CLASSIFIER,
    FACETS,
    HAZARDS,
    PROVIDERS,
    RESOLVERS,
    ROUTES,
    SOCIAL_CATEGORIES,
    TRADE_PHASES,
    AcceptanceNeedsOpenOffer,
    DeclaredActNamesItsTool,
    DeclaredActNeedsAskingRoute,
    DefenceNeedsOpenFight,
    DefenceYieldsToHazard,
    HazardFacet,
    HazardForcesRisk,
    InvitedReplyWins,
    ObservationNeedsReadRoute,
    OfferProvider,
    PersonsProvider,
    PresenceProvider,
    ReadFacet,
    ReplyInheritsReferents,
    RouteFacet,
    SocialBindingsFollowTheRoute,
    SocialFacet,
    TurnClassification,
    build_turn_classifier,
    classification_prompt,
    classifier_for,
    policy_from,
)
from narrator.facets import (  # noqa: E402
    TRUSTED,
    ClassificationContext,
    CompositionError,
)
from narrator.interactions import named_dead_target  # noqa: E402
from narrator.policy_types import (  # noqa: E402
    CombatSnapshot,
    InteractionCue,
    TrustedScope,
)
from narrator.social import SocialCategory  # noqa: E402

GOLDEN = REPO_ROOT / "tests_narrator" / "golden"

# The scene configurations the goldens were captured against. Kept in lockstep with
# ``scripts/capture_classifier_goldens.py``; a case added there needs a golden here.
DECLARATION = "I take the rope and punch the trader. What next?"
EMPTY = TrustedScope("", "", ())
MARKET = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell", "maren"), ("maren",),
    (("rade", "alive"), ("orso-pell", "alive"), ("maren", "alive")),
)
CORPSE = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell"), ("rill",),
    (("rade", "dead"), ("orso-pell", "alive")),
)
UNKNOWN = TrustedScope("session-1", "the-eel-market", ("rade", "ghost"), (), (("rade", "alive"),))
FIGHT_NPC_TURN = CombatSnapshot(
    active=True, round=2, active_actor="reed-thug", order=("reed-thug", "rill", "ossa"),
    sides=(("rill", "pc"), ("ossa", "pc"), ("reed-thug", "npc")),
)
FIGHT_PC_TURN = CombatSnapshot(
    active=True, round=2, active_actor="rill", order=("rill", "reed-thug"),
    sides=(("rill", "pc"), ("reed-thug", "npc")),
)
FIGHT_NO_ACTOR = CombatSnapshot(
    active=True, round=1, active_actor="", order=("rill", "reed-thug"),
    sides=(("rill", "pc"), ("reed-thug", "npc")),
)
FIGHT_NO_ALLIES = CombatSnapshot(
    active=True, round=1, active_actor="reed-thug", order=("reed-thug",),
    sides=(("reed-thug", "npc"),),
)
FIGHT_NO_NPCS = CombatSnapshot(
    active=True, round=1, active_actor="rill", order=("rill",), sides=(("rill", "pc"),)
)


PERSONS = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell", "maren"), ("maren",),
    (("rade", "alive"), ("orso-pell", "alive"), ("maren", "alive")),
    ("salt-magistrate-clerk", "sera-vane"),
)
PERSONS_ONLY = TrustedScope("session-1", "the-eel-market", (), ("rill",), (), ("salt-magistrate-clerk",))
GOLDEN_CASES: dict[str, tuple[TrustedScope | None, CombatSnapshot | None, bool]] = {
    "empty": (EMPTY, None, False),
    "persons": (PERSONS, None, False),
    "persons-only": (PERSONS_ONLY, None, False),
    "empty-none-scope": (None, None, False),
    "market": (MARKET, None, False),
    "market-offer": (MARKET, None, True),
    "corpse": (CORPSE, CombatSnapshot(), False),
    "unknown-status": (UNKNOWN, None, False),
    "fight-npc-turn": (CORPSE, FIGHT_NPC_TURN, False),
    "fight-pc-turn": (CORPSE, FIGHT_PC_TURN, False),
    "fight-no-actor": (MARKET, FIGHT_NO_ACTOR, False),
    "fight-no-allies": (MARKET, FIGHT_NO_ALLIES, True),
    "fight-no-npcs": (MARKET, FIGHT_NO_NPCS, False),
    "fight-none-scope": (None, FIGHT_NPC_TURN, False),
}


def _golden(name: str) -> str:
    return (GOLDEN / name).read_text(encoding="utf-8")


def _strict_schema(model) -> str:
    return json.dumps(to_strict_json_schema(model), indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# The bytes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_the_composed_classifier_prompt_matches_its_golden(name: str):
    scope, combat, offer = GOLDEN_CASES[name]
    assert classification_prompt(DECLARATION, scope, combat, offer) == _golden(
        f"classifier_prompt.{name}.txt"
    ), f"classifier prompt for {name} changed; see the module docstring before updating"


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_the_composed_assessor_prompt_matches_its_golden(name: str):
    scope, combat, _ = GOLDEN_CASES[name]
    assert assessment_prompt(DECLARATION, scope, combat) == _golden(
        f"assessor_prompt.{name}.txt"
    ), f"assessor prompt for {name} changed; see the module docstring before updating"


def test_the_strict_schemas_match_their_goldens_including_property_order():
    """``sort_keys`` hides order, so order is asserted on its own beside the bytes."""
    assert _strict_schema(TurnClassification) == _golden("classifier_schema.json")
    assert _strict_schema(HazardAssessment) == _golden("assessor_schema.json")
    expected = [name for facet in FACETS for name in facet.fields] + ["reason"]
    assert list(to_strict_json_schema(TurnClassification)["properties"]) == expected
    assert expected == [
        "route", "hazard", "social_category", "social_mode", "trade_phase",
        "named_person_ids", "names_unrecorded_person", "interlocutor_id",
        "accepts_offer", "offered_price", "departs", "bare_answer",
        "replies_to_question", "defence_method", "reads_by_observing",
        "romance_escalation", "romance_coercive", "reason",
    ]


def test_the_system_prompts_match_their_goldens():
    assert CLASSIFIER_SYSTEM_PROMPT == _golden("classifier_system_prompt.txt")
    assert ASSESSOR_SYSTEM_PROMPT == _golden("assessor_system_prompt.txt")


def test_every_golden_file_is_covered_by_a_case():
    """A golden nothing reads is a golden that silently stopped pinning anything."""
    expected = {f"classifier_prompt.{name}.txt" for name in GOLDEN_CASES}
    expected |= {f"assessor_prompt.{name}.txt" for name in GOLDEN_CASES}
    expected |= {f"defence_companion_prompt.{name}.txt" for name in COMPANION_CASES}
    expected |= {f"hazard_companion_prompt.{name}.txt" for name in COMPANION_CASES}
    expected |= {f"persons_companion_prompt.{name}.txt" for name in COMPANION_CASES}
    expected |= {f"route_companion_prompt.{name}.txt" for name in COMPANION_CASES}
    expected |= {
        "classifier_schema.json", "assessor_schema.json",
        "defence_companion_schema.json", "hazard_companion_schema.json",
        "persons_companion_schema.json", "route_companion_schema.json",
        "classifier_system_prompt.txt", "assessor_system_prompt.txt", "ACCEPTANCE.md",
    }
    assert {path.name for path in GOLDEN.iterdir()} == expected


# ---------------------------------------------------------------------------
# The vocabularies are derived, not restated.
# ---------------------------------------------------------------------------


def test_the_social_vocabulary_is_the_enum_and_nothing_else():
    """The schema, the prompt line and the mechanical set all follow one source.

    A hand-written copy once carried ``information`` where the enum says
    ``information_gathering``; the consequence was an eaten turn, not a bad label.
    """
    assert SOCIAL_CATEGORIES == tuple(sorted(c.value for c in SocialCategory)) + ("none",)
    assert set(TurnClassification.model_fields["social_category"].annotation.__args__) == set(
        SOCIAL_CATEGORIES
    )
    line = next(rule for rule in SocialFacet().rules() if rule.startswith("social_category:"))
    for category in SocialCategory:
        assert category.value in line
    assert set(TurnClassification.model_fields["hazard"].annotation.__args__) == set(HAZARDS)
    assert set(TurnClassification.model_fields["route"].annotation.__args__) == set(ROUTES)
    assert set(TurnClassification.model_fields["trade_phase"].annotation.__args__) == set(
        TRADE_PHASES
    )


# ---------------------------------------------------------------------------
# The seams.
# ---------------------------------------------------------------------------


def test_a_provider_outside_the_two_trust_classes_is_refused():
    """Narration, the canon digest and tool results have no trust value to declare."""

    class DigestProvider:
        name = "digest"
        trust = "model_output"

        def lines(self, context):
            return ["The narration said the stall holds a lantern."]

    assert "model_output" not in TRUSTED
    with pytest.raises(CompositionError, match="trust"):
        build_turn_classifier(
            system_prompt="x", providers=[DigestProvider()], facets=FACETS, resolvers=RESOLVERS
        )


def test_two_facets_claiming_one_field_are_refused():
    """Two facets on one field would let the later derivation read the earlier answer."""

    class SecondHazard:
        name = "second_hazard"
        fields = {"hazard": (str, "none")}

        def rules(self):
            return []

        def shape(self):
            return []

        def derive(self, verdict, context, policy):
            return policy

    with pytest.raises(CompositionError, match="claimed by two facets"):
        build_turn_classifier(
            system_prompt="x", providers=PROVIDERS, facets=[HazardFacet(), SecondHazard()],
            resolvers=RESOLVERS,
        )
    # The same facet twice is refused one step earlier, by name.
    with pytest.raises(CompositionError, match="duplicate facet names"):
        build_turn_classifier(
            system_prompt="x", providers=PROVIDERS, facets=[HazardFacet(), HazardFacet()],
            resolvers=RESOLVERS,
        )


def test_a_narrower_composition_asks_fewer_questions_and_still_derives():
    """Session zero: no fight, no stock, so no trade, no defence, no romance facets.

    The schema loses their fields, the prompt loses their lines, the derivation leaves
    their policy fields at defaults, and the resolvers that read those fields are
    inert rather than crashing.
    """
    narrow = build_turn_classifier(
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        providers=(PresenceProvider(),),
        facets=(RouteFacet(), HazardFacet(), SocialFacet(), ReadFacet()),
        resolvers=RESOLVERS,
    )
    fields = set(narrow.schema.model_fields)
    assert fields == {"route", "hazard", "social_category", "social_mode", "reads_by_observing", "reason"}
    prompt = narrow.prompt("I look around", ClassificationContext(scope=MARKET))
    assert "defence_method" not in prompt and "trade_phase" not in prompt
    assert "No fight is currently running." not in prompt  # no fight provider composed
    assert '"reads_by_observing": true|false, "reason": "one short sentence"}' in prompt
    verdict = narrow.schema(
        route="read", hazard="none", social_category="none", social_mode="none",
        reads_by_observing=True, reason="looks around",
    )
    policy = policy_from(verdict, scope=MARKET)
    assert classifier_for(verdict) is narrow
    assert policy.route == "read" and policy.reads_by_observing is True
    assert policy.defence_method == "" and policy.trade_phase == "" and policy.accepts_offer is False


def test_a_custom_composition_derives_with_its_own_rules_through_policy_from():
    """The verdict knows the contract it answered in; call sites thread nothing extra."""

    class EverythingIsRisk:
        name = "everything_is_risk"

        def __call__(self, policy, verdict, context):
            from dataclasses import replace

            return replace(policy, route="risk", risk_category="destructive")

    custom = build_turn_classifier(
        system_prompt="x", providers=PROVIDERS, facets=FACETS,
        resolvers=(EverythingIsRisk(),), schema_name="CustomVerdict",
    )
    verdict = custom.schema(
        route="read", hazard="none", social_category="none", social_mode="none",
        trade_phase="none", reason="r",
    )
    assert policy_from(verdict, scope=MARKET).risk_category == "destructive"
    # The default is untouched by the custom composition existing.
    default = TurnClassification(
        route="read", hazard="none", social_category="none", social_mode="none",
        trade_phase="none", reason="r",
    )
    assert policy_from(default, scope=MARKET).risk_category == "none"


def test_the_default_composition_is_the_one_the_engine_sends():
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=REPO_ROOT))
    assert engine.turn_classifier is DEFAULT_TURN_CLASSIFIER
    assert engine.turn_classifier.schema is TurnClassification
    injected = build_turn_classifier(
        system_prompt="x", providers=PROVIDERS, facets=(RouteFacet(), HazardFacet()),
        resolvers=(HazardForcesRisk(),), schema_name="Injected",
    )
    assert NarratorEngine(NarratorConfig(campaign_root=REPO_ROOT), turn_classifier=injected).turn_classifier is injected


def test_the_offer_provider_is_the_only_service_state_in_the_prompt():
    """Every other provider renders recorded campaign state, except the one
    ``model_sourced`` provider Phase 3 admits for routing (``PersonsProvider``)."""
    assert [p.trust for p in PROVIDERS] == ["recorded_state", "recorded_state", "service_state"]
    assert isinstance(PROVIDERS[-1], OfferProvider)
    # The one ``model_sourced`` provider lives in the persons companion, never here.
    assert "model_sourced" in TRUSTED and "model_output" not in TRUSTED
    assert [p.trust for p in PERSONS_COMPANION.providers] == [
        "recorded_state", "model_sourced", "recorded_state",
    ]


def test_persons_reach_the_classifier_for_routing_only():
    """The persons line appears only when the scope holds persons; a person may be the
    interlocutor (``scene_person``); a person named as a *target* is dropped from
    ``named_person_ids`` and sets ``names_unrecorded_person`` -- so the consent floor
    reads exactly what it read before.
    """
    from narrator.classify import _validated_cue, policy_from
    from narrator.facets import ClassificationContext
    from narrator.interactions import named_dead_target

    assert PersonsProvider().lines(ClassificationContext(scope=MARKET)) == []
    assert PersonsProvider().lines(ClassificationContext(scope=PERSONS)) == [
        "Persons the narration has introduced, not yet recorded as NPCs. A message that "
        "addresses one of them, by name or by role and in any language, names that "
        "identifier as interlocutor_id: salt-magistrate-clerk, sera-vane."
    ]
    # The full prompt never carries persons: its bytes for a persons scene are the
    # bytes for the same scene without them, so no capture that predates persons moved.
    assert classification_prompt(DECLARATION, PERSONS) == classification_prompt(DECLARATION, MARKET)
    assert persons_companion_applies(ClassificationContext(scope=PERSONS))
    assert not persons_companion_applies(ClassificationContext(scope=MARKET))
    assert "Persons the narration has introduced" in PERSONS_COMPANION.prompt(DECLARATION, ClassificationContext(scope=PERSONS))

    # merge_persons: monotone and routing-only.
    nobody = TurnClassification(route="social", hazard="none", **_BASE)
    companion = PERSONS_COMPANION.schema(route="social", interlocutor_id="salt-magistrate-clerk", **_BASE)
    merged = merge_persons(nobody, companion, PERSONS)
    assert merged.interlocutor_id == "salt-magistrate-clerk" and merged.hazard == "none"
    rade = TurnClassification(route="social", hazard="none", interlocutor_id="rade", **_BASE)
    assert merge_persons(rade, companion, PERSONS) is rade  # a resolved NPC stands
    ghost = PERSONS_COMPANION.schema(route="social", interlocutor_id="ghost", **_BASE)
    assert merge_persons(nobody, ghost, PERSONS) is nobody  # not a present person
    assert merge_persons(nobody, None, PERSONS) is nobody  # a faulted companion changes nothing
    violent = TurnClassification(route="risk", hazard="violence", named_person_ids=["salt-magistrate-clerk"], **_BASE)
    merged = merge_persons(violent, companion, PERSONS)
    assert merged.hazard == "violence" and policy_from(merged, scope=PERSONS).names_unrecorded_person is True
    cue = _validated_cue("salt-magistrate-clerk", PERSONS)
    assert cue is not None and cue.kind == "scene_person" and cue.public_npc_id == "salt-magistrate-clerk"
    assert _validated_cue("salt-magistrate-clerk", MARKET) is None
    assert _validated_cue("rade", PERSONS).kind == "canonical_npc"

    addressed = TurnClassification.model_validate(
        {**_BASE, "route": "social", "hazard": "none", "interlocutor_id": "salt-magistrate-clerk"}
    )
    policy = policy_from(addressed, scope=PERSONS)
    assert policy.route == "social"
    assert policy.interaction_cue == InteractionCue("scene_person", "salt-magistrate-clerk")

    targeted = TurnClassification.model_validate(
        {
            **_BASE, "route": "risk", "hazard": "violence",
            "named_person_ids": ["salt-magistrate-clerk"], "names_unrecorded_person": False,
        }
    )
    policy = policy_from(targeted, scope=PERSONS)
    assert policy.named_person_ids == ()
    assert policy.names_unrecorded_person is True
    assert named_dead_target(policy) is False

    # A dead NPC looted beside a person: the person keeps the ask, as any unrecorded
    # person did before Phase 3.
    corpse_and_person = TrustedScope(
        "session-1", "the-eel-market", ("rade",), ("rill",), (("rade", "dead"),), ("salt-magistrate-clerk",)
    )
    loot = TurnClassification.model_validate(
        {**_BASE, "route": "risk", "hazard": "theft",
         "named_person_ids": ["rade", "salt-magistrate-clerk"], "names_unrecorded_person": False}
    )
    policy = policy_from(loot, scope=corpse_and_person)
    assert policy.named_person_ids == ("rade",)
    assert policy.names_unrecorded_person is True
    assert named_dead_target(policy) is False


# ---------------------------------------------------------------------------
# The resolvers, one each, in declared order.
# ---------------------------------------------------------------------------

_BASE = dict(social_category="none", social_mode="none", trade_phase="none", reason="x")
_FIGHT = CombatSnapshot(
    active=True, round=1, active_actor="reed-thug", order=("reed-thug", "rill"),
    sides=(("rill", "pc"), ("reed-thug", "npc")),
)


def test_the_resolver_order_is_the_declared_contract():
    assert [r.name for r in RESOLVERS] == [
        "hazard_forces_risk",
        "social_bindings_follow_the_route",
        "observation_needs_read_route",
        "declared_act_needs_asking_route",
        "declared_act_names_its_tool",
        "defence_needs_open_fight",
        "defence_yields_to_hazard",
        "acceptance_needs_open_offer",
        "invited_reply_wins",
        "reply_inherits_referents",
    ]
    assert isinstance(RESOLVERS[0], HazardForcesRisk)
    assert isinstance(RESOLVERS[1], SocialBindingsFollowTheRoute)
    assert isinstance(RESOLVERS[2], ObservationNeedsReadRoute)
    assert isinstance(RESOLVERS[3], DeclaredActNeedsAskingRoute)
    # The tool table reads the kind the rule above has already cleared off every route
    # where the declaration says nothing, so it can only ever name a tool for a live one.
    assert isinstance(RESOLVERS[4], DeclaredActNamesItsTool)
    assert isinstance(RESOLVERS[5], DefenceNeedsOpenFight)
    assert isinstance(RESOLVERS[6], DefenceYieldsToHazard)
    assert isinstance(RESOLVERS[7], AcceptanceNeedsOpenOffer)
    assert isinstance(RESOLVERS[8], InvitedReplyWins)
    assert isinstance(RESOLVERS[9], ReplyInheritsReferents)


def test_reply_inherits_referents_over_an_empty_answer():
    """Test reply inherits referents over an empty answer.
    """
    verdict = TurnClassification(route="risk", hazard="theft", **_BASE)
    policy = policy_from(
        verdict, scope=CORPSE,
        last_named_person_ids=("rade",), awaiting_referent_reply=True,
    )
    assert policy.named_person_ids == ("rade",)
    assert policy.names_unrecorded_person is False
    assert named_dead_target(policy) is True


def test_reply_inherits_referents_only_when_the_message_names_no_one():
    """A reply that names someone new is never overridden -- inheriting can only
    add referents a silent message would otherwise have none of."""
    verdict = TurnClassification(
        route="risk", hazard="theft", named_person_ids=["orso-pell"], **_BASE
    )
    policy = policy_from(
        verdict, scope=CORPSE,
        last_named_person_ids=("rade",), awaiting_referent_reply=True,
    )
    assert policy.named_person_ids == ("orso-pell",)


def test_reply_inherits_referents_only_right_after_a_referent_question():
    """Without ``awaiting_referent_reply``, a referent-less message stays referent-less
    -- inheritance is bounded to the turn immediately following the question that
    invited it, never an arbitrary earlier one."""
    verdict = TurnClassification(route="risk", hazard="theft", **_BASE)
    policy = policy_from(verdict, scope=CORPSE, last_named_person_ids=("rade",))
    assert policy.named_person_ids == ()
    assert named_dead_target(policy) is False


def test_hazard_forces_risk():
    verdict = TurnClassification(route="planner", hazard="violence", **_BASE)
    policy = policy_from(verdict, scope=MARKET)
    assert policy.route == "risk" and policy.risk_category == "violence"


def test_social_bindings_follow_the_route():
    social = TurnClassification(
        route="social", hazard="none", social_category="persuasion",
        social_mode="concise_intent", trade_phase="none", reason="x",
    )
    policy = policy_from(social, scope=MARKET)
    assert policy.social_category == "persuasion" and policy.social_test_required is True
    # In-character speech narrates rather than tests.
    spoken = social.model_copy(update={"social_mode": "ic_speech"})
    assert policy_from(spoken, scope=MARKET).social_test_required is False
    # A hazard has already taken the route, so no social test rides into the ask.
    violent = social.model_copy(update={"hazard": "violence"})
    policy = policy_from(violent, scope=MARKET)
    assert policy.route == "risk"
    assert policy.social_category == "" and policy.social_test_required is False


def test_observation_needs_read_route():
    verdict = TurnClassification(route="planner", hazard="none", reads_by_observing=True, **_BASE)
    assert policy_from(verdict, scope=MARKET).reads_by_observing is False
    reading = verdict.model_copy(update={"route": "read"})
    assert policy_from(reading, scope=MARKET).reads_by_observing is True


def test_declared_act_needs_asking_route():
    """The route companion's answer is kept exactly where a framing says "you asked".

    The default schema has no ``also_declares_act`` field -- the full prompt's bytes are
    unchanged on purpose -- so an unmerged verdict answers False, which is the behavior
    every recorded measurement was taken against. A merged one carries the flag on the
    two asking routes and nowhere else, and never moves the route itself.

    ``declared_act_kind`` is cleared with the flag, so no later rule can name a tool for
    a declaration this resolver has just discarded.
    """
    from narrator.policy_types import turn_framing_for

    asked = TurnClassification(route="read", hazard="none", **_BASE)
    assert policy_from(asked, scope=MARKET).also_declares_act is False

    declared = merge_route(
        asked,
        ROUTE_COMPANION.schema(
            route="read", also_declares_act=True, declared_act_kind="rest", reason="r"
        ),
    )
    policy = policy_from(declared, scope=MARKET)
    assert policy.also_declares_act is True
    assert policy.declared_act_kind == "rest"
    assert policy.route == "read", "the companion informs the framing, it never promotes the route"
    assert turn_framing_for(policy.route, policy.also_declares_act) == "question_with_act"

    ooc = merge_route(
        TurnClassification(route="out_of_character", hazard="none", **_BASE),
        ROUTE_COMPANION.schema(route="out_of_character", also_declares_act=True, reason="r"),
    )
    assert policy_from(ooc, scope=MARKET).also_declares_act is True

    # Off an asking route the flag says nothing: a planner turn is already framed as a
    # declaration, and a hazard has taken the risk route one resolver earlier.
    planning = merge_route(
        TurnClassification(route="planner", hazard="none", **_BASE),
        ROUTE_COMPANION.schema(
            route="planner", also_declares_act=True, declared_act_kind="rest", reason="r"
        ),
    )
    dropped = policy_from(planning, scope=MARKET)
    assert dropped.also_declares_act is False
    assert dropped.declared_act_kind == "" and dropped.declared_act_tool == ""
    violent = merge_route(
        TurnClassification(route="read", hazard="violence", **_BASE),
        ROUTE_COMPANION.schema(route="read", also_declares_act=True, reason="r"),
    )
    policy = policy_from(violent, scope=MARKET)
    assert policy.route == "risk" and policy.also_declares_act is False


def test_declared_act_names_its_tool():
    """The kind is the model's answer; the tool it implies is the engine's own table.

    The framing that carries this was measured naming no tool at all ("the tool that
    owns it"), and its residual failures were turns the model agreed had declared an
    action. Naming the tool is engine policy over ``narrator.policy.MCP_TOOLS``, so it
    is asserted here rather than left to whatever the model would have volunteered.
    """
    from narrator.classify import DECLARED_ACT_TOOLS
    from narrator.policy import MCP_TOOLS

    assert set(DECLARED_ACT_TOOLS) == set(DECLARED_ACT_KINDS)
    named = {tool for tool in DECLARED_ACT_TOOLS.values() if tool}
    assert named <= MCP_TOOLS, f"the table names a tool the server does not serve: {named - MCP_TOOLS}"

    def _tool(kind: str, combat=None) -> str:
        verdict = merge_route(
            TurnClassification(route="read", hazard="none", **_BASE),
            ROUTE_COMPANION.schema(
                route="read", also_declares_act=True, declared_act_kind=kind, reason="r"
            ),
        )
        return policy_from(verdict, scope=MARKET, combat=combat).declared_act_tool

    assert _tool("rest") == "rest"
    assert _tool("move") == "scene_commit"
    # Two candidate tools is not a tool this framing may name: the generic wording
    # stands rather than nudging the narrator toward the wrong call.
    assert _tool("search") == "" and _tool("watch") == "" and _tool("other") == ""
    # Inside a fight the rulebook gives movement its own action, and so does the table.
    assert _tool("move", combat=CombatSnapshot(active=True, round=1)) == "combat_move"
    assert _tool("rest", combat=CombatSnapshot(active=True, round=1)) == "rest"
    # A verdict the companion never widened names nothing at all.
    plain = TurnClassification(route="read", hazard="none", **_BASE)
    assert policy_from(plain, scope=MARKET).declared_act_tool == ""


def test_defence_needs_open_fight():
    """Fails against the fused ``policy_from``, which carried ``dodge`` with no fight."""
    verdict = TurnClassification(route="planner", hazard="none", defence_method="dodge", **_BASE)
    assert policy_from(verdict, scope=MARKET).defence_method == ""
    assert policy_from(verdict, scope=MARKET, combat=CombatSnapshot()).defence_method == ""
    assert policy_from(verdict, scope=MARKET, combat=_FIGHT).defence_method == "dodge"


def test_defence_yields_to_hazard():
    """Fails against the fused ``policy_from``, which carried ``parry`` beside violence.

    The live probe records that the classifier does sometimes bind ``dodge`` for "I
    dodge his swing and stab the barkeep"; the hazard stands and the defence drops.
    """
    verdict = TurnClassification(route="planner", hazard="violence", defence_method="parry", **_BASE)
    policy = policy_from(verdict, scope=MARKET, combat=_FIGHT)
    assert policy.route == "risk" and policy.risk_category == "violence"
    assert policy.defence_method == ""


def test_acceptance_needs_open_offer():
    """Fails against the fused ``policy_from``, which had no ``offer_open`` at all."""
    verdict = TurnClassification(route="social", hazard="none", accepts_offer=True, **_BASE)
    assert policy_from(verdict, scope=MARKET).accepts_offer is False
    assert policy_from(verdict, scope=MARKET, offer_open=True).accepts_offer is True


def test_invited_reply_wins_and_wins_last():
    focus = InteractionCue("canonical_npc", "rade")
    verdict = TurnClassification(
        route="planner", hazard="none", bare_answer="yes", replies_to_question=True, **_BASE
    )
    # Without an invitation the answer is an ordinary planner turn carrying the answer.
    plain = policy_from(verdict, scope=MARKET, active_focus=focus, awaiting_reply=False)
    assert plain.route == "planner" and plain.bare_answer == "yes"
    # With one, the reply lands on the interlocutor and nothing else survives.
    reply = policy_from(verdict, scope=MARKET, active_focus=focus, awaiting_reply=True)
    assert reply.route == "social" and reply.interaction_cue == focus
    assert reply.retains_focus is True and reply.bare_answer == ""
    # A name is an answer too: ``replies_to_question`` alone suffices.
    named = verdict.model_copy(update={"bare_answer": "none"})
    assert policy_from(named, scope=MARKET, active_focus=focus, awaiting_reply=True).route == "social"
    # No focus to land on means no reply branch, whatever the narration asked.
    assert policy_from(verdict, scope=MARKET, active_focus=None, awaiting_reply=True).route == "planner"


def test_a_hallucinated_interlocutor_is_dropped_and_an_unrecorded_name_is_kept():
    verdict = TurnClassification(
        route="risk", hazard="violence", interlocutor_id="nobody-here",
        named_person_ids=["rade", "nobody-here"], **_BASE,
    )
    policy = policy_from(verdict, scope=CORPSE)
    assert policy.interaction_cue is None
    assert policy.named_person_ids == ("rade",)
    assert policy.names_unrecorded_person is True


# ---------------------------------------------------------------------------
# The duplicate classification is gone: the engine plans on the service's verdict.
# ---------------------------------------------------------------------------


async def test_plan_turn_reuses_the_services_verdict_instead_of_classifying_again(tmp_path: Path):
    """One classification per turn, across the service/engine boundary too.

    Before ``plan_turn`` took ``policy``, the engine re-classified the same declaration
    the service had just classified -- a second model call per planner-routed turn.
    """
    from dataclasses import replace

    from fake_classifier import classification_for, policy_for

    from narrator.channels.base import ChannelMessage, InboundTurn
    from narrator.config import NarratorConfig
    from narrator.decisions import ProgressiveDecisionSession
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))
    classified: list[str] = []

    async def spy(declaration, *, scope=None, combat=None, offer_open=False):
        classified.append(declaration)
        return classification_for(declaration, scope=scope, combat=combat, offer_open=offer_open)

    engine.classify_intent = spy
    text = "What are the traders selling?"
    turn = InboundTurn("terminal", ChannelMessage("Rill", text))

    # With the service's verdict for this exact text, nothing is classified again.
    outcome = await engine.plan_turn(turn, (), (), policy=policy_for(text))
    assert outcome.plan.kind == "proceed"
    assert classified == []

    # Without it, the engine classifies once -- the behavior every older caller keeps.
    outcome = await engine.plan_turn(turn, (), ())
    assert outcome.plan.kind == "proceed"
    assert classified == [text]

    # A later round planning a different effective action classifies that action: the
    # service's verdict was about the turn's text, not about a planner-authored label.
    classified.clear()
    relabeled = replace(ProgressiveDecisionSession.start(text), effective_action="Look at the traders")
    await engine.plan_turn(turn, (), (), session=relabeled, policy=policy_for(text))
    assert classified == ["Look at the traders"]


async def test_the_service_hands_the_policy_to_a_planner_that_accepts_it_and_not_to_one_that_does_not():
    from narrator.service import NarratorService

    received: dict = {}

    async def current(turn, eligible, resolutions, *, session=None, interaction_cue=None, policy=None):
        received.update(session=session, interaction_cue=interaction_cue, policy=policy)
        return "planned"

    async def legacy(turn, eligible, resolutions, *, session=None):
        received.update(session=session)
        return "planned-legacy"

    async def oldest(turn, eligible, resolutions):
        return "planned-oldest"

    marker = object()
    assert await NarratorService._plan(None, current, "t", (), (), "s", interaction_cue="c", policy=marker) == "planned"
    assert received["policy"] is marker and received["interaction_cue"] == "c"
    assert await NarratorService._plan(None, legacy, "t", (), (), "s", policy=marker) == "planned-legacy"
    assert await NarratorService._plan(None, oldest, "t", (), (), "s", policy=marker) == "planned-oldest"


# ---------------------------------------------------------------------------
# Companions: narrow compositions selected by typed state, merged monotonically.
# ---------------------------------------------------------------------------

from narrator.classify import (  # noqa: E402
    DEFENCE_COMPANION,
    HAZARD_COMPANION,
    HAZARD_COMPANION_RULE,
    PERSONS_COMPANION,
    ROUTE_COMPANION,
    DeclaredActFacet,
    defence_companion_applies,
    hazard_companion_applies,
    merge_defence,
    merge_hazard,
    merge_persons,
    merge_route,
    persons_companion_applies,
    route_companion_applies,
)

#: Scene configurations the companion goldens are captured against: the state that
#: selects each companion, plus one where it is not selected (the prompt must still
#: render, because the compositions are injectable and a test may drive them directly).
COMPANION_CASES: dict[str, tuple[TrustedScope | None, CombatSnapshot | None, bool]] = {
    "persons": (PERSONS, None, False),
    "persons-only": (PERSONS_ONLY, None, False),
    "fight-npc-turn": (CORPSE, FIGHT_NPC_TURN, False),
    "fight-pc-turn": (CORPSE, FIGHT_PC_TURN, False),
    "market": (MARKET, None, False),
    "market-offer": (MARKET, None, True),
}


@pytest.mark.parametrize("name", sorted(COMPANION_CASES))
def test_the_companion_prompts_match_their_goldens(name: str):
    scope, combat, offer = COMPANION_CASES[name]
    context = ClassificationContext(scope=scope, combat=combat, offer_open=offer)
    assert PERSONS_COMPANION.prompt(DECLARATION, context) == _golden(
        f"persons_companion_prompt.{name}.txt"
    )
    assert DEFENCE_COMPANION.prompt(DECLARATION, context) == _golden(
        f"defence_companion_prompt.{name}.txt"
    )
    assert HAZARD_COMPANION.prompt(DECLARATION, context) == _golden(
        f"hazard_companion_prompt.{name}.txt"
    )
    assert ROUTE_COMPANION.prompt(DECLARATION, context) == _golden(
        f"route_companion_prompt.{name}.txt"
    )


def test_the_companion_schemas_match_their_goldens():
    assert _strict_schema(DEFENCE_COMPANION.schema) == _golden("defence_companion_schema.json")
    assert _strict_schema(HAZARD_COMPANION.schema) == _golden("hazard_companion_schema.json")
    assert _strict_schema(ROUTE_COMPANION.schema) == _golden("route_companion_schema.json")
    # Field order is answer order (``facets.compose_schema``). ``also_declares_act`` is
    # the load-bearing, already-shipped field and the kind is only read through it, so
    # the kind is additive after it rather than ahead of it.
    assert list(ROUTE_COMPANION.schema.model_fields) == [
        "route", "also_declares_act", "declared_act_kind", "reason",
    ]
    assert list(DEFENCE_COMPANION.schema.model_fields) == ["route", "defence_method", "reason"]
    assert list(HAZARD_COMPANION.schema.model_fields) == [
        "route", "hazard", "named_person_ids", "names_unrecorded_person", "interlocutor_id",
        "accepts_offer", "offered_price", "reason",
    ]


def test_the_hazard_companion_carries_its_one_extra_rule_and_the_defence_companion_none():
    """The rule that fixed the theft and cost the parry lives only where no parry is asked."""
    hazard_prompt = HAZARD_COMPANION.prompt(DECLARATION, ClassificationContext(scope=MARKET))
    assert HAZARD_COMPANION_RULE in hazard_prompt
    assert "defence_method" not in hazard_prompt
    assert HAZARD_COMPANION_RULE not in classification_prompt(DECLARATION, MARKET)
    defence_prompt = DEFENCE_COMPANION.prompt(DECLARATION, ClassificationContext(scope=CORPSE, combat=FIGHT_NPC_TURN))
    assert "hazard" not in defence_prompt and HAZARD_COMPANION_RULE not in defence_prompt
    assert "It is reed-thug's turn to act, so this character is the one defending." in defence_prompt


def test_the_route_companions_rule_lives_only_in_its_own_composition():
    """The trailing-question rule for the *route* field, asked where nothing else is.

    The hazard field's version of this rule is in the full prompt and was measured
    there; this one is not, and must not be -- the full prompt's bytes are the
    measured artifact every golden pins. So the rule appears in the route companion's
    prompt and in no other, the companion carries only the two questions it needs, and
    the full prompt for the same scene is unchanged.
    """
    rule = DeclaredActFacet().rules()[0]
    context = ClassificationContext(scope=MARKET)
    route_prompt = ROUTE_COMPANION.prompt(DECLARATION, context)
    assert rule in route_prompt
    assert rule not in classification_prompt(DECLARATION, MARKET)
    assert rule not in HAZARD_COMPANION.prompt(DECLARATION, context)
    assert rule not in DEFENCE_COMPANION.prompt(DECLARATION, context)
    assert rule not in PERSONS_COMPANION.prompt(DECLARATION, context)
    # It asks the route (so the social branch's presence list is load-bearing) and the
    # declared act, and nothing else: no hazard, no defence, no referents, no offer.
    assert "People recorded present in the scene: rade" in route_prompt
    for absent in ("hazard", "defence_method", "named_person_ids", "accepts_offer", "fight"):
        assert absent not in route_prompt, absent
    assert [p.name for p in ROUTE_COMPANION.providers] == ["presence"]
    assert [p.trust for p in ROUTE_COMPANION.providers] == ["recorded_state"]
    assert [f.name for f in ROUTE_COMPANION.facets] == ["route", "declared_act"]


def test_the_route_companion_is_selected_only_on_the_asking_routes():
    """One extra call on a question-shaped turn, none on any other."""
    context = ClassificationContext(scope=MARKET)
    for route in ("read", "out_of_character"):
        verdict = TurnClassification(route=route, hazard="none", **_BASE)
        assert route_companion_applies(verdict, context) is True, route
    for route in ("risk", "social", "planner"):
        verdict = TurnClassification(route=route, hazard="none", **_BASE)
        assert route_companion_applies(verdict, context) is False, route


def test_the_route_merge_is_monotone_and_never_moves_the_route():
    base = TurnClassification(route="read", hazard="none", **_BASE)
    declared = ROUTE_COMPANION.schema(route="planner", also_declares_act=True, reason="r")
    nothing = ROUTE_COMPANION.schema(route="read", also_declares_act=False, reason="r")
    merged = merge_route(base, declared)
    assert merged.also_declares_act is True
    # The companion's own route answer is discarded: only the full verdict routes a turn.
    assert merged.route == "read"
    assert merge_route(base, nothing) is base
    assert merge_route(base, None) is base
    # Already declared: nothing to add, and the flag is never cleared.
    assert merge_route(merged, nothing) is merged
    assert merge_route(merged, declared) is merged


def test_the_defence_companion_is_selected_exactly_when_the_fight_provider_states_the_turn():
    assert defence_companion_applies(ClassificationContext(combat=FIGHT_NPC_TURN)) is True
    assert defence_companion_applies(ClassificationContext(combat=FIGHT_PC_TURN)) is False
    assert defence_companion_applies(ClassificationContext(combat=FIGHT_NO_ACTOR)) is False
    assert defence_companion_applies(ClassificationContext(combat=FIGHT_NO_NPCS)) is False
    assert defence_companion_applies(ClassificationContext(combat=CombatSnapshot())) is False
    assert defence_companion_applies(ClassificationContext()) is False


def test_the_hazard_companion_is_selected_on_the_typed_signature_only():
    signature = TurnClassification(
        route="social", hazard="none", accepts_offer=True, named_person_ids=["rade"], **_BASE
    )
    closed = ClassificationContext(scope=MARKET, offer_open=False)
    assert hazard_companion_applies(signature, closed) is True
    # An unrecorded person counts as a person named.
    unrecorded = signature.model_copy(update={"named_person_ids": [], "names_unrecorded_person": True})
    assert hazard_companion_applies(unrecorded, closed) is True
    # Each leg of the signature missing means no second question.
    assert hazard_companion_applies(signature, ClassificationContext(scope=MARKET, offer_open=True)) is False
    assert hazard_companion_applies(signature.model_copy(update={"hazard": "theft"}), closed) is False
    assert hazard_companion_applies(signature.model_copy(update={"accepts_offer": False}), closed) is False
    assert hazard_companion_applies(signature.model_copy(update={"named_person_ids": []}), closed) is False


def test_merges_are_monotone():
    base = TurnClassification(route="planner", hazard="none", **_BASE)
    parry = DEFENCE_COMPANION.schema(route="planner", defence_method="parry", reason="r")
    nothing = DEFENCE_COMPANION.schema(route="planner", defence_method="none", reason="r")
    # A binding is added only to a verdict that bound nothing and declared no hazard.
    assert merge_defence(base, parry).defence_method == "parry"
    assert merge_defence(base, nothing) is base
    assert merge_defence(base, None) is base
    assert merge_defence(base.model_copy(update={"defence_method": "dodge"}), parry).defence_method == "dodge"
    assert merge_defence(base.model_copy(update={"hazard": "violence"}), parry).defence_method == "none"
    theft = HAZARD_COMPANION.schema(route="risk", hazard="theft", reason="r")
    none = HAZARD_COMPANION.schema(route="planner", hazard="none", reason="r")
    # A hazard is added, never removed or downgraded.
    assert merge_hazard(base, theft).hazard == "theft"
    assert merge_hazard(base, none) is base
    assert merge_hazard(base, None) is base
    assert merge_hazard(base.model_copy(update={"hazard": "violence"}), theft).hazard == "violence"
    # The merged verdict still derives with the default composition, and the resolver
    # order makes the added hazard route the turn.
    assert policy_from(merge_hazard(base, theft), scope=MARKET).route == "risk"


async def _engine_with(
    monkeypatch, full, defence=None, hazard=None, persons=None, route=None, tmp_path=None
):
    from narrator.config import NarratorConfig
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._client = object()  # noqa: SLF001 - started-engine seam
    calls: list[str] = []

    async def fake_full(prompt):
        calls.append("classify")
        return full

    async def fake_companion(classifier, declaration, context, *, origin):
        calls.append(origin)
        if origin == "classify_defence":
            if isinstance(defence, Exception):
                raise defence
            return defence
        if origin == "classify_persons":
            if isinstance(persons, Exception):
                raise persons
            return persons
        if origin == "classify_route":
            if isinstance(route, Exception):
                raise route
            return route
        if isinstance(hazard, Exception):
            raise hazard
        return hazard

    monkeypatch.setattr(engine, "_classify_once", fake_full)
    monkeypatch.setattr(engine, "_companion_once", fake_companion)
    return engine, calls


async def test_outside_a_fight_the_engine_makes_exactly_one_call(monkeypatch, tmp_path):
    full = TurnClassification(route="planner", hazard="none", **_BASE)
    engine, calls = await _engine_with(monkeypatch, full, tmp_path=tmp_path)
    verdict = await engine.classify_intent("I look around", scope=MARKET, combat=None)
    assert verdict is full and calls == ["classify"]
    assert engine._companion_log[-1] == {"defence": "skipped", "hazard": "skipped", "persons": "skipped", "route": "skipped"}  # noqa: SLF001


async def test_on_an_enemys_turn_the_defence_companion_runs_beside_the_full_call_and_binds(monkeypatch, tmp_path):
    full = TurnClassification(route="planner", hazard="none", **_BASE)
    parry = DEFENCE_COMPANION.schema(route="planner", defence_method="parry", reason="r")
    engine, calls = await _engine_with(monkeypatch, full, defence=parry, tmp_path=tmp_path)
    verdict = await engine.classify_intent("ich pariere", scope=CORPSE, combat=FIGHT_NPC_TURN)
    assert verdict.defence_method == "parry"
    assert sorted(calls) == ["classify", "classify_defence"]
    assert engine._companion_log[-1]["defence"] == "bound"  # noqa: SLF001
    # The merged verdict reaches the policy the service binds on.
    assert policy_from(verdict, scope=CORPSE, combat=FIGHT_NPC_TURN).defence_method == "parry"
    # On the player's own turn the companion is not selected at all.
    engine, calls = await _engine_with(monkeypatch, full, defence=parry, tmp_path=tmp_path)
    await engine.classify_intent("ich pariere", scope=CORPSE, combat=FIGHT_PC_TURN)
    assert calls == ["classify"]


async def test_a_companion_fault_leaves_the_full_verdict_and_a_full_fault_still_fails_closed(monkeypatch, tmp_path):
    full = TurnClassification(route="planner", hazard="none", **_BASE)
    engine, _ = await _engine_with(monkeypatch, full, defence=RuntimeError("endpoint"), tmp_path=tmp_path)
    # ``_companion_once`` itself swallows faults in production; the stub raises to show the
    # merge treats any non-verdict as "no change".
    engine._companion_once = lambda *a, **k: _none()  # noqa: SLF001
    verdict = await engine.classify_intent("ich pariere", scope=CORPSE, combat=FIGHT_NPC_TURN)
    assert verdict is full
    assert engine._companion_log[-1]["defence"] == "fault"  # noqa: SLF001

    async def broken(prompt):
        raise RuntimeError("endpoint")

    engine, _ = await _engine_with(monkeypatch, full, tmp_path=tmp_path)
    monkeypatch.setattr(engine, "_classify_once", broken)
    assert await engine.classify_intent("ich pariere", scope=CORPSE, combat=FIGHT_NPC_TURN) is None
    assert engine._companion_log[-1]["full"] == "fault"  # noqa: SLF001


async def _none():
    return None


async def test_the_hazard_companion_runs_after_the_full_verdict_on_its_signature_and_adds_the_theft(monkeypatch, tmp_path):
    signature = TurnClassification(
        route="social", hazard="none", accepts_offer=True, named_person_ids=["rade"], **_BASE
    )
    theft = HAZARD_COMPANION.schema(route="risk", hazard="theft", reason="r")
    engine, calls = await _engine_with(monkeypatch, signature, hazard=theft, tmp_path=tmp_path)
    verdict = await engine.classify_intent("I take it from rade. What next?", scope=MARKET)
    assert verdict.hazard == "theft" and calls == ["classify", "classify_hazard"]
    assert engine._companion_log[-1] == {"defence": "skipped", "hazard": "added", "persons": "skipped", "route": "skipped"}  # noqa: SLF001
    assert policy_from(verdict, scope=MARKET).route == "risk"
    # A gift: same signature, companion answers none, nothing changes, one ask avoided.
    none = HAZARD_COMPANION.schema(route="social", hazard="none", reason="r")
    engine, calls = await _engine_with(monkeypatch, signature, hazard=none, tmp_path=tmp_path)
    verdict = await engine.classify_intent("I take the coin rade offers me", scope=MARKET)
    assert verdict is signature and engine._companion_log[-1]["hazard"] == "none"  # noqa: SLF001
    # With an offer open the signature is not a signature.
    engine, calls = await _engine_with(monkeypatch, signature, hazard=theft, tmp_path=tmp_path)
    await engine.classify_intent("I take it", scope=MARKET, offer_open=True)
    assert calls == ["classify"]


async def test_the_route_companion_runs_after_a_read_verdict_and_keeps_the_declared_act(monkeypatch, tmp_path):
    """The engine half of the trailing-question route defect, end to end.

    The shape is the short rest that resolved through no tool: the full verdict routes
    ``read`` because the message does ask what is perceptible, the companion reports
    that it also declared an act, and the merged verdict reaches a policy whose framing
    tells the narrator to answer *and* resolve. The route is untouched throughout.
    """
    from narrator.policy_types import turn_framing_for

    asked = TurnClassification(route="read", hazard="none", **_BASE)
    declared = ROUTE_COMPANION.schema(
        route="read", also_declares_act=True, declared_act_kind="rest", reason="r"
    )
    engine, calls = await _engine_with(monkeypatch, asked, route=declared, tmp_path=tmp_path)
    verdict = await engine.classify_intent(
        "We take a short rest and watch the water. Does anything find us?", scope=MARKET
    )
    assert calls == ["classify", "classify_route"]
    assert engine._companion_log[-1]["route"] == "added"  # noqa: SLF001
    policy = policy_from(verdict, scope=MARKET)
    assert policy.route == "read" and policy.also_declares_act is True
    assert turn_framing_for(policy.route, policy.also_declares_act) == "question_with_act"
    # The whole point of the second field: the framing can name the tool this rest
    # owes, instead of telling the narrator to find "the tool that owns it".
    assert policy.declared_act_kind == "rest" and policy.declared_act_tool == "rest"

    # A question that declares nothing: the companion answers False, the verdict is
    # untouched, and the framing is the measured ``question`` string it has always been.
    plain = ROUTE_COMPANION.schema(route="read", also_declares_act=False, reason="r")
    engine, calls = await _engine_with(monkeypatch, asked, route=plain, tmp_path=tmp_path)
    verdict = await engine.classify_intent("What do I see around the market?", scope=MARKET)
    assert verdict is asked and engine._companion_log[-1]["route"] == "none"  # noqa: SLF001
    policy = policy_from(verdict, scope=MARKET)
    assert turn_framing_for(policy.route, policy.also_declares_act) == "question"

    # A declaring route never asks the second question at all.
    planning = TurnClassification(route="planner", hazard="none", **_BASE)
    engine, calls = await _engine_with(monkeypatch, planning, route=declared, tmp_path=tmp_path)
    await engine.classify_intent("I walk down to the quay", scope=MARKET)
    assert calls == ["classify"]

    # A companion fault leaves the full verdict exactly as it was: today's behavior.
    engine, calls = await _engine_with(monkeypatch, asked, route=None, tmp_path=tmp_path)
    verdict = await engine.classify_intent(
        "We take a short rest. Does anything find us?", scope=MARKET
    )
    assert verdict is asked and engine._companion_log[-1]["route"] == "fault"  # noqa: SLF001
    assert policy_from(verdict, scope=MARKET).also_declares_act is False


async def test_with_persons_present_the_persons_companion_runs_beside_the_full_call_and_binds(monkeypatch, tmp_path):
    """Test with persons present the persons companion runs beside the full call and binds.
    """
    full = TurnClassification(route="social", hazard="none", **_BASE)
    clerk = PERSONS_COMPANION.schema(route="social", interlocutor_id="salt-magistrate-clerk", **_BASE)
    engine, calls = await _engine_with(monkeypatch, full, persons=clerk, tmp_path=tmp_path)
    verdict = await engine.classify_intent("Je demande au clerc.", scope=PERSONS, combat=None)
    assert sorted(calls) == ["classify", "classify_persons"]
    assert verdict.interlocutor_id == "salt-magistrate-clerk" and verdict.hazard == "none"
    assert engine._companion_log[-1]["persons"] == "bound"  # noqa: SLF001
    assert policy_from(verdict, scope=PERSONS).interaction_cue == InteractionCue("scene_person", "salt-magistrate-clerk")

    # A full verdict that resolved a recorded NPC is not overridden.
    rade = TurnClassification(route="social", hazard="none", interlocutor_id="rade", **_BASE)
    engine, calls = await _engine_with(monkeypatch, rade, persons=clerk, tmp_path=tmp_path)
    verdict = await engine.classify_intent("Rade!", scope=PERSONS, combat=None)
    assert verdict is rade and engine._companion_log[-1]["persons"] == "none"  # noqa: SLF001

    # No persons in the scene: the companion is not selected at all.
    engine, calls = await _engine_with(monkeypatch, full, persons=clerk, tmp_path=tmp_path)
    await engine.classify_intent("Je demande au clerc.", scope=MARKET, combat=None)
    assert calls == ["classify"]


def test_a_scene_person_is_never_a_seller_and_renders_as_the_interlocutor():
    """Phase 3's "what it does not change", pinned: the trade lane refuses a person
    as seller on both of its seams, and the narrator prompt names the cue kind."""
    from types import SimpleNamespace

    from narrator.prompt import turn_prompt
    from narrator.service import NarratorService
    from narrator.social import SocialState, TradeTerms

    person = InteractionCue("scene_person", "salt-magistrate-clerk")
    terms = TradeTerms("salt-magistrate-clerk", "rope", "rope", 1, "copper", 5, stock_version=1)
    state = SocialState()
    assert state.accept_offer("market", scope=PERSONS, seller=person, terms=terms, delivered=True) is None
    npc_terms = TradeTerms("rade", "rope", "rope", 1, "copper", 5, stock_version=1)
    assert state.accept_offer("market", scope=PERSONS, seller=InteractionCue("canonical_npc", "rade"), terms=npc_terms, delivered=True) is not None

    policy = policy_from(
        TurnClassification(route="social", hazard="none", trade_phase="price_inquiry", interlocutor_id="salt-magistrate-clerk",
                           **{k: v for k, v in _BASE.items() if k != "trade_phase"}),
        scope=PERSONS,
    )
    assert policy.interaction_cue == person
    fake_service = SimpleNamespace(_purchase_executor=object())
    turn = SimpleNamespace(mention=SimpleNamespace(text="how much for the rope?"))
    assert NarratorService._trade_offer(fake_service, turn, policy) is None  # noqa: SLF001

    prompt = turn_prompt("hello", interaction_cue=person)
    assert "kind: scene_person" in prompt and "public_npc_id: salt-magistrate-clerk" in prompt
