"""The corpse rule: when the risk floor's own claim stops being true.

``verify_plan``'s risk branch authors a confirmation whose text makes a factual claim,
and two functions check that claim before the ask goes out.
``combat_sanctions_violence`` checks the violence question's \"who is not already
fighting you\"; ``named_dead_target`` checks its \"someone\". Neither is a special case:
between them, no ask is authored whose premise is false.

The bypass is now gated on the hazard category. The premise fails for violence (\"can
wound or kill someone\") and theft (\"takes property from someone\"), because neither is
true of a corpse. It does not fail for destruction (\"can destroy or permanently alter
something\") -- a corpse is a something, and burning or dismembering one is the most
content-sensitive act in the family. An audit found the ungated version skipping that
ask entirely, so the repair for a spurious confirmation had taken a warranted one with
it.

The rule now reads classified identifiers instead of English nouns. The subtle half is
``names_unrecorded_person``: the retired implementation matched ``_BYSTANDER_NOUNS``
(\"barkeep\", \"merchant\", ...) union campaign identifiers, and the noun half was what made
\"I attack the barkeep and loot rade\" refuse the bypass in a scene recording no barkeep.
A port keyed on validated identifiers alone would drop the barkeep as unresolvable,
reduce the declaration to \"only rade's body\", and bypass a consent ask over a living
person.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fake_classifier import fake_classify_intent  # noqa: E402

from narrator.decisions import (  # noqa: E402
    PlanOutcome,
    ProceedPlan,
    ProgressiveDecisionSession,
    RequestDecisionPlan,
    verify_plan,
)
from narrator.interactions import combat_sanctions_violence, named_dead_target  # noqa: E402
from narrator.policy_types import CombatSnapshot, TrustedScope, TurnPolicy  # noqa: E402

#: rade is a corpse the campaign records; orso-pell is alive and present.
_SCOPE = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell"), ("rill",),
    (("rade", "dead"), ("orso-pell", "alive")),
)


def _policy(
    category: str,
    *,
    named: tuple[str, ...] = (),
    unrecorded: bool = False,
) -> TurnPolicy:
    """A risk-routed policy carrying exactly the classifier facts the rule reads."""
    return TurnPolicy(
        "risk", category, None, None, _SCOPE, False, False,
        named_person_ids=named, names_unrecorded_person=unrecorded,
    )


# ---------------------------------------------------------------------------
# The rule itself.
# ---------------------------------------------------------------------------


def test_the_bypass_needs_at_least_one_person_and_all_of_them_dead():
    """Positive on both halves, so absence never suppresses a confirmation."""
    assert named_dead_target(_policy("theft", named=("rade",))) is True
    # A living person named alongside the corpse is not accounted for by it.
    assert named_dead_target(_policy("theft", named=("rade", "orso-pell"))) is False
    assert named_dead_target(_policy("theft", named=("orso-pell",))) is False
    # Naming nobody is not the same as naming only corpses.
    assert named_dead_target(_policy("theft")) is False


def test_an_unrecorded_person_refuses_the_bypass():
    """The half a port keyed on identifiers alone would lose.

    "I attack the barkeep and loot rade" in a scene recording no barkeep must not
    bypass. The classifier reports the barkeep through ``names_unrecorded_person``
    rather than as an identifier, because there is no identifier to report, and the
    rule refuses on it: an unrecorded person cannot be shown to be a corpse.

    Without this the declaration reduces to its resolvable half -- "only rade's body" --
    and skips a consent ask over someone alive.
    """
    assert named_dead_target(_policy("violence", named=("rade",), unrecorded=True)) is False
    # And an unrecorded person alone, with no corpse named at all.
    assert named_dead_target(_policy("violence", unrecorded=True)) is False


def test_the_unrecorded_refusal_is_what_closed_a_non_english_safety_hole():
    """The retired rule failed OPEN here, and it is why this field exists.

        \"I attack the barkeep and loot rade\"          -> no bypass  (correct)
        \"J'attaque le tavernier et je fouille rade\"   -> BYPASS     (wrong)

    The English noun list was what counted living bystanders. It has no \"tavernier\", so
    the French declaration reduced to the one word it could resolve -- ``rade``, a
    corpse -- and \"every person named is dead\" became vacuously true. The table-safety
    ask over an assault on a living person was skipped.

    The replacement cannot reduce this way: an unresolvable person is reported rather
    than dropped, and any unresolved person refuses the bypass outright.
    """
    # The French shape, as the classifier now reports it: rade resolves, the
    # tavern-keeper does not, and the unresolved person refuses the bypass.
    assert named_dead_target(_policy("violence", named=("rade",), unrecorded=True)) is False
    # The same declaration with every person recorded still refuses, by the other half.
    assert named_dead_target(_policy("violence", named=("rade", "orso-pell"))) is False


def test_a_scene_with_no_corpse_never_bypasses():
    """``dead`` is read from campaign state, so an empty set of corpses refuses."""
    living_only = TrustedScope(
        "session-1", "the-eel-market", ("orso-pell",), ("rill",), (("orso-pell", "alive"),)
    )
    policy = TurnPolicy(
        "risk", "theft", None, None, living_only, False, False,
        named_person_ids=("orso-pell",),
    )
    assert named_dead_target(policy) is False


# ---------------------------------------------------------------------------
# The category gate, through ``verify_plan``.
# ---------------------------------------------------------------------------


async def _floor(category: str) -> object:
    """What the risk floor authors for a declaration naming only the corpse."""
    session = ProgressiveDecisionSession.start("I do something to rade")
    return await verify_plan(
        PlanOutcome(plan=ProceedPlan(kind="proceed")),
        "I do something to rade",
        session,
        combat=None,
        scope=_SCOPE,
        classify=fake_classify_intent,
        policy=_policy(category, named=("rade",)),
    )


async def test_violence_and_theft_against_a_corpse_skip_the_confirmation():
    """Neither question's claim survives contact with a corpse: it cannot be wounded or
    killed, and it cannot be deprived of property.
    """
    for category in ("violence", "theft"):
        outcome = await _floor(category)
        assert isinstance(outcome, PlanOutcome), category
        assert isinstance(outcome.plan, ProceedPlan), (
            f"{category} against a corpse should not ask: {outcome.plan}"
        )


async def test_destroying_a_corpse_still_asks():
    """The half the ungated bypass was wrongly taking with it.
    """
    outcome = await _floor("destructive")
    assert isinstance(outcome, PlanOutcome)
    assert isinstance(outcome.plan, RequestDecisionPlan), (
        "destroying a corpse must still reach the table-safety ask, not proceed"
    )
    assert "destroy" in outcome.plan.decision.question.casefold()


# ---------------------------------------------------------------------------
# The sibling premise check: "who is not already fighting you".
# ---------------------------------------------------------------------------

_FIGHT_SCOPE = TrustedScope(
    "session-1", "vey-docks", ("reed-thug", "maren"), ("rill",),
    (("reed-thug", "alive"), ("maren", "alive")),
)
_FIGHT = CombatSnapshot(
    active=True, round=1, sides=(("reed-thug", "npc"), ("rill", "pc"))
)


def _fight_policy(named: tuple[str, ...] = (), unrecorded: bool = False) -> TurnPolicy:
    return TurnPolicy(
        "risk", "violence", None, None, _FIGHT_SCOPE, False, False,
        named_person_ids=named, names_unrecorded_person=unrecorded,
    )


def test_the_fight_sanctions_violence_against_its_own_enemy():
    """While a fight runs, attacking a combatant is the mechanic, not an assault."""
    assert combat_sanctions_violence(_fight_policy(("reed-thug",)), _FIGHT) is True
    assert combat_sanctions_violence(_fight_policy(), _FIGHT) is True


def test_the_fight_withholds_the_sanction_for_anyone_it_does_not_cover():
    """A bystander or a party member is not covered, so the ask stands.

        \"I attack maren\"   -> withheld   (correct)
        \"J'attaque maren\"  -> withheld   (correct; the identifier is ASCII)
        \"マレンを攻撃する\"   -> SANCTIONED (wrong -- the ask was skipped)
        \"Я атакую Марен\"   -> SANCTIONED (wrong)

    Names now arrive resolved from the classifier, so the script the player types in
    stops mattering. ``test_probe_classifier`` measures the resolution itself live.
    """
    assert combat_sanctions_violence(_fight_policy(("maren",)), _FIGHT) is False
    assert combat_sanctions_violence(_fight_policy(("rill",)), _FIGHT) is False
    assert combat_sanctions_violence(_fight_policy(("reed-thug", "maren")), _FIGHT) is False


def test_an_unresolved_reference_never_defeats_the_sanction():
    """That rule needs every person accounted for, because its claim is that all of them
    are corpses, so an unresolved reference refuses it. This one asks the opposite
    question -- is anyone named who is *outside* the fight -- and an unresolved
    reference is not evidence of one.

    The requirement exists because treating it as evidence was measured failing live:
    one two-round fight asked the lethal-harm confirmation three times about its only
    enemy. Re-measured while porting to the classifier, with the same result: \"stab
    him\" and \"Я атакую бандита\" both carry no resolved name, and both must stay
    sanctioned. The disclosed residual is unchanged -- \"stab the innocent hostage\"
    names nobody the scene records and does not withhold; the tool boundary protects
    them, because prose alone moves no dice.
    """
    assert combat_sanctions_violence(_fight_policy(unrecorded=True), _FIGHT) is True
    assert combat_sanctions_violence(_fight_policy(("reed-thug",), unrecorded=True), _FIGHT) is True
    # A resolved outsider still withholds, even alongside an unresolved reference.
    assert combat_sanctions_violence(_fight_policy(("maren",), unrecorded=True), _FIGHT) is False


def test_a_closed_or_absent_fight_sanctions_nothing():
    assert combat_sanctions_violence(_fight_policy(("reed-thug",)), None) is False
    closed = CombatSnapshot(active=False, sides=(("reed-thug", "npc"),))
    assert combat_sanctions_violence(_fight_policy(("reed-thug",)), closed) is False
