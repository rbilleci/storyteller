"""The offline ``classify_intent`` every fake engine exposes.

``NarratorService`` reaches the classifier through a duck-typed ``classify_intent`` on
whatever object the channel handed it, and an engine without one routes no turn at all
(see ``NarratorService._classify``). That is deliberate in production -- there is no
lexical fallback behind the model -- but it means every offline fake engine needs a
classifier of its own or the suite would only ever measure the fault notice.

This wraps ``tests_narrator/lexical_double.py`` -- the retired English classifier, kept
as test code -- and renders its ``TurnPolicy`` back into the ``TurnClassification``
shape the model answers in. So the offline suite exercises the real wiring
(``policy_from``, the fail-closed branches, the threading through ``plan_turn`` and
``verify_plan``) against deterministic verdicts, and never reaches an endpoint.

Read ``lexical_double``'s docstring for what this cannot establish: the double is
English-only and measurably worse than the model it stands in for, so a passing offline
test says the engine is wired correctly, never that classification is good. Only
``test_probe_classifier.py`` speaks to that.
"""

from __future__ import annotations

import lexical_double
from lexical_double import counteroffer_price

from narrator.classify import TurnClassification
from narrator.policy_types import TrustedScope
from narrator.social import SocialCategory

_EMPTY_SCOPE = TrustedScope("", "", ())

#: ``TurnPolicy.route`` carries a ``gm`` value the classifier schema deliberately lacks:
#: the game-master designator lane is decided by syntax before any classification runs.
#: It cannot arrive here, and mapping it to ``out_of_character`` rather than crashing
#: keeps a mis-wired test readable instead of raising inside a validator.
_ROUTES = {
    "risk": "risk",
    "social": "social",
    "read": "read",
    "out_of_character": "out_of_character",
    "planner": "planner",
    "gm": "out_of_character",
}

#: Derived from the enum rather than restated. An earlier hand-written copy here held
#: the same invented values production briefly carried ("information", "casual"), which
#: filtered every real category to "none" and made nine of the fourteen unreachable
#: offline -- so a social test that should have bound a mechanic silently did not.
_SOCIAL_CATEGORIES = {category.value for category in SocialCategory}
_TRADE_PHASES = {"price_inquiry", "counteroffer", "agreement", "purchase_intent"}
_SOCIAL_MODES = {
    "concise_intent", "ic_speech", "ooc_action_request", "ooc_rules_inquiry", "mixed",
}


def classification_for(
    declaration: str,
    *,
    scope: TrustedScope | None = None,
    combat=None,
    offer_open: bool = False,
) -> TurnClassification:
    """Render the lexical double's verdict in the model classifier's schema."""
    scene = scope if scope is not None else _EMPTY_SCOPE
    policy = lexical_double.classify_turn(declaration, scope=scene)

    normalized = " ".join(declaration.casefold().split())
    observing = normalized.removeprefix("i ").startswith(
        lexical_double._OBSERVATION_PREFIXES  # noqa: SLF001 - the double is test code
    )

    bare = "none"
    if lexical_double.is_bare_yes_or_no(declaration):
        bare = "no" if lexical_double._word_set(declaration) & {  # noqa: SLF001
            "no", "nope", "nah", "never", "not"
        } else "yes"

    # Reproduce the retired ``_person_words`` split: recorded identifiers become
    # ``named_person_ids``, and a generic English person-noun the scene does not record
    # becomes ``names_unrecorded_person``. Without the second half the double would let
    # "I attack the barkeep and loot rade" bypass a consent ask the lexicon refused.
    words = lexical_double._word_set(declaration)  # noqa: SLF001 - the double is test code
    known = tuple(scene.present_npc_ids) + (
        tuple(actor for actor, _ in combat.sides) if combat is not None else ()
    )
    recorded = tuple(
        person for person in known
        if lexical_double._identifier_tokens(person) & words  # noqa: SLF001
    )
    unrecorded = bool(
        (words & lexical_double._BYSTANDER_NOUNS)  # noqa: SLF001
        - {token for npc_id in recorded for token in lexical_double._identifier_tokens(npc_id)}  # noqa: SLF001
    )

    return TurnClassification(
        route=_ROUTES.get(policy.route, "planner"),
        hazard=policy.risk_category,
        social_category=(
            policy.social_category if policy.social_category in _SOCIAL_CATEGORIES else "none"
        ),
        social_mode=policy.social_mode if policy.social_mode in _SOCIAL_MODES else "none",
        trade_phase=policy.trade_phase if policy.trade_phase in _TRADE_PHASES else "none",
        interlocutor_id=(
            policy.interaction_cue.public_npc_id or ""
            if policy.interaction_cue is not None
            else ""
        ),
        departs=policy.clears_focus,
        # ``accepts_offer`` reproduces what ``negotiating=True`` used to unlock, and
        # ``offered_price`` what ``counteroffer_price`` read, so the offline suite still
        # drives the trade lane through the same decisions.
        # ``offer_open`` is what the retired lexicon took as ``negotiating=True``,
        # which is exactly the flag that let a bare "take" read as acceptance.
        accepts_offer=bool(words & {"buy", "deal", "agreed"}) or (offer_open and "take" in words),
        offered_price=counteroffer_price(declaration) or 0,
        named_person_ids=list(recorded),
        names_unrecorded_person=unrecorded,
        bare_answer=bare,
        replies_to_question=lexical_double.reply_shaped(
            declaration, scene.party_name_tokens
        ),
        defence_method=lexical_double.defence_method(declaration, combat) or "none",
        reads_by_observing=observing and policy.route == "read",
        romance_escalation=policy.romance_escalation,
        romance_coercive=policy.romance_coercive,
        reason="offline lexical double",
    )


async def fake_classify_intent(
    declaration: str,
    *,
    scope: TrustedScope | None = None,
    combat=None,
    offer_open: bool = False,
) -> TurnClassification:
    """The awaitable a fake engine exposes as ``classify_intent``."""
    return classification_for(
        declaration, scope=scope, combat=combat, offer_open=offer_open
    )


class ClassifyingEngine:
    """Mixin giving a fake engine the classifier the service now requires.

    Inherited rather than copied so that a future change to the classifier seam is one
    edit here instead of one per fake. ``classify_calls`` records what was classified,
    which a test asserting the turn was read once can read.
    """

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - cooperative init
        super().__init__(*args, **kwargs)
        self.classify_calls: list[str] = []

    async def classify_intent(
        self, declaration: str, *, scope=None, combat=None, offer_open: bool = False
    ):
        self.classify_calls.append(declaration)
        return classification_for(
            declaration, scope=scope, combat=combat, offer_open=offer_open
        )


def policy_for(
    declaration: str,
    scope: TrustedScope | None = None,
    combat=None,
    offer_open: bool = False,
):
    """The ``TurnPolicy`` this declaration would carry, for a test that has only text.

    ``SocialState.advance_trade`` and the trade gates now read the turn's classification
    instead of re-reading the declaration, which is what removed the last English
    keyword ladder from the trade lane. Tests written against the old ``player_text=``
    signature keep their exact inputs and intent by routing them through the double
    here, rather than hand-building a policy whose fields were chosen to make them pass.

    ``offer_open`` mirrors what ``NarratorService._offer_is_open`` supplies in
    production: a test exercising acceptance against a live frame has to pass it, or it
    measures the classifier without the one fact that disambiguates "I take it".
    """
    from narrator.classify import policy_from

    scene = scope if scope is not None else _EMPTY_SCOPE
    return policy_from(
        classification_for(declaration, scope=scene, combat=combat, offer_open=offer_open),
        scope=scene,
        combat=combat,
        offer_open=offer_open,
    )
