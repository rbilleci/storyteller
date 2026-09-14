"""Deterministic social mechanics and engine-only purchase tests.

``narrator.classify`` replaced the English lexical classifier with a model call, so the
routing verdicts here come from ``fake_classifier`` -- the retired lexicon, kept verbatim
as ``tests_narrator/lexical_double.py`` and rendered into the schema the model answers
in. That keeps this suite offline and deterministic while the service's real classifier
seam still runs. It measures wiring, never classification quality; the live gate for that
is ``tests_narrator/test_probe_classifier.py``.
"""

from __future__ import annotations

from hashlib import sha256

from fake_classifier import ClassifyingEngine, policy_for
from lexical_double import SocialInputMode, classify_social_input, classify_turn

from bsh_mcp.models import EnginePurchaseRequest, MerchantStock
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission
from narrator.delivery import TurnOutcome
from narrator.policy_types import InteractionCue, TrustedScope
from narrator.resolution_guard import ResolutionGuard
from narrator.service import NarratorService
from narrator.social import (
    RomancePolicy,
    SocialCategory,
    SocialState,
    SocialTestGuard,
    SocialTestOutcome,
    SocialTestRequest,
    TradeConfirmationGate,
    TradePhase,
    TradeTerms,
    classify_romance,
    social_attributes,
)


def _scope() -> TrustedScope:
    return TrustedScope("session-1", "market", ("rade",))


def _terms() -> TradeTerms:
    return TradeTerms("rade", "rope", "rope", 1, "copper", 5, 4)


def _fingerprint(character: str) -> str:
    return sha256(character.encode("utf-8")).hexdigest()


def _buyer(service, name: str) -> str:
    created = service.character_create(
        discord_user_id=f"social-{name.casefold()}",
        name=name,
        origin="barbarian",
        backgrounds=["scout", "hunter", "survivor"],
        weapons=["long knife"],
    )
    assert created["ok"], created
    return created["character_id"]


def test_input_taxonomy_preserves_ooc_and_quoted_counteroffer_semantics():
    quoted = classify_social_input('"Two copper," I say.')
    offered = classify_social_input("I offer two copper.")
    assert quoted.category is SocialCategory.NEGOTIATION
    assert quoted.trade_phase is TradePhase.COUNTEROFFER
    assert offered.trade_phase is TradePhase.COUNTEROFFER
    assert classify_social_input("OOC: how does bargaining work?").mode is SocialInputMode.OOC_RULES_INQUIRY
    request = classify_social_input("OOC: I bargain for a better price.")
    assert request.mode is SocialInputMode.OOC_ACTION_REQUEST
    assert request.requires_test is True
    mixed = classify_social_input("I smile. OOC: I am deceiving him.")
    assert mixed.mode is SocialInputMode.MIXED
    assert mixed.category is SocialCategory.DECEPTION
    romantic = classify_social_input("I smile. OOC: I am seducing him.")
    assert romantic.mode is SocialInputMode.MIXED
    assert romantic.category is SocialCategory.ROMANCE
    assert romantic.romance_escalation is True


def test_unlisted_escalation_phrases_still_gate():
    """Unlisted escalation phrases still gate."""
    undress_command = classify_social_input('Maren, undress now')
    assert undress_command.category is SocialCategory.ROMANCE
    assert undress_command.romance_escalation is True
    love_scene = classify_social_input("Make love to her")
    assert love_scene.category is SocialCategory.ROMANCE
    assert love_scene.romance_escalation is True
    # Adjacent ordinary uses of the same words must never trip the gate.
    assert classify_social_input("I love this sturdy backpack").romance_escalation is False
    assert classify_social_input("I dry my wet clothes by the fire").romance_escalation is False


def test_social_categories_and_attribute_mappings_are_data_driven():
    cases = {
        "I persuade the clerk.": SocialCategory.PERSUASION,
        "I deceive the clerk.": SocialCategory.DECEPTION,
        "I rally the crowd.": SocialCategory.LEADERSHIP,
        "I debate the facts.": SocialCategory.DEBATE,
        "I intimidate the guard.": SocialCategory.INTIMIDATION,
        "I perform a song.": SocialCategory.PERFORMANCE,
        "I ask for information.": SocialCategory.INFORMATION,
        "I study their motive for insight.": SocialCategory.INSIGHT,
        "I observe etiquette at court.": SocialCategory.ETIQUETTE,
        "I promise to disclose the secret.": SocialCategory.PROMISE,
        "I bribe the clerk.": SocialCategory.BRIBERY,
    }
    assert {classify_social_input(text).category for text in cases} == set(cases.values())
    assert social_attributes(SocialCategory.INSIGHT) == ("WIS",)
    assert social_attributes(SocialCategory.ETIQUETTE) == ("INT",)
    assert social_attributes(SocialCategory.INTIMIDATION) == ("CHA",)
    assert social_attributes(SocialCategory.INTIMIDATION, physical_support=True) == ("CHA", "STR")


def test_social_policy_keeps_active_cue_without_pronouns_or_questions():
    policy = classify_turn("I bargain for a better price.", scope=_scope())
    assert policy.route == "social"
    assert policy.social_test_required is True
    assert policy.social_category == SocialCategory.NEGOTIATION.value
    assert policy.interaction_cue is not None
    assert policy.interaction_cue.kind == "scene_interlocutor"


def test_social_guard_rejects_spoofed_actor_method_fingerprint_and_repeat():
    request = SocialTestRequest(
        SocialCategory.NEGOTIATION, "rill", "Lower the quoted price.", "Terms stay unchanged.",
        ("CHA",), _fingerprint("bargain"),
    )
    guard = SocialTestGuard()
    assert guard.resolve(request, actor_id="ossa", attribute="CHA", action_fingerprint=request.action_fingerprint, success=True) is None
    assert guard.resolve(request, actor_id="rill", attribute="INT", action_fingerprint=request.action_fingerprint, success=True) is None
    assert guard.resolve(request, actor_id="rill", attribute="CHA", action_fingerprint=_fingerprint("other"), success=True) is None
    outcome = guard.resolve(request, actor_id="rill", attribute="CHA", action_fingerprint=request.action_fingerprint, success=True)
    assert outcome is not None and outcome.frame_effect == "terms_improved"
    assert guard.resolve(request, actor_id="rill", attribute="CHA", action_fingerprint=request.action_fingerprint, success=False) is None


def test_romance_policy_never_uses_a_test_as_consent():
    assert classify_romance(RomancePolicy(), escalation=False).status == "rapport"
    assert classify_romance(RomancePolicy(), escalation=True).status == "unavailable"
    policy = RomancePolicy(allow_escalation=True)
    assert classify_romance(policy, escalation=True, affected_principals=("a", "b"), authenticated_consents=("a",)).status == "unavailable"
    assert classify_romance(policy, escalation=True, affected_principals=("a", "b"), authenticated_consents=("a", "b"), coercive_or_exploitative=True).status == "unavailable"
    assert classify_romance(policy, escalation=True, affected_principals=("a",), authenticated_consents=("a",)).allows_test is False


def test_trade_frames_keep_terms_channel_local_and_require_delivery():
    state = SocialState()
    seller = InteractionCue("canonical_npc", "rade")
    assert state.accept_offer("market", scope=_scope(), seller=seller, terms=_terms(), delivered=False) is None
    frame = state.accept_offer("market", scope=_scope(), seller=seller, terms=_terms(), delivered=True)
    assert frame is not None
    counter = state.advance_trade("market", scope=_scope(), policy=policy_for("I would pay 2 copper.", scope=_scope()))
    assert counter is not None and counter.phase is TradePhase.COUNTEROFFER
    assert counter.terms.item_id == "rope" and counter.terms.price == 5 and counter.counteroffer_price == 2
    assert state.trade_frame("other", _scope()) is None
    assert state.advance_trade("market", scope=_scope(), policy=policy_for("Deal.", scope=_scope())) == counter
    state.apply_social_outcome(
        "market", scope=_scope(),
        outcome=SocialTestOutcome(SocialCategory.NEGOTIATION, "attribute:CHA", "success", "terms_improved"),
    )
    agreement = state.advance_trade("market", scope=_scope(), policy=policy_for("Deal.", scope=_scope()))
    assert agreement is not None and agreement.phase is TradePhase.AGREEMENT and agreement.agreed_price == 2
    purchase = state.advance_trade("market", scope=_scope(), policy=policy_for("I buy it now.", scope=_scope()))
    assert purchase is not None and purchase.phase is TradePhase.PURCHASE_INTENT


def test_confirmed_purchase_is_atomic_and_idempotent(service):
    """A confirmed purchase is the only legal way narration-adjacent coin and inventory
    figures can change: it mutates ``Character.coins`` and ``Character.equipment``
    atomically inside one ``store.transaction``, and that transaction's own commit
    appends exactly one audit event -- naming the tool, the buyer, and the confirmed
    outcome -- to ``campaign/logs/events.jsonl``. Pinning the event's own fields, not
    merely its count, is what lets an auditor tell a sanctioned acquisition apart from
    any other write to the log.
    """
    buyer_id = _buyer(service, "Rill")
    with service.store.transaction("test", buyer_id, "fund buyer") as transaction:
        character = transaction.character(buyer_id)
        character.coins = 5
        transaction.touch_character(buyer_id)
        transaction.state.merchant_stocks["rade"] = MerchantStock(version=4, items={"rope": 1}, prices={"rope": 2})
        transaction.commit({"outcome": "setup"})
    action = _fingerprint("purchase")
    request = EnginePurchaseRequest(
        decision_key="decision-key-0001", action_fingerprint=action, trade_fingerprint=_fingerprint("trade"),
        idempotency_key=_fingerprint("receipt"), buyer_id=buyer_id, seller_id="rade", item_id="rope",
        quantity=1, price_copper=2, stock_version=4,
    )
    events_before = service.store.read_events(100)
    rejected = service.engine_confirm_purchase(request)
    assert rejected["ok"] is False and rejected["error"] == "purchase_capability_required"
    confirmed = service.narrator_purchase_executor().confirm(request)
    assert confirmed["ok"] is True and confirmed["outcome"] == "confirmed"
    replay = service.narrator_purchase_executor().confirm(request)
    assert replay["ok"] is True and replay["outcome"] == "idempotent"
    after = service.store.read_character(buyer_id)
    state = service.store.read_state()
    assert after.coins == 3 and after.equipment == ["rope"]
    assert state.merchant_stocks["rade"].items["rope"] == 0
    events_after = service.store.read_events(100)
    # The rejected capability-less call and the idempotent replay wrote zero events:
    # exactly one real mutation happened, and exactly one audit event records it.
    assert len(events_after) == len(events_before) + 1
    purchase_event = events_after[-1]
    assert purchase_event["tool"] == "engine_purchase"
    assert purchase_event["actor_id"] == buyer_id
    assert purchase_event["reason"] == "confirmed_purchase"
    assert purchase_event["outcome"] == "confirmed_purchase"
    assert purchase_event["purchase"] == "confirmed"
    assert "confirmed engine purchase" in purchase_event["changes"]
    assert isinstance(purchase_event["seq"], int) and purchase_event["seq"] > 0
    # The receipt is the idempotency record: it carries the same event sequence the
    # audit log recorded, so a later reconciliation can join the two by that number.
    receipt = state.purchase_receipts[request.idempotency_key]
    assert receipt.event_seq == purchase_event["seq"]
    assert receipt.trade_fingerprint == request.trade_fingerprint


def test_stale_stock_insufficient_funds_and_decline_write_nothing(service):
    buyer_id = _buyer(service, "Mara")
    with service.store.transaction("test", buyer_id, "setup") as transaction:
        character = transaction.character(buyer_id)
        character.coins = 1
        transaction.touch_character(buyer_id)
        transaction.state.merchant_stocks["rade"] = MerchantStock(version=7, items={"rope": 1}, prices={"rope": 2})
        transaction.commit({"outcome": "setup"})
    before = service.store.read_character(buyer_id).model_dump(), service.store.read_state().model_dump()
    stale_request = {
        "decision_key": "decision-key-0002", "action_fingerprint": _fingerprint("a"),
        "trade_fingerprint": _fingerprint("b"), "idempotency_key": _fingerprint("c"),
        "buyer_id": buyer_id, "seller_id": "rade", "item_id": "rope", "quantity": 1,
        "price_copper": 2, "stock_version": 6,
    }
    assert service.engine_confirm_purchase(stale_request)["error"] == "purchase_capability_required"
    stale = service.narrator_purchase_executor().confirm(
        EnginePurchaseRequest(**stale_request)
    )
    assert stale["ok"] is False and stale["error"] == "stale_trade_terms"
    insufficient = service.narrator_purchase_executor().confirm(
        EnginePurchaseRequest(
            decision_key="decision-key-0003",
            action_fingerprint=_fingerprint("d"),
            trade_fingerprint=_fingerprint("e"),
            idempotency_key=_fingerprint("f"),
            buyer_id=buyer_id,
            seller_id="rade",
            item_id="rope",
            quantity=1,
            price_copper=2,
            stock_version=7,
        )
    )
    assert insufficient["ok"] is False and insufficient["error"] == "insufficient_funds"
    after = service.store.read_character(buyer_id).model_dump(), service.store.read_state().model_dump()
    assert before == after


def test_trade_confirmation_gate_rejects_principal_spoof_before_purchase():
    state = SocialState()
    seller = InteractionCue("canonical_npc", "rade")
    state.accept_offer("market", scope=_scope(), seller=seller, terms=_terms(), delivered=True)
    state.advance_trade("market", scope=_scope(), policy=policy_for("I offer two copper.", scope=_scope()))
    state.apply_social_outcome(
        "market", scope=_scope(),
        outcome=SocialTestOutcome(SocialCategory.NEGOTIATION, "attribute:CHA", "success", "terms_improved"),
    )
    state.advance_trade("market", scope=_scope(), policy=policy_for("Agreed.", scope=_scope()))
    state.advance_trade("market", scope=_scope(), policy=policy_for("I buy it now.", scope=_scope()))
    frame = state.begin_confirmation("market", scope=_scope(), decision_key="decision-key-0004")
    assert frame is not None
    action = _fingerprint("gate")
    gate = TradeConfirmationGate()
    assert gate.open(decision_key=frame.decision_key, actor_id="rill", action_fingerprint=action, frame=frame)
    called = []
    assert state.confirm_purchase("market", scope=_scope(), decision_key=frame.decision_key, actor_id="ossa", action_fingerprint=action, purchase=lambda *_: called.append(True) or "confirmed", gate=gate) == "unauthorized"
    assert called == []
    assert state.confirm_purchase("market", scope=_scope(), decision_key=frame.decision_key, actor_id="rill", action_fingerprint=action, purchase=lambda *_: called.append(True) or "confirmed", gate=gate) == "confirmed"
    assert called == [True]


def test_bound_social_guard_rejects_unbound_tests_and_wrong_substitutions():
    request = SocialTestRequest(
        SocialCategory.DECEPTION,
        "rill",
        "Resolve one deception.",
        "The stated effect fails.",
        ("CHA",),
        _fingerprint("social"),
        (("sophist_lie", "CHA"), ("bookworm_substitution", "INT")),
        lambda actor_id, ability_id: actor_id == "rill" and ability_id == "bookworm_substitution",
    )
    empty = ResolutionGuard()
    assert empty.validate_social_test(actor_id="rill", attribute="CHA", action_fingerprint=request.action_fingerprint)
    guard = ResolutionGuard(social_request=request)
    assert guard.validate("attribute_test", {"character_id": "ossa", "attribute": "CHA"})
    assert guard.validate("use_ability", {"character_id": "rill", "ability_id": "other", "mode": "use"})
    assert guard.validate("use_ability", {"character_id": "rill", "ability_id": "bookworm_substitution", "mode": "use"}) is None
    assert guard.validate("attribute_test", {"character_id": "rill", "attribute": "CHA"})
    assert guard.validate("attribute_test", {"character_id": "rill", "attribute": "INT"}) is None
    guard.record_tool_result(
        "attribute_test", {"character_id": "rill", "attribute": "INT"},
        {"ok": True, "outcome": "success"},
    )
    first = guard.social_outcome
    guard.record_tool_result(
        "attribute_test", {"character_id": "rill", "attribute": "INT"},
        {"ok": True, "outcome": "success"},
    )
    assert guard.social_outcome == first
    assert guard.unused_error() == ""


class _TradeAdapter:
    name = "terminal"
    decision_capabilities = ChannelCapabilities(structured_decisions=True, atomic_decision_delivery=True)

    def __init__(self) -> None:
        self.posted: list[tuple[str, str]] = []
        self.views = []

    async def turns(self):
        principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        for text in ("Rade, what is the price for rope?", "I would pay 2 copper.", "Deal.", "I buy it now."):
            yield InboundTurn("market", ChannelMessage("Rill", text, principal))

    async def post(self, channel_id, text):
        self.posted.append((channel_id, text))

    async def close(self):
        return None

    async def deliver_decision_views(self, views):
        self.views.extend(views)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(tuple(views)))

    async def collect_decision(self, views):
        view = views[0]
        return (
            ChannelPrincipal("terminal", "terminal-player", "Rill"),
            DecisionSubmission(presentation_token=view.presentation_token, selection_id="confirm"),
        )

    async def acknowledge_decision(self, result):
        raise AssertionError(result)


class _TradeEngine(ClassifyingEngine):
    """A narrator that answers every turn, carrying the classifier the service requires.

    ``NarratorService`` routes no turn from an engine without ``classify_intent`` -- it
    posts ``classifier_fault_notice`` instead -- so the mixin supplies the offline
    double. See ``tests_narrator/fake_classifier.py`` for what that can establish.
    """

    def __init__(self) -> None:
        super().__init__()
        self.turns = []

    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    async def run_turn(self, turn, **_kwargs):
        self.turns.append(turn)
        outcome = None
        if turn.social_test is not None:
            outcome = SocialTestOutcome(
                SocialCategory.NEGOTIATION, "attribute:CHA", "success", "terms_improved"
            )
        return TurnOutcome("The trader answers.", ratified=True, withheld=False, social_test_outcome=outcome)


async def test_service_owned_trade_requires_delivery_and_authenticated_confirmation(service):
    buyer_id = _buyer(service, "Rill")
    with service.store.transaction("test", buyer_id, "setup social purchase") as transaction:
        buyer = transaction.character(buyer_id)
        buyer.coins = 5
        transaction.touch_character(buyer_id)
        transaction.state.merchant_stocks["rade"] = MerchantStock(version=4, items={"rope": 1}, prices={"rope": 2})
        transaction.commit({"outcome": "setup"})
    root = service.store.root
    (root / "campaign" / "scene.md").write_text(
        "---\nsession: 1\nlocation_id: market\n---\n\n## Present NPCs\n\n- rade\n",
        encoding="utf-8",
    )
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: rill\n  display_name: Rill\n",
        encoding="utf-8",
    )
    adapter = _TradeAdapter()
    engine = _TradeEngine()
    config = NarratorConfig(campaign_root=root, max_decision_rounds=2)
    await NarratorService(
        config,
        adapter,
        engine,
        purchase_executor=service.narrator_purchase_executor(),
    ).run()
    buyer = service.store.read_character(buyer_id)
    stock = service.store.read_state().merchant_stocks["rade"]
    assert len(adapter.views) == 1
    assert engine.turns[0].trade_offer is not None
    assert buyer.coins == 3 and buyer.equipment == ["rope"] and stock.items["rope"] == 0
    assert adapter.posted[-1] == ("market", config.trade_completed_notice)


def test_failed_purchase_releases_confirmation_for_a_later_retry():
    state = SocialState()
    seller = InteractionCue("canonical_npc", "rade")
    state.accept_offer("market", scope=_scope(), seller=seller, terms=_terms(), delivered=True)
    state.advance_trade("market", scope=_scope(), policy=policy_for("I offer two copper.", scope=_scope()))
    state.apply_social_outcome(
        "market", scope=_scope(),
        outcome=SocialTestOutcome(SocialCategory.NEGOTIATION, "attribute:CHA", "success", "terms_improved"),
    )
    state.advance_trade("market", scope=_scope(), policy=policy_for("Deal.", scope=_scope()))
    state.advance_trade("market", scope=_scope(), policy=policy_for("I buy it now.", scope=_scope()))
    frame = state.begin_confirmation("market", scope=_scope(), decision_key="retry-key-0001")
    assert frame is not None
    gate = TradeConfirmationGate()
    action = _fingerprint("retry")
    assert gate.open(decision_key=frame.decision_key, actor_id="rill", action_fingerprint=action, frame=frame)
    assert state.confirm_purchase(
        "market", scope=_scope(), decision_key=frame.decision_key, actor_id="rill",
        action_fingerprint=action, purchase=lambda *_: "failed", gate=gate,
    ) == "failed"
    recovered = state.trade_frame("market", _scope())
    assert recovered is not None and recovered.phase is TradePhase.AGREEMENT
    assert not gate.validate(
        decision_key="retry-key-0001", actor_id="rill", action_fingerprint=action, frame=recovered
    )
    retry = state.advance_trade("market", scope=_scope(), policy=policy_for("I buy it now.", scope=_scope()))
    assert retry is not None and retry.phase is TradePhase.PURCHASE_INTENT


def test_social_substitution_spends_an_actual_eligible_ability_before_the_roll(service):
    created = service.character_create(
        discord_user_id="social-sophist",
        name="Vey",
        origin="barbarian",
        backgrounds=["sophist", "scout", "hunter"],
        weapons=["long knife"],
    )
    assert created["ok"], created
    character_id = str(created["character_id"])
    port = service.narrator_purchase_executor()
    assert port.reserve_social_ability(character_id, "sophist_lie")["ok"] is True
    assert service.store.read_character(character_id).pools["sophist_lie"] == 0
    # The subsequent model-visible tool call acknowledges the same reserved spend,
    # rather than silently giving the model a second free substitution.
    assert service.use_ability(character_id, "sophist_lie")["ok"] is True
    assert service.use_ability(character_id, "sophist_lie")["error"] == "ability_exhausted"
