"""Engine-owned trade: the merchant stock schema, the trusted purchase port, and
``TradeMixin``'s tools.

``GameService`` is referenced only as a lazily-evaluated annotation here (the
``TYPE_CHECKING`` import below never runs), so importing this module never
triggers the circular import that a real ``from . import GameService`` would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from .. import results
from ..models import EnginePurchaseRequest, MerchantStock, PurchaseReceipt, slugify
from ..store import CampaignError

if TYPE_CHECKING:
    from . import GameService

# The engine accepts only these normalized item identifiers. Narrator tools never expose
# the purchase method, so narration cannot create catalog entries or transfer resources.
TRADE_ITEM_SCHEMA: dict[str, str] = {
    "rope": "rope",
    "fish": "fish",
    "lantern": "lantern",
    "rations": "rations",
}

# This is engine configuration, not narrator-local dialogue data.  Scene
# provisioning copies it into durable merchant stock only when a present seller has
# no configured inventory; existing campaigns retain their own authoritative terms.
DEFAULT_MERCHANT_STOCK: dict[str, tuple[str, int, int]] = {
    "rope": ("rope", 5, 3),
    "fish": ("fish", 1, 6),
    "lantern": ("lantern", 8, 1),
    "rations": ("rations", 2, 4),
}


class EnginePurchaseExecutor:
    """A non-serializable engine capability for confirmed trade mutation."""

    def __init__(self, service: GameService, capability: object) -> None:
        self._service = service
        self._capability = capability

    def confirm(self, request: EnginePurchaseRequest) -> dict:
        """Submit only a service-created request through the private capability."""
        if not isinstance(request, EnginePurchaseRequest):
            return results.failure("purchase_contract_required", "The purchase terms are unavailable.")
        return self._service.engine_confirm_purchase(request, _capability=self._capability)

    def quote_for_text(self, seller_id: str, text: str) -> dict:
        """Read a current offer through the same engine-owned port."""
        return self._service.engine_quote_trade(seller_id, text, _capability=self._capability)

    def reserve_social_ability(self, actor_id: str, ability_id: str) -> dict:
        """Spend an approved substitution before permitting its bound test."""
        return self._service.reserve_social_ability(
            actor_id, ability_id, _capability=self._capability
        )


class TradeMixin:
    """Engine-owned trade tools: quote, confirm, and the social-ability reservation."""

    def narrator_purchase_executor(self) -> EnginePurchaseExecutor:
        """Issue the trusted narrator host port during production construction."""
        return EnginePurchaseExecutor(self, self._purchase_capability)

    def provision_scene_merchants(self, seller_ids: tuple[str, ...]) -> None:
        """Idempotently provision default stock for trusted scene merchants.

        This is an explicit engine-side scene-initialization operation.  It only
        creates absent records, so it cannot overwrite configured stock or prices.
        """
        valid = tuple(sorted({seller for seller in seller_ids if seller and seller == slugify(seller)}))
        if not valid:
            return
        with self.store.transaction("merchant_scene_provision", "engine", "scene_initialization") as transaction:
            created: list[str] = []
            for seller_id in valid:
                if seller_id in transaction.state.merchant_stocks:
                    continue
                transaction.state.merchant_stocks[seller_id] = MerchantStock(
                    version=1,
                    items={item_id: count for item_id, (_label, _price, count) in DEFAULT_MERCHANT_STOCK.items()},
                    prices={item_id: price for item_id, (_label, price, _count) in DEFAULT_MERCHANT_STOCK.items()},
                )
                created.append(seller_id)
            if created:
                transaction.record("provisioned scene merchant stock")
                transaction.commit({"outcome": "merchant_scene_provision", "sellers": created})

    def engine_quote_trade(
        self, seller_id: str, player_text: str, *, _capability: object | None = None
    ) -> dict:
        """Return one current, versioned offer selected from engine inventory."""
        if _capability is not self._purchase_capability:
            return results.failure("purchase_capability_required", "The trade terms are unavailable.")
        words = set(player_text.casefold().replace("-", " ").split())
        state = self.store.read_state()
        stock = state.merchant_stocks.get(seller_id)
        if stock is None:
            return results.failure("seller_unavailable", "The seller no longer offers this item.")
        available = [
            item_id for item_id in stock.items
            if item_id.replace("-", " ") in " ".join(words) and stock.items[item_id] > 0
        ]
        if len(available) != 1:
            return results.failure("trade_item_unresolved", "Name one available item for a quote.")
        item_id = available[0]
        label = TRADE_ITEM_SCHEMA.get(item_id)
        price = stock.prices.get(item_id)
        if label is None or price is None:
            return results.failure("trade_terms_unavailable", "The seller has no current terms for that item.")
        return results.success(
            "Current merchant terms are available.",
            seller_id=seller_id,
            item_id=item_id,
            public_label=label,
            quantity=1,
            currency="copper",
            price_copper=price,
            stock_version=stock.version,
        )

    def reserve_social_ability(
        self, character_id: str, ability_id: str, *, _capability: object | None = None
    ) -> dict:
        """Atomically validate and spend one substitution before its social roll."""
        if _capability is not self._purchase_capability:
            return results.failure("purchase_capability_required", "The social ability is unavailable.")
        used = self.use_ability(character_id, ability_id, mode="use")
        if used.get("ok") is not True:
            return used
        try:
            with self.store.transaction(
                "reserve_social_ability", actor_id=character_id, reason=ability_id
            ) as transaction:
                character = transaction.character(character_id)
                receipt = f"{character.id}:{ability_id}"
                transaction.state.social_ability_receipts[receipt] = "pending"
                transaction.record("reserved spent social ability for bound test")
                transaction.commit({"outcome": "social_ability_reserved", "ability_id": ability_id})
        except CampaignError as error:
            return error.as_dict()
        return results.success(
            "The selected social ability is spent for the bound test.",
            ability_id=ability_id,
        )

    def engine_confirm_purchase(
        self,
        request: EnginePurchaseRequest | dict,
        *,
        _capability: object | None = None,
    ) -> dict:
        """Apply a confirmation only when the private engine capability authenticated it."""
        if _capability is not self._purchase_capability:
            return results.failure("purchase_capability_required", "The purchase is not authorized.")
        try:
            request = EnginePurchaseRequest.model_validate(request)
        except ValidationError:
            return results.failure("invalid_purchase", "The purchase confirmation is invalid.")
        label = TRADE_ITEM_SCHEMA.get(request.item_id)
        if label is None:
            return results.failure("unknown_trade_item", "The requested item is unavailable.")
        try:
            with self.store.transaction("engine_purchase", request.buyer_id, "confirmed_purchase") as transaction:
                prior = transaction.state.purchase_receipts.get(request.idempotency_key)
                if prior is not None:
                    if prior.trade_fingerprint != request.trade_fingerprint:
                        return results.failure("purchase_replay_conflict", "The confirmation no longer matches.")
                    return results.success("The confirmed purchase was already recorded.", outcome="idempotent")
                stock = transaction.state.merchant_stocks.get(request.seller_id)
                if stock is None:
                    return results.failure("seller_unavailable", "The seller no longer offers this item.")
                if request.stock_version is None or stock.version != request.stock_version:
                    return results.failure("stale_trade_terms", "The quoted terms changed before confirmation.")
                available = stock.items.get(request.item_id)
                if available is None or available < request.quantity:
                    return results.failure("out_of_stock", "The seller no longer has that quantity.")
                current_price = stock.prices.get(request.item_id)
                # A quoted list price is authoritative.  The trusted narrator port
                # may submit a lower agreed price only after its bound social test
                # succeeded; it can never inflate the list price or invent an item.
                if current_price is None or request.price_copper > current_price:
                    return results.failure("stale_trade_terms", "The quoted terms changed before confirmation.")
                buyer = transaction.character(request.buyer_id)
                if buyer.status != "ok":
                    return results.failure("buyer_unavailable", "The buyer cannot complete that purchase.")
                if buyer.coins < request.price_copper:
                    return results.failure("insufficient_funds", "The buyer lacks the agreed coins.")
                buyer.coins -= request.price_copper
                buyer.equipment = buyer.equipment + [label] * request.quantity
                transaction.touch_character(buyer.id)
                stock.items[request.item_id] = available - request.quantity
                stock.version += 1
                transaction.state.purchase_receipts[request.idempotency_key] = PurchaseReceipt(
                    trade_fingerprint=request.trade_fingerprint,
                    event_seq=transaction.state.event_seq + 1,
                )
                transaction.record("confirmed engine purchase")
                sequence = transaction.commit({"outcome": "confirmed_purchase", "purchase": "confirmed"})
        except CampaignError as error:
            return error.as_dict()
        return results.success(
            "The confirmed purchase was recorded.", sequence=sequence, outcome="confirmed",
            state_changes=["coins and equipment changed atomically"],
        )
