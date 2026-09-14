"""Typed, process-local social and trade mechanics."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from narrator.identifiers import IDENTIFIER
from narrator.policy_types import InteractionCue, TrustedScope


class SocialCategory(str, Enum):  # noqa: UP042 -- StrEnum's serialization differs subtly; not worth the risk here
    CASUAL = "casual_conversation"
    NEGOTIATION = "negotiation"
    PERSUASION = "persuasion"
    DECEPTION = "deception"
    LEADERSHIP = "leadership"
    DEBATE = "debate"
    INTIMIDATION = "intimidation"
    ROMANCE = "romance"
    BRIBERY = "bribery"
    PERFORMANCE = "performance"
    INFORMATION = "information_gathering"
    INSIGHT = "insight"
    ETIQUETTE = "etiquette"
    PROMISE = "promise_disclosure"


class TradePhase(str, Enum):  # noqa: UP042 -- same reasoning as SocialCategory above
    INQUIRY = "price_inquiry"
    COUNTEROFFER = "counteroffer"
    AGREEMENT = "agreement"
    PURCHASE_INTENT = "purchase_intent"
    CONFIRMATION = "authenticated_confirmation"
    COMPLETED = "atomic_mutation"
    CANCELLED = "cancelled"


TERMINAL_TRADE_PHASES = frozenset({TradePhase.COMPLETED, TradePhase.CANCELLED})


ADVANCING_TRADE_PHASES = frozenset(
    {TradePhase.AGREEMENT, TradePhase.PURCHASE_INTENT, TradePhase.CONFIRMATION}
)


_MECHANICAL_SOCIAL_CATEGORIES = frozenset(
    {
        SocialCategory.NEGOTIATION,
        SocialCategory.PERSUASION,
        SocialCategory.DECEPTION,
        SocialCategory.LEADERSHIP,
        SocialCategory.DEBATE,
        SocialCategory.INTIMIDATION,
        SocialCategory.BRIBERY,
        SocialCategory.PERFORMANCE,
        SocialCategory.INSIGHT,
    }
)


@dataclass(frozen=True)
class SocialFrame:
    """Delivery-committed channel context containing trusted mechanics only."""

    scope: TrustedScope
    interlocutor: InteractionCue | None
    category: SocialCategory
    frame_effect: str = ""


@dataclass(frozen=True)
class TradeTerms:
    """Validated non-mutating terms. Values never originate from free-form prose."""

    seller_id: str
    item_id: str
    public_label: str
    quantity: int
    currency: str
    price: int
    stock_version: int | None = None

    def __post_init__(self) -> None:
        if not IDENTIFIER.fullmatch(self.seller_id) or not IDENTIFIER.fullmatch(self.item_id):
            raise ValueError("trade identifiers are invalid")
        if not self.public_label or len(self.public_label) > 120:
            raise ValueError("trade label is invalid")
        if self.quantity < 1 or self.price < 0 or self.currency != "copper":
            raise ValueError("trade terms are invalid")
        if self.stock_version is not None and self.stock_version < 0:
            raise ValueError("trade stock version is invalid")


@dataclass(frozen=True)
class TradeFrame:
    """One channel-local offer with no transcript, principal, or generated prose."""

    scope: TrustedScope
    seller: InteractionCue
    terms: TradeTerms
    phase: TradePhase
    context_fingerprint: str
    staged_effect: str = ""
    counteroffer_price: int | None = None
    agreed_price: int | None = None
    decision_key: str = ""

    def __post_init__(self) -> None:
        if not self.context_fingerprint or len(self.context_fingerprint) != 64:
            raise ValueError("trade context fingerprint is invalid")
        if self.agreed_price is not None and self.agreed_price < 0:
            raise ValueError("trade agreement is invalid")
        if self.counteroffer_price is not None and self.counteroffer_price < 0:
            raise ValueError("trade counteroffer is invalid")


@dataclass(frozen=True)
class SocialTestRequest:
    """One bounded public objective, stakes, and allowed mechanics."""

    category: SocialCategory
    actor_id: str
    objective: str
    failure_stakes: str
    allowed_attributes: tuple[str, ...]
    action_fingerprint: str
    ability_substitutions: tuple[tuple[str, str], ...] = ()
    ability_authorizer: Callable[[str, str], bool] | None = None


@dataclass(frozen=True)
class SocialTestOutcome:
    """A typed result with no dialogue, consent, belief, or action mutation."""

    category: SocialCategory
    mechanic: str
    branch: str
    frame_effect: str


@dataclass(frozen=True)
class RomancePolicy:
    """Session-zero consent policy. Escalation stays disabled by default."""

    allow_escalation: bool = False


@dataclass(frozen=True)
class RomanceDecision:
    """A public disposition that omits safety and consent details."""

    status: str
    allows_test: bool


def social_attributes(category: SocialCategory, *, physical_support: bool = False) -> tuple[str, ...]:
    """Return exact default attributes. Strength needs explicit fiction support."""
    if category is SocialCategory.INSIGHT:
        return ("WIS",)
    if category in {SocialCategory.DEBATE, SocialCategory.ETIQUETTE}:
        return ("INT",) if category is SocialCategory.ETIQUETTE else ("INT", "CHA")
    if category is SocialCategory.INTIMIDATION and physical_support:
        return ("CHA", "STR")
    return ("CHA",)


def classify_romance(
    policy: RomancePolicy,
    *,
    escalation: bool,
    affected_principals: Iterable[str] = (),
    authenticated_consents: Iterable[str] = (),
    coercive_or_exploitative: bool = False,
) -> RomanceDecision:
    """Reject escalation before mechanics. A test never supplies consent."""
    if not escalation:
        return RomanceDecision("rapport", False)
    if not policy.allow_escalation or coercive_or_exploitative:
        return RomanceDecision("unavailable", False)
    affected = frozenset(affected_principals)
    consented = frozenset(authenticated_consents)
    if affected and not affected <= consented:
        return RomanceDecision("unavailable", False)
    return RomanceDecision("available", False)


def trade_fingerprint(scope: TrustedScope, terms: TradeTerms) -> str:
    """Create a stable opaque binding for one current offer."""
    material = "|".join(
        (
            scope.campaign_session,
            scope.location_id,
            terms.seller_id,
            terms.item_id,
            str(terms.quantity),
            terms.currency,
            str(terms.price),
            str(terms.stock_version),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def confirmation_fingerprint(frame: TradeFrame) -> str:
    """Bind a confirmation to the exact current terms, phase, and accepted price."""
    material = "|".join(
        (
            frame.context_fingerprint,
            frame.phase.value,
            str(frame.terms.quantity),
            frame.terms.currency,
            str(frame.terms.price),
            str(frame.terms.stock_version),
            str(frame.counteroffer_price),
            str(frame.agreed_price),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class _ChannelSocial:
    """One channel's own social and trade state.

    Replaces two dicts (``_social``, ``_trades``) that shared this exact key space but
    not a single invalidation rule -- most methods below still touch only one field,
    exactly as the two dicts did independently; only the container is unified.
    """

    social: SocialFrame | None = None
    trade: TradeFrame | None = None


class SocialState:
    """Manage social and trade frames. All contents disappear on process shutdown."""

    def __init__(self) -> None:
        self._channels: dict[str, _ChannelSocial] = {}

    def social_frame(self, channel_id: str, scope: TrustedScope) -> SocialFrame | None:
        channel = self._channels.get(channel_id)
        frame = channel.social if channel else None
        if frame is not None and frame.scope != scope:
            self._channels.pop(channel_id, None)
            return None
        return frame

    def trade_frame(self, channel_id: str, scope: TrustedScope) -> TradeFrame | None:
        """The channel's live frame, or ``None``. A settled trade is never live.
        """
        self.social_frame(channel_id, scope)
        channel = self._channels.get(channel_id)
        frame = channel.trade if channel else None
        if frame is not None and (
            frame.phase in TERMINAL_TRADE_PHASES
            or frame.scope != scope
            or frame.seller.public_npc_id not in scope.present_npc_ids
        ):
            channel.trade = None
            return None
        return frame

    def commit_social(
        self,
        channel_id: str,
        *,
        scope: TrustedScope,
        cue: InteractionCue | None,
        category: SocialCategory | None,
        delivered: bool,
        clear: bool = False,
        frame_effect: str = "",
    ) -> None:
        """Change social state only after a successful delivery."""
        if not delivered:
            return
        if clear:
            self._channels.pop(channel_id, None)
            return
        if category is not None:
            self._channels.setdefault(channel_id, _ChannelSocial()).social = SocialFrame(
                scope, cue, category, frame_effect
            )

    def accept_offer(
        self,
        channel_id: str,
        *,
        scope: TrustedScope,
        seller: InteractionCue,
        terms: TradeTerms,
        delivered: bool,
        staged_effect: str = "offer",
    ) -> TradeFrame | None:
        """Accept typed narrator terms after delivery and strict catalog validation."""
        if not delivered or seller.public_npc_id != terms.seller_id:
            return None
        if seller.public_npc_id not in scope.present_npc_ids:
            return None
        if terms.stock_version is None:
            return None
        frame = TradeFrame(
            scope,
            seller,
            terms,
            TradePhase.INQUIRY,
            trade_fingerprint(scope, terms),
            staged_effect,
        )
        self._channels.setdefault(channel_id, _ChannelSocial()).trade = frame
        return frame

    def advance_trade(
        self, channel_id: str, *, scope: TrustedScope, policy
    ) -> TradeFrame | None:
        """Advance typed trade stages without a resource or equipment mutation.

        Reads the turn's classification rather than the declaration's words. The live
        frame is still the fact the classification lacks, and it is still supplied here
        rather than to the model: ``policy.accepts_offer`` says the message takes up an
        offer, and only this method knows whether one is open. That keeps the frame out
        of the classifier prompt, which the context-discipline measurement in
        ``narrator.classify`` is a reason to want -- naming a merchant's goods there
        already turned a theft into ordinary commerce once.
        """
        frame = self.trade_frame(channel_id, scope)
        if frame is None:
            return None
        phase = TradePhase(policy.trade_phase) if policy.trade_phase else None
        # A bare acceptance is a purchase only against a frame already advancing toward
        # one. The retired lexicon needed ``negotiating=True`` to read "take" this way,
        # and read it as a purchase on channels that had never attempted one when the
        # flag was absent.
        if policy.accepts_offer and frame.phase in ADVANCING_TRADE_PHASES:
            phase = TradePhase.PURCHASE_INTENT
        if phase is None:
            return frame
        if phase is TradePhase.COUNTEROFFER:
            price = policy.offered_price or None
            if price is None:
                return frame
            frame = replace(
                frame, phase=phase, counteroffer_price=price, agreed_price=None,
                staged_effect="counteroffer",
            )
        elif phase is TradePhase.AGREEMENT:
            if frame.phase is TradePhase.COUNTEROFFER and frame.staged_effect != "terms_improved":
                return frame
            agreed = frame.counteroffer_price if frame.counteroffer_price is not None else frame.terms.price
            frame = replace(frame, phase=phase, agreed_price=agreed, staged_effect="agreement")
        elif phase is TradePhase.PURCHASE_INTENT and frame.phase is TradePhase.AGREEMENT:
            frame = replace(frame, phase=phase, staged_effect="purchase_intent")
        self._channels.setdefault(channel_id, _ChannelSocial()).trade = frame
        return frame

    def apply_social_outcome(
        self, channel_id: str, *, scope: TrustedScope, outcome: SocialTestOutcome | None
    ) -> TradeFrame | None:
        """Apply only an actual bound test result to its current trade frame."""
        frame = self.trade_frame(channel_id, scope)
        if frame is None or outcome is None or outcome.category is not SocialCategory.NEGOTIATION:
            return frame
        if frame.phase is not TradePhase.COUNTEROFFER:
            return frame
        effect = outcome.frame_effect if outcome.branch == "success" else "terms_unchanged"
        frame = replace(frame, staged_effect=effect)
        self._channels.setdefault(channel_id, _ChannelSocial()).trade = frame
        return frame

    def begin_confirmation(
        self, channel_id: str, *, scope: TrustedScope, decision_key: str
    ) -> TradeFrame | None:
        """Open one exact confirmation after a current purchase intent."""
        frame = self.trade_frame(channel_id, scope)
        if frame is None or frame.phase is not TradePhase.PURCHASE_INTENT or not decision_key:
            return None
        frame = replace(frame, phase=TradePhase.CONFIRMATION, decision_key=decision_key)
        self._channels.setdefault(channel_id, _ChannelSocial()).trade = frame
        return frame

    def confirm_purchase(
        self,
        channel_id: str,
        *,
        scope: TrustedScope,
        decision_key: str,
        actor_id: str,
        action_fingerprint: str,
        purchase: Callable[[TradeFrame, str, str], str],
        gate: TradeConfirmationGate | None = None,
    ) -> str:
        """Call an engine-only purchaser once. Declines and stale state write nothing."""
        frame = self.trade_frame(channel_id, scope)
        if frame is None or frame.phase is not TradePhase.CONFIRMATION:
            return "stale"
        if frame.decision_key != decision_key or not actor_id or not action_fingerprint:
            return "unauthorized"
        if gate is not None and not gate.validate(
            decision_key=decision_key,
            actor_id=actor_id,
            action_fingerprint=action_fingerprint,
            frame=frame,
        ):
            return "unauthorized"
        result = purchase(frame, actor_id, action_fingerprint)
        if result in {"confirmed", "idempotent"}:
            # ``COMPLETED`` is terminal, and ``trade_frame`` above drops a terminal frame
            # instead of returning it, so this write records the outcome for this turn
            # and retires the frame for every later one. Nothing reads the frame again
            # after this call inside the confirming turn.
            self._channels.setdefault(channel_id, _ChannelSocial()).trade = replace(
                frame, phase=TradePhase.COMPLETED
            )
        elif result not in {"unauthorized", "stale"}:
            # A current failure such as stale stock or insufficient funds must not
            # strand the channel behind an irrevocable confirmation gate.  Keep the
            # quoted, delivery-committed context so the player can revise or retry.
            self._channels.setdefault(channel_id, _ChannelSocial()).trade = replace(
                frame, phase=TradePhase.AGREEMENT, decision_key="", staged_effect="purchase_failed"
            )
        if gate is not None:
            gate.close(decision_key)
        return result

    def cancel(self, channel_id: str) -> None:
        """Cancel a channel trade without durable side effects."""
        channel = self._channels.get(channel_id)
        if channel is not None and channel.trade is not None:
            channel.trade = replace(channel.trade, phase=TradePhase.CANCELLED)

    def close(self) -> None:
        """Discard every process-local frame during service shutdown."""
        self._channels.clear()


class SocialTestGuard:
    """Bind one social test to its actor, method, and action fingerprint."""

    def __init__(self) -> None:
        self._resolved: set[tuple[str, str, str]] = set()

    def resolve(
        self,
        request: SocialTestRequest,
        *,
        actor_id: str,
        attribute: str,
        action_fingerprint: str,
        success: bool,
    ) -> SocialTestOutcome | None:
        """Reject wrong bindings and repeat outcomes without exposing internals."""
        key = (request.actor_id, request.action_fingerprint, request.category.value)
        if (
            actor_id != request.actor_id
            or action_fingerprint != request.action_fingerprint
            or attribute not in (
                request.allowed_attributes
                + tuple(attribute for _ability, attribute in request.ability_substitutions)
            )
            or key in self._resolved
        ):
            return None
        self._resolved.add(key)
        return SocialTestOutcome(
            request.category,
            f"attribute:{attribute}",
            "success" if success else "failure",
            "terms_improved" if success and request.category is SocialCategory.NEGOTIATION else "bounded_effect",
        )


class TradeConfirmationGate:
    """Keep actor-bound confirmation data outside durable trade frames."""

    def __init__(self) -> None:
        self._open: dict[str, tuple[str, str, str]] = {}

    def open(self, *, decision_key: str, actor_id: str, action_fingerprint: str, frame: TradeFrame) -> bool:
        """Bind one decision to its actor and current trade fingerprint."""
        if not decision_key or not actor_id or len(action_fingerprint) != 64:
            return False
        if decision_key in self._open:
            return False
        self._open[decision_key] = (actor_id, action_fingerprint, confirmation_fingerprint(frame))
        return True

    def validate(
        self, *, decision_key: str, actor_id: str, action_fingerprint: str, frame: TradeFrame
    ) -> bool:
        """Check the actor, exact action, and current terms before a purchase call."""
        return self._open.get(decision_key) == (
            actor_id, action_fingerprint, confirmation_fingerprint(frame)
        )

    def close(self, decision_key: str) -> None:
        """Forget a completed or cancelled process-local confirmation."""
        self._open.pop(decision_key, None)
