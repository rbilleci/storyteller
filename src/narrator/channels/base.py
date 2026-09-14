"""The channel contract. Discord, Slack, a terminal and a transcript are interchangeable.

An adapter owns inbound mechanics and nothing else: deciding when a message addresses
the narrator, buffering the chatter since the previous reply, and putting one string on
the wire when told to. It never decides whether to post, never sees the fiction-debt
ledger, and never talks to the model.

The engine returns an outcome; ``src/narrator/delivery.py`` decides; the adapter posts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

#: The decision-recovery commands a channel consumes, without their leading marker. They
#: live here rather than in one channel because two modules must agree on them. The
#: terminal parses them, and ``narrator.interactions`` must refuse to read one as an
#: answer to an interlocutor. An audit found those two lists able to drift apart.
RECOVERY_CONTROLS = frozenset({"continue", "dismiss", "retry", "revise"})

#: Commands a channel accepts whether or not a recovery is open.
LOCAL_COMMANDS = frozenset({"help", "quit", "thinking", "language"})


STRUCTURED_DECISION_CAPABILITY_VERSION = 1


@dataclass(frozen=True)
class ChannelCapabilities:
    """Versioned channel contract. Unknown future versions fail closed."""

    version: int = STRUCTURED_DECISION_CAPABILITY_VERSION
    structured_decisions: bool = False
    atomic_decision_delivery: bool = False

    def supports_structured_decisions(self) -> bool:
        return (
            self.version == STRUCTURED_DECISION_CAPABILITY_VERSION
            and self.structured_decisions
            and self.atomic_decision_delivery
        )


@dataclass(frozen=True)
class DecisionDeliveryReceipt:
    """One atomic presentation result without a player identity or decision text."""

    status: Literal["delivered", "unavailable"]
    decision_count: int


def structured_decision_capabilities(adapter) -> ChannelCapabilities:
    """Read only the versioned capability object and reject unknown adapter shapes."""
    capabilities = getattr(adapter, "decision_capabilities", None)
    return capabilities if isinstance(capabilities, ChannelCapabilities) else ChannelCapabilities()


@dataclass(frozen=True)
class ChannelPrincipal:
    """Authenticated channel subject.  Display names never authorize decisions."""

    adapter_name: str
    subject_id: str
    display_name: str = ""


@dataclass(frozen=True)
class DecisionCollectionCancelled:
    """An explicit non-answer from a structured-decision input session."""

    reason: Literal["dismissed", "session_ended"]


@dataclass(frozen=True)
class ChannelMessage:
    """One line of channel traffic, from a player or the narrator."""

    author: str
    text: str
    principal: ChannelPrincipal | None = field(default=None, compare=False)


@dataclass(frozen=True)
class InboundTurn:
    """One narrator turn: the message that triggered it plus preceding chatter.

    ``backfill`` is oldest first and capped at ``NarratorConfig.backfill_limit``. It
    reproduces the Discord history backfill the Hermes profile configured at 50, so a
    narrator that has been silent for a while still sees what the table discussed.
    """

    channel_id: str
    mention: ChannelMessage
    backfill: tuple[ChannelMessage, ...] = field(default_factory=tuple)

    def channel_text(self) -> str:
        """Render the turn the way the narrator reads it.
        """
        lines = [f"{message.author}: {message.text}" for message in self.backfill]
        lines.append(f"@GM {self.mention.text}")
        return "\n".join(lines)


@runtime_checkable
class ChannelAdapter(Protocol):
    """Four members. Anything more belongs in the engine or the delivery path."""

    name: str

    def turns(self) -> AsyncIterator[InboundTurn]:
        """Yield one turn each time a message addresses the narrator."""
        ...

    async def post(self, channel_id: str, text: str) -> None:
        """Put one message on the channel. Called only by the delivery path."""
        ...

    async def close(self) -> None:
        """Release transport resources. Called during service shutdown."""
        ...


@runtime_checkable
class StatusLineAdapter(Protocol):
    """Additive renderer contract for a channel that shows a persistent status summary.

    Optional, like ``StructuredDecisionAdapter``: most channels lack this method, and
    ``narrator.delivery`` checks for it with ``getattr`` before calling it. ``snapshot``
    carries only the mechanical, already-durable state ``narrator.status`` reads directly
    from ``campaign/`` -- never model narration, never the fiction-debt ledger -- so a
    channel that renders it is not deciding whether to post; it is redrawing a summary of
    what already posted.
    """

    async def update_status(self, snapshot: dict) -> None:
        """Render one status snapshot. Safe to drop; never gates a turn."""
        ...


@runtime_checkable
class ThinkingIndicatorAdapter(Protocol):
    """Additive renderer contract for a channel that shows a live processing indicator.

    Optional, like ``StatusLineAdapter`` beside it: most channels lack this method, and
    ``NarratorService.run`` checks for it with ``getattr`` before calling it, wrapping
    the whole stretch from picking up a turn to whichever notice or narration answers
    it -- decision-phase routing included, not just the engine's own call, since a
    player waiting on a reply cannot tell those apart and neither should the signal
    they see. ``thinking=True`` starts it, ``thinking=False`` stops it and restores
    whatever this channel was showing before; a channel renders this however fits its
    own transport (an animated status line, a typing indicator, nothing at all). It
    carries no campaign state and decides nothing, so it is safe to drop and never
    gates a turn.
    """

    async def set_thinking(self, channel_id: str, thinking: bool) -> None:
        """Toggle the busy indicator for one channel. Safe to drop; never gates a turn."""
        ...


@dataclass(frozen=True)
class ThinkingLevelControl:
    """What a channel needs to let a player read and set the narrator's thinking level.

    ``levels`` is the ordered vocabulary a player may type; ``read`` returns the level
    in force; ``write`` applies one and returns the level as applied, or raises
    ``ValueError`` for a name outside ``levels``. The service builds this from the
    engine and hands it to the adapter at startup, so the adapter never holds the
    engine itself: it still owns inbound mechanics only, and the one thing it can reach
    through here is a session setting, never the model, the ledger or campaign state.
    """

    levels: tuple[str, ...]
    read: Callable[[], str]
    write: Callable[[str], str]


@runtime_checkable
class ThinkingLevelAdapter(Protocol):
    """Additive contract for a channel whose players can change the thinking level.

    Optional, like the renderer contracts above: ``NarratorService.run`` checks for
    ``bind_thinking_level`` with ``getattr`` once at startup and passes a
    ``ThinkingLevelControl``; a channel without it simply runs at the launch level.
    """

    def bind_thinking_level(self, control: ThinkingLevelControl) -> None:
        """Accept the control this channel's ``/thinking`` command acts through."""
        ...


@dataclass(frozen=True)
class LanguageControl:
    """What a channel needs to let a player read and switch the table's language.

    The same shape as ``ThinkingLevelControl``, for the same reason: the adapter
    receives callables, never the engine or the config, so it still owns inbound
    mechanics only. ``languages`` is the vocabulary a player may type (the language
    directories shipping a catalog); ``read`` returns the tag in force; ``write``
    applies one and returns the tag as applied, or raises ``ValueError`` for a tag no
    catalog covers. ``locale_root`` is here because a switch is two catalogs, not one:
    the write flips the engine's narrator-domain strings, and the adapter must reload
    its *own* domain (``terminal``, a future ``discord``) from the same root, which it
    otherwise never learns -- its catalog arrives pre-loaded at construction.
    """

    languages: tuple[str, ...]
    read: Callable[[], str]
    write: Callable[[str], str]
    locale_root: Path


@runtime_checkable
class LanguageAdapter(Protocol):
    """Additive contract for a channel whose players can switch the table's language.

    Optional, discovered by ``NarratorService.run`` with ``getattr`` once at startup;
    a channel without it simply runs at the launch language (``BSH_LANGUAGE``,
    defaulting to en-US).
    """

    def bind_language(self, control: LanguageControl) -> None:
        """Accept the control this channel's ``/language`` command acts through."""
        ...


@dataclass(frozen=True)
class MentionCandidate:
    """One entity a player may ``@``-mention: the token to insert and its kind.

    Typing-aid data only -- inserting one is exactly as if the player had typed the
    sigil themselves. No module reads ``kind`` to change how a turn routes; it exists
    so a completion menu can label a candidate PC, NPC, GM or person without a second
    lookup. ``"person"`` is a ``bsh_mcp.models.ScenePerson`` -- someone the fiction
    named who holds no mechanical ``NPC`` record (a clerk, a fishmonger) -- kept
    distinct from ``"npc"`` because conflating the two would misrepresent which
    campaign-state dict actually holds the entry.
    """

    sigil: str
    kind: Literal["gm", "pc", "npc", "person"]


@dataclass(frozen=True)
class MentionDirectoryControl:
    """What a channel needs to offer ``@``-mention completion.

    The same shape as ``ThinkingLevelControl``/``LanguageControl``, for the same
    reason: the adapter receives one callable, never the engine, the campaign store
    or the config, so it still owns inbound mechanics only. ``candidates`` returns
    the current roster fresh on every call -- campaign state changes turn to turn,
    and the source files are small enough that no caching is warranted.
    """

    candidates: Callable[[], tuple[MentionCandidate, ...]]


@runtime_checkable
class MentionDirectoryAdapter(Protocol):
    """Additive contract for a channel that offers ``@``-mention completion.

    Optional, discovered by ``NarratorService.run`` with ``getattr`` once at startup,
    like ``bind_thinking_level``/``bind_language`` above; a channel without it simply
    offers no mention completion.
    """

    def bind_mentions(self, control: MentionDirectoryControl) -> None:
        """Accept the control this channel's ``@``-mention completion reads from."""
        ...


@runtime_checkable
class StructuredDecisionAdapter(Protocol):
    """Additive renderer contract for assignment-specific structured decisions."""

    async def present_decision(self, view) -> bool:
        """Render one authorized view.  Return false when this target is unavailable."""
        ...

    async def collect_decision(self, views):
        """Return one answer or an explicit cancellation for the active request."""
        ...

    async def acknowledge_decision(self, result) -> None:
        """Render a semantic answer validation result without exposing raw state."""
        ...

    @property
    def decision_capabilities(self) -> ChannelCapabilities:
        """Advertise the supported structured-decision protocol version."""
        ...

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        """Deliver all routable views as one acknowledged decision presentation."""
        ...
