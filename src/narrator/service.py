"""Process lifecycle for the narrator. One service, one Model Context Protocol session.

Turns run one at a time through a single consumer. The MCP server's campaign lock is the
hard guard against concurrent mutation; this queue is the soft guard against two turns
narrating over each other in the same channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import signal
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol

from narrator.assess import assessment_downgrades
from narrator.channels.base import (
    ChannelAdapter,
    ChannelMessage,
    DecisionCollectionCancelled,
    InboundTurn,
    LanguageControl,
    MentionCandidate,
    MentionDirectoryControl,
    ThinkingLevelControl,
    structured_decision_capabilities,
)
from narrator.config import NarratorConfig
from narrator.decision_coordinator import DecisionCoordinator, DecisionCoordinatorError
from narrator.decisions import (
    ApproachDraft,
    CombatDefendDirective,
    ConfirmationDraft,
    CurrentAudience,
    DecisionResolution,
    DeclinedPlan,
    PlanOutcome,
    ProceedPlan,
    ProgressiveDecisionSession,
    RequestDecisionPlan,
    enemy_turn_defence,
    semantic_fingerprint,
    verify_plan,
)
from narrator.delivery import (
    TurnPost,
    advertises_recovery_controls,
    deliver,
    deliver_decision_decline,
    deliver_decision_fault,
    deliver_decision_recovery,
    deliver_no_action,
    present_decisions,
    refresh_status,
)
from narrator.engine import (
    REPETITION_SIMILARITY_THRESHOLD,
    DecisionPlanningError,
    narration_similarity,
)
from narrator.interactions import (
    InteractionTracker,
    combat_sanctions_violence,
    gm_discussion_policy,
    gm_discussion_remainder,
    open_npc_turn,
    read_combat_snapshot,
    read_trusted_scope,
)
from narrator.player_directory import PlayerDirectory, PlayerDirectoryError
from narrator.policy_types import InteractionCue, TurnPolicy, turn_framing_for
from narrator.social import (
    ADVANCING_TRADE_PHASES,
    RomancePolicy,
    SocialCategory,
    SocialState,
    SocialTestRequest,
    TradeConfirmationGate,
    TradeFrame,
    TradePhase,
    TradeTerms,
    classify_romance,
    social_attributes,
)

if TYPE_CHECKING:
    pass


DECISION_DIAGNOSTIC_ALLOWLIST = frozenset(
    {"category", "decision_kind", "target_count", "answer_category", "failure_category"}
)
DECISION_DIAGNOSTIC_CATEGORIES = frozenset(
    {"answered", "assessed", "cancelled", "dismissed", "fault", "presented", "retry", "segment"}
)


def repeats_answered_question(session: ProgressiveDecisionSession, draft) -> bool:
    """Whether one planned narrowing draft re-asks a question this chain already answered.

    A ``ConfirmationDraft`` is never a repeat. ``remember_question`` explains why: the
    risk floor authors one fixed wording for every unconfirmed hazard, so treating it as
    a duplicate would wave a genuine second hazard through with no confirmation at all.
    """
    if isinstance(draft, ConfirmationDraft):
        return False
    return any(
        narration_similarity(draft.question, asked.question)
        >= REPETITION_SIMILARITY_THRESHOLD
        for asked in session.asked_questions
    )


class _Engine(Protocol):
    """The engine operations the service sequences."""

    def start(self) -> None:
        ...

    async def run_turn(self, turn):
        ...

    def stop(self) -> None:
        ...


class _PurchaseExecutor(Protocol):
    """The non-serializable engine port that owns confirmed purchase mutation."""

    def confirm(self, request) -> dict:
        ...

    def quote_for_text(self, seller_id: str, text: str) -> dict:
        ...

    def reserve_social_ability(self, actor_id: str, ability_id: str) -> dict:
        ...


def _routing_row(policy) -> dict:
    """The typed routing facts of one turn, for the soak's ``routing`` instrument.
    """
    if policy is None:
        return {"route": "fault", "risk_category": "", "names_unrecorded_person": False,
                "interlocutor_kind": "", "interlocutor_id": "",
                "also_declares_act": False, "declared_act_kind": "", "declared_act_tool": ""}
    cue = policy.interaction_cue
    return {
        "route": policy.route,
        "risk_category": policy.risk_category,
        "names_unrecorded_person": bool(policy.names_unrecorded_person),
        "interlocutor_kind": cue.kind if cue is not None else "",
        "interlocutor_id": (cue.public_npc_id or "") if cue is not None else "",
        "also_declares_act": bool(policy.also_declares_act),
        "declared_act_kind": policy.declared_act_kind,
        "declared_act_tool": policy.declared_act_tool,
    }


@dataclass
class ServiceReport:
    """What one service run produced. Returned so a caller can assert on it."""

    turns: int = 0
    delivered: int = 0
    withheld: int = 0
    leaks_scrubbed: int = 0
    settle_attempts: int = 0
    commits: int = 0
    waives: int = 0
    decision_pending: int = 0
    decision_faults: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass(frozen=True)
class PreparedNarrationTurn:
    """Service-owned execution wrapper that preserves the channel body byte-for-byte."""

    source: InboundTurn
    interaction_cue: InteractionCue | None
    social_test: SocialTestRequest | None = None
    trade_offer: TradeTerms | None = None
    romance_escalation_blocked: bool = False


    withheld_note: str | None = None


    turn_framing: str = ""
    #: The MCP tool that resolves the action a ``question_with_act`` turn declared
    #: (``TurnPolicy.declared_act_tool``), or ``""`` when the engine knows of no single
    #: one. Only the ``question_with_act`` framing reads it, and it is a datum rather
    #: than a framing of its own: which tool a declared rest needs is not a different
    #: kind of turn, it is one blank filled in the same instruction.
    declared_act_tool: str = ""
    #: True when the player opened the turn with the configured game-master designator
    #: (``NarratorConfig.gm_address``): out-of-fiction discussion with the game master.
    #: ``NarratorEngine.run_turn`` reads it to refuse every tool outside
    #: ``narrator.policy.GM_DISCUSSION_TOOLS`` and to keep the answer out of the scene
    #: sweep, the narration tail, and the traversal clocks -- out-of-fiction talk must
    #: never move the fiction. The designator itself is already stripped from the
    #: mention text by the time this turn is built.
    meta: bool = False

    @property
    def channel_id(self) -> str:
        return self.source.channel_id

    @property
    def mention(self) -> ChannelMessage:
        return self.source.mention

    @property
    def backfill(self) -> tuple[ChannelMessage, ...]:
        return self.source.backfill

    def channel_text(self) -> str:
        return self.source.channel_text()


#: Every value ``_decision_phase``, ``_routed_decision_phase`` and the phase methods
#: they delegate to (``_trade_confirmation_phase``, ``_social_choice_phase``,
#: ``_enemy_turn_defence_phase``) return, and the complete set ``run``'s dispatch below
#: switches on. Distinct from
#: ``delivery.DecisionPhaseOutcome.status`` (``"fault"``/``"presented"``), a smaller,
#: unrelated vocabulary answering a different question (did presenting a decision UI
#: succeed) despite the similar name.
RoutingPhase = Literal[
    "continue",
    "risk_confirmation_required",
    "fault",
    "pending",
    "segment",
    "declined",
    "stopped",
    "trade_confirmation_required",
    "trade_declined",
    "trade_completed",
    "social_choice_required",
    "classifier_fault",
]


@dataclass(frozen=True)
class _Recovery:
    """One recovery control's adopted state, threaded into the routed phase.

    ``_decision_phase`` consumes the control and re-classifies the retained
    declaration before any route-dependent gate runs, so the routed phase below
    receives the already-adopted session rather than re-deriving it after those
    gates have already read a policy computed for the control token.
    """

    #: The retained progressive session the control acts on.
    session: ProgressiveDecisionSession
    #: Answers already collected for that session.
    resolutions: tuple
    #: The draft a view boundary refused to present, which ``/continue`` resumes.
    pending_draft: object
    #: ``continue``, ``retry`` or ``revise`` -- ``dismiss`` never reaches the routed phase.
    control: str


@dataclass
class _ChannelState:
    """One channel's own decision/consent/withhold state.

    Replaces three dicts (``_progressive_sessions``, ``_romance_consents``,
    ``_withheld_notes``) that shared this exact key space but not a single
    invalidation rule -- each field below still clears on its own trigger, exactly as
    the three dicts did independently; only the container is unified.
    """


    progressive_session: tuple[ProgressiveDecisionSession, tuple, object] | None = None
    #: Authenticated, process-local romance consents for an enabled table policy.
    romance_consents: set[str] = field(default_factory=set)


    withheld_note: str | None = None


class NarratorService:
    """Own the engine's lifetime and drive one adapter through it."""

    def __init__(
        self,
        config: NarratorConfig,
        adapter: ChannelAdapter,
        engine: _Engine | None = None,
        purchase_executor: _PurchaseExecutor | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        if engine is None:
            from narrator.engine import NarratorEngine

            engine = NarratorEngine(config)
        self.engine = engine
        self._stopping = asyncio.Event()
        self._decisions = DecisionCoordinator()
        self._player_directory = PlayerDirectory(config.campaign_root)
        #: This channel's own decision/consent/withhold state -- see ``_ChannelState``.
        self._channels: dict[str, _ChannelState] = {}
        self._interactions = InteractionTracker()


        self._routing_log: list[dict] = []


        self.post_log: list[TurnPost] = []
        self._social = SocialState()
        self._trade_gate = TradeConfirmationGate()
        # ``scripts/narrator_serve.py`` always supplies this production port.  Tests
        # and alternate hosts without the Black Sword Hack runtime get no quote or
        # mutation path rather than an unsafe narrator-local fallback.
        self._purchase_executor = purchase_executor
        self._romance_policy = RomancePolicy(config.romance_escalation_enabled)


        self._last_fault_category: str = ""

    def record_romance_consent(self, channel_id: str, principal) -> bool:
        """Record one authenticated, process-local consent for an enabled table policy."""
        subject = getattr(principal, "subject_id", "")
        if not isinstance(subject, str) or not subject:
            return False
        self._channels.setdefault(channel_id, _ChannelState()).romance_consents.add(subject)
        return True

    def request_stop(self) -> None:
        self._stopping.set()

    async def _next_turn(self, turns) -> object | None:
        """Wait for one turn or shutdown, cancelling only an idle receive."""
        receive = asyncio.create_task(anext(turns))
        stopping = asyncio.create_task(self._stopping.wait())
        done, pending = await asyncio.wait(
            (receive, stopping), return_when=asyncio.FIRST_COMPLETED
        )

        if stopping in done:
            if not receive.done():
                receive.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive
            return None

        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping
        try:
            return receive.result()
        except StopAsyncIteration:
            return None

    def _retain_recovery(self, turn) -> None:
        """Hold one resumable session for a channel whose notice armed the controls.

        The retained session carries the declaration's own text and zero prior answers.
        A ``/retry`` after an engine-side withhold re-plans the action the player
        actually typed rather than replaying answers that belonged to a round the notice
        already closed. A retained session created inside ``_decision_phase`` is richer
        -- it carries the collected answers and any pending view -- so this never
        overwrites one.

        The session starts from ``mention.text``, never ``channel_text()``, matching
        ``_decision_phase``'s own ``ProgressiveDecisionSession.start``. ``channel_text``
        renders the turn the way the *narrator* reads it: backfill lines first, then the
        mention behind an ``@GM`` marker. An effective action built from that would be
        planned and narrated as ``\"@GM I force the door\"``, and ``run_turn`` would then
        prefix it a second time.
        """
        existing = self._channels.get(turn.channel_id)
        if existing is not None and existing.progressive_session is not None:
            return
        self._channels.setdefault(turn.channel_id, _ChannelState()).progressive_session = (
            ProgressiveDecisionSession.start(turn.mention.text), (), None,
        )

    def _bind_thinking_level(self) -> None:
        """Hand a channel that can set the thinking level the engine's control for it.

        Optional on both sides and discovered with ``getattr``, like ``set_thinking``
        below: a channel without ``bind_thinking_level`` runs at the launch level, and
        an engine without ``set_turn_thinking_level`` (a test double) binds nothing.
        The adapter receives callables, never the engine, so the channel contract --
        inbound mechanics only -- still holds.
        """
        bind = getattr(self.adapter, "bind_thinking_level", None)
        write = getattr(self.engine, "set_turn_thinking_level", None)
        if not callable(bind) or not callable(write):
            return
        from narrator.config import THINKING_LEVELS

        bind(
            ThinkingLevelControl(
                levels=THINKING_LEVELS,
                read=lambda: str(getattr(self.engine, "turn_thinking_level", "off")),
                write=write,
            )
        )

    def _bind_language(self) -> None:
        """Hand a channel that can switch the table's language the control for it.

        Optional on both sides and discovered with ``getattr``, mirroring
        ``_bind_thinking_level`` above. The callables close over the *config* rather
        than the engine, because the live language is session state the config carries
        (``narrator.config.LanguageState``): every notice property and
        ``config.catalog`` read in the process follows a write at call time. The
        adapter still receives callables only -- it reloads its own domain's catalog
        through ``LanguageControl.locale_root`` and reaches nothing else.
        """
        bind = getattr(self.adapter, "bind_language", None)
        write = getattr(self.config, "set_language", None)
        if not callable(bind) or not callable(write):
            return
        from narrator.locale import available_languages

        config = self.config
        bind(
            LanguageControl(
                languages=available_languages(config.locale_root),
                read=lambda: str(config.active_language),
                write=write,
                locale_root=config.locale_root,
            )
        )

    def _bind_mentions(self) -> None:
        """Hand a channel that offers @-mention completion the live roster to read.

        Optional on both sides and discovered with ``getattr``, mirroring
        ``_bind_thinking_level``/``_bind_language`` above. Typing-aid only: the
        candidates this control returns carry no routing meaning, so building the
        roster never touches the classifier or the fiction-debt ledger.
        """
        bind = getattr(self.adapter, "bind_mentions", None)
        if not callable(bind):
            return
        bind(MentionDirectoryControl(candidates=self._mention_candidates))

    def _mention_candidates(self) -> tuple[MentionCandidate, ...]:
        """The current @-mentionable roster: the GM, every linked PC, every living NPC,
        and every scene person -- someone the fiction named who holds no ``NPC``
        record (``bsh_mcp.models.ScenePerson``: a clerk, a fishmonger).

        Reads ``campaign/state.json`` directly rather than through a service call,
        the same precedent ``narrator.ledger`` sets for a read this cheap and this
        far from the fiction-debt path. Fails open on every source: an unreadable or
        unlinked player directory yields no PCs, an unreadable or malformed state
        file yields no NPCs or scene persons, and any one failure still leaves the GM
        candidate and whatever the other sources found.
        """
        candidates: list[MentionCandidate] = [
            MentionCandidate(sigil=self.config.gm_address, kind="gm")
        ]
        try:
            for _character_id, character_name in self._player_directory.snapshot().eligible_characters:
                candidates.append(MentionCandidate(sigil=character_name, kind="pc"))
        except PlayerDirectoryError:
            pass
        import json

        state_path = self.config.campaign_root / "campaign" / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            npcs = state.get("npcs") or {}
            for npc_id, npc in npcs.items():
                if not isinstance(npc, dict) or npc.get("status") != "alive":
                    continue
                name = npc.get("name") or npc_id
                candidates.append(MentionCandidate(sigil=str(name), kind="npc"))
            persons = ((state.get("scene") or {}).get("persons")) or {}
            for person_id, person in persons.items():
                if not isinstance(person, dict):
                    continue
                name = person.get("name") or person_id
                candidates.append(MentionCandidate(sigil=str(name), kind="person"))
        except (OSError, ValueError, AttributeError):
            pass
        return tuple(candidates)

    async def _set_thinking(self, channel_id: str, thinking: bool) -> None:
        """Toggle the channel's own busy indicator, if it has one.

        Optional and best-effort, the same ``getattr`` pattern
        ``narrator.delivery.refresh_status`` uses for ``update_status``: most adapters
        lack ``set_thinking`` entirely, and this carries no campaign state, so it never
        gates or blocks a turn.
        """
        set_thinking = getattr(self.adapter, "set_thinking", None)
        if callable(set_thinking):
            await set_thinking(channel_id, thinking)

    async def _post_notice(self, turn, notice: str, notice_key: str = "") -> None:
        """Post one boundary notice and keep any control it advertises meaningful."""
        self.post_log.append(
            await deliver_no_action(
                self.adapter, turn, notice, self.config.campaign_root, notice_key=notice_key
            )
        )
        if advertises_recovery_controls(notice):
            self._retain_recovery(turn)

    def _record_withhold(self, channel_id: str, notice: str) -> None:
        """Remember the true reason a turn was withheld, for the next turn on record.
        """
        self._channels.setdefault(channel_id, _ChannelState()).withheld_note = notice

    def _fault_notice_for(self, failure_category: str) -> tuple[str, str]:
        """Returns ``(catalog_key, rendered_text)`` so the posted ``TurnPost`` carries
        the key as provenance while the text stays the bytes rendered here.

        \"context\" is `_routed_decision_phase`'s stale-directory-fingerprint site, raised
        when the table's own state moved between a decision's proposal and its answer.
        \"confirmation\" is its stale `PlannerPolicyError` site: an answer arrived for an
        action the session no longer recognises as the one it proposed, which is not
        the same claim as an explicit decline and must not read like one. Every other
        of the method's dozen fault sites -- each of `engine.PLANNER_FAILURE_REASONS`, a
        directory error, a malformed presentation, an unanswered collection -- is a
        genuinely unpreparable proposal and keeps the original, general notice. Those
        planner reasons are split for the diagnostic transcript, not for the wording:
        which of them fired is an operator's question, and none of the answers changes
        what the table can usefully do next.
        """
        return {
            "context": (
                "decision_stale_context", self.config.decision_stale_context_notice
            ),
            "confirmation": (
                "decision_stale_confirmation",
                self.config.decision_stale_confirmation_notice,
            ),
        }.get(failure_category, ("decision_fault", self.config.decision_fault_notice))

    def _record_decision(
        self, *, category: str, kind: str = "", count: int = 0, answer: str = "",
        failure: str = "",
    ) -> None:
        """Record redacted lifecycle semantics without decision text or identifiers."""
        if category == "fault":


            self._last_fault_category = failure
        recorder = getattr(self.adapter, "recorder", None)
        if recorder is None or category not in DECISION_DIAGNOSTIC_CATEGORIES:
            return
        fields = {"category": category}
        if kind:
            fields["decision_kind"] = kind
        if count:
            fields["target_count"] = count
        if answer:
            fields["answer_category"] = answer
        if failure:
            fields["failure_category"] = failure
        recorder.record(
            "decision_lifecycle",
            **{key: value for key, value in fields.items() if key in DECISION_DIAGNOSTIC_ALLOWLIST},
        )

    def _record_tool_events(self, outcome) -> None:
        """Record each executed tool's redacted disposition to the diagnostic transcript.

        The engine already redacted these to name, ``ok``, error code, and event id.
        A disposable campaign deletes its logs at exit, so this is the only retained
        answer to whether a tool ran and whether it rolled. It records nothing more.
        """
        recorder = getattr(self.adapter, "recorder", None)
        if recorder is None:
            return
        for event in getattr(outcome, "tool_events", ()) or ():
            if not isinstance(event, dict):
                continue
            recorder.record(
                "tool_call",
                tool=str(event.get("tool", "")),
                ok=bool(event.get("ok")),
                error=str(event.get("error", "")),
                event_id=event.get("event_id"),
            )

    async def _plan(
        self, planner, turn, eligible, resolutions, session, interaction_cue=None, policy=None,
        retry: bool = False,
    ):
        """Pass progressive state to current engines while retaining legacy adapters.

        ``policy`` is this turn's routing verdict, handed to a current engine so it
        plans on the verdict the turn was routed on instead of classifying the same
        text a second time; an engine that predates it never receives it either.

        ``retry`` says the player already read a no-action notice for this declaration
        and typed ``/retry``. A current engine answers a second planner failure with its
        own floor instead of a second fault; an engine that predates the keyword falls
        back to the behaviour that shipped before it.

        Which of these an engine accepts is read from its own signature rather than
        discovered by calling it and pattern-matching a raised ``TypeError``'s message:
        a genuine bug inside a current engine's ``plan_turn`` -- one that happens to
        raise a ``TypeError`` whose text mentions ``session`` or ``policy`` for reasons
        that have nothing to do with its signature -- used to be indistinguishable from
        \"this engine predates the parameter\" and was silently retried with less
        context instead of surfacing. ``inspect.signature`` decides once, before any
        call is made, so a real failure inside the call still propagates.
        """
        try:
            accepted = inspect.signature(planner).parameters
        except (TypeError, ValueError):
            accepted = {}
        takes_any_keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()
        )
        offered = {
            "session": session, "interaction_cue": interaction_cue,
            "policy": policy, "retry": retry,
        }
        kwargs = {
            name: value
            for name, value in offered.items()
            if takes_any_keyword or name in accepted
        }
        return await planner(turn, eligible, resolutions, **kwargs)

    async def _run_engine(self, turn, resolutions, action_fingerprint):
        """Bind the current action fingerprint without breaking legacy engine adapters."""
        try:
            return await self.engine.run_turn(
                turn,
                decision_resolutions=resolutions,
                decision_action_fingerprint=action_fingerprint,
            )
        except TypeError as error:
            if "decision_action_fingerprint" not in str(error):
                raise
            return await self.engine.run_turn(turn, decision_resolutions=resolutions)

    def _combat_turn_pace_bypass(self, turn) -> bool:
        """Whether an open fight makes this committed declaration the game master's
        to rule directly, with no planner questionnaire round trip.

        Nothing protective is lost, because the planner was never the protective
        layer. Hazard confirmations run on the risk route, ahead of this; social
        consent, trade confirmation, and the dodge/parry binding all route before
        the planner loop; and the mechanics stay engine-owned regardless -- an
        outlandish declaration can be refused or priced by the narrator, but its
        *outcome* can only ever be what audited tool rolls return, and a test rolled
        on the actor's own turn spends a real combat action.

        The bypass binds to the fight's own record: the declaring principal must
        resolve to a character the active fight enrolls on the player side. Every
        miss -- no principal, no readable directory, no fight, a fight that does not
        enroll this character -- falls through to the planner exactly as before.
        """
        principal = turn.mention.principal
        if principal is None:
            return False
        combat = read_combat_snapshot(self.config.campaign_root)
        if not combat.active:
            return False
        try:
            player = self._player_directory.snapshot().for_principal(
                principal.adapter_name, principal.subject_id
            )
        except PlayerDirectoryError:
            return False
        return player is not None and combat.side_of(player.character_id) == "pc"

    def _combat_defence_resolution(
        self, turn: PreparedNarrationTurn, policy: TurnPolicy
    ) -> DecisionResolution | None:
        """Author a bound ``combat_defend`` for a lone dodge or parry inside an open fight.

        The game master narrates the enemy's strike and asks the target to parry or dodge.
        The player's one-word answer is the decision itself. The Standard Reference
        Document fixes the defence space at those two literals. A planner call therefore
        adds latency and one failure mode while contributing no judgment. This binds the
        defence to the authenticated principal's character instead, verified against the
        fight's own combatant list. The answer then resolves through ``combat_defend``,
        rather than reaching the planner that once answered that no active threat existed.

        It returns None whenever a binding premise is missing. The premises are a present
        principal, an open fight, a declaration carrying nothing besides the defence, a
        readable player directory, and a character the fight records on the player side.
        Each miss falls through to the routing this method precedes. That is the behaviour
        which shipped before it.

        Membership reads ``side_of`` rather than the order line, because the side list is
        what names a combatant. A render missing its order line would otherwise refuse a
        defence the fight permits.
        """
        principal = turn.mention.principal
        if principal is None:
            return None
        combat = read_combat_snapshot(self.config.campaign_root)
        # The bound defence comes from this turn's own classification rather
        # than from a re-read of the text. ``defence_method`` matched the two
        # English literals and subtracted a stop-word set to insist the
        # declaration carried nothing else; the classifier answers the same
        # question in any language, and the fight membership check below is
        # unchanged and still what authorises the binding.
        method = policy.defence_method
        if not method or not combat.active:
            return None
        try:
            player = self._player_directory.snapshot().for_principal(
                principal.adapter_name, principal.subject_id
            )
        except PlayerDirectoryError:
            return None
        if player is None or combat.side_of(player.character_id) != "pc":
            return None
        return DecisionResolution(
            character_id=player.character_id,
            kind="approach",
            answer_kind="option",
            selection_id="defend",
            directive=CombatDefendDirective(
                kind="combat_defend", character_id=player.character_id, method=method
            ),
        )

    def _offer_is_open(self, channel_id: str) -> bool:
        """Whether this channel holds a trade frame still advancing toward a purchase.

        Read before classification and passed as a bare flag. Failure is silent and
        false: an unreadable scope or an absent frame simply means the classifier
        judges the message without the hint, which is the state every non-trade turn
        is in anyway.
        """
        try:
            scope = read_trusted_scope(self.config.campaign_root)
        except Exception:  # noqa: BLE001 - an unreadable scope is not a trade frame
            return False
        frame = self._social.trade_frame(channel_id, scope)
        return frame is not None and frame.phase in ADVANCING_TRADE_PHASES

    async def _classify(self, player_text: str, *, scope=None, combat=None, offer_open: bool = False):
        """This turn's routing verdict, or ``None`` when the classifier cannot answer.

        Duck-typed like ``assess_hazard`` above, and for the same reason: an engine is
        whatever the channel handed this service. The difference is what an absent
        capability means. A missing ``assess_hazard`` leaves a floor standing, so it
        returns "no downgrade" and play continues; a missing ``classify_intent`` leaves
        nothing standing at all, because the English word lists that used to answer this
        question were deleted rather than kept as a fallback. So this returns ``None``
        and the caller withholds the turn.

        That is the intended offline contract too: a fake engine that does not implement
        ``classify_intent`` reaches no endpoint and routes no turn, which makes the
        dependency loud instead of letting a suite silently exercise a path production
        cannot reach.
        """
        classify = getattr(self.engine, "classify_intent", None)
        if not callable(classify):
            return None
        try:
            return await classify(
                player_text, scope=scope, combat=combat, offer_open=offer_open
            )
        except Exception:  # noqa: BLE001 - every classifier fault withholds the turn
            return None

    async def _hazard_assessment_downgrades(self, turn, policy: TurnPolicy) -> bool:
        """Whether the typed speech-act assessment removes this turn's risk floor.

        Duck-typed like every other engine capability this service consumes: an
        engine without ``assess_hazard`` (every pre-existing fake in the offline
        suites) keeps the lexical floor unchanged, and so does any assessor fault.
        A downgrade is recorded to the diagnostic transcript with the assessed kind,
        because a silently vanished confirmation is the one thing an operator
        reading a session log could not otherwise explain.
        """
        assess = getattr(self.engine, "assess_hazard", None)
        if not callable(assess):
            return False
        try:
            assessment = await assess(
                turn.mention.text,
                scope=policy.scope,
                combat=read_combat_snapshot(self.config.campaign_root),
            )
        except Exception:  # noqa: BLE001 - every assessor fault keeps the confirmation
            return False
        if not assessment_downgrades(assessment):
            return False
        self._record_decision(category="assessed", kind=assessment.kind)
        return True

    async def _classify_turn_policy(self, turn) -> TurnPolicy | None:
        """This channel's routing verdict for ``turn.mention.text``, or ``None``.

        The one classification call the service makes. ``run`` uses it for a freshly
        typed turn and ``_decision_phase`` uses it for the declaration a recovery
        control retries, so both reach the classifier with the same context -- the
        channel's own open-offer flag bound into the prompt *and* handed to
        ``policy_from`` -- rather than one of them running a thinner call.

        Whether an offer is open is service state this class owns, and the classifier
        needs it: without it "I take it" reads as acceptance idiomatically in English
        and German and not in French, Japanese or Russian, the model correctly noting
        that no offer was mentioned. It is the same fact the retired
        ``social._trade_phase`` took as ``negotiating=True``, and it reaches the prompt
        as a bare flag naming no goods -- naming a seller's stock there was measured
        turning a theft into commerce.
        """
        offer_open = self._offer_is_open(turn.channel_id)
        policy = await self._interactions.policy(
            turn.channel_id,
            turn.mention.text,
            self.config.campaign_root,
            classify=partial(self._classify, offer_open=offer_open),
            offer_open=offer_open,
        )
        self._routing_log.append(_routing_row(policy))
        return policy

    def _consume_recovery_control(self, turn) -> tuple[str, tuple | None]:
        """Take this turn's recovery control and the session it acts on.
        """
        consume = getattr(self.adapter, "consume_decision_recovery_control", None)
        control = consume() if callable(consume) else ""
        channel = self._channels.get(turn.channel_id)
        retained = channel.progressive_session if channel is not None else None
        if control and channel is not None:
            channel.progressive_session = None
        return control, retained

    async def _decision_phase(
        self, turn: InboundTurn, policy: TurnPolicy
    ) -> tuple[
        RoutingPhase, tuple[DecisionResolution, ...], InboundTurn, TurnPolicy | None
    ]:
        """Adopt any recovery control, then plan, gate, present, and collect decisions.

        Returns the policy the rest of the turn must read alongside the phase, because
        a recovery control replaces it. ``run`` classified whatever text the channel
        yielded, and for ``/retry`` and ``/continue`` that text is the control token
        itself, not the declaration being retried. Every route-dependent gate in
        ``_routed_decision_phase`` -- the enemy-turn defence, the risk floor, the trade
        and social lanes, the bound defence -- used to run against that token's verdict,
        and so did ``run``'s own downstream reads (turn framing, the social test
        request, the romance gate, the trade offer, the ``meta`` flag). A control token
        that happened to classify ``read`` therefore bypassed the hazard floor outright:
        a violent declaration could narrate on retry with no confirmation at all.

        So the control is consumed and the retained declaration re-classified *here*,
        ahead of every one of those gates, and the fresh verdict is threaded back out.
        ``/revise`` is deliberately left alone: the channel yields the revised text as
        the mention, so ``run`` already classified exactly the action being planned.

        This spends a second classification on a ``/retry`` or ``/continue`` turn, since
        ``run`` has already classified the token by the time the control can be
        discovered -- the channel contract offers no way to read a pending control
        without consuming it. The cost is one model call on a turn that only happens
        after the table has already read a no-action notice, which is the cheapest
        moment in a session to spend one.
        """
        control, retained = self._consume_recovery_control(turn)
        if not control:
            return (*await self._routed_decision_phase(turn, policy, None), policy)
        if retained is None:
            # A channel must not turn an orphaned recovery command into narration.
            return "pending", (), turn, policy
        session, resolutions, pending_draft = retained
        if control == "dismiss":
            self._record_decision(category="dismissed")
            return "pending", (), turn, policy
        if control == "retry":
            # ``/retry`` re-plans from the retained action; ``/continue`` resumes the
            # pending view below without a planner call. Discarding the pending draft
            # here is what makes the two controls genuinely different rather than two
            # spellings of one behaviour.
            pending_draft = None
        if control == "revise":
            try:
                session = session.revised(turn.mention.text)
            except Exception:  # noqa: BLE001 - invalid revision never reaches narration
                self._record_decision(category="fault", failure="revision")
                return "fault", (), turn, policy
            # A revised action replaces the one the pending view was about, and
            # ``revised`` already dropped every answer and closed dimension with it.
            resolutions, pending_draft = (), None
        # Controls are transport syntax.  Both planner and narrator see the retained
        # effective action, never a literal ``/continue`` or ``/retry`` command.
        turn = InboundTurn(
            channel_id=turn.channel_id,
            mention=ChannelMessage(
                turn.mention.author,
                session.effective_action,
                principal=turn.mention.principal,
            ),
            backfill=turn.backfill,
        )
        if control in {"continue", "retry"}:
            policy = await self._classify_turn_policy(turn)
            if policy is None:
                # Fail closed, and closed here means the turn is withheld outright
                # rather than merely floored: the engine does not know what the
                # retained declaration does, so it resolves nothing. That is strictly
                # stronger than the floor ``requires_risk_confirmation(None)`` would
                # have applied deeper in, and it is the same answer ``run`` already
                # gives a freshly typed turn the classifier cannot read. The session
                # is put back so the control the notice advertises still has something
                # to act on.
                self._channels.setdefault(
                    turn.channel_id, _ChannelState()
                ).progressive_session = (session.next_segment(), resolutions, pending_draft)
                return "classifier_fault", (), turn, None
        recovery = _Recovery(session, resolutions, pending_draft, control)
        return (*await self._routed_decision_phase(turn, policy, recovery), policy)

    async def _routed_decision_phase(
        self,
        turn: InboundTurn,
        policy: TurnPolicy,
        recovery: _Recovery | None,
    ) -> tuple[RoutingPhase, tuple[DecisionResolution, ...], InboundTurn]:
        """Plan, gate, present, and collect decisions before narrator execution."""


        if policy.route not in {"read", "out_of_character", "gm"} and not policy.defence_method:
            open_turn = open_npc_turn(self.config.campaign_root)
            if open_turn is not None:
                phase, resolutions, execution_turn = await self._enemy_turn_defence_phase(
                    turn, policy, open_turn
                )
                if resolutions:
                    return phase, resolutions, execution_turn
        if policy.route == "risk":


            if combat_sanctions_violence(
                policy, read_combat_snapshot(self.config.campaign_root)
            ):
                return "continue", (), turn
            # The lexical risk trigger cannot tell a declaration from a report about
            # one. Before any confirmation machinery runs -- including the
            # capability short-circuit below, so a channel that cannot present
            # decisions is not withheld over a message that declares nothing -- the
            # typed assessor is asked the one open question: is this message a
            # declaration at all? Every assessor failure keeps the floor exactly as
            # it was.
            if await self._hazard_assessment_downgrades(turn, policy):
                return "continue", (), turn
            capabilities = structured_decision_capabilities(self.adapter)
            if (
                self.config.max_decision_rounds <= 0
                or not capabilities.supports_structured_decisions()
                or not callable(getattr(self.engine, "plan_turn", None))
                or turn.mention.principal is None
            ):
                return "risk_confirmation_required", (), turn


        if policy.route in {"social", "planner"} and self._advancing_purchase(turn, policy) is not None:
            return await self._trade_confirmation_phase(turn, policy)
        if policy.route == "social" and policy.social_test_required:
            return await self._social_choice_phase(turn, policy)
        if policy.route in {"social", "read", "out_of_character", "gm"}:
            return "continue", (), turn
        # A lone dodge or parry inside an open fight binds its defence here.
        #
        # ``defence_method`` holds the gate that keeps a violent compound out, through its
        # residual-word subtraction. This route test is a second barrier, and it is
        # currently redundant. Enumerating one defence word against every subset of two
        # filler words yields nothing the subtraction admits and this test then rejects.
        # Do not read it as the thing protecting the risk floor. Widening
        # ``_DEFENCE_INTENT_WORDS`` toward a hazard verb would breach the floor whatever
        # this line says, and only then would this test begin to carry weight.
        #
        # The binding needs neither a planner nor a structured-decision channel, so it
        # stands ahead of both capability gates.
        if policy.route == "planner":
            defence = self._combat_defence_resolution(turn, policy)
            if defence is not None:
                return "continue", (defence,), turn
        if self.config.max_decision_rounds <= 0:
            return "continue", (), turn
        if not structured_decision_capabilities(self.adapter).supports_structured_decisions():
            return "continue", (), turn
        planner = getattr(self.engine, "plan_turn", None)
        if not callable(planner):
            return "continue", (), turn
        principal = turn.mention.principal
        if principal is None:
            self._record_decision(category="fault", failure="principal_absent")
            return "fault", (), turn
        if recovery is None:


            retained_channel = self._channels.get(turn.channel_id)
            if retained_channel is not None:
                retained_channel.progressive_session = None
            session, resolutions, pending_draft = (
                ProgressiveDecisionSession.start(turn.mention.text), (), None,
            )
        else:
            # ``_decision_phase`` already consumed the control, adopted the retained
            # session, rebuilt this turn around its effective action, and -- for
            # ``/retry`` and ``/continue`` -- re-classified that action, which is the
            # ``policy`` every gate above judged. Nothing is re-derived here.
            session = recovery.session
            resolutions = recovery.resolutions
            pending_draft = recovery.pending_draft
        # Session zero: an authenticated principal who owns no character yet has
        # nothing a structured decision could bind to -- every prepared view names a
        # character id, and audience authorization would fault on an empty directory.
        # The turn is conversation with the game master (boundaries, origin and
        # background choices, ``character_create``), so it proceeds straight to the
        # narrator. This runs after the recovery-control adoption above so a typed
        # ``/retry`` or ``/revise`` still resolves to its effective action rather
        # than leaking transport syntax into narration. An unreadable directory is a
        # different thing entirely and still fails closed, exactly as the planner
        # path always has.
        try:
            if (
                self._player_directory.snapshot().for_principal(
                    principal.adapter_name, principal.subject_id
                )
                is None
            ):
                return "continue", (), turn
        except PlayerDirectoryError:
            # Mirrors ``retain_for_recovery`` below, which is not in scope yet: the
            # session is held so the fault notice's recovery controls have something
            # to act on.
            self._channels.setdefault(turn.channel_id, _ChannelState()).progressive_session = (
                session.next_segment(), resolutions, None,
            )
            self._record_decision(category="fault", failure="directory")
            return "fault", (), turn
        original_turn = turn.channel_text()

        def retain_for_recovery(pending=None) -> None:
            """Hold this channel's session, its answers, and any unpresented view.

            ``pending`` is the draft a view boundary refused to present, which
            ``/continue`` resumes without a planner call. Every other caller is a fault:
            nothing was pending, so a control re-plans.
            """
            self._channels.setdefault(turn.channel_id, _ChannelState()).progressive_session = (
                session.next_segment(), resolutions, pending,
            )

        def budget_boundary(decision, *, failure: str, kind: str = "", count: int = 0):
            """End a segment at a view boundary without ever discarding an answer.

            A refused *narrowing* view (clarification or approach) is a request for more
            precision, so the turn runs on the precision it already has: the collected
            answers reach the engine in the same turn. A refused ``ConfirmationDraft`` is
            a gate, not a narrowing question, and proceeding past one would run an
            unconfirmed hazard, so it still ends the segment -- but it ends it holding
            both the answers and the confirmation itself, which ``/continue`` then
            resumes rather than re-planning.
            """
            if resolutions and not isinstance(decision, ConfirmationDraft):
                return "continue", resolutions, turn
            retain_for_recovery(decision)
            self._record_decision(
                category="segment", kind=kind, count=count, failure=failure
            )
            return "segment", (), turn

        while True:
            try:
                if pending_draft is not None:


                    plan = PlanOutcome(
                        plan=RequestDecisionPlan(kind="request", decision=pending_draft)
                    )
                    pending_draft = None
                    snapshot = self._player_directory.snapshot()
                elif policy.route == "risk":


                    plan = await verify_plan(
                        PlanOutcome(plan=ProceedPlan(kind="proceed")),
                        session.effective_action,
                        session,
                        combat=read_combat_snapshot(self.config.campaign_root),
                        scope=policy.scope,
                        classify=self._classify,
                        policy=policy,
                    )
                elif self._combat_turn_pace_bypass(turn):


                    plan = PlanOutcome(plan=ProceedPlan(kind="proceed"))
                else:
                    snapshot = self._player_directory.snapshot()
                    plan = await self._plan(
                        planner, turn, snapshot.eligible_characters, resolutions, session,
                        interaction_cue=policy.interaction_cue, policy=policy,


                        retry=recovery is not None and recovery.control == "retry",
                    )
            except PlayerDirectoryError:
                retain_for_recovery()
                self._record_decision(category="fault", failure="directory")
                return "fault", (), turn
            except DecisionPlanningError as error:


                retain_for_recovery()
                self._record_decision(
                    category="fault", failure=getattr(error, "reason", "") or "planner"
                )
                return "fault", (), turn
            except Exception:  # noqa: BLE001 - planner faults never invoke narrator
                retain_for_recovery()
                self._record_decision(category="fault", failure="planner")
                return "fault", (), turn
            if isinstance(plan, DeclinedPlan):
                return "declined", (), turn
            if isinstance(plan.plan, ProceedPlan):
                return "continue", resolutions, turn
            if repeats_answered_question(session, plan.plan.decision):


                return "continue", resolutions, turn
            if session.completed_views >= self.config.max_decision_rounds:
                return budget_boundary(plan.plan.decision, failure="view_limit")
            try:
                if policy.route == "risk":
                    snapshot = self._player_directory.snapshot()
                session = session.preparing(plan.plan.decision)
                fingerprint = self._player_directory.context_fingerprint(
                    snapshot=snapshot,
                    original_turn=original_turn,
                    prior_resolutions=resolutions,
                )
                prepared, views = await self._decisions.prepare(
                    draft=plan.plan.decision,
                    snapshot=snapshot,
                    principal=principal,
                    context_fingerprint=fingerprint,
                )
            except (DecisionCoordinatorError, PlayerDirectoryError):
                retain_for_recovery()
                self._record_decision(category="fault", failure="audience")
                return "fault", (), turn
            remaining_views = self.config.max_decision_rounds - session.completed_views
            if len(views) > remaining_views:
                await self._decisions.cancel(prepared.request_id, reason="facilitator")
                return budget_boundary(
                    plan.plan.decision, failure="view_capacity",
                    kind=prepared.kind, count=len(views),
                )
            delivery = await present_decisions(self.adapter, views, self.config.campaign_root)
            if delivery.status != "presented":
                await self._decisions.cancel(prepared.request_id, reason="facilitator")
                retain_for_recovery()
                self._record_decision(
                    category="fault", kind=prepared.kind, count=len(views),
                    failure=delivery.failure_category,
                )
                return "fault", (), turn
            if not await self._decisions.activate(prepared.request_id):
                await self._decisions.cancel(prepared.request_id, reason="facilitator")
                retain_for_recovery()
                self._record_decision(category="fault", failure="activation")
                return "fault", (), turn
            self._record_decision(category="presented", kind=prepared.kind, count=len(views))
            while not await self._decisions.is_complete(prepared.request_id):
                collected, stopped = await self._collect_decision_submission(views)
                if stopped:
                    await self._decisions.cancel(prepared.request_id, reason="shutdown")
                    self._record_decision(category="cancelled", kind=prepared.kind)
                    return "stopped", (), turn
                if isinstance(collected, DecisionCollectionCancelled):
                    if collected.reason == "dismissed":
                        await self._decisions.cancel(prepared.request_id, reason="dismissed")
                        self._record_decision(category="dismissed", kind=prepared.kind)
                        # The terminal's normal turn iterator now resumes.  A dismissed
                        # request is tombstoned and can never reopen on a later input.
                        return "pending", (), turn
                    await self._decisions.cancel(prepared.request_id, reason="shutdown")
                    self.request_stop()
                    self._record_decision(category="cancelled", kind=prepared.kind)
                    return "stopped", (), turn
                if collected is None:
                    # Structured adapters must make non-answers explicit.  Failing
                    # closed prevents a lost response from leaving a live request.
                    await self._decisions.cancel(prepared.request_id, reason="facilitator")
                    retain_for_recovery()
                    self._record_decision(category="fault", failure="collection_disposition")
                    return "fault", (), turn
                responder, submission = collected
                try:
                    current = self._player_directory.context_fingerprint(
                        snapshot=self._player_directory.snapshot(),
                        original_turn=original_turn,
                        prior_resolutions=resolutions,
                    )
                except PlayerDirectoryError:
                    await self._decisions.cancel(prepared.request_id, reason="stale")
                    retain_for_recovery()
                    self._record_decision(category="fault", failure="context")
                    return "fault", (), turn
                result = await self._decisions.submit(
                    principal=responder,
                    submission=submission,
                    context_fingerprint=current,
                )
                if result.status in {"accepted", "completed"}:
                    answer_category = (
                        submission.selection_id
                        if submission.selection_id in {"confirm", "decline"}
                        else "custom" if submission.selection_id == "own_approach" else "option"
                    )
                    self._record_decision(
                        category="answered", kind=prepared.kind,
                        count=len(views), answer=answer_category,
                    )
                    continue
                if result.status == "idempotent":
                    # A matching retry confirms an earlier answer but cannot occupy
                    # another target's collection slot.
                    self._record_decision(category="retry", kind=prepared.kind)
                    continue
                await self.adapter.acknowledge_decision(result)
                if result.status in {"invalid", "conflict"}:
                    continue
                self._record_decision(
                    category="fault", kind=prepared.kind,
                    failure=result.failure_category or result.status,
                )
                retain_for_recovery()
                return "fault", (), turn
            answered = await self._decisions.resolutions(prepared.request_id)
            if not answered:
                retain_for_recovery()
                self._record_decision(category="fault", failure="incomplete")
                return "fault", (), turn
            try:
                session = session.accept(plan.plan.decision, answered)
            except Exception:  # noqa: BLE001 - stale confirmation never reaches narration
                retain_for_recovery()
                self._record_decision(category="fault", failure="confirmation")
                return "fault", (), turn
            if any(item.answer_kind == "custom" for item in answered):
                resolutions = answered
            else:
                resolutions = resolutions + answered
            resolutions = tuple(
                item.model_copy(update={"action_fingerprint": session.action_fingerprint})
                for item in resolutions
            )

    def _advancing_purchase(self, turn, policy: TurnPolicy) -> TradeFrame | None:
        """The live frame this turn's declaration could actually advance to a purchase.

        Returning ``None`` sends the turn to ordinary handling with no notice posted,
        which is the correct answer for a channel that is not mid-purchase.
        """
        frame = self._social.trade_frame(turn.channel_id, policy.scope)
        if frame is None or frame.phase not in ADVANCING_TRADE_PHASES:
            return None
        # ``accepts_offer`` is the language fact; the advancing frame found just above
        # is the context fact. The retired lexicon folded both into one call by taking
        # ``negotiating=True`` on trust; keeping them apart is what stops a bare "take"
        # meaning a purchase on a channel with no open offer.
        return frame if (
            policy.trade_phase == TradePhase.PURCHASE_INTENT.value or policy.accepts_offer
        ) else None

    async def _trade_confirmation_phase(
        self, turn: PreparedNarrationTurn, policy: TurnPolicy
    ) -> tuple[RoutingPhase, tuple[DecisionResolution, ...], PreparedNarrationTurn]:
        """Present and settle one exact purchase confirmation before any debit occurs.

        The early return below still reports a genuinely stalled negotiation -- a frame
        the gate found advancing that ``advance_trade`` could not carry to
        ``PURCHASE_INTENT``, such as one abandoned at ``CONFIRMATION``. It is no longer
        reachable from an absent frame, because ``_advancing_purchase`` above refuses to
        route a turn here without one.
        """
        frame = self._social.advance_trade(
            turn.channel_id, scope=policy.scope, policy=policy
        )
        if frame is None or frame.phase is not TradePhase.PURCHASE_INTENT:
            return "trade_confirmation_required", (), turn
        principal = turn.mention.principal
        if (
            principal is None
            or self._purchase_executor is None
            or self.config.max_decision_rounds <= 0
            or not structured_decision_capabilities(self.adapter).supports_structured_decisions()
        ):
            return "trade_confirmation_required", (), turn
        try:
            snapshot = self._player_directory.snapshot()
            player = snapshot.for_principal(principal.adapter_name, principal.subject_id)
            if player is None:
                return "trade_confirmation_required", (), turn
            action_fingerprint = semantic_fingerprint(
                {
                    "channel": turn.channel_id,
                    "trade": frame.context_fingerprint,
                    "actor": player.character_id,
                }
            )
            draft = ConfirmationDraft(
                kind="confirmation",
                question="Confirm this purchase?",
                context=(
                    f"{frame.terms.quantity} {frame.terms.public_label} for "
                    f"{frame.agreed_price or frame.terms.price} copper."
                ),
                audience=CurrentAudience(kind="current"),
            )
            context = self._player_directory.context_fingerprint(
                snapshot=snapshot, original_turn=turn.mention.text, prior_resolutions=()
            )
            prepared, views = await self._decisions.prepare(
                draft=draft, snapshot=snapshot, principal=principal, context_fingerprint=context
            )
        except (DecisionCoordinatorError, PlayerDirectoryError):
            return "trade_confirmation_required", (), turn
        opened = self._social.begin_confirmation(
            turn.channel_id, scope=policy.scope, decision_key=prepared.request_id
        )
        if opened is None or not self._trade_gate.open(
            decision_key=prepared.request_id,
            actor_id=player.character_id,
            action_fingerprint=action_fingerprint,
            frame=opened,
        ):
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            return "trade_confirmation_required", (), turn
        delivery = await present_decisions(self.adapter, views, self.config.campaign_root)
        if delivery.status != "presented" or not await self._decisions.activate(prepared.request_id):
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            self._social.cancel(turn.channel_id)
            self._trade_gate.close(prepared.request_id)
            return "trade_confirmation_required", (), turn
        self._record_decision(category="presented", kind="confirmation", count=len(views))
        while not await self._decisions.is_complete(prepared.request_id):
            collected, stopped = await self._collect_decision_submission(views)
            if stopped or isinstance(collected, DecisionCollectionCancelled) or collected is None:
                await self._decisions.cancel(
                    prepared.request_id,
                    reason="shutdown" if stopped else "facilitator",
                )
                self._social.cancel(turn.channel_id)
                self._trade_gate.close(prepared.request_id)
                return ("stopped" if stopped else "trade_confirmation_required"), (), turn
            responder, submission = collected
            try:
                current = self._player_directory.context_fingerprint(
                    snapshot=self._player_directory.snapshot(), original_turn=turn.mention.text,
                    prior_resolutions=(),
                )
            except PlayerDirectoryError:
                await self._decisions.cancel(prepared.request_id, reason="stale")
                self._social.cancel(turn.channel_id)
                self._trade_gate.close(prepared.request_id)
                return "trade_confirmation_required", (), turn
            result = await self._decisions.submit(
                principal=responder, submission=submission, context_fingerprint=current
            )
            if result.status in {"accepted", "completed", "idempotent"}:
                continue
            if result.status in {"invalid", "conflict"}:
                await self.adapter.acknowledge_decision(result)
                continue
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            self._social.cancel(turn.channel_id)
            self._trade_gate.close(prepared.request_id)
            return "trade_confirmation_required", (), turn
        resolutions = await self._decisions.resolutions(prepared.request_id)
        if not resolutions or resolutions[0].selection_id != "confirm":
            self._social.cancel(turn.channel_id)
            self._trade_gate.close(prepared.request_id)
            return "trade_declined", (), turn

        def purchase(current_frame, actor_id: str, current_fingerprint: str) -> str:
            from bsh_mcp.models import EnginePurchaseRequest

            price = current_frame.agreed_price
            request = EnginePurchaseRequest(
                decision_key=prepared.request_id,
                action_fingerprint=current_fingerprint,
                trade_fingerprint=current_frame.context_fingerprint,
                idempotency_key=semantic_fingerprint(
                    {"decision": prepared.request_id, "trade": current_frame.context_fingerprint}
                ),
                buyer_id=actor_id,
                seller_id=current_frame.terms.seller_id,
                item_id=current_frame.terms.item_id,
                quantity=current_frame.terms.quantity,
                price_copper=current_frame.terms.price if price is None else price,
                stock_version=current_frame.terms.stock_version,
            )
            result = self._purchase_executor.confirm(request)
            if result.get("ok") is True:
                return str(result.get("outcome", ""))
            return "failed"

        disposition = self._social.confirm_purchase(
            turn.channel_id,
            scope=policy.scope,
            decision_key=prepared.request_id,
            actor_id=player.character_id,
            action_fingerprint=action_fingerprint,
            purchase=purchase,
            gate=self._trade_gate,
        )
        phase = "trade_completed" if disposition in {"confirmed", "idempotent"} else "trade_confirmation_required"
        return phase, (), turn

    async def _social_choice_phase(
        self, turn: PreparedNarrationTurn, policy: TurnPolicy
    ) -> tuple[RoutingPhase, tuple[DecisionResolution, ...], PreparedNarrationTurn]:
        """Bind an attribute through typed UI when a social test has real alternatives."""
        request = self._social_test_request(turn, policy)
        if request is None or len(request.allowed_attributes) < 2:
            return "continue", (), turn
        principal = turn.mention.principal
        if (
            principal is None
            or self.config.max_decision_rounds <= 0
            or not structured_decision_capabilities(self.adapter).supports_structured_decisions()
        ):
            return "social_choice_required", (), turn
        try:
            snapshot = self._player_directory.snapshot()
            context = self._player_directory.context_fingerprint(
                snapshot=snapshot,
                original_turn=turn.mention.text,
                prior_resolutions=(),
            )
            draft = ApproachDraft(
                kind="approach",
                question="Choose the social approach.",
                context="This action needs one bound attribute test.",
                audience=CurrentAudience(kind="current"),
                approaches=tuple(
                    {
                        "id": f"attribute_{attribute.lower()}",
                        "label": f"Use {attribute}.",
                        "directive": {"kind": "attribute_test", "attribute": attribute},
                    }
                    for attribute in request.allowed_attributes
                ),
            )
            prepared, views = await self._decisions.prepare(
                draft=draft,
                snapshot=snapshot,
                principal=principal,
                context_fingerprint=context,
            )
        except (DecisionCoordinatorError, PlayerDirectoryError, ValueError):
            return "social_choice_required", (), turn
        delivery = await present_decisions(self.adapter, views, self.config.campaign_root)
        if delivery.status != "presented" or not await self._decisions.activate(prepared.request_id):
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            return "social_choice_required", (), turn
        self._record_decision(category="presented", kind="approach", count=len(views))
        while not await self._decisions.is_complete(prepared.request_id):
            collected, stopped = await self._collect_decision_submission(views)
            if stopped:
                await self._decisions.cancel(prepared.request_id, reason="shutdown")
                return "stopped", (), turn
            if isinstance(collected, DecisionCollectionCancelled) or collected is None:
                await self._decisions.cancel(prepared.request_id, reason="facilitator")
                return "social_choice_required", (), turn
            responder, submission = collected
            try:
                current = self._player_directory.context_fingerprint(
                    snapshot=self._player_directory.snapshot(),
                    original_turn=turn.mention.text,
                    prior_resolutions=(),
                )
            except PlayerDirectoryError:
                await self._decisions.cancel(prepared.request_id, reason="stale")
                return "social_choice_required", (), turn
            result = await self._decisions.submit(
                principal=responder,
                submission=submission,
                context_fingerprint=current,
            )
            if result.status in {"accepted", "completed", "idempotent"}:
                continue
            if result.status in {"invalid", "conflict"}:
                await self.adapter.acknowledge_decision(result)
                continue
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            return "social_choice_required", (), turn
        resolutions = await self._decisions.resolutions(prepared.request_id)
        if len(resolutions) != 1 or resolutions[0].directive is None:
            return "social_choice_required", (), turn
        bound = resolutions[0].model_copy(
            update={
                "action_fingerprint": semantic_fingerprint(
                    {
                        "channel": turn.channel_id,
                        "category": request.category.value,
                        "attribute": resolutions[0].directive.attribute,
                    }
                )
            }
        )
        return "continue", (bound,), turn

    async def _enemy_turn_defence_phase(
        self, turn: PreparedNarrationTurn, policy: TurnPolicy, open_turn
    ) -> tuple[RoutingPhase, tuple[DecisionResolution, ...], PreparedNarrationTurn]:
        """Present the fixed dodge/parry choice for an enemy's own open combat turn.

        ``open_turn`` (``narrator.interactions.OpenNpcTurn``) is already resolved by
        the caller. This never blocks the turn on a missing capability: every early
        exit returns empty resolutions, which tells ``_decision_phase`` to fall
        through to its own ordinary routing exactly as if this check had never run --
        the new capability is additive, never a new way to withhold a turn a channel
        cannot present a decision on.

        Structurally the same non-model-planned shape ``_social_choice_phase`` uses
        for its own fixed choice: no ``plan_turn`` call and no custom-text option,
        because the Standard Reference Document fixes the defence space at exactly
        two literals, the same reason ``_combat_defence_resolution`` needs neither.
        """
        principal = turn.mention.principal
        if principal is None:
            return "continue", (), turn
        try:
            snapshot = self._player_directory.snapshot()
            player = snapshot.for_principal(principal.adapter_name, principal.subject_id)
        except PlayerDirectoryError:
            return "continue", (), turn
        combat = read_combat_snapshot(self.config.campaign_root)
        if player is None or combat.side_of(player.character_id) != "pc":
            return "continue", (), turn
        if (
            self.config.max_decision_rounds <= 0
            or not structured_decision_capabilities(self.adapter).supports_structured_decisions()
        ):
            return "continue", (), turn
        try:
            context = self._player_directory.context_fingerprint(
                snapshot=snapshot, original_turn=turn.mention.text, prior_resolutions=(),
            )
            draft = enemy_turn_defence(open_turn.name, player.character_name, self.config.catalog)
            prepared, views = await self._decisions.prepare(
                draft=draft, snapshot=snapshot, principal=principal, context_fingerprint=context,
            )
        except (DecisionCoordinatorError, PlayerDirectoryError, ValueError):
            return "continue", (), turn
        delivery = await present_decisions(self.adapter, views, self.config.campaign_root)
        if delivery.status != "presented" or not await self._decisions.activate(prepared.request_id):
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            return "continue", (), turn
        self._record_decision(category="presented", kind="approach", count=len(views))
        while not await self._decisions.is_complete(prepared.request_id):
            collected, stopped = await self._collect_decision_submission(views)
            if stopped:
                await self._decisions.cancel(prepared.request_id, reason="shutdown")
                return "stopped", (), turn
            if isinstance(collected, DecisionCollectionCancelled) or collected is None:
                await self._decisions.cancel(prepared.request_id, reason="facilitator")
                return "continue", (), turn
            responder, submission = collected
            try:
                current = self._player_directory.context_fingerprint(
                    snapshot=self._player_directory.snapshot(),
                    original_turn=turn.mention.text, prior_resolutions=(),
                )
            except PlayerDirectoryError:
                await self._decisions.cancel(prepared.request_id, reason="stale")
                return "continue", (), turn
            result = await self._decisions.submit(
                principal=responder, submission=submission, context_fingerprint=current,
            )
            if result.status in {"accepted", "completed", "idempotent"}:
                continue
            if result.status in {"invalid", "conflict"}:
                await self.adapter.acknowledge_decision(result)
                continue
            await self._decisions.cancel(prepared.request_id, reason="facilitator")
            return "continue", (), turn
        resolutions = await self._decisions.resolutions(prepared.request_id)
        if len(resolutions) != 1 or resolutions[0].directive is None:
            return "continue", (), turn
        return "continue", resolutions, turn

    def _social_test_request(self, turn, policy: TurnPolicy, resolutions=()) -> SocialTestRequest | None:
        """Build a threshold-approved, actor-bound mechanic before the narrator runs."""
        if not policy.social_test_required or not policy.social_category:
            return None
        principal = turn.mention.principal
        if principal is None:
            return None
        try:
            player = self._player_directory.snapshot().for_principal(
                principal.adapter_name, principal.subject_id
            )
        except PlayerDirectoryError:
            return None
        if player is None:
            return None
        try:
            category = SocialCategory(policy.social_category)
        except ValueError:
            return None
        # ``social_test_needed`` read three fields off a re-classification of the same
        # text -- ``is_rules_inquiry``, ``category`` and ``requires_test`` -- and this
        # method has already established all three from the turn's own policy: the
        # guard above returns early without ``social_test_required`` or a category, and
        # a rules inquiry carries neither. What is left of the threshold is feasibility,
        # which is whether there is anyone to test against.
        if policy.interaction_cue is None:
            return None
        substitutions: tuple[tuple[str, str], ...] = ()
        if category is SocialCategory.DECEPTION:
            substitutions = (("sophist_lie", "CHA"), ("bookworm_substitution", "INT"))
        elif category in {SocialCategory.DEBATE, SocialCategory.INSIGHT}:
            substitutions = (("bookworm_substitution", "INT"),)
        selected = None
        if resolutions:
            directive = getattr(resolutions[0], "directive", None)
            selected = getattr(directive, "attribute", None)
            if selected not in social_attributes(category):
                return None
        return SocialTestRequest(
            category=category,
            actor_id=player.character_id,
            objective="Resolve the stated bounded social objective.",
            failure_stakes="The stated objective does not gain its requested effect.",
            allowed_attributes=(selected,) if selected else social_attributes(category),
            action_fingerprint=(
                resolutions[0].action_fingerprint
                if resolutions
                else semantic_fingerprint(
                    {"channel": turn.channel_id, "category": category.value, "text": turn.mention.text}
                )
            ),
            ability_substitutions=substitutions,
            ability_authorizer=self._reserve_social_ability,
        )

    def _principal_owns_no_character(self, turn) -> bool:
        """True only when a readable directory positively holds no link for this turn.

        The session-zero discriminator: an authenticated player with no character is
        having a conversation, not taking a mechanical action, so gates that exist to
        force a character's dice must stand aside for them. Every failure reads as
        False -- an absent principal and an unreadable directory keep whatever
        fail-closed handling the calling gate already has.
        """
        principal = turn.mention.principal
        if principal is None:
            return False
        try:
            return (
                self._player_directory.snapshot().for_principal(
                    principal.adapter_name, principal.subject_id
                )
                is None
            )
        except PlayerDirectoryError:
            return False

    def _reserve_social_ability(self, actor_id: str, ability_id: str) -> bool:
        """Use the engine port to validate and spend a substitution before a roll."""
        if self._purchase_executor is None:
            return False
        result = self._purchase_executor.reserve_social_ability(actor_id, ability_id)
        return bool(result.get("ok") is True)

    def _romance_principals(self, turn, policy: TurnPolicy) -> tuple[str, ...] | None:
        """Resolve consent principals before accepting a remembered NPC focus."""
        if not policy.romance_escalation:
            return ()
        principal = turn.mention.principal
        if principal is None:
            return None
        affected = self._romance_affected_principals(turn, policy)
        if affected is None:
            return None
        decision = classify_romance(
            self._romance_policy,
            escalation=True,
            affected_principals=affected,
            authenticated_consents=(
                self._channels[turn.channel_id].romance_consents
                if turn.channel_id in self._channels else set()
            ),
            coercive_or_exploitative=policy.romance_coercive,
        )
        return affected if decision.status == "available" else None

    def _romance_affected_principals(self, turn, policy: TurnPolicy) -> tuple[str, ...] | None:
        """Resolve every non-NPC escalation target to an authenticated principal."""
        principal = turn.mention.principal
        if principal is None:
            return None
        try:
            snapshot = self._player_directory.snapshot()
        except PlayerDirectoryError:
            return None
        actor = snapshot.for_principal(principal.adapter_name, principal.subject_id)
        if actor is None:
            return None
        words = frozenset(re.findall(r"[a-z0-9]+", turn.mention.text.casefold()))
        matches = []
        for player in snapshot.players:
            if player.subject_id == principal.subject_id:
                continue
            names = frozenset(re.findall(r"[a-z0-9]+", player.character_name.casefold()))
            identifier = frozenset(player.character_id.replace("-", "_").split("_"))
            if (names and names <= words) or (identifier and identifier <= words):
                matches.append(player)
        if len(matches) > 1:
            return None
        if matches:
            cue = policy.interaction_cue
            if cue is not None and cue.kind == "canonical_npc" and cue.public_npc_id:
                npc_words = frozenset(
                    re.findall(r"[a-z0-9]+", cue.public_npc_id.replace("-", " ").casefold())
                )
                if npc_words and npc_words <= words:
                    return None
            return (principal.subject_id, matches[0].subject_id)
        if policy.interaction_cue is not None and policy.interaction_cue.kind == "canonical_npc":
            return ()
        return None

    async def _collect_decision_submission(self, views):
        """Wait for a request-level answer or shutdown without leaving a task behind."""
        collect = asyncio.create_task(self.adapter.collect_decision(views))
        stopping = asyncio.create_task(self._stopping.wait())
        done, _ = await asyncio.wait((collect, stopping), return_when=asyncio.FIRST_COMPLETED)
        if stopping in done:
            if not collect.done():
                collect.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await collect
            return None, True
        stopping.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopping
        return collect.result(), False

    async def run(self) -> ServiceReport:
        """Start the engine, drive every turn the adapter yields, then shut down.

        Startup order matters. ``engine.start`` asserts the frozen agent surface before
        this method touches ``adapter.turns``, so a policy violation stops the process
        rather than reaching a player. An audit found an earlier arrangement that
        asserted lazily on a channel's first turn, which left a channel yielding zero
        turns unchecked and made this paragraph false.
        """
        report = ServiceReport()
        try:
            self.engine.start()
            self._bind_thinking_level()
            self._bind_language()
            self._bind_mentions()
            # A channel that renders a status line (``StatusLineAdapter``) otherwise
            # shows nothing until the first turn's own ``deliver``/``_post_notice`` call
            # refreshes it -- every other refresh point is a delivery reaction, and
            # nothing has been delivered yet. The freshly bootstrapped campaign already
            # has real values to show (starting HP, the seeded scene), so this reads
            # them before the first prompt rather than leaving it blank.
            await refresh_status(self.adapter, self.config.campaign_root)
            turns = self.adapter.turns()
            while not self._stopping.is_set():
                turn = await self._next_turn(turns)
                if turn is None or self._stopping.is_set():
                    break
                report.turns += 1
                await self._set_thinking(turn.channel_id, True)
                # The game-master designator decides the route before any text
                # classification could: an addressed turn is out-of-fiction discussion
                # in whatever language the table speaks, and the lexicon-based
                # classifier below must never see it. The designator is stripped here
                # so the narrator reads the question, not the address, and the
                # interaction tracker is deliberately not consulted -- an
                # out-of-fiction aside must not disturb an in-fiction focus.
                gm_body = gm_discussion_remainder(turn.mention.text, self.config.gm_address)
                if gm_body is not None:
                    turn = replace(turn, mention=replace(turn.mention, text=gm_body))
                    policy = gm_discussion_policy()
                else:
                    policy = await self._classify_turn_policy(turn)
                    if policy is None:
                        # Fail closed: the engine does not know what this turn was
                        # doing, so it resolves nothing. No dice were committed --
                        # classification runs before any tool call -- which is why
                        # this notice may invite a retry where ``fault_notice`` may
                        # not.
                        self._record_withhold(
                            turn.channel_id, self.config.classifier_fault_notice
                        )
                        await self._post_notice(
                            turn, self.config.classifier_fault_notice, "classifier_fault"
                        )
                        continue
                # ``policy`` comes back because a recovery control replaces it: for
                # ``/retry`` and ``/continue`` the text this loop classified was the
                # control token, and ``_decision_phase`` re-classified the retained
                # declaration the turn is actually about. Everything below -- the
                # romance gate, the social test, the turn framing, the trade offer,
                # the ``meta`` flag -- must read that verdict, not the token's.
                phase, resolutions, execution_turn, policy = await self._decision_phase(
                    turn, policy
                )
                if phase == "stopped":
                    break
                if phase == "classifier_fault":
                    # The retained declaration a control retried could not be read.
                    # Same answer as a freshly typed turn the classifier cannot read:
                    # nothing resolves, and the notice may invite another retry
                    # because no dice were committed.
                    self._record_withhold(turn.channel_id, self.config.classifier_fault_notice)
                    await self._post_notice(
                        turn, self.config.classifier_fault_notice, "classifier_fault"
                    )
                    continue
                if phase == "risk_confirmation_required":
                    self._record_withhold(turn.channel_id, self.config.risk_confirmation_notice)
                    await self._post_notice(
                        turn, self.config.risk_confirmation_notice, "risk_confirmation"
                    )
                    continue
                if phase == "trade_confirmation_required":
                    self._record_withhold(turn.channel_id, self.config.trade_confirmation_notice)
                    await self._post_notice(
                        turn, self.config.trade_confirmation_notice, "trade_confirmation"
                    )
                    continue
                if phase == "social_choice_required":
                    self._record_withhold(turn.channel_id, self.config.decision_fault_notice)
                    await self._post_notice(turn, self.config.decision_fault_notice, "decision_fault")
                    continue
                if phase == "trade_declined":
                    self._record_withhold(turn.channel_id, self.config.decision_declined_notice)
                    self.post_log.append(
                        await deliver_decision_decline(self.adapter, turn, self.config)
                    )
                    continue
                if phase == "trade_completed":
                    # A completed purchase is not a withhold: it is the one successful
                    # outcome this notice announces, so the next turn carries no note.
                    await self._post_notice(turn, self.config.trade_completed_notice, "trade_completed")
                    continue
                if phase == "pending":
                    # The one outcome that answers a turn with no reply at all: control
                    # returns to the player, who is about to be prompted for their own
                    # next line, not left waiting on one -- every other exit posts
                    # something and stops the indicator from inside ``adapter.post``.
                    await self._set_thinking(turn.channel_id, False)
                    report.decision_pending += 1
                    continue
                if phase == "segment":
                    report.decision_pending += 1
                    self._record_withhold(turn.channel_id, self.config.decision_segment_notice)
                    self.post_log.append(
                        await deliver_decision_recovery(self.adapter, turn, self.config)
                    )
                    continue
                if phase == "declined":
                    self._record_withhold(execution_turn.channel_id, self.config.decision_declined_notice)
                    self.post_log.append(
                        await deliver_decision_decline(self.adapter, execution_turn, self.config)
                    )
                    continue
                if phase == "fault":
                    report.decision_faults += 1
                    fault_key, fault_notice = self._fault_notice_for(self._last_fault_category)
                    self._record_withhold(turn.channel_id, fault_notice)
                    self.post_log.append(
                        await deliver_decision_fault(
                            self.adapter, turn, self.config, fault_notice, notice_key=fault_key
                        )
                    )
                    continue
                romance_principals = self._romance_principals(execution_turn, policy)
                if romance_principals is None:
                    self._record_withhold(execution_turn.channel_id, self.config.romance_boundary_notice)
                    await self._post_notice(
                        execution_turn, self.config.romance_boundary_notice, "romance_boundary"
                    )
                    continue
                if len(romance_principals) > 1:
                    # A resolved player target takes precedence over a remembered NPC
                    # focus. The prompt must not imply that the NPC is the participant.
                    policy = replace(policy, interaction_cue=None, focus_candidate=None)
                social_test = self._social_test_request(execution_turn, policy, resolutions)
                if policy.social_test_required and social_test is None:
                    # An unlinked principal can never derive a social test -- there is
                    # no character whose attribute the roll would bind -- so during
                    # session zero every socially-phrased line ("I want to play a
                    # charming duellist") dead-ended in this notice. Conversation with
                    # a player who owns no character proceeds to the narrator instead;
                    # a live TUI ceremony recorded exactly this turn being eaten. The
                    # withhold stands for every linked player, and an unreadable
                    # directory stays failed-closed: only a readable directory that
                    # positively holds no link for this principal proceeds.
                    if self._principal_owns_no_character(execution_turn):
                        social_test = None
                    else:
                        self._record_withhold(execution_turn.channel_id, self.config.decision_fault_notice)
                        await self._post_notice(
                            execution_turn, self.config.decision_fault_notice, "decision_fault"
                        )
                        continue


                framing = turn_framing_for(policy.route, policy.also_declares_act)
                withheld_note = None
                if framing:
                    withheld_channel = self._channels.get(execution_turn.channel_id)
                    if withheld_channel is not None:
                        withheld_note = withheld_channel.withheld_note
                        withheld_channel.withheld_note = None
                prepared_turn = PreparedNarrationTurn(
                    execution_turn,
                    policy.interaction_cue,
                    social_test,
                    self._trade_offer(execution_turn, policy),
                    False,
                    withheld_note,
                    framing,
                    policy.declared_act_tool,
                    meta=policy.route == "gm",
                )
                if resolutions:
                    outcome = await self._run_engine(
                        prepared_turn, resolutions, resolutions[0].action_fingerprint
                    )
                else:
                    outcome = await self.engine.run_turn(prepared_turn)
                self._record_tool_events(outcome)
                report.leaks_scrubbed += outcome.leaks_scrubbed
                report.settle_attempts += outcome.settle_attempts
                if outcome.settle == "commit":
                    report.commits += 1
                elif outcome.settle == "waive":
                    report.waives += 1
                if outcome.error:
                    report.errors.append(outcome.error)
                if outcome.withheld:
                    report.withheld += 1
                else:
                    report.delivered += 1
                post = await deliver(self.adapter, turn, outcome, self.config)
                self.post_log.append(post)


                if post.kind == "notice" and advertises_recovery_controls(post.text):
                    self._retain_recovery(execution_turn)
                if (
                    not outcome.withheld
                    and not outcome.error
                    and outcome.narration.strip()
                    and post.kind == "narration"
                ):
                    # ``post.kind`` established that this text is the model's own
                    # narration rather than a fault or withheld notice. Only that
                    # text may set the reply invitation, so an engine-authored notice
                    # ending in a question mark can never make the next turn social.
                    self._interactions.complete_delivery(
                        turn.channel_id, policy, self.config.campaign_root, post.text
                    )
                    category = None
                    if policy.social_category:
                        try:
                            category = SocialCategory(policy.social_category)
                        except ValueError:
                            category = None
                    self._social.commit_social(
                        turn.channel_id,
                        scope=policy.scope,
                        cue=policy.interaction_cue,
                        category=category,
                        delivered=True,
                        clear=policy.clears_focus,
                    )
                    terms = prepared_turn.trade_offer
                    if terms is not None and policy.interaction_cue is not None:
                        self._social.accept_offer(
                            turn.channel_id,
                            scope=policy.scope,
                            seller=policy.interaction_cue,
                            terms=terms,
                            delivered=True,
                        )
                    elif policy.trade_phase:
                        self._social.advance_trade(
                            turn.channel_id, scope=policy.scope, policy=policy
                        )
                    self._social.apply_social_outcome(
                        turn.channel_id,
                        scope=policy.scope,
                        outcome=getattr(outcome, "social_test_outcome", None),
                    )
        finally:
            try:
                # C2: a sweep dispatched in the background on this loop's last turn
                # must still run to completion, or the fact it would have committed
                # is lost for good -- there may be no further turn on any channel to
                # resolve it at the top of. Before ``adapter.close()`` too, so an
                # instrumented adapter's own shutdown reporting (the soak harness)
                # reads a resolved ``_sweep_log``, not a trailing ``pending``.
                await self.engine.flush_pending_sweep()
                await self._decisions.cancel_all_shutdown()
                self._social.close()
                await self.adapter.close()
            finally:
                self._interactions.close()
                self.engine.stop()
        return report

    def _trade_offer(self, turn, policy: TurnPolicy) -> TradeTerms | None:
        """Project only a current, engine-authoritative merchant quote into narration."""
        cue = policy.interaction_cue
        if (
            policy.trade_phase != TradePhase.INQUIRY.value
            or cue is None
            or cue.kind != "canonical_npc"
            or not cue.public_npc_id
            or self._purchase_executor is None
        ):
            return None
        quote = self._purchase_executor.quote_for_text(cue.public_npc_id, turn.mention.text)
        if quote.get("ok") is not True:
            return None
        try:
            return TradeTerms(
                str(quote["seller_id"]),
                str(quote["item_id"]),
                str(quote["public_label"]),
                int(quote["quantity"]),
                str(quote["currency"]),
                int(quote["price_copper"]),
                int(quote["stock_version"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def install_signal_handlers(service: NarratorService) -> None:
    """Stop cleanly on SIGINT and SIGTERM so the server observes end of file."""
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, service.request_stop)
        except (NotImplementedError, RuntimeError):
            # Platforms without signal handler support fall back to default behaviour.
            pass
