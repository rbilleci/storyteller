"""The typed routing verdict a turn classifier fills in, and the values it carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrustedScope:
    """Validated campaign scope that may bound one channel's ephemeral focus.

    Hostility is deliberately absent here. No campaign field records it: the only
    evidence the engine holds that an NPC is an enemy is that an open fight lists it on
    the ``npc`` side, and that fact travels beside this value in ``CombatSnapshot``
    rather than inside it, for the reason ``CombatSnapshot`` documents.
    """

    campaign_session: str
    location_id: str
    present_npc_ids: tuple[str, ...]
    party_name_tokens: tuple[str, ...] = ()
    npc_states: tuple[tuple[str, str], ...] = ()


    present_person_ids: tuple[str, ...] = ()

    def status_of(self, npc_id: str) -> str:
        """The recorded status, or the empty string when the campaign omits the NPC."""
        for recorded_id, status in self.npc_states:
            if recorded_id == npc_id:
                return status
        return ""

    def is_dead(self, npc_id: str) -> bool:
        """Whether the campaign records this NPC as a corpse rather than a person.

        Only ``dead`` answers True. ``fled`` and ``captured`` describe an NPC who is
        still a person the risk floor owes a confirmation over; they merely describe one
        who is no longer here.
        """
        return self.status_of(npc_id) == "dead"


@dataclass(frozen=True)
class CombatSnapshot:
    """One fight's typed public facts, parsed from the scene render.

    This value travels beside ``TrustedScope`` and never inside it.
    ``InteractionTracker.policy`` clears a channel's focus whenever the scope changes,
    so a round counter folded into the scope would discard the interlocutor every
    combat exchange. Keeping the two apart leaves focus semantics untouched.
    """

    active: bool = False
    round: int = 0
    active_actor: str = ""
    order: tuple[str, ...] = ()
    sides: tuple[tuple[str, str], ...] = ()

    def side_of(self, actor_id: str) -> str:
        """The recorded side, or the empty string when the fight omits the actor."""
        for recorded_id, side in self.sides:
            if recorded_id == actor_id:
                return side
        return ""

    @property
    def npc_combatants(self) -> tuple[str, ...]:
        return tuple(actor_id for actor_id, side in self.sides if side == "npc")


@dataclass(frozen=True)
class InteractionCue:
    """Trusted in-world interlocutor data permitted in the narrator prompt."""

    #: ``scene_person``: a ``Scene.persons`` entry the classifier resolved as the
    #: addressee (Phase 3). Carries the person's id in ``public_npc_id`` so every
    #: prompt renderer reads it the same way; every mechanic that needs a *recorded*
    #: NPC -- a trade seller, a romance counterpart -- checks for ``canonical_npc``
    #: and refuses this kind exactly as it refuses ``scene_interlocutor``.
    kind: Literal["canonical_npc", "scene_interlocutor", "scene_person"]
    public_npc_id: str | None = None


ROUTE_TURN_FRAMINGS = {
    "read": "question",
    "out_of_character": "out_of_character",
    "gm": "gm_discussion",
}


#: The framing a question-shaped turn owes when the same message also declared an act.
#: Not in ``ROUTE_TURN_FRAMINGS``, because it is not a route: it is the ``read`` route
#: plus one fact about the message, answered by ``classify.ROUTE_COMPANION``.
#:
#: A player who writes "we take a short rest and watch the water. Does anything find
#: us?" routes ``read`` -- the message does ask what is perceptible -- and the
#: ``question`` framing then tells the model on that same turn not to resolve an action
#: the player has not declared, while ``skills/bsh-gm`` tells it to call ``rest``. The
#: model resolves the contradiction by doing neither, and the rest is silently lost.
#: This is the route field's analogue of the trailing-question rule
#: ``classify.HazardFacet`` already carries for the hazard field.
QUESTION_WITH_ACT_FRAMING = "question_with_act"


def turn_framing_for(route: str, also_declares_act: bool = False) -> str:
    """The framing this route owes the narrator, or the empty string for a declaration.

    ``also_declares_act`` is ``TurnPolicy``'s own field, and it moves only the ``read``
    route's framing. The out-of-character and game-master framings answer in a
    different voice -- as the game master, about the game -- so folding a declared act
    into either would mean rewriting a measured string rather than adding one, and the
    game-master lane refuses every mechanical tool by design.
    """
    framing = ROUTE_TURN_FRAMINGS.get(route, "")
    if also_declares_act and framing == "question":
        return QUESTION_WITH_ACT_FRAMING
    return framing


@dataclass(frozen=True)
class TurnPolicy:
    """One immutable routing decision, made before planning or mechanics."""

    route: Literal["risk", "out_of_character", "social", "read", "planner", "gm"]
    risk_category: Literal["theft", "violence", "destructive", "none"]
    interaction_cue: InteractionCue | None
    focus_candidate: InteractionCue | None
    scope: TrustedScope
    retains_focus: bool
    clears_focus: bool
    social_mode: str = ""
    social_category: str = ""
    social_test_required: bool = False
    trade_phase: str = ""
    romance_escalation: bool = False
    romance_coercive: bool = False
    #: A ``read`` turn that looks at something, as against one that asks a question
    #: about the record. ``decisions.planning_bypass`` selects between two engine-owned
    #: bypass behaviors on it. It replaces ``_OBSERVATION_PREFIXES``, an English
    #: prefix tuple ("look", "observe", "examine", ...) that a French or Japanese
    #: read turn never matched.
    reads_by_observing: bool = False
    #: Every person the message names as a referent of the act, by recorded identifier,
    #: already filtered to the presence list. ``named_dead_target`` reads it.
    named_person_ids: tuple[str, ...] = ()
    #: The message named a person the scene does not record. Any rule that proceeds
    #: *because* everyone named is a corpse must refuse when this is true: an
    #: unrecorded person cannot be shown to be dead.
    names_unrecorded_person: bool = False
    #: The message takes up an offer just made. ``SocialState.advance_trade`` reads it
    #: alongside the live frame; neither is sufficient alone.
    accepts_offer: bool = False
    #: A price in copper the message names, or 0. Replaces ``social.counteroffer_price``.
    offered_price: int = 0
    #: ``"dodge"``, ``"parry"``, or ``""``. The engine binds ``combat_defend`` on it
    #: only inside an open fight, for a principal the fight records on the player side.
    #: Replaces ``defence_method``'s ``_DEFENCE_METHODS``/``_DEFENCE_INTENT_WORDS`` pair,
    #: which could only recognize the two English literals.
    defence_method: Literal["dodge", "parry", ""] = ""
    #: ``"yes"``, ``"no"``, or ``""`` when the message is not a bare answer. The engine
    #: honours it only alongside ``narration_invites_reply``, so a stray "yes" cannot
    #: resolve a question nobody asked. Replaces ``is_bare_yes_or_no``, whose
    #: ``_YES_NO_WORDS`` admitted "aye" and "nope" and refused "oui", "ja" and "はい".
    bare_answer: Literal["yes", "no", ""] = ""
    #: The message asked a question *and* declared an action -- a rest, a move, a
    #: search -- that still owes the table its tool call. Answered by
    #: ``classify.ROUTE_COMPANION`` beside nothing else it could disturb, kept only on
    #: a route that carries an asking framing (``declared_act_needs_asking_route``),
    #: and read by ``turn_framing_for`` to select ``QUESTION_WITH_ACT_FRAMING``. It
    #: never moves the route itself: a real hazard is already promoted to ``risk`` by
    #: ``hazard_forces_risk`` whatever the model answered for ``route``.
    also_declares_act: bool = False
    #: Which act it declared, from ``classify.DECLARED_ACT_KINDS``, or ``""`` when the
    #: turn declared none (or declared one on a route where the flag says nothing).
    #: Answered by the model; kept beside the tool below so a soak report can separate
    #: "the classifier read the message wrong" from "the classifier was right and the
    #: model ignored the framing" without re-running the session.
    declared_act_kind: str = ""
    #: The MCP tool that resolves that act, or ``""`` where the tool surface answers
    #: with more than one and the framing must stay generic. Engine policy over
    #: ``declared_act_kind`` (``classify.DeclaredActNamesItsTool``), never the model's
    #: choice; ``prompt``'s ``question_with_act`` framing names it to the narrator.
    #: The empty string rather than ``None`` for the same reason ``defence_method`` and
    #: ``bare_answer`` use it: every optional string on this value is absent-as-empty,
    #: and ``service._routing_row`` serializes it straight into the soak report.
    declared_act_tool: str = ""
