"""Typed scene facts, channel focus, and the routing value the classifier fills in.

- the campaign-file readers that build the typed values ``narrator.policy_types``
  carries (``read_trusted_scope``, ``read_combat_snapshot``,
  ``read_character_display_names``),
- ``InteractionTracker``, the per-channel focus store, which now awaits a classifier
  the service passes in rather than importing one,
- ``gm_discussion_remainder``, which is pure syntax and always was.

That change closed a safety hole, not merely a multilingual gap, and the distinction is
worth keeping because the first analysis got it backwards. The retired version matched
``_BYSTANDER_NOUNS`` (about seventy English nouns for people) union campaign
identifiers, and the noun half was what counted living bystanders. Measured in a scene
recording ``rade`` dead and no tavern-keeper:

    \"I attack the barkeep and loot rade\"          -> no bypass  (correct)
    \"J'attaque le tavernier et je fouille rade\"   -> BYPASS     (wrong)

With no \"tavernier\" in the list the French declaration reduced to the one word it could
resolve -- a corpse -- so \"every person named is dead\" held vacuously and the ask over
an assault on a living person was skipped. An unmatched *corpse* reference fails toward
asking and is safe; an unmatched *living person* reference fails toward proceeding and
is not. Non-English tables got the second.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from narrator.identifiers import IDENTIFIER, words
from narrator.policy_types import (
    CombatSnapshot,
    InteractionCue,
    TrustedScope,
    TurnPolicy,
)


def gm_discussion_remainder(text: str, designator: str) -> str | None:
    """The out-of-fiction body of a game-master-addressed turn, or ``None``.

    A turn opening with ``@`` followed by the configured designator addresses the game
    master out of fiction. Detection is pure syntax -- the fixed ``@`` sigil, the
    configured token compared casefolded, and a boundary check -- so it holds for any
    designator in any script (``@MJ``, ``@SL``, ``@ВЕД``, ``@進行``) and never reads a
    word of the turn's own language. That was written when the risk lexicon and the
    out-of-character markers in this module were English-bound best effort and this lane
    was the one route that must not inherit the limit. ``narrator.classify`` has since
    removed the limit everywhere else, and this stays syntax anyway: the designator lane
    must not depend on a model verdict, because a model that could be argued out of
    recognizing the address could be argued into resolving a mechanic on it.

    The boundary check requires the character after the token to be a non-alphanumeric
    separator (or the end of the turn), so a designator of ``GM`` never claims
    ``@GMX``. The casefold comparison is prefix-shaped and assumes the typed token and
    the configured token casefold to equal lengths; a token whose casefold expands
    (``ß`` to ``ss``) simply fails to match in the safe direction -- the turn stays
    in fiction and every existing gate applies.

    Returns the turn with the address stripped -- what the game master is actually
    being asked -- with one leading ``:`` or ``,`` separator also removed, or ``None``
    when the turn does not open with the address.
    """
    token = designator.strip()
    if not token:
        return None
    stripped = text.lstrip()
    if not stripped.startswith("@"):
        return None
    candidate = stripped[1:]
    if candidate[: len(token)].casefold() != token.casefold():
        return None
    remainder = candidate[len(token) :]
    if remainder[:1].isalnum():
        return None
    remainder = remainder.lstrip()
    if remainder[:1] in {":", ","}:
        remainder = remainder[1:]
    return remainder.strip()


def gm_discussion_policy() -> TurnPolicy:
    """The fixed routing verdict for a game-master-addressed turn.

    Built directly rather than through ``InteractionTracker.policy`` because the
    designator decides the route before any text classification could, and because an
    out-of-fiction aside must not disturb the tracker's in-fiction focus: a player who
    pauses a negotiation to ask the game master a question is still mid-negotiation
    when they drop back in. The empty ``TrustedScope`` is safe because every consumer
    of ``scope`` sits on routes this policy never takes -- the risk floor, the social
    tests, the trade frames all key off routes and flags this policy pins to their
    inert values.
    """
    return TurnPolicy(
        route="gm",
        risk_category="none",
        interaction_cue=None,
        focus_candidate=None,
        scope=TrustedScope("", "", ()),
        retains_focus=True,
        clears_focus=False,
    )


@dataclass(frozen=True)
class _Focus:
    scope: TrustedScope
    cue: InteractionCue
    sequence: int
    awaiting_reply: bool = False


def _read_party_name_tokens(campaign_root: Path) -> tuple[str, ...]:
    """Word tokens naming the party's own characters, so a named answer is recognizable.

    A player answering "What is your name?" types the character's name and nothing else.
    Recognizing that as an answer requires knowing the names, and ``campaign/players.yaml``
    is where the campaign records them.

    This reads character identifiers and display names only. It never reads the account
    identifier beside them, which is the private binding the player directory owns. Any
    read or parse failure returns an empty tuple, so a turn never fails on this.
    """
    try:
        import yaml

        raw = (campaign_root / "campaign" / "players.yaml").read_bytes()
        payload = yaml.safe_load(raw) or {}
    except Exception:  # noqa: BLE001 - an unreadable directory must never fail a turn
        return ()
    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, list):
        return ()
    tokens: set[str] = set()
    for link in players:
        if not isinstance(link, dict):
            continue
        for key in ("character_id", "display_name"):
            value = link.get(key)
            if isinstance(value, str):
                tokens.update(words(value))
    return tuple(sorted(tokens))


def read_character_display_names(campaign_root: Path) -> dict[str, str]:
    """Map each party character's id to its display name, from the same source
    ``_read_party_name_tokens`` reads.

    Unlike that function, this keeps the id-to-name pairing rather than flattening it
    into a token set: constructing a roll announcement needs the specific name that
    goes with a specific id, not just a set of recognisable words. Any read or parse
    failure returns an empty mapping, so a caller can always fall back to the bare id
    rather than fail a turn over a display label.
    """
    try:
        import yaml

        raw = (campaign_root / "campaign" / "players.yaml").read_bytes()
        payload = yaml.safe_load(raw) or {}
    except Exception:  # noqa: BLE001 - an unreadable directory must never fail a turn
        return {}
    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, list):
        return {}
    names: dict[str, str] = {}
    for link in players:
        if not isinstance(link, dict):
            continue
        character_id = link.get("character_id")
        display_name = link.get("display_name")
        if isinstance(character_id, str) and isinstance(display_name, str) and display_name:
            names[character_id] = display_name
    return names


#: The lifecycle statuses ``bsh_mcp.models.NPC`` records. Anything else the state file
#: carries is not a status this module understands, and it is dropped rather than
#: guessed at, so an unknown value can never read as "dead" and suppress a confirmation.
_NPC_STATUSES = frozenset({"alive", "dead", "fled", "captured"})


def _read_npc_states(campaign_root: Path) -> tuple[tuple[str, str], ...]:
    """Read each recorded NPC's lifecycle status from the campaign state file.

    Every failure returns an empty tuple. An unreadable, absent, or malformed state file
    therefore leaves every NPC's status unknown, and every scope-aware check below fails
    safe toward the confirmation rather than toward suppressing it.
    """
    try:
        payload = json.loads(
            (campaign_root / "campaign" / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ()
    npcs = payload.get("npcs") if isinstance(payload, dict) else None
    if not isinstance(npcs, dict):
        return ()
    states: list[tuple[str, str]] = []
    for npc_id, record in npcs.items():
        if not isinstance(npc_id, str) or not IDENTIFIER.fullmatch(npc_id):
            continue
        status = record.get("status") if isinstance(record, dict) else None
        if isinstance(status, str) and status in _NPC_STATUSES:
            states.append((npc_id, status))
    return tuple(sorted(states))


def read_trusted_scope(campaign_root: Path | str) -> TrustedScope:
    """Read only validated scene scope, never player or narrator content."""
    try:
        scene = (Path(campaign_root) / "campaign" / "scene.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return TrustedScope("", "", ())
    session = ""
    location = ""
    for line in scene.splitlines()[:16]:
        if line.startswith("session:"):
            value = line.split(":", 1)[1].strip()
            session = value if IDENTIFIER.fullmatch(value) else ""
        elif line.startswith("location_id:"):
            value = line.split(":", 1)[1].strip()
            location = value if IDENTIFIER.fullmatch(value) else ""
    present: list[str] = []
    persons: list[str] = []
    section = ""
    for line in scene.splitlines():
        if line.startswith("## "):
            section = line.strip()
            continue
        if section == "## Present NPCs" and line.startswith("- "):
            # The render annotates non-alive NPCs ("- rade (dead)"); the scope keeps
            # the bare id -- alive-or-dead stays ``_read_npc_states``'s to answer,
            # from the state file the annotation itself is derived from.
            npc_id = re.sub(r"\s*\([^)]*\)$", "", line[2:].strip())
            if IDENTIFIER.fullmatch(npc_id) and npc_id not in present:
                present.append(npc_id)
        elif section == "## Other persons present" and line.startswith("- "):


            person_id = line[2:].split(":", 1)[0].strip()
            if IDENTIFIER.fullmatch(person_id) and person_id not in persons:
                persons.append(person_id)
    return TrustedScope(
        session,
        location,
        tuple(present),
        _read_party_name_tokens(Path(campaign_root)),
        _read_npc_states(Path(campaign_root)),
        tuple(persons),
    )


_COMBATANT_LINE = re.compile(
    r"^- combatant:\s+(?P<id>[a-z0-9][a-z0-9_-]{0,63})\s+side=(?P<side>pc|npc)\b"
)


def read_combat_snapshot(campaign_root: Path | str) -> CombatSnapshot:
    """Parse the scene render's Combat section. Never raises.

    ``CampaignStore._combat_section`` writes the section this function reads, and it
    writes nothing while no fight runs. An absent section, an unreadable file, and a
    closed fight therefore produce the same inactive snapshot, which is the value every
    caller must treat as "no combat facts available". Failing to an inactive snapshot
    keeps every gate that consults it at its pre-combat behaviour.
    """
    try:
        scene = (Path(campaign_root) / "campaign" / "scene.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return CombatSnapshot()
    round_number = 0
    active_actor = ""
    order: tuple[str, ...] = ()
    sides: list[tuple[str, str]] = []
    in_combat = False
    for line in scene.splitlines():
        if line.startswith("## "):
            in_combat = line.strip() == "## Combat"
            continue
        if not in_combat or not line.startswith("- "):
            continue
        body = line[2:].strip()
        if body.startswith("round:"):
            value = body.split(":", 1)[1].strip()
            round_number = int(value) if value.isdigit() else 0
        elif body.startswith("active actor:"):
            value = body.split(":", 1)[1].strip()
            # The renderer writes the literal "none" for an unset actor. It is a valid
            # identifier, so it must be read as absence rather than as an actor named
            # "none". This slice is the first production consumer, so it lands here.
            active_actor = value if value != "none" and IDENTIFIER.fullmatch(value) else ""
        elif body.startswith("order:"):
            value = body.split(":", 1)[1].strip()
            order = tuple(
                part.strip()
                for part in value.split(",")
                if part.strip() != "none" and IDENTIFIER.fullmatch(part.strip())
            )
        else:
            match = _COMBATANT_LINE.match(line)
            if match and all(match.group("id") != known for known, _ in sides):
                sides.append((match.group("id"), match.group("side")))
    if not order and not sides:
        return CombatSnapshot()
    return CombatSnapshot(
        active=True,
        round=round_number,
        active_actor=active_actor,
        order=order,
        sides=tuple(sides),
    )


@dataclass(frozen=True)
class OpenNpcTurn:
    """One NPC's own open combat turn, with what a defence decision needs to name."""

    actor_id: str
    name: str
    actions_remaining: int


def open_npc_turn(campaign_root: Path | str) -> OpenNpcTurn | None:
    """The NPC whose combat turn is open with at least one action left, or ``None``.

    Reads ``campaign/state.json`` directly, never the scene render's own Combat
    section ``read_combat_snapshot`` parses: that section is written for the canon
    digest a model reads, and its per-combatant line carries ``actions={remaining}/
    {max}`` for whichever combatant but no ``turn_open`` flag at all, so it cannot
    tell a closed turn with zero actions used from an open one with one spent of two.
    ``NarratorEngine._combat_recovery_view`` reads the same file for the identical
    reason: this decides a real mechanic from the campaign's own record, never from
    anything rendered for a model to read.
    """
    try:
        raw = json.loads(
            (Path(campaign_root) / "campaign" / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    combat = raw.get("combat") if isinstance(raw, dict) else None
    if not isinstance(combat, dict) or not combat.get("active"):
        return None
    active_id = str(combat.get("active_actor") or "")
    actors = combat.get("actors")
    actor = actors.get(active_id) if isinstance(actors, dict) else None
    if not isinstance(actor, dict) or actor.get("side") != "npc" or not actor.get("turn_open"):
        return None
    remaining = int(actor.get("actions_max", 0)) - int(actor.get("actions_used", 0))
    if remaining <= 0:
        return None
    npcs = raw.get("npcs") if isinstance(raw, dict) else None
    npc = npcs.get(active_id) if isinstance(npcs, dict) else None
    if not isinstance(npc, dict) or npc.get("status") != "alive":
        return None
    name = str(npc.get("name") or active_id)
    return OpenNpcTurn(actor_id=active_id, name=name, actions_remaining=remaining)


def hazard_scope(
    scope: TrustedScope | None, combat: CombatSnapshot | None = None
) -> TrustedScope:
    """The scope a hazard decision should judge against, never the empty placeholder.

    An open fight is the fallback, and it is a real one rather than a placeholder. A
    caller that holds a ``CombatSnapshot`` but no scope still knows which NPCs are
    present and engaged, because the fight lists them, so the fight's own ``npc`` side
    becomes the present-NPC set. That is strictly more than the empty scope every hazard
    call site built before, and it is what lets a caller outside this slice's declared
    paths reach a populated scope without changing its own call.

    Only a caller with neither returns the empty scope, which classifies exactly as the
    whole hazard floor did before this requirement.
    """
    if scope is not None:
        return scope
    if combat is not None and combat.active and combat.npc_combatants:
        return TrustedScope("", "", combat.npc_combatants)
    return TrustedScope("", "", ())


def named_dead_target(policy: TurnPolicy) -> bool:
    """Whether every person this declaration names is an NPC the campaign records dead.

    The test is positive on both halves, so absence never suppresses a confirmation. It
    requires that the declaration name at least one person, and that every person it
    names be a present NPC recorded ``dead``. A declaration naming a living bystander
    alongside a corpse still confirms, because the bystander is not accounted for.

    ``names_unrecorded_person`` is the third refusal and the one a port can lose. This
    used to read English nouns (``_BYSTANDER_NOUNS``: \"barkeep\", \"merchant\", ...) union
    campaign identifiers, and the noun half was doing real work -- it is what made \"I
    attack the barkeep and loot rade\" answer False in a scene recording no barkeep.
    Keying on validated identifiers alone would drop the barkeep as unresolvable and
    reduce the declaration to \"only rade's body\", bypassing a consent ask over a living
    person. So the classifier reports unrecorded people explicitly and this refuses on
    them, which is also what makes the rule work in any language: identifiers are
    script-independent and the classifier resolves \"le marchand\" to one.
    """
    named = frozenset(policy.named_person_ids)
    if not named or policy.names_unrecorded_person:
        return False
    dead = frozenset(
        npc_id for npc_id in policy.scope.present_npc_ids if policy.scope.is_dead(npc_id)
    )
    return bool(dead) and named <= dead


def combat_sanctions_violence(policy: TurnPolicy, combat: CombatSnapshot | None) -> bool:
    """Whether an open fight makes a violence declaration a combat action, not an assault.

    Consent lives in the campaign's typed combat state, never in the declaration's
    vocabulary. Everyone in an open fight entered it through ``combat_start``, the one
    gate that puts a person into a fight, and the risk floor already confirmed that gate
    when the player instigated it. While the fight runs, violence is the mechanic the
    fight exists to resolve, and the combat tools refuse any target the roster does not
    list (``target_not_in_combat``), so no one outside the fight can lose a hit point
    whatever this answers.

    The sanction is withheld on one kind of evidence: the declaration names a person the
    fight's enemy side does not cover -- a party member, or a present, living NPC the
    roster omits. A corpse withholds nothing, and neither does a phrasing: no words are
    inspected here at all.

        \"I attack maren\"      -> withheld  (correct)
        \"J'attaque maren\"     -> withheld  (correct, the identifier is ASCII)
        \"マレンを攻撃する\"      -> SANCTIONED (wrong)
        \"Я атакую Марен\"      -> SANCTIONED (wrong)

    Retiring the token match also retired ``_TARGETLESS_WORDS``, which existed only to
    stop a bare function word inside an identifier (\"the\" in ``the-wanderer``) matching
    every declaration -- a problem that does not exist once names arrive resolved.
    """
    if combat is None or not combat.active:
        return False
    engaged = frozenset(combat.npc_combatants)
    if not engaged:
        return False
    scope = policy.scope
    outside = {actor_id for actor_id, side in combat.sides if side == "pc"}
    outside |= {
        npc_id
        for npc_id in scope.present_npc_ids
        if npc_id not in engaged and not scope.is_dead(npc_id)
    }
    return not (frozenset(policy.named_person_ids) & (outside - engaged))


_CLOSING_MARKS = "\"'`*_)]}’”"


def narration_invites_reply(narration: str) -> bool:
    """Whether a delivered narration ends by asking the player something.

    An interlocutor's direct question ends the narration with a question mark, and the
    player answers it in bare words. That answer carries no speech verb, so nothing in the
    declaration itself marks it as conversation. This supplies the missing signal.

    It reads narrator output only, never player text. ``InteractionTracker``'s rule
    against retaining player-derived state therefore still holds.

    It reads the last non-empty line and asks whether that line ends in a question mark,
    ignoring the closing marks ``_CLOSING_MARKS`` names. A narration that asks something
    and then narrates past it does not count. That direction of failure returns the turn
    to the routing which shipped before this function.
    """
    for line in reversed(narration.strip().splitlines()):
        stripped = line.strip().rstrip(_CLOSING_MARKS).strip()
        if stripped:
            return stripped.endswith("?")
    return False


@dataclass
class _ChannelInteraction:
    """One channel's own interaction-tracking state.

    Replaces two dicts (``_focuses``, ``_last_referents``) that shared this exact key
    space but not a single invalidation rule -- each field below still clears on its
    own trigger, exactly as the two dicts did independently; only the container is
    unified.
    """

    #: The channel's active interlocutor focus, or ``None``.
    focus: _Focus | None = None
    #: The immediately preceding delivered turn's own
    #: ``(named_person_ids, names_unrecorded_person, awaiting_reply)``, read by
    #: ``classify.ReplyInheritsReferents``. Independent of ``focus`` above because a
    #: declaration can name a referent -- a corpse looted, never addressed -- without
    #: ever setting an interlocutor focus.
    last_referents: tuple[tuple[str, ...], bool, bool] | None = None


class InteractionTracker:
    """Service-local channel focus that stores validated scope and no transcript data."""

    def __init__(self) -> None:
        self._channels: dict[str, _ChannelInteraction] = {}
        self._sequence = 0

    async def policy(
        self,
        channel_id: str,
        player_text: str,
        campaign_root: Path | str,
        *,
        classify,
        offer_open: bool = False,
    ) -> TurnPolicy | None:
        """One routing verdict for this turn, or ``None`` when it cannot be read.

        ``classify`` is the awaitable the service supplies -- in production
        ``NarratorEngine.classify_intent``, in the offline suites a fake. Passing it in
        rather than importing it keeps this module free of both the model path and the
        engine, which is what lets the tracker stay a pure focus store.

        ``offer_open`` is the service's own fact about this channel's trade frame. The
        caller already bound it into ``classify`` so the prompt carries it; it is passed
        here as well so the derived policy can refuse an acceptance the channel cannot
        honor (``classify.AcceptanceNeedsOpenOffer``).

        ``None`` is every classifier failure collapsed to one value, and it routes
        nothing: the service posts ``config.classifier_fault_notice`` and resolves no
        action. There is no lexical fallback behind it on purpose; see
        ``narrator.classify`` for the measurement that retired the word lists rather
        than demoting them to a backstop.
        """
        scope = read_trusted_scope(campaign_root)
        channel = self._channels.get(channel_id)
        focus = channel.focus if channel else None
        if focus and (
            focus.scope != scope
            or (
                focus.cue.public_npc_id
                and focus.cue.public_npc_id not in scope.present_npc_ids + scope.present_person_ids
            )
        ):
            channel.focus = None
            focus = None
        combat = read_combat_snapshot(campaign_root)
        classification = await classify(player_text, scope=scope, combat=combat)
        if classification is None:
            return None
        from narrator.classify import policy_from

        last_referents = channel.last_referents if channel else None
        return policy_from(
            classification,
            scope=scope,
            combat=combat,
            active_focus=focus.cue if focus else None,
            awaiting_reply=focus.awaiting_reply if focus else False,
            offer_open=offer_open,
            last_named_person_ids=last_referents[0] if last_referents else (),
            last_names_unrecorded_person=last_referents[1] if last_referents else False,
            awaiting_referent_reply=last_referents[2] if last_referents else False,
        )

    def complete_delivery(
        self,
        channel_id: str,
        policy: TurnPolicy,
        campaign_root: Path | str,
        narration: str = "",
    ) -> None:
        """Commit focus only after delivery and only if the post-turn scope still matches.

        ``narration`` is the text the channel actually posted. Whether it ended by asking
        the player something rides with the focus, so every path that drops the focus also
        drops the invitation. A caller that omits it records no invitation, which keeps the
        older two-argument call site behaving exactly as it did.

        The invitation always describes the most recent delivered narration, never an
        older one. A turn that keeps an existing focus without replacing it, which is the
        shape of the out-of-character and read policies, therefore rewrites the flag
        rather than leaving it. An audit found the earlier version letting an invitation
        outlive the narration that armed it across such a turn.

        Also records this turn's own referents and invitation for
        ``classify.ReplyInheritsReferents``, independent of the focus logic below: a
        loot declaration naming a corpse sets ``candidate`` to ``None`` (it addresses
        no one) and would otherwise leave nothing behind for a referent-less
        follow-up to inherit.
        """
        scope = read_trusted_scope(campaign_root)
        invites = narration_invites_reply(narration)
        if scope != policy.scope:
            self._channels.pop(channel_id, None)
            return
        channel = self._channels.setdefault(channel_id, _ChannelInteraction())
        channel.last_referents = (
            policy.named_person_ids, policy.names_unrecorded_person, invites
        )
        if policy.clears_focus:
            channel.focus = None
            return
        candidate = policy.focus_candidate
        if candidate is None:
            if channel.focus is not None and channel.focus.awaiting_reply != invites:
                channel.focus = replace(channel.focus, awaiting_reply=invites)
            return
        if (
            candidate.public_npc_id
            and candidate.public_npc_id not in scope.present_npc_ids + scope.present_person_ids
        ):
            channel.focus = None
            return
        self._sequence += 1
        channel.focus = _Focus(scope, candidate, self._sequence, invites)

    def close(self) -> None:
        """Discard every channel focus when the service shuts down."""
        self._channels.clear()
