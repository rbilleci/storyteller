"""The sweep contract: recovering durable facts the model no longer volunteers.

Two invariants bound what the sweep can do. It runs only on delivered turns, so every
fact it can see is text the party has already read: swept facts are party knowledge by
construction, which is why this schema carries no ``hidden_changes`` field at all
under the knowledge-ownership contract in ``bsh_mcp.server.scene_commit``. And it
holds the same zero new capability property the settle step recorded: a swept commit
writes nothing the model could not already write through its own ``scene_commit``,
and a declined sweep changes nothing — the sweep narrows what silence loses, not what
words can do. Failure is open: the sweep is recovery, so a sweep that breaks must
never become a new way to lose a turn.

Three defects in earlier versions of this guard, all found by independent audit and
all fixed here, shape its exact contract. First, detection scope: the guard reads
only the record's own ``public_summary`` and ``visible_changes`` — the words actually
headed for ``scene_commit`` — never the turn's full narration. An NPC quoting an
unrelated price, or a character's HP coming up in unrelated banter, must not cost an
unrelated, genuine record its ratification merely because the two shared a turn.
Second, verification: a claimed coin or hit-point figure is checked against the real,
current character-sheet values the caller reads fresh at ratification time
(``narrator.engine.NarratorEngine._sweep`` supplies them; this module reads no files
itself), not against \"did any tool call succeed this turn.\" An earlier version used
that coarser proxy, and it ratified a fabricated coin total whenever one unrelated,
non-mutating tool call — a ``character_sheet`` lookup, say — happened to also run in
the same turn; a call succeeding has no necessary relationship to the specific figure
being claimed. Third, attribution: the claimed figure is checked against the one
character the record's own words name, never the party's pooled state. A second
earlier version checked a claimed coin figure against every character's coins at
once (`value in actual_coins`, a flat tuple), so \"Ossa now has 3 coins\" ratified as
long as *some* character in the party — Rill, say — really had 3 coins, regardless of
what Ossa's own sheet said. ``_match_character`` resolves which character a claim
names by matching each character's id or name as a whole word in the claim text; one
match verifies against that character alone, and a claim naming no one is resolved
only when the party holds exactly one character (the common case, where \"you\" has no
other referent) — otherwise the subject cannot be determined and the guard declines
rather than guess. Checking the claim against real state is also strictly more
permissive where permissiveness is safe: a figure the model merely recalls correctly,
including one a purchase confirmed on an earlier turn, still ratifies, because it is
still true right now. An item claim has no single figure to check the same way, so it
instead requires the caller's own observation that the *named character's own*
equipment differs from a snapshot taken at the start of the turn. A narration stating
no mechanical fact, or a verifiably true one about the character it actually names,
ratifies exactly as before; the guard changes nothing about the common case this
module's docstring above already covers.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from narrator.shape import example_json

SWEEPER_SYSTEM_PROMPT = (
    "You keep a game's scene record. After a delivered turn that wrote no record "
    "entry, you copy any durable change the narration itself states into one entry, "
    "or you decline. You transcribe; you never invent."
)


class SweepRecord(BaseModel):
    """Copy the narration's durable changes into one scene entry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    kind: Literal["record"]
    public_summary: str = Field(
        min_length=1,
        max_length=600,
        description="One or two sentences of durable change, copied from the narration.",
    )
    visible_changes: list[Annotated[str, StringConstraints(max_length=240)]] = Field(
        default_factory=list,
        max_length=4,
        description="Player-visible changes, one clause each.",
    )


class SweepNone(BaseModel):
    """Decline: the narration states nothing durable. The common case."""

    kind: Literal["none"]


class SweepOutcome(BaseModel):
    """Exactly one of: a record, or a decline."""

    outcome: SweepRecord | SweepNone = Field(discriminator="kind")


OBJECT_CLAIMS: frozenset[str] = frozenset({"locked", "unlocked", "barred", "open", "closed"})


class SweepMention(BaseModel):
    """One reference the narration made to a recorded identifier, with any state it
    asserted. In-memory guard input, never a campaign record."""

    model_config = ConfigDict(str_strip_whitespace=True)

    entity_id: Annotated[str, StringConstraints(max_length=64)]
    kind: Literal["npc", "character", "object", "exit"]
    claim: Literal["none", "alive", "dead", "absent", "locked", "unlocked", "barred", "open", "closed"] = "none"


class SweepEntityRecord(SweepRecord):
    """``SweepRecord`` plus the identifiers the narration referred to.

    A subclass rather than a field on ``SweepRecord`` because the OpenAI client's
    strict schema marks every property required: the arm that runs without
    entities must send today's ``SweepOutcome`` bytes unchanged, so the legacy class
    cannot grow a field. Parent fields come first in the schema, so the record text
    is answered before the mentions.
    """

    mentions: list[SweepMention] = Field(
        default_factory=list,
        max_length=12,
        description="Recorded identifiers the narration refers to, with any state it asserts.",
    )


    claimed_coins: int | None = Field(
        default=None, ge=0, le=99999, examples=[40],
        description="The coin total the record states for a party member, or null.",
    )
    claimed_hp: int | None = Field(
        default=None, ge=0, le=999, examples=[9],
        description="The hit-point total the record states for a party member, or null.",
    )


class SweepIntroducingRecord(SweepEntityRecord):
    """``SweepEntityRecord`` plus the persons the narration introduced (Slice B)."""

    introduced_persons: list[Annotated[str, StringConstraints(min_length=1, max_length=60)]] = Field(
        default_factory=list,
        max_length=4,
        description="People the narration names or describes whom no recorded identifier covers.",
    )


class SweepEntityOutcome(BaseModel):
    """Exactly one of: a record with mentions, or a decline."""

    outcome: SweepEntityRecord | SweepNone = Field(discriminator="kind")


class SweepIntroducingOutcome(BaseModel):
    """Exactly one of: a record with mentions and introductions, or a decline."""

    outcome: SweepIntroducingRecord | SweepNone = Field(discriminator="kind")


def empty_roster() -> dict:
    """The roster shape ``render_roster`` and ``validated_mentions`` read.

    ``npcs`` maps id to lifecycle status, ``persons`` and ``characters`` map id to
    display name, ``objects`` maps id to barrier state, ``exits`` lists ids. Every
    value comes from recorded campaign state; the engine's ``_entity_roster`` reads
    them and nothing model-generated is ever placed here.
    """
    return {
        "npcs": {}, "persons": {}, "characters": {}, "objects": {}, "exits": [],
        # Second iteration: display names of recorded NPCs (id to name) for the
        # engine's introduction cross-check, and the scene's current summary so
        # the arrival guard can keep a record's facts under it (change 3). Neither
        # is rendered into the prompt.
        "npc_names": {}, "summary": "",
    }


def roster_ids(roster: dict, kind: str) -> set[str]:
    """The identifiers a mention of ``kind`` may name."""
    if kind == "npc":
        return set(roster.get("npcs", {})) | set(roster.get("persons", {}))
    if kind == "character":
        return set(roster.get("characters", {}))
    if kind == "object":
        return set(roster.get("objects", {}))
    if kind == "exit":
        return set(roster.get("exits", []))
    return set()


def validated_mentions(mentions, roster: dict) -> list[SweepMention]:
    """Keep only mentions whose identifier the roster lists under that kind.

    The model copies ids from lists the engine wrote; anything else is a
    hallucination and is dropped, exactly as ``narrator.classify._validated_cue``
    drops an invented interlocutor. Duplicates collapse to the first occurrence.
    """
    kept: list[SweepMention] = []
    seen: set[tuple[str, str]] = set()
    for mention in mentions or ():
        key = (mention.kind, mention.entity_id)
        if key in seen or mention.entity_id not in roster_ids(roster, mention.kind):
            continue
        seen.add(key)
        kept.append(mention)
    return kept


_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """``bsh_mcp.models.slugify``, reproduced: the narrator package imports nothing
    from ``bsh_mcp`` (the two interpreters split on ``mcp``'s major version), the same
    reason ``narrator.engine`` duplicates ``TRAVERSAL_CLOCK_PREFIX``."""
    slug = _SLUG_SEPARATORS.sub("-", value.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:64] or "unnamed"


def _label_names(label: str, name: str) -> bool:
    """Whether ``name`` occurs in ``label`` as a whole word, case-folded; the same
    rule ``bsh_mcp.service.resolve_person_label`` applies on the server."""
    name = name.strip()
    if len(name) < 2:
        return False
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", label, re.IGNORECASE) is not None


def introduced_person_names(labels, roster: dict) -> list[str]:
    """The introduced-person labels that name no NPC and no party member.

    Slice B's cross-check, as a cheap filter before the server's authoritative
    resolution: a label whose slug matches a recorded NPC or party member, or that
    carries one of their names as a whole word ("Rade, the fishmonger"), is the model
    re-describing someone known and is dropped rather than sent on. A label naming a
    recorded *person* or an authored index entry is kept: the server updates or
    converges it (``resolve_person_label``). Duplicates within the turn collapse to
    the first spelling.
    """
    known_ids = set(roster.get("npcs", {})) | set(roster.get("characters", {}))
    known_names = list(roster.get("npc_names", {}).values()) + list(roster.get("characters", {}).values())
    kept: list[str] = []
    seen: set[str] = set()
    for label in labels or ():
        name = str(label).strip()
        slug = _slug(name)
        if not name or slug in known_ids or slug in seen:
            continue
        if any(_label_names(name, known) for known in known_names):
            continue
        seen.add(slug)
        kept.append(name)
    return kept


def departed_person_ids(mentions, roster: dict) -> list[str]:
    """The recorded *persons* the record states have left the scene.

    Persons only: a recorded NPC's presence stays tool-driven (a fight roster, a
    location), and an ``absent`` claim about one is ignored here. Removal of a person
    is the safe direction -- a person feeds no hazard rule, and a departed one draws
    the consent ask again if named -- so narration may drive it, through the same
    ``scene_commit`` argument the model has (rule 1 of the record).
    """
    persons = set(roster.get("persons", {}))
    departed: list[str] = []
    for mention in mentions or ():
        if mention.claim == "absent" and mention.entity_id in persons and mention.entity_id not in departed:
            departed.append(mention.entity_id)
    return departed


def render_roster(roster: dict) -> list[str]:
    """The roster block: recorded identifiers the model may copy into ``mentions``.

    Each list is rendered even when empty, so the model is told a kind has nothing to
    name rather than left to guess whether the list was omitted.
    """
    people = [f"{npc_id} (npc, {status or 'status unknown'})" for npc_id, status in roster.get("npcs", {}).items()]
    people += [f"{person_id} (person)" for person_id in roster.get("persons", {})]
    objects = [f"{object_id} ({state})" for object_id, state in roster.get("objects", {}).items()]
    party = list(roster.get("characters", {}))
    exits = list(roster.get("exits", []))
    join = lambda items: ", ".join(items) if items else "(none)"  # noqa: E731 - one-line formatter
    return [
        "Recorded in the scene, by identifier:",
        f"  people: {join(people)}",
        f"  party: {join(party)}",
        f"  objects: {join(objects)}",
        f"  exits: {join(exits)}",
    ]


#: A stated coin figure: "40 coins", "3 copper", "12 gp". Captures the integer so the
#: live retcon probe can compare the exact figure, not merely detect the category; the
#: sweep guard below only needs the boolean. Comma-grouped figures ("1,200 coins")
#: parse too, because a live model has produced them.
_COIN_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:coins?|copper|silver|gold|cp|sp|gp)\b", re.IGNORECASE)

#: A stated hit-point figure: "7 hit points", "9 HP", "at 3 hp". Captures the integer
#: for the same reason ``_COIN_PATTERN`` does: the guard verifies the exact figure,
#: not merely that one was stated.
_HP_PATTERN = re.compile(r"\b(\d+)\s*(?:hit points?|hp)\b", re.IGNORECASE)

#: An item gained or lost, stated as a possession change. Deliberately narrow: this
#: names the verbs the sweep prompt's own instructions already use for "an object
#: gained, lost, or hidden" so detection stays anchored to the sweep's own vocabulary
#: rather than trying to parse arbitrary narration for inventory changes.
_ITEM_CLAIM_PATTERN = re.compile(
    r"\b(?:now (?:has|have|carries|carry|holds|hold)|gains?|loses?|"
    r"picks? up|drops?|hands? over|steals?)\b",
    re.IGNORECASE,
)


def stated_coin_totals(text: str) -> list[int]:
    """Every coin figure the text's own words state, as plain integers.

    Reads text alone, exactly like the rest of this module's contract: this never
    asks whether a figure is true, only what the text says. Used by
    ``narration_claims_mechanical_fact`` below, by ``guard_ratification`` to verify a
    record's own claim, and directly by the live retcon probe, which compares each
    figure against the buyer's actual ``coins`` at the same point in the run rather
    than merely detecting that a figure was stated.
    """
    return [int(match.group(1).replace(",", "")) for match in _COIN_PATTERN.finditer(text)]


def stated_hp_totals(text: str) -> list[int]:
    """Every hit-point figure the text's own words state, as plain integers.

    The hit-point counterpart to ``stated_coin_totals``, read the same way and used
    the same two places: category detection below, and ``guard_ratification``'s own
    verification of a record's claimed figure against the real current value.
    """
    return [int(match.group(1)) for match in _HP_PATTERN.finditer(text)]


def narration_claims_mechanical_fact(text: str) -> str:
    """The mechanical-fact category the text's own words assert, or "".

    Returns ``"coins"``, ``"hp"``, or ``"item"`` for the first category matched, in
    that order, or ``""`` when the text states none of them. Detection stays lexical
    and narrow by design: a false negative here costs nothing the sweep's own "none
    is the common, correct answer" contract does not already cover, while a false
    positive would decline text the sweep should keep — the fixture cases
    ``guard_ratification`` is pinned against. Callers pass the specific text whose
    claim matters: ``guard_ratification`` passes the record's own words, never a
    whole turn's narration, so an unrelated mention elsewhere in the same turn cannot
    trigger a decline on text that makes no claim itself.
    """
    if _COIN_PATTERN.search(text):
        return "coins"
    if _HP_PATTERN.search(text):
        return "hp"
    if _ITEM_CLAIM_PATTERN.search(text):
        return "item"
    return ""


_LIFE_LINKING_PATTERN = re.compile(
    r"\b(?:is|are|was|were|remains?|stands?|stays?)\s+((?:\w+\s+){0,2}?)(alive|dead)\b",
    re.IGNORECASE,
)

#: A death stated as a verb: "killed", "slain", "dies". The same narrow-vocabulary
#: posture as ``_ITEM_CLAIM_PATTERN``: these are the words a kill narration actually
#: uses, not an attempt to parse arbitrary prose for mortality.
_LIFE_DEATH_VERB_PATTERN = re.compile(
    r"\b(?:kill(?:s|ed|ing)?|slay(?:s|ing)?|slain|slew|dies|died|perish(?:es|ed)?)\b",
    re.IGNORECASE,
)

#: A claimed arrival: "arrives", "enters", "reaches" a place. The same narrow-vocabulary
#: posture as ``_ITEM_CLAIM_PATTERN`` and ``_LIFE_DEATH_VERB_PATTERN``: these are the
#: words an arrival narration actually uses. Load-bearing for ``guard_ratification``'s
#: own contract rather than any lexical subtlety: the sweep step that reads this text
#: fires only when the scene record provably did not change this turn (``run_turn``'s
#: own fingerprint gate), and a genuine arrival always changes it -- a new
#: ``location_id`` marks the scene dirty on its own. So this claim is never true of a
#: turn the sweep ever actually reaches; there is no scene to compare the claimed place
#: against, because the claim itself is what the sweep's own precondition already rules
#: out.
_ARRIVAL_CLAIM_PATTERN = re.compile(
    r"\b(?:arrives?|arrived|enters?|entered|entering|reaches|reached)\b", re.IGNORECASE
)


def states_new_arrival(text: str) -> bool:
    """Whether the text's own words claim the party arrived or entered somewhere.
    """
    return bool(_ARRIVAL_CLAIM_PATTERN.search(text))


#: Subjects that name the party as a whole. Read only by ``party_arrival`` below, and
#: only beside the real character names the roster supplies; it is a subtraction of
#: pronouns and collective nouns, not a vocabulary of referents.
_PARTY_SUBJECT_WORDS = frozenset(
    {"party", "players", "group", "characters", "adventurers", "companions", "you", "we", "they", "everyone"}
)


def party_arrival(text: str, party_names) -> bool:
    """Whether an arrival claim in the text has the *party* as its subject.
    """
    names = {str(name).casefold() for name in party_names if str(name).strip()}
    for match in _ARRIVAL_CLAIM_PATTERN.finditer(text):
        head = text[: match.start()]
        sentence = re.split(r"[.!?;\n]", head)[-1]
        words = [word.casefold() for word in re.findall(r"[^\W\d_]+", sentence)][-6:]
        if any(word in _PARTY_SUBJECT_WORDS or word in names for word in words):
            return True
        joined = " ".join(words)
        if any(name in joined for name in names if " " in name):
            return True
    return False


def stated_life_statuses(text: str) -> list[str]:
    """Every life status the text's own words assert, as ``"alive"``/``"dead"``.

    Read the same way as ``stated_coin_totals``: text alone, never truth. A negated
    linking claim flips polarity ("is not dead" asserts alive; "is no longer alive"
    asserts dead), and a death verb contributes one ``"dead"`` regardless of
    phrasing. Used by ``guard_ratification`` to verify a record's life claim about a
    named NPC against that NPC's real recorded status.
    """
    statuses: list[str] = []
    for match in _LIFE_LINKING_PATTERN.finditer(text):
        polarity = match.group(2).casefold()
        qualifiers = match.group(1).casefold().split()
        if "not" in qualifiers or "no" in qualifiers:
            polarity = "alive" if polarity == "dead" else "dead"
        statuses.append(polarity)
    if _LIFE_DEATH_VERB_PATTERN.search(text):
        statuses.append("dead")
    return statuses


def _named_npcs(claim_text: str, npcs: dict) -> list[dict]:
    """Every NPC the claim text names by id or name, as a whole word.

    Unlike ``_match_character`` this carries no lone-member fallback: "you" in a
    sweep record refers to a player character, never to the campaign's only NPC, so
    a life claim naming no recorded NPC simply stays outside the guard's reach.
    """
    named: list[dict] = []
    for npc in npcs.values():
        candidates = {
            str(npc.get("id", "")).casefold(),
            str(npc.get("name", "")).casefold(),
        }
        candidates.discard("")
        if any(
            re.search(rf"\b{re.escape(candidate)}\b", claim_text, re.IGNORECASE)
            for candidate in candidates
        ):
            named.append(npc)
    return named


def _record_claim_text(record: SweepRecord) -> str:
    """The record's own player-visible words: what is actually headed for canon.

    ``public_summary`` and ``visible_changes`` are the only fields ``_sweep`` copies
    into ``scene_commit``; nothing outside them can become the ratified fact, so
    nothing outside them should decide whether ratification is safe.
    """
    return " ".join([record.public_summary, *record.visible_changes])


def _match_character(claim_text: str, characters: dict) -> dict | None:
    """The one character a claim names, or ``None`` when that cannot be determined.

    ``characters`` maps character id to a dict carrying at least ``id`` and
    ``name``. Each character's id and name are matched as a whole word against the
    claim text, case-insensitively; exactly one character matching identifies the
    claim's subject. A claim naming no one resolves only when the party holds
    exactly one character — "you now have 3 coins" has no other referent when
    there is only one character to be "you" — because guessing among several would
    reproduce the pooled-state defect this function exists to close. Zero
    resolvable characters, or more than one matching by name, both return ``None``:
    the caller must decline rather than verify against the wrong entity, or an
    ambiguous one.
    """
    if not characters:
        return None
    matches = []
    for character in characters.values():
        candidates = {
            str(character.get("id", "")).casefold(),
            str(character.get("name", "")).casefold(),
        }
        candidates.discard("")
        if any(
            re.search(rf"\b{re.escape(candidate)}\b", claim_text, re.IGNORECASE)
            for candidate in candidates
        ):
            matches.append(character)
    if len(matches) == 1:
        return matches[0]
    if not matches and len(characters) == 1:
        return next(iter(characters.values()))
    return None


def guard_ratification(
    outcome: SweepOutcome,
    *,
    characters_now: dict | None = None,
    characters_before: dict | None = None,
    npcs_now: dict | None = None,
    objects_now: dict | None = None,
    scene_summary: str = "",
) -> SweepOutcome:
    """Downgrade a record this turn's actual character-sheet state cannot back.

    Detection scopes to the record's own text (see ``_record_claim_text``), never the
    turn's full narration. Verification reads real state the caller supplies fresh at
    ratification time — ``characters_now`` maps character id to that character's
    current ``coins``, ``hp``, and ``equipment``; ``characters_before`` is the same
    shape, snapshotted at the start of the turn, for the equipment comparison — rather
    than inferring causality from tool-call success, which has no necessary
    connection to the specific figure claimed.

    A coin or hit-point claim first resolves which character it names
    (``_match_character``) and then checks every figure the claim states against
    that character's own real value; a claim whose subject cannot be resolved
    declines, because verifying against the wrong character, or against the whole
    party's pooled state, is not verification at all. An item claim declines the
    same way when its subject cannot be resolved, and otherwise ratifies only when
    that specific character's own equipment differs from the turn-start snapshot. A
    record making no mechanical claim, or one whose claim verifies true for the
    character it names, ratifies exactly as the model decided; this is one further
    disqualifying condition on the existing record/decline choice, not a
    replacement for it.
    """
    if outcome.outcome.kind != "record":
        return outcome
    record = outcome.outcome
    claim_text = _record_claim_text(record)

    salvaged = _salvage_party_arrival(outcome, record, claim_text, characters_now, scene_summary)
    if salvaged is None:
        return _declined()
    outcome, record, claim_text = salvaged
    mentions = list(getattr(record, "mentions", ()) or ())
    if not _mention_claims_hold(mentions, npcs_now, objects_now):
        return _declined()
    if not _regex_life_holds(claim_text, npcs_now):
        return _declined()
    return _numeric_verdict(
        outcome, record, claim_text, mentions,
        characters_now=characters_now, characters_before=characters_before,
    )


# -- guard_ratification's checks, in the driver's order ------------------------------
#
# The guard grew to 180 lines as each measured union arm went inline; these are its
# mechanical decomposition. Each check owns its own relocated commentary, error
# budget and union arms; the driver above owns only the order. Behavior is
# byte-for-byte the inline original's, held by the guard's own pins.


def _declined() -> SweepOutcome:
    """A fresh decline, never a shared instance a caller could mutate."""
    return SweepOutcome(outcome=SweepNone(kind="none"))


def _salvage_party_arrival(outcome, record, claim_text, characters_now, scene_summary):
    """The arrival check with the entity record's salvage.

    Returns ``None`` to decline, otherwise the (possibly rewritten) outcome, record
    and claim text -- the one check that may modify the record rather than only
    judge it, which is why it runs first and hands its result to every later check.
    """
    if states_new_arrival(claim_text):
        if not hasattr(record, "mentions"):
            return None
        # An entity record carries a roster: decline the party's own arrival, never a
        # stranger's (``party_arrival``). Names come from the recorded character
        # sheets the caller supplies, never from the record.
        party = [
            value
            for character in (characters_now or {}).values()
            for value in (character.get("id", ""), character.get("name", ""))
        ]
        if party_arrival(claim_text, party):


            survivors = [
                change
                for change in record.visible_changes
                if not (states_new_arrival(change) and party_arrival(change, party))
            ]
            if not survivors or not scene_summary.strip():
                return None
            record = record.model_copy(
                update={"public_summary": scene_summary.strip(), "visible_changes": survivors}
            )
            outcome = outcome.model_copy(update={"outcome": record})
            claim_text = _record_claim_text(record)


    return outcome, record, claim_text


def _mention_claims_hold(mentions, npcs_now, objects_now) -> bool:
    """The typed union arm: id-resolved life and barrier claims against real state."""
    for mention in mentions:
        if mention.kind == "npc" and mention.claim in ("alive", "dead"):
            npc = (npcs_now or {}).get(mention.entity_id)
            if npc is None:
                continue
            actual = "dead" if str(npc.get("status", "")).casefold() == "dead" else "alive"
            if actual != mention.claim:
                return False
        elif mention.kind == "object" and mention.claim in OBJECT_CLAIMS:
            actual_state = (objects_now or {}).get(mention.entity_id)
            if actual_state is not None and str(actual_state) != mention.claim:
                return False

    return True


def _regex_life_holds(claim_text, npcs_now) -> bool:
    """Regex life holds.
    """
    claimed_life = stated_life_statuses(claim_text)
    if claimed_life:
        named = _named_npcs(claim_text, npcs_now or {})
        if named:
            polarities = set(claimed_life)
            backed_life = len(polarities) == 1 and all(
                ("dead" if str(npc.get("status", "")).casefold() == "dead" else "alive")
                in polarities
                for npc in named
            )
            if not backed_life:
                return False

    return True


def _numeric_verdict(outcome, record, claim_text, mentions, *, characters_now, characters_before):
    """Coin, hit-point and item claims: regex and typed figures as one union, the
    narrowed digit tripwire, and subject resolution by name or single mention."""
    category = narration_claims_mechanical_fact(claim_text)
    claimed_coins = getattr(record, "claimed_coins", None)
    claimed_hp = getattr(record, "claimed_hp", None)
    if not category:
        # The typed fields are the language-independent half of the union (spec §3.5):
        # a record the regexes read nothing from may still state a figure the model
        # extracted. And the digit tripwire (spec §3.2), narrowed: a digit in a record
        # that names a party member, with no figure extracted by either signal, is
        # unverified and declines -- the model reporting on its own record must not
        # be able to hide a figure by phrasing it. A digit about anything else ("high
        # water 2 hours after sunset", the authored tide-table fact) ratifies: the
        # spec's unnarrowed rule would have declined exactly the clue the continuity
        # probe plants.
        if claimed_coins is not None:
            category = "coins"
        elif claimed_hp is not None:
            category = "hp"
        elif hasattr(record, "mentions") and re.search(r"\d", claim_text) and any(
            mention.kind == "character" for mention in mentions
        ):
            return SweepOutcome(outcome=SweepNone(kind="none"))
        else:
            return outcome
    characters_now = characters_now or {}
    characters_before = characters_before or {}
    character = _match_character(claim_text, characters_now)
    if character is None:
        # The regex path resolves Latin-script names only (``\b`` matches nothing in
        # Japanese). A record whose mentions name exactly one character resolves the
        # subject by identifier instead; two or more stay ambiguous and decline.
        named_characters = {
            mention.entity_id
            for mention in mentions
            if mention.kind == "character" and mention.entity_id in characters_now
        }
        if len(named_characters) == 1:
            character = characters_now[next(iter(named_characters))]
    if character is None:
        return SweepOutcome(outcome=SweepNone(kind="none"))
    if category == "coins":
        claimed = stated_coin_totals(claim_text) + ([claimed_coins] if claimed_coins is not None else [])
        backed = bool(claimed) and all(value == character.get("coins") for value in claimed)
    elif category == "hp":
        claimed = stated_hp_totals(claim_text) + ([claimed_hp] if claimed_hp is not None else [])
        backed = bool(claimed) and all(value == character.get("hp") for value in claimed)
    else:
        before = characters_before.get(character.get("id"))
        backed = before is not None and before.get("equipment") != character.get("equipment")
    if backed:
        return outcome
    return SweepOutcome(outcome=SweepNone(kind="none"))


def sweep_prompt(narration: str, roster: dict) -> str:
    """Render the delivered narration into the sweeper's user message.
    """
    categories = (
        "discovery made, a place entered or left, an object gained, lost, or "
        "hidden, an agreement struck, an injury taken, "
        "a person newly named or described in the scene, "
        "or a fixed physical "
        "detail newly described (an object's mounting, structure, or placement "
        "in the scene). "
    )
    lines = [
        "This turn's narration was delivered to the players, and the scene "
        "record did not change:",
        "",
        narration.strip() or "(the turn produced no deliverable narration)",
        "",
        "Decide whether that narration itself states a durable change: a "
        + categories
        + "Copy durable changes "
        "from the narration's own words; record nothing the narration does not "
        "state. Banter, atmosphere, and unresolved intentions are not durable. "
        "When nothing durable happened, answer none — that is the common case "
        "and the correct answer for most turns.",
        "",
    ]
    # Derived from the schema (``narrator.shape.example_json``) rather than a second,
    # hand-typed copy of it, so the two cannot drift apart silently; computed once,
    # ahead of the prose paragraphs below that describe the same fields in order.
    record_shape = f'{{"outcome": {example_json(SweepIntroducingRecord)}}}'
    decline_shape = f'{{"outcome": {example_json(SweepNone)}}}'
    lines += render_roster(roster)
    lines += [
        "",


        "mentions: every recorded identifier above that your record (public_summary "
        "and visible_changes) refers to, copied exactly from these lists, each with "
        "the state your record asserts for it — alive or dead for a person, absent "
        "when your record states a person has left the scene, locked, unlocked, "
        "barred, open, or closed for an object — or none when your record asserts "
        "no state. Empty when your record refers to none of them.",
    ]
    lines += [
        "claimed_coins and claimed_hp: the coin total and the hit-point total your "
        "record states for a party member, as plain integers, or null when your "
        "record states none. A number about anything else — hours, distances, a "
        "clock — is null here.",
    ]
    lines += [
        "introduced_persons: a person the narration names or describes whom no "
        "identifier above covers, in the narration's own words — one individual "
        "per entry; never a group, a crowd, or a kind of people. Empty when "
        "nobody new appears.",
    ]
    lines.append("")
    return "\n".join(
        lines
        + [


            "Answer with one JSON object, every field present, in this exact "
            "order. To record: " + record_shape + " — visible_changes may be an empty "
            "list, but the field must appear. To decline: " + decline_shape + ".",
        ]
    )
