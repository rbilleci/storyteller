"""Scorer, gate, and transcript-generation logic for the soak harness.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from narrator.delivery import _ARG_DELIMITER, _CLOSERS, _OPENERS

REPO_ROOT = Path(__file__).resolve().parents[2]

PLAYERS = ("Rill", "Ossa")

#: The five locations shipped in ``world/locations/index.yaml`` and the recurring cast.
#: Grounding the chatter in real identifiers keeps the narrator's tool calls resolvable.
LOCATIONS = (
    "the eel market",
    "the road shrine",
    "the drowned customs house",
    "the black bell crypt",
    "the river cave",
)
FIGURES = ("Sera Vane", "the keeper", "the salt magistrate", "the fishmongers", "the ferryman")
OBJECTS = ("the torn ledger", "the bell fragment", "the salt writ", "the eel knife", "the crypt key")

_CHATTER = (
    "{other} is right, {figure} knows more than they said about {object}.",
    "keep your voice down near {place}, the walls carry.",
    "we still owe the ferryman for the last crossing.",
    "my torch is guttering, note the usage die before we go deeper.",
    "if {figure} lied about {object}, the trail starts at {place}.",
    "I marked the tide line, we have until it turns.",
    "{other}, watch the doom die, you have tempted it twice today.",
    "the mud took my boot at {place} last time, walk the planks.",
    "ration check, we split the last dried eel this morning.",
    "I say we sell {object} and be done with this town.",
    "no, {object} is the only leverage we hold over {figure}.",
    "did you hear bells last night, or was that the wind again.",
    "count the exits before we commit, {place} only has two.",
    "{other}, your bowstring is frayed, deal with it before trouble.",
    "the magistrate's guards double at dusk, we move before then.",
    "I trust {figure} exactly as far as the coin they owe us.",
    "sketch the route, {place} floods at high water.",
    "we should rest before this, my hit points are not what they were.",
    "quiet, someone is listening from the stalls.",
    "the writ names a bearer, it does not name us.",
    "whatever is in the crypt, it rang that bell for a reason.",
    "one plan, {other}. we do not split up again.",
    "the salt in the air is worse near {place}, cover the rations.",
    "if this goes wrong, we meet at the ferry stone.",
)


_MENTIONS = (
    "@GM {actor} leads the way to {place}. What do we find when we arrive?",
    "@GM {actor} searches {place} for any trace of {object}. Roll whatever is needed.",
    "@GM {actor} approaches {figure} and asks about {object}, keeping it civil.",
    "@GM {actor} checks the torch. Roll its usage die before we go on.",
    "@GM we take a short rest and watch the water. Does anything find us?",
    "@GM both of us listen at the door of {place}. Test whoever is better suited.",
)


#: The continuity probe's fact tokens. Chosen by corpus search: zero occurrences in
#: rules/, world/, skills/, the chatter and mention pools, and the recall text below,
#: so a pass can come only from the narrator producing them at the recall turn. The
#: plant mention necessarily contains them; the replay grammar guarantees the recall
#: turn's backfill starts after the previous mention, so the plant text cannot ride
#: into the recall turn's input.
CONTINUITY_TOKENS: tuple[str, ...] = ("tin whistle", "salt-cellar")

CONTINUITY_PLANT = (
    "@GM {actor} hides the cracked tin whistle inside the old salt-cellar beneath "
    "the customs house floor, and marks the spot. Record this hiding place; it must "
    "survive whatever comes."
)

#: Deliberately token-free: it names neither the item nor the place, so the narrator
#: must supply both from canon, not from echoing the question.
CONTINUITY_RECALL = (
    "@GM {actor} doubles back alone to retrieve the thing we hid earlier. Say exactly "
    "what the hidden item is and exactly where it lies."
)


AUTHORED_TOKENS: tuple[str, ...] = ("two hours after sunset",)


AUTHORED_SUBJECT_TERMS: tuple[str, ...] = ("high water", "high tide")
AUTHORED_SUNSET_TERMS: tuple[str, ...] = (
    "sunset", "sundown", "sun has dipped", "sun dips", "sun sets", "sun has set",
    "setting of the sun",
)
AUTHORED_INTERVAL_TERMS: tuple[str, ...] = ("two", "2", "second")
#: The authored sentence places high water after sunset. A sentence carrying this term
#: states some other relation, so the gate rejects it rather than scoring the interval
#: alone. The same audit executed "Tonight's high water arrives two hours before sunset."
#: against a rule without it, and that rule returned true for a reversed fact.
AUTHORED_EXCLUDED_TERMS: tuple[str, ...] = ("before",)

AUTHORED_RECALL = (
    "@GM {actor} reads the tide notice nailed to the bell-tower door aloud, word for "
    "word. When exactly does tonight's high water arrive?"
)


CLOCK_ID = "brackwater-levy"
CLOCK_NAME = "brackwater levy"
CLOCK_NOTE = "hulls counted at the mole"
CLOCK_SEGMENTS = 6
CLOCK_FILLED = 3
CLOCK_TOKENS: tuple[str, ...] = (CLOCK_NAME, CLOCK_NOTE)

#: The one served tool that reports clock state (``tests/test_tools.py``). A reply
#: naming the clock after this tool ran proves retrieval, not digest delivery, so the
#: probe records the turn's tool names beside the score instead of assuming either.
CLOCK_READ_TOOLS: tuple[str, ...] = ("campaign_status",)

#: Token-free by construction: it names neither the clock, nor its note, nor its fill.
CLOCK_RECALL = (
    "@GM {actor} calls a halt to take stock before anything else happens. Name every "
    "countdown the table is tracking against us right now, exactly as the record "
    "words it, and say how many segments each one has filled."
)


ZONE_B_TOKENS: tuple[str, ...] = ("wicker eel-trap", "third piling")
ZONE_C_TOKENS: tuple[str, ...] = ("tide-medal", "waxed pouch")
ZONE_A_TOKENS: tuple[str, ...] = ("rope ladder", "canvas sleeve")


ZONE_B_FACT = (
    "Before the venture began the party stowed a wicker eel-trap under the third "
    "piling of the fish dock."
)

ZONE_C_PLANT = (
    "@GM {actor} threads the tide-medal into the waxed pouch at her belt and knots it "
    "shut. Record this; it must survive whatever comes."
)

ZONE_A_PLANT = (
    "@GM {actor} coils the rope ladder into the canvas sleeve on her back and buckles "
    "it down. Record this; we come back for it."
)

#: Each question names its plant by circumstance and carries zero scored tokens.
ZONE_B_RECALL = (
    "@GM {actor} thinks back to the very first thing this party stowed away, before "
    "any of this began. Say exactly what it was and exactly where it sits."
)

ZONE_C_RECALL = (
    "@GM {actor} asks about the small thing one of us knotted shut at the belt "
    "partway through the venture. Say exactly what it is and exactly what holds it."
)

ZONE_A_RECALL = (
    "@GM {actor} asks about the thing we stowed a few moments ago. Say exactly what it "
    "is and exactly what holds it."
)


ORDER_OBJECT = "lacquered oar-case"
ORDER_PLACES: tuple[str, ...] = ("oilcloth wrap", "bone-clasp case", "tarred tube")

#: The first resting place plants at bootstrap, which puts it at the head of the
#: visible-facts list and therefore inside the retained block at every depth. The two
#: moves plant by mention at mid-session turns, which places them in the band where a
#: superseded digest is the only carrier: too old for the conversation window at the
#: recall turn, and past the retained head once the list grows.
ORDER_MOVE_FIRST = (
    "@GM {actor} lifts the lacquered oar-case out of what holds it and seals it into "
    "the bone-clasp case instead. Record the move; we must know what holds it."
)

ORDER_MOVE_SECOND = (
    "@GM {actor} lifts the lacquered oar-case again and seals it into the tarred tube. "
    "Record the move; we must know what holds it."
)

#: Token-free against the score: it names the object, which carries no score, and none
#: of the three places, which carry all of it.
ORDER_RECALL = (
    "@GM {actor} asks the table to account for the lacquered oar-case. Name every "
    "container it has rested in, in the order it moved, from the first to what holds "
    "it now."
)


EXCHANGE_OBJECT = "brass fid"
EXCHANGE_GIVER, EXCHANGE_TAKER = PLAYERS
EXCHANGE_FINAL_HOLDER = EXCHANGE_TAKER


EXCHANGE_GIVE_FIRST = (
    f"@GM {EXCHANGE_GIVER} hands the {EXCHANGE_OBJECT} to {EXCHANGE_TAKER} and lets go "
    "of it. Record who carries it now."
)

EXCHANGE_RETURN = (
    f"@GM {EXCHANGE_TAKER} hands the {EXCHANGE_OBJECT} back to {EXCHANGE_GIVER} and "
    "lets go of it. Record who carries it now."
)

EXCHANGE_GIVE_SECOND = (
    f"@GM {EXCHANGE_GIVER} hands the {EXCHANGE_OBJECT} to {EXCHANGE_TAKER} a second "
    "time and lets go of it. Record who carries it now."
)


HAZARD_STRIKE_PERSON = "@GM Rill punches Orso Pell in the face."
HAZARD_STRIKE_UNRECORDED = "@GM Ossa shoves the fishmonger off the planks into the mud."
HAZARD_LOOT_CORPSE = "@GM Rill searches the dead guard's pockets and takes his purse."
HAZARD_PLANTS: dict[str, tuple[str, str, str]] = {
    # name: (text, target kind, what the ask is)
    "hazard_strike_person": (HAZARD_STRIKE_PERSON, "person", "owed-first-strike"),
    "hazard_strike_unrecorded": (HAZARD_STRIKE_UNRECORDED, "unrecorded-role", "owed-first-strike"),
    "hazard_loot_corpse": (HAZARD_LOOT_CORPSE, "unrecorded-corpse", "spurious-if-asked"),
}


PREFILL_FACT = (
    "Prefilled record entry {index:02d}: the party logged a durable change at the "
    "eel market."
)


def authored_fact_stated(reply: str) -> bool:
    """Whether one sentence of the reply states when high water arrives.
    """
    interval = re.compile(r"\b(?:" + "|".join(AUTHORED_INTERVAL_TERMS) + r")\b")
    sentences = re.split(r"[.!?\n]+", reply.lower())
    for index, sentence in enumerate(sentences):
        if any(term in sentence for term in AUTHORED_EXCLUDED_TERMS):
            continue
        if not (
            any(term in sentence for term in AUTHORED_SUNSET_TERMS)
            and interval.search(sentence)
        ):
            continue
        window = sentences[max(0, index - 1):index + 1]
        if any(term in text for text in window for term in AUTHORED_SUBJECT_TERMS):
            return True
    return False


AUTHORED_QUOTE_CUES: tuple[str, ...] = (
    "read", "reads", "reading", "states", "stated", "notice", "table", "aloud", "words",
    "wording", "shows", "says", "said", "written", "writes", "inscription", "inscribed",
)


AUTHORED_QUOTE_CUE_WINDOW = 120

#: A span the reply delimits as reproduced text: straight double quotes, curly double
#: quotes, or markdown emphasis. Single quotes stay out because an apostrophe opens one in
#: ordinary prose. Newlines stay out because a pattern crossing them joins two unrelated
#: delimiters into one span, which invents a quotation the reply never wrote. Both choices
#: undercount rather than overcount, so a measured rate reads as a floor.
_AUTHORED_QUOTE_SPAN = re.compile(
    r'"([^"\n]{4,300})"'
    r"|“([^”\n]{4,300})”"
    r"|\*([^*\n]{4,300})\*"
    r"|_([^_\n]{4,300})_"
)

#: Joins two canon sources so no normalized match spans both. A model reply reproducing
#: this token would have to type it, and no authored file contains it.
_CANON_JOIN = "qqsepqq"


def authored_quotation_spans(reply: str, canon: str) -> dict:
    """Score every span the reply presents as a written source's own words.

    A span counts as attributed when one ``AUTHORED_QUOTE_CUES`` word sits in the
    ``AUTHORED_QUOTE_CUE_WINDOW`` characters before its opening delimiter. It counts as
    word-supported when its normalized word sequence appears in ``canon``, and as
    verbatim-supported when its exact characters do. ``fabricated`` reports an attributed
    span whose words canon never states, which is invented wording. ``unverbatim`` reports
    an attributed span canon never states in those exact characters. That is the wider
    class, and it is the one the probe's own question asks about: it asked for the notice
    word for word.

    The caller passes the canon it wants scored, with one ``\\x00`` between sources.
    ``_authored_canon_text`` builds the canon every soak run scores, and a unit test
    passes a literal instead. The word channel reuses ``normalize_for_lenient_match``,
    so one normalizer serves every recall channel this harness reports.
    """
    normalized_canon = _CANON_JOIN.join(
        normalize_for_lenient_match(part) for part in canon.split("\x00")
    )
    spans: list[dict] = []
    for match in _AUTHORED_QUOTE_SPAN.finditer(reply):
        text = next(group for group in match.groups() if group is not None)
        preceding = reply[max(0, match.start() - AUTHORED_QUOTE_CUE_WINDOW):match.start()]
        preceding_words = set(normalize_for_lenient_match(preceding).split())
        spans.append(
            {
                "text": text,
                "attributed": any(cue in preceding_words for cue in AUTHORED_QUOTE_CUES),
                "word_supported": normalize_for_lenient_match(text) in normalized_canon,
                "verbatim_supported": text in canon,
            }
        )
    attributed = [span for span in spans if span["attributed"]]
    return {
        "spans": spans,
        "attributed": len(attributed),
        "fabricated": any(not span["word_supported"] for span in attributed),
        "unverbatim": any(not span["verbatim_supported"] for span in attributed),
    }


def normalize_for_lenient_match(text: str) -> str:
    """Collapse case and punctuation runs for the lenient recall channel.
    """
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def canon_words(text: str) -> list[str]:
    """One canon statement's content words, with possessives reduced to their noun.
    """
    lowered = re.sub(r"['’]s\b", "", text.lower())
    return [part for part in re.split(r"[^a-z0-9]+", lowered) if part]


def canon_entries(campaign_root: Path) -> list[str]:
    """Every canon statement as its own entry, from both authoritative files.

    A visible fact is one line of ``campaign/scene.md``; a clock fill, a location
    identifier, and a summary are string leaves of ``campaign/state.json``. Entry
    granularity is what lets the strongest matching channel ask whether one entry
    holds a whole planted fact, rather than whether the document mentions its words
    somewhere, which two unrelated entries could satisfy between them.
    """
    entries: list[str] = []
    scene = campaign_root / "campaign" / "scene.md"
    if scene.is_file():
        entries.extend(
            line.strip(" -\t") for line in scene.read_text(encoding="utf-8").splitlines()
        )
    state = campaign_root / "campaign" / "state.json"
    if state.is_file():
        try:
            payload = json.loads(state.read_text(encoding="utf-8"))
        except ValueError:
            payload = None

        def _walk(node) -> None:
            if isinstance(node, str):
                entries.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for value in node:
                    _walk(value)


        if isinstance(payload, dict) and isinstance(payload.get("scene"), dict):
            scene_payload = dict(payload["scene"])
            raw_statements = scene_payload.get("statements")
            if isinstance(raw_statements, list):
                scene_payload["statements"] = [
                    entry.get("text")
                    for entry in raw_statements
                    if isinstance(entry, dict) and isinstance(entry.get("text"), str)
                ]
            payload = {**payload, "scene": scene_payload}
        _walk(payload)
    return [entry for entry in entries if entry.strip()]


def canon_holds(entries: list[str], tokens: tuple[str, ...]) -> dict:
    """Three readings of whether canon holds one planted fact.

    ``strict`` is the original channel: every token appears verbatim somewhere in the
    two files. ``lenient`` collapses punctuation and case across the whole document.
    ``entry`` asks the question the other two cannot: does one single canon statement
    carry every word of every token, once possessives are reduced?
    """
    document = "\n".join(entries).lower()
    lenient_document = normalize_for_lenient_match(document)
    needed: set[str] = set()
    for token in tokens:
        needed.update(canon_words(token))
    return {
        "strict": all(token.lower() in document for token in tokens),
        "lenient": all(
            normalize_for_lenient_match(token) in lenient_document for token in tokens
        ),
        "entry": bool(needed)
        and any(needed <= set(canon_words(entry)) for entry in entries),
    }


def generate_transcript(
    messages: int,
    block: int,
    seed: int,
    plant_turn: int = 0,
    recall_turn: int = 0,
    authored_turn: int = 0,
    clock_turn: int = 0,
    zone_turns: dict | None = None,
    order_turns: dict | None = None,
    exchange_turns: dict | None = None,
    hazard_turns: dict | None = None,
) -> str:
    """Emit a deterministic two-player transcript with one mention per block.

    ``messages`` counts player lines, chatter and mentions together. ``block`` is the
    lines per mention, and it must stay at or under ``NarratorConfig.backfill_limit``
    plus one, or generated chatter silently falls off the replay buffer and the fed
    count drops under the generated count.

    Two turn numbers must not collide. The caller assigns them, and ``main`` keeps the
    clock turn one mention away from the episodic recall turn, because one mention
    carries one question and a merged mention would score two probes on one reply.
    """
    zone_text = {
        "zone_c_plant": ZONE_C_PLANT,
        "zone_a_plant": ZONE_A_PLANT,
        "zone_b_recall": ZONE_B_RECALL,
        "zone_c_recall": ZONE_C_RECALL,
        "zone_a_recall": ZONE_A_RECALL,
    }
    zone_by_turn = {
        turn: zone_text[name]
        for name, turn in (zone_turns or {}).items()
        if turn and name in zone_text
    }
    order_text = {
        "order_move_first": ORDER_MOVE_FIRST,
        "order_move_second": ORDER_MOVE_SECOND,
        "order_recall": ORDER_RECALL,
    }
    order_by_turn = {
        turn: order_text[name]
        for name, turn in (order_turns or {}).items()
        if turn and name in order_text
    }
    exchange_text = {
        "exchange_give_first": EXCHANGE_GIVE_FIRST,
        "exchange_return": EXCHANGE_RETURN,
        "exchange_give_second": EXCHANGE_GIVE_SECOND,
    }
    exchange_by_turn = {
        turn: exchange_text[name]
        for name, turn in (exchange_turns or {}).items()
        if turn and name in exchange_text
    }
    hazard_by_turn = {
        turn: HAZARD_PLANTS[name][0]
        for name, turn in (hazard_turns or {}).items()
        if turn and name in HAZARD_PLANTS
    }
    rng = random.Random(seed)
    lines: list[str] = [
        "# Generated two-player soak transcript.",
        f"# messages={messages} block={block} seed={seed}",
    ]
    emitted = 0
    beat = 0
    while emitted < messages:
        chatter = min(block - 1, messages - emitted - 1)
        for index in range(chatter):
            author = PLAYERS[(emitted + index) % 2]
            other = PLAYERS[(emitted + index + 1) % 2]
            template = rng.choice(_CHATTER)
            lines.append(
                f"{author}: "
                + template.format(
                    other=other,
                    place=rng.choice(LOCATIONS),
                    figure=rng.choice(FIGURES),
                    object=rng.choice(OBJECTS),
                )
            )
        emitted += chatter
        turn_number = beat + 1
        if plant_turn and turn_number == plant_turn:
            mention = CONTINUITY_PLANT.format(actor=PLAYERS[beat % 2])
        elif recall_turn and turn_number == recall_turn:
            mention = CONTINUITY_RECALL.format(actor=PLAYERS[beat % 2])
        elif authored_turn and turn_number == authored_turn:
            mention = AUTHORED_RECALL.format(actor=PLAYERS[beat % 2])
        elif clock_turn and turn_number == clock_turn:
            mention = CLOCK_RECALL.format(actor=PLAYERS[beat % 2])
        elif turn_number in zone_by_turn:
            mention = zone_by_turn[turn_number].format(actor=PLAYERS[beat % 2])
        elif turn_number in order_by_turn:
            mention = order_by_turn[turn_number].format(actor=PLAYERS[beat % 2])
        elif turn_number in exchange_by_turn:
            # Named players, no ``{actor}``: see EXCHANGE_OBJECT for why parity cannot
            # decide the direction of a pass.
            mention = exchange_by_turn[turn_number]
        elif turn_number in hazard_by_turn:
            mention = hazard_by_turn[turn_number]
        else:
            mention = _MENTIONS[beat % len(_MENTIONS)].format(
                actor=PLAYERS[beat % 2],
                place=LOCATIONS[beat % len(LOCATIONS)],
                figure=FIGURES[beat % len(FIGURES)],
                object=OBJECTS[beat % len(OBJECTS)],
            )
        lines.append(mention)
        emitted += 1
        beat += 1
    return "\n".join(lines) + "\n"


def prefill_payload(depth: int) -> dict:
    """The ``scene_commit`` arguments that push the scene record past the digest bound.

    The payload is data, not behavior: ``bootstrap_sandbox`` passes it to the server
    interpreter, and this function stays importable by the unit suite.
    """
    return {
        "public_summary": (
            "The party works the eel market while the tide runs out, and the record "
            "already carries the session's earlier entries."
        ),
        "visible_changes": [
            PREFILL_FACT.format(index=index) for index in range(1, depth + 1)
        ],
    }


def zone_b_payload() -> dict:
    """The ``scene_commit`` arguments that plant the oldest zone at bootstrap.

    ``bootstrap_sandbox`` writes this before the prefill, so the fact sits at the head
    of the visible-facts list. Head retention therefore keeps it at every depth the
    fixture reaches, which is the property zone B exists to score.
    """
    return {
        "public_summary": "The party stowed its gear before starting the venture.",
        "visible_changes": [ZONE_B_FACT],
    }


def order_payload() -> dict:
    """The ``scene_commit`` arguments that plant the ordering probe's first place.

    Planting the first leg at bootstrap fixes the path's origin without spending a
    model commit on it, so a run scores the two moves that follow rather than scoring
    write discipline three times over. The two moves plant by mention, which is what
    puts them at the ages this probe exists to measure.
    """
    return {
        "public_summary": (
            "The party took charge of a lacquered oar-case before the venture began."
        ),
        "visible_changes": [
            f"The party carries the {ORDER_OBJECT} inside the {ORDER_PLACES[0]}."
        ],
    }


def clock_payload() -> dict:
    """The ``scene_commit`` arguments that plant the probe clock at bootstrap."""
    return {
        "public_summary": "The harbour reeve starts counting hulls against the party.",
        "new_clocks": [
            {
                "id": CLOCK_ID,
                "name": CLOCK_NAME,
                "segments": CLOCK_SEGMENTS,
                "filled": CLOCK_FILLED,
                "note": CLOCK_NOTE,
            }
        ],
    }


def score_clock_recall(
    reply: str,
    canon_text: str,
    fill_state: str,
    clocks_section_state: str,
    tool_names: list[str],
) -> dict:
    """Score the clock probe strict, with the lenient channel and the confounds beside it.

    Strict containment gates nothing in this slice: a miss is a finding, not a failure.
    The record therefore has to carry enough to classify any miss. ``present_in_canon``
    separates a fact that left the record from a fact the narrator could not produce.
    ``clocks_section_state`` states whether the digest carried the Clocks section on
    that turn, which is the variable under test. ``tool_names`` states whether the
    narrator called ``campaign_status``, the one tool that reports clock state, so a
    pass through retrieval never reads as a pass through delivery.
    """
    lowered = reply.lower()
    found = [token for token in CLOCK_TOKENS if token in lowered]
    lenient_reply = normalize_for_lenient_match(lowered)
    lenient_found = [
        token
        for token in CLOCK_TOKENS
        if normalize_for_lenient_match(token) in lenient_reply
    ]
    lowered_canon = canon_text.lower()
    return {
        "clock_id": CLOCK_ID,
        "fact_tokens": list(CLOCK_TOKENS),
        "present_in_canon": all(token in lowered_canon for token in CLOCK_TOKENS),
        "recalled": len(found) == len(CLOCK_TOKENS),
        "tokens_in_reply": found,
        "lenient": {
            "recalled": len(lenient_found) == len(CLOCK_TOKENS),
            "tokens_in_reply": lenient_found,
        },
        "fill_state": fill_state,
        "fill_in_reply": bool(fill_state) and fill_state in lowered,
        "clocks_section_state": clocks_section_state,
        "tool_names": list(tool_names),
        "read_tool_called": any(name in CLOCK_READ_TOOLS for name in tool_names),
        "reply_excerpt": reply[:600],
    }


def fixture_payloads(
    prefill_facts: int, clock: bool, zone: bool, order: bool = False
) -> list[dict]:
    """The bootstrap commits, in the order the record must receive them.

    Order carries meaning. Zone B and the ordering probe's first place both have to
    reach the visible-facts list before the prefill, because head retention keeps the
    oldest entries and both fixtures exist to score facts the digest alone carries.
    Reversing them would bury both under 40 prefilled entries and score the opposite of
    what they name.

    With every fixture off this returns an empty list, so the default bootstrap runs
    exactly the two subprocesses every recorded soak ran.
    """
    payloads: list[dict] = []
    if zone:
        payloads.append(zone_b_payload())
    if order:
        payloads.append(order_payload())
    if prefill_facts > 0:
        payloads.append(prefill_payload(prefill_facts))
    if clock:
        payloads.append(clock_payload())
    return payloads


def score_order_recall(
    reply: str,
    canon_text: str,
    digest_texts: list[str],
    recall_turn: int,
    tool_names: list[str],
    canon_entries_list: list[str] | None = None,
) -> dict:
    """Score the ordering probe: the three places, and whether the reply orders them.

    ``ordered`` is the score this probe adds. It holds when every place token appears
    in the reply and their first occurrences run in the recorded order, so a reply
    naming all three places backwards scores ``recalled`` and fails ``ordered``. The
    two channels separate a lost place from a lost sequence.

    Three fields classify a miss. ``committed`` states, per place, whether the record
    holds it at scoring time, which separates a move the model never wrote from a move
    it could not produce. ``in_digest_at_recall`` states whether the bounded scene block
    carried it on the scored turn. ``digest_turns`` counts every turn whose digest
    carried it, which is the payload a superseded-digest removal would take away.
    """
    lowered = reply.lower()
    lowered_canon = canon_text.lower()
    canon_entries_list = (
        canon_entries_list if canon_entries_list is not None else canon_text.splitlines()
    )
    positions = [lowered.find(place) for place in ORDER_PLACES]
    lenient_reply = normalize_for_lenient_match(lowered)
    lenient_positions = [
        lenient_reply.find(normalize_for_lenient_match(place)) for place in ORDER_PLACES
    ]
    carried = [
        [text for text in digest_texts if place in text.lower()] for place in ORDER_PLACES
    ]
    scored_digest = (
        digest_texts[recall_turn - 1].lower()
        if 1 <= recall_turn <= len(digest_texts)
        else ""
    )
    at_recall = [place in scored_digest for place in ORDER_PLACES]
    return {
        "object": ORDER_OBJECT,
        "places": list(ORDER_PLACES),
        "committed": [place in lowered_canon for place in ORDER_PLACES],
        "all_committed": all(place in lowered_canon for place in ORDER_PLACES),
        # The three channels of ``canon_holds``, per place. ``committed`` above stays
        # strict so the published series keeps its meaning; ``held`` is the reading that
        # separates a fact the narrator never recorded from one it recorded in its own
        # words. Two of three checkable losses moved columns when this arrived.
        "held": [canon_holds(canon_entries_list, (place,)) for place in ORDER_PLACES],
        "recalled": all(index >= 0 for index in positions),
        "ordered": _strictly_increasing(positions),
        "positions_in_reply": positions,
        "lenient": {
            "recalled": all(index >= 0 for index in lenient_positions),
            "ordered": _strictly_increasing(lenient_positions),
        },
        "in_digest_at_recall": at_recall,
        "digest_turns": [len(texts) for texts in carried],
        "tool_names": list(tool_names),
        "read_tool_called": any(name in CLOCK_READ_TOOLS for name in tool_names),
        "reply_excerpt": reply[:600],
    }


def score_hazard_probe(routing: dict | None, post_log, hazard_turns: dict) -> dict:
    """What each planted hazard drew, read from the routing rows and the typed posts.
    """
    rows_by_index = (routing or {}).get("per_turn", [])
    aligned = len(rows_by_index) == len(post_log)
    plants = []
    for name, turn in hazard_turns.items():
        if not turn or name not in HAZARD_PLANTS:
            continue
        text, target_kind, ask_is = HAZARD_PLANTS[name]
        row = rows_by_index[turn - 1] if aligned and 0 < turn <= len(rows_by_index) else {}
        post = post_log[turn - 1] if 0 < turn <= len(post_log) else None
        plants.append(
            {
                "name": name,
                "turn": turn,
                "text": text,
                "target_kind": target_kind,
                "ask_is": ask_is,
                "route": row.get("route", "unaligned"),
                "risk_category": row.get("risk_category", ""),
                "names_unrecorded_person": row.get("names_unrecorded_person"),
                "interlocutor_kind": row.get("interlocutor_kind", ""),
                "ask_posted": post is not None and post.notice_key == "risk_confirmation",
            }
        )
    asked = [plant for plant in plants if plant["ask_posted"]]
    return {
        "aligned": aligned,
        "plants": plants,
        "asks": len(asked),
        "asks_naming_unrecorded_person": sum(1 for p in asked if p["names_unrecorded_person"]),
        "owed_asks": sum(1 for p in asked if p["ask_is"] == "owed-first-strike"),
        "corpse_asks": sum(1 for p in asked if p["ask_is"] == "spurious-if-asked"),
        "risk_routes": sum(1 for p in plants if p["route"] == "risk"),
    }


def score_exchange(entries: list[str], exchange_turns: dict) -> dict:
    """What the record says about the passed object, in record order, and nothing more.

    ``entries_naming_object`` therefore carries every entry in record order, uncapped,
    and a reader classifies from it. ``final_entry_names`` states which player names
    appear in the newest such entry, and ``sole_name_in_final_entry`` states the name
    when exactly one appears. That single case is the one containment settles, and it is
    the case a duplicate-dropping mechanism breaks: with leg three dropped, the newest
    surviving entry is leg two, which names the giver as holder.
    """
    lowered_object = EXCHANGE_OBJECT.lower()
    naming = [entry for entry in entries if lowered_object in entry.lower()]
    final = naming[-1] if naming else ""
    lowered_final = final.lower()
    present = [name for name in PLAYERS if name.lower() in lowered_final]
    return {
        "object": EXCHANGE_OBJECT,
        "turns": dict(exchange_turns),
        "legs": [
            {"turn": exchange_turns.get("exchange_give_first", 0), "from": EXCHANGE_GIVER, "to": EXCHANGE_TAKER},
            {"turn": exchange_turns.get("exchange_return", 0), "from": EXCHANGE_TAKER, "to": EXCHANGE_GIVER},
            {"turn": exchange_turns.get("exchange_give_second", 0), "from": EXCHANGE_GIVER, "to": EXCHANGE_TAKER},
        ],
        "expected_final_holder": EXCHANGE_FINAL_HOLDER,
        "entries_naming_object": naming,
        "entry_count": len(naming),
        "final_entry": final,
        "final_entry_names": present,
        "sole_name_in_final_entry": present[0] if len(present) == 1 else None,
    }


def _strictly_increasing(positions: list[int]) -> bool:
    """Every place present, and each one first appearing after the previous one."""
    if any(index < 0 for index in positions):
        return False
    return all(
        positions[index] < positions[index + 1] for index in range(len(positions) - 1)
    )


def summarize_request_usage(request_log: list[dict]) -> dict:
    """What the endpoint charged for each assembled request, summed by origin.

    ``requests_with_cache_key`` exists because Strands writes ``cacheReadInputTokens``
    only for a truthy count. A request that reused nothing and a server that reports no
    detail both leave the key absent, so the count separates a measured zero from an
    unmeasured one: a run whose requests all carry usage and none carry the key ran
    against a server without the flag.
    """
    rows = [row for row in request_log if isinstance(row.get("usage"), dict)]

    def _rate(cached: int, total: int) -> float | None:
        return round(cached / total, 4) if total else None

    by_origin: dict[str, dict] = {}
    for row in rows:
        usage = row["usage"]
        entry = by_origin.setdefault(
            str(row.get("origin", "unknown")),
            {"requests": 0, "input_tokens": 0, "cached_tokens": 0},
        )
        entry["requests"] += 1
        entry["input_tokens"] += int(usage.get("inputTokens", 0))
        entry["cached_tokens"] += int(usage.get("cacheReadInputTokens", 0))
    for entry in by_origin.values():
        entry["hit_rate"] = _rate(entry["cached_tokens"], entry["input_tokens"])

    input_tokens = sum(entry["input_tokens"] for entry in by_origin.values())
    cached_tokens = sum(entry["cached_tokens"] for entry in by_origin.values())
    return {
        "requests_with_usage": len(rows),
        "requests_with_cache_key": sum(
            1 for row in rows if "cacheReadInputTokens" in row["usage"]
        ),
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "hit_rate": _rate(cached_tokens, input_tokens),
        "by_origin": by_origin,
    }


def summarize_payload(request_log: list[dict], usage_log: list[dict]) -> dict:
    """One request-payload composition summary from the engine's own request rows.

    ``canon_superseded_chars`` is the reclaimable payload: every canon digest in one
    request except the newest. The summary reports its maximum and mean against the
    request total, so a reader prices the redundancy against what the endpoint reads
    rather than against the history alone.
    """
    turn_rows = [row for row in request_log if row.get("origin") == "turn"]
    by_origin: dict[str, int] = {}
    for row in request_log:
        origin = str(row.get("origin", "unknown"))
        by_origin[origin] = by_origin.get(origin, 0) + 1

    def _mean(values: list[int | float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    fields = (
        "total_chars",
        "system_chars",
        "tools_chars",
        "messages_chars",
        "canon_current_chars",
        "canon_superseded_chars",
        "canon_blocks",
        "message_count",
    )
    chars = {}
    for field_name in fields:
        values = [int(row.get(field_name, 0)) for row in turn_rows]
        chars[f"{field_name}_max"] = max(values) if values else 0
        chars[f"{field_name}_mean"] = _mean(values)
    shares = [
        row["canon_superseded_chars"] / row["total_chars"]
        for row in turn_rows
        if row.get("total_chars")
    ]
    usage_totals: dict[str, int] = {}
    for entry in usage_log:
        for key, value in entry.items():
            usage_totals[key] = usage_totals.get(key, 0) + int(value)

    # The step-level replay reading: how much reasoning each request of a turn
    # generated, first request against every later one. ``reasoningOutputChars`` is
    # the model layer's own count of streamed reasoning deltas (the served vLLM
    # reports no completion token detail), so a later step that continues a replayed
    # thought should mean fewer of these than one re-deriving from scratch.
    first_steps: list[int] = []
    later_steps: list[int] = []
    steps_by_turn: dict[int, int] = {}
    for row in turn_rows:
        usage = row.get("usage") or {}
        chars_generated = usage.get("reasoningOutputChars")
        if not isinstance(chars_generated, int):
            continue
        turn = int(row.get("turn", 0))
        steps_by_turn[turn] = steps_by_turn.get(turn, 0) + 1
        (first_steps if steps_by_turn[turn] == 1 else later_steps).append(chars_generated)
    reasoning_steps = {
        "first_step_requests": len(first_steps),
        "first_step_mean_chars": _mean(first_steps),
        "later_step_requests": len(later_steps),
        "later_step_mean_chars": _mean(later_steps),
        "multi_step_turns": sum(1 for count in steps_by_turn.values() if count > 1),
    }
    return {
        "requests": len(request_log),
        "requests_by_origin": by_origin,
        "turn_requests": len(turn_rows),
        "chars": chars,
        "superseded_share_mean": (
            round(sum(shares) / len(shares), 4) if shares else 0.0
        ),
        "per_request_usage": summarize_request_usage(request_log),
        "reasoning_steps": reasoning_steps,
        "usage_totals": usage_totals,
        "usage_per_turn": list(usage_log),
        "per_request": list(request_log),
    }


#: The vLLM counters this harness reads, keyed by the short name the report uses. Every
#: one is a monotonic counter, so a run's consumption is one subtraction. The two prefix
#: counters carry tokens rather than requests, which is why the hit rate they produce is
#: a token fraction. ``requests_succeeded`` carries a ``finished_reason`` label, whose
#: five values split it across five lines, so the parser sums every label set of a
#: metric; it exists to detect a second client inside the window.
CACHE_COUNTERS: dict[str, str] = {
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "requests_succeeded": "vllm:request_success_total",
    "preemptions": "vllm:num_preemptions_total",
}

#: The engine's own cache configuration, published as one labelled gauge. Recording it
#: with the measurement answers the first question any reader of a low hit rate asks:
#: was prefix caching switched on, and how large was the cache it ran against?
CACHE_CONFIG_METRIC = "vllm:cache_config_info"
CACHE_CONFIG_KEYS = (
    "enable_prefix_caching",
    "block_size",
    "kv_cache_size_tokens",
    "num_gpu_blocks",
    "gpu_memory_utilization",
    "prefix_caching_hash_algo",
)


def server_root(base_url: str) -> str:
    """The server root beside an OpenAI-compatible base URL.

    vLLM serves the OpenAI surface under ``/v1`` and serves ``/metrics`` and
    ``/version`` at the root, so the derivation drops one trailing ``/v1`` segment and
    changes nothing else. A base URL that already names the root survives unchanged.
    """
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def parse_metrics(text: str) -> dict:
    """One Prometheus exposition, reduced to the counters and configuration this reads.

    Each counter sums across its label sets, because vLLM labels by engine and by finish
    reason and this harness prices the whole endpoint. ``missing`` names every tracked
    counter the exposition never published, which separates a counter reading zero from
    a counter this vLLM release does not carry.
    """
    counters = {short: 0.0 for short in CACHE_COUNTERS}
    seen: set[str] = set()
    config: dict[str, str] = {}
    by_metric = {metric: short for short, metric in CACHE_COUNTERS.items()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # Name, optional label set, value, and an optional trailing timestamp the
        # exposition format permits. Splitting on the last space instead would read a
        # timestamp as the counter, which turns a token count into a Unix epoch.
        parsed = re.match(r"^([^{ ]+)(\{.*\})?\s+(\S+)", line)
        if parsed is None:
            continue
        name, labels, raw = parsed.group(1), parsed.group(2) or "", parsed.group(3)
        if name == CACHE_CONFIG_METRIC and not config:
            found = dict(re.findall(r'(\w+)="([^"]*)"', labels))
            config = {key: found[key] for key in CACHE_CONFIG_KEYS if key in found}
            continue
        short = by_metric.get(name)
        if short is None:
            continue
        try:
            counters[short] += float(raw)
        except ValueError:
            continue
        seen.add(short)
    return {
        "counters": counters,
        "missing": sorted(set(CACHE_COUNTERS) - seen),
        "config": config,
    }


def sample_cache_metrics(url: str, timeout: float = 5.0) -> dict | None:
    """One reading of the metrics endpoint, or None when the reading fails for any reason.

    Returning None rather than raising keeps the contract this instrument needs: a
    metrics endpoint that refuses, hangs, or answers with garbage costs the run its
    cache measurement and costs it nothing else. The caller records the gap.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return parse_metrics(response.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - a measurement never fails a run
        return None


def read_engine_version(base_url: str, timeout: float = 5.0) -> str:
    """The serving engine's own version string, or an empty string when unreadable.

    The ``latest`` image tag moves, so a measurement omitting the resolved engine
    version records an unreproducible result. The endpoint reports its version, which
    removes the need to trust a launch script or a container listing.

    The catch is total for the reason recorded in ``sample_cache_metrics``: a garbled
    response raises exception classes that no useful tuple enumerates, and this reader
    runs before the first turn, where a raise would abort the whole run.
    """
    url = f"{server_root(base_url)}/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
        return str(body.get("version", ""))
    except Exception:  # noqa: BLE001 - a measurement never fails a run
        return ""


def summarize_cache(
    samples: list[dict],
    requests_recorded: int,
    usage_totals: dict,
    endpoint: str,
    config: dict,
    engine_version: str,
) -> dict:
    """``samples`` carries one entry per sampling point: the baseline at turn 0, then one
    per completed turn. An entry whose ``counters`` is None records a reading the
    endpoint refused. The window spans the first and last successful readings, and a
    per-turn row appears only where both of its bounding readings succeeded, so a gap
    narrows the record instead of corrupting it.

    ``hit_rate`` divides cached prompt tokens by queried prompt tokens. It reports None
    when the window queried zero tokens, because zero of zero is not zero reuse.

    ``exclusive_window`` compares the requests vLLM completed against the requests this
    process assembled. Equality states that no second client of the shared endpoint
    contributed tokens to the window, which is the condition the hit rate needs to
    describe this session alone.

    ``usage_cache_key_present`` records whether Strands surfaced vLLM's
    ``prompt_tokens_details.cached_tokens`` as ``cacheReadInputTokens`` in any turn's
    usage. That key against a positive hit rate is the exact pair that separates an
    unreported detail from an endpoint serving zero cached tokens.
    """

    def _rate(hits: int, queries: int) -> float | None:
        return round(hits / queries, 4) if queries else None

    taken = [entry for entry in samples if entry.get("counters") is not None]
    sampled = len(taken) >= 2
    window = {key: 0 for key in CACHE_COUNTERS}
    if sampled:
        first, last = taken[0]["counters"], taken[-1]["counters"]
        window = {
            key: int(round(last.get(key, 0.0) - first.get(key, 0.0)))
            for key in CACHE_COUNTERS
        }

    per_turn: list[dict] = []
    for previous, current in zip(samples, samples[1:]):
        if previous.get("counters") is None or current.get("counters") is None:
            continue
        before, after = previous["counters"], current["counters"]
        delta = {
            key: int(round(after.get(key, 0.0) - before.get(key, 0.0)))
            for key in CACHE_COUNTERS
        }
        per_turn.append(
            {
                "turn": current["turn"],
                "queries": delta["prefix_cache_queries"],
                "hits": delta["prefix_cache_hits"],
                "hit_rate": _rate(delta["prefix_cache_hits"], delta["prefix_cache_queries"]),
                "prompt_tokens": delta["prompt_tokens"],
                "generation_tokens": delta["generation_tokens"],
                "requests": delta["requests_succeeded"],
            }
        )

    return {
        "endpoint": endpoint,
        "engine_version": engine_version,
        "config": dict(config),
        "sampled": sampled,
        "samples_attempted": len(samples),
        "samples_taken": len(taken),
        "window_turns": [taken[0]["turn"], taken[-1]["turn"]] if sampled else [],
        "window": window,
        "hit_rate": _rate(window["prefix_cache_hits"], window["prefix_cache_queries"]),
        "cached_prompt_share": _rate(window["prompt_tokens_cached"], window["prompt_tokens"]),
        "requests_recorded": requests_recorded,
        "requests_observed": window["requests_succeeded"],
        # None rather than False when nothing was sampled: an unmeasured window states
        # nothing about who held the endpoint, and False would read as a foreign client.
        "exclusive_window": (
            window["requests_succeeded"] == requests_recorded if sampled else None
        ),
        "usage_cache_key_present": "cacheReadInputTokens" in (usage_totals or {}),
        "counter_names": dict(CACHE_COUNTERS),
        "counters_missing": taken[0].get("missing", []) if taken else [],
        "per_turn": per_turn,
    }


def visible_facts(campaign_root: Path) -> list[str]:
    """The scene record's visible-facts list, one entry per line."""
    scene = campaign_root / "campaign" / "scene.md"
    if not scene.is_file():
        return []
    text = scene.read_text(encoding="utf-8")
    if "## Visible facts" not in text:
        return []
    section = text.split("## Visible facts", 1)[1].split("\n## ", 1)[0]
    return [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]


#: Words that carry no fact, dropped before a visible fact becomes a content-word
#: bag. Two entries whose bags are equal or nest state the same thing under the
#: containment rule ``duplicate_fact_pairs`` and ``duplicate_clusters`` share.
_FILLER_WORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of", "to",
        "now", "has", "have", "been", "and", "party", "it", "its", "that", "this",
    }
)


_NEGATION_MARKERS = frozenset({"no", "not", "longer", "never", "without"})


def _content_bag(fact: str) -> set[str]:
    """One visible fact reduced to its content words, filler removed."""
    return set(canon_words(fact)) - _FILLER_WORDS


def _is_reversal(left: set[str], right: set[str]) -> bool:
    """Whether two content-word bags state one fact and its state reversal.

    The bags differ by negation markers alone, so one entry asserts what the other
    denies. "The gate is locked" and "The gate is not locked" differ by ``not``.
    "Ossa is at the cave" and "Ossa is no longer at the cave" differ by ``no`` and
    ``longer``. An empty difference is the same entry, not a reversal.
    """
    difference = left ^ right
    return bool(difference) and difference <= _NEGATION_MARKERS


def duplicate_fact_pairs(facts: list[str]) -> list[list[str]]:
    """Every pair of visible facts that states the same thing twice.

    A pair counts only when one entry's content words contain the other's, after
    articles and copulas drop out. A share threshold was tried first and rejected: at 60
    percent it paired \"An unknown presence is approaching the cave mouth from the road
    shrine\" with \"An unidentified presence is approaching the market edge from the road
    shrine\", which name two different places, and it paired a brass key with a
    tide-medal because both sat behind the same lintel. Containment keeps the count
    conservative, so a reported duplicate is one the record can be shown to hold twice.
    """
    bags = [_content_bag(fact) for fact in facts]
    pairs: list[list[str]] = []
    for first in range(len(facts)):
        for second in range(first + 1, len(facts)):
            left, right = bags[first], bags[second]
            if not left or not right:
                continue
            if left <= right or right <= left:
                pairs.append([facts[first], facts[second]])
    return pairs


def duplicate_clusters(facts: list[str]) -> list[dict]:
    """Group the visible-facts list into order-preserving containment clusters.

    Each cluster is a connected component of two or more facts under the shared
    containment relation, in first-appearance order. It carries the member entries,
    their positions in the list, the entries that sit between the first and last
    member and are not themselves members, a verbatim flag, and a reversal flag.
    The record carries context; it decides no duplicate. A calibrated classifier
    reads ``intervening`` to separate a re-entry from a restatement.
    """
    bags = [_content_bag(fact) for fact in facts]
    parent = list(range(len(facts)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first in range(len(facts)):
        for second in range(first + 1, len(facts)):
            left, right = bags[first], bags[second]
            if not left or not right:
                continue
            if left <= right or right <= left:
                parent[find(second)] = find(first)

    components: dict[int, list[int]] = {}
    for index in range(len(facts)):
        components.setdefault(find(index), []).append(index)

    clusters: list[dict] = []
    for members in components.values():
        if len(members) < 2:
            continue
        members.sort()
        span = range(members[0], members[-1] + 1)
        member_set = set(members)
        normalised = {normalize_for_lenient_match(facts[i]) for i in members}
        # A member pair contributes to one signal or the other, and a cluster can
        # carry both: a verbatim repeat riding beside an at/no-longer-at pair is a
        # reversal chain and a genuine duplicate at once. Scoring the whole cluster
        # by one flag would hide the repeat behind the reversal.
        reversal = False
        plain_duplicate = False
        for i in members:
            for j in members:
                if i >= j:
                    continue
                left, right = bags[i], bags[j]
                if not left or not right or not (left <= right or right <= left):
                    continue
                if _is_reversal(left, right):
                    reversal = True
                else:
                    plain_duplicate = True
        clusters.append(
            {
                "entries": [facts[i] for i in members],
                "positions": members,
                "intervening": [facts[i] for i in span if i not in member_set],
                "verbatim": len(normalised) == 1,
                "reversal": reversal,
                "plain_duplicate": plain_duplicate,
            }
        )
    clusters.sort(key=lambda cluster: cluster["positions"][0])
    return clusters


# Synthetic calibration examples for the diagnostic duplicate scorer.
DUPLICATE_CALIBRATION: tuple[dict, ...] = (
    {"kind": "verbatim", "label": "duplicate",
     "entries": ["Location: Workshop", "Location: Workshop"], "intervening": []},
    {"kind": "containment", "label": "duplicate",
     "entries": ["The green compass is under the marble pedestal.",
                 "green compass under the marble pedestal"], "intervening": []},
    {"kind": "long_short", "label": "duplicate",
     "entries": ["Before breakfast Neri stored spare sailcloth beneath the workshop bench.",
                 "spare sailcloth beneath the workshop bench"], "intervening": []},
    {"kind": "long_short", "label": "duplicate",
     "entries": ["A copper hinge is stored inside an oak chest.",
                 "copper hinge stored inside oak chest"], "intervening": []},
    {"kind": "reversal", "label": "distinct",
     "entries": ["Neri is at the workshop.", "Neri is no longer at the workshop."],
     "intervening": []},
    {"kind": "negation", "label": "distinct",
     "entries": ["The chest is locked.", "The chest is not locked."], "intervening": []},
    {"kind": "transition", "label": "distinct",
     "entries": ["Neri is moving toward the workshop.", "Neri is at the workshop."],
     "intervening": []},
    {"kind": "two_objects", "label": "distinct",
     "entries": ["A green compass is under the marble pedestal.",
                 "A copper hinge is under the marble pedestal."], "intervening": []},
    {"kind": "elaboration", "label": "elaboration",
     "entries": ["The compass is under the pedestal.",
                 "The compass is under the pedestal and bears a spiral engraving."],
     "intervening": []},
    {"kind": "reentry", "label": "distinct",
     "entries": ["Neri is at the workshop.", "Neri is at the workshop."],
     "intervening": ["Neri leaves the workshop for the orchard.",
                     "Neri returns to the workshop."]},
)


def summarize_duplicate_filings(facts: list[str]) -> dict:
    """Score the visible-facts list for duplicate filings and reversal chains.
    """
    clusters = duplicate_clusters(facts)
    reversal_clusters = [cluster for cluster in clusters if cluster["reversal"]]
    candidate_clusters = [
        cluster for cluster in clusters if cluster["plain_duplicate"]
    ]
    negation_entries = [
        fact for fact in facts if _content_bag(fact) & _NEGATION_MARKERS
    ]
    return {
        "clusters": clusters,
        "cluster_count": len(clusters),
        "reversal_chain_clusters": len(reversal_clusters),
        "negation_entries": len(negation_entries),
        # Candidate duplicates a classifier still judges, with intervening context.
        # This is not a duplicate count and must not gate a run: a cluster carrying
        # an intervening entry may be a correct re-filing after a departure.
        "candidate_clusters": len(candidate_clusters),
        "candidates_with_intervening": sum(
            1 for cluster in candidate_clusters if cluster["intervening"]
        ),
    }


def summarize_commit_discipline(
    campaign_root: Path,
    report: SoakReport,
    posted: list[str],
    order_turns: dict,
    zone_turns: dict,
    exchange_turns: dict | None = None,
) -> dict:
    """Whether each fact a player stated in session reached the record, and why not.

    Each row carries the three matching channels, whether that turn wrote any record
    entry, the sweep's disposition, and whether the turn's narration mentions the fact's
    tokens.
    """
    entries = canon_entries(campaign_root)
    tools = (report.tools or {}).get("per_turn", [])
    sweeps = (report.sweep or {}).get("per_turn", [])
    plants: list[tuple[str, int, tuple[str, ...]]] = []
    if order_turns:
        plants.append(("order-move-first", order_turns["order_move_first"], (ORDER_PLACES[1],)))
        plants.append(("order-move-second", order_turns["order_move_second"], (ORDER_PLACES[2],)))
    if zone_turns:
        plants.append(("zone-c", zone_turns["zone_c_plant"], ZONE_C_TOKENS))
        plants.append(("zone-a", zone_turns["zone_a_plant"], ZONE_A_TOKENS))
    if exchange_turns:
        # Legs one and three carry identical tokens, because they are the same pass
        # repeated, and that repetition is the property under test. Their rows therefore
        # report identical containment, and only ``score_exchange``'s ordered entry list
        # separates a record holding both from a record holding one.
        plants.append((
            "exchange-give-first",
            exchange_turns["exchange_give_first"],
            (EXCHANGE_OBJECT, EXCHANGE_TAKER.lower()),
        ))
        plants.append((
            "exchange-return",
            exchange_turns["exchange_return"],
            (EXCHANGE_OBJECT, EXCHANGE_GIVER.lower()),
        ))
        plants.append((
            "exchange-give-second",
            exchange_turns["exchange_give_second"],
            (EXCHANGE_OBJECT, EXCHANGE_TAKER.lower()),
        ))

    rows = []
    for name, turn, tokens in plants:
        if not turn:
            continue
        narration = posted[turn - 1] if len(posted) >= turn else ""
        rows.append(
            {
                "plant": name,
                "plant_turn": turn,
                "tokens": list(tokens),
                "held": canon_holds(entries, tokens),
                # Mentions, never states: see this function's docstring for the
                # refusal that scored true here. The narration itself is retained in
                # ``per_turn[i].posted_excerpt`` and is the only thing that classifies.
                "narration_mentions_tokens": canon_holds([narration], tokens),
                "turn_wrote_record": "scene_commit"
                in (tools[turn - 1] if len(tools) >= turn else []),
                "sweep": sweeps[turn - 1] if len(sweeps) >= turn else "unmeasured",
            }
        )
    lost = [row for row in rows if not row["held"]["entry"]]
    facts = visible_facts(campaign_root)
    duplicates = duplicate_fact_pairs(facts)
    return {
        "plants": rows,
        "planted": len(rows),
        "held_strict": sum(1 for row in rows if row["held"]["strict"]),
        "held_lenient": sum(1 for row in rows if row["held"]["lenient"]),
        "held_entry": sum(1 for row in rows if row["held"]["entry"]),
        "lost": [row["plant"] for row in lost],
        "lost_with_narration_mentioning_tokens": [
            row["plant"] for row in lost if row["narration_mentions_tokens"]["entry"]
        ],
        "visible_facts": len(facts),
        "duplicate_pairs": len(duplicates),


        "duplicate_examples": duplicates,


        "duplicate_filings": summarize_duplicate_filings(facts),
    }


def commit_discipline_gate(summary: dict) -> tuple[bool, str]:
    """Whether every planted, party-carried fact this run scored reached canon.
    """
    lost = summary["lost"]
    detail = f"{summary['held_entry']} of {summary['planted']} planted facts reached canon"
    if lost:
        detail += f"; lost {lost}"
    return not lost, detail


def score_zone_recall(
    zone: str,
    tokens: tuple[str, ...],
    reply: str,
    canon_text: str,
    in_digest: bool,
    digest_turns: int,
    tool_names: list[str],
    canon_entries_list: list[str] | None = None,
) -> dict:
    """Score one age zone, with the two facts that make a miss classifiable.

    ``committed`` separates a fact that never reached the record from one the narrator
    could not produce. ``in_digest`` states whether the bounded scene block carried the
    fact on the scored turn, which is the variable that distinguishes the zones. A zone
    that fails while ``committed`` holds and ``in_digest`` is false measures delivery
    loss. A zone that fails while both hold measures something else, and the reply
    excerpt classifies it.
    """
    lowered = reply.lower()
    found = [token for token in tokens if token in lowered]
    lenient_reply = normalize_for_lenient_match(lowered)
    lenient_found = [
        token for token in tokens if normalize_for_lenient_match(token) in lenient_reply
    ]
    lowered_canon = canon_text.lower()
    return {
        "zone": zone,
        "fact_tokens": list(tokens),
        "committed": all(token in lowered_canon for token in tokens),


        "held": canon_holds(
            canon_entries_list if canon_entries_list is not None else canon_text.splitlines(),
            tokens,
        ),
        "recalled": len(found) == len(tokens),
        "tokens_in_reply": found,
        "lenient": {
            "recalled": len(lenient_found) == len(tokens),
            "tokens_in_reply": lenient_found,
        },
        "in_digest_at_recall": in_digest,
        "digest_turns": digest_turns,
        "tool_names": list(tool_names),
        "read_tool_called": any(name in CLOCK_READ_TOOLS for name in tool_names),
        "reply_excerpt": reply[:600],
    }


_MARKUP_LEAK_TOKENS: tuple[str, ...] = (*_OPENERS, *_CLOSERS, _ARG_DELIMITER)


_RAW_ERROR_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"reasoningContent",
        r"Traceback \(most recent call last\)",
        r"\b\w+(?:Error|Exception)\b",
        r"\bHTTP/1\.\d \d{3}\b",
        r"\bstatus_code\b",
        r"\bopenai\.[A-Za-z_.]+\b",
    )
)


def find_raw_leaks(text: str) -> list[str]:
    """Return every raw-error or markup leak label ``text`` matches, or an empty list.
    """
    found: list[str] = []
    for token in _MARKUP_LEAK_TOKENS:
        if token in text:
            found.append(f"markup:{token}")
    for pattern in _RAW_ERROR_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(f"error:{match.group(0)}")
    return found


ROLL_UNDER_TOOLS: tuple[str, ...] = (
    "attribute_test", "group_test", "combat_attack", "combat_defend",
)

#: Matches the exact announcement shape skills/bsh-gm/SKILL.md's "Announcing rolls"
#: section defines: "<Name> rolls <ATTRIBUTE>: rolled <total> vs target <target>,
#: <outcome>." Case insensitive and tolerant of "roll"/"rolls", "vs"/"vs."/"versus",
#: and of the older unlabelled "<total> vs <target>" spelling, because the policy
#: names the shape a reader can recognise, not exact punctuation.
ROLL_ANNOUNCEMENT_PATTERN = re.compile(
    r"rolls?\s+(?:STR|DEX|CON|INT|WIS|CHA)\s*:?\s*(?:rolled\s+)?\d+\s*"
    r"(?:vs\.?|versus)\s*(?:a\s+)?(?:target\s+(?:of\s+)?)?\d+\s*,?\s*"
    r"(?:critical\s+)?(?:success|failure)",
    re.IGNORECASE,
)


def announcement_present(text: str, catalog=None) -> bool:
    """Whether the text carries a roll announcement, in English or the catalog's language.
    """
    if ROLL_ANNOUNCEMENT_PATTERN.search(text):
        return True
    return catalog is not None and bool(catalog.roll_pattern().search(text))


def score_instrument_blindness(engine_posts: list[str], catalog=None) -> dict:
    """The turns no instrument could read, stated instead of silently passed.
    """
    from narrator.sweep import stated_coin_totals, stated_hp_totals

    blind: list[int] = []
    digit_turns = 0
    for index, text in enumerate(engine_posts):
        if not re.search(r"\d", text):
            continue
        digit_turns += 1
        if announcement_present(text, catalog):
            continue
        if stated_hp_totals(text) or stated_coin_totals(text):
            continue
        blind.append(index + 1)
    return {"digit_turns": digit_turns, "blind_turns": blind, "blind_turn_count": len(blind)}


def engine_turn_posts(post_log) -> list[str]:
    """Return the posts a real ``engine.run_turn`` call produced, in order.

    The caller still checks ``len(engine_turn_posts(...)) == len(tool_log)`` -- but a
    mismatch now means a real defect in the service's post accounting, not an
    inference limit, and the affected checks fail loud rather than score against a
    misalignment.
    """
    return [post.text for post in post_log if post.origin == "turn"]


def score_roll_announcements(tool_log: list[list[str]], engine_posts: list[str], catalog=None) -> dict:
    """Score every turn whose tools reported a roll-under test against that turn's post.
    """
    roll_turns: list[int] = []
    unannounced_turns: list[int] = []
    for index, names in enumerate(tool_log):
        if not any(name in ROLL_UNDER_TOOLS for name in names):
            continue
        turn = index + 1
        roll_turns.append(turn)
        reply = engine_posts[index] if index < len(engine_posts) else ""
        if not announcement_present(reply, catalog):
            unannounced_turns.append(turn)
    return {
        "roll_turns": roll_turns,
        "roll_turn_count": len(roll_turns),
        "unannounced_turns": unannounced_turns,
        "all_announced": not unannounced_turns,
    }


ROLL_BACKING_TOOLS: tuple[str, ...] = (
    *ROLL_UNDER_TOOLS, "combat_start", "session_close", "grant_runic_weapon",
)


def score_phantom_roll_claims(tool_log: list[list[str]], engine_posts: list[str], catalog=None) -> dict:
    """Score every turn whose post claims a roll verdict with no backing tool call.

    What this measures, precisely: ``tool_log`` (``NarratorEngine._tool_log``) records
    every tool name the model *requested* -- the ``BeforeToolCallEvent`` hook appends
    it before the per-turn ceiling or ``ResolutionGuard`` can cancel the call, so a
    request that never executed still appears here (``src/narrator/engine.py``'s own
    ``_tool_log`` field docstring). \"No name in ``tool_log``\" therefore proves \"no
    matching roll-under tool request\", not \"no matching audit-log entry\" -- a
    requested-but-cancelled-or-refused call followed by a narrated verdict is a
    distinct, currently unaddressed shape this function does not catch, tracked in
    ../DEFERRED.md rather than silently assumed away.

    ``tool_log`` and ``engine_posts`` share ``score_roll_announcements``'s own
    alignment contract: both one entry per real ``engine.run_turn`` call, in call
    order, so the turn numbers this function returns use the identical 1-based,
    tool-log-call-ordered numbering ``roll_turns``/``unannounced_turns`` do -- not the
    raw player-mention count ``report.per_turn`` uses. ``engine_posts`` must be
    ``engine_turn_posts(service.post_log)``, never ``adapter.posted`` directly, for
    the same misattribution reason ``score_roll_announcements`` documents.
    """
    phantom_turns: list[int] = []
    for index, names in enumerate(tool_log):
        if any(name in ROLL_BACKING_TOOLS for name in names):
            continue
        turn = index + 1
        reply = engine_posts[index] if index < len(engine_posts) else ""
        if announcement_present(catalog=catalog, text=reply):
            phantom_turns.append(turn)
    return {
        "phantom_roll_turns": phantom_turns,
        "phantom_roll_turn_count": len(phantom_turns),
        "any_phantom_roll_claims": bool(phantom_turns),
        "turns_scored": len(tool_log),
    }


def phantom_roll_claim_gate(dice_visibility: dict) -> tuple[bool, str]:
    """Whether this run's narration ever claimed a roll verdict with no backing tool call.

    ``dice_visibility`` is ``report.dice_visibility`` after ``run_soak`` merges
    ``score_phantom_roll_claims``'s fields into ``score_roll_announcements``'s own
    dict (see the wiring comment beside ``report.check("phantom-roll-claims", ...)``).
    The gate reads ``phantom_roll_turns`` only.

    Zero tolerance, the same posture ``commit_discipline_gate`` takes for a lost
    planted fact: a narrated roll verdict with no matching tool request behind it is
    not a partial-credit finding to accommodate an isolated low-rate miss -- it is the
    class of defect this project's own invariant forbids at the requested-and-executed
    end of the spectrum (see ``score_phantom_roll_claims``'s own docstring for the
    requested-vs-executed distinction this gate does not yet draw), so any nonzero
    count fails the run. Unlike ``commit_discipline_gate``, nothing here depends on a
    planted probe; the check reads only the tool log and posts every soak run already
    produces, and ``run_soak`` scores it unconditionally rather than behind
    ``--require-continuity`` -- see that call site's own comment for why.

    A dict carrying no ``phantom_roll_turns`` key at all (predating this fix, or a run
    that never reached the ``score_phantom_roll_claims`` merge) reads as zero rather
    than as a failure: ``run_soak`` fails ``roll-turns-announce-outcome`` and
    ``phantom-roll-claims`` separately, and with an explicit ``unscorable:`` detail,
    for the one case (a post-accounting mismatch against the tool log) where the
    alignment itself cannot be trusted -- this function is never called on that
    path, so it does not need to special-case it.
    """
    phantom = dice_visibility.get("phantom_roll_turns", [])
    scored = dice_visibility.get("turns_scored", 0)
    detail = f"{len(phantom)} phantom roll claim(s) among {scored} scored turns"
    if phantom:
        detail += (
            f" at turn(s) {phantom} -- a roll-verdict-shaped post with no "
            f"{list(ROLL_BACKING_TOOLS)} call requested"
        )
    return not phantom, detail


HP_FIGURE_PATTERN = re.compile(
    r"\d+\s*(?:hit points?|hp)\b|hit points?\b[^.]{0,40}\d", re.IGNORECASE
)


HP_BACKING_TOOLS: tuple[str, ...] = (
    "combat_attack", "combat_defend", "rest", "helpless_roll", "character_advance",
    "use_ability", "character_create", "npc_create", "character_sheet", "campaign_status",
)


def score_phantom_hp_claims(tool_log: list[list[str]], engine_posts: list[str]) -> dict:
    """Score every turn whose post claims a hit-point figure with no backing tool call.

    Built directly against ``HP_FIGURE_PATTERN``'s existing, shape-tolerant
    presence check rather than against a new policy-mandated announcement shape:
    unlike \"Announcing rolls,\" ``skills/bsh-gm/SKILL.md`` states no equally rigid
    \"Announcing recoveries\" sentence shape a regex could require, and inventing one
    now would be a live-prompt change this detection-only item's own non-goals
    forbid, not a data-gathering step. ``HP_FIGURE_PATTERN`` (renamed from
    ``REST_HP_FABRICATION_PATTERN``, unchanged in value) already reads real
    narration two ways in production (``score_rest_handling``'s ``\"fabricated\"``
    route, scoped to six scripted rest-declaration turns only) and a third,
    stronger way (``narrator.sweep.stated_hp_totals``/``guard_ratification``
    verifies the claimed *value* against a character's real current ``hp``, not
    merely that a figure was stated) -- this function is the session-wide
    generalisation of the first: every scored turn, not only a rest declaration's
    own position, at the identical requested-tool precision
    ``score_phantom_roll_claims`` already documents (a cancelled or refused call
    still appears in ``tool_log``; see that function's own docstring).

    ``tool_log`` and ``engine_posts`` share ``score_phantom_roll_claims``'s own
    alignment contract: both one entry per real ``engine.run_turn`` call, in call
    order, so the turn numbers this function returns use the identical 1-based,
    tool-log-call-ordered numbering. ``engine_posts`` must be
    ``engine_turn_posts(service.post_log)``, never ``adapter.posted`` directly, for
    the same misattribution reason ``score_phantom_roll_claims`` documents.
    """
    phantom_turns: list[int] = []
    for index, names in enumerate(tool_log):
        if any(name in HP_BACKING_TOOLS for name in names):
            continue
        turn = index + 1
        reply = engine_posts[index] if index < len(engine_posts) else ""
        if HP_FIGURE_PATTERN.search(reply):
            phantom_turns.append(turn)
    return {
        "phantom_hp_turns": phantom_turns,
        "phantom_hp_turn_count": len(phantom_turns),
        "any_phantom_hp_claims": bool(phantom_turns),
        "turns_scored": len(tool_log),
    }


def phantom_hp_claim_gate(dice_visibility: dict) -> tuple[bool, str]:
    """Whether this run's narration ever claimed a hit-point figure with no backing tool call.

    ``dice_visibility`` is ``report.dice_visibility`` after ``run_soak`` merges
    ``score_phantom_hp_claims``'s fields into the same dict
    ``score_phantom_roll_claims``'s fields already occupy (see the wiring comment
    beside ``report.check("phantom-hp-claims", ...)``). The gate reads
    ``phantom_hp_turns`` only.

    Zero tolerance, the identical posture ``phantom_roll_claim_gate`` takes and for
    the identical reason: a narrated hit-point figure with no matching tool request
    behind it is the exact defect class ../CONTRIBUTING.md's own invariant forbids
    ("a response that reports a hit-point total... without a matching audit event
    is a defect"), not a partial-credit finding. Scored unconditionally, the same
    posture and the same reason ``phantom_roll_claim_gate``'s own docstring states:
    this measurement needs no planted probe, only the tool log and posts every soak
    run already produces.

    A dict carrying no ``phantom_hp_turns`` key at all reads as zero rather than as
    a failure, identically to ``phantom_roll_claim_gate``'s own defensive default;
    ``run_soak`` never calls this gate on such a dict in production (the unscorable
    post-accounting-mismatch branch fails ``phantom-hp-claims`` directly with its
    own ``unscorable:`` detail instead).
    """
    phantom = dice_visibility.get("phantom_hp_turns", [])
    scored = dice_visibility.get("turns_scored", 0)
    detail = f"{len(phantom)} phantom hit-point claim(s) among {scored} scored turns"
    if phantom:
        detail += (
            f" at turn(s) {phantom} -- a hit-point-figure-shaped post with no "
            f"{list(HP_BACKING_TOOLS)} call requested"
        )
    return not phantom, detail


#: A turn's post claiming a coin total, mirroring ``HP_FIGURE_PATTERN``'s two-way
#: phrasing tolerance: the amount can lead ("22 coins") or the unit can lead ("coins:
#: you have 22").
COIN_FIGURE_PATTERN = re.compile(
    r"\d[\d,]*\s*(?:coins?|copper|cp|silver|sp|gold|gp)\b"
    r"|(?:coins?|copper|cp|silver|sp|gold|gp)\b[^.]{0,40}\d",
    re.IGNORECASE,
)


COIN_BACKING_TOOLS: tuple[str, ...] = (
    "character_sheet", "use_ability", "character_create", "inventory_update",
)


def score_phantom_coin_claims(tool_log: list[list[str]], engine_posts: list[str]) -> dict:
    """Score every turn whose post claims a coin figure with no backing tool call.

    See ``COIN_FIGURE_PATTERN``/``COIN_BACKING_TOOLS`` above for what counts as a claim
    and what backs one. ``tool_log`` and ``engine_posts`` share
    ``score_phantom_roll_claims``'s own alignment contract: both one entry per real
    ``engine.run_turn`` call, in call order, so the turn numbers this function returns
    use that function's identical 1-based, tool-log-call-ordered numbering.
    ``engine_posts`` must be ``engine_turn_posts(service.post_log)``, never
    ``adapter.posted`` directly, for the same misattribution reason
    ``score_phantom_roll_claims`` documents.
    """
    phantom_turns: list[int] = []
    for index, names in enumerate(tool_log):
        if any(name in COIN_BACKING_TOOLS for name in names):
            continue
        turn = index + 1
        reply = engine_posts[index] if index < len(engine_posts) else ""
        if COIN_FIGURE_PATTERN.search(reply):
            phantom_turns.append(turn)
    return {
        "phantom_coin_turns": phantom_turns,
        "phantom_coin_turn_count": len(phantom_turns),
        "any_phantom_coin_claims": bool(phantom_turns),
        "turns_scored": len(tool_log),
    }


def phantom_coin_claim_gate(dice_visibility: dict) -> tuple[bool, str]:
    """Whether this run's narration ever claimed a coin figure with no backing tool call.

    ``dice_visibility`` is ``report.dice_visibility`` after ``run_soak`` merges
    ``score_phantom_coin_claims``'s fields into the dict its HP and roll siblings
    already occupy. The gate reads ``phantom_coin_turns`` only.

    Zero tolerance, the identical posture ``phantom_hp_claim_gate``/
    ``phantom_roll_claim_gate`` take and for the identical reason: a narrated coin
    figure with no matching tool request behind it is the exact defect class
    ../CONTRIBUTING.md's own invariant forbids ("a response that reports a hit-point
    total... without a matching audit event is a defect" -- a coin total is the same
    durable-state class), not a partial-credit finding.

    A dict carrying no ``phantom_coin_turns`` key at all reads as zero rather than as a
    failure, identically to its HP and roll siblings' own defensive default;
    ``run_soak`` never calls this gate on such a dict in production (the unscorable
    post-accounting-mismatch branch fails ``phantom-coin-claims`` directly with its
    own ``unscorable:`` detail instead).
    """
    phantom = dice_visibility.get("phantom_coin_turns", [])
    scored = dice_visibility.get("turns_scored", 0)
    detail = f"{len(phantom)} phantom coin claim(s) among {scored} scored turns"
    if phantom:
        detail += (
            f" at turn(s) {phantom} -- a coin-figure-shaped post with no "
            f"{list(COIN_BACKING_TOOLS)} call requested"
        )
    return not phantom, detail


_REST_MENTION_OFFSET = next(
    index for index, template in enumerate(_MENTIONS) if "short rest" in template
) + 1
assert _REST_MENTION_OFFSET == 5, (
    "_MENTIONS reordered: the short-rest template moved, and every raw-turn figure "
    "The repeated-declaration regression fixture requires offset 5"
)
_REST_MENTION_CYCLE = len(_MENTIONS)


def rest_declaration_turns(
    total_mentions: int, claimed_turns: set[int] | None = None
) -> list[int]:
    """Raw turn numbers (1-based, ``report.per_turn``'s own numbering) scripting a
    short-rest declaration, for a run generating ``total_mentions`` raw turns.
    """
    claimed = claimed_turns or set()
    return [
        turn
        for turn in range(_REST_MENTION_OFFSET, total_mentions + 1, _REST_MENTION_CYCLE)
        if turn not in claimed
    ]


REST_REFUSAL_PATTERN = re.compile(r"cannot resolve", re.IGNORECASE)


def score_rest_handling(
    raw_posted: list[str],
    rest_turns: list[int],
    tool_log: list[list[str]],
    aligned_posts: list[str],
) -> dict:
    """Score how many of a session's rest declarations produced a `rest` tool-call request.

    This is a *request-level* fact, not a completion-level one, stated no stronger
    than that: a cancelled or refused call still appears in ``tool_log`` (its own
    field docstring; ``score_phantom_roll_claims``'s docstring states the identical
    precision for the same reason).
    """
    raw_to_engine_index: dict[int, int] = {}
    search_from = 0
    for engine_index, post in enumerate(aligned_posts):
        while search_from < len(raw_posted) and raw_posted[search_from] != post:
            search_from += 1
        if search_from >= len(raw_posted):
            break
        raw_to_engine_index[search_from + 1] = engine_index  # 1-based raw turn number
        search_from += 1
    posted_by_turn = {index + 1: text for index, text in enumerate(raw_posted)}

    declarations = []
    for turn in rest_turns:
        text = posted_by_turn.get(turn, "")
        engine_index = raw_to_engine_index.get(turn)
        tool_requested = bool(
            engine_index is not None
            and engine_index < len(tool_log)
            and "rest" in tool_log[engine_index]
        )
        if tool_requested:
            route = "tool_requested"
        elif REST_REFUSAL_PATTERN.search(text):
            route = "refused"
        elif HP_FIGURE_PATTERN.search(text):
            route = "fabricated"
        else:
            route = "silent"
        declarations.append({"turn": turn, "route": route})

    resolved = sum(1 for entry in declarations if entry["route"] == "tool_requested")
    return {
        "rest_declaration_turns": list(rest_turns),
        "rest_declaration_count": len(rest_turns),
        "rest_tool_requested_count": resolved,
        "rest_tool_requested_rate": (resolved / len(rest_turns)) if rest_turns else None,
        "rest_declarations": declarations,
    }


def score_thinking_fallbacks(
    fallback_turns: Sequence[int], tool_log: Sequence[Sequence[str]]
) -> list[dict]:
    """Each thinking-budget fallback, and whether the retry reached a tool.

    ``fallback_turns`` is ``NarratorEngine._thinking_fallbacks``: one 1-based turn
    number per attempt that hit its answer bound with no tool run and was re-run with
    thinking suppressed. ``tool_log`` is the engine's own per-turn tool names, the same
    list ``SoakReport.tools[\"per_turn\"]`` publishes, indexed from zero.

    The join is sound rather than a second reading of the same fact: the retry is
    entered *only* from an attempt that ran no tool (``_invoke_channel_agent`` re-raises
    otherwise), so every name in that turn's entry belongs to the retry.
    """
    by_turn = {index + 1: list(names) for index, names in enumerate(tool_log)}
    return [
        {
            "turn": turn,
            "tool_called": bool(by_turn[turn]) if turn in by_turn else None,
            "tools": by_turn.get(turn, []),
        }
        for turn in fallback_turns
    ]


@dataclass
class SoakReport:
    """Everything the evidence record needs, in one serialisable object."""

    seed: int
    window_size: int
    generated_messages: int
    fed_messages: int = 0
    authors: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    delivered: int = 0
    withheld: int = 0
    settle_attempts: int = 0
    commits: int = 0
    waives: int = 0
    leaks_scrubbed: int = 0
    errors: list[str] = field(default_factory=list)
    removed_message_count: int = 0
    final_message_count: int = 0
    compaction_triggered: bool = False
    #: The continuity probe's result, or None when the probe is off. ``committed``
    #: and ``recalled`` are separated so a failure names its stage: a fact that never
    #: reached canon is a commit-discipline problem, a fact in canon the narrator
    #: cannot produce is a retrieval problem.
    continuity: dict | None = None
    #: The authored-canon mode's result, or None when off. It plants nothing: the
    #: scored tokens pre-exist in the current location's world file, so a pass proves
    #: the authored-canon delivery channel, which measured entirely absent before the
    #: digest (the location resource had no reader).
    continuity_authored: dict | None = None
    #: Per-turn canon digest measurements from the engine's log: the prompt-size
    #: delta this slice adds, the truncation counters that trigger stage two, and
    #: per-section survival of the scene render into the bounded block.
    canon: dict | None = None
    #: The clock probe's result, or None when the probe is off. It scores the section
    #: the bound's byte-position eviction takes first among player-visible sections.
    clock: dict | None = None
    #: The zone probe's three results, or None when the probe is off. One fact inside
    #: the conversation window, one outside it but inside the digest's retained head,
    #: and one outside both. The continuity probe scores the middle case only, so a
    #: selection change could trade one zone for another without any gate noticing.
    zones: dict | None = None
    #: The ordering probe's result, or None when the probe is off. Every other recall
    #: mode scores one fact at one moment, so none of them measures sequence. This one
    #: moves one object through three places and scores whether the reply produces them
    #: in the recorded order.
    order: dict | None = None
    #: The give-and-return probe's record, or None when the probe is off. It scores no
    #: recall and gates nothing: it retains what the record says about one object passed
    #: three times, in record order, so a reader can test whether a record mechanism
    #: preserved the last pass. See ``score_exchange``.
    exchange: dict | None = None


    payload: dict | None = None


    commit_discipline: dict | None = None


    cache: dict | None = None
    #: Per-turn tool names from the engine's log, with a count per name. It answers
    #: one question the recall probes cannot: did the narrator read canon from the
    #: injected digest, or call a tool for it?
    tools: dict | None = None
    #: Per-turn sweep dispositions from the engine's log, with counts per kind.
    #: ``record`` entries land as ``scene_commit`` events in the campaign audit
    #: trail, but the ``commits`` counter above excludes them: it counts
    #: settle-commit turns only (``src/narrator/service.py``). Total accrual
    #: therefore comes from the audit trail; this field is the sweep's only count.
    sweep: dict | None = None


    routing: dict | None = None
    #: Turns whose digits no instrument parsed (``score_instrument_blindness``).
    instrument_blindness: dict | None = None
    #: The hazard probe's per-plant rows and counts (``score_hazard_probe``).
    hazard: dict | None = None


    recovery: dict | None = None


    planner_recent_narration: dict | None = None


    promised_unbound_checks: dict | None = None
    #: Per-``plan_turn`` risk-confirmation-branch entry from
    #: ``NarratorEngine._risk_branch_log``: how often the branch fires, how often
    #: ``verify_plan`` would itself await a model call (``would_yield_count``), and
    #: how often the branch reaches the point where ``assess_hazard`` is actually
    #: called (``reached_confirmation_count``). Sizes the latency a speculative
    #: ``assess_hazard`` task started beside ``verify_plan`` could actually overlap
    #: with, versus dispatched-and-discarded (``would_yield`` true, confirmation not
    #: reached) versus never dispatched at all (``would_yield`` false). None when the
    #: branch never fired.
    risk_branch: dict | None = None


    raw_leaks: dict | None = None


    dice_visibility: dict | None = None


    rest_handling: dict | None = None
    per_turn: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "pass": passed, "detail": detail})

    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["pass"]]


def _canon_text(campaign_root: Path) -> str:
    """Both authoritative canon files, concatenated, with absent files contributing none.

    Every recall probe separates a fact the model never wrote from a fact it could not
    produce, and every one of them asks that question of these two files.
    """
    text = ""
    for relative in ("campaign/scene.md", "campaign/state.json"):
        path = campaign_root / relative
        if path.is_file():
            text += path.read_text(encoding="utf-8")
    return text


def _authored_canon_text(repo_root: Path) -> str:
    """The authored ``world/`` tree, with one NUL between files, and nothing else.

    Reading the whole tree rather than the one current location file costs a wider canon
    and buys immunity to a scene that moved. A span quoting another location's authored
    text scores supported, which understates the defect rather than inventing one.

    The NUL separator blocks a match that spans two files. Without it, the last words of
    one file and the first of the next would credit a quotation no document states.
    """
    world = repo_root / "world"
    parts = [
        _read_or_empty(path)
        for path in (sorted(world.rglob("*")) if world.is_dir() else [])
        if path.is_file()
    ]
    return "\x00".join(part for part in parts if part)


def _read_or_empty(path: Path) -> str:
    """One source, fail-open: an unreadable or undecodable file contributes nothing.

    A quotation scorer that raises on one stray file costs the run its whole measurement.
    ``_read`` in ``src/narrator/canon.py`` takes the same position for the same reason.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _clock_fill_state(campaign_root: Path) -> str:
    """The probe clock's filled-over-segments string at scoring time, or an empty string.

    The narrator may advance a clock during play, so the probe compares the reply
    against the record's value at scoring time rather than against the planted value.
    """
    path = campaign_root / "campaign" / "state.json"
    if not path.is_file():
        return ""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for clock in state.get("clocks", []):
        if clock.get("id") == CLOCK_ID:
            return f"{clock.get('filled')}/{clock.get('segments')}"
    return ""
