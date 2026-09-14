"""A deterministic, offline stand-in for the model turn classifier.

This is TEST CODE. It is the English lexical classifier that used to live in
``src/narrator/interactions.py`` and decide the risk floor, moved here verbatim when
``narrator.classify`` replaced it. Production no longer contains it, and no module
under ``src/`` imports this file.

Why keep it at all. The offline suite must route turns without reaching the served
endpoint, and it must do so identically on every run, so it needs a deterministic
double. Reusing the retired lexicon as that double means several hundred existing
assertions keep measuring the behavior they were written for, rather than being
rewritten against a stub whose answers were chosen to make them pass.

What this does NOT establish. The double is English-only and it is measurably wrong --
that is why it was retired. ``tests_narrator/declaration_corpus.py`` records six cases
it gets wrong that the model classifier gets right. So an offline test passing here is
evidence about the *engine's wiring*, never about classification quality; the only
acceptance for that is ``tests_narrator/test_probe_classifier.py`` against the live
endpoint. Do not add a case here expecting it to certify the classifier.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from narrator.channels.base import LOCAL_COMMANDS, RECOVERY_CONTROLS
from narrator.policy_types import (  # shared production types, never redefined here
    CombatSnapshot,
    InteractionCue,
    TrustedScope,
    TurnPolicy,
)
from narrator.social import (
    _MECHANICAL_SOCIAL_CATEGORIES,
    SocialCategory,
    TradePhase,
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_WORDS = re.compile(r"[a-z]+")
_VOWELS = frozenset("aeiou")
_QUESTION_PREFIXES = (
    "what ", "who ", "where ", "when ", "why ", "which ", "how ",
    "can ", "do ", "does ", "is ", "are ", "whats ", "what's ",
    "what’s ",
)
_OBSERVATION_PREFIXES = (
    "look", "observe", "watch", "examine", "inspect", "listen", "read", "scan", "study",
    "survey",
)
_THEFT_ACTIONS = frozenset(
    {"filch", "grab", "lift", "loot", "pilfer", "pickpocket", "rob", "snatch", "steal", "swipe", "take"}
)
_VIOLENT_ACTIONS = frozenset(
    {
        "assault", "attack", "batter", "beat", "bite", "bludgeon", "choke", "club", "cut",
        "elbow", "eviscerate", "fight", "fling", "garrote", "gouge", "harm", "headbutt", "hit",
        "hurl", "kick", "kill", "knee", "maim", "maul", "murder", "punch", "shoot", "slap",
        "slash", "smother", "stab", "stomp", "strangle", "strike", "tackle", "throttle",
        "trample", "wound",
    }
)
_DESTRUCTIVE_ACTIONS = frozenset(
    {
        "abduct", "annihilate", "banish", "break", "burn", "collapse", "demolish", "destroy",
        "disfigure", "dismantle", "erase", "execute", "exile", "flood", "obliterate", "poison",
        "pulverize", "raze", "ruin", "sabotage", "seal", "sever", "shatter", "smash", "torch",
        "void", "wreck",
    }
)
_OOC_MARKERS = frozenset(
    {"ooc", "gm", "narrator", "system", "rules", "rule", "mechanic", "mechanics"}
)
_SOCIAL_WORDS = frozenset(
    {"hello", "hi", "greetings", "greet", "approach", "speak", "talk", "say", "ask", "introduce"}
)
_CONSEQUENTIAL_SOCIAL_WORDS = frozenset(
    {"threat", "threaten", "bribe", "deceive", "deception", "lie", "buy", "sell", "trade", "promise", "secret", "disclose", "reveal"}
)
_COMMITTED_ACTIONS = frozenset(
    {"bind", "cast", "climb", "close", "enter", "give", "go", "move", "open", "take"}
)
_DEPARTURE_WORDS = frozenset({"depart", "leave", "retreat", "withdraw"})


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
                tokens.update(_WORDS.findall(value.casefold()))
    return tuple(sorted(tokens))


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
        if not isinstance(npc_id, str) or not _IDENTIFIER.fullmatch(npc_id):
            continue
        status = record.get("status") if isinstance(record, dict) else None
        if isinstance(status, str) and status in _NPC_STATUSES:
            states.append((npc_id, status))
    return tuple(sorted(states))


_COMBATANT_LINE = re.compile(
    r"^- combatant:\s+(?P<id>[a-z0-9][a-z0-9_-]{0,63})\s+side=(?P<side>pc|npc)\b"
)


#: Common non-hostile person nouns. A declaration that names one of these targets a
#: bystander, so the risk floor keeps its confirmation even mid-fight. The set need not
#: be complete: an unrecognized target noun survives as a residual word, which also
#: withholds the sanction. This fails safe toward the confirmation.
_BYSTANDER_NOUNS = frozenset(
    {
        "acolyte", "acolytes", "barkeep", "barkeeps", "bartender", "boy", "boys",
        "bystander", "captain", "captains", "child", "children", "citizen", "citizens",
        "civilian", "civilians", "cook", "cooks", "doctor", "elder", "elders", "farmer",
        "farmers", "friend", "girl", "girls", "guard", "guards", "guy", "guys",
        "innkeeper", "innocent", "king", "kings", "knight", "knights", "librarian",
        "man", "men", "merchant", "merchants", "monk", "monks", "noble", "nobles",
        "patron", "patrons", "peasant", "person", "priest", "priests", "queen", "queens",
        "sailor", "sailors", "servant", "servants", "shopkeeper", "soldier", "soldiers",
        "stranger", "student", "thief", "thieves", "trader", "traders", "vendor",
        "vendors", "villager", "villagers", "waiter", "witness", "woman",
    }
)

#: Function words a violence declaration may carry without naming a target: articles,
#: prepositions, conjunctions, and pronouns. The sole-NPC sanction fires only when the
#: declaration, minus the violence verb, the named enemy, and these, leaves no residual
#: content word. Any residual word is a possible other target, so it withholds the sanction.
_TARGETLESS_WORDS = frozenset(
    {
        "a", "an", "and", "at", "back", "down", "for", "he", "her", "here", "him", "his",
        "in", "into", "it", "its", "me", "my", "near", "next", "of", "on", "one", "our",
        "over", "she", "that", "the", "their", "them", "then", "there", "they", "this", "to",
        "up", "us", "we", "with", "you", "your",
    }
)


def _identifier_tokens(identifier: str) -> frozenset[str]:
    return frozenset(_WORDS.findall(identifier.replace("-", " ")))


_DEFENCE_METHODS = frozenset({"dodge", "parry"})

#: Intent words a defence answer may carry while still declaring only the defence.
#: A player answering the game master's parry-or-dodge question writes "dodge", "I'll try
#: to dodge", or "parry it". Those add no second action, so they keep the binding. Every
#: other content word withholds it, which is what keeps a compound declaration out.
#:
#: The first-person pronoun sits here rather than in ``_TARGETLESS_WORDS``, which omits
#: it. That set also feeds ``combat_sanctions_violence``, so widening it would change
#: which declarations the risk floor sanctions. This set reaches only the defence
#: binding, so it carries the pronoun without touching the floor.
_DEFENCE_INTENT_WORDS = frozenset(
    {
        "am", "attempt", "attempts", "aside", "away", "going", "i", "just", "ll",
        "quickly", "try", "tries", "trying", "want", "wants", "will", "would",
    }
)


def defence_method(declaration: str, combat: CombatSnapshot | None) -> str:
    """Return the defence a lone dodge or parry declaration selects, or the empty string.

    The Black Sword Hack defences are exactly dodge and parry. The player rolls one on the
    enemy's turn. This returns that word when a fight runs and the declaration reduces to
    it alone. The engine then binds ``combat_defend`` with no planner round trip.

    The residual subtraction on the last guard is this slice's operative safety property.
    "I dodge his swing and stab him" leaves "swing", "his", and "stab" behind. Those words
    survive the subtraction, so this returns the empty string. The declaration then reaches
    the risk floor that owns violence, which is the outcome that matters.

    It also returns the empty string outside an open fight. It returns it when neither or
    both defence words appear. Each of those answers falls through to ordinary routing.
    """
    if combat is None or not combat.active:
        return ""
    words = _word_set(declaration)
    methods = words & _DEFENCE_METHODS
    if len(methods) != 1:
        return ""
    if words - _DEFENCE_METHODS - _TARGETLESS_WORDS - _DEFENCE_INTENT_WORDS:
        return ""
    return next(iter(methods))


_CLOSING_MARKS = "\"'`*_)]}’”"


#: Words an answer carries besides the fact it supplies: affirmations, negations, and the
#: copulas and naming words that frame one. They are not function words, so
#: ``_TARGETLESS_WORDS`` omits them, and an answer needs them.
_ANSWER_WORDS = frozenset(
    {
        "am", "are", "aye", "be", "been", "call", "called", "is", "maybe", "nah", "name",
        "named", "never", "no", "nope", "not", "okay", "only", "perhaps", "still", "sure",
        "was", "were", "yeah", "yep", "yes",
    }
)

#: A first-person subject marks a declared act rather than an answer. "I pick the lock"
#: carries one; "My name is Rill" does not, because "my" is possessive.
_FIRST_PERSON_SUBJECTS = frozenset({"i", "we", "let", "lets"})

#: The copulas and naming verbs an answer puts in front of the fact it supplies. An
#: unrecognized word earns its place only directly behind one of these, which is what
#: separates "my name is Torvald" from "then run" and "back away". Both of those put an
#: unrecognized verb behind an ordinary function word, and neither is an answer.
_ANSWER_PREDECESSORS = frozenset(
    {"am", "are", "be", "been", "call", "called", "is", "name", "named", "was", "were"}
)


#: Transport words a channel reserves for decision recovery and local commands. The
#: service consumes a recovery control on a branch below the social route, so an answer
#: must never claim one. These read the channel contract's own names rather than
#: restating them, because an audit found a restated copy able to drift from the parser.
_RECOVERY_CONTROLS = RECOVERY_CONTROLS | LOCAL_COMMANDS


def _mechanic_vocabulary() -> frozenset[str]:
    """Every word this module already reserves for a mechanic, named rather than listed.

    An answer never carries one of these, and a declaration that does is asking for the
    mechanic. Composing the set from the defining sets means a verb added to any of them
    tightens this bound with no edit here, which is the property a hand-copied list loses.
    """
    return (
        _COMMITTED_ACTIONS
        | _THEFT_ACTIONS
        | _VIOLENT_ACTIONS
        | _DESTRUCTIVE_ACTIONS
        | _DEPARTURE_WORDS
        | _SOCIAL_WORDS
        | _CONSEQUENTIAL_SOCIAL_WORDS
        | _OOC_MARKERS
        | _DEFENCE_METHODS
        | _RECOVERY_CONTROLS
        | frozenset(_OBSERVATION_PREFIXES)
    )


def reply_shaped(declaration: str, party_name_tokens: tuple[str, ...] = ()) -> bool:
    """Whether a declaration reads as an answer rather than as a declared act.

    An armed invitation alone cannot decide this. The player may answer the question, and
    the player may ignore it and act. Both arrive at ``classify_turn``'s fallback, so
    without this test the invitation would route "I pick the lock" as conversation.

    Three conditions bound it, and each fails toward the planner route. The declaration
    carries no first-person subject, because one marks a declared act. At most one word
    sits outside the recognized vocabulary. That outside word stands directly behind a
    copula or naming verb, which is the position an answer puts its fact in.

    The recognized vocabulary is ``_TARGETLESS_WORDS``, ``_ANSWER_WORDS``, and the party's
    own character names. Names are recognized rather than exempted, and an audit
    established why that distinction matters. An earlier version exempted every one-word
    declaration instead, which admitted "dodge" and "parry" and so defeated the accepted
    slice that binds them to ``combat_defend``. It admitted the decision-recovery controls
    for the same reason. Under the rule as written, a one-word declaration passes only
    when the vocabulary already holds it, so no unlisted verb reaches conversation.

    The predecessor condition is what an audit forced, and it replaced a weaker rule that
    only refused an unrecognized word in the opening position. That rule admitted "then
    run", "back away", and "into the water", because each puts its unrecognized verb
    behind an ordinary function word. Requiring a copula instead admits "my name is
    Torvald" and refuses all three.

    The bound declines answers it cannot recognize, which is the safe direction. "Rill of
    the low water" leaves two unrecognized words beside the name and keeps the planner
    route, which is the behavior that shipped before the invitation.
    """
    words = [word for word in _WORDS.findall(declaration.casefold()) if word]
    if not words or len(words) > 6:
        return False
    if any(word in _FIRST_PERSON_SUBJECTS for word in words):
        return False
    if any(word in _mechanic_vocabulary() for word in words):
        return False
    known = _TARGETLESS_WORDS | _ANSWER_WORDS | frozenset(party_name_tokens)
    outside = [index for index, word in enumerate(words) if word not in known]
    if len(outside) > 1:
        return False
    if outside and (outside[0] == 0 or words[outside[0] - 1] not in _ANSWER_PREDECESSORS):
        return False
    return True


#: The subset of ``_ANSWER_WORDS`` that actually commits to yes or no, rather than an
#: identity fact ("my name is Torvald") or genuine uncertainty ("maybe", "perhaps").
_YES_NO_WORDS = frozenset(
    {"yes", "yeah", "yep", "aye", "sure", "okay", "no", "nope", "nah", "never", "not"}
)


def is_bare_yes_or_no(declaration: str) -> bool:
    """Whether a declaration is nothing but a yes/no-committing answer word.

    "Yes", "yeah", "no", "nope" and similar, with nothing else surviving besides the
    function words ``_TARGETLESS_WORDS`` already permits, is unambiguous only against
    whatever question was just asked -- see ``narration_invites_reply``. This names the
    shape; it says nothing about what the reply answers.
    """
    words = _word_set(declaration)
    if not words & _YES_NO_WORDS:
        return False
    return not (words - _TARGETLESS_WORDS - _YES_NO_WORDS)


def _word_set(text: str) -> frozenset[str]:
    return frozenset(_WORDS.findall(text.casefold()))


def _is_question(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return "?" in normalized or normalized.startswith(_QUESTION_PREFIXES)


#: Sentence boundaries let the double distinguish interrogative clauses from
#: declarations elsewhere in the same message.
_CLAUSE_BOUNDARY = re.compile(r"[.!?]")

#: Openers ``_committed_question`` treats as making their own clause interrogative.
#: ``_QUESTION_PREFIXES`` alone is not enough here: "I stab the trader?" is a declared
#: act voiced with a raised, uncertain question mark, and it must stay committed, so an
#: ending "?" alone can never disqualify a clause. What must disqualify one is an
#: opening word that actually asks something. The plain wh-/aux openers this module
#: already recognizes cover most of that; the modal openers below extend them, because
#: "Should I take him?" carries none of the former and would otherwise let the first-
#: person regex match its embedded "i take" as if it were a plain declaration.
_COMMITTED_QUESTION_OPENERS = _QUESTION_PREFIXES + (
    "should ", "would ", "could ", "might ", "may ", "will ", "shall ",
    "did ", "was ", "were ", "have ", "has ",
)


#: Marks a new committed action within the same clause, regardless of how the clause
#: itself opens. ``_committed_question`` splits on this in addition to sentence-ending
#: punctuation: an audit found that a compound clause like "Should I open the door, then
#: I stab reed thug?" lost its committed action entirely, because the whole clause opens
#: with "should" and the earlier version discarded everything in a disqualified clause,
#: including a later "then <verb>" segment the module elsewhere explicitly supports.
_THEN_BOUNDARY = re.compile(r"\bthen\b")


def _committed_question(text: str, actions: frozenset[str]) -> bool:
    """Committed question. Synthetic fixtures exercise this contract."""
    normalized = " ".join(text.casefold().split())
    if "?" not in normalized:
        return False
    alternatives = "|".join(sorted(actions))
    direct = re.compile(rf"\b(?:i|we)\s+(?:will\s+|am\s+going\s+to\s+)?(?:{alternatives})\b")
    bare_action = re.compile(rf"^(?:{alternatives})\b")
    position = 0
    clauses: list[str] = []
    for boundary in _CLAUSE_BOUNDARY.finditer(normalized):
        clauses.append(normalized[position:boundary.start()].strip())
        position = boundary.end()
    tail = normalized[position:].strip()
    if tail:
        clauses.append(tail)
    for clause in clauses:
        if not clause:
            continue
        for segment in _THEN_BOUNDARY.split(clause):
            segment = segment.strip()
            if not segment or segment.startswith(_COMMITTED_QUESTION_OPENERS):
                continue
            if direct.search(segment) or bare_action.match(segment):
                return True
    return False


#: Possessive pronouns that name a person as the possessor when a theft-shaped
#: declaration carries one: "take HIS sword", "steal HER necklace", "grab THEIR purse".
#: An audit found a bare "their" carried no signal here even outside the engaged-focus
#: path ``related`` already covers, which is the gap this set closes. A second audit
#: later found this set matched anywhere in the declaration was itself too broad --
#: see ``_possessive_names_a_person``, which is the position-scoped check that actually
#: reads it now.
_POSSESSIVE_PERSON_PRONOUNS = frozenset({"his", "her", "their"})


def _person_words(scope: TrustedScope) -> frozenset[str]:
    """Every word that denotes a specific, actual person in this turn's context: every
    generic bystander noun, plus every present NPC's own identifier tokens.

    Combining the two into one vocabulary is what lets each position-scoped check below
    (source clause, genitive owner, bare object) test a present NPC's name and an
    ordinary bystander noun through the same structural rule, rather than repeating the
    same three positions once per vocabulary -- the inconsistency an audit found between
    an already-scoped source-clause check and a still-unscoped bystander-noun check.
    """
    words = set(_BYSTANDER_NOUNS)
    for npc_id in scope.present_npc_ids:
        words |= _identifier_tokens(npc_id)
    return frozenset(words)


#: Prepositions that introduce a source a theft removes something from. "off" joins
#: "from" because "steal the queen off the chessboard" is exactly as natural a phrasing
#: as "... from the chessboard", and an audit used it as one of its adversarial cases.
_SOURCE_PREPOSITIONS = frozenset({"from", "off"})

#: Object pronouns a source clause commonly uses when the person was already named:
#: "take it from him". Kept apart from ``_TARGETLESS_WORDS``, which also lists "him"/
#: "her" as function words a violence declaration may carry without naming a target --
#: that set answers a different question (is this word content-bearing at all) than
#: this one (does this word itself denote a person).
_SOURCE_OBJECT_PRONOUNS = frozenset({"him", "her", "them"})

#: Coordinating words that end a clause a source or bare-object scan must never cross:
#: each starts content with no established relationship to the object or person the
#: earlier clause named. "take the sword AND the guard fled" must never let "guard"
#: read as part of the same clause "the sword" belongs to.
_CLAUSE_CONJUNCTIONS = frozenset({"and", "but", "then", "while"})

#: Punctuation marks that also end a clause. ``_WORDS`` extracts letters only, so a
#: comma or period between two words normally leaves no trace once tokenized; the
#: combined word-and-punctuation tokenizer below keeps this visible specifically so a
#: clause scan can stop on it.
_CLAUSE_PUNCTUATION = frozenset({",", ".", ";", ":", "!", "?"})

_WORD_OR_CLAUSE_PUNCTUATION = re.compile(r"[a-z]+|[,.;:!?]")

#: Every verb whose own object could otherwise be mistaken for still belonging to an
#: earlier clause's object or source: "take the sword, ATTACK the guard" must not let
#: the second clause's "guard" read as though "take" still governed it.
_CLAUSE_VERB_BOUNDARIES = _THEFT_ACTIONS | _VIOLENT_ACTIONS | _DESTRUCTIVE_ACTIONS


def _clause_after(tokens: list[str], start: int) -> list[str]:
    """Every token from ``start`` up to, but not including, the next clause boundary: a
    comma or sentence-ending punctuation mark, a coordinating conjunction, another verb
    that would start unrelated content, or a source preposition ("from"/"off") opening
    a different clause of its own. An audit found a fixed token-count window here the
    wrong fix entirely: three tokens covers one descriptive word before a person noun,
    four covers two, and no fixed number covers all of them, because the real boundary a
    natural-language description needs is grammatical, not numeric. Scanning to the
    clause boundary instead handles an arbitrary run of descriptive words for free, the
    same way a comma or "and" already bounds any other scan in this module.

    The source-preposition boundary is what keeps a theft verb's own object clause from
    silently absorbing a trailing "from"/"off" source clause's content: an audit found
    "take the QUEEN from the TARNISHED board" reading as one clause let "tarnished" -- a
    trailing signal that genuinely describes a person elsewhere in this module -- get
    misattributed to "queen", when it actually describes "board", three words later and
    past an intervening "from". ``_source_names_a_person`` already starts its own scan
    immediately after the preposition, so the two clauses now never overlap.
    """
    clause: list[str] = []
    for token in tokens[start:]:
        if (
            token in _CLAUSE_PUNCTUATION
            or token in _CLAUSE_CONJUNCTIONS
            or token in _CLAUSE_VERB_BOUNDARIES
            or token in _SOURCE_PREPOSITIONS
        ):
            break
        clause.append(token)
    return clause


#: Prepositions that can open a trailing phrase describing *where* something or someone
#: is, rather than naming a second, separate object: "the man IN THE CORNER" (a fixture
#: this module has carried since AUD-2), "the guard NEAR THE DOOR". ``_clause_head``
#: drops the entire phrase from the preposition onward as one unit before asking what a
#: clause's own head noun is, because the object of that preposition ("corner", "door")
#: is not what the clause is about.
_TRAILING_LOCATION_PREPOSITIONS = frozenset(
    {"in", "on", "at", "near", "next", "over", "under", "behind", "beside", "by"}
)

#: Adverbs describing where something or someone is, positioned after the noun rather
#: than before it: "the guard NEARBY", "the guard standing THERE". Deliberately its own
#: small set rather than a reuse of ``_TARGETLESS_WORDS``: an audit found that reusing
#: the broader set here (which also lists plain articles like "the") let a bare article
#: anywhere in a longer tail validate it, which silently defeated the very check this
#: exists to run -- "take the queen FROM THE board" has a "the" sitting right in its
#: tail, and that "the" must never be read as evidence the tail describes "queen".
_TRAILING_LOCATION_ADVERBS = frozenset({"nearby", "away", "there", "here"})

#: Words that open a relative clause, an infinitive, or an "X of Y" role phrase
#: describing the noun before them: "the guard WHO arrived late", "the guard sent TO
#: watch", "the guard captain OF the watch". Recognizing any of these signals that the
#: material after it still describes the same person, rather than naming a second,
#: distinct object the way "guard RAIL" or "guard STATION" does.
_RELATIVE_OR_INFINITIVE_SIGNALS = frozenset({"who", "that", "which", "to", "of"})


def _clause_head_word(clause: list[str], person_words: frozenset[str]) -> str | None:
    """The clause's own head noun, if it is a person word, once trailing material that
    describes it, rather than naming something else, is accounted for; ``None`` if the
    clause does not reduce to a person word at all.

    "the OLD GUARD standing there", "the GUARD nearby", and "the MAN in the corner" all
    reduce to a person word regardless of how many descriptive words came before it:
    "the very tall old grizzled bearded guard" reduces the same way "the old guard"
    does, because nothing about resolving the head noun depends on how long its own
    lead-in was. A trailing locational prepositional phrase ("in the corner") drops as
    one unit first. What remains is scanned for its rightmost person word; if nothing
    else follows that word, or if a recognized trailing signal -- a locational adverb, a
    relative pronoun, an infinitive "to", or a present participle or past-tense/past-
    participle verb form (either morphological ending, "-ing" or "-ed") -- appears
    anywhere after it, the clause still reduces to that person: "the guard STANDING
    WATCH" (an audit found this specific shape regressed a prior repair: the trailing
    "watch" is not itself a recognized shape, but "standing" beside it is, so the
    person word it describes still counts), "the guard SENT TO watch", "the guard WHO
    ARRIVED late". "the guard RAIL" and "the old wooden guard STATION" do not reduce to
    a person word, because their tail ("rail", "station") carries no such signal at
    all -- it is the clause's own real head noun, not material describing "guard".

    Returning the matched word itself, rather than only whether one exists, is what lets
    a caller judge that specific word's own ambiguity (see ``_AMBIGUOUS_PERSON_WORDS``)
    without re-deriving which word actually matched.
    """
    tokens = list(clause)
    for index, token in enumerate(tokens):
        if token in _TRAILING_LOCATION_PREPOSITIONS:
            tokens = tokens[:index]
            break
    last_person_index: int | None = None
    for index, token in enumerate(tokens):
        if token in person_words:
            last_person_index = index
    if last_person_index is None:
        return None
    tail = tokens[last_person_index + 1 :]
    if not tail:
        return tokens[last_person_index]
    if any(
        word in _TRAILING_LOCATION_ADVERBS
        or word in _RELATIVE_OR_INFINITIVE_SIGNALS
        or word.endswith("ing")
        or word.endswith("ed")
        for word in tail
    ):
        return tokens[last_person_index]
    return None


def _clause_head(clause: list[str], person_words: frozenset[str]) -> bool:
    """Whether a clause's own head noun is a person word. See ``_clause_head_word``."""
    return _clause_head_word(clause, person_words) is not None


def _source_names_a_person(text: str, scope: TrustedScope) -> bool:
    """Whether a "from"/"off" source clause names a person, not a container or place.

    An audit found an earlier version of this signal fired on the bare source
    preposition alone, regardless of what followed it, which misclassified ordinary
    container phrasing ("take the key from the drawer", "grab the sword from the rack")
    as theft. The repair after that required a positive match against ``_person_words``
    in the single word directly following the preposition -- rather than trying to
    enumerate every container or piece of furniture a source clause could instead name,
    the same "positive signal, not an exclusion list" principle ``_violent_words``'
    regular-inflection matching already applies. A second audit found that single-token
    capture too narrow (any descriptive word defeated it); a fixed-width window fixed
    for one descriptive word then failed the same way for two. This reads the whole
    clause after the preposition, bounded by ``_clause_after``, and asks ``_clause_head``
    whether it reduces to a person word, which is correct regardless of how many
    descriptive words the clause carries.
    """
    normalized = " ".join(text.casefold().replace("’", "'").split())
    tokens = _WORD_OR_CLAUSE_PUNCTUATION.findall(normalized)
    person_words = _person_words(scope) | _SOURCE_OBJECT_PRONOUNS
    for index, token in enumerate(tokens):
        if token not in _SOURCE_PREPOSITIONS:
            continue
        if _clause_head(_clause_after(tokens, index + 1), person_words):
            return True
    return False


#: A word immediately owning something via "'s": "the merchant's purse", "reed thug's
#: coin purse". Both the straight and the curly apostrophe appear in player text
#: elsewhere in this module (see ``_is_name_question``), so both are normalized before
#: matching.
_GENITIVE_OWNER = re.compile(r"\b([a-z]+)'s\b")


def _genitive_names_a_person(text: str, scope: TrustedScope) -> bool:
    """Whether an "X's" owner is a person: "the merchant's purse", "reed thug's coin
    purse". "his sword" is the same ownership relation in a different grammatical shape
    (a pronoun rather than a named owner) and lives in ``_possessive_names_a_person``.
    """
    normalized = " ".join(text.casefold().replace("’", "'").split())
    person_words = _person_words(scope)
    return any(
        match.group(1) in person_words for match in _GENITIVE_OWNER.finditer(normalized)
    )


#: Bystander nouns with a real, common, competing *non-person* sense: "queen"/"king"/
#: "knight" name a chess piece (a king or queen also names a playing card) at least as
#: often in ordinary usage as they name an actual person, so only the surrounding
#: context -- specifically, whether a following "from"/"off" clause names a chess/game
#: board (``_CHESS_CONTEXT_NOUNS`` below) -- can resolve which one a declaration means.
#: An audit traced AUD-7's and AUD-12's whole bug class to exactly these three: no other
#: bystander noun has a comparable competing sense ("guard", "merchant", "stranger",
#: "thief" name only a person, the same way "captain" names only a person or a rank and
#: not, unlike "queen", a common object in this game's own vocabulary), so only these
#: three ever need a following source clause consulted at all.
_AMBIGUOUS_PERSON_WORDS = frozenset({"king", "kings", "queen", "queens", "knight", "knights"})


#: The concrete objects that give ``_AMBIGUOUS_PERSON_WORDS`` their competing
#: non-person sense: a literal chess/game board. AUD-13 found that suppressing an
#: ambiguous word's bare-object match whenever the following source merely *fails* to
#: name a person -- rather than when it positively names the competing sense -- broke
#: the single most foundational case this whole check exists to recognize: "take the
#: guard from the tower", "steal the queen from the throne room", and "grab the
#: merchant from the crowd" are all a person taken from an ordinary place, not a chess
#: piece taken off a board, and an everyday, non-chess location does not carry the same
#: disambiguating weight a real board does. Recognizing the board itself, rather than
#: merely the absence of a recognized person, is what lets "take the queen from the
#: board" (reject -- the chess piece) and "steal the queen from the throne room"
#: (confirm -- the person) reach different answers even though "board" and "throne
#: room" are equally not people.
_CHESS_CONTEXT_NOUNS = frozenset({"board", "chessboard", "gameboard", "chess"})


#: Nouns that combine with an otherwise-unambiguous person word to name a piece of
#: equipment rather than a person: "SHIN guard", "ARM guard", "WRIST guard", "MOUTH
#: guard" name protective gear, not a person named "guard" described by a body part.
#: This is a different shape of ambiguity than ``_AMBIGUOUS_PERSON_WORDS`` above --
#: "guard" has no competing sense on its own the way "queen" does, but the specific
#: two-word combination "<body part> guard" does. Recognizing that combination is what
#: lets "the OLD guard" (a person, an ordinary adjective) and "the SHIN guard"
#: (equipment, a body-part compound) reach different answers even though both are "one
#: modifier word directly before guard". Without this, the word directly before the
#: matched noun is invisible to the from-clause deferral decision below, and "take the
#: shin guard from the locker" -- one of the audit's own original required negative
#: cases -- would wrongly confirm once an unambiguous word like "guard" stopped
#: deferring to the source clause unconditionally.
_COMPOUND_FORMING_MODIFIERS = frozenset(
    {"shin", "arm", "wrist", "mouth", "elbow", "knee", "ear", "leg", "hand", "foot", "neck"}
)


def _bare_object_is_a_person(text: str, scope: TrustedScope) -> bool:
    """Whether a theft verb's own clause reduces to a person word standing alone as its
    object, rather than modifying, or being modified by, another object noun.

    "pickpocket the stranger" and "rob the merchant" name the person directly as the
    verb's object -- that is what those two verbs mean -- rather than through a source
    clause or a possessive. This reads the clause after each theft verb in the
    declaration, bounded by ``_clause_after`` (so "take the sword AND the guard fled"
    never lets the second clause's "guard" read as though "take" still governed it, and
    "take the queen FROM the board" never lets this check's own clause run into the
    source clause's content at all), and asks ``_clause_head_word`` the same question
    the source-clause check asks: once a trailing locational phrase, adverb, or
    participle is stripped, does the clause reduce to a person word? "pickpocket the
    young stranger nearby" still does; "take the guard rail" does not, because "rail" is
    the clause's own real head noun, not trailing material. "take the guard from the
    tower" still does too, and stands on its own regardless of what "from" names, because
    "guard" is not one of the ``_AMBIGUOUS_PERSON_WORDS`` -- there is no common
    object-reading of "guard" the way there is for "queen".

    The match stands on its own -- confirming immediately, the same way "pickpocket the
    stranger" always has -- except for two narrow suppressions, each gated on a
    different thing because each exists for a different reason:

    * The word directly before the match forms an equipment compound
      (``_COMPOUND_FORMING_MODIFIERS``: "SHIN guard", "ARM guard") -- "the shin guard"
      never names a person, so this suppresses the match unconditionally, regardless of
      whether any source clause follows at all. "take the shin guard" (bare, nothing
      else in the declaration) and "take the shin guard from the locker" both reject
      for the same reason: the compound itself, not what (if anything) comes after it.
    * The matched word is one of the few genuinely ambiguous ones
      (``_AMBIGUOUS_PERSON_WORDS``: "queen", "king", "knight"), *and* a "from"/"off"
      source clause follows *and* that source clause positively names the competing
      sense (``_CHESS_CONTEXT_NOUNS``: a chess/game board) -- "take the queen from the
      board" is the chess piece, not the person. Unlike the compound check, this one
      genuinely depends on the source: with no source clause, or an ordinary one, the
      ambiguous word still confirms on its own ("take the queen" -- no board in sight --
      still reads as the person).

    Two audits, in sequence, each traced this suppression logic getting one of those two
    gates wrong. The first found an earlier version of the ambiguous-word suppression
    backwards: it fired whenever the source clause merely *failed* to name a recognized
    person, rather than when the source positively named the competing sense -- a much
    broader condition ("the throne room", "the crowd", "the cell", "the tower", and "his
    post" all equally fail to name a recognized person, but none of them is a
    chessboard) that broke the single most foundational case this function exists to
    recognize: "take the guard from the tower", "steal the queen from the throne room",
    and "grab the merchant from the crowd" all silently lost their confirmation. The
    second found the *unrelated* compound-modifier suppression incorrectly reusing that
    same "only when a source clause follows" gate, even though the compound itself is
    already sufficient reason to reject with no source clause considered at all -- so
    "take the shin guard", declared bare with nothing following it, wrongly confirmed.
    The two suppressions look similar (both key off the word immediately surrounding
    the match) but answer different questions -- "is the source what disambiguates this
    word" versus "is this word part of a compound at all" -- and needed two independent
    gates, not one shared one.
    """
    normalized = " ".join(text.casefold().replace("’", "'").split())
    tokens = _WORD_OR_CLAUSE_PUNCTUATION.findall(normalized)
    person_words = _person_words(scope)
    for index, token in enumerate(tokens):
        if token not in _THEFT_ACTIONS:
            continue
        clause = _clause_after(tokens, index + 1)
        matched = _clause_head_word(clause, person_words)
        if matched is None:
            continue
        boundary_index = index + 1 + len(clause)
        followed_by_source = (
            boundary_index < len(tokens) and tokens[boundary_index] in _SOURCE_PREPOSITIONS
        )
        matched_index = len(clause) - 1 - clause[::-1].index(matched)
        preceded_by_compound_modifier = (
            matched_index > 0 and clause[matched_index - 1] in _COMPOUND_FORMING_MODIFIERS
        )
        if preceded_by_compound_modifier:
            continue
        if followed_by_source and matched in _AMBIGUOUS_PERSON_WORDS:
            source_clause = _clause_after(tokens, boundary_index + 1)
            if any(word in _CHESS_CONTEXT_NOUNS for word in source_clause):
                continue
        return True
    return False


#: A theft verb immediately followed by a possessive pronoun: "TAKE his sword", "STEAL
#: her necklace". A possessive pronoun replaces the article in English rather than
#: joining it ("take his sword", never "take the his sword"), so it sits exactly where
#: the taken object's own determiner belongs only when it directly follows the verb.
_POSSESSIVE_OBJECT = re.compile(
    r"\b(?:"
    + "|".join(sorted(_THEFT_ACTIONS))
    + r")\s+(?:"
    + "|".join(sorted(_POSSESSIVE_PERSON_PRONOUNS))
    + r")\b"
)


def _possessive_names_a_person(text: str) -> bool:
    """Whether a possessive pronoun immediately follows the theft verb, owning the
    object the declaration takes.

    An audit found the original whole-declaration match on ``_POSSESSIVE_PERSON_PRONOUNS``
    -- kept unscoped on the reasoning that the pronouns have no competing noun sense --
    still fired on a pronoun that never possessed the taken object at all: "take the
    sword before HER arrival", "loot the chest while waiting for THEIR signal". The risk
    that check exists to close was never about lexical ambiguity; it was about whether
    the pronoun's presence reliably signals a person's relationship to the object being
    taken, the same "present anywhere" failure mode every other check in this module
    already closed. Requiring the pronoun to sit directly after the verb -- the taken
    object's own determiner position -- is what actually distinguishes ownership of the
    object from an unrelated mention of the same pronoun elsewhere in the sentence.
    """
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return bool(_POSSESSIVE_OBJECT.search(normalized))


def _person_signal(
    text: str, scope: TrustedScope, related: InteractionCue | None
) -> bool:
    """Whether a declaration names a person: a possessive pronoun immediately owning the
    taken object, the engaged interlocutor a pronoun in this declaration already points
    back at, a "from"/"off" source clause naming a person, an "X's" owner who is a
    person, or a bare person-word standing alone as the verb's object.

    An object-taking declaration ("take the copper hinge") names no one; it targets an
    unattended object, and the normal planner narrates picking it up. The same verb
    aimed at someone is a theft. Four audits, across three repair rounds, found four
    different ways an earlier version of this function got that judgment wrong: the
    first under-detected it, because ``_BYSTANDER_NOUNS`` is a closed list that can never
    enumerate every person-denoting noun a player might type ("guard", "man", "thief"
    were all absent, and are now members); the fix for that then over-detected it three
    times over -- first by treating the bare word "from" as sufficient regardless of
    what followed it; then, even after that repair scoped the source clause correctly,
    by leaving the bystander-noun and present-NPC checks matching anywhere in the whole
    declaration, so a chess piece ("queen", "king", "knight") or a piece of equipment
    ("guard rail", "shin guard") that merely shares a word with a real bystander noun
    still confirmed; and then, even after that rewrite unified every noun-shaped check
    under one structural principle, by leaving the possessive-pronoun check as the one
    exception still matching anywhere, so a pronoun that never possessed the taken
    object at all ("before HER arrival") still confirmed too, while the same rewrite's
    single-token capture and empty-residual requirements turned out to admit no
    adjective at all ("the OLD guard", "the young STRANGER" both failed).

    Every check here now shares one structural principle: a person word or pronoun only
    counts where a person-introducing construction actually places it -- the object of
    "from"/"off" (read across a bounded window, so a leading descriptive word no longer
    breaks it), the owner in an "X's" construction, the sole trailing content word once
    the verb and ordinary function words are stripped away (so a leading descriptive
    word is admitted but a trailing one still refuses the match), or the determiner
    position directly after the verb -- never merely present somewhere in the sentence.
    ``_person_words`` is the one combined vocabulary (bystander nouns and present NPC
    identifiers) every noun-shaped position checks against, so a present NPC's name and
    an ordinary bystander noun are recognized the same way in every position.
    """
    if _possessive_names_a_person(text):
        return True
    if related is not None:
        return True
    if _source_names_a_person(text, scope):
        return True
    if _genitive_names_a_person(text, scope):
        return True
    return _bare_object_is_a_person(text, scope)


def _regular_inflections(root: str) -> frozenset[str]:
    """Every regularly spelled inflection of one violent-action root: the plural/
    third-person form, and both the doubled- and undoubled-consonant spelling of the
    past tense and present participle.

    English doubles a short root's final consonant before "-ed"/"-ing" only when the
    last syllable is stressed ("stab" -> "stabbing", "club" -> "clubbing"), and does not
    for a longer root stressed earlier ("murder" -> "murdering", not "murderring").
    Telling those apart needs syllable stress, which a three-letter trigram check cannot
    recover, so this generates both spellings whenever the trigram allows doubling at
    all, rather than guessing which single one the specific root wants. The unspelled
    extra form (e.g. "murderring") matches no real English word and so creates no new
    false positive; the real form it might otherwise have missed is what matters.

    Every member is matched later as a complete word, never as a prefix or suffix of
    something longer. That whole-word boundary -- not a hand-picked exception list -- is
    what keeps "skill", "harmony", "kneeling", "heartbeat", "sidekick", and "nightclub"
    out while still admitting "stabbing" and "attacking": none of those false positives
    is itself the exact spelling of a generated inflection, even though several contain
    a root as a literal substring.
    """
    forms = {root}
    forms.add(root + "es" if root.endswith(("s", "x", "z", "ch", "sh")) else root + "s")
    if root.endswith("ee"):
        forms.add(root + "d")
        forms.add(root + "ing")
    elif root.endswith("e"):
        forms.add(root[:-1] + "ed")
        forms.add(root[:-1] + "ing")
    else:
        forms.add(root + "ed")
        forms.add(root + "ing")
        if (
            len(root) >= 3
            and root[-1] not in _VOWELS
            and root[-2] in _VOWELS
            and root[-3] not in _VOWELS
        ):
            doubled = root + root[-1]
            forms.add(doubled + "ed")
            forms.add(doubled + "ing")
    return frozenset(forms)


#: Every regularly inflected surface form of every violent-action root, precomputed
#: once. An audit found the previous prefix/suffix substring approach here misclassified
#: ordinary English words -- "skill" (contains "kill" as a suffix), "stable" ("stab" as
#: a prefix), "upbeat" and "heartbeat" ("beat" as a suffix), "harmony"/"harmless" ("harm"
#: as a prefix), "kneeling" ("knee" as a prefix, but of the unrelated verb "kneel"),
#: "sidekick"/"nightclub"/"clubhouse"/"roadkill"/"overkill"/"kickstart"/"kickoff" (all
#: ordinary compounds) -- because bare substring containment cannot tell a genuine
#: inflection or compound of a violent verb from an unrelated word that merely contains
#: the same letters. Exact membership in this generated, closed set of real inflected
#: spellings has no such ambiguity: every member is verified against a 104,000-word
#: system dictionary to carry no unintended entry beyond the roots and their own
#: inflections. The tradeoff this accepts: a violent verb compounded with an unrelated
#: word ("backstab") is no longer recognized, because no rule distinguishes that shape
#: from "heartbeat" or "sidekick" without either a violence-specific compound allowlist
#: or a safe-word denylist, and both of those are the same closed-list trap this fix
#: replaces, just moved one level down.
_VIOLENT_INFLECTIONS: frozenset[str] = frozenset().union(
    *(_regular_inflections(root) for root in _VIOLENT_ACTIONS)
)


def _violent_words(words: frozenset[str]) -> bool:
    """Whether any token is a violent-action word or one of its regular inflections."""
    return bool(words & _VIOLENT_INFLECTIONS)


#: Contracted and bare forms of a negation directly modifying a verb. Matched against
#: the whole contraction with its apostrophe intact, rather than against the same
#: letters-only token stream every other check in this module uses, because splitting
#: on the apostrophe the way ``_WORDS`` does turns "won't" into the fragments "won" and
#: "t" -- and "won" is also the past tense of "win", so treating the bare fragment as a
#: negation marker would misread "I won the arm wrestle, so I attack next" as denied.
#: Every entry here is either a plain adverb ("not", "never") or a full contraction with
#: no such collision.
_NEGATION_MARKERS = frozenset({
    "not", "never", "no", "cannot",
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "wouldn't", "couldn't", "shouldn't",
    "haven't", "hasn't", "hadn't",
})

_WORD_OR_CLAUSE_PUNCTUATION_WITH_APOSTROPHE = re.compile(r"[a-z']+|[,.;:!?]")


def _clause_before(tokens: list[str], index: int) -> list[str]:
    """Every token back to ``tokens[index]``'s own clause start: a comma or
    sentence-ending mark, a coordinating conjunction, or another clause-defining verb.
    Mirrors ``_clause_after``'s forward scan in the opposite direction, for the same
    reason: a clause's own start is grammatical, not a fixed token count back.
    """
    clause: list[str] = []
    for token in reversed(tokens[:index]):
        if (
            token in _CLAUSE_PUNCTUATION
            or token in _CLAUSE_CONJUNCTIONS
            or token in _CLAUSE_VERB_BOUNDARIES
        ):
            break
        clause.append(token)
    return clause


def _negates_violence(text: str) -> bool:
    """Whether every violent-action word the text carries is itself denied by a
    negation in its own clause.

    A clause that carries a violent word without a negation in the same clause still
    counts as violent: \"I said I wouldn't, then I stab him anyway\" keeps its hazard,
    because \"wouldn't\" sits in the first clause and \"then\" starts a new one before
    \"stab\" ever appears. This can only ever remove a hazard the words themselves
    supplied, never add one, so an actually-declared attack is never let through
    unconfirmed by this check.
    """
    tokens = _WORD_OR_CLAUSE_PUNCTUATION_WITH_APOSTROPHE.findall(
        text.casefold().replace("’", "'")
    )
    violent_indices = [
        index for index, token in enumerate(tokens) if token in _VIOLENT_INFLECTIONS
    ]
    if not violent_indices:
        return False
    return all(
        any(marker in _NEGATION_MARKERS for marker in _clause_before(tokens, index))
        for index in violent_indices
    )


#: Balanced double-quoted spans, straight or curly. Single quotes are deliberately
#: excluded: apostrophes make them unmatchable in ordinary prose.
_QUOTED_SPEECH = re.compile(r'"[^"]*"|“[^”]*”')


def _spoken_spans_removed(text: str) -> str:
    """The declaration with its quoted speech removed.
    """
    return _QUOTED_SPEECH.sub(" ", text)


#: Auxiliaries a passive construction hangs a participle from. "get" forms count:
#: "I got stabbed" reports suffering violence, not committing it.
_PASSIVE_AUXILIARIES = frozenset(
    {"am", "are", "be", "been", "being", "get", "gets", "getting", "got", "is", "was", "were"}
)

#: Irregular past participles of the violent-action list, which a suffix test misses.
_IRREGULAR_VIOLENT_PARTICIPLES = frozenset(
    {"beaten", "bitten", "cut", "hit", "hurt", "shot", "slain", "struck"}
)


def _passive_violence(text: str) -> bool:
    """Whether every violent word the text carries reports violence *suffered*.

    "I'm being attacked by Rade" declares no attack: the speaker is its object.
    The test is deliberately narrow -- a passive auxiliary immediately before a
    past-participle form -- and every violent token must match, so "I was attacked,
    now I stab him back" keeps its hazard through the active "stab". Like
    ``_negates_violence`` beside it, this can only remove a hazard the words
    themselves supplied, never add one.
    """
    tokens = _WORD_OR_CLAUSE_PUNCTUATION_WITH_APOSTROPHE.findall(
        text.casefold().replace("’", "'")
    )
    violent_indices = [
        index for index, token in enumerate(tokens) if token in _VIOLENT_INFLECTIONS
    ]
    if not violent_indices:
        return False

    def is_passive(index: int) -> bool:
        token = tokens[index]
        participle = token.endswith("ed") or token in _IRREGULAR_VIOLENT_PARTICIPLES
        return (
            participle and index > 0 and tokens[index - 1] in _PASSIVE_AUXILIARIES
        )

    return all(is_passive(index) for index in violent_indices)


def _violence_precedes_the_taking(text: str) -> bool:
    """Whether a violent word appears before the first theft verb in the declaration.

    See the fall-through comment in ``_hazard`` for why position decides: the
    declaration's leading act is what the player is actually doing, and every
    audited take-idiom carries its ambiguous word after the "take"."""
    tokens = re.findall(r"[a-z]+", text.casefold())
    first_theft = next(
        (index for index, token in enumerate(tokens) if token in _THEFT_ACTIONS), None
    )
    if first_theft is None:
        return False
    return any(token in _VIOLENT_INFLECTIONS for token in tokens[:first_theft])


def _hazard(
    words: frozenset[str],
    text: str,
    question: bool,
    scope: TrustedScope,
    related: InteractionCue | None,
) -> Literal["theft", "violence", "destructive", "none"]:
    """Classify committed verbs while keeping hypothetical questions readable."""
    # Judge the act, not the speech: quoted words are said, not done.
    acted = _spoken_spans_removed(text)
    if acted != text:
        text = acted
        words = _word_set(acted)
    if question and not _committed_question(text, _THEFT_ACTIONS | _VIOLENT_ACTIONS | _DESTRUCTIVE_ACTIONS):
        return "none"
    # The theft branch's person-signal miss returns "none" for the audited
    # take-idiom family ("take a break", "take the elbow guard") -- but that early
    # return must not swallow a declaration whose PRIMARY act is violent: a live
    # turn declared "I punch the trader in the face and take the rope", the miss
    # ("take" steals from no one; the trader is punched) returned "none", and a
    # first-strike assault on a bystander reached the ordinary planner with no
    # confirmation at all. The fall-through fires only when a violent word comes
    # BEFORE the theft verb -- the declaration leads with its main act -- because
    # the audited idioms all carry their ambiguous word AFTER "take" ("take the
    # elbow guard", "take the sword, then the guard attacked", "take a break"),
    # where "elbow" is equipment, "attacked" is a report, and "break" is a rest.
    # The reversed compound ("take the rope and punch the trader") is a disclosed
    # residual of this ordering rule, chosen over re-flagging every audited idiom.
    if words & _THEFT_ACTIONS:
        if _person_signal(text, scope, related):
            return "theft"
        if not _violence_precedes_the_taking(text):
            return "none"
    if _violent_words(words):
        return "none" if _negates_violence(text) or _passive_violence(text) else "violence"
    if words & _DESTRUCTIVE_ACTIONS:
        return "destructive"
    return "none"


def _cue_for_named_target(words: frozenset[str], scope: TrustedScope) -> InteractionCue | None:
    for npc_id in scope.present_npc_ids:
        target_words = frozenset(_WORDS.findall(npc_id.replace("-", " ")))
        if target_words and target_words <= words:
            return InteractionCue("canonical_npc", npc_id)
    return None


def _is_name_question(text: str) -> bool:
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return normalized in {"whats your name", "what's your name", "what is your name"} or (
        normalized.startswith("whats your name")
        or normalized.startswith("what's your name")
        or normalized.startswith("what is your name")
    )


def _is_ooc(words: frozenset[str], text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return bool(words & _OOC_MARKERS) or normalized.startswith(("out of character", "game master"))


def _is_ooc_question(text: str, is_rules_inquiry: bool) -> bool:
    """Whether the player stepped out of the fiction to ask something.

    Two signals must both hold, and requiring both is what keeps every existing precedence
    edge where it was.

    The turn must read as a question. An out-of-character *declaration* -- \"ooc I stab the
    Choir Listener\" -- is still a declaration, and the hazard branch keeps it.

    The out-of-character marker must open the turn, in the first sentence, or the social
    classifier must already read the whole turn as a rules inquiry. A marker that surfaces
    only after the declared act -- \"I stab the guard. gm?\" -- leaves the act first and the
    hazard branch keeps it, so a trailing marker cannot lower the risk floor.
    """
    if not _is_question(text):
        return False
    if is_rules_inquiry:
        return True
    opening = _CLAUSE_BOUNDARY.split(" ".join(text.casefold().split()), 1)[0]
    return _is_ooc(_word_set(opening), opening)


#: Vocabulary a question about the mechanical state of play carries. Deliberately
#: short: only words that name the combat bookkeeping itself, never scene content.
_MECHANICS_STATE_WORDS = frozenset(
    {"action", "actions", "initiative", "round", "rounds", "turn", "turns"}
)

#: How such a question opens when the player omits the question mark.
_MECHANICS_QUESTION_OPENERS = (
    "am i", "do i", "how many", "how much", "is it", "what", "when", "which",
    "who", "who's", "whos", "whose",
)


def _is_mechanics_state_question(words: frozenset[str], text: str) -> bool:
    """Whether this turn asks about turn order, rounds, or actions rather than acting.

    Both signals must hold: mechanics vocabulary, and an interrogative shape (a
    question mark, or an opening interrogative word). Any committed, violent,
    thieving, or destructive word withholds it, so "I take my turn and stab him"
    can never route here.
    """
    if not words & _MECHANICS_STATE_WORDS:
        return False
    if (
        words & _COMMITTED_ACTIONS
        or _violent_words(words)
        or words & _THEFT_ACTIONS
        or words & _DESTRUCTIVE_ACTIONS
    ):
        return False
    if _is_question(text):
        return True
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return normalized.startswith(_MECHANICS_QUESTION_OPENERS)


def _is_low_impact_social(words: frozenset[str], text: str, active: InteractionCue | None) -> bool:
    if words & _CONSEQUENTIAL_SOCIAL_WORDS:
        return False
    if _is_name_question(text):
        return active is not None
    if _is_question(text):
        return active is not None
    return bool(words & _SOCIAL_WORDS)


def _is_read_only(words: frozenset[str], text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return _is_question(text) or normalized.startswith(_OBSERVATION_PREFIXES) or normalized.startswith(
        tuple(f"i {word}" for word in _OBSERVATION_PREFIXES)
    )


def _departure(words: frozenset[str], text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return bool(words & _DEPARTURE_WORDS) or "walk away" in normalized


def classify_turn(
    player_text: str,
    *,
    scope: TrustedScope,
    active_focus: InteractionCue | None = None,
    awaiting_reply: bool = False,
) -> TurnPolicy:
    """Apply the fixed risk, OOC, social, read, then planner precedence.

    ``awaiting_reply`` says the previous delivered narration ended by asking the player
    something. It changes only the final fallback, never an earlier route, so a hazard,
    a rules inquiry, a committed action, and a read all classify as they did before.
    """
    social = classify_social_input(player_text)
    words = _word_set(player_text)
    question = _is_question(player_text)
    named = _cue_for_named_target(words, scope)
    related = active_focus if active_focus and (question or bool(words & {"you", "your", "them", "their"})) else None


    hazard = _hazard(words, player_text, question, scope, related)
    if hazard != "none" and _is_ooc_question(player_text, social.is_rules_inquiry):
        hazard = "none"
    if hazard != "none":
        return TurnPolicy("risk", hazard, related, None, scope, related is not None, True)


    if hazard == "none" and _is_mechanics_state_question(words, player_text):
        return TurnPolicy("out_of_character", "none", None, None, scope, False, False)
    if social.is_rules_inquiry:
        return TurnPolicy(
            "out_of_character", "none", None, None, scope, False, False,
            social.mode.value, "", False, "",
        )
    if social.category is not None:
        cue = named or active_focus
        if cue is None and social.mode.value != "ooc_action_request":
            cue = InteractionCue("scene_interlocutor")
        return TurnPolicy(
            "social", "none", cue, cue, scope, cue is not None, _departure(words, player_text),
            social.mode.value, social.category.value, social.requires_test,
            social.trade_phase.value if social.trade_phase else "", social.romance_escalation,
            social.romance_coercive,
        )
    if _is_ooc(words, player_text):
        return TurnPolicy("out_of_character", "none", None, None, scope, False, False)
    if words & _COMMITTED_ACTIONS:
        return TurnPolicy("planner", "none", related, None, scope, related is not None, related is None)
    if _is_low_impact_social(words, player_text, active_focus):
        cue = named or active_focus
        if cue is None and not question:
            cue = InteractionCue("scene_interlocutor")
        return TurnPolicy(
            "social", "none", cue, cue, scope, cue is not None, _departure(words, player_text)
        )
    if _is_read_only(words, player_text):
        return TurnPolicy("read", "none", related, None, scope, related is not None, False)
    # An answer to the interlocutor's own question lands here and nowhere earlier. It
    # names no target, carries no speech verb, and asks nothing, so every route above
    # declines it and the planner inherits a turn that is plainly conversation.
    #
    # The invitation alone cannot take the turn, and an audit established why. A player
    # may answer the question or ignore it and act, and both land here, so state without
    # ``reply_shaped`` routed "I pick the lock" as conversation with the interlocutor.
    if (
        awaiting_reply
        and active_focus is not None
        and reply_shaped(player_text, scope.party_name_tokens)
    ):
        return TurnPolicy(
            "social", "none", active_focus, active_focus, scope, True,
            _departure(words, player_text),
        )


    return TurnPolicy(
        "planner", "none", related, None, scope, related is not None, _departure(words, player_text)
    )


# Moved from ``narrator.social`` when the trade lane stopped re-reading the
# declaration. ``TurnClassification.offered_price`` replaced it in production:
# this read English number words and a ``[0-9]{1,4} copper`` regex, so "douze
# cuivres" named no price at all. Kept here so the offline double still reproduces
# what the retired reader saw.
_AMOUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
def counteroffer_price(text: str) -> int | None:
    """Read a bounded copper amount only from an already active offer."""
    words = _words(text)
    for word in words:
        if word in _AMOUNTS:
            return _AMOUNTS[word]
    match = re.search(r"\b([0-9]{1,4})\s+copper\b", text.casefold())
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# The social half of the retired classifier, moved out of ``narrator.social`` when
# the trade and romance lanes stopped re-reading the declaration. Production now
# reads ``TurnClassification``'s social_category, trade_phase, accepts_offer,
# offered_price, romance_escalation and romance_coercive instead, so none of the
# English vocabulary below reaches a player any more.
# ---------------------------------------------------------------------------

def _words(text: str) -> frozenset[str]:
    return frozenset(_WORDS.findall(text.casefold()))

SOCIAL_CATEGORY_WORDS: Mapping[SocialCategory, frozenset[str]] = {
    SocialCategory.CASUAL: frozenset({"hello", "hi", "greet", "talk", "speak", "chat"}),
    SocialCategory.NEGOTIATION: frozenset({"bargain", "negotiate", "offer", "counteroffer", "deal", "agreed", "price", "buy", "sell", "trade", "pay"}),
    SocialCategory.PERSUASION: frozenset({"persuade", "convince", "appeal", "plead"}),
    SocialCategory.DECEPTION: frozenset({"deceive", "deceiving", "deception", "lie", "mislead", "bluff"}),
    SocialCategory.LEADERSHIP: frozenset({"lead", "rally", "command", "direct"}),
    SocialCategory.DEBATE: frozenset({"debate", "argue", "rebut", "prove"}),
    SocialCategory.INTIMIDATION: frozenset({"intimidate", "threaten", "menace", "frighten"}),
    SocialCategory.ROMANCE: frozenset({"romance", "flirt", "flirting", "kiss", "date", "intimate", "seduce", "seducing"}),
    SocialCategory.BRIBERY: frozenset({"bribe", "bribery", "payoff"}),
    SocialCategory.PERFORMANCE: frozenset({"perform", "sing", "dance", "recite", "play"}),
    SocialCategory.INFORMATION: frozenset({"ask", "question", "inquire", "information", "rumor"}),
    SocialCategory.INSIGHT: frozenset({"insight", "motive", "sense"}),
    SocialCategory.ETIQUETTE: frozenset({"etiquette", "courtesy", "protocol", "bow"}),
    SocialCategory.PROMISE: frozenset({"promise", "swear", "disclose", "reveal", "secret"}),
}

_OOC_PREFIX = re.compile(r"\booc\s*:\s*", re.IGNORECASE)

class SocialInputMode(str, Enum):  # noqa: UP042 -- matches the production enums it doubles for
    IC_SPEECH = "ic_speech"
    CONCISE_INTENT = "concise_intent"
    OOC_RULES_INQUIRY = "ooc_rules_inquiry"
    OOC_ACTION_REQUEST = "ooc_action_request"
    MIXED = "mixed"

_ROMANCE_ESCALATION_WORDS = frozenset(
    {
        "kiss", "intimate", "seduce", "seducing", "undress", "undressing",
        "naked", "nude", "sex", "sexual",
    }
)


_ROMANCE_ESCALATION_PHRASES = (
    re.compile(r"\bmake\s+love\b"),
    re.compile(r"\bclothes\s+off\b"),
    re.compile(r"\btake\s+off\s+(?:your|her|his|their|my|the)\s+clothes\b"),
)

def _has_romance_escalation_phrase(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(pattern.search(normalized) for pattern in _ROMANCE_ESCALATION_PHRASES)

_ROMANCE_COERCION_WORDS = frozenset({"threaten", "force", "coerce", "blackmail", "drug"})

@dataclass(frozen=True)
class SocialIntent:
    """A normalized classification with no copied player dialogue."""

    mode: SocialInputMode
    category: SocialCategory | None
    requires_test: bool = False
    is_rules_inquiry: bool = False
    trade_phase: TradePhase | None = None
    romance_escalation: bool = False
    romance_coercive: bool = False

def _split_ooc(text: str) -> tuple[str, str]:
    match = _OOC_PREFIX.search(text)
    if match is None:
        return text.strip(), ""
    return text[:match.start()].strip(), text[match.end():].strip()

def _category(words: frozenset[str], text: str = "") -> SocialCategory | None:
    if (
        words & (SOCIAL_CATEGORY_WORDS[SocialCategory.ROMANCE] | _ROMANCE_ESCALATION_WORDS)
        or _has_romance_escalation_phrase(text)
    ):
        return SocialCategory.ROMANCE
    for category, triggers in SOCIAL_CATEGORY_WORDS.items():
        if words & triggers:
            return category
    return None

def _trade_phase(
    words: frozenset[str], text: str, *, negotiating: bool = False
) -> TradePhase | None:
    """Read one trade stage. ``negotiating`` says a live frame is advancing to a purchase.

    \"Take\" does mean a purchase inside a negotiation that has already agreed terms, which
    is the one context that supplies the missing fact. ``negotiating`` carries that fact,
    and only a caller holding the channel's live frame -- ``SocialState.advance_trade``
    below, and ``NarratorService`` at its trade routing gate -- ever sets it. The
    context-free caller leaves it false, so classification alone can no longer manufacture
    a purchase out of an ordinary verb.
    """
    normalized = " ".join(text.casefold().split())
    if "buy" in words or (negotiating and "take" in words) or ("pay" in words and "now" in words):
        return TradePhase.PURCHASE_INTENT
    if words & {"deal", "agreed", "agree"}:
        return TradePhase.AGREEMENT
    if "would pay" in normalized or "could offer" in normalized or "how about" in normalized:
        return TradePhase.COUNTEROFFER
    if "copper" in words and (any(word in _AMOUNTS for word in words) or re.search(r"\b[0-9]{1,4}\b", normalized)):
        return TradePhase.COUNTEROFFER
    if words & {"offer", "counteroffer", "bargain", "negotiate"}:
        return TradePhase.COUNTEROFFER
    if words & {"price", "cost", "inventory", "sell"}:
        return TradePhase.INQUIRY
    return None

def classify_social_input(text: str, *, negotiating: bool = False) -> SocialIntent:
    """Classify semantics without treating quotation marks as a control channel.

    ``negotiating`` is the caller's trusted statement that this channel already holds a
    trade frame advancing toward a purchase; ``_trade_phase`` above owns what it changes.
    It never originates in player text, and it reaches no other decision here.
    """
    ic, ooc = _split_ooc(text)
    ooc_words = _words(ooc)
    body_words = _words(ic or text)
    all_words = body_words | ooc_words
    body_category = _category(body_words, ic or text)
    if ooc:
        rules = bool(ooc_words & {"how", "rule", "rules", "work", "works", "mechanic", "mechanics"})
        action = _category(ooc_words, ooc)
        if rules and action is None:
            return SocialIntent(SocialInputMode.OOC_RULES_INQUIRY, None, is_rules_inquiry=True)
        category = action or body_category
        if category is SocialCategory.CASUAL:
            return SocialIntent(SocialInputMode.OOC_ACTION_REQUEST, None)
        escalation = bool(all_words & _ROMANCE_ESCALATION_WORDS) or _has_romance_escalation_phrase(ooc)
        return SocialIntent(
            SocialInputMode.MIXED if ic else SocialInputMode.OOC_ACTION_REQUEST,
            category,
            requires_test=category in _MECHANICAL_SOCIAL_CATEGORIES,
            trade_phase=_trade_phase(ooc_words, ooc, negotiating=negotiating),
            romance_escalation=escalation,
            romance_coercive=escalation and bool(all_words & _ROMANCE_COERCION_WORDS),
        )
    phase = _trade_phase(body_words, text, negotiating=negotiating)
    category = body_category or (SocialCategory.NEGOTIATION if phase is not None else None)
    concise = bool(body_words & {"i", "we"}) and category is not None
    requires = category in _MECHANICAL_SOCIAL_CATEGORIES and concise
    escalation = bool(body_words & _ROMANCE_ESCALATION_WORDS) or _has_romance_escalation_phrase(text)
    return SocialIntent(
        SocialInputMode.CONCISE_INTENT if concise else SocialInputMode.IC_SPEECH,
        category,
        requires_test=requires,
        trade_phase=phase,
        romance_escalation=escalation,
        romance_coercive=escalation and bool(body_words & _ROMANCE_COERCION_WORDS),
    )

def social_test_needed(
    intent: SocialIntent,
    *,
    feasible: bool,
    uncertain: bool,
    opposed: bool,
    meaningful_branches: bool,
    roleplay_resolved: bool = False,
) -> bool:
    """Apply the threshold without grading performance quality."""
    if intent.is_rules_inquiry or intent.category is None:
        return False
    if intent.requires_test:
        return feasible and meaningful_branches
    return feasible and uncertain and opposed and meaningful_branches and not roleplay_resolved
