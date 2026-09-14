"""Interactions risk lexicon. Synthetic fixtures exercise this contract."""

from __future__ import annotations

import pytest
from fake_classifier import policy_for
from lexical_double import classify_turn

from narrator.interactions import combat_sanctions_violence
from narrator.policy_types import CombatSnapshot, TrustedScope, turn_framing_for

_NO_ONE_PRESENT = TrustedScope("session-1", "cellar", ())
_NPC_PRESENT = TrustedScope("session-1", "market", ("reed-thug",))


@pytest.mark.parametrize(
    "declaration",
    (
        "take the copper hinge",
        'I take the copper hinge',
        "grab the rusty key",
        "I lift the fallen banner",
        "loot the chest",
    ),
)
def test_object_taking_with_no_person_signal_is_never_theft(declaration):
    """The original assertion was ``policy.route != \"risk\"``, which every case satisfied but
    two of them satisfied for the wrong reason. \"take the copper hinge\" and \"I take the
    lamp bracket\" left the risk floor correctly and were then claimed by
    ``narrator.social._trade_phase``, which read any text containing \"take\" as
    ``PURCHASE_INTENT``; ``classify_social_input`` promoted that to a negotiation
    category, and both declarations routed to ``social`` on a channel with no merchant,
    no trade, and no NPC present at all. A weaker assertion concealed it: not-``risk``
    was true of the misroute too.

    Each case now states the route it actually takes. ``take``, ``lift``, and ``loot`` are
    unattended-object verbs; with no person signal the ordinary planner narrates picking
    the thing up, so ``planner`` is the answer for all five. The theft assertion this test
    was written for is unchanged and still first.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.risk_category != "theft"
    assert policy.route == "planner"
    # The trade classifier claims none of them, which is the misroute above, stated
    # directly rather than inferred from the route.
    assert policy.trade_phase == ""


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the merchant",
        "steal the merchant's purse",
        "pickpocket the stranger",
    ),
)
def test_object_taking_with_a_bystander_noun_stays_theft(declaration):
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


def test_object_taking_from_a_present_npc_stays_theft():
    policy = classify_turn("take reed thug's coin purse", scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


def test_object_taking_from_the_engaged_interlocutor_stays_theft():
    """A pronoun pointing back at an already-engaged interlocutor is a person signal too."""
    scope = TrustedScope("session-1", "market", ())
    active = classify_turn("approach the traders and say hello", scope=scope).focus_candidate
    assert active is not None
    policy = classify_turn("steal their rope", scope=scope, active_focus=active)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the purse from the guard",
        "steal it from the guard",
        "take it from the man in the corner",
        "steal the coin from the thief",
        "take his sword",
        "steal her necklace",
    ),
)
def test_aud_2_person_nouns_and_possessives_outside_the_allowlist_still_confirm(declaration):
    """Test aud 2 person nouns and possessives outside the allowlist still confirm.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the key from the drawer",
        "take the coin from the chest",
        "grab the sword from the rack",
        "take the potion from the shelf",
        "loot the gold from the crate",
    ),
)
def test_aud_6_a_from_clause_naming_a_container_is_never_a_person_signal(declaration):
    """AUD-6: the second repair round's own "from" signal (added to resolve AUD-2)
    treated the bare preposition as sufficient regardless of what followed it, so an
    ordinary "take X from <container>" declaration -- naming no person anywhere --
    spuriously confirmed too. The auditor reproduced every one of these directly against
    the first repair round's candidate. The fix requires the word "from" actually
    introduces to itself be a recognized person word (a bystander noun, an object
    pronoun, or a present NPC's identifier), not the bare preposition alone.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the purse from the guard",
        "steal it from the guard",
        "take it from the man in the corner",
        "steal the coin from the thief",
        "take the coin from him",
        "steal the ring from her",
    ),
)
def test_aud_6_a_from_clause_naming_a_person_still_confirms(declaration):
    """The AUD-6 repair narrows the "from" signal to a positive person match; these
    are the AUD-2 cases (plus the object-pronoun shape "from him"/"from her") that
    signal must keep catching, with no NPC scoped present and no bystander-noun word
    elsewhere in the declaration.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


def test_aud_6_a_from_clause_naming_a_present_npc_still_confirms():
    policy = classify_turn("take the pouch from reed thug", scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the queen from the board",
        "take the king from the chessboard",
        "take the knight from the board",
        "steal the queen off the chessboard",
        "take the guard rail",
        "take the shin guard from the locker",
    ),
)
def test_aud_7_a_bystander_word_used_as_a_chess_piece_or_equipment_is_never_theft(declaration):
    """AUD-7: expanding ``_BYSTANDER_NOUNS`` for AUD-2/6 restored real person nouns
    ("guard", "man", "thief", ...) but the bystander-noun check still matched anywhere
    in the declaration, unscoped -- the same shape of bug ``_source_names_a_person`` was
    already built to avoid for the word after "from". A chess/game piece ("queen",
    "king", "knight") or a piece of protective/structural equipment named after a person
    word ("guard rail", "shin guard") names no person at all; the auditor reproduced
    every one of these directly against the second repair round's candidate.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the arm guard from the chest",
        "take the elbow guard from the rack",
        "take the wrist guard from the shelf",
        "take the mouth guard from the bag",
        "take the queen statue from the shelf",
        "take the king piece from the box",
    ),
)
def test_aud_7_own_adversarial_pass_more_equipment_and_titled_object_compounds(declaration):
    """A further adversarial pass beyond the auditor's own six examples, covering
    equipment named after a body part beside a person word ("arm guard", "elbow guard",
    "wrist guard", "mouth guard") and a titled object ("queen statue", "king piece")
    naming a collectible or a decoration, not a person. None of these should reach a
    theft confirmation with no NPC scoped present.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "grab the captain's chest",
        "loot the king's tomb",
        "take the queen's crown",
    ),
)
def test_aud_7_a_title_as_a_genitive_owner_still_confirms(declaration):
    """A title word ("captain", "king", "queen") in the *owner* position of an "X's Y"
    construction still names a person and must still confirm -- the auditor's own result
    explicitly called this shape (same as the already-accepted "the merchant's purse")
    defensible, not a bug, distinct from the same words appearing as the bare object of
    the theft verb itself ("take the queen") or after "from"/"off" naming a chess board
    or other non-person source.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the old guard",
        "steal the crown from the young queen",
        "take it from the wounded guard",
        "steal the ring from the sleeping merchant",
        "take the wounded guard",
        "grab the old man",
        "pickpocket the young stranger",
    ),
)
def test_aud_8_an_adjective_before_the_person_word_still_confirms(declaration):
    """AUD-8: a single descriptive word between the article and the person noun --
    naming its age, condition, or disposition -- must not defeat detection, in either
    the source-clause ("from the old guard") or the bare-object ("the wounded guard",
    "pickpocket the young stranger") shape. The auditor called out "pickpocket the young
    stranger" as the most severe case: "pickpocket" is a verb whose entire meaning
    requires a person target, and the unqualified form ("pickpocket the stranger")
    already confirmed correctly before this repair -- one adjective should never have
    been able to break it.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "I take the potion, thinking about her",
        "grab the key, then look for her",
        "take the sword before her arrival",
        "loot the chest while waiting for their signal",
    ),
)
def test_aud_9_an_incidental_pronoun_never_confirms(declaration):
    """AUD-9: "her"/"their" appearing somewhere in the declaration without possessing
    the taken object must never confirm -- the same "present anywhere in the sentence"
    failure mode AUD-7 already closed for nouns, reproduced here for the one check the
    AUD-7 rewrite had left deliberately unscoped. None of these four names the potion,
    key, sword, or chest as belonging to the referenced person; the pronoun's own
    ordinary use (a thought, a search, an arrival, a signal) is unrelated to the object
    being taken.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    ("take his sword", "steal her necklace", "grab their purse"),
)
def test_aud_9_a_possessive_pronoun_immediately_after_the_verb_still_confirms(declaration):
    """A possessive pronoun sitting exactly where the taken object's own determiner
    belongs -- directly after the theft verb -- still confirms; this is the AUD-2
    fixture shape the AUD-9 scoping must not regress.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the potion from the old wooden guard station",
        "take the old wooden guard station",
        "take the sword, her idea was clever",
        "grab the potion near her, quietly",
    ),
)
def test_aud_8_and_9_own_adversarial_pass_stays_negative(declaration):
    """A further self-directed adversarial pass beyond the auditor's own cases: a
    compound noun phrase whose own head noun is not a person word at all ("the old
    wooden guard station" -- "station", not "guard", is what the clause actually
    reduces to once its content is read to the end, regardless of how many words
    precede it), and a possessive pronoun that sits near the verb by coincidence but
    still does not immediately follow it. None of these should reach a theft
    confirmation with no NPC scoped present.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the tall old bearded guard standing there",
        "take the coin purse from the tall old bearded guard nearby",
    ),
)
def test_aud_10_two_or_more_adjectives_still_confirm(declaration):
    """AUD-10: the AUD-8 repair's fixed-width token window covered one descriptive word
    but not two -- "guard" sat one position past the window in both of the auditor's
    exact cases, and the bare-object fallback did not rescue either one because the
    trailing phrase ("standing there"/"nearby") pushed the sentence's actual last
    content word past "guard" too. The repair replaces the token count with a clause
    boundary (a comma, sentence-ending punctuation, a conjunction, or another verb) and
    reads the whole bounded clause, which is correct for an arbitrary run of descriptive
    words rather than merely a wider fixed number.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the very tall old bearded guard",
        "take the sword from the very tall old grizzled bearded guard",
        "pickpocket the very tall old grizzled bearded stranger",
        "take the very tall old grizzled bearded guard standing there",
    ),
)
def test_aud_10_own_adversarial_pass_three_and_four_adjectives_still_confirm(declaration):
    """Pushed further than the auditor's own two-adjective cases: three and four
    descriptive words before the person noun, in both the source-clause and bare-object
    positions, combined with a trailing participial phrase in the last case. A fix that
    is genuinely clause-scoped rather than count-scoped handles an arbitrary run of
    these for free; if any of these failed, the repair would not actually be
    clause-scoped yet.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword and the guard fled",
        "take the sword, then the guard attacked",
        "take the sword but the guard noticed",
        "take the sword while the guard slept",
        "take the sword from the chest and the guard shouted",
        "take the sword from the old chest, near the guard",
    ),
)
def test_aud_10_a_genuinely_separate_clause_never_extends_the_match(declaration):
    """The clause-boundary repair must never reach *past* a coordinating conjunction, a
    comma, or another verb into a genuinely separate clause: "take the sword and the
    guard fled" must not suddenly start matching "guard" as the object just because the
    scan is no longer bounded by a fixed token count. Every one of these names a person
    only in a clause the taken object's own clause does not extend into.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the pouch from the tall guard standing watch",
        "take the sword from the guard standing watch",
    ),
)
def test_aud_11_a_regression_a_trailing_word_outside_the_three_shapes_still_confirms(declaration):
    """AUD-11 regression: these two correctly confirmed in the immediately preceding
    candidate. The round-5 trailing-strip loop popped one word at a time from the end
    and gave up the instant it hit a word outside its three recognized shapes; "watch"
    (the clause's actual last word) is a plain noun, not a locational adverb, relative
    signal, or participle, so the scan never even reached "standing" (which would have
    stripped fine) or "guard". The repair reads the whole tail after the clause's own
    rightmost person word at once, rather than requiring every word back from the end to
    individually strip in sequence, so an unrecognized word like "watch" no longer
    aborts the match when a recognized signal ("standing") sits elsewhere in the same
    tail. The control case "take the pouch from the guard standing there" -- where the
    tail's own last word already falls in a recognized shape -- correctly confirmed
    before this repair and must keep confirming after it.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the guard who arrived late",
        "take the guard who arrived late",
        "take the sword from the guard sent to watch",
        "steal the coin from the merchant who just arrived",
    ),
)
def test_aud_11_a_trailing_relative_clause_or_infinitive_still_confirms(declaration):
    """The same root cause the "standing watch" regression traces to -- a trailing word
    outside the three originally recognized shapes aborting the whole match -- also
    blocked a trailing relative clause ("who arrived late") or infinitive ("sent to
    watch") describing the person. Recognizing a relative pronoun ("who"/"that"/
    "which") or an infinitive "to" as a signal that the material following it still
    describes the person named earlier in the clause closes this the same way the
    regression closes, without reintroducing the "guard rail"/"guard station" false
    positives: neither of those tails carries any such signal at all.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the guard rail",
        "take the potion from the old wooden guard station",
        "take the old wooden guard station",
        "take the queen from the board",
        "take the shin guard from the locker",
    ),
)
def test_aud_11_own_adversarial_pass_the_prior_negatives_still_reject(declaration):
    """The AUD-11 repair reads the whole tail after a clause's rightmost person word
    rather than stopping at the first unrecognized trailing word, which is a genuinely
    more permissive rule than round 5's. Reconfirming the exact prior-round negative
    cases with no NPC scoped present is this round's own required proof that the wider
    tolerance did not quietly let one of them back in: none of "rail", "station" (twice,
    with and without a leading "from"), or "board" carries a locational adverb,
    relative/infinitive signal, or "-ing"/"-ed" verb form anywhere in its own tail, so
    none of them reduce to the person word that precedes them.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the queen from the tarnished board",
        "take the king from the polished chessboard",
    ),
)
def test_aud_12_a_bare_objects_own_clause_never_absorbs_a_trailing_source_clause(declaration):
    """AUD-12 regression: ``_bare_object_is_a_person``'s clause was anchored at the
    theft verb but did not stop at "from"/"off", so for "take the queen from the
    tarnished board" it absorbed both "queen" (the actual taken object) and "the
    tarnished board" (the unrelated source clause) into one scanning unit. The tail-scan
    then found "queen" as the rightmost person word and misread "tarnished" -- an "-ed"
    word meant to recognize a participial phrase still describing a *person* -- as
    describing "queen", even though it actually describes "board", three words later
    and past an intervening "from". This reopened the exact AUD-7 chess-piece bug class
    through the "-ed"/"-ing" signal instead of a coincidental "the". The repair adds
    "from"/"off" to the shared clause boundary both scans respect, so the bare-object
    clause never reaches into a trailing source clause at all.

    "take the queen from the rusted cabinet" was originally paired with these two as a
    third non-theft case, on the theory that any source clause failing to name a person
    should defer the whole match to non-theft. AUD-13 (below) found that theory too
    broad -- it silently rejected the foundational "take the guard from the tower"
    shape too -- and narrowed the trigger to a source clause that positively names the
    competing chess/game sense, not merely one that fails to name a person. "rusted
    cabinet" names neither a person nor a chessboard, so it now confirms as theft; see
    ``test_aud_13_the_foundational_bare_person_case_still_confirms_from_an_ordinary_source``.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


def test_aud_12_the_established_comma_bounded_control_still_rejects():
    """The auditor's own control case, already correct before this repair: a comma
    boundary between the object and its trailing description keeps working via the
    pre-existing clause-punctuation boundary, isolating AUD-12's cause to the missing
    "from"/"off" boundary specifically rather than the clause-boundary logic itself.
    """
    policy = classify_turn("take the queen from the board, gleaming", scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the guard captain of the watch",
        "take the sword from the guard captain of the watch",
    ),
)
def test_aud_12_a_title_connected_by_of_still_confirms(declaration):
    """A secondary, smaller gap the auditor bundled into the same round: a person title
    joined to a role by "of" ("the guard captain OF the watch") is a third grammatical
    shape for naming a person, alongside the genitive ("the merchant's purse") and the
    relative/infinitive signals ("who"/"sent to"). Recognizing "of" the same way closes
    it in both the bare-object and source-clause positions.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the sword from the old guard",
        "steal the crown from the young queen",
        "take it from the wounded guard",
        "steal the ring from the sleeping merchant",
        "take the sword from the tall old bearded guard standing there",
        "take the coin purse from the tall old bearded guard nearby",
        "take the sword from the very tall old bearded guard",
        "take the sword from the very tall old grizzled bearded guard",
        "take the sword from the guard who arrived late",
        "take the sword from the guard sent to watch",
        "steal the coin from the merchant who just arrived",
        "take the pouch from the tall guard standing watch",
        "take the sword from the guard standing watch",
        "take the pouch from the guard standing there",
    ),
)
def test_aud_12_every_prior_adjective_case_still_scopes_to_the_right_clause_with_a_from_clause(
    declaration,
):
    """Every adjective- or trailing-material-qualified person reference this slice's
    history has added, re-verified together with a "from" clause following it, since an
    interaction between the leading/trailing material of one clause and a subsequent
    "from" clause is exactly the shape that has caused every regression in this slice so
    far (AUD-10's fixed window, AUD-11's abort-on-first-miss, and now AUD-12's
    clause-boundary gap all surfaced here first).
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the guard from the tower",
        "take the guard from his post",
        "steal the queen from the throne room",
        "grab the merchant from the crowd",
        "take the stranger from the cell",
        "take the queen from the rusted cabinet",
    ),
)
def test_aud_13_the_foundational_bare_person_case_still_confirms_from_an_ordinary_source(
    declaration,
):
    """Test aud 13 the foundational bare person case still confirms from an ordinary source.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the queen from the board",
        "take the king from the chessboard",
        "take the knight from the board",
        "steal the queen off the chessboard",
        "take the queen from the tarnished board",
        "take the king from the polished chessboard",
    ),
)
def test_aud_13_the_ambiguous_word_set_still_defers_to_a_genuine_chess_context(declaration):
    """The narrowed AUD-13 fix must still resolve every AUD-7/AUD-12 chess-context case
    the same way it always has: "queen"/"king"/"knight" only defer to their source
    clause, and only reject, when that source clause positively names the competing
    sense (``_CHESS_CONTEXT_NOUNS``: "board", "chessboard"), not merely whenever a
    source clause is present. Reconfirming the exact prior fixture text is this round's
    proof that narrowing the trigger from "source fails to name a person" to "source
    positively names a board" did not quietly let the original bug back in.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the shin guard from the locker",
        "take the arm guard from the chest",
        "take the elbow guard from the rack",
        "take the wrist guard from the shelf",
        "take the mouth guard from the bag",
        "take the knee guard from the shelf",
        "take the ear guard from the rack",
        "take the hand guard from the box",
        "take the leg guard from the crate",
        "take the neck guard from the peg",
        "take the foot guard from the bin",
        "take the old shin guard from the locker",
        "take the tarnished shin guard from the locker",
    ),
)
def test_aud_13_own_adversarial_pass_equipment_compounds_still_reject_with_a_from_clause(
    declaration,
):
    """A collateral regression this round's own repair introduced and then closed
    before submitting: once an unambiguous word like "guard" stopped deferring to its
    source clause unconditionally, "take the shin guard from the locker" -- one of
    AUD-7's own original required negative cases -- started confirming again, because
    the truncated bare-object clause "the shin guard" reduces to the person word
    "guard" with an empty tail once "from the locker" is no longer part of the same
    scan. "shin"/"arm"/"wrist"/"mouth"/... directly before "guard" form an equipment
    compound, not an adjective describing a person, and ``_COMPOUND_FORMING_MODIFIERS``
    recognizes that combination specifically so it can keep rejecting these regardless
    of what follows "from", the same way the equipment compound never named a person
    with no source clause at all (AUD-7). Leading material before the body-part modifier
    ("the OLD shin guard", "the TARNISHED shin guard") does not change the answer,
    matching AUD-10/AUD-12's adjective-tolerance precedent applied to the modifier that
    actually forms the compound.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


def test_aud_13_an_ambiguous_word_with_a_genuine_person_source_still_confirms():
    """"take the queen from the guard" already confirms via ``_source_names_a_person``
    independently (the composition checks that before falling back to
    ``_bare_object_is_a_person``), but this exercises the same shape end to end to
    confirm the AUD-13 narrowing did not disturb it: an ambiguous word taken from a
    source that names an actual person is still theft, regardless of the chess-context
    check.
    """
    policy = classify_turn("take the queen from the guard", scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the shin guard",
        "take the arm guard",
        "take the elbow guard",
        "take the wrist guard",
        "take the mouth guard",
        "take the knee guard",
        "take the ear guard",
        "take the hand guard",
        "take the leg guard",
        "take the neck guard",
        "take the foot guard",
    ),
)
def test_aud_14_a_bare_equipment_compound_with_no_source_clause_never_confirms(declaration):
    """AUD-14 regression: the AUD-13 repair's ``_COMPOUND_FORMING_MODIFIERS``
    suppression was nested inside ``if followed_by_source and
    preceded_by_compound_modifier``, even though the compound itself -- not anything a
    source clause could name -- is what disqualifies the match. With no "from"/"off"
    clause at all, ``followed_by_source`` is False, so the suppression never ran and the
    bare match fell through to confirm. "take the shin guard", declared with nothing
    else in the sentence, is at least as natural as "take the shin guard from the
    locker" (already required to reject since AUD-7), and this shape had zero fixture
    coverage for five rounds: it worked under the original AUD-7-era residual-
    subtraction design, broke silently when that was replaced by the "last content
    word" scan (which never inspects a leading modifier), and stayed broken through
    every subsequent round because every AUD-7/AUD-13 fixture in this file happens to
    include a "from ..." clause. The fix drops the ``followed_by_source`` condition
    from this suppression specifically -- it now fires whenever the word immediately
    before the match forms a compound, regardless of what, if anything, follows -- while
    leaving the chess-context suppression's own source-clause gate untouched, since that
    one genuinely does depend on what follows "from"/"off".
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        "take the old shin guard",
        "take the tarnished shin guard",
        "take the shin guard, standing there",
    ),
)
def test_aud_14_own_adversarial_pass_a_bare_compound_with_a_leading_adjective_or_trailing_material(
    declaration,
):
    """A further adversarial pass beyond the auditor's own two examples: a bare
    equipment compound still rejects with a leading adjective before the body-part
    modifier, or trailing material after "guard", since neither changes which word sits
    immediately before the match.
    """
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"


@pytest.mark.parametrize(
    "declaration",
    (
        'Could I defeat him? Is he armed?',
        'He seems tough. Could I defeat him? Is he armed?',
        "Should I take him?",
        "Would I be able to take him?",
        "Could I take him?",
        "What if I take him? What does he carry?",
        "What would happen if I take him?",
    ),
)
def test_hypothetical_questions_never_commit_the_embedded_action(declaration):
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "theft"
    assert policy.risk_category != "violence"


@pytest.mark.parametrize(
    "declaration",
    (
        "I take it from reed thug. What next?",
        "I stab reed thug. What happens now?",
    ),
)
def test_a_real_declaration_still_commits_beside_an_unrelated_question(declaration):
    """A trailing, unrelated question must never mask a genuinely declared act."""
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route == "risk"


def test_a_raised_declaration_with_its_own_trailing_question_mark_still_commits():
    """"I stab reed thug?" is a declared act voiced uncertainly, not a question."""
    policy = classify_turn("I stab reed thug?", scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "violence"


def test_aud_3_a_question_opener_governs_only_its_own_segment_before_a_then_chain():
    """AUD-3: "Should I open the door, then I stab reed thug?" asks about the door and
    separately commits to the attack in the same sentence. The first repair round's
    clause-splitting discarded the whole clause -- including the chained action --
    once it saw the clause open with "should"; this pins the fix, which judges the
    segment after "then" on its own rather than inheriting the question opener's
    disqualification. Reproduced by the auditor against base commit
    55518c3b98574cb0ea95ce8262f62201dcf720fa, where this was risk/violence.
    """
    policy = classify_turn("Should I open the door, then I stab reed thug?", scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "violence"


@pytest.mark.parametrize(
    "declaration",
    (
        "Should I open the door? Then I stab reed thug.",
        "What if I open the door, then I take his purse?",
    ),
)
def test_a_then_chained_action_commits_across_clause_and_sentence_boundaries(declaration):
    """The same "question, then committed act" shape holds whether "then" sits inside
    one clause or opens a new sentence, and whether the chained act is violence or
    theft."""
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route == "risk"


@pytest.mark.parametrize(
    "declaration",
    (
        "I am attacking reed thug",  # a listed root ("attack") inflected, not exact
        "I keep stabbing reed thug",  # "stabbing" is not itself a member of the list
        "reed thug is punching me",  # "punching" likewise
        "I killed reed thug",  # past tense of "kill"
        "I harmed reed thug",  # past tense of "harm"
    ),
)
def test_a_regular_inflection_outside_the_exact_violent_list_still_reads_as_violence(declaration):
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "violence"


@pytest.mark.parametrize(
    "declaration",
    (
        # AUD-1: every one of these misclassified as risk/violence in the first
        # repair round's prefix/suffix substring approach. Each contains a violent
        # root as a literal substring but is not itself the root or one of its
        # regular inflections, and the auditor reproduced every one directly.
        "I use my persuasion skill to calm the guard",  # "skill" ends with "kill"
        "I lead the horse into the stable",  # "stable" starts with "stab"
        "I feel upbeat about our chances",  # "upbeat" ends with "beat"
        "there is harmony in the camp",  # "harmony" starts with "harm"
        "this place feels harmless",  # "harmless" starts with "harm"
        "I am kneeling before the altar",  # "kneeling" starts with "knee" (a different verb, "kneel")
        "I hear a heartbeat",  # "heartbeat" ends with "beat"
        "he is my sidekick",  # "sidekick" ends with "kick"
        "we enter the nightclub",  # "nightclub" ends with "club"
        "the clubhouse is empty",  # "clubhouse" starts with "club"
        "she plays the harmonica",  # "harmonica" starts with "harm"
        "I see roadkill on the path",  # "roadkill" ends with "kill"
        "that would be overkill",  # "overkill" ends with "kill"
        "I kickstart the engine",  # "kickstart" starts with "kick"
        "the kickoff is soon",  # "kickoff" starts with "kick"
    ),
)
def test_aud_1_ordinary_words_containing_a_violent_root_are_never_violence(declaration):
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "violence"


def test_a_short_violent_root_needs_no_length_based_exclusion_under_exact_matching():
    """Test a short violent root needs no length based exclusion under exact matching.
    """
    policy = classify_turn("I take the shortcut through the market", scope=_NPC_PRESENT)
    assert policy.route != "risk"


def test_a_violent_compound_word_is_a_known_accepted_gap():
    """"backstab" is no longer recognized: AUD-1 established that no rule distinguishes
    a genuine violent compound from an ordinary one ("heartbeat", "sidekick") without
    either a violence-specific compound allowlist or a safe-word denylist, and both are
    the same closed-list trap word-boundary matching replaces. This is a known,
    documented gap, not a silent regression: the live probe's unlisted-violence
    scenario now declares an inflected form ("stabbing") instead."""
    policy = classify_turn("I backstab reed thug", scope=_NPC_PRESENT)
    assert policy.route != "risk"


@pytest.mark.parametrize(
    "declaration",
    (
        "I'm not attacking; I only want to comfort her",
        "I am not attacking, I just want to talk",
        "I will not attack",
        "I don't attack the guard",
        "I wouldn't stab him",
        "no, I never hit anyone",
    ),
)
def test_a_negated_violent_verb_never_reads_as_violence(declaration):
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route != "risk"
    assert policy.risk_category != "violence"


@pytest.mark.parametrize(
    "declaration",
    (
        "I attack the guard",  # the plain positive control
        "I said I wouldn't, then I stab him anyway",  # the negation sits in an earlier clause
        "I won't back down, so I attack",  # the negation modifies a different verb entirely
        "not now, but I attack the guard",  # the negation shares no clause with the verb
    ),
)
def test_a_genuine_violent_clause_still_confirms_beside_an_unrelated_negation(declaration):
    policy = classify_turn(declaration, scope=_NPC_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "violence"


_FIGHT_SCOPE = TrustedScope("session-1", "market", ("choir-listener",))


def _fight(*npc_ids: str, active: bool = True) -> CombatSnapshot:
    return CombatSnapshot(
        active=active,
        round=1,
        active_actor="rill",
        order=("rill", *npc_ids),
        sides=(("rill", "pc"), *((npc, "npc") for npc in npc_ids)),
    )


@pytest.mark.parametrize(
    "declaration",
    [


        '@GM I attack the choir listener immediately, why was I refused?',
        'ooc I stab the choir listener now -- does that need a roll?',
        'gm, do I need to confirm when I stab the choir listener now now?',
        "ooc I punch the barkeep, is that allowed?",
    ],
)
def test_m13_an_out_of_character_question_is_never_claimed_by_the_hazard_branch(declaration):
    """Each of these quotes a violence verb, so each reached ``route == \"risk\"`` before the
    reorder and was answered with a confirmation the player never asked for.
    """
    policy = classify_turn(declaration, scope=_FIGHT_SCOPE)
    assert policy.route == "out_of_character"
    assert policy.risk_category == "none"


@pytest.mark.parametrize(
    ("declaration", "route"),
    [
        # An out-of-character *declaration* is still a declaration. The marker alone must
        # never lower the floor, or "ooc" becomes a bypass for every hazard.
        ('ooc I stab the choir listener now', "risk"),
        ('gm I stab the choir listener now', "risk"),
        # A marker that surfaces only after the declared act leaves the act first.
        ('I stab the choir listener now. gm?', "risk"),
        ('I stab the choir listener now! ooc?', "risk"),
        # No marker at all: a raised declaration keeps committing, which is the edge
        # ``test_a_raised_declaration_with_its_own_trailing_question_mark_still_commits``
        # already pins and which this reorder must not disturb.
        ('I stab the choir listener now now?', "risk"),
    ],
)
def test_m13_the_marker_alone_never_lowers_the_hazard_floor(declaration, route):
    """Test m13 the marker alone never lowers the hazard floor.
    """
    assert classify_turn(declaration, scope=_FIGHT_SCOPE).route == route


def test_m13_the_ooc_question_yields_to_the_social_branch_rather_than_claiming_it():
    """An audit (``AUD-1``) rejected a first candidate that returned ``out_of_character`` from
    this branch ahead of every branch below it. Its differential found that two thirds of
    the classifications it changed were turns the *social* branch had been taking, which
    lost their social fields and with them three engine-owned gates.

    \"rules question: if I stab the choir listener do I roll?\" is the case that shows the
    bound: the social classifier claims it, so the social branch keeps it. What the
    milestone owes this declaration is that the hazard branch no longer claims it, which is
    what stops the spurious confirmation -- not that it reach any particular other route.
    """
    policy = classify_turn(
        'rules question: if I stab the choir listener now do I roll?', scope=_FIGHT_SCOPE
    )
    assert policy.route != "risk"
    assert policy.risk_category == "none"


@pytest.mark.parametrize(
    ("declaration", "field", "expected"),
    [
        # The romance consent and coercion gate. The audit reproduced this one end to end:
        # the rejected candidate delivered in-character narration where the base tree
        # withheld the turn and posted the consent boundary notice.
        ("ooc I kiss the stranger?", "romance_escalation", True),
        ("@GM I kiss the stranger, is that allowed?", "romance_escalation", True),
        # The trade confirmation phase, which is milestone M12's own closed mechanism.
        ("ooc i buy a rope for 5 coin?", "trade_phase", "purchase_intent"),
        ("gm, I buy the ring?", "trade_phase", "purchase_intent"),
        # The social-test phase.
        ("gm i persuade the guard to let us pass?", "social_test_required", True),
        ("ooc I threaten the barkeep?", "social_test_required", True),
    ],
)
def test_m13_an_ooc_marked_social_turn_still_reaches_its_engine_owned_gate(
    declaration, field, expected
):
    """Each declaration below carries an out-of-character marker in its opening clause and
    reads as a question, so the rejected candidate routed every one of them to
    ``out_of_character`` with every social field cleared. The service keys the romance
    consent gate, the trade confirmation phase, and the social-test phase off exactly these
    fields, so clearing them bypassed all three. A prompt-level framing instruction is not
    a substitute for a mechanical consent boundary.
    """
    policy = classify_turn(declaration, scope=_FIGHT_SCOPE)
    assert policy.route == "social"
    assert getattr(policy, field) == expected


@pytest.mark.parametrize(
    "declaration",
    [
        "stab the Choir Listener with my dagger",
        "stab the choir listener in the throat",
        "attack the choir listener with my sword",
        "kick the choir listener hard",
        "punch the choir listener in the face twice",
        "swing my axe at the choir listener",
        "stab the choir listener again",
    ],
)
def test_m13_naming_a_weapon_body_part_or_manner_keeps_the_fight_sanction(declaration):
    """Each declaration below is the sanctioned \"stab the choir listener\" plus a word saying
    how the blow lands. Before M13 every one of them withheld the sanction, because the
    residual-word subtraction treated \"dagger\", \"throat\", or \"twice\" as a possible second
    target -- so naming the weapon re-triggered the confirmation the bare declaration
    never got.
    """
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight("choir-listener")), _fight("choir-listener")) is True


@pytest.mark.parametrize(
    "declaration",
    [
        # Third-person, dictionary-root control -- already sanctioned before M14.
        "attack the choir listener",
        # First-person pronoun alone (no tense gap): only "i" survives pre-fix.
        'I attack the choir listener immediately',
        'i stab the choir listener now',
        # First-person contraction: tokenizes to "i", "m", and the progressive form, so
        # both gaps this milestone closes stack in one declaration.
        "I'm attacking the choir listener",
        # First-person "am" copula plus progressive tense: same double gap, spelled out.
        "I am stabbing the choir listener",
        "I am attacking the choir listener",
        # Progressive tense alone, no subject at all: the inflection gap in isolation,
        # independent of the pronoun gap.
        "attacking the choir listener",
    ],
)
def test_m14_pronoun_and_verb_inflection_never_defeat_the_sanction(declaration):
    """Before this milestone, \"i\" (and \"am\"/\"m\" from a contraction) survived the residual
    subtraction as an unrecognized word, and a violent verb's own progressive-tense
    inflection (\"attacking\", \"stabbing\") survived it too, because the subtraction read
    ``_VIOLENT_ACTIONS`` -- exact roots only -- rather than ``_VIOLENT_INFLECTIONS``. Either
    gap alone withheld the sanction; a declaration carrying both (first-person and
    progressive tense together, \"I'm attacking ...\") failed it twice over. Every phrasing
    below now sanctions exactly like the bare dictionary-root control.
    """
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight("choir-listener")), _fight("choir-listener")) is True


#: The withholding evidence for a second target is the campaign record: these NPCs are
#: present and alive but never rostered into the fight, so naming one withholds the
#: sanction whatever else the declaration carries.
_SECOND_TARGET_SCOPE = TrustedScope(
    "session-1",
    "market",
    ("choir-listener", "fishwife", "smuggler", "market-child", "acolyte"),
    ("rill",),
    (
        ("choir-listener", "alive"),
        ("fishwife", "alive"),
        ("smuggler", "alive"),
        ("market-child", "alive"),
        ("acolyte", "alive"),
    ),
)


@pytest.mark.parametrize(
    "declaration",
    [
        # A genuinely different target still withholds: a recorded, living person the
        # fight's roster omits, named by their own campaign identifier.
        "stab the choir listener and the fishwife",
        "kill the smuggler next to the choir listener",
        "stab the choir listener and the child",
        "stab the choir listener with my dagger and then the acolyte",
        # M14: the same second-target shapes, first-person and progressive-tense phrased.
        # A leading pronoun or a verb's own inflection tolerating the subject and the verb
        # must not also tolerate a recorded second target riding along with them.
        'I stab the choir listener now and the fishwife',
        "I'm stabbing the choir listener and the fishwife",
        "I am attacking the smuggler next to the choir listener",
    ],
)
def test_m13_a_second_target_still_withholds_the_sanction(declaration):
    """A recorded person outside the fight withholds the sanction in any phrasing.

    The evidence is typed campaign state (present NPCs and their statuses), not a
    noun list: the residual-word subtraction this test once exercised treated every
    unrecognized word as a possible person, which broke live play once per unlisted
    word. Unrecorded referents are protected at the tool boundary instead
    (``target_not_in_combat``)."""
    assert (
        combat_sanctions_violence(policy_for(declaration, scope=_SECOND_TARGET_SCOPE, combat=_fight("choir-listener")), _fight("choir-listener"))
        is False
    )


@pytest.mark.parametrize(
    "declaration",
    [


        "I stab Rade with my long knife",
        "stab the choir listener with my long knife",
        "I hit the choir listener with my heavy iron mace",
        "swing my old rusty axe at the choir listener",
        "cut the choir listener with my sharp short blade",
    ],
)
def test_a_weapon_adjective_keeps_the_fight_sanction(declaration):
    """A weapon's adjectives describe the instrument, never a person; none of them
    may withhold the sanction the bare declaration receives."""
    npc = "rade" if "Rade" in declaration else "choir-listener"
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight(npc)), _fight(npc)) is True


@pytest.mark.parametrize(
    "declaration",
    [


        'I yell "help, help! I\'m being attacked by Rade!!!"',
        'I shout "he attacked me, someone help!"',
        # Passive voice without quotation marks: violence suffered, not declared.
        "I'm being attacked by Rade",
        "I was attacked on the road last night",
        "we got attacked at the shrine",
    ],
)
def test_reported_or_suffered_violence_is_never_a_violence_declaration(declaration):
    """Words a character says, and violence a character suffers, are not deeds the
    character does; neither may reach the risk floor's confirmation."""
    policy = classify_turn(declaration, scope=_FIGHT_SCOPE)
    assert policy.route != "risk"
    assert policy.risk_category == "none"


@pytest.mark.parametrize(
    "declaration",
    [
        # Quoted speech beside a real act: the act outside the quotes keeps its hazard.
        'I say "I\'ve got a lot of gutting to do too" and stab the stranger',
        # A passive report beside an active answer: the active verb keeps its hazard.
        "I was attacked first, so now I stab the stranger back",
    ],
)
def test_an_act_beside_speech_or_a_passive_report_keeps_its_hazard(declaration):
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == "violence"


@pytest.mark.parametrize(
    "declaration",
    [


        "Who's turn is it",
        "whose turn is it?",
        "What round is it",
        "how many actions do I have left",
    ],
)
def test_a_mechanics_state_question_routes_out_of_character(declaration):
    policy = classify_turn(declaration, scope=_FIGHT_SCOPE)
    assert policy.route == "out_of_character"


@pytest.mark.parametrize(
    ("declaration", "route"),
    [
        # Mechanics vocabulary in a committed act never routes to the answer branch.
        ("I spend my turn stabbing the choir listener", "risk"),
        ("I turn around", "planner"),
    ],
)
def test_mechanics_vocabulary_inside_an_act_keeps_its_route(declaration, route):
    assert classify_turn(declaration, scope=_FIGHT_SCOPE).route == route


@pytest.mark.parametrize(
    "declaration",
    [


        "I move up to striking distance",
        "I close the distance and strike the choir listener",
        "I step forward and attack",
        "advance and cut the choir listener",
    ],
)
def test_positioning_words_keep_the_fight_sanction(declaration):
    """Movement and positioning vocabulary describes where the actor stands, never
    who they strike; none of it may withhold the sanction mid-fight."""
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight("choir-listener")), _fight("choir-listener")) is True


def test_positioning_words_never_widen_the_sanction_to_a_recorded_bystander():
    scope = TrustedScope(
        "session-1",
        "market",
        ("choir-listener", "market-child"),
        ("rill",),
        (("choir-listener", "alive"), ("market-child", "alive")),
    )
    assert (
        combat_sanctions_violence(policy_for("I move toward the child and strike", scope=scope, combat=_fight("choir-listener")), _fight("choir-listener"))
        is False
    )


@pytest.mark.parametrize(
    ("declaration", "category"),
    [
        # The live shape: "take" hit the theft branch, its person-signal missed (the
        # trader is punched, not stolen from), and the old early return swallowed the
        # violence check -- a first-strike assault on a bystander reached the planner
        # with no confirmation at all. The violence LEADS the declaration, which is
        # what licenses the fall-through past the theft-miss.
        ("I punch the trader in the face and take the rope", "violence"),
        ("stab the guard and take the lamp", "violence"),
        # When the taking DOES steal from a person, theft still classifies first --
        # a confirmation gates the compound either way.
        ("I slash at the merchant, then take his stall", "theft"),
        ("take the sword from the merchant", "theft"),
    ],
)
def test_a_leading_violent_act_is_never_swallowed_by_a_theft_miss(declaration, category):
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.route == "risk"
    assert policy.risk_category == category


@pytest.mark.parametrize(
    "declaration",
    [
        # The audited take-idiom family the ordering rule exists to protect: the
        # ambiguous word sits after "take", where it is equipment, a report, or a
        # rest -- never the player's own declared violence.
        "take the elbow guard from the rack",
        "take a break",
        "take the sword, then the guard attacked",
        # The reversed compound is the disclosed residual of the ordering rule:
        # violence after the theft verb stays with the planner rather than
        # re-flagging every audited idiom above.
        "take the rope and punch the trader",
    ],
)
def test_a_trailing_ambiguous_word_never_confirms_through_the_fall_through(declaration):
    policy = classify_turn(declaration, scope=_NO_ONE_PRESENT)
    assert policy.risk_category != "violence"


def test_m13_a_bystander_noun_inside_the_combatants_own_name_no_longer_exempts_it():
    """``_BYSTANDER_NOUNS`` lists \"guard\", \"monk\", and \"priest\" unconditionally, so a fight
    whose recorded enemy is ``temple-guard`` had every attack on it read as an attack on a
    bystander and re-confirmed. The scene, not the vocabulary, says what the referent is.
    """
    fight = _fight("temple-guard")
    assert combat_sanctions_violence(policy_for("attack the temple guard", combat=fight), fight) is True
    # A recorded person the fight does not account for still withholds.
    child_scope = TrustedScope(
        "session-1",
        "temple",
        ("temple-guard", "market-child"),
        ("rill",),
        (("temple-guard", "alive"), ("market-child", "alive")),
    )
    assert (
        combat_sanctions_violence(policy_for("attack the temple guard and the child", scope=child_scope, combat=fight), fight)
        is False
    )


def test_m13_a_bystander_noun_naming_a_recorded_corpse_no_longer_exempts_it():
    """Test m13 a bystander noun naming a recorded corpse no longer exempts it.
    """
    dead_monk = TrustedScope(
        "session-1", "market", ("choir-listener", "brother-monk"),
        (), (("brother-monk", "dead"), ("choir-listener", "alive")),
    )
    fight = _fight("choir-listener")
    # "monk" names brother-monk, and the campaign records this monk as dead.
    assert combat_sanctions_violence(policy_for("stab the choir listener over the monk", scope=dead_monk, combat=fight), fight) is True
    # A living recorded bystander is never exempted.
    living = TrustedScope(
        "session-1", "market", ("choir-listener", "brother-monk"),
        (), (("brother-monk", "alive"), ("choir-listener", "alive")),
    )
    assert combat_sanctions_violence(policy_for("stab the choir listener over the monk", scope=living, combat=fight), fight) is False


def test_m14_a_pronoun_led_counter_attack_on_the_fights_own_combatant_never_confirms():
    """Test m14 a pronoun led counter attack on the fights own combatant never confirms.
    """
    fight = _fight("reed-thug")
    assert combat_sanctions_violence(policy_for("I stab the reed thug back", combat=fight), fight) is True
    assert combat_sanctions_violence(policy_for("I'm striking back at the reed thug", combat=fight), fight) is True
    assert combat_sanctions_violence(policy_for("I strike back at the reed thug", combat=fight), fight) is True


@pytest.mark.parametrize(
    "declaration",
    [
        "I attack the wanderer",
        "I'm attacking the wanderer",
        "I am stabbing the wanderer",
        "attack the wanderer",
    ],
)
def test_m14_no_fight_open_still_confirms_regardless_of_phrasing(declaration):
    """Outside a fight ``combat_sanctions_violence`` always withholds, and the pronoun and
    inflection tolerance this milestone adds must not change that: it binds to a
    recorded, engaged combatant, not to the mere presence of a subject pronoun or a
    violent verb.
    """
    assert combat_sanctions_violence(policy_for(declaration, combat=None), None) is False
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight("reed-thug", active=False)), _fight("reed-thug", active=False)) is False


@pytest.mark.parametrize(
    "declaration",
    [
        "I attack the wanderer",
        "I'm attacking the wanderer",
        "I am stabbing the wanderer",
        "attack the wanderer",
    ],
)
def test_m14_a_different_recorded_combatant_still_confirms_regardless_of_phrasing(declaration):
    """The fight is open and engaged, but it records ``reed-thug``, while the campaign
    records ``the-wanderer`` as present and alive outside the roster. Naming the
    wanderer withholds the sanction and the floor still confirms, first-person and
    progressive-tense phrasing included -- the evidence is the campaign's own record
    of the person, not the phrasing.
    """
    scope = TrustedScope(
        "session-1",
        "market",
        ("reed-thug", "the-wanderer"),
        ("rill",),
        (("reed-thug", "alive"), ("the-wanderer", "alive")),
    )
    assert combat_sanctions_violence(policy_for(declaration, scope=scope, combat=_fight("reed-thug")), _fight("reed-thug")) is False


def test_m13_turn_framing_maps_only_the_asking_routes():
    """Test m13 turn framing maps only the asking routes.
    """
    assert turn_framing_for("read") == "question"
    assert turn_framing_for("out_of_character") == "out_of_character"
    for route in ("risk", "social", "planner", "", "unknown"):
        assert turn_framing_for(route) == ""


# -- Consent lives in combat state, never in the declaration's vocabulary ----------------


@pytest.mark.parametrize(
    "declaration",
    [


        "I attack Rade with my razor whip",
        "attack rade with my engraved longbow",
        "I strike Rade with the salt-crusted boarding hook",
        "I bring grandfather's notched harpoon down on Rade",
        "I hurl the eel barrel at rade and follow with a wild haymaker",
    ],
)
def test_no_vocabulary_defeats_the_open_fights_sanction(declaration):
    """Test no vocabulary defeats the open fights sanction.
    """
    assert combat_sanctions_violence(policy_for(declaration, combat=_fight("rade")), _fight("rade")) is True


def test_a_recorded_person_outside_the_fight_still_withholds_the_sanction():
    """The withholding evidence is typed campaign data, not English nouns: a party
    member's recorded identifier, or a present, living NPC the fight's roster does
    not include. A corpse withholds nothing -- it is not a person the table protects."""
    fight = _fight("choir-listener")
    scope = TrustedScope(
        "session-1",
        "market",
        ("choir-listener", "orso-pell"),
        ("rill",),
        (("choir-listener", "alive"), ("orso-pell", "alive")),
    )
    assert combat_sanctions_violence(policy_for("I stab orso pell", scope=scope, combat=fight), fight) is False
    assert (
        combat_sanctions_violence(
            policy_for("attack the choir listener, then orso", scope=scope, combat=fight),
            fight,
        )
        is False
    )
    assert combat_sanctions_violence(policy_for("stab rill the healer", scope=scope, combat=fight), fight) is False
    assert combat_sanctions_violence(policy_for('I attack the choir listener immediately', scope=scope, combat=fight), fight) is True
    corpse_scope = TrustedScope(
        "session-1",
        "market",
        ("choir-listener", "orso-pell"),
        ("rill",),
        (("choir-listener", "alive"), ("orso-pell", "dead")),
    )
    assert (
        combat_sanctions_violence(policy_for("I stab the choir listener now beside orso pell's body", scope=corpse_scope, combat=fight), fight)
        is True
    )
