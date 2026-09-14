"""Pin the soak harness's canon-bound instruments, without a model or a campaign.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from narrator import soak_instruments as soak_session  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402

SCENE_BOUND = NarratorConfig.canon_scene_max_chars
TRAILING_BUFFER = 572

#: Characters one prefilled entry renders: the 81-character line plus its newline.
PREFILL_ENTRY_CHARS = 82


def test_the_generator_is_byte_identical_with_every_probe_off():
    """The evidence path must not move when an instrument lands beside it.
    """
    for seed, expected in (
        (1234, "37f5183bf48d5a3572e21726e16b4fd13abff3cd957c1f793622f0aa591a2d9a"),
        (20260803, "c3a61da35ccd1f8f5f6e1a3a5dcd74eabc8f50c570480f72b9fb6d4a94a0443d"),
    ):
        text = soak_session.generate_transcript(1050, 25, seed)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected


def test_the_clock_question_replaces_exactly_one_mention():
    """A probe that added a line would change every later turn's history."""
    plain = soak_session.generate_transcript(1050, 25, 1234).splitlines()
    probed = soak_session.generate_transcript(1050, 25, 1234, clock_turn=41).splitlines()

    assert len(plain) == len(probed)
    differing = [index for index, line in enumerate(plain) if line != probed[index]]
    assert len(differing) == 1
    assert probed[differing[0]].startswith("@GM")
    assert soak_session.CLOCK_RECALL.split("{actor}")[1][:20] in probed[differing[0]]


def test_the_clock_question_names_neither_the_clock_nor_its_fill():
    """Scoring a question that carries its own answer measures echo, not recall."""
    question = soak_session.CLOCK_RECALL.lower()
    for token in soak_session.CLOCK_TOKENS:
        assert token not in question
    assert f"{soak_session.CLOCK_FILLED}/{soak_session.CLOCK_SEGMENTS}" not in question

    transcript = soak_session.generate_transcript(1050, 25, 1234, clock_turn=41).lower()
    for token in soak_session.CLOCK_TOKENS:
        assert transcript.count(token) == 0


def test_the_prefill_fixture_reaches_past_the_trailing_buffer():
    """The depth parameter must reach the rung under the sections that buffer the facts.
    """
    depth = (SCENE_BOUND + TRAILING_BUFFER) // PREFILL_ENTRY_CHARS + 1
    payload = soak_session.prefill_payload(depth)
    facts = payload["visible_changes"]

    assert len(facts) == depth
    assert len(set(facts)) == depth
    rendered = sum(len(f"- {fact}") + 1 for fact in facts)
    assert rendered > SCENE_BOUND + TRAILING_BUFFER
    assert rendered - PREFILL_ENTRY_CHARS <= SCENE_BOUND + TRAILING_BUFFER
    assert soak_session.prefill_payload(depth) == payload
    assert soak_session.prefill_payload(0)["visible_changes"] == []


def test_the_clock_fixture_plants_one_clock_with_the_scored_tokens():
    """The probe scores the record's own rendering, so the plant must carry both tokens."""
    clocks = soak_session.clock_payload()["new_clocks"]

    assert len(clocks) == 1
    clock = clocks[0]
    assert clock["id"] == soak_session.CLOCK_ID
    assert clock["name"] == soak_session.CLOCK_NAME
    assert clock["note"] == soak_session.CLOCK_NOTE
    assert 0 < clock["filled"] < clock["segments"]
    rendered = f"- {clock['name']} ({clock['filled']}/{clock['segments']}) {clock['note']}"
    for token in soak_session.CLOCK_TOKENS:
        assert token in rendered


def test_the_clock_probe_scores_strict_with_lenient_and_the_confounds_beside_it():
    """Strict gates nothing here, so the record must classify every miss it reports."""
    canon = "- brackwater levy (3/6) hulls counted at the mole"

    hit = soak_session.score_clock_recall(
        "The brackwater levy stands at 3/6, hulls counted at the mole.",
        canon, "3/6", "absent", ["attribute_test"],
    )
    assert hit["recalled"] is True
    assert hit["lenient"]["recalled"] is True
    assert hit["fill_in_reply"] is True
    assert hit["present_in_canon"] is True
    assert hit["clocks_section_state"] == "absent"
    assert hit["read_tool_called"] is False

    miss = soak_session.score_clock_recall(
        "Nothing presses on you that I can name.", canon, "3/6", "absent", []
    )
    assert miss["recalled"] is False
    assert miss["lenient"]["recalled"] is False
    assert miss["tokens_in_reply"] == []
    assert miss["present_in_canon"] is True


def test_the_lenient_channel_separates_formatting_drift_from_loss():
    """A hyphenated or line-broken answer holds the fact; strict containment cannot see it."""
    canon = "- brackwater levy (3/6) hulls counted at the mole"
    drifted = soak_session.score_clock_recall(
        "The Brackwater-Levy clock: hulls\ncounted at the Mole.",
        canon, "3/6", "absent", [],
    )

    assert drifted["recalled"] is False
    assert drifted["lenient"]["recalled"] is True
    assert len(drifted["lenient"]["tokens_in_reply"]) == 2


def test_a_read_tool_call_at_the_probe_turn_is_recorded_not_hidden():
    """One served tool reports clock state, so a pass through it is not a pass through canon."""
    assert soak_session.CLOCK_READ_TOOLS == ("campaign_status",)
    scored = soak_session.score_clock_recall(
        "The brackwater levy stands at 3/6, hulls counted at the mole.",
        "- brackwater levy (3/6) hulls counted at the mole",
        "3/6",
        "absent",
        ["campaign_status", "scene_commit"],
    )

    assert scored["recalled"] is True
    assert scored["read_tool_called"] is True
    assert scored["tool_names"] == ["campaign_status", "scene_commit"]


def test_no_zone_question_carries_its_own_answer():
    """A question naming its plant's content would score echo, not recall.
    """
    zones = (
        (soak_session.ZONE_B_TOKENS, soak_session.ZONE_B_RECALL),
        (soak_session.ZONE_C_TOKENS, soak_session.ZONE_C_RECALL),
        (soak_session.ZONE_A_TOKENS, soak_session.ZONE_A_RECALL),
    )
    every_token = [t for tokens, _ in zones for t in tokens]
    for _, question in zones:
        for token in every_token:
            assert token not in question.lower()
            assert (
                soak_session.normalize_for_lenient_match(token)
                not in soak_session.normalize_for_lenient_match(question)
            )
    assert len(set(every_token)) == 6


def test_the_zone_fixture_plants_the_oldest_fact_before_the_prefill():
    """Head retention keeps the oldest entries, so zone B must reach the list first.

    Reversing this order would bury zone B under 40 prefilled entries, and the zone
    named 'inside the digest' would measure the zone named 'outside it'.
    """
    payloads = soak_session.fixture_payloads(40, clock=True, zone=True)

    assert len(payloads) == 3
    assert payloads[0]["visible_changes"] == [soak_session.ZONE_B_FACT]
    assert len(payloads[1]["visible_changes"]) == 40
    assert payloads[2]["new_clocks"][0]["id"] == soak_session.CLOCK_ID
    for token in soak_session.ZONE_B_TOKENS:
        assert token in soak_session.ZONE_B_FACT.lower()
    assert soak_session.fixture_payloads(0, clock=False, zone=False) == []


def test_each_zone_mention_replaces_exactly_one_mention():
    """Five zone turns must not add or drop a line, or every later turn shifts."""
    zone_turns = {
        "zone_c_plant": 20,
        "zone_a_plant": 38,
        "zone_b_recall": 39,
        "zone_c_recall": 40,
        "zone_a_recall": 41,
    }
    plain = soak_session.generate_transcript(1050, 25, 1234).splitlines()
    probed = soak_session.generate_transcript(
        1050, 25, 1234, zone_turns=zone_turns
    ).splitlines()

    assert len(plain) == len(probed)
    differing = [index for index, line in enumerate(plain) if line != probed[index]]
    assert len(differing) == 5
    for index in differing:
        assert probed[index].startswith("@GM")


def test_the_zone_score_separates_delivery_loss_from_every_other_miss():
    """A miss is only a delivery miss when the record holds the fact and the digest does not."""
    canon = "- The party knotted the tide-medal into the waxed pouch at her belt."

    delivery_loss = soak_session.score_zone_recall(
        "C_outside_both", soak_session.ZONE_C_TOKENS,
        "I have no record of that.", canon, False, 0, [],
    )
    assert delivery_loss["committed"] is True
    assert delivery_loss["recalled"] is False
    assert delivery_loss["in_digest_at_recall"] is False

    hit = soak_session.score_zone_recall(
        "C_outside_both", soak_session.ZONE_C_TOKENS,
        "The tide-medal sits in the waxed pouch at her belt.",
        canon, True, 42, ["campaign_status"],
    )
    assert hit["recalled"] is True
    assert hit["lenient"]["recalled"] is True
    assert hit["in_digest_at_recall"] is True
    assert hit["digest_turns"] == 42
    assert hit["read_tool_called"] is True

    never_written = soak_session.score_zone_recall(
        "C_outside_both", soak_session.ZONE_C_TOKENS,
        "I have no record of that.", "- None recorded.", False, 0, [],
    )
    assert never_written["committed"] is False


def test_the_ordering_question_names_no_place_it_scores():
    """A question carrying a place would score echo, not sequence.
    """
    question = soak_session.ORDER_RECALL.lower()
    for place in soak_session.ORDER_PLACES:
        assert place not in question
        assert (
            soak_session.normalize_for_lenient_match(place)
            not in soak_session.normalize_for_lenient_match(question)
        )
    assert soak_session.ORDER_OBJECT in question
    assert len(set(soak_session.ORDER_PLACES)) == 3


def test_each_ordering_place_appears_only_where_the_probe_puts_it():
    """Test each ordering place appears only where the probe puts it.
    """
    probed = soak_session.generate_transcript(
        1050, 25, 1234,
        order_turns={
            "order_move_first": 12, "order_move_second": 22, "order_recall": 37
        },
    ).lower()

    assert probed.count(soak_session.ORDER_PLACES[0]) == 0
    assert probed.count(soak_session.ORDER_PLACES[1]) == 1
    assert probed.count(soak_session.ORDER_PLACES[2]) == 1


def test_the_three_ordering_mentions_replace_exactly_three_lines():
    """A probe that added a line would change every later turn's history."""
    plain = soak_session.generate_transcript(1050, 25, 1234).splitlines()
    probed = soak_session.generate_transcript(
        1050, 25, 1234,
        order_turns={
            "order_move_first": 12, "order_move_second": 22, "order_recall": 37
        },
    ).splitlines()

    assert len(plain) == len(probed)
    differing = [index for index, line in enumerate(plain) if line != probed[index]]
    assert len(differing) == 3
    for index in differing:
        assert probed[index].startswith("@GM")


def test_the_ordering_fixture_plants_the_first_place_before_the_prefill():
    """The path's origin has to sit at the head of the list, like zone B does."""
    payloads = soak_session.fixture_payloads(40, clock=True, zone=True, order=True)

    assert len(payloads) == 4
    assert payloads[0]["visible_changes"] == [soak_session.ZONE_B_FACT]
    assert soak_session.ORDER_PLACES[0] in payloads[1]["visible_changes"][0]
    assert len(payloads[2]["visible_changes"]) == 40
    assert soak_session.fixture_payloads(0, clock=False, zone=False, order=False) == []


def test_ordered_recall_demands_the_recorded_sequence_not_only_the_places():
    """The score this probe adds: three places in the wrong order is not a path.

    Every other recall mode plants one fact and scores its return, so a reply naming
    all three places backwards would pass every one of them.
    """
    canon = "\n".join(f"- moved to the {place}" for place in soak_session.ORDER_PLACES)
    first, second, third = soak_session.ORDER_PLACES

    in_order = soak_session.score_order_recall(
        f"It sat in the {first}, then the {second}, and now under the {third}.",
        canon, [], 0, [],
    )
    assert in_order["ordered"] is True
    assert in_order["recalled"] is True
    assert in_order["all_committed"] is True

    reversed_order = soak_session.score_order_recall(
        f"It sits under the {third}, before that the {second}, before that the {first}.",
        canon, [], 0, [],
    )
    assert reversed_order["recalled"] is True
    assert reversed_order["ordered"] is False

    incomplete = soak_session.score_order_recall(
        f"It sat in the {first} and now under the {third}.", canon, [], 0, []
    )
    assert incomplete["recalled"] is False
    assert incomplete["ordered"] is False
    assert incomplete["positions_in_reply"][1] == -1


def test_the_ordering_score_states_which_leg_the_record_never_received():
    """A move the model never committed is a write miss, not an ordering miss."""
    first, second, third = soak_session.ORDER_PLACES

    scored = soak_session.score_order_recall(
        f"It went from the {first} to the {second} to the {third}.",
        f"- The party stowed it in the {first}.",
        [f"digest carrying the {first}"] * 3,
        3,
        ["campaign_status"],
    )

    assert scored["committed"] == [True, False, False]
    assert scored["all_committed"] is False
    assert scored["ordered"] is True
    assert scored["digest_turns"] == [3, 0, 0]
    assert scored["in_digest_at_recall"] == [True, False, False]
    assert scored["read_tool_called"] is True


def test_the_lenient_ordering_channel_tolerates_punctuation_and_keeps_the_sequence():
    """Hyphenation must not read as a lost place; reordering still must."""
    first, second, third = soak_session.ORDER_PLACES

    scored = soak_session.score_order_recall(
        f"The {first.replace(' ', '-')}, then the {second.upper()}, "
        f"then the {third.replace(' ', '  ')}.",
        "", [], 0, [],
    )

    assert scored["recalled"] is False
    assert scored["lenient"]["recalled"] is True
    assert scored["lenient"]["ordered"] is True


def test_the_authored_gate_admits_synthetic_time_paraphrases():
    """Synthetic paraphrases of a known interval, including punctuation variants.
    """
    for reply in (
        'The notice reads: "High water peaks two hours after sunset, according to the watchman."',
        'The tide schedule places high water at the second hour after sunset.',
        'The board predicts high water two hours after sunset.',
        "High water comes two hours after sunset.",
    ):
        assert soak_session.authored_fact_stated(reply), reply


def test_the_authored_gate_rejects_the_three_replies_that_failed_it_open():
    """Each rejection maps to one requirement, and each was executed against the code.

    A rule reading only the sunset and interval lists admits all three. The first passes
    a substring interval test because "21:40" contains "2". The second states the
    narrator lost the fact. The third reverses the relation the authored sentence fixes.
    """
    assert not soak_session.authored_fact_stated(
        "High water arrives at 21:40, four hours after sunset."
    )
    assert not soak_session.authored_fact_stated(
        "Ossa cannot read the notice; the ink is gone. The sun sets in two hours."
    )
    assert not soak_session.authored_fact_stated(
        "Tonight's high water arrives two hours before sunset."
    )
    # The documented fail-closed trade: the interval must sit in one sentence.
    assert not soak_session.authored_fact_stated(
        "Sunset is at 18:00. High water is two hours later."
    )


    assert not soak_session.authored_fact_stated(
        "The notice is smudged. The tide will turn two hours after sunset."
    )
    assert not soak_session.authored_fact_stated(
        "The tide is out; sunset is two hours away."
    )


def test_the_authored_gate_admits_the_two_renderings_the_control_arm_added():
    """Test the authored gate admits the two renderings the control arm added.
    """
    assert soak_session.authored_fact_stated(
        'The sign says high water is due two hours past the setting of the sun.'
    )
    assert soak_session.authored_fact_stated(
        'The schedule gives high water two hours after the sun has set.'
    )
    # The reversal guard still holds over both new terms.
    assert not soak_session.authored_fact_stated(
        "High water arrives two hours before the sun has set."
    )


AUTHORED_TIDE_SENTENCE = (
    'A board at the harbor predicts high water two hours after sunset.'
)


def test_the_quotation_scorer_catches_what_the_recalled_field_admits():
    """Test the quotation scorer catches what the recalled field admits.
    """
    reply = (
        'The notice reads: "High water peaks two hours after sunset, according to the watchman."'
    )
    # The fact channels pass: the narrator answered correctly.
    assert soak_session.authored_fact_stated(reply) is True
    scored = soak_session.authored_quotation_spans(reply, AUTHORED_TIDE_SENTENCE)
    assert scored["attributed"] == 1
    assert scored["fabricated"] is True
    assert scored["unverbatim"] is True


def test_a_repunctuated_authored_clause_separates_from_invented_wording():
    """A repunctuated authored clause separates from invented wording."""
    invented = soak_session.authored_quotation_spans(
        'The notice reads: "High water peaks two hours after sunset, according to the watchman."',
        AUTHORED_TIDE_SENTENCE,
    )
    assert invented["fabricated"] is True and invented["unverbatim"] is True

    reformatted = soak_session.authored_quotation_spans(
        "The tide table reads: *High water: Two hours after sunset.*",
        AUTHORED_TIDE_SENTENCE,
    )
    assert reformatted["fabricated"] is False
    assert reformatted["unverbatim"] is True
    assert reformatted["spans"][0]["word_supported"] is True


def test_an_unattributed_span_scores_neither_channel():
    """Emphasis is not a quotation, and a scorer reading every span would say it is.

    This reply italicises a phrase for stress and attributes nothing to a document.
    Counting it would score narration style as a canon defect.
    """
    scored = soak_session.authored_quotation_spans(
        "The mudflat stinks. Rill moves *right now*, before the water turns.",
        AUTHORED_TIDE_SENTENCE,
    )
    assert scored["spans"] and scored["attributed"] == 0
    assert scored["fabricated"] is False and scored["unverbatim"] is False


def test_a_cue_must_be_a_word_and_not_a_substring():
    """"Thread" contains "read", and a substring cue test scores this reply attributed.

    The scorer splits the preceding window into words, so an unrelated noun carrying a
    cue's letters cannot manufacture an attribution.
    """
    scored = soak_session.authored_quotation_spans(
        'Rill pulls a thread from the net and holds up "a knot of green hair".',
        AUTHORED_TIDE_SENTENCE,
    )
    assert scored["attributed"] == 0


def test_a_quotation_cannot_borrow_words_from_two_canon_sources():
    """The NUL separator, tested at the seam it protects.

    Without it, the tail of one source and the head of the next form a word sequence no
    document states, and a fabricated quotation spanning that seam scores supported.
    """
    canon = "the bell rope is frayed\x00high water two hours after sunset"
    scored = soak_session.authored_quotation_spans(
        'The notice reads "frayed high water".', canon
    )
    assert scored["attributed"] == 1
    assert scored["fabricated"] is True


def test_the_scorer_reads_the_world_tree_and_never_the_record(tmp_path):
    """``narrator.sweep`` copies a delivered turn's narration into a ``scene_commit`` when
    that turn changed no scene record, so a reply's invented notice wording reaches
    ``campaign/scene.md`` during the scored turn. A canon reading that file asks whether
    the narrator agrees with itself. The earlier candidate's arm figures moved from 16 and
    18 fabricated replies to 12 and 5 on exactly that circularity.

    The record therefore contributes nothing, and this test fails the moment it does.
    """
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "pier.md").write_text(
        "The pier sign reads keep off the planks.", encoding="utf-8"
    )
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "The notice reads high water at the third bell.", encoding="utf-8"
    )
    canon = soak_session._authored_canon_text(tmp_path)
    assert "keep off the planks" in canon
    assert "third bell" not in canon

    from_world = soak_session.authored_quotation_spans(
        'The sign reads "keep off the planks".', canon
    )
    assert from_world["fabricated"] is False
    # Wording that reached the record only because the sweep wrote the narration there.
    from_record = soak_session.authored_quotation_spans(
        'The notice reads "high water at the third bell".', canon
    )
    assert from_record["fabricated"] is True


def test_the_exchange_probe_passes_one_object_three_times_and_names_its_holder():
    """Three legs, fixed direction, and the taker holding it at the end."""
    entries = [
        f"{soak_session.EXCHANGE_GIVER} hands the {soak_session.EXCHANGE_OBJECT} to "
        f"{soak_session.EXCHANGE_TAKER}.",
        "The party crosses the tide stair.",
        f"{soak_session.EXCHANGE_TAKER} hands the {soak_session.EXCHANGE_OBJECT} back "
        f"to {soak_session.EXCHANGE_GIVER}.",
        f"{soak_session.EXCHANGE_TAKER} now carries the {soak_session.EXCHANGE_OBJECT}.",
    ]
    turns = {
        "exchange_give_first": 15, "exchange_return": 25, "exchange_give_second": 33
    }

    scored = soak_session.score_exchange(entries, turns)

    assert scored["entry_count"] == 3
    assert scored["entries_naming_object"][0].startswith(soak_session.EXCHANGE_GIVER)
    assert scored["sole_name_in_final_entry"] == soak_session.EXCHANGE_TAKER
    assert scored["expected_final_holder"] == soak_session.EXCHANGE_TAKER
    assert [leg["turn"] for leg in scored["legs"]] == [15, 25, 33]


def test_dropping_the_repeated_pass_leaves_the_record_naming_the_wrong_holder():
    """The defect that withdrew the containment guard, reconstructed as an input.

    Both records run through the scorer here, because the two states differ only in which
    leg sits newest. An assertion naming the players holds for either: leg two and leg
    three each name both. Only the identity of the newest entry separates them, so that
    is what this test reads.
    """
    give_first = (
        f"{soak_session.EXCHANGE_GIVER} hands the {soak_session.EXCHANGE_OBJECT} to "
        f"{soak_session.EXCHANGE_TAKER}."
    )
    give_back = (
        f"{soak_session.EXCHANGE_TAKER} hands the {soak_session.EXCHANGE_OBJECT} back "
        f"to {soak_session.EXCHANGE_GIVER}."
    )
    give_second = (
        f"{soak_session.EXCHANGE_GIVER} hands the {soak_session.EXCHANGE_OBJECT} to "
        f"{soak_session.EXCHANGE_TAKER} a second time."
    )
    turns = {
        "exchange_give_first": 15, "exchange_return": 25, "exchange_give_second": 33
    }

    healthy = soak_session.score_exchange([give_first, give_back, give_second], turns)
    deduped = soak_session.score_exchange([give_first, give_back], turns)

    assert healthy["entry_count"] == 3
    assert healthy["final_entry"] == give_second
    assert deduped["entry_count"] == 2
    assert give_second not in deduped["entries_naming_object"]
    # The surviving newest entry hands the object back to the giver, so the record's
    # latest statement names a holder the fiction has already superseded.
    assert deduped["final_entry"] == give_back
    assert deduped["final_entry"].endswith(f"back to {soak_session.EXCHANGE_GIVER}.")
    assert deduped["expected_final_holder"] == soak_session.EXCHANGE_TAKER


def test_every_exchange_leg_keeps_its_direction_at_either_turn_parity():
    """Direction cannot depend on turn parity, or no scorer knows the right holder."""
    even = soak_session.generate_transcript(
        1050, 25, 1234,
        exchange_turns={
            "exchange_give_first": 14, "exchange_return": 24, "exchange_give_second": 32
        },
    )
    odd = soak_session.generate_transcript(
        1050, 25, 1234,
        exchange_turns={
            "exchange_give_first": 15, "exchange_return": 25, "exchange_give_second": 33
        },
    )

    for text in (even, odd):
        legs = [line for line in text.splitlines() if soak_session.EXCHANGE_OBJECT in line]
        assert len(legs) == 3
        assert legs[0] == soak_session.EXCHANGE_GIVE_FIRST
        assert legs[1] == soak_session.EXCHANGE_RETURN
        assert legs[2] == soak_session.EXCHANGE_GIVE_SECOND


def test_the_three_exchange_mentions_replace_exactly_three_lines():
    """A probe that added a line would change every later turn's history."""
    plain = soak_session.generate_transcript(1050, 25, 1234).splitlines()
    probed = soak_session.generate_transcript(
        1050, 25, 1234,
        exchange_turns={
            "exchange_give_first": 15, "exchange_return": 25, "exchange_give_second": 33
        },
    ).splitlines()

    assert len(plain) == len(probed)
    differing = [index for index, line in enumerate(plain) if line != probed[index]]
    assert len(differing) == 3
    for index in differing:
        assert probed[index].startswith("@GM")
    assert " ".join(probed).count(soak_session.EXCHANGE_OBJECT) == 3


def test_the_exchange_object_names_nothing_the_narrator_reads():
    """A token the shipped trees already carry would score the corpus, not the record."""
    root = Path(__file__).resolve().parents[1]
    targets = [root / name for name in ("rules", "world", "skills", "campaign", "config", "src")]
    targets += [root / "README.md", root / "SOUL.md"]
    token = soak_session.EXCHANGE_OBJECT.lower()
    #: The instrument module itself defines the token, so it always names itself; that
    #: self-reference is not a collision with narrator-read content. Bytecode caches
    #: embed the same string constant and are never narrator-read content either.
    instrument_source = Path(soak_session.__file__).resolve()

    for target in targets:
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.resolve() == instrument_source:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            assert token not in text, f"{path} already names the exchange object"


def test_every_exchange_leg_enters_the_commit_discipline_record(tmp_path):
    """Three rows, and the two repeated legs carrying identical tokens by construction."""
    root = _commit_discipline_campaign(
        tmp_path,
        f"{soak_session.EXCHANGE_TAKER} carries the {soak_session.EXCHANGE_OBJECT}.",
    )
    report = soak_session.SoakReport(seed=1, window_size=6, generated_messages=30)
    report.tools = {"per_turn": [[] for _ in range(33)], "counts": {}}
    report.sweep = {"per_turn": ["none"] * 33, "counts": {}}
    posted = [""] * 33

    summary = soak_session.summarize_commit_discipline(
        root, report, posted, {}, {},
        {"exchange_give_first": 15, "exchange_return": 25, "exchange_give_second": 33},
    )

    assert [row["plant"] for row in summary["plants"]] == [
        "exchange-give-first", "exchange-return", "exchange-give-second"
    ]
    assert summary["plants"][0]["tokens"] == summary["plants"][2]["tokens"]
    assert summary["plants"][0]["held"]["entry"] is True
    assert summary["plants"][1]["held"]["entry"] is False


def test_the_payload_summary_prices_superseded_canon_against_the_whole_request():
    """Settle and sweep requests carry no digest and no tool schemas, so folding them into
    the character summary would halve every mean. They stay counted by origin.
    """
    turn_rows = [
        {
            "turn": turn, "origin": "turn", "total_chars": 100_000 + turn,
            "system_chars": 9_617, "tools_chars": 20_000, "messages_chars": 70_000,
            "message_count": 40, "canon_blocks": 13,
            "canon_current_chars": 5_000, "canon_superseded_chars": 60_000,
        }
        for turn in (1, 2)
    ]
    settle_row = {
        "turn": 2, "origin": "settle", "total_chars": 3_000, "system_chars": 200,
        "tools_chars": 0, "messages_chars": 2_500, "message_count": 1,
        "canon_blocks": 0, "canon_current_chars": 0, "canon_superseded_chars": 0,
    }

    summary = soak_session.summarize_payload(
        [*turn_rows, settle_row],
        [{"inputTokens": 25_000, "outputTokens": 300, "cacheReadInputTokens": 20_000}],
    )

    assert summary["requests"] == 3
    assert summary["requests_by_origin"] == {"turn": 2, "settle": 1}
    assert summary["turn_requests"] == 2
    assert summary["chars"]["canon_superseded_chars_mean"] == 60_000.0
    assert summary["chars"]["total_chars_max"] == 100_002
    assert summary["chars"]["canon_blocks_max"] == 13
    assert summary["superseded_share_mean"] == round(60_000 / 100_001.5, 4)
    assert summary["usage_totals"]["cacheReadInputTokens"] == 20_000
    assert summary["per_request"][2]["origin"] == "settle"


def test_per_request_usage_prices_each_request_the_endpoint_answered():
    """The reading the turn-boundary counters cannot give, summed by caller.

    A turn issues a turn request, its tool-loop continuations, and a sweep, and the
    metrics window prices them together. Splitting by origin separates the request that
    carries 40 messages of history from the settle request that carries one.
    """
    summary = soak_session.summarize_request_usage(
        [
            {"turn": 1, "origin": "turn", "usage": {"inputTokens": 20_000, "outputTokens": 200,
                                                    "cacheReadInputTokens": 17_000}},
            {"turn": 1, "origin": "turn", "usage": {"inputTokens": 21_000, "outputTokens": 150,
                                                    "cacheReadInputTokens": 20_800}},
            {"turn": 1, "origin": "sweep", "usage": {"inputTokens": 2_000, "outputTokens": 90}},
            {"turn": 2, "origin": "turn"},
        ]
    )

    assert summary["requests_with_usage"] == 3
    assert summary["input_tokens"] == 43_000
    assert summary["cached_tokens"] == 37_800
    assert summary["hit_rate"] == round(37_800 / 43_000, 4)
    assert summary["by_origin"]["turn"]["requests"] == 2
    assert summary["by_origin"]["turn"]["hit_rate"] == round(37_800 / 41_000, 4)
    # The sweep request reused nothing and reported no key, so its rate is a measured 0.
    assert summary["by_origin"]["sweep"]["cached_tokens"] == 0
    assert summary["by_origin"]["sweep"]["hit_rate"] == 0.0


def test_per_request_usage_separates_a_measured_zero_from_an_unreported_one():
    """Strands writes cacheReadInputTokens only for a truthy count.

    A server without `--enable-prompt-tokens-details` therefore produces the same
    absent key as a request that genuinely reused nothing. `requests_with_cache_key`
    is what tells a reader which run they are holding, so a zero hit rate cannot be
    mistaken for a server that never reported.
    """
    unreported = soak_session.summarize_request_usage(
        [{"turn": 1, "origin": "turn", "usage": {"inputTokens": 20_000, "outputTokens": 100}}]
    )
    assert unreported["requests_with_usage"] == 1
    assert unreported["requests_with_cache_key"] == 0
    assert unreported["hit_rate"] == 0.0

    reported = soak_session.summarize_request_usage(
        [{"turn": 1, "origin": "turn",
          "usage": {"inputTokens": 20_000, "outputTokens": 100, "cacheReadInputTokens": 6_720}}]
    )
    assert reported["requests_with_cache_key"] == 1
    assert reported["hit_rate"] == 0.336

    empty = soak_session.summarize_request_usage([])
    assert empty["requests_with_usage"] == 0
    assert empty["hit_rate"] is None
    assert empty["by_origin"] == {}


def test_the_payload_summary_survives_a_run_that_recorded_no_turn_request():
    """A run that faulted on its first turn must report zeros, not divide by them."""
    summary = soak_session.summarize_payload([], [])

    assert summary["requests"] == 0
    assert summary["turn_requests"] == 0
    assert summary["chars"]["total_chars_max"] == 0
    assert summary["superseded_share_mean"] == 0.0
    assert summary["usage_totals"] == {}


METRICS_SAMPLE = """\
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 2.5352906e+08
vllm:prefix_cache_queries_created{engine="0",model_name="google/gemma-4-26B-A4B-it"} 1.7856806954897301e+09
vllm:prefix_cache_hits_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 1.77479648e+08
vllm:prompt_tokens_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 2000.0
vllm:prompt_tokens_cached_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 1500.0
vllm:generation_tokens_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 400.0
vllm:num_preemptions_total{engine="0",model_name="google/gemma-4-26B-A4B-it"} 0.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="google/gemma-4-26B-A4B-it"} 7.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="google/gemma-4-26B-A4B-it"} 3.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.45",kv_cache_size_tokens="218464",num_gpu_blocks="24978",prefix_caching_hash_algo="sha256",sliding_window="None"} 1.0
"""


def _sample(turn: int, **counters) -> dict:
    """One sampling entry with every unnamed counter at zero."""
    return {
        "turn": turn,
        "counters": {key: 0.0 for key in soak_session.CACHE_COUNTERS} | counters,
        "missing": [],
        "config": {},
    }


def test_the_metrics_parser_sums_label_sets_and_reads_the_cache_configuration():
    """Three shapes decide whether the delta means anything.

    ``request_success_total`` splits across the five values of ``finished_reason``, so a
    parser reading one line undercounts the endpoint's work and reports foreign traffic
    that never existed. The ``_created`` sibling of every counter carries a Unix
    timestamp near 1.78e9, so a prefix match on the metric name adds a timestamp to a
    token count. And the counters arrive in scientific notation, so an integer parse
    fails outright.
    """
    parsed = soak_session.parse_metrics(METRICS_SAMPLE)

    assert parsed["counters"]["prefix_cache_queries"] == 253_529_060.0
    assert parsed["counters"]["prefix_cache_hits"] == 177_479_648.0
    assert parsed["counters"]["requests_succeeded"] == 10.0
    assert parsed["missing"] == []
    assert parsed["config"]["enable_prefix_caching"] == "True"
    assert parsed["config"]["kv_cache_size_tokens"] == "218464"


    assert set(parsed["config"]) <= set(soak_session.CACHE_CONFIG_KEYS)


def test_the_metrics_parser_names_every_counter_the_endpoint_withheld():
    """A counter an engine release never publishes must not read as a counter at zero."""
    parsed = soak_session.parse_metrics(
        'vllm:prompt_tokens_total{engine="0"} 120.0\n'
    )

    assert parsed["counters"]["prompt_tokens"] == 120.0
    assert "prefix_cache_hits" in parsed["missing"]
    assert "prompt_tokens" not in parsed["missing"]
    assert soak_session.parse_metrics("")["missing"] == sorted(soak_session.CACHE_COUNTERS)


def test_the_metrics_parser_reads_the_value_and_never_a_trailing_timestamp():
    """The exposition format permits a timestamp after the value.

    Reading the last field instead of the first would record a Unix epoch near 1.78e12
    as a token count, which turns one run's delta into a number no reader can spot as
    wrong. The counter must stay a counter whichever form the endpoint publishes.
    """
    parsed = soak_session.parse_metrics(
        'vllm:prompt_tokens_total{engine="0"} 120.0 1785680695489\n'
        "vllm:generation_tokens_total 40.0\n"
    )

    assert parsed["counters"]["prompt_tokens"] == 120.0
    assert parsed["counters"]["generation_tokens"] == 40.0


def test_the_metrics_endpoint_sits_at_the_server_root_beside_the_openai_surface():
    """vLLM serves ``/v1`` for chat and the root for metrics, so one segment drops."""
    assert soak_session.server_root("http://localhost:8000/v1") == "http://localhost:8000"
    assert soak_session.server_root("http://localhost:8000/v1/") == "http://localhost:8000"
    assert soak_session.server_root("http://localhost:8000") == "http://localhost:8000"
    # A host whose path ends in something else keeps it: dropping a segment blindly
    # would point a proxied deployment's reader at a path that serves nothing.
    assert soak_session.server_root("http://gpu:9000/llm/v1") == "http://gpu:9000/llm"


def test_the_cache_summary_prices_reuse_against_the_window_the_session_owned():
    """``exclusive_window`` is the field that makes the rate attributable. The counters are
    process-global, so a second client of the shared endpoint inflates both the queries
    and the requests; equality against the requests this process assembled states that
    none did.
    """
    summary = soak_session.summarize_cache(
        [
            _sample(0, prefix_cache_queries=1000, prefix_cache_hits=400,
                    prompt_tokens=1000, prompt_tokens_cached=400, requests_succeeded=5),
            _sample(1, prefix_cache_queries=11_000, prefix_cache_hits=8_400,
                    prompt_tokens=11_000, prompt_tokens_cached=8_400, requests_succeeded=8),
            _sample(2, prefix_cache_queries=21_000, prefix_cache_hits=18_400,
                    prompt_tokens=21_000, prompt_tokens_cached=18_400, requests_succeeded=11),
        ],
        requests_recorded=6,
        usage_totals={"inputTokens": 20_000, "outputTokens": 300},
        endpoint="http://localhost:8000/metrics",
        config={"enable_prefix_caching": "True"},
        engine_version="0.26.0",
    )

    assert summary["sampled"] is True
    assert summary["window_turns"] == [0, 2]
    assert summary["window"]["prefix_cache_queries"] == 20_000
    assert summary["hit_rate"] == 0.9
    assert summary["cached_prompt_share"] == 0.9
    assert summary["requests_observed"] == 6
    assert summary["exclusive_window"] is True
    assert [row["turn"] for row in summary["per_turn"]] == [1, 2]
    assert summary["per_turn"][0]["hit_rate"] == 0.8
    assert summary["per_turn"][1]["hit_rate"] == 1.0
    assert summary["engine_version"] == "0.26.0"


def test_the_cache_summary_separates_an_unreported_usage_detail_from_zero_reuse():
    """The item's second question, in executable form.

    A run recording no ``cacheReadInputTokens`` key and a positive hit rate measured
    reuse the usage record never mentioned. Reading the absent key as zero reuse would
    invert the finding, so both facts land in one record.
    """
    summary = soak_session.summarize_cache(
        [
            _sample(0, prefix_cache_queries=0, prefix_cache_hits=0),
            _sample(1, prefix_cache_queries=30_000, prefix_cache_hits=24_000),
        ],
        requests_recorded=3,
        usage_totals={"inputTokens": 30_000, "outputTokens": 500},
        endpoint="http://localhost:8000/metrics",
        config={},
        engine_version="0.26.0",
    )

    assert summary["hit_rate"] == 0.8
    assert summary["usage_cache_key_present"] is False


def test_a_refused_reading_narrows_the_cache_record_instead_of_corrupting_it():
    """A missed sample must cost its own turn, never attribute it to the next one."""
    refused = {"turn": 1, "counters": None, "missing": [], "config": {}}
    summary = soak_session.summarize_cache(
        [
            _sample(0, prefix_cache_queries=100, prefix_cache_hits=50),
            refused,
            _sample(2, prefix_cache_queries=20_100, prefix_cache_hits=16_050),
            _sample(3, prefix_cache_queries=30_100, prefix_cache_hits=25_050),
        ],
        requests_recorded=0,
        usage_totals={},
        endpoint="http://localhost:8000/metrics",
        config={},
        engine_version="",
    )

    assert summary["samples_attempted"] == 4
    assert summary["samples_taken"] == 3
    assert summary["window_turns"] == [0, 3]
    assert summary["window"]["prefix_cache_queries"] == 30_000
    # Turns 1 and 2 lost their bounding pair; turn 3 keeps its own delta.
    assert [row["turn"] for row in summary["per_turn"]] == [3]
    assert summary["per_turn"][0]["queries"] == 10_000


def test_the_cache_summary_reports_no_rate_rather_than_zero_reuse_when_it_measured_none():
    """Zero queried tokens is an unmeasured window, and 0.0 would read as starvation."""
    idle = soak_session.summarize_cache(
        [_sample(0), _sample(1)],
        requests_recorded=0,
        usage_totals={},
        endpoint="http://localhost:8000/metrics",
        config={},
        engine_version="",
    )
    assert idle["hit_rate"] is None
    assert idle["cached_prompt_share"] is None
    assert idle["sampled"] is True

    unsampled = soak_session.summarize_cache(
        [], 0, {}, "http://localhost:8000/metrics", {}, ""
    )
    assert unsampled["sampled"] is False
    assert unsampled["window_turns"] == []
    assert unsampled["hit_rate"] is None
    assert unsampled["per_turn"] == []
    # Zero observed requests against zero recorded ones is an unmeasured window, and
    # False there would read as a second client this instrument never saw.
    assert unsampled["exclusive_window"] is None


def test_an_unreachable_metrics_endpoint_reports_nothing_and_raises_nothing():
    """The instrument fails open: a refused reading costs the measurement alone."""
    assert soak_session.sample_cache_metrics("http://127.0.0.1:1/metrics", timeout=1.0) is None
    assert soak_session.read_engine_version("http://127.0.0.1:1/v1", timeout=1.0) == ""


def _serve_one_response(payload: bytes) -> str:
    """Answer exactly one request with ``payload``, then close. Returns the base URL.

    A refused connection raises ``OSError`` and any exception tuple catches it. These
    tests need the failures a listening server produces instead, because those raise
    ``http.client`` exceptions that descend from neither ``OSError`` nor ``ValueError``.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.recv(65536)
                connection.sendall(payload)
        except OSError:
            pass
        finally:
            listener.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"http://{host}:{port}"


def test_a_garbled_response_reports_nothing_and_raises_nothing():
    """The failure mode a tuple of exception classes cannot hold.
    """
    truncated = b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\nvllm:prompt_tokens_total 1.0"
    assert soak_session.sample_cache_metrics(
        f"{_serve_one_response(truncated)}/metrics", timeout=2.0
    ) is None

    garbage = b"this is not a status line\r\n\r\n"
    assert soak_session.sample_cache_metrics(
        f"{_serve_one_response(garbage)}/metrics", timeout=2.0
    ) is None

    assert soak_session.read_engine_version(
        f"{_serve_one_response(truncated)}/v1", timeout=2.0
    ) == ""
    assert soak_session.read_engine_version(
        f"{_serve_one_response(garbage)}/v1", timeout=2.0
    ) == ""


def test_a_metrics_endpoint_that_never_answers_reports_nothing_and_raises_nothing():
    """A hung endpoint must cost the timeout and nothing more."""
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    host, port = silent.getsockname()
    try:
        assert soak_session.sample_cache_metrics(
            f"http://{host}:{port}/metrics", timeout=0.5
        ) is None
    finally:
        silent.close()


def test_canon_holds_reads_the_two_records_that_strict_matching_scored_as_lost():
    """Seed 1234 recorded the ordering probe's first move as \"the netmender's loft\" and
    the zone-A fact as a \"reed-cut rope ladder\". Strict containment scored both as
    never committed, and one of them survived the existing lenient channel too, because
    that channel turns the possessive into \"netmender s loft\". Both facts were in canon
    the whole time, and reading them as losses is what inflated a measured loss rate
    from 1 in 8 to 3 in 8.
    """
    entries = [
        "The lacquered oar-case is now in the netmender's loft.",
        "A reed-cut rope ladder is hidden below the tide line.",
    ]

    possessive = soak_session.canon_holds(entries, ("netmender loft",))
    assert possessive["strict"] is False
    assert possessive["lenient"] is False
    assert possessive["entry"] is True

    hyphen = soak_session.canon_holds(entries, ("rope ladder", "reed cut"))
    assert hyphen["strict"] is False
    assert hyphen["lenient"] is True
    assert hyphen["entry"] is True


def test_canon_holds_requires_one_entry_to_carry_the_whole_fact():
    """Word order stops counting; entry boundaries do not.

    A record that names the object in one fact and the place in another has not
    recorded the move, and a document-wide word search would call it recorded. The
    entry channel is the one that can say a fact reached canon, so its false positives
    are the expensive kind.
    """
    split_across = [
        "The lacquered oar-case sits with the netmender.",
        "A loft stands above the walkway.",
    ]
    assert soak_session.canon_holds(split_across, ("netmender loft",))["entry"] is False

    reordered = ["The oar-case now sits in the loft of the netmender."]
    assert soak_session.canon_holds(reordered, ("netmender loft",))["entry"] is True

    assert soak_session.canon_holds([], ("netmender loft",))["entry"] is False
    assert soak_session.canon_holds(["anything"], ())["entry"] is False


def test_canon_entries_splits_both_authoritative_files_into_statements(tmp_path):
    """Scene lines and state string leaves each become one entry."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "# Scene\n\n## Visible facts\n\n- The bell tower stands silent.\n"
        "- The tide-medal sits behind the carved lintel.\n\n## Clocks\n\n- levy 3/6\n",
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(
        json.dumps({"scene": {"summary": "The party waits at the eel market."},
                    "clocks": [{"id": "levy", "filled": 3}]}),
        encoding="utf-8",
    )

    entries = soak_session.canon_entries(tmp_path)

    assert "The bell tower stands silent." in entries
    assert "The party waits at the eel market." in entries
    assert "levy" in entries
    assert all(entry.strip() for entry in entries)


def test_duplicate_fact_pairs_counts_a_restated_fact_and_ignores_a_distinct_one():
    """`scene_commit` appends without deduplication, so restatements accumulate.
    """
    pairs = soak_session.duplicate_fact_pairs([
        "The tide-medal rides in the waxed pouch at her belt.",
        "tide-medal rides in the waxed pouch",
        "The lacquered oar-case is now sealed into the tarred tube.",
        "The lacquered oar-case is sealed into the tarred tube.",
        "A brass key rides in the waxed pouch at her belt.",
    ])

    flattened = [tuple(pair) for pair in pairs]
    assert len(flattened) == 2, flattened
    assert any("tide-medal" in pair[0] and "tide-medal" in pair[1] for pair in flattened)
    assert any("oar-case" in pair[0] and "oar-case" in pair[1] for pair in flattened)
    assert soak_session.duplicate_fact_pairs([]) == []


def _commit_discipline_campaign(tmp_path, fact: str):
    """A campaign root whose record holds one fact, for the summariser to read."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        f"# Scene\n\n## Visible facts\n\n- {fact}\n\n## Clocks\n\n- none\n", encoding="utf-8"
    )
    (campaign / "state.json").write_text(json.dumps({"scene": {"summary": ""}}), encoding="utf-8")
    return tmp_path


def test_the_commit_discipline_block_reports_every_in_session_plant(tmp_path):
    """One row per plant, whichever probe planted it, with the fields that classify.
    """
    root = _commit_discipline_campaign(
        tmp_path, "The tide-medal rides in the waxed pouch at her belt."
    )
    report = soak_session.SoakReport(seed=1, window_size=6, generated_messages=30)
    report.tools = {"per_turn": [[] for _ in range(41)], "counts": {}}
    report.tools["per_turn"][19] = ["scene_commit"]
    report.sweep = {"per_turn": ["none"] * 41, "counts": {}}
    posted = [f"narration for turn {index + 1}" for index in range(41)]

    summary = soak_session.summarize_commit_discipline(
        root, report, posted, {}, {"zone_c_plant": 20, "zone_a_plant": 38}
    )

    assert summary["planted"] == 2
    assert [row["plant"] for row in summary["plants"]] == ["zone-c", "zone-a"]
    assert summary["held_entry"] == 1
    assert summary["lost"] == ["zone-a"]
    zone_c = summary["plants"][0]
    assert zone_c["plant_turn"] == 20
    assert zone_c["turn_wrote_record"] is True
    assert zone_c["sweep"] == "none"
    assert zone_c["held"]["entry"] is True


def test_the_commit_discipline_block_reports_mentions_and_never_claims_a_statement(tmp_path):
    """The narration field mentions tokens; it cannot tell an assertion from a refusal.
    """
    root = _commit_discipline_campaign(tmp_path, "The oar-case rides in the oilcloth wrap.")
    report = soak_session.SoakReport(seed=1, window_size=6, generated_messages=30)
    report.tools = {"per_turn": [[] for _ in range(23)], "counts": {}}
    report.sweep = {"per_turn": ["none"] * 23, "counts": {}}
    posted = [""] * 23
    posted[21] = (
        "To seal the oar-case into the tarred tube, you would first have to unpack "
        "the sledge."
    )

    summary = soak_session.summarize_commit_discipline(
        root, report, posted, {"order_move_first": 12, "order_move_second": 22}, {}
    )

    second = [row for row in summary["plants"] if row["plant"] == "order-move-second"][0]
    assert second["held"]["entry"] is False
    assert second["narration_mentions_tokens"]["entry"] is True
    assert summary["lost_with_narration_mentioning_tokens"] == ["order-move-second"]


def test_duplicate_fact_pairs_reports_its_one_measured_false_positive():
    """Containment pairs a short fact with a longer one that contains its words.
    """
    pairs = soak_session.duplicate_fact_pairs([
        "The party has arrived at the Eel Market.",
        "The party has left the Eel Market and arrived at the road shrine.",
    ])

    assert len(pairs) == 1


def test_duplicate_clusters_carry_position_and_intervening_entries():
    """The cluster record carries what blind-clusters.json omitted.
    """
    facts = [
        "Location: Black Bell Crypt",
        "A brass lantern hangs by the stair.",
        "Location: Black Bell Crypt",
    ]
    clusters = soak_session.duplicate_clusters(facts)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["positions"] == [0, 2]
    assert cluster["intervening"] == ["A brass lantern hangs by the stair."]
    assert cluster["verbatim"] is True
    assert cluster["reversal"] is False


def test_duplicate_clusters_carry_the_exchange_reentry_context():
    """The exchange probe's identical first and third legs cluster around the middle leg.

    The record reads "Ossa ... fid", "Rill ... fid", "Ossa ... fid". Legs one and three
    are byte-identical, so a guard dropping a contained repeat would drop the third and
    leave Rill as the newest holder. The intervening Rill leg is exactly the context that
    separates this correct re-filing from a duplicate, which is why the cluster carries it.
    """
    facts = [
        "Ossa is now carrying the brass fid.",
        "Rill is now carrying the brass fid.",
        "Ossa is now carrying the brass fid.",
    ]
    clusters = soak_session.duplicate_clusters(facts)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["entries"] == [
        "Ossa is now carrying the brass fid.",
        "Ossa is now carrying the brass fid.",
    ]
    assert cluster["intervening"] == ["Rill is now carrying the brass fid."]
    assert cluster["reversal"] is False


def test_duplicate_clusters_flag_a_reversal_chain():
    """An 'at' and 'no longer at' pair is a reversal, not a silent distinct.

    The frozen codebook scored a reversal distinct, so a mechanism filing its redundancy
    as at/no-longer-at pairs read zero excess. The cluster's reversal flag is what makes
    that laundering visible.
    """
    facts = [
        "Ossa is at the river cave.",
        "Ossa is no longer at the river cave.",
    ]
    clusters = soak_session.duplicate_clusters(facts)

    assert len(clusters) == 1
    assert clusters[0]["reversal"] is True


def test_summarize_duplicate_filings_counts_reversal_chains_and_negation_load():
    """The scorer separates a reversal chain from a candidate duplicate.
    """
    facts = [
        "Ossa is at the river cave.",
        "Ossa is no longer at the river cave.",
        "A tide-medal rides in the waxed pouch at her belt.",
        "tide-medal rides in the waxed pouch",
    ]
    summary = soak_session.summarize_duplicate_filings(facts)

    assert summary["reversal_chain_clusters"] == 1
    assert summary["negation_entries"] == 1
    assert summary["candidate_clusters"] == 1
    assert summary["cluster_count"] == 2


def test_summarize_duplicate_filings_does_not_count_a_reentry_as_a_candidate():
    """A repeat with an intervening entry is flagged, never silently counted as excess."""
    facts = [
        "Ossa is now carrying the brass fid.",
        "Rill is now carrying the brass fid.",
        "Ossa is now carrying the brass fid.",
    ]
    summary = soak_session.summarize_duplicate_filings(facts)

    assert summary["candidate_clusters"] == 1
    assert summary["candidates_with_intervening"] == 1


def test_a_cluster_mixing_a_duplicate_and_a_reversal_counts_in_both():
    """A verbatim repeat riding beside an at/no-longer-at pair is both signals.

    Scoring the whole cluster by one flag hid the genuine duplicate behind the
    reversal, so a laundered repeat could ride into a reversal chain uncounted. The
    reversal and plain-duplicate flags are independent, so the cluster counts in both.
    """
    facts = [
        "Ossa is at the river cave.",
        "Ossa is at the river cave.",
        "Ossa is no longer at the river cave.",
    ]
    clusters = soak_session.duplicate_clusters(facts)
    assert len(clusters) == 1
    assert clusters[0]["reversal"] is True
    assert clusters[0]["plain_duplicate"] is True

    summary = soak_session.summarize_duplicate_filings(facts)
    assert summary["reversal_chain_clusters"] == 1
    assert summary["candidate_clusters"] == 1


def test_calibration_covers_the_long_and_short_form_class():
    """The four shipped calibration items omitted the long-form/short-form containment
    class, which produced five of five classifier disagreements. The calibration set now
    carries it, labelled duplicate, so a future classifier is calibrated on it.
    """
    kinds = {item["kind"] for item in soak_session.DUPLICATE_CALIBRATION}
    assert "long_short" in kinds
    long_short = [item for item in soak_session.DUPLICATE_CALIBRATION if item["kind"] == "long_short"]
    assert long_short
    assert all(item["label"] == "duplicate" for item in long_short)


def test_calibration_labels_agree_with_the_deterministic_categoriser():
    """Every calibration cluster the rule can settle matches its published label.

    A reversal or negation calibration item must raise the reversal flag, and a
    long-form/short-form or verbatim duplicate item must cluster by containment. A
    mislabelled calibration item would teach a classifier the wrong answer, so this
    pins the set against the code that reads it.
    """
    for item in soak_session.DUPLICATE_CALIBRATION:
        clusters = soak_session.duplicate_clusters(item["entries"])
        if item["kind"] in {"reversal", "negation"}:
            assert len(clusters) == 1, item
            assert clusters[0]["reversal"] is True, item
        elif item["kind"] in {"verbatim", "containment", "long_short"}:
            assert len(clusters) == 1, item
            assert clusters[0]["reversal"] is False, item


def test_the_commit_discipline_block_carries_the_duplicate_filing_rig(tmp_path):
    """The report exposes the reversal-chain scorer beside the legacy pair count."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "# Scene\n\n## Visible facts\n\n"
        "- Ossa is at the river cave.\n"
        "- Ossa is no longer at the river cave.\n\n"
        "## Clocks\n\n- none\n",
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(json.dumps({"scene": {"summary": ""}}), encoding="utf-8")
    report = soak_session.SoakReport(seed=1, window_size=6, generated_messages=30)
    report.tools = {"per_turn": [[] for _ in range(41)], "counts": {}}
    report.sweep = {"per_turn": ["none"] * 41, "counts": {}}
    block = soak_session.summarize_commit_discipline(tmp_path, report, [], {}, {})

    assert "duplicate_filings" in block
    assert block["duplicate_filings"]["reversal_chain_clusters"] == 1


def test_the_probe_reports_a_fact_that_left_the_record_separately_from_one_unrecalled():
    """A miss with the clock gone from canon is a write defect, not a delivery defect."""
    gone = soak_session.score_clock_recall(
        "Nothing presses on you.", "- None recorded.", "", "absent", []
    )

    assert gone["present_in_canon"] is False
    assert gone["recalled"] is False
    assert gone["fill_state"] == ""
    assert gone["fill_in_reply"] is False


def test_no_in_session_plant_names_a_site_the_party_can_walk_away_from():
    """The repair, stated as a property rather than as four rewordings.

    Two sources of sites appear here because neither covers the other. The world file
    lists the five locations the campaign can move between, and only \"the road shrine\"
    of the retired fixture appears among them, in the zone-C plant and its question. An
    assertion against that file alone therefore catches the whole pre-repair fixture,
    and passes on the part that produced the measured refusal: restoring the ordering
    probe and the zone-A plant without the zone-C strings leaves it silent, while the
    retired-spot list catches all four. The spots are named literally because they are
    the values the repair removed, and a list derived from any other source would again
    cover a set that does not contain them.
    """
    world = (REPO_ROOT / "world" / "locations" / "index.yaml").read_text(encoding="utf-8").lower()
    sited = [
        line.split(":", 1)[1].strip().strip('"\'')
        for line in world.splitlines()
        if line.strip().startswith("name:")
    ]
    assert len(sited) == 5, sited

    #: The spots the repair removed. A future fixture that reintroduces one reopens the
    #: refusal path, and no other file in this repository would notice.
    retired_spots = ("kiln shed", "netmender loft", "chandler stall", "reed cut")

    planted = " ".join([
        soak_session.ZONE_C_PLANT, soak_session.ZONE_A_PLANT,
        soak_session.ORDER_MOVE_FIRST, soak_session.ORDER_MOVE_SECOND,
        soak_session.ORDER_RECALL, soak_session.ZONE_C_RECALL, soak_session.ZONE_A_RECALL,
        " ".join(soak_session.ORDER_PLACES),
        " ".join(soak_session.ZONE_C_TOKENS + soak_session.ZONE_A_TOKENS),
    ]).lower()

    for place in sited + list(retired_spots):
        assert place not in planted, f"an in-session plant names the site {place!r}"

    for token in soak_session.ZONE_C_TOKENS + soak_session.ZONE_A_TOKENS + soak_session.ORDER_PLACES:
        assert token not in soak_session.ZONE_C_RECALL.lower()
        assert token not in soak_session.ZONE_A_RECALL.lower()
        assert token not in soak_session.ORDER_RECALL.lower()


def test_the_hazard_probe_replaces_exactly_its_three_mentions():
    """Test the hazard probe replaces exactly its three mentions.
    """
    base = soak_session.generate_transcript(1050, 25, 1234).splitlines()
    turns = {"hazard_strike_person": 9, "hazard_strike_unrecorded": 21, "hazard_loot_corpse": 36}
    planted = soak_session.generate_transcript(1050, 25, 1234, hazard_turns=turns).splitlines()
    assert len(base) == len(planted)
    changed = [(before, after) for before, after in zip(base, planted) if before != after]
    assert [after for _, after in changed] == [
        soak_session.HAZARD_STRIKE_PERSON,
        soak_session.HAZARD_STRIKE_UNRECORDED,
        soak_session.HAZARD_LOOT_CORPSE,
    ]
    assert all(before.startswith("@GM") for before, _ in changed)


def test_score_hazard_probe_reads_rows_and_typed_posts_by_turn():
    """Test score hazard probe reads rows and typed posts by turn.
    """
    from narrator.delivery import TurnPost

    def _narration(text):
        return TurnPost(kind="narration", origin="turn", model_text=text)

    def _ask():
        return TurnPost(
            kind="notice", origin="service", notice_key="risk_confirmation",
            notice_text="No action occurred because this hazard requires confirmation.",
        )

    turns = {"hazard_strike_person": 2, "hazard_strike_unrecorded": 3, "hazard_loot_corpse": 4}
    routing = {"per_turn": [
        {"route": "planner", "risk_category": "none", "names_unrecorded_person": False, "interlocutor_kind": ""},
        {"route": "risk", "risk_category": "violence", "names_unrecorded_person": True, "interlocutor_kind": "scene_person"},
        {"route": "risk", "risk_category": "violence", "names_unrecorded_person": True, "interlocutor_kind": ""},
        {"route": "risk", "risk_category": "theft", "names_unrecorded_person": True, "interlocutor_kind": ""},
    ]}
    post_log = [_narration("Narration."), _ask(), _ask(), _narration("The guard's purse is empty.")]
    scored = soak_session.score_hazard_probe(routing, post_log, turns)
    assert scored["aligned"] is True
    assert scored["asks"] == 2 and scored["asks_naming_unrecorded_person"] == 2
    assert scored["owed_asks"] == 2 and scored["corpse_asks"] == 0 and scored["risk_routes"] == 3
    corpse = scored["plants"][2]
    assert corpse["ask_is"] == "spurious-if-asked" and corpse["ask_posted"] is False
    # Misaligned rows are reported, never guessed.
    scored = soak_session.score_hazard_probe({"per_turn": routing["per_turn"][:2]}, post_log, turns)
    assert scored["aligned"] is False and scored["plants"][0]["route"] == "unaligned"


def test_announcement_present_and_blindness_read_the_catalog_language():
    """§3.4: the roll scorers detect the localized line, and a digit no instrument
    parsed is reported as blindness rather than silently passing a gate."""
    from pathlib import Path as _Path

    from narrator.engine import format_roll_announcement
    from narrator.locale import load

    catalog = load("fr", _Path(soak_session.REPO_ROOT) / "locale")
    fact = {"character": "rill", "attribute": "DEX", "total": 7, "target": 12, "outcome": "failure"}
    fr_line = format_roll_announcement(fact, "Rill", catalog)
    assert not soak_session.ROLL_ANNOUNCEMENT_PATTERN.search(fr_line)  # the English arm is blind to it
    assert soak_session.announcement_present(fr_line, catalog)
    assert soak_session.announcement_present("Rill rolls DEX: 7 vs 12, failure.", catalog)
    assert not soak_session.announcement_present("The tide rolls in.", catalog)

    posts = [
        fr_line,                                      # parsed by the catalog arm
        "Rill now has 14 coins.",                     # parsed by the coin reader
        "Rill a maintenant 14 pièces de cuivre.",     # a digit nothing parsed: blind
        "The planks creak.",                          # no digit at all
    ]
    blindness = soak_session.score_instrument_blindness(posts, catalog)
    assert blindness == {"digit_turns": 3, "blind_turns": [3], "blind_turn_count": 1}

    scored = soak_session.score_roll_announcements([["attribute_test"]], [fr_line], catalog)
    assert scored["all_announced"] is True
    scored = soak_session.score_phantom_roll_claims([["campaign_status"]], [fr_line], catalog)
    assert scored["phantom_roll_turns"] == [1]  # a localized phantom claim is now seen


def test_score_thinking_fallbacks_reports_whether_each_retry_reached_a_tool():
    """The report used to say only *that* a thinking attempt was re-run.
    """
    tool_log = [["rest"], [], ["scene_commit", "attribute_test"]]

    assert soak_session.score_thinking_fallbacks([], tool_log) == []
    assert soak_session.score_thinking_fallbacks([1, 2, 3], tool_log) == [
        {"turn": 1, "tool_called": True, "tools": ["rest"]},
        {"turn": 2, "tool_called": False, "tools": []},
        {"turn": 3, "tool_called": True, "tools": ["scene_commit", "attribute_test"]},
    ]
    # A turn the tool log never closed is unmeasured, not failed: None, never False.
    assert soak_session.score_thinking_fallbacks([9], tool_log) == [
        {"turn": 9, "tool_called": None, "tools": []}
    ]
    # A run with no tool log at all reports every fallback as unmeasured rather than
    # asserting that none of them called anything.
    assert [row["tool_called"] for row in soak_session.score_thinking_fallbacks([1], [])] == [None]
