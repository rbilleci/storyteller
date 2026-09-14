"""Pin the roll-announcement injection that replaces asking the model to write it.
"""

from __future__ import annotations

from narrator.engine import (
    ROLL_UNDER_TOOLS,
    format_roll_announcement,
    inject_missing_roll_announcements,
)


def _fact(tool="attribute_test", character="rill", attribute="STR", total=6, target=12, outcome="failure"):
    return {
        "tool": tool, "character": character, "attribute": attribute,
        "total": total, "target": target, "outcome": outcome,
    }


def test_format_roll_announcement_matches_the_policy_shape():
    """Both numbers are labelled: the bare "6 vs 12" form left a live table unable
    to tell which number the dice produced and which the sheet supplied."""
    assert (
        format_roll_announcement(_fact(), "Rill")
        == "Rill rolls STR: rolled 6 vs target 12, failure."
    )


def test_format_roll_announcement_spaces_the_critical_outcomes():
    """``bsh_mcp.rules.Outcome`` spells it ``critical_failure``; the policy's own
    worked example spells it with a space."""
    assert (
        format_roll_announcement(_fact(outcome="critical_failure"), "Rill")
        == "Rill rolls STR: rolled 6 vs target 12, critical failure."
    )
    assert (
        format_roll_announcement(_fact(outcome="critical_success"), "Rill")
        == "Rill rolls STR: rolled 6 vs target 12, critical success."
    )


def test_injection_is_a_no_op_when_every_fact_is_already_announced():
    narration = "Rill rolls STR: rolled 6 vs target 12, failure.\n\nShe stumbles back, rubbing her arm."
    amended, injected = inject_missing_roll_announcements(
        narration, (_fact(),), {"rill": "Rill"}
    )
    assert amended == narration
    assert injected == 0


def test_injection_prepends_a_missing_announcement_with_the_resolved_name():
    narration = "Rill throws her shoulder against the door, but it does not budge."
    amended, injected = inject_missing_roll_announcements(
        narration, (_fact(),), {"rill": "Rill"}
    )
    assert amended == (
        "Rill rolls STR: rolled 6 vs target 12, failure.\n\n"
        "Rill throws her shoulder against the door, but it does not budge."
    )
    assert injected == 1


def test_injection_orders_multiple_missing_lines_oldest_fact_first():
    narration = "Neither of them can budge it."
    facts = (
        _fact(character="rill", attribute="STR", total=6),
        _fact(tool="group_test", character="ossa", attribute="STR", total=3, outcome="failure"),
    )
    amended, injected = inject_missing_roll_announcements(
        narration, facts, {"rill": "Rill", "ossa": "Ossa"}
    )
    assert amended == (
        "Rill rolls STR: rolled 6 vs target 12, failure.\n"
        "Ossa rolls STR: rolled 3 vs target 12, failure.\n\n"
        "Neither of them can budge it."
    )
    assert injected == 2


def test_injection_matches_on_attribute_and_total_not_on_the_character_name():
    """The same deliberate choice ``verdict_mismatches`` documents: identifying the
    roll, not the spelling of a name, is this function's job."""
    narration = "R rolls STR: 6 vs 12, failure.\n\nShe stumbles back."
    amended, injected = inject_missing_roll_announcements(
        narration, (_fact(),), {"rill": "Rill"}
    )
    assert amended == narration
    assert injected == 0


def test_injection_ignores_facts_from_tools_outside_the_announcement_policy():
    narration = "The party settles in for the night."
    facts = (_fact(tool="rest", attribute="", total=None, target=None, outcome=""),)
    amended, injected = inject_missing_roll_announcements(narration, facts, {})
    assert amended == narration
    assert injected == 0


def test_injection_falls_back_to_a_readable_transform_of_an_unresolved_id():
    """Never expected in practice -- every ``ROLL_UNDER_TOOLS`` fact resolves to a
    party character already in ``campaign/players.yaml`` -- but a missing mapping
    must still produce a readable line, not a crash or a silently dropped one."""
    narration = "The blade skitters off the ancient lock."
    amended, injected = inject_missing_roll_announcements(
        narration, (_fact(character="reed-thug"),), {}
    )
    assert amended.startswith("Reed Thug rolls STR: rolled 6 vs target 12, failure.\n\n")
    assert injected == 1


def test_injection_on_empty_narration_produces_only_the_announcement():
    amended, injected = inject_missing_roll_announcements("", (_fact(),), {"rill": "Rill"})
    assert amended == "Rill rolls STR: rolled 6 vs target 12, failure."
    assert injected == 1


def test_roll_under_tools_matches_the_four_tools_the_announcement_policy_covers():
    assert ROLL_UNDER_TOOLS == {"attribute_test", "group_test", "combat_attack", "combat_defend"}


def test_injection_covers_a_combat_start_initiative_roll():
    """Test injection covers a combat start initiative roll.
    """
    narration = "Vessa closes the distance and swings."
    fact = _fact(tool="combat_start", character="vessa", attribute="WIS", total=6, target=8, outcome="success")
    amended, injected = inject_missing_roll_announcements(narration, (fact,), {"vessa": "Vessa"})
    assert amended == (
        "Vessa rolls WIS: rolled 6 vs target 8, success.\n\n"
        "Vessa closes the distance and swings."
    )
    assert injected == 1


def test_injection_covers_a_runic_weapons_session_test_from_both_tools():
    """A runic weapon's start-of-session INT test is a real, audited d20 whose
    success arms a lethal verdict (``_become_helpless`` kills the wielder outright),
    and the grant-time fact sat outside ``ANNOUNCED_ROLL_TOOLS`` -- the identical
    gap ``combat_start`` had, fixed one day earlier for that tool alone --
    while ``session_close``'s re-roll never reached its envelope at all. The
    roller is the weapon: its target is the weapon's own INT score, not the
    wielder's, so the fact carries the weapon's ``name`` for display and the
    injector prefers it over the wielder's id in ``names``. A name is used
    verbatim -- ``"the Widow's Tooth"`` must not become ``"The Widow'S Tooth"``
    through the bare-id fallback's ``.title()``.
    """
    names = {"mara": "Mara"}
    granted = {**_fact(tool="grant_runic_weapon", character="mara", attribute="INT",
                       total=1, target=9, outcome="critical_success"),
               "name": "the Widow's Tooth"}
    amended, injected = inject_missing_roll_announcements("She lifts it.", (granted,), names)
    assert injected == 1
    assert amended == (
        "the Widow's Tooth rolls INT: rolled 1 vs target 9, critical success.\n\n"
        "She lifts it."
    )
    closed = {**_fact(tool="session_close", character="mara", attribute="INT",
                      total=20, target=9, outcome="critical_failure"),
              "name": "the Widow's Tooth"}
    amended, injected = inject_missing_roll_announcements("", (closed,), names)
    assert injected == 1
    assert amended == "the Widow's Tooth rolls INT: rolled 20 vs target 9, critical failure."


def test_a_runic_fact_carries_its_stake_line_beside_the_announcement():
    """Test a runic fact carries its stake line beside the announcement.
    """
    names = {"mara": "Mara"}
    armed = {**_fact(tool="grant_runic_weapon", character="mara", attribute="INT",
                     total=1, target=9, outcome="critical_success"),
             "name": "Sorrow", "kills_helpless": True}
    amended, injected = inject_missing_roll_announcements("She lifts it.", (armed,), names)
    assert injected == 1
    assert amended == (
        "Sorrow rolls INT: rolled 1 vs target 9, critical success.\n"
        "Sorrow has taken the measure of Mara: if Mara falls Helpless this session, "
        "the blade kills Mara outright.\n\n"
        "She lifts it."
    )
    disarmed_next = {**_fact(tool="session_close", character="mara", attribute="INT",
                             total=20, target=9, outcome="critical_failure"),
                     "name": "Sorrow", "kills_helpless": False}
    amended, injected = inject_missing_roll_announcements("", (disarmed_next,), names)
    assert injected == 1
    assert amended == (
        "Sorrow rolls INT: rolled 20 vs target 9, critical failure.\n"
        "Sorrow has not taken the measure of Mara for the next session: a Helpless "
        "fall in it is not the blade's kill."
    )
    # A model that wrote the announcement itself still gets the stake line, once.
    written = "Sorrow rolls INT: rolled 1 vs target 9, critical success.\nIt hums."
    amended, injected = inject_missing_roll_announcements(written, (armed,), names)
    assert injected == 0
    assert amended.count("has taken the measure of Mara") == 1
    assert amended.endswith("It hums.")
    # And never twice.
    assert inject_missing_roll_announcements(amended, (armed,), names)[0] == amended


def test_a_fact_without_a_name_still_resolves_the_wielder_through_names():
    """``name`` is additive: every pre-existing fact shape carries none, and those
    keep resolving the roller's display name from ``names`` exactly as before."""
    amended, _ = inject_missing_roll_announcements("", (_fact(),), {"rill": "Rill"})
    assert amended.startswith("Rill rolls STR")


def test_roll_facts_reads_the_roller_from_every_identifying_key():
    """Test roll facts reads the roller from every identifying key.
    """
    from narrator.engine import roll_facts

    envelope = {
        "ok": True,
        "attribute": "STR",
        "outcome": "failure",
        "roll": {"total": 18, "target": 10},
    }
    (test_fact,) = roll_facts("attribute_test", {**envelope, "character_id": "rill"})
    assert test_fact["character"] == "rill"
    (attack_fact,) = roll_facts("combat_attack", {**envelope, "attacker_id": "rill"})
    assert attack_fact["character"] == "rill"
    # combat_defend's envelope carries BOTH ids; the roller is the defender. The
    # first version of this chain preferred attacker_id, so a defence announced the
    # enemy as the one rolling.
    (defend_fact,) = roll_facts(
        "combat_defend",
        {**envelope, "attribute": "DEX", "defender_id": "rill", "attacker_id": "rade"},
    )
    assert defend_fact["character"] == "rill"
    assert defend_fact["attribute"] == "DEX"


def test_both_announcement_spellings_parse_and_neither_earns_a_duplicate():
    """The labelled shape is what the engine writes now; the bare shape is what a
    disobedient model may still write itself. Both must parse identically, so the
    verdict check can read either and the injection never adds a second line
    beside one already present in either spelling."""
    from narrator.engine import announced_rolls

    labelled = announced_rolls("Rill rolls STR: rolled 6 vs target 12, failure.")
    bare = announced_rolls("Rill rolls STR: 6 vs 12, failure.")
    wordy = announced_rolls("Rill rolls STR: rolled 6 vs a target of 12, failure.")
    assert labelled == bare == wordy
    assert labelled[0]["total"] == 6 and labelled[0]["target"] == 12

    narration = "Rill rolls STR: 6 vs 12, failure.\n\nShe stumbles back."
    amended, injected = inject_missing_roll_announcements(
        narration, (_fact(),), {"rill": "Rill"}
    )
    assert amended == narration
    assert injected == 0
