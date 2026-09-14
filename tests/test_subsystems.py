"""Pin the dark-pacts subsystem data and its deterministic mechanics.

``rules/subsystems.json`` is derived from ``docs/srd/dark-pacts.md`` by a one-shot
generator. These tests hold an independent second copy of the counts, the runic
personality map, and the d100 boundaries, so a regenerated or reordered file
fails rather than passing on structure alone, matching the weapon-table golden in
``test_rules.py``.
"""
from __future__ import annotations

import json
import random

import pytest
from conftest import REPO_ROOT, ScriptedRoller

from bsh_mcp import rules
from bsh_mcp.data import RulesData, RulesDataError
from bsh_mcp.dice import Roller


@pytest.fixture
def data() -> RulesData:
    return RulesData.load(REPO_ROOT / "rules")


def _subsystems() -> dict:
    return json.loads((REPO_ROOT / "rules" / "subsystems.json").read_text(encoding="utf-8"))


def test_subsystem_counts_match_the_srd(data: RulesData):
    """Test subsystem counts match the srd.
    """
    subs = _subsystems()
    assert len(subs["demonic_pacts"]["demons"]) == 13
    assert len(subs["spirit_alliances"]["spirits"]) == 9
    assert len(subs["faerie_ties"]["ties"]) == 12
    assert len(subs["twisted_science"]["marvels"]) == 15


def test_the_sorcery_spell_table_covers_d100_contiguously(data: RulesData):
    """Every d100 result 1..100 maps to exactly one spell, so a draw never fails."""
    spells = _subsystems()["sorcery"]["spell_table"]["spells"]
    covered: dict[int, int] = {}
    for spell in spells:
        assert spell["low"] <= spell["high"]
        for face in range(spell["low"], spell["high"] + 1):
            covered[face] = covered.get(face, 0) + 1
    assert sorted(covered) == list(range(1, 101))
    assert max(covered.values()) == 1  # no overlap
    ids = [s["id"] for s in spells]
    assert len(set(ids)) == len(ids)  # unique ids


def test_the_backlash_tables_hold_all_six_faces(data: RulesData):
    """Demon's Revenge and Torn Veil are the deterministic d6 tables the engine rolls."""
    for face in range(1, 7):
        assert data.demon_revenge(face)
        assert data.torn_veil(face)
    # Spot-check the mechanically load-bearing rows against the SRD.
    assert "3d6 damage" in data.demon_revenge(6)
    assert "permanently lose 1 INT" in data.torn_veil(5)


def test_spell_for_roll_maps_the_die_ranges(data: RulesData):
    """Range boundaries resolve to the SRD spell, so the d100 draw is exact."""
    assert data.spell_for_roll(1)["id"] == "acid_blood"
    assert data.spell_for_roll(2)["id"] == "acid_blood"
    assert data.spell_for_roll(3)["id"] == "animate_mirror"
    assert data.spell_for_roll(100)["id"] == "withering"
    assert data.spell_for_roll(98)["id"] == "withering"


def test_spell_for_roll_rejects_a_result_off_the_table(data: RulesData):
    with pytest.raises(RulesDataError):
        data.spell_for_roll(101)


def test_draw_starting_spells_draws_four_by_default(data: RulesData):
    """Forbidden Knowledge draws four spells on d100; the count comes from the data."""
    assert data.subsystem("sorcery")["starting_spells"] == 4
    roller = ScriptedRoller(rng=random.Random(0), script=[1, 3, 100, 54])
    draws = rules.draw_starting_spells(roller, data)
    assert [d["roll"] for d in draws] == [1, 3, 100, 54]
    assert [d["id"] for d in draws] == ["acid_blood", "animate_mirror", "withering", "inquisition"]
    assert all(d["name"] and d["effect"] for d in draws)


def test_draw_starting_spells_respects_an_explicit_count(data: RulesData):
    roller = ScriptedRoller(rng=random.Random(0), script=[11, 12])
    draws = rules.draw_starting_spells(roller, data, count=2)
    assert len(draws) == 2
    assert draws[0]["id"] == draws[1]["id"] == "curse_of_the_mute"  # both in 11..12


def test_draw_starting_spells_is_deterministic_under_a_seed(data: RulesData):
    first = rules.draw_starting_spells(Roller.seeded(20260805), data)
    second = rules.draw_starting_spells(Roller.seeded(20260805), data)
    assert first == second
    assert len(first) == 4


def test_runic_personalities_map_to_attributes(data: RulesData):
    """The six runic personalities fix damage to one attribute each."""
    expected = {
        "brutal": "STR",
        "vicious": "DEX",
        "patient": "CON",
        "cunning": "INT",
        "judgemental": "WIS",
        "prideful": "CHA",
    }
    assert _subsystems()["runic_weapons"]["personalities"] == expected
    for personality, attribute in expected.items():
        assert data.runic_damage_attribute(personality) == attribute


def test_unknown_runic_personality_raises(data: RulesData):
    with pytest.raises(RulesDataError):
        data.runic_damage_attribute("merciful")


def test_missing_subsystem_raises(data: RulesData):
    with pytest.raises(RulesDataError):
        data.subsystem("necromancy")


def test_pact_alliance_and_tie_slots_cap_at_two(data: RulesData):
    """Warlock, Shaman, and Changeling each hold two, matching the backgrounds."""
    subs = _subsystems()
    assert subs["demonic_pacts"]["pact_slots"] == 2
    assert subs["spirit_alliances"]["alliance_slots"] == 2
    assert subs["faerie_ties"]["tie_slots"] == 2


def test_faerie_doom_triggers_mark_only_the_four_srd_ties(data: RulesData):
    """The SRD marks four ties with a Doom trigger; every other tie carries none."""
    triggers = {
        tie["id"]: tie.get("doom_trigger")
        for tie in _subsystems()["faerie_ties"]["ties"]
    }
    assert triggers["barrow_wisdom"] == "step_down"
    assert triggers["doomed_to_greatness"] == "roll_advantage"
    assert triggers["true_faith"] == "roll"
    assert triggers["elfin_secret"] == "roll_on_repeat"
    doomed_to_greatness = next(
        tie for tie in _subsystems()["faerie_ties"]["ties"]
        if tie["id"] == "doomed_to_greatness"
    )
    assert doomed_to_greatness["daily_limit"] == 1
    assert doomed_to_greatness["recovery_long_rests"] == "d3"
    marked = {tie_id for tie_id, trigger in triggers.items() if trigger}
    assert marked == {"barrow_wisdom", "doomed_to_greatness", "true_faith", "elfin_secret"}


def test_marvel_crafting_math(data: RulesData):
    """The invention-point economy: 20 coins per point, half-cost maintenance, INT per week."""
    assert data.marvel_usage_die() == "d6"
    assert data.marvel_materials_cost(4) == 80  # 20 coins per invention point
    assert data.marvel_maintenance_points(6) == 3  # half the cost, rounded up
    assert data.marvel_maintenance_points(5) == 3
    # An Inventor with INT 12 and no maintenance load builds a 6-point marvel in one week.
    assert data.marvel_build_weeks(6, 12, 0) == 1
    # A 10-point marvel at INT 4 takes three weeks: ceil(10 / 4).
    assert data.marvel_build_weeks(10, 4, 0) == 3
    # A maintenance load of 10 slows a build: rate = max(1, 12 - 10) = 2, ceil(6 / 2) = 3.
    assert data.marvel_build_weeks(6, 12, 10) == 3
