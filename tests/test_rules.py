"""Unit tests for the deterministic rules engine."""

from __future__ import annotations

import random

import pytest
from conftest import REPO_ROOT, ScriptedRoller

from bsh_mcp import rules
from bsh_mcp.data import RulesData
from bsh_mcp.dice import (
    DAMAGE_CHAIN,
    DEPLETED,
    USAGE_CHAIN,
    DiceError,
    Roller,
    resolve_edge,
    step_down,
    step_up,
)
from bsh_mcp.models import Attributes, Character, UsageResource


@pytest.fixture
def data() -> RulesData:
    return RulesData.load(REPO_ROOT / "rules")


def scripted(*values: int) -> ScriptedRoller:
    return ScriptedRoller(rng=random.Random(0), script=list(values))


# -- dice selection ---------------------------------------------------------


def test_advantage_selects_the_lower_d20():
    roller = scripted(17, 4)
    result = rules.resolve_test(roller, target=12, advantage=True)
    assert result.roll.dice == [17, 4]
    assert result.roll.selected == 4
    assert result.outcome == "success"


def test_disadvantage_selects_the_higher_d20():
    roller = scripted(4, 17)
    result = rules.resolve_test(roller, target=12, disadvantage=True)
    assert result.roll.selected == 17
    assert result.outcome == "failure"


def test_advantage_and_disadvantage_cancel_to_one_die():
    roller = scripted(9)
    result = rules.resolve_test(roller, target=12, advantage=True, disadvantage=True)
    assert result.roll.dice == [9]
    assert result.roll.notation == "1d20"
    assert resolve_edge(True, True) == "single"


def test_natural_one_is_a_critical_success_even_below_target():
    roller = scripted(1)
    result = rules.resolve_test(roller, target=8)
    assert result.outcome == "critical_success"
    assert result.succeeded


def test_natural_twenty_is_a_critical_failure_even_with_a_high_attribute():
    roller = scripted(20)
    result = rules.resolve_test(roller, target=20)
    assert result.outcome == "critical_failure"
    assert not result.succeeded


def test_threat_level_modifier_can_turn_a_success_into_a_failure():
    roller = scripted(11)
    modifier = rules.threat_modifier(actor_level=1, opponent_level=3)
    assert modifier == 2
    result = rules.resolve_test(roller, target=12, modifier=modifier)
    assert result.roll.total == 13
    assert result.outcome == "failure"


def test_threat_level_never_grants_a_bonus_for_a_weaker_opponent():
    assert rules.threat_modifier(actor_level=5, opponent_level=1) == 0
    assert rules.threat_modifier(actor_level=5, opponent_level=None) == 0


def test_criticals_read_the_unmodified_die_not_the_total():
    roller = scripted(20)
    result = rules.resolve_test(roller, target=12, modifier=-19)
    assert result.roll.total == 1
    assert result.outcome == "critical_failure"


# -- widened critical-success band (Battle Hardened) ------------------------


def test_classify_widens_the_critical_band_when_the_kept_die_is_inside_it():
    assert rules.classify(3, 15, 10, crit_success_max=3) == "critical_success"


def test_classify_a_die_outside_the_widened_band_resolves_by_total():
    # Die shows 4, outside a crit_success_max of 3: resolved by total (3 < 10), not
    # a critical, even though the same total (3) crits when the die itself is 3.
    assert rules.classify(4, 3, 10, crit_success_max=3) == "success"


def test_classify_default_band_is_unchanged():
    # A die of 2 does not crit under the default (unwidened) band.
    assert rules.classify(2, 5, 10) == "success"
    assert rules.classify(2, 15, 10) == "failure"


@pytest.mark.parametrize("crit_success_max", [3, 19, 20, 100])
def test_classify_natural_twenty_is_never_inside_the_widened_band(crit_success_max: int):
    assert rules.classify(20, 5, 10, crit_success_max=crit_success_max) == "critical_failure"


# -- usage dice -------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 2])
def test_usage_die_downgrades_on_one_or_two(value: int):
    outcome = rules.roll_usage(scripted(value), "d8")
    assert outcome.downgraded
    assert outcome.current_die == "d6"
    assert not outcome.depleted


@pytest.mark.parametrize("value", [3, 4, 5, 6, 7, 8])
def test_usage_die_holds_on_three_or_more(value: int):
    outcome = rules.roll_usage(scripted(value), "d8")
    assert not outcome.downgraded
    assert outcome.current_die == "d8"


def test_usage_die_chain_order():
    assert USAGE_CHAIN == ("d20", "d12", "d10", "d8", "d6", "d4")
    chain = ["d20"]
    while chain[-1] != DEPLETED:
        chain.append(step_down(chain[-1]))
    assert chain == ["d20", "d12", "d10", "d8", "d6", "d4", DEPLETED]


def test_a_downgraded_d4_is_depleted():
    outcome = rules.roll_usage(scripted(1), "d4")
    assert outcome.current_die == DEPLETED
    assert outcome.depleted


def test_a_depleted_usage_die_cannot_be_rolled():
    with pytest.raises(ValueError):
        rules.roll_usage(scripted(3), DEPLETED)


def test_invalid_die_notation_is_rejected():
    with pytest.raises(DiceError):
        rules.roll_usage(scripted(3), "d0")
    with pytest.raises(DiceError):
        step_down("d7")


def test_usage_advantage_keeps_the_higher_die():
    outcome = rules.roll_usage(scripted(1, 6), "d6", advantage=True)
    assert outcome.roll.selected == 6
    assert not outcome.downgraded


# -- damage-die step-up chain (Bloodlust) ------------------------------------
#
# DAMAGE_CHAIN/step_up is a deliberately separate table and function from
# USAGE_CHAIN/step_down: the Usage Die chain depletes downward and includes
# d20, neither of which a damage-die upgrade shares.


def test_damage_chain_order_and_membership_differs_from_usage_chain():
    assert DAMAGE_CHAIN == ("d4", "d6", "d8", "d10", "d12")
    assert "d20" not in DAMAGE_CHAIN
    assert set(DAMAGE_CHAIN) != set(USAGE_CHAIN)


def test_step_up_walks_the_chain_one_step_at_a_time():
    chain = ["d4"]
    while chain[-1] != "d12":
        chain.append(step_up(chain[-1]))
    assert chain == ["d4", "d6", "d8", "d10", "d12"]


def test_step_up_clamps_at_d12():
    assert step_up("d12") == "d12"
    assert step_up("d10", 5) == "d12"


def test_step_up_honors_a_multi_step_count():
    assert step_up("d4", 2) == "d8"


@pytest.mark.parametrize("die", ["d20", "d7", DEPLETED])
def test_step_up_rejects_a_die_off_the_damage_chain(die: str):
    with pytest.raises(DiceError):
        step_up(die)


def test_step_up_default_is_zero_steps_is_the_caller_convention_not_step_up_itself():
    # step_up always steps at least once when called; combat_attack's own
    # "only call step_up when steps is non-zero" guard is exercised in
    # tests/test_tools.py, not here -- this pins step_up's own contract.
    assert step_up("d4", 1) == "d6"


# -- doom -------------------------------------------------------------------


def test_call_on_doom_always_steps_the_die_down():
    outcome = rules.call_on_doom(scripted(6), "d6")
    assert outcome.roll.selected == 6
    assert outcome.current_die == "d4"
    assert outcome.downgraded


def test_call_on_doom_from_d4_leaves_the_character_doomed():
    outcome = rules.call_on_doom(scripted(4), "d4")
    assert outcome.current_die == DEPLETED
    assert outcome.depleted


def test_doom_roll_follows_usage_die_mechanics():
    held = rules.roll_doom(scripted(5), "d6")
    assert held.current_die == "d6"
    stepped = rules.roll_doom(scripted(2), "d6")
    assert stepped.current_die == "d4"


def test_doomed_condition_forces_disadvantage_on_every_test():
    character = Character(
        id="mara",
        name="Mara",
        origin="barbarian",
        attributes=Attributes(STR=12, DEX=12, CON=12, INT=12, WIS=12, CHA=12),
        hp=12,
        hp_max=12,
        doom_die=DEPLETED,
        conditions=[
            {
                "id": "doomed",
                "label": "Doomed",
                "effect": "disadvantage_all",
                "scope": "until_long_rest",
            }
        ],
    )
    advantage, disadvantage, notes = rules.character_edges(character, "STR", False, False)
    assert disadvantage
    assert notes
    two_handed, damage_disadvantage, _ = rules.damage_edges(character, two_handed=False)
    assert damage_disadvantage


def test_impaired_only_affects_dexterity():
    character = Character(
        id="ulf",
        name="Ulf",
        origin="civilised",
        attributes=Attributes(STR=12, DEX=12, CON=12, INT=12, WIS=12, CHA=12),
        hp=12,
        hp_max=12,
        conditions=[
            {"id": "impaired", "label": "Impaired", "effect": "disadvantage_dex", "scope": "session"}
        ],
    )
    _, dex_disadvantage, _ = rules.character_edges(character, "DEX", False, False)
    _, str_disadvantage, _ = rules.character_edges(character, "STR", False, False)
    assert dex_disadvantage
    assert not str_disadvantage


# -- damage and armour ------------------------------------------------------


def test_critical_damage_is_maximum_base_plus_one_die():
    outcome = rules.roll_damage(scripted(3), "d6", critical=True)
    assert outcome.total == 9
    assert outcome.critical


def test_two_handed_damage_keeps_the_higher_die():
    outcome = rules.roll_damage(scripted(2, 5), "d6", two_handed=True)
    assert outcome.total == 5


def test_disadvantaged_damage_keeps_the_lower_die():
    outcome = rules.roll_damage(scripted(2, 5), "d6", disadvantage=True)
    assert outcome.total == 2


def test_brutal_rerolls_a_damage_die_showing_one():
    outcome = rules.roll_damage(scripted(1, 4), "d6", brutal=True)
    assert outcome.total == 4
    assert len(outcome.rolls) == 2


def test_armour_reduces_damage_but_never_below_zero():
    assert rules.apply_armour(5, 2) == (3, 2)
    assert rules.apply_armour(1, 3) == (0, 1)
    assert rules.apply_armour(5, 3, ignore_armour=True) == (5, 0)


def test_armour_protection_ratings():
    assert rules.armour_protection("none") == 0
    assert rules.armour_protection("light") == 1
    assert rules.armour_protection("medium") == 2
    assert rules.armour_protection("heavy") == 3


# -- group tests ------------------------------------------------------------


@pytest.mark.parametrize(
    "successes,participants,expected",
    [(0, 2, False), (1, 2, True), (1, 3, False), (2, 3, True), (2, 4, True), (0, 0, False)],
)
def test_group_succeeds_at_half(successes: int, participants: int, expected: bool):
    assert rules.group_succeeds(successes, participants) is expected


# -- recovery ---------------------------------------------------------------


@pytest.mark.parametrize("con,healed", [(8, 4), (11, 5), (13, 6), (1, 0)])
def test_short_rest_restores_half_constitution(con: int, healed: int):
    assert rules.short_rest_healing(con) == healed


# -- ranges -----------------------------------------------------------------


def test_one_move_changes_one_band():
    assert rules.move_band("nearby", "closer") == "close"
    assert rules.move_band("nearby", "away") == "far_away"
    assert rules.move_band("close", "closer") == "close"
    assert rules.move_band("distant", "away") == "distant"


# -- character creation -----------------------------------------------------


def test_attribute_score_table_covers_two_to_twelve():
    assert [rules.score_from_2d6(total) for total in range(2, 13)] == [
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
        12,
        13,
    ]
    with pytest.raises(ValueError):
        rules.score_from_2d6(13)


def test_background_selection_requires_two_from_the_origin(data: RulesData):
    errors = rules.validate_backgrounds("barbarian", ["scout", "hunter", "vicious"], data)
    assert errors == []
    errors = rules.validate_backgrounds("barbarian", ["scout", "vicious", "warlock"], data)
    assert any("at least 2" in error for error in errors)


def test_only_one_unique_background_is_allowed(data: RulesData):
    errors = rules.validate_backgrounds("barbarian", ["berserker", "scout", "assassin"], data)
    assert any("unique" in error for error in errors)


def test_background_selection_rejects_unknown_and_duplicate_ids(data: RulesData):
    assert any(
        "unknown" in error
        for error in rules.validate_backgrounds("barbarian", ["scout", "hunter", "wizard"], data)
    )
    assert any(
        "twice" in error
        for error in rules.validate_backgrounds("barbarian", ["scout", "scout", "hunter"], data)
    )


def test_generated_character_hit_points_equal_constitution(data: RulesData):
    roller = scripted(*([6, 6] * 6))
    generated = rules.generate_character(roller, "barbarian", ["scout", "hunter", "survivor"], data)
    assert generated.attributes.CON == 14  # 13 from a 2d6 total of 12, plus Survivor
    assert generated.hp == generated.attributes.CON
    assert generated.coins == 25


# -- NPCs -------------------------------------------------------------------


def test_npc_table_matches_the_srd(data: RulesData):
    assert rules.npc_stats(1, "none", data) == (5, 4)
    assert rules.npc_stats(10, "none", data) == (50, 13)
    assert rules.npc_stats(3, "medium", data) == (18, 6)


def test_npc_level_bounds_are_enforced(data: RulesData):
    with pytest.raises(ValueError):
        rules.npc_stats(0, "none", data)
    with pytest.raises(ValueError):
        rules.npc_stats(11, "none", data)


def test_npc_morale_warning_thresholds(data: RulesData):
    assert rules.npc_morale_warning(2, 0.0, data) is not None
    assert rules.npc_morale_warning(9, 0.5, data) is not None
    assert rules.npc_morale_warning(9, 0.25, data) is None


# -- advancement ------------------------------------------------------------


def test_cumulative_story_requirements():
    assert rules.stories_required_for_level(1) == 0
    assert rules.stories_required_for_level(2) == 1
    assert rules.stories_required_for_level(3) == 3
    assert rules.stories_required_for_level(4) == 6
    assert rules.eligible_level(0) == 1
    assert rules.eligible_level(1) == 2
    assert rules.eligible_level(3) == 3


def _leveller(level: int = 1, **overrides) -> Character:
    """Return a minimal character for advancement-choice tests."""
    fields = dict(
        id="mara",
        name="Mara",
        origin="barbarian",
        level=level,
        attributes=Attributes(STR=13, DEX=13, CON=13, INT=13, WIS=13, CHA=13),
        hp=13,
        hp_max=13,
    )
    fields.update(overrides)
    return Character(**fields)


def test_advancement_benefits_match_the_srd_table(data: RulesData):
    # Levels 2 and 6 raise one attribute; 4 and 8 raise two; gifts at 3,5,7,9.
    assert rules.advancement_benefits(2, data) == {
        "target_level": 2,
        "hit_point_gain": 1,
        "attribute_increases": 1,
        "grants_gift": False,
        "doom_die": "",
        "attribute_max": 18,
    }
    assert rules.advancement_benefits(4, data)["attribute_increases"] == 2
    assert rules.advancement_benefits(3, data)["grants_gift"] is True
    assert rules.advancement_benefits(3, data)["attribute_increases"] == 0


def test_level_ten_grants_the_doom_die_and_no_hit_point(data: RulesData):
    benefits = rules.advancement_benefits(10, data)
    assert benefits["doom_die"] == "d8"
    assert benefits["hit_point_gain"] == 0
    assert benefits["grants_gift"] is False
    assert benefits["attribute_increases"] == 0


def test_advancement_benefits_reject_a_level_outside_the_range(data: RulesData):
    with pytest.raises(ValueError):
        rules.advancement_benefits(1, data)
    with pytest.raises(ValueError):
        rules.advancement_benefits(11, data)


def test_choice_validation_requires_the_exact_attribute_count(data: RulesData):
    gifts = data.gift_index()
    benefits = rules.advancement_benefits(4, data)  # two attributes
    assert rules.validate_advancement_choices(_leveller(3), benefits, ["STR", "DEX"], "", gifts) == []
    one = rules.validate_advancement_choices(_leveller(3), benefits, ["STR"], "", gifts)
    assert any("exactly 2" in violation for violation in one)


def test_choice_validation_rejects_a_repeated_attribute(data: RulesData):
    benefits = rules.advancement_benefits(4, data)
    violations = rules.validate_advancement_choices(
        _leveller(3), benefits, ["STR", "STR"], "", data.gift_index()
    )
    assert any("distinct" in violation for violation in violations)


def test_choice_validation_holds_the_attribute_ceiling_at_eighteen(data: RulesData):
    capped = _leveller(1, attributes=Attributes(STR=18, DEX=13, CON=13, INT=13, WIS=13, CHA=13))
    benefits = rules.advancement_benefits(2, data)
    violations = rules.validate_advancement_choices(capped, benefits, ["STR"], "", data.gift_index())
    assert any("maximum of 18" in violation for violation in violations)


def test_choice_validation_governs_the_gift(data: RulesData):
    gifts = data.gift_index()
    gift_benefits = rules.advancement_benefits(3, data)
    assert rules.validate_advancement_choices(_leveller(2), gift_benefits, [], "meditation", gifts) == []
    missing = rules.validate_advancement_choices(_leveller(2), gift_benefits, [], "", gifts)
    assert any("grants a Gift" in violation for violation in missing)
    unknown = rules.validate_advancement_choices(_leveller(2), gift_benefits, [], "no-such-gift", gifts)
    assert any("unknown gift" in violation for violation in unknown)
    owned = rules.validate_advancement_choices(
        _leveller(2, gifts=["meditation"]), gift_benefits, [], "meditation", gifts
    )
    assert any("already has" in violation for violation in owned)
    no_gift_level = rules.advancement_benefits(2, data)
    stray = rules.validate_advancement_choices(_leveller(1), no_gift_level, ["STR"], "meditation", gifts)
    assert any("grants no Gift" in violation for violation in stray)


def test_roller_rejects_impossible_dice():
    with pytest.raises(DiceError):
        Roller().die(1)


# -- SRD table completeness --------------------------------------------------


def _rules_json(name: str) -> dict:
    import json

    return json.loads((REPO_ROOT / "rules" / name).read_text(encoding="utf-8"))


def test_every_gift_is_recorded_five_per_category():
    """Advancement grants a Gift at levels 3, 5, 7, and 9. A table missing entries offers
    a player a choice the rules do not contain, so completeness is a correctness
    property rather than a content preference.
    """
    gifts = _rules_json("gifts.json")
    structure = gifts["structure"]
    expected = len(structure["categories"]) * structure["gifts_per_category"]

    assert gifts["completeness"] == "complete"
    assert len(gifts["gifts"]) == expected == 15
    for category in structure["categories"]:
        matching = [g for g in gifts["gifts"] if g["category"] == category]
        assert len(matching) == structure["gifts_per_category"], category
    identifiers = [g["id"] for g in gifts["gifts"]]
    assert len(set(identifiers)) == len(identifiers)
    assert all(g["feature"].strip() and g["name"].strip() for g in gifts["gifts"])
    assert {g["category"] for g in gifts["gifts"]} == set(structure["categories"])


def test_battle_hardened_keeps_the_srd_restriction_to_combat():
    """One transcription defect this file exists to prevent, pinned by name.

    The pre-transcription entry read "Attribute test results of 1 to 3 count as
    critical successes", which granted on every attribute test. The SRD restricts the
    benefit to combat and to an unmodified roll. rules/attribution.md records the
    correction.
    """
    gift = next(g for g in _rules_json("gifts.json")["gifts"] if g["id"] == "battle-hardened")

    assert "combat" in gift["feature"].lower()
    assert "unmodified" in gift["feature"].lower()


def test_battle_hardened_automation_is_registry_driven():
    """The registry drives the Gift, so the model must not apply it a second time.

    rules/gifts.json's automation field is the one place a reader (model or test)
    learns whether a Gift's text is a mechanized effect or a narrator ruling; a
    Gift whose effect ships but whose automation still reads 'manual' would get
    applied twice. This pins the flip Step 6 of the Battle Hardened slice made.
    """
    gift = next(g for g in _rules_json("gifts.json")["gifts"] if g["id"] == "battle-hardened")
    assert gift["automation"] == "effect"

    effects_registry = _rules_json("effects.json")["effects"]
    assert effects_registry["battle_hardened"]["source"] == "gift:battle-hardened"


def test_bloodlust_and_riddle_of_steel_automation_is_registry_driven():
    """Both flip 'manual' -> 'effect' in this slice; neither gains a frequency.

    The design doc's own correction: unlike Survivor's Luck, the SRD states no
    frequency for either Gift, so neither entry may carry a pool, a reset, or a
    resource_effect.
    """
    gifts_by_id = {g["id"]: g for g in _rules_json("gifts.json")["gifts"]}
    effects_registry = _rules_json("effects.json")["effects"]

    bloodlust_gift = gifts_by_id["bloodlust"]
    assert bloodlust_gift["automation"] == "effect"
    bloodlust_effect = effects_registry["bloodlust"]
    assert bloodlust_effect["source"] == "gift:bloodlust"
    assert bloodlust_effect["activation"] == "passive"
    assert "pool" not in bloodlust_effect
    assert "resource_effect" not in bloodlust_effect

    riddle_gift = gifts_by_id["riddle-of-steel"]
    assert riddle_gift["automation"] == "effect"
    riddle_effect = effects_registry["riddle_of_steel"]
    assert riddle_effect["source"] == "gift:riddle-of-steel"
    assert riddle_effect["activation"] == "intent"
    assert riddle_effect["choices_from"] == "weapons"
    assert "pool" not in riddle_effect
    assert "resource_effect" not in riddle_effect


SRD_WEAPON_TABLES: dict[str, list[str]] = {
    "barbarian": [
        "bone bow (two-handed)", "chakram", "claymore (two-handed)", "hunting knife",
        "iwisa", "spear (two-handed)", "nomad scimitar",
        "raider's great axe (two-handed)", "warhammer",
    ],
    "civilised": [
        "cestus", "dagger", "engraved longbow (two-handed)",
        "executioner's cleaver (two-handed)", "flail", "katana", "legion gladius",
        "rapier", "pilgrim's staff (two-handed)",
    ],
    "decadent": [
        "blood metal sickle", "crossbow (two-handed)",
        "inquisitor's long sword (two-handed)", "maul", "razor whip", "rusted harpoon",
        "scythe (two-handed)", "shiv", "serrated sword",
    ],
}


def test_each_origin_carries_the_srd_weapon_table_in_die_order():
    """Nine entries in die order, so entry N is the result of N.

    A d10 result of 10 exceeds the table and yields no weapon, which is why the tables
    hold nine entries rather than ten. ``rules.roll_starting_weapons`` selects
    ``table[index - 1]``, so the array position is the die result and nothing else
    records the mapping.
    """
    backgrounds = _rules_json("backgrounds.json")

    assert backgrounds["weapon_tables_source"] == "srd"
    assert set(backgrounds["origins"]) == set(SRD_WEAPON_TABLES)
    for origin, expected in SRD_WEAPON_TABLES.items():
        table = backgrounds["origins"][origin]["weapon_table"]
        assert table == expected, origin
        assert len(table) == 9, origin
        assert len(set(table)) == 9, origin
        # The property the golden protects, stated so a future reader cannot satisfy
        # this test by sorting both copies.
        assert table != sorted(table), origin


def test_the_background_roster_matches_the_srd_counts():
    """The SRD lists 10 barbarian, 9 civilised, and 7 decadent backgrounds.
    """
    origins = _rules_json("backgrounds.json")["origins"]

    assert {name: len(block["backgrounds"]) for name, block in origins.items()} == {
        "barbarian": 10,
        "civilised": 9,
        "decadent": 7,
    }


BACKGROUND_AUTOMATION_NOTES = {
    "combat_modifier",
    "session_resource",
    "test_category_advantage",
    "attribute_substitution",
    "narrative_ruling",
    "static_flag",
}

#: Every ``automation_tag`` the engine acts on. A tag outside this set is a typo
#: that silently applies no effect, so the classification test rejects it.
BACKGROUND_AUTOMATION_TAGS = {
    "initiative_advantage",
    "long_rest_anywhere",
    "extra_languages_2",
    "unarmed_as_weapon",
    "damage_die_d8",
    "draw_spells_4",
}


def _all_backgrounds() -> dict[str, dict]:
    origins = _rules_json("backgrounds.json")["origins"]
    return {b["id"]: b for block in origins.values() for b in block["backgrounds"]}


def _effect_automated_background_ids() -> set[str]:
    """Background ids the effect registry drives (source ``background:<id>``)."""
    registry = _rules_json("effects.json")["effects"]
    ids = set()
    for effect in registry.values():
        source = effect.get("source", "")
        if source.startswith("background:"):
            ids.add(source.split(":", 1)[1])
    return ids


def test_every_background_carries_one_automation_classification():
    """Each background is automated, subsystem-linked, or classified with the
    surface it still needs. A background is automated when the effect registry
    drives it (an ``effects.json`` entry sourced ``background:<id>``), when it
    carries a legacy ``automation_tag``, or when it links a ``subsystem``. A noted
    background is none of those, so a manual background cannot silently regress to
    unclassified, and a new one must declare its state. ``automation_note`` uses a
    fixed vocabulary.
    """
    effect_automated = _effect_automated_background_ids()
    for bid, background in _all_backgrounds().items():
        automated = (
            bid in effect_automated
            or "automation_tag" in background
            or "subsystem" in background
        )
        note = background.get("automation_note")
        assert automated or note, f"{bid} carries no automation classification"
        if "automation_tag" in background:
            tag = background["automation_tag"]
            assert tag in BACKGROUND_AUTOMATION_TAGS, f"{bid} tag {tag!r} is off-vocabulary"
        if note is not None:
            assert not automated, f"{bid} is both noted and automated"
            assert note in BACKGROUND_AUTOMATION_NOTES, f"{bid} note {note!r} is off-vocabulary"


def test_every_subsystem_background_names_a_real_subsystem():
    """A ``subsystem`` value must key a block in subsystems.json, so the warning
    and the canon resource resolve.
    """
    subsystems = _rules_json("subsystems.json")
    for bid, background in _all_backgrounds().items():
        name = background.get("subsystem")
        if name is not None:
            assert name in subsystems, f"{bid} names unknown subsystem {name!r}"


# -- UsageResource.maximum backfill (Resourceful) ----------------------------


def test_usage_resource_maximum_backfills_to_the_constructed_die():
    resource = UsageResource(id="oil", name="Lamp oil", die="d8")
    assert resource.maximum == "d8"


def test_usage_resource_explicit_maximum_survives_construction():
    resource = UsageResource(id="oil", name="Lamp oil", die="d6", maximum="d10")
    assert resource.maximum == "d10"


def test_usage_resource_legacy_dict_without_maximum_backfills_on_load():
    legacy = {"id": "rations", "name": "Rations", "die": "d8"}  # no "maximum" key
    resource = UsageResource(**legacy)
    assert resource.maximum == "d8"


def test_usage_resource_legacy_depleted_dict_keeps_maximum_unknown():
    legacy = {"id": "rations", "name": "Rations", "die": DEPLETED}
    resource = UsageResource(**legacy)
    assert resource.maximum == ""
