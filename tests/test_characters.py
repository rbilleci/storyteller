"""Character creation and advancement tool tests."""

from __future__ import annotations

import json

from conftest import REPO_ROOT, make_character
from tools_shared import _award_stories, make_fighter, open_fight

from bsh_mcp.service import GameService

# -- character creation ------------------------------------------------------


def test_character_creation_sets_hit_points_doom_and_coins(service: GameService, roller):
    result = make_fighter(service, roller)
    sheet = result["sheet"]
    assert sheet["hp"] == sheet["hp_max"] == sheet["attributes"]["CON"]
    assert sheet["doom_die"] == "d6"
    assert sheet["coins"] == 25
    assert sheet["weapon_damage"] == "d6"
    assert sheet["unarmed_damage"] == "d4"


def test_illegal_background_selection_is_rejected_without_writing(service: GameService, roller):
    result = service.character_create(
        discord_user_id="1", name="Ulf", origin="barbarian", backgrounds=["berserker", "assassin", "scout"]
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_backgrounds"
    assert service.store.character_ids() == []


def test_vicious_background_upgrades_the_damage_dice(service: GameService, roller):
    result = make_character(
        service,
        roller,
        name="Sela",
        origin="decadent",
        backgrounds=("vicious", "snake-blood", "pit-fighter"),
        weapons=("rapier",),
    )
    assert result["sheet"]["weapon_damage"] == "d8"
    assert result["sheet"]["unarmed_damage"] == "d8"  # pit-fighter matches the weapon die


def test_forbidden_knowledge_draws_four_starting_spells(service: GameService, roller):
    subs = json.loads((REPO_ROOT / "rules" / "subsystems.json").read_text(encoding="utf-8"))
    spell_names = {s["name"] for s in subs["sorcery"]["spell_table"]["spells"]}

    result = make_character(
        service,
        roller,
        name="Vex",
        origin="decadent",
        backgrounds=("forbidden-knowledge", "snake-blood", "assassin"),
    )
    spells = result["sheet"]["spells"]
    assert len(spells) == 4
    assert all(name in spell_names for name in spells)
    draws = result["spell_draws"]
    assert len(draws) == 4
    assert all(1 <= d["roll"] <= 100 for d in draws)
    assert all(d["name"] in spell_names for d in draws)
    assert any("Forbidden knowledge spells:" in line for line in result["background_features"])
    # The draw persists to the authoritative sheet.
    assert service.store.read_character("vex").spells == spells
    # Forbidden knowledge links to the sorcery subsystem but declares no slot cap,
    # so its warning names the subsystem and the canon without a capacity number.
    fk_warnings = [w for w in result["warnings"] if "Forbidden knowledge" in w]
    assert fk_warnings, result["warnings"]
    assert "sorcery" in fk_warnings[0]
    assert "bsh://rules/subsystems" in fk_warnings[0]
    assert "up to" not in fk_warnings[0]


def test_subsystem_background_warning_names_the_subsystem(service: GameService, roller):
    result = make_character(
        service,
        roller,
        name="Grimm",
        origin="decadent",
        backgrounds=("warlock", "snake-blood", "assassin"),
    )
    warlock_warnings = [w for w in result["warnings"] if "Warlock" in w]
    assert warlock_warnings, result["warnings"]
    warning = warlock_warnings[0]
    assert "demonic pacts" in warning
    assert "up to 2" in warning
    assert "bsh://rules/subsystems" in warning


def test_non_subsystem_background_keeps_the_generic_ruling_warning(service: GameService, roller):
    result = make_character(
        service,
        roller,
        name="Sten",
        origin="barbarian",
        backgrounds=("berserker", "chieftain", "scout"),
    )
    berserker_warnings = [w for w in result["warnings"] if "Berserker" in w]
    assert berserker_warnings, result["warnings"]
    assert "narrative-only effects" in berserker_warnings[0]
    assert "bsh://rules/subsystems" not in berserker_warnings[0]


def test_character_sheet_reports_derived_values(service: GameService, roller):
    make_fighter(service, roller, armour="medium")
    sheet = service.character_sheet("mara")["sheet"]
    assert sheet["armour_protection"] == 2
    assert sheet["doomed"] is False
    assert sheet["eligible_level"] == 1


# -- advancement --------------------------------------------------------------


def test_subsystem_gift_warning_points_at_the_canon(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 3)  # enough to reach level 3
    service.character_advance("mara", attribute_increases=["STR"])  # to level 2
    result = service.character_advance("mara", gift_id="spirit-alliance")  # level 3 Gift
    assert result["ok"], result
    gift_warnings = [w for w in result["warnings"] if "Spirit alliance" in w]
    assert gift_warnings, result["warnings"]
    assert "spirit alliances" in gift_warnings[0]
    assert "bsh://rules/subsystems" in gift_warnings[0]


def test_non_subsystem_gift_keeps_the_generic_ruling_warning(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 3)
    service.character_advance("mara", attribute_increases=["STR"])  # to level 2
    result = service.character_advance("mara", gift_id="meditation")  # level 3 Gift, no subsystem
    assert result["ok"], result
    gift_warnings = [w for w in result["warnings"] if "Meditation" in w]
    assert gift_warnings, result["warnings"]
    assert "applies this as a ruling" in gift_warnings[0]
    assert "bsh://rules/subsystems" not in gift_warnings[0]


def test_character_advance_refuses_an_ineligible_character(service: GameService, roller):
    make_fighter(service, roller)
    before = service.store.read_character("mara").model_dump()
    result = service.character_advance("mara", attribute_increases=["STR"])
    assert result["ok"] is False
    assert result["error"] == "not_eligible"
    # Nothing was written.
    assert service.store.read_character("mara").model_dump() == before


def test_character_advance_raises_one_level_and_applies_the_choices(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 1)
    before = service.store.read_character("mara")
    result = service.character_advance("mara", attribute_increases=["STR"])
    assert result["ok"], result
    assert result["to_level"] == 2
    mara = service.store.read_character("mara")
    assert mara.level == 2
    assert mara.attributes.STR == before.attributes.STR + 1
    assert mara.hp_max == before.hp_max + 1
    assert mara.hp == before.hp + 1
    assert mara.gifts == []


def test_character_advance_takes_exactly_one_level_per_call(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 3)  # enough to reach level 3
    first = service.character_advance("mara", attribute_increases=["STR"])
    assert first["ok"], first
    assert first["to_level"] == 2
    assert first["further_advancement_available"] is True
    # Level 3 grants a Gift and no attribute increase.
    second = service.character_advance("mara", gift_id="meditation")
    assert second["ok"], second
    assert second["to_level"] == 3
    mara = service.store.read_character("mara")
    assert mara.level == 3
    assert mara.gifts == ["meditation"]
    assert second["further_advancement_available"] is False


def test_character_advance_rejects_a_gift_at_a_non_gift_level(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 1)
    result = service.character_advance("mara", attribute_increases=["STR"], gift_id="meditation")
    assert result["ok"] is False
    assert result["error"] == "invalid_advancement_choices"
    assert service.store.read_character("mara").level == 1


def test_character_advance_requires_the_gift_at_a_gift_level(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 3)
    service.character_advance("mara", attribute_increases=["STR"])  # to level 2
    result = service.character_advance("mara")  # level 3 needs a gift
    assert result["ok"] is False
    assert result["error"] == "invalid_advancement_choices"
    assert service.store.read_character("mara").level == 2


def test_character_advance_rejects_a_repeated_gift(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 10)  # enough for level 5
    service.character_advance("mara", attribute_increases=["STR"])  # 2
    service.character_advance("mara", gift_id="meditation")  # 3
    service.character_advance("mara", attribute_increases=["STR", "DEX"])  # 4
    result = service.character_advance("mara", gift_id="meditation")  # 5, repeat
    assert result["ok"] is False
    assert result["error"] == "invalid_advancement_choices"


def test_character_advance_reaches_level_ten_and_upgrades_the_doom_die(service: GameService, roller):
    make_fighter(service, roller)
    base_hp_max = service.store.read_character("mara").hp_max
    _award_stories(service, "mara", 45)  # cumulative Stories for level 10
    plan = {
        2: (["STR"], ""),
        3: ([], "meditation"),
        4: (["DEX", "CON"], ""),
        5: ([], "second-wind"),
        6: (["STR"], ""),
        7: ([], "fortress-of-the-mind"),
        8: (["DEX", "CON"], ""),
        9: ([], "survivors-luck"),
        10: ([], ""),
    }
    for target in range(2, 11):
        attributes, gift = plan[target]
        result = service.character_advance("mara", attribute_increases=attributes, gift_id=gift)
        assert result["ok"], (target, result)
        assert result["to_level"] == target
    mara = service.store.read_character("mara")
    assert mara.level == 10
    assert mara.doom_die == "d8"
    assert mara.doom_max == "d8"
    assert len(mara.gifts) == 4
    # Eight +1 hit points across levels 2 through 9, none at level 10.
    assert mara.hp_max == base_hp_max + 8


def test_character_advance_refuses_during_combat(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 1)  # eligible for level 2
    open_fight(service, roller)
    assert service.store.read_state().combat.active is True
    result = service.character_advance("mara", attribute_increases=["STR"])
    assert result["ok"] is False
    assert result["error"] == "combat_active"
    # An eligible character is not advanced while the fight runs.
    assert service.store.read_character("mara").level == 1


def test_character_advance_refuses_past_the_maximum_level(service: GameService, roller):
    make_fighter(service, roller)
    _award_stories(service, "mara", 60)
    for target in range(2, 11):
        attributes, gift = {
            2: (["STR"], ""), 3: ([], "meditation"), 4: (["DEX", "CON"], ""),
            5: ([], "second-wind"), 6: (["STR"], ""), 7: ([], "fortress-of-the-mind"),
            8: (["DEX", "CON"], ""), 9: ([], "survivors-luck"), 10: ([], ""),
        }[target]
        assert service.character_advance("mara", attribute_increases=attributes, gift_id=gift)["ok"]
    result = service.character_advance("mara")
    assert result["ok"] is False
    assert result["error"] == "already_max_level"
