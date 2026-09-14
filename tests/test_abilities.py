"""Herbalist, use_ability (toggles and resources), Second Wind, Resourceful, Battle
Hardened, Bloodlust, Riddle of Steel, and Survivor's Luck tool tests."""

from __future__ import annotations

from conftest import ScriptedRoller, make_character
from tools_shared import _award_stories, drop_to_helpless, make_assassin, make_fighter, open_fight

from bsh_mcp.dice import DEPLETED
from bsh_mcp.service import GameService

HERBALIST_BACKGROUNDS = ("herbalist", "scout", "survivor")


def make_herbalist(service: GameService, roller: ScriptedRoller, **kwargs) -> dict:
    """A barbarian with INT 14 (herbalist's +1), STR/DEX/CON/WIS/CHA 13."""
    kwargs.setdefault("name", "Mara")
    kwargs.setdefault("origin", "barbarian")
    kwargs.setdefault("backgrounds", HERBALIST_BACKGROUNDS)
    return make_character(service, roller, **kwargs)


def prepare_doses(
    service: GameService, roller: ScriptedRoller, character_id: str, dose_type: str, count: int
) -> None:
    """Declare, tag the scene natural, and long-rest to seed a dose stock for a test."""
    declared = service.use_ability(character_id, "herbalist_stock", choice=dose_type)
    assert declared["ok"], declared
    tagged = service.scene_commit("a stand of useful herbs", environment_tags=["natural"])
    assert tagged["ok"], tagged
    roller.queue(count)
    rested = service.rest([character_id], "long", safe_environment=True, reason="prepare supplies")
    assert rested["ok"], rested


# -- Herbalist: standing declarations resolved on a class-neutral rest -------


def test_declaring_a_preparation_choice_records_it(service: GameService, roller):
    make_herbalist(service, roller)
    result = service.use_ability("mara", "herbalist_stock", choice="poison")
    assert result["ok"], result
    assert result["details"]["choice"] == "poison"
    assert service.store.read_character("mara").declared_choices == {"herbalist_stock": "poison"}


def test_declaring_an_illegal_choice_is_refused(service: GameService, roller):
    make_herbalist(service, roller)
    result = service.use_ability("mara", "herbalist_stock", choice="acid")
    assert result["ok"] is False
    assert result["error"] == "invalid_intent_choice"
    assert service.store.read_character("mara").declared_choices == {}


def test_withdrawing_a_declaration_clears_it(service: GameService, roller):
    make_herbalist(service, roller)
    service.use_ability("mara", "herbalist_stock", choice="poison")
    result = service.use_ability("mara", "herbalist_stock", mode="deactivate")
    assert result["ok"], result
    assert service.store.read_character("mara").declared_choices == {}


def test_a_long_rest_in_a_natural_scene_resolves_the_declared_preparation(
    service: GameService, roller
):
    make_herbalist(service, roller)
    service.use_ability("mara", "herbalist_stock", choice="poison")
    tagged = service.scene_commit("camp beneath the pines", environment_tags=["natural"])
    assert tagged["ok"], tagged
    roller.queue(4)  # the d6 quantity roll -- no other dice this rest
    result = service.rest(["mara"], "long", safe_environment=True, reason="brew through the night")
    assert result["ok"], result
    assert result["characters"][0]["doses_prepared"] == [{"dose_type": "poison", "count": 4}]
    assert any("poison" in fact for fact in result["narration_facts"])
    character = service.store.read_character("mara")
    assert character.doses == {"poison": 4}
    assert character.declared_choices == {}


def test_a_long_rest_without_the_natural_tag_leaves_the_declaration_standing(
    service: GameService, roller
):
    make_herbalist(service, roller)
    service.use_ability("mara", "herbalist_stock", choice="healing_balm")
    result = service.rest(["mara"], "long", safe_environment=True, reason="sleep in the tower")
    assert result["ok"], result
    assert "doses_prepared" not in result["characters"][0]
    character = service.store.read_character("mara")
    assert character.doses == {}
    assert character.declared_choices == {"herbalist_stock": "healing_balm"}
    assert any("natural" in warning for warning in result["warnings"])


def test_environment_tags_replace_rather_than_accumulate(service: GameService, roller):
    make_herbalist(service, roller)
    service.scene_commit("in the pines", environment_tags=["natural"])
    service.scene_commit("back in the city", environment_tags=["urban"])
    assert service.store.read_state().scene.environment_tags == ["urban"]
    unspecified = service.scene_commit("still in the city")
    assert unspecified["ok"], unspecified
    assert service.store.read_state().scene.environment_tags == ["urban"]


def test_rest_stays_class_neutral_for_a_non_herbalist(service: GameService, roller):
    make_fighter(service, roller)  # hunter/survivor/raider -- no herbalist
    service.scene_commit("in the pines", environment_tags=["natural"])
    result = service.rest(["mara"], "long", safe_environment=True, reason="sleep")
    assert result["ok"], result
    assert "doses_prepared" not in result["characters"][0]


def test_two_herbalists_can_each_declare_a_different_type_and_both_resolve(
    service: GameService, roller
):
    """Test two herbalists can each declare a different type and both resolve.
    """
    make_herbalist(service, roller, name="Mara")
    make_herbalist(service, roller, name="Ulf")
    service.use_ability("mara", "herbalist_stock", choice="poison")
    service.use_ability("ulf", "herbalist_stock", choice="healing_balm")
    service.scene_commit("a mossy hollow", environment_tags=["natural"])
    roller.queue(4, 5)  # mara's d6, then ulf's d6, in character_ids order
    result = service.rest(
        ["mara", "ulf"], "long", safe_environment=True, reason="camp together"
    )
    assert result["ok"], result
    mara_entry = next(e for e in result["characters"] if e["character_id"] == "mara")
    ulf_entry = next(e for e in result["characters"] if e["character_id"] == "ulf")
    assert mara_entry["doses_prepared"] == [{"dose_type": "poison", "count": 4}]
    assert ulf_entry["doses_prepared"] == [{"dose_type": "healing_balm", "count": 5}]
    assert service.store.read_character("mara").doses == {"poison": 4}
    assert service.store.read_character("ulf").doses == {"healing_balm": 5}


def test_a_wildling_herbalist_prepares_on_an_unsafe_long_rest(service: GameService, roller):
    make_character(
        service, roller, name="Mara", origin="barbarian",
        backgrounds=("herbalist", "wildling", "scout"),
    )
    service.use_ability("mara", "herbalist_stock", choice="hallucinogen")
    service.scene_commit("deep wilderness", environment_tags=["natural"])
    roller.queue(3)
    result = service.rest(["mara"], "long", safe_environment=False, reason="camp in the open")
    assert result["ok"], result
    assert result["characters"][0]["doses_prepared"] == [{"dose_type": "hallucinogen", "count": 3}]


def test_a_bookworm_resting_alongside_a_herbalist_still_gets_its_own_pool_reset(
    service: GameService, roller
):
    make_herbalist(service, roller, name="Mara")
    make_character(
        service, roller, name="Ulf", origin="civilised",
        backgrounds=("bookworm", "diplomat", "sophist"),
    )
    spent = service.use_ability("ulf", "bookworm_substitution")
    assert spent["ok"], spent
    assert service.store.read_character("ulf").pools.get("bookworm_substitution") == 0

    result = service.rest(["mara", "ulf"], "long", safe_environment=True, reason="camp together")
    assert result["ok"], result
    # A missing key reads as a full pool (see GameService._reset_pools).
    assert "bookworm_substitution" not in service.store.read_character("ulf").pools


def test_a_character_saved_before_herbalist_fields_still_loads(service: GameService, roller):
    from bsh_mcp.models import Character

    make_herbalist(service, roller)
    sheet = service.store.read_character("mara").model_dump()
    del sheet["doses"]
    del sheet["declared_choices"]
    reloaded = Character(**sheet)
    assert reloaded.doses == {}
    assert reloaded.declared_choices == {}


def test_a_scene_saved_before_environment_tags_still_loads(service: GameService, roller):
    from bsh_mcp.models import Scene

    sheet = service.store.read_state().scene.model_dump()
    del sheet["environment_tags"]
    reloaded = Scene(**sheet)
    assert reloaded.environment_tags == []


def test_the_sheet_lists_the_herbalist_declared_choice_and_dose_stock(
    service: GameService, roller
):
    make_herbalist(service, roller)
    service.use_ability("mara", "herbalist_stock", choice="poison")
    result = service.character_sheet("mara")
    abilities = result["sheet"]["abilities"]
    stock_ability = next(a for a in abilities if a["ability_id"] == "herbalist_stock")
    assert stock_ability["declared"] == "poison"
    assert set(stock_ability["choices"]) == {"healing_balm", "poison", "hallucinogen"}
    dose_ability = next(a for a in abilities if a["ability_id"] == "herbalist_dose")
    assert dose_ability["doses"] == {"healing_balm": 0, "poison": 0, "hallucinogen": 0}


# -- Herbalist: spending a prepared dose --------------------------------------


def test_a_healing_balm_dose_heals_d6_plus_level(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "healing_balm", 2)
    npc = open_fight(service, roller)
    roller.queue(19)
    service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=10)
    injured = service.store.read_character("mara")
    assert injured.hp == injured.hp_max - 10

    roller.queue(4)
    result = service.use_ability("mara", "herbalist_dose", choice="healing_balm")
    assert result["ok"], result
    assert result["details"]["roll"] == 4
    assert result["details"]["level_bonus"] == 1
    assert result["details"]["healed"] == 5
    assert result["details"]["remaining"] == 1
    healed = service.store.read_character("mara")
    assert healed.hp == injured.hp + 5
    assert healed.doses == {"healing_balm": 1}


def test_a_balm_can_heal_an_ally(service: GameService, roller):
    make_herbalist(service, roller, name="Mara")
    make_character(
        service, roller, name="Ulf", origin="barbarian",
        backgrounds=("scout", "survivor", "raider"),
    )
    prepare_doses(service, roller, "mara", "healing_balm", 1)
    roller.queue(19)
    service.combat_defend("ulf", method="dodge", incoming_damage=6)
    injured = service.store.read_character("ulf")
    assert injured.hp < injured.hp_max

    roller.queue(3)
    result = service.use_ability("mara", "herbalist_dose", choice="healing_balm", target_id="ulf")
    assert result["ok"], result
    assert result["details"]["target_id"] == "ulf"
    healed = service.store.read_character("ulf")
    assert healed.hp == injured.hp + 3 + 1  # roll + Mara's level
    assert service.store.read_character("mara").doses == {}


def test_a_balm_wakes_a_helpless_character(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "healing_balm", 1)
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    assert service.store.read_character("mara").status == "helpless"

    roller.queue(5)
    result = service.use_ability("mara", "herbalist_dose", choice="healing_balm")
    assert result["ok"], result
    healed = service.store.read_character("mara")
    assert healed.status == "ok"
    assert healed.hp == 6


def test_spending_a_dose_without_stock_is_refused(service: GameService, roller):
    make_herbalist(service, roller)
    result = service.use_ability("mara", "herbalist_dose", choice="healing_balm")
    assert result["ok"] is False
    assert result["error"] == "no_doses"


def test_spending_an_illegal_dose_type_is_refused(service: GameService, roller):
    make_herbalist(service, roller)
    result = service.use_ability("mara", "herbalist_dose", choice="acid")
    assert result["ok"] is False
    assert result["error"] == "invalid_dose_type"


def test_a_hallucinogen_dose_only_decrements_and_is_adjudicated(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "hallucinogen", 1)
    result = service.use_ability("mara", "herbalist_dose", choice="hallucinogen")
    assert result["ok"], result
    assert result["details"]["adjudicated"] is True
    assert "roll" not in result["details"]
    assert service.store.read_character("mara").doses == {}


# -- Herbalist: poison delivered through combat_attack ------------------------


def test_a_poisoned_attack_adds_d6_damage_and_spends_the_dose(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "poison", 1)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4, 3)  # d20 hit, weapon damage die, poison d6
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee", poisoned=True)
    assert result["ok"], result
    assert result["poisoned"] is True
    assert result["damage"]["poison"] == 3
    assert result["applied_damage"] == 7
    assert service.store.read_character("mara").doses == {}


def test_a_poisoned_miss_still_spends_the_dose(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "poison", 1)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(19)  # a miss -- no damage die, no poison die rolled
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee", poisoned=True)
    assert result["ok"], result
    assert result["outcome"] == "failure"
    assert result["applied_damage"] == 0
    assert "poison" not in (result["damage"] or {})
    assert service.store.read_character("mara").doses == {}


def test_a_poisoned_attack_without_a_dose_is_refused_before_dice(service: GameService, roller):
    make_herbalist(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    before = service.store.read_state().event_seq
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee", poisoned=True)
    assert result["ok"] is False
    assert result["error"] == "no_poison_dose"
    assert service.store.read_state().event_seq == before


def test_poison_needs_a_damaging_attack(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "poison", 1)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", weapon_effect="disarm", poisoned=True
    )
    assert result["ok"] is False
    assert result["error"] == "poison_needs_a_damaging_attack"


def test_a_runic_strike_takes_no_poison(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "poison", 1)
    roller.queue(5, 6, 3)  # 2d6 weapon INT, then the session test
    grant = service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    assert grant["ok"], grant
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", runic=True, poisoned=True
    )
    assert result["ok"] is False
    assert result["error"] == "runic_attack_conflict"


def test_an_unarmed_strike_takes_no_poison(service: GameService, roller):
    make_herbalist(service, roller)
    prepare_doses(service, roller, "mara", "poison", 1)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", unarmed=True, poisoned=True
    )
    assert result["ok"] is False
    assert result["error"] == "inconsistent_weapon_declaration"


# -- use_ability: toggles and resources --------------------------------------


def _berserker(service: GameService, roller: ScriptedRoller) -> dict:
    return make_character(
        service, roller, name="Ulf", origin="barbarian",
        backgrounds=("berserker", "raider", "hunter"),
    )


def _resource_holder(service: GameService, roller: ScriptedRoller) -> dict:
    return make_character(
        service, roller, name="Sable", origin="civilised",
        backgrounds=("legionnaire", "sophist", "bookworm"),
    )


def test_use_ability_toggles_berserker_rage(service: GameService, roller):
    _berserker(service, roller)
    activated = service.use_ability("ulf", "berserker_rage", mode="activate")
    assert activated["ok"] and activated["details"]["active"] is True
    ulf = service.store.read_character("ulf")
    assert any(c.effect_id == "berserker_rage" for c in ulf.conditions)
    dropped = service.use_ability("ulf", "berserker_rage", mode="deactivate")
    assert dropped["ok"] and dropped["details"]["active"] is False
    assert not service.store.read_character("ulf").conditions


def test_use_ability_refuses_a_passive_effect(service: GameService, roller):
    _berserker(service, roller)
    result = service.use_ability("ulf", "raider_crit", mode="use")
    assert result["ok"] is False and result["error"] == "not_activatable"


def test_use_ability_refuses_an_unknown_ability(service: GameService, roller):
    _berserker(service, roller)
    result = service.use_ability("ulf", "not_a_real_ability")
    assert result["ok"] is False and result["error"] == "unknown_ability"


def test_use_ability_spends_and_exhausts_a_resource(service: GameService, roller):
    _resource_holder(service, roller)
    for expected_left in (2, 1, 0):
        result = service.use_ability("sable", "legionnaire_reroll")
        assert result["ok"] and result["details"]["remaining"] == expected_left
    exhausted = service.use_ability("sable", "legionnaire_reroll")
    assert exhausted["ok"] is False and exhausted["error"] == "ability_exhausted"


def test_session_close_refreshes_session_pools(service: GameService, roller):
    _resource_holder(service, roller)
    service.use_ability("sable", "sophist_lie")  # 1/session, now exhausted
    assert service.use_ability("sable", "sophist_lie")["error"] == "ability_exhausted"
    service.session_close("Between adventures", "The party counts its scars.")
    assert service.use_ability("sable", "sophist_lie")["ok"] is True


def test_long_rest_refreshes_a_per_rest_pool(service: GameService, roller):
    _resource_holder(service, roller)
    service.use_ability("sable", "bookworm_substitution")  # 1/long_rest
    assert service.use_ability("sable", "bookworm_substitution")["error"] == "ability_exhausted"
    service.rest(["sable"], rest_type="long", safe_environment=True)
    assert service.use_ability("sable", "bookworm_substitution")["ok"] is True


def test_abilities_block_lists_activatable_abilities(service: GameService, roller):
    _resource_holder(service, roller)
    abilities = {a["ability_id"]: a for a in service.character_sheet("sable")["sheet"]["abilities"]}
    assert abilities["legionnaire_reroll"]["remaining"] == 3
    assert abilities["legionnaire_reroll"]["activation"] == "resource"
    assert abilities["bookworm_substitution"]["reset"] == "long_rest"
    # A passive effect never appears in the activatable block.
    _berserker(service, roller)
    ulf_abilities = {a["ability_id"] for a in service.character_sheet("ulf")["sheet"]["abilities"]}
    assert "berserker_rage" in ulf_abilities
    assert "raider_crit" not in ulf_abilities


def test_bodyguard_ward_targets_an_ally(service: GameService, roller):
    make_character(
        service, roller, name="Kell", origin="civilised",
        backgrounds=("bodyguard", "sophist", "bookworm"),
    )
    make_fighter(service, roller)  # mara, the ally
    warded = service.use_ability("kell", "bodyguard_ward", target_id="mara", mode="activate")
    assert warded["ok"] and warded["details"]["target_id"] == "mara"
    kell = service.store.read_character("kell")
    assert any(c.effect_id == "bodyguard_ward" and c.target_id == "mara" for c in kell.conditions)
    # Self-warding and a missing target are refused.
    assert service.use_ability("kell", "bodyguard_ward", target_id="kell")["error"] == "invalid_target"
    assert service.use_ability("kell", "bodyguard_ward")["error"] == "missing_target"
    dropped = service.use_ability("kell", "bodyguard_ward", mode="deactivate")
    assert dropped["ok"] and not service.store.read_character("kell").conditions


def test_bodyguard_absorbs_half_of_a_warded_allys_damage(service: GameService, roller):
    make_character(
        service, roller, name="Kell", origin="civilised",
        backgrounds=("bodyguard", "sophist", "bookworm"),
    )
    make_fighter(service, roller)  # mara, the warded ally
    service.use_ability("kell", "bodyguard_ward", target_id="mara", mode="activate")
    kell_before = service.store.read_character("kell").hp
    npc = open_fight(service, roller)  # combat with mara
    roller.queue(18)  # mara fails the dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["incoming_damage"] == 4
    # The guard takes the larger half (ceil(4/2)=2); mara takes the remainder.
    assert result["applied_damage"] == 2
    assert kell_before - service.store.read_character("kell").hp == 2


def test_bodyguard_rounds_the_absorbed_half_up(service: GameService, roller):
    """On odd damage the guard takes the larger half; a regression to floor would fail."""
    make_character(
        service, roller, name="Kell", origin="civilised",
        backgrounds=("bodyguard", "sophist", "bookworm"),
    )
    make_fighter(service, roller)  # mara, warded
    service.use_ability("kell", "bodyguard_ward", target_id="mara", mode="activate")
    kell_before = service.store.read_character("kell").hp
    npc = open_fight(service, roller, npc_level=2)  # a level-2 NPC deals 5
    roller.queue(18)
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["incoming_damage"] == 5
    assert result["applied_damage"] == 2  # mara takes floor(5/2)
    assert kell_before - service.store.read_character("kell").hp == 3  # guard takes ceil(5/2)


def _warlock(service: GameService, roller: ScriptedRoller) -> dict:
    return make_character(
        service, roller, name="Grim", origin="decadent",
        backgrounds=("warlock", "assassin", "snake-blood"),
    )


def test_warlock_binds_and_invokes_a_demon(service: GameService, roller):
    _warlock(service, roller)
    bound = service.use_ability("grim", "demon:wrath", mode="bind")
    assert bound["ok"] and bound["details"]["bound"] is True
    assert "demon:wrath" in service.store.read_character("grim").powers
    roller.queue(4)  # Doom d6 rolls 4: no step-down, no side effect
    inv = service.use_ability("grim", "demon:wrath", mode="invoke", target_id="npc-cultist")
    assert inv["ok"]
    invocation = inv["details"]["invocation"]
    assert invocation["power"] == "Wrath" and invocation["side_effect"] is False
    assert service.store.read_character("grim").doom_die == "d6"


def test_pact_slots_cap_demon_binding(service: GameService, roller):
    _warlock(service, roller)  # Warlock grants two demon pact slots
    assert service.use_ability("grim", "demon:wrath", mode="bind")["ok"]
    assert service.use_ability("grim", "demon:abyss", mode="bind")["ok"]
    third = service.use_ability("grim", "demon:fear", mode="bind")
    assert third["ok"] is False and third["error"] == "subsystem_full"


def test_binding_requires_the_subsystem(service: GameService, roller):
    make_fighter(service, roller)  # a barbarian with no demonic pact
    result = service.use_ability("mara", "demon:wrath", mode="bind")
    assert result["ok"] is False and result["error"] == "no_subsystem_access"


def test_invoking_an_unbound_demon_is_refused(service: GameService, roller):
    _warlock(service, roller)
    result = service.use_ability("grim", "demon:wrath", mode="invoke")
    assert result["ok"] is False and result["error"] == "power_not_bound"


def test_demon_revenge_applies_on_doom_depletion(service: GameService, roller):
    _warlock(service, roller)
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # Doom d6 -> d4 (a rolled 1 steps it down)
    service.use_ability("grim", "demon:wrath", mode="invoke")
    assert service.store.read_character("grim").doom_die == "d4"
    hp_before = service.store.read_character("grim").hp
    # Deplete the d4 with a 1; Demon's Revenge rolls face 4 -> lose d6 HP (rolled 3).
    roller.queue(1, 4, 3)
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    revenge = inv["details"]["invocation"]["demon_revenge"]
    assert revenge["face"] == 4 and revenge["hp_lost"] == 3
    assert service.store.read_character("grim").doom_die == DEPLETED
    assert hp_before - service.store.read_character("grim").hp == 3


def test_demon_revenge_face_six_breaks_the_pact(service: GameService, roller):
    _warlock(service, roller)
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # d6 -> d4
    service.use_ability("grim", "demon:wrath", mode="invoke")
    hp_before = service.store.read_character("grim").hp
    # Deplete the d4; Demon's Revenge face 6 -> 3d6 damage (2+3+4=9) and the pact breaks.
    roller.queue(1, 6, 2, 3, 4)
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    revenge = inv["details"]["invocation"]["demon_revenge"]
    assert revenge["face"] == 6 and revenge["hp_lost"] == 9 and revenge["pact_broken"] is True
    grim = service.store.read_character("grim")
    assert hp_before - grim.hp == 9
    assert "demon:wrath" not in grim.powers  # the pact is broken


def _warlock_to_revenge_face_two(service: GameService, roller) -> str:
    """Bind Wrath, step the Doom die to d4, then deplete it onto Revenge face 2."""
    _warlock(service, roller)
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # d6 -> d4
    service.use_ability("grim", "demon:wrath", mode="invoke")
    roller.queue(1, 2)  # deplete d4, Demon's Revenge face 2 (steals a possession)
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    assert inv["details"]["adjudication_needed"] is True
    return inv["details"]["invocation"]["demon_revenge"]["ruling_id"]


def test_demon_revenge_face_two_queues_a_pending_ruling(service: GameService, roller):
    ruling_id = _warlock_to_revenge_face_two(service, roller)
    rulings = service.campaign_status()["pending_rulings"]
    assert len(rulings) == 1
    assert rulings[0]["id"] == ruling_id and rulings[0]["kind"] == "demon_revenge_steal"
    assert "long knife" in rulings[0]["options"]  # the caster's possession is stealable


def test_ability_apply_ruling_applies_the_choice_and_is_idempotent(service: GameService, roller):
    ruling_id = _warlock_to_revenge_face_two(service, roller)
    result = service.ability_apply_ruling(ruling_id, choice="long knife", source="model")
    assert result["ok"] and result["details"]["choice"] == "long knife"
    assert "long knife" not in service.store.read_character("grim").weapons  # stolen
    assert service.campaign_status()["pending_rulings"] == []  # cleared
    # Re-applying a resolved ruling is refused, not applied twice.
    again = service.ability_apply_ruling(ruling_id, choice="long knife")
    assert again["ok"] is False and again["error"] == "unknown_ruling"


def test_ability_apply_ruling_falls_to_default_on_a_bad_choice(service: GameService, roller):
    ruling_id = _warlock_to_revenge_face_two(service, roller)
    # An off-list choice falls to the ruling's default strategy (the lone option here).
    result = service.ability_apply_ruling(ruling_id, choice="a nonexistent relic")
    assert result["ok"] and result["details"]["source"] == "default"
    assert result["details"]["choice"] == "long knife"
    assert "long knife" not in service.store.read_character("grim").weapons


def test_ability_apply_ruling_default_picks_from_multiple_options(service: GameService, roller):
    """With several possessions the random default still picks a real, owned option."""
    make_character(
        service, roller, name="Grim", origin="decadent",
        backgrounds=("warlock", "assassin", "snake-blood"), weapons=("long knife", "spear"),
    )
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)
    service.use_ability("grim", "demon:wrath", mode="invoke")
    roller.queue(1, 2)
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    ruling_id = inv["details"]["invocation"]["demon_revenge"]["ruling_id"]
    result = service.ability_apply_ruling(ruling_id, choice="", source="model")
    assert result["ok"] and result["details"]["source"] == "default"
    stolen = result["details"]["choice"]
    assert stolen in ("long knife", "spear")  # a real, owned option
    assert stolen not in service.store.read_character("grim").weapons


def _revenge_face_three(
    service: GameService, roller, allies=(("Sira", ("sword",)),)
) -> dict:
    """Create Grim and each (name, weapons) ally, then deplete the Doom die onto face 3."""
    _warlock(service, roller)
    for ally_name, ally_weapons in allies:
        make_character(service, roller, name=ally_name, weapons=ally_weapons)
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # d6 -> d4
    service.use_ability("grim", "demon:wrath", mode="invoke")
    roller.queue(1, 3)  # deplete d4, Demon's Revenge face 3 (destroys an ally's weapon)
    return service.use_ability("grim", "demon:wrath", mode="invoke")


def test_demon_revenge_face_three_queues_an_ally_weapon_ruling(service: GameService, roller):
    inv = _revenge_face_three(service, roller, allies=(("Sira", ("sword",)),))
    assert inv["ok"] and inv["details"]["adjudication_needed"] is True
    revenge = inv["details"]["invocation"]["demon_revenge"]
    assert revenge["face"] == 3
    assert "no_ally_weapons" not in revenge
    rulings = service.campaign_status()["pending_rulings"]
    assert len(rulings) == 1
    ruling = rulings[0]
    assert ruling["id"] == revenge["ruling_id"]
    assert ruling["kind"] == "demon_revenge_destroy_ally_weapon"
    assert ruling["actor_id"] == "grim"
    assert ruling["default_strategy"] == "random"
    assert ruling["options"] == ["sira: sword"]
    # the invoker's own weapon is excluded, in either form
    assert "long knife" not in ruling["options"]
    assert all(not option.startswith("grim: ") for option in ruling["options"])


def test_face_three_ruling_destroys_only_the_ally_weapon_and_is_idempotent(
    service: GameService, roller
):
    inv = _revenge_face_three(service, roller, allies=(("Sira", ("sword",)),))
    ruling_id = inv["details"]["invocation"]["demon_revenge"]["ruling_id"]
    result = service.ability_apply_ruling(ruling_id, choice="sira: sword", source="model")
    assert result["ok"] and result["details"]["choice"] == "sira: sword"
    assert result["details"]["source"] == "model"
    assert "sword" not in service.store.read_character("sira").weapons  # ally disarmed
    assert service.store.read_character("grim").weapons == ["long knife"]  # invoker untouched
    assert service.campaign_status()["pending_rulings"] == []  # cleared
    # Re-applying a resolved ruling is refused, not applied twice.
    again = service.ability_apply_ruling(ruling_id, choice="sira: sword")
    assert again["ok"] is False and again["error"] == "unknown_ruling"


def test_face_three_off_list_choice_falls_to_the_default_strategy(service: GameService, roller):
    inv = _revenge_face_three(
        service, roller, allies=(("Sira", ("sword",)), ("Brand", ("axe",)))
    )
    ruling_id = inv["details"]["invocation"]["demon_revenge"]["ruling_id"]
    # An off-list choice falls to the ruling's default strategy.
    result = service.ability_apply_ruling(ruling_id, choice="a nonexistent relic")
    assert result["ok"] and result["details"]["source"] == "default"
    chosen = result["details"]["choice"]
    assert chosen in ("sira: sword", "brand: axe")  # a real, owned option
    owner_id, _, weapon = chosen.partition(": ")
    assert weapon not in service.store.read_character(owner_id).weapons
    other_id = "brand" if owner_id == "sira" else "sira"
    other_weapon = "axe" if other_id == "brand" else "sword"
    assert other_weapon in service.store.read_character(other_id).weapons  # untouched
    assert service.store.read_character("grim").weapons == ["long knife"]


def test_face_three_disambiguates_two_allies_holding_the_same_weapon(
    service: GameService, roller
):
    inv = _revenge_face_three(
        service, roller, allies=(("Sira", ("sword",)), ("Brand", ("sword",)))
    )
    revenge = inv["details"]["invocation"]["demon_revenge"]
    ruling_id = revenge["ruling_id"]
    rulings = service.campaign_status()["pending_rulings"]
    ruling = next(r for r in rulings if r["id"] == ruling_id)
    assert set(ruling["options"]) == {"sira: sword", "brand: sword"}
    assert len(ruling["options"]) == 2  # two distinct options, not collapsed
    result = service.ability_apply_ruling(ruling_id, choice="sira: sword")
    assert result["ok"]
    assert service.store.read_character("sira").weapons == []
    assert service.store.read_character("brand").weapons == ["sword"]  # the other ally keeps theirs


def test_face_three_without_an_armed_ally_queues_no_ruling(service: GameService, roller):
    inv = _revenge_face_three(service, roller, allies=(("Sira", ()),))
    assert inv["ok"]
    revenge = inv["details"]["invocation"]["demon_revenge"]
    assert revenge["face"] == 3
    assert revenge["no_ally_weapons"] is True
    assert "ruling_id" not in revenge
    assert "adjudication_needed" not in inv["details"]
    assert revenge["text"]  # the SRD line still reaches the narrator
    assert service.campaign_status()["pending_rulings"] == []  # campaign not wedged


def test_face_three_excludes_a_dead_ally_but_not_a_helpless_one(service: GameService, roller):
    """The oracle for the ally status boundary: dead is excluded, Helpless is not."""
    _warlock(service, roller)
    make_character(service, roller, name="Sira", weapons=("sword",))
    make_character(service, roller, name="Vale", weapons=("axe",))
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    roller.queue(10, 10)  # one initiative d20 per enrolled character
    service.combat_start(
        pc_ids=["sira", "vale"], npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"}, reason="ambush on the plank walk",
    )
    roller.queue(19)
    service.combat_defend("sira", npc["npc_id"], method="dodge", incoming_damage=99)
    roller.queue(6)  # the Helpless table's Killed face
    service.helpless_roll("sira")
    roller.queue(19)
    service.combat_defend("vale", npc["npc_id"], method="dodge", incoming_damage=99)
    assert service.store.read_character("sira").status == "dead"
    assert service.store.read_character("vale").status == "helpless"

    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # d6 -> d4
    service.use_ability("grim", "demon:wrath", mode="invoke")
    roller.queue(1, 3)  # deplete d4, Demon's Revenge face 3
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    revenge = inv["details"]["invocation"]["demon_revenge"]
    rulings = service.campaign_status()["pending_rulings"]
    ruling = next(r for r in rulings if r["id"] == revenge["ruling_id"])
    assert "vale: axe" in ruling["options"]  # Helpless but alive: still an eligible ally
    assert all(not option.startswith("sira: ") for option in ruling["options"])  # dead: excluded


def test_shaman_invokes_a_spirit_with_no_revenge_table(service: GameService, roller):
    make_character(
        service, roller, name="Aya", origin="barbarian",
        backgrounds=("shaman", "hunter", "survivor"),
    )
    assert service.use_ability("aya", "spirit:fire_spirit", mode="bind")["ok"]
    roller.queue(1)  # Doom d6 -> d4, a rolled 1 flags the spirit's side effect
    inv = service.use_ability("aya", "spirit:fire_spirit", mode="invoke")
    invocation = inv["details"]["invocation"]
    assert invocation["power"] == "Fire spirit" and invocation["side_effect"] is True
    assert "demon_revenge" not in invocation  # spirits carry no Revenge table
    assert service.store.read_character("aya").doom_die == "d4"


def _changeling(service: GameService, roller: ScriptedRoller) -> dict:
    return make_character(
        service, roller, name="Sable", origin="decadent",
        backgrounds=("changeling", "assassin", "snake-blood"),
    )


def test_changeling_invokes_a_passive_tie_without_a_doom_roll(service: GameService, roller):
    """Faerie ties carry no shared Doom-die pattern, so a passive tie rolls nothing."""
    _changeling(service, roller)
    assert service.use_ability("sable", "faerie:witchsight", mode="bind")["ok"]
    inv = service.use_ability("sable", "faerie:witchsight", mode="invoke")
    invocation = inv["details"]["invocation"]
    assert invocation["power"] == "Witchsight" and invocation["doom_trigger"] is None
    assert "doom" not in invocation
    assert service.store.read_character("sable").doom_die == "d6"  # the die is untouched


def test_barrow_wisdom_steps_the_doom_die(service: GameService, roller):
    _changeling(service, roller)
    assert service.use_ability("sable", "faerie:barrow_wisdom", mode="bind")["ok"]
    roller.queue(5)  # the compelled answer decreases the Doom die; the step is deterministic
    inv = service.use_ability("sable", "faerie:barrow_wisdom", mode="invoke")
    invocation = inv["details"]["invocation"]
    assert invocation["doom_trigger"] == "step_down"
    assert invocation["doom"]["current_die"] == "d4"
    assert service.store.read_character("sable").doom_die == "d4"


def test_doomed_to_greatness_rolls_the_doom_die_with_advantage(service: GameService, roller):
    _changeling(service, roller)
    assert service.use_ability("sable", "faerie:doomed_to_greatness", mode="bind")["ok"]
    # Advantage on the Doom die keeps the higher face away from 1 and 2: a 1 and a 6 keep
    # the 6, so the die does not step. A single roll would consume only the 1 and step down.
    roller.queue(1, 6, 2)  # the recovery requires two completed long rests
    inv = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    doom = inv["details"]["invocation"]["doom"]
    assert inv["details"]["invocation"]["doom_trigger"] == "roll_advantage"
    assert doom["recovery_long_rests"] == 2
    character = service.store.read_character("sable")
    assert character.doom_die == "d6"
    assert character.faerie_uses_today == {"faerie:doomed_to_greatness": 1}
    assert character.faerie_recovery_rests == {"faerie:doomed_to_greatness": 2}

    same_day = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    assert same_day["ok"] is False and same_day["error"] == "faerie_recovery_pending"
    assert service.rest(["sable"], "long", safe_environment=True)["ok"]
    assert service.store.read_character("sable").faerie_recovery_rests == {
        "faerie:doomed_to_greatness": 1
    }
    assert service.rest(["sable"], "long", safe_environment=True)["ok"]
    assert service.store.read_character("sable").faerie_recovery_rests == {}

    daily_cap = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    assert daily_cap["ok"] is False and daily_cap["error"] == "faerie_daily_limit"
    with service.store.transaction("test_setup", actor_id="sable", reason="depleted daily cap") as transaction:
        character = transaction.character("sable")
        character.doom_die = DEPLETED
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})
    depleted_cap = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    assert depleted_cap["ok"] is False and depleted_cap["error"] == "faerie_daily_limit"
    with service.store.transaction("test_setup", actor_id="sable", reason="restore Doom") as transaction:
        character = transaction.character("sable")
        character.doom_die = "d6"
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})
    assert service.scene_commit("Dawn breaks.", in_game_time_delta_minutes=720)["ok"]
    roller.queue(6, 5, 3)
    next_day = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    assert next_day["ok"] and next_day["details"]["invocation"]["doom"]["recovery_long_rests"] == 3


def test_doomed_to_greatness_rejects_depleted_doom_on_a_fresh_day(service: GameService, roller):
    _changeling(service, roller)
    assert service.use_ability("sable", "faerie:doomed_to_greatness", mode="bind")["ok"]
    assert service.scene_commit("The sun rises.", in_game_time_delta_minutes=1440)["ok"]
    with service.store.transaction("test_setup", actor_id="sable", reason="depleted fresh day") as transaction:
        character = transaction.character("sable")
        character.doom_die = DEPLETED
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    first = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")
    second = service.use_ability("sable", "faerie:doomed_to_greatness", mode="invoke")

    assert first["ok"] is False and first["error"] == "doom_depleted"
    assert second["ok"] is False and second["error"] == "doom_depleted"
    character = service.store.read_character("sable")
    assert character.faerie_uses_today == {}
    assert character.faerie_recovery_rests == {}


def test_elfin_secret_rolls_the_doom_die_only_on_the_day_s_repeat(service: GameService, roller):
    _changeling(service, roller)
    assert service.use_ability("sable", "faerie:elfin_secret", mode="bind")["ok"]
    first = service.use_ability("sable", "faerie:elfin_secret", mode="invoke")
    assert "doom" not in first["details"]["invocation"]  # the day's first use is free
    assert service.store.read_character("sable").doom_die == "d6"
    roller.queue(1)  # the repeat rolls the Doom die; a 1 steps it down
    second = service.use_ability("sable", "faerie:elfin_secret", mode="invoke")
    assert second["details"]["invocation"]["doom"]["current_die"] == "d4"
    assert service.store.read_character("sable").doom_die == "d4"


def _inventor(service: GameService, roller: ScriptedRoller) -> dict:
    return make_character(
        service, roller, name="Cog", origin="civilised",
        backgrounds=("inventor", "sophist", "bookworm"),
    )


def test_inventor_builds_a_marvel_deducting_materials_and_time(service: GameService, roller):
    _inventor(service, roller)  # civilised Inventor: 50 coins, INT 14
    assert service.store.read_character("cog").coins == 50
    assert service.store.read_state().in_game_minutes == 0
    result = service.use_ability("cog", "marvel:acid_spray", mode="bind")  # cost 2
    assert result["ok"] and result["details"]["built"] is True
    assert result["details"]["materials_coins"] == 40  # 20 coins per point
    assert result["details"]["build_weeks"] == 1  # ceil(2 / 14)
    cog = service.store.read_character("cog")
    assert cog.coins == 10 and "marvel:acid_spray" in cog.powers
    assert cog.marvel_usage["marvel:acid_spray"] == "d6"  # a reusable marvel arms its Usage Die
    assert service.store.read_state().in_game_minutes == 7 * 1440  # one week of workshop time


def test_building_a_marvel_without_materials_is_refused(service: GameService, roller):
    _inventor(service, roller)  # 50 coins
    result = service.use_ability("cog", "marvel:bomb", mode="bind")  # cost 4 -> 80 coins
    assert result["ok"] is False and result["error"] == "insufficient_materials"
    cog = service.store.read_character("cog")
    assert cog.coins == 50 and "marvel:bomb" not in cog.powers  # nothing changed
    assert service.store.read_state().in_game_minutes == 0


def test_building_a_marvel_requires_the_inventor_background(service: GameService, roller):
    make_fighter(service, roller)  # a barbarian with no twisted science
    result = service.use_ability("mara", "marvel:acid_spray", mode="bind")
    assert result["ok"] is False and result["error"] == "no_subsystem_access"


def test_single_use_marvel_is_consumed_on_use(service: GameService, roller):
    _inventor(service, roller)
    assert service.use_ability("cog", "marvel:terror_gas_grenade", mode="bind")["ok"]  # cost 2
    inv = service.use_ability("cog", "marvel:terror_gas_grenade", mode="invoke")
    assert inv["ok"] and inv["details"]["invocation"]["consumed"] is True
    cog = service.store.read_character("cog")
    assert "marvel:terror_gas_grenade" not in cog.powers  # a single-use marvel is spent


def test_reusable_marvel_steps_its_usage_die_and_breaks(service: GameService, roller):
    _inventor(service, roller)
    service.use_ability("cog", "marvel:acid_spray", mode="bind")
    roller.queue(1)  # Usage d6 -> d4
    first = service.use_ability("cog", "marvel:acid_spray", mode="invoke")
    assert first["details"]["invocation"]["usage"]["current_die"] == "d4"
    assert service.store.read_character("cog").marvel_usage["marvel:acid_spray"] == "d4"
    roller.queue(1)  # Usage d4 -> depleted; the marvel breaks
    second = service.use_ability("cog", "marvel:acid_spray", mode="invoke")
    assert second["details"]["invocation"]["broken"] is True
    cog = service.store.read_character("cog")
    assert "marvel:acid_spray" not in cog.powers
    assert "marvel:acid_spray" not in cog.marvel_usage


def test_reusable_marvel_rolls_usage_after_a_fight(service: GameService, roller):
    _inventor(service, roller)
    assert service.use_ability("cog", "marvel:acid_spray", mode="bind")["ok"]
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    roller.queue(5)
    assert service.combat_start(["cog"], [npc["npc_id"]], {npc["npc_id"]: "close"})["ok"]
    assert service.combat_begin_turn("cog")["ok"]
    roller.queue(5, 6, 1)  # attack, damage, then the marvel's d6 Usage Die
    # The killing blow ends the fight itself, so the post-combat marvel Usage Die
    # rolls inside the same attack rather than on a combat_end_turn the narrator
    # must remember to make.
    ended = service.combat_attack("cog", npc["npc_id"])
    assert ended["target_npc"]["status"] == "dead"

    assert ended["combat_over"] is True
    assert ended["marvel_usage"][0]["power_id"] == "marvel:acid_spray"
    assert ended["marvel_usage"][0]["usage"]["current_die"] == "d4"
    assert service.store.read_character("cog").marvel_usage["marvel:acid_spray"] == "d4"
    event = [event for event in service.store.read_events(limit=10) if event["tool"] == "combat_attack"][-1]
    assert event["marvel_usage"][0]["usage"]["roll"]["dice"] == [1]


def test_marvel_maintenance_load_counts_only_reusable_held_marvels(service: GameService, roller):
    _inventor(service, roller)
    cog = service.store.read_character("cog")
    cog.powers = ["marvel:firelance", "marvel:bomb"]  # firelance reusable (6); bomb single-use (4)
    # Only the reusable firelance draws maintenance: ceil(6 / 2) = 3 points per week.
    assert service._marvel_maintenance_load(cog) == 3


def test_grant_runic_weapon_sets_the_weapon_and_seeds_the_flag(service: GameService, roller):
    make_fighter(service, roller)  # mara, STR 14
    roller.queue(5, 6, 3)  # 2d6 yields INT 12; the session test succeeds
    result = service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    assert result["ok"] and result["details"]["damage_attribute"] == "STR"
    assert result["details"]["kills_helpless"] is True
    assert result["details"]["weapon_int_roll"] == {
        "notation": "2d6", "dice": [5, 6], "total": 11, "score": 12
    }
    weapon = service.store.read_character("mara").runic_weapon
    assert weapon.name == "Sorrow" and weapon.personality == "brutal"
    assert weapon.weapon_int == 12 and weapon.kills_helpless is True
    event = [event for event in service.store.read_events(limit=10) if event["tool"] == "grant_runic_weapon"][-1]
    assert event["weapon_int_roll"]["dice"] == [5, 6]


def test_grant_runic_weapon_refuses_while_one_is_already_held(service: GameService, roller):
    """Test grant runic weapon refuses while one is already held.
    """
    make_fighter(service, roller)
    roller.queue(5, 6, 1)  # weapon INT 12; session test succeeds
    first = service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    assert first["ok"]
    second = service.grant_runic_weapon("mara", name="Grief", personality="cunning")
    assert second["ok"] is False and second["error"] == "runic_weapon_held"
    assert "Sorrow" in second["message"]
    assert any("inventory_update" in step for step in second["allowed_next_steps"])
    weapon = service.store.read_character("mara").runic_weapon
    assert weapon.name == "Sorrow" and weapon.weapon_int == 12 and weapon.kills_helpless is True
    # Releasing it reopens the slot, which then rolls fresh.
    assert service.inventory_update("mara", reason="cast Sorrow into the river", remove_weapons=["Sorrow"])["ok"]
    roller.queue(1, 1, 20)  # 2d6 total 2 -> INT 8; natural 20 fails the session test
    third = service.grant_runic_weapon("mara", name="Grief", personality="cunning")
    assert third["ok"] and third["details"]["weapon_int"] == 8 and third["details"]["kills_helpless"] is False


def test_grant_runic_weapon_rejects_an_unknown_personality(service: GameService, roller):
    make_fighter(service, roller)
    result = service.grant_runic_weapon("mara", name="Sorrow", personality="merciful")
    assert result["ok"] is False and result["error"] == "invalid_personality"
    assert service.store.read_character("mara").runic_weapon is None


def test_runic_attack_deals_attribute_equal_damage(service: GameService, roller):
    make_fighter(service, roller)  # mara, STR 14
    roller.queue(5, 6, 1)  # grant: weapon INT 12, then a successful session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller, npc_level=3)  # 15 hit points, survives 14 damage
    service.combat_begin_turn("mara")
    roller.queue(2)  # attack test: natural 2, +2 Threat Level = 4, under STR 14 -> a hit
    result = service.combat_attack("mara", npc["npc_id"], runic=True)
    assert result["ok"] and result["applied_damage"] == 14  # damage equals STR, flat, no die
    assert result["damage"]["runic"] is True and result["damage"]["attribute"] == "STR"
    assert result["runic_on_kill"] is None  # the NPC survived, so no on-kill roll


def test_a_runic_attack_without_a_runic_weapon_is_refused(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    result = service.combat_attack("mara", npc["npc_id"], runic=True)
    assert result["ok"] is False and result["error"] == "no_runic_weapon"


def test_a_runic_strike_takes_no_weapon_effect(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 6, 1)
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    result = service.combat_attack("mara", npc["npc_id"], runic=True, weapon_effect="brutal")
    assert result["ok"] is False and result["error"] == "runic_attack_conflict"


def test_a_flagged_runic_weapon_kills_its_helpless_wielder(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 6, 1)  # grant: weapon INT 12 and a successful session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller)
    roller.queue(19)  # the dodge fails; 99 damage drops mara to 0 hit points
    result = service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=99)
    assert result["defender"]["status"] == "dead"  # the runic weapon claims the helpless wielder
    assert any("claims its helpless wielder" in warning for warning in result["warnings"])


def test_an_unflagged_runic_weapon_leaves_its_wielder_helpless(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 6, 20)  # grant: weapon INT 12 and a failed session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller)
    roller.queue(19)
    result = service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=99)
    assert result["defender"]["status"] == "helpless"  # the weapon's test failed, so no kill


def test_session_close_returns_the_runic_weapons_re_roll_for_the_narrator(
    service: GameService, roller
):
    """The weapon's start-of-session INT test is a real, audited d20 whose success
    arms a lethal verdict, and before this the re-roll ``session_close`` makes for the
    next session reached the audit event only, never the caller: the player learned
    the verdict on the day of discovery (``grant_runic_weapon``'s own envelope) and
    never again. The envelope now carries one entry per runic weapon, shaped so
    ``narrator.engine.roll_facts`` reads it the way it reads the grant's own test.
    """
    make_fighter(service, roller)
    roller.queue(5, 6, 1)  # grant: weapon INT 12 and a successful session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    roller.queue(20)  # the next session's test: a critical failure disarms the weapon
    result = service.session_close(
        session_title="The Ashen Bell",
        public_summary="The party found the bell-keeper's body beneath the shrine.",
    )
    assert result["ok"], result
    (entry,) = result["runic_session_tests"]
    assert entry["character_id"] == "mara" and entry["weapon_name"] == "Sorrow"
    assert entry["weapon_int"] == 12
    assert entry["roll"]["selected"] == 20 and entry["roll"]["target"] == 12
    assert entry["outcome"] == "critical_failure" and entry["kills_helpless"] is False
    assert service.store.read_character("mara").runic_weapon.kills_helpless is False
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "session_close"][-1]
    assert event["runic_session_tests"][0]["roll"]["selected"] == 20


def test_inventory_update_relinquishes_a_runic_weapon(service: GameService, roller):
    """A runic weapon is a possession like any other, and ``inventory_update`` is the
    one path possessions move through, so naming it in ``remove_weapons`` must let it
    go. Before this, nothing ever set ``runic_weapon`` back to ``None``: the tool
    refused with ``item_not_held`` (the weapon is a typed field, never an entry in
    ``weapons``), so a player told the blade was armed to kill them that session had
    no way to put it down. The match is by name, case-insensitively, the same rule
    the two lists already use.
    """
    make_fighter(service, roller)
    roller.queue(5, 6, 1)  # grant: weapon INT 12 and a successful (lethal) session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    result = service.inventory_update(
        "mara", reason="left Sorrow in the barrow", remove_weapons=["sorrow"]
    )
    assert result["ok"], result
    assert result["runic_weapon_relinquished"] == "Sorrow"
    assert "Sorrow" in result["summary"]
    assert service.store.read_character("mara").runic_weapon is None
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "inventory_update"][-1]
    assert event["runic_weapon_relinquished"] == "Sorrow"
    assert any("relinquished runic weapon Sorrow" in line for line in result["state_changes"])
    # The blade's lethal verdict leaves with it: the wielder falls Helpless, not dead.
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    refused = service.combat_attack("mara", npc["npc_id"], runic=True)
    assert refused["ok"] is False and refused["error"] == "no_runic_weapon"
    roller.queue(19)
    fallen = service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=99)
    assert fallen["defender"]["status"] == "helpless"


def test_inventory_update_removing_an_unheld_name_leaves_a_runic_weapon_in_hand(
    service: GameService, roller
):
    """The refusal keeps its no-partial-write guarantee: an unheld name beside a
    held runic weapon changes nothing, and the refusal text names the weapon so the
    narrator can pass the exact name."""
    make_fighter(service, roller)
    roller.queue(5, 6, 1)
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    result = service.inventory_update(
        "mara", reason="dropped Grief", remove_weapons=["Grief"]
    )
    assert result["ok"] is False and result["error"] == "item_not_held"
    assert "Sorrow" in result["message"]
    assert service.store.read_character("mara").runic_weapon is not None


def test_runic_on_kill_heal_face_restores_the_wielder(service: GameService, roller):
    make_fighter(service, roller)  # CON 14 -> 14 hit points
    roller.queue(5, 6, 1)  # grant: weapon INT 12 and a successful session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller, npc_level=1)  # 5 hit points, dies to 14 damage
    roller.queue(19)  # a failed dodge drops mara from 14 to 8 before her turn
    service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=6)
    service.combat_begin_turn("mara")
    roller.queue(2, 1, 5)  # attack hit; on-kill d6 = 1 (the heal face); heal d6 = 5
    result = service.combat_attack("mara", npc["npc_id"], runic=True)
    assert result["target_npc"]["status"] == "dead"
    assert result["runic_on_kill"]["face"] == 1 and result["runic_on_kill"]["healed"] == 5
    assert service.store.read_character("mara").hp == 13  # 8 + 5, under the maximum of 14
    event = [event for event in service.store.read_events(limit=10) if event["tool"] == "combat_attack"][-1]
    assert event["runic_on_kill"]["roll"]["dice"] == [1]
    assert event["runic_on_kill"]["healing_roll"]["dice"] == [5]


def test_runic_on_kill_audits_the_raw_heal_when_hit_points_cap(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 6, 1)
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    npc = open_fight(service, roller, npc_level=1)
    service.combat_begin_turn("mara")
    roller.queue(2, 1, 6)  # attack, kill face, then a heal capped at full hit points
    result = service.combat_attack("mara", npc["npc_id"], runic=True)

    assert result["runic_on_kill"]["healed"] == 0
    event = [event for event in service.store.read_events(limit=10) if event["tool"] == "combat_attack"][-1]
    assert event["runic_on_kill"]["roll"]["dice"] == [1]
    assert event["runic_on_kill"]["healing_roll"]["dice"] == [6]


def test_session_close_rerolls_the_runic_helpless_flag(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 6, 20)  # grant: weapon INT 12 and a failed session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    assert service.store.read_character("mara").runic_weapon.kills_helpless is False
    roller.queue(1)  # session_close re-rolls the weapon's INT test: 1 <= 12 succeeds
    service.session_close("Night falls", "The party makes camp.")
    assert service.store.read_character("mara").runic_weapon.kills_helpless is True
    event = [event for event in service.store.read_events(limit=10) if event["tool"] == "session_close"][-1]
    assert event["runic_session_tests"][0]["roll"]["dice"] == [1]


# -- Second Wind and Resourceful: the shared resource_effect dispatch --------


def _gifted_fighter(service: GameService, roller: ScriptedRoller, gift_id: str):
    """A level-3 fighter holding exactly one Gift, for the resource_effect tests."""
    make_fighter(service, roller)
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id=gift_id)
    assert to_level_3["ok"], to_level_3
    return service.store.read_character("mara")


def _set_hp(service: GameService, character_id: str, hp: int) -> None:
    with service.store.transaction(
        "test_setup", actor_id=character_id, reason="wound for a test"
    ) as transaction:
        character = transaction.character(character_id)
        character.hp = hp
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})


def test_second_wind_heals_exactly_level_and_decrements_the_pool(service: GameService, roller):
    character = _gifted_fighter(service, roller, "second-wind")
    assert character.level == 3
    _set_hp(service, "mara", character.hp_max - 10)

    result = service.use_ability("mara", "second_wind")
    assert result["ok"], result
    assert result["details"]["healed"] == 3
    assert result["details"]["remaining"] == 0
    healed = service.store.read_character("mara")
    assert healed.hp == character.hp_max - 10 + 3

    exhausted = service.use_ability("mara", "second_wind")
    assert exhausted["ok"] is False
    assert exhausted["error"] == "ability_exhausted"


def test_second_wind_refreshes_lazily_on_a_day_boundary_only(service: GameService, roller):
    character = _gifted_fighter(service, roller, "second-wind")
    _set_hp(service, "mara", character.hp_max - 5)
    spent = service.use_ability("mara", "second_wind")
    assert spent["ok"], spent
    assert service.store.read_character("mara").pools["second_wind"] == 0

    before_day = service.store.read_state().day

    # A long rest heals fully but must NOT refresh a day-reset pool.
    rested = service.rest(["mara"], rest_type="long", safe_environment=True)
    assert rested["ok"], rested
    assert service.store.read_state().day == before_day, "the long rest crossed a day boundary"
    still_exhausted = service.use_ability("mara", "second_wind")
    assert still_exhausted["ok"] is False
    assert still_exhausted["error"] == "ability_exhausted"

    # Nor does session_close, on its own, refresh it.
    closed = service.session_close(
        "Between adventures", "The party rests uneasily.", accept_uncommitted=True
    )
    assert closed["ok"], closed
    assert service.store.read_state().day == before_day
    still_exhausted_after_close = service.use_ability("mara", "second_wind")
    assert still_exhausted_after_close["ok"] is False
    assert still_exhausted_after_close["error"] == "ability_exhausted"

    # Advancing the in-game day through scene_commit is the only thing that
    # refreshes it, and only lazily, on the next spend attempt.
    _set_hp(service, "mara", character.hp_max - 5)
    advanced = service.scene_commit("The sun rises on a new day.", in_game_time_delta_minutes=1440)
    assert advanced["ok"], advanced
    assert service.store.read_state().day == before_day + 1

    refreshed = service.use_ability("mara", "second_wind")
    assert refreshed["ok"], refreshed
    assert refreshed["details"]["remaining"] == 0
    assert service.store.read_character("mara").pools_day == before_day + 1


def test_second_wind_heal_caps_at_hp_max(service: GameService, roller):
    character = _gifted_fighter(service, roller, "second-wind")
    _set_hp(service, "mara", character.hp_max - 1)  # wounded by less than level

    result = service.use_ability("mara", "second_wind")
    assert result["ok"], result
    assert result["details"]["healed"] == 1
    assert service.store.read_character("mara").hp == character.hp_max


def test_second_wind_refuses_hp_already_full_and_leaves_the_pool_unspent(
    service: GameService, roller
):
    character = _gifted_fighter(service, roller, "second-wind")
    assert character.hp == character.hp_max  # untouched, already full

    result = service.use_ability("mara", "second_wind")
    assert result["ok"] is False
    assert result["error"] == "hp_already_full"
    unspent = service.store.read_character("mara")
    assert unspent.hp == character.hp_max
    assert unspent.pools.get("second_wind", 1) == 1  # the day's only use was not consumed


def test_second_wind_wakes_a_helpless_character(service: GameService, roller):
    _gifted_fighter(service, roller, "second-wind")
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    dropped = service.store.read_character("mara")
    assert dropped.status == "helpless" and dropped.hp == 0

    result = service.use_ability("mara", "second_wind")
    assert result["ok"], result
    woken = service.store.read_character("mara")
    assert woken.status == "ok"
    assert woken.hp == 3  # min(hp_max - 0, level 3)


def test_second_wind_refuses_a_dead_character(service: GameService, roller):
    _gifted_fighter(service, roller, "second-wind")
    with service.store.transaction(
        "test_setup", actor_id="mara", reason="kill for a test"
    ) as transaction:
        character = transaction.character("mara")
        character.hp = 0
        character.status = "dead"
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    result = service.use_ability("mara", "second_wind")
    assert result["ok"] is False
    assert result["error"] == "character_dead"


def test_second_wind_works_with_combat_active(service: GameService, roller):
    character = _gifted_fighter(service, roller, "second-wind")
    open_fight(service, roller)
    assert service.store.read_state().combat.active is True
    _set_hp(service, "mara", character.hp_max - 4)

    result = service.use_ability("mara", "second_wind")
    assert result["ok"], result
    assert result["details"]["healed"] == 3
    assert service.store.read_state().combat.active is True  # use_ability carries no combat gate


def test_a_character_saved_before_pools_day_still_loads(service: GameService, roller):
    from bsh_mcp.models import Character

    make_fighter(service, roller)
    sheet = service.store.read_character("mara").model_dump()
    del sheet["pools_day"]
    reloaded = Character(**sheet)
    assert reloaded.pools_day is None


def test_character_sheet_reports_the_refreshed_second_wind_count_on_a_new_day(
    service: GameService, roller
):
    make_character(
        service, roller, name="Mara", origin="barbarian",
        backgrounds=("legionnaire", "hunter", "survivor"),
    )
    _award_stories(service, "mara", 3)
    service.character_advance("mara", attribute_increases=["STR"])
    service.character_advance("mara", gift_id="second-wind")
    character = service.store.read_character("mara")
    _set_hp(service, "mara", character.hp_max - 5)

    assert service.use_ability("mara", "second_wind")["ok"]
    # Also spend the session-reset legionnaire pool, so the test can confirm the
    # day-scoped reporting rule does not leak onto it.
    assert service.use_ability("mara", "legionnaire_reroll")["ok"]

    day_boundary = service.scene_commit(
        "A new day begins.", in_game_time_delta_minutes=1440
    )
    assert day_boundary["ok"], day_boundary

    sheet = service.character_sheet("mara")["sheet"]
    abilities = {a["ability_id"]: a for a in sheet["abilities"]}
    assert abilities["second_wind"]["remaining"] == 1  # the day pool reports refreshed
    assert abilities["legionnaire_reroll"]["remaining"] == 2  # the session pool does not


def test_resourceful_restores_to_the_recorded_maximum_and_decrements_the_pool(
    service: GameService, roller
):
    _gifted_fighter(service, roller, "resourceful")
    for value in (1, 1):  # d6 -> d4 -> depleted
        roller.queue(value)
        service.usage_roll("mara", "rations", "eat on the road")
    assert service.store.read_character("mara").resources[0].die == DEPLETED

    result = service.use_ability("mara", "resourceful", choice="rations")
    assert result["ok"], result
    assert result["details"]["die"] == "d6"
    assert result["details"]["remaining"] == 0
    restored = service.store.read_character("mara")
    assert restored.resources[0].die == "d6"
    assert restored.resources[0].maximum == "d6"


def test_resourceful_second_spend_refuses_and_session_close_refreshes(
    service: GameService, roller
):
    _gifted_fighter(service, roller, "resourceful")
    roller.queue(1)
    service.usage_roll("mara", "rations", "eat on the road")  # d6 -> d4
    assert service.use_ability("mara", "resourceful", choice="rations")["ok"]

    exhausted = service.use_ability("mara", "resourceful", choice="rations")
    assert exhausted["ok"] is False
    assert exhausted["error"] == "ability_exhausted"

    assert service.session_close(
        "Between adventures", "The party regroups.", accept_uncommitted=True
    )["ok"]
    roller.queue(1)
    service.usage_roll("mara", "rations", "eat on the road")  # back down to d4
    refreshed = service.use_ability("mara", "resourceful", choice="rations")
    assert refreshed["ok"], refreshed


def test_resourceful_refuses_an_unknown_resource(service: GameService, roller):
    _gifted_fighter(service, roller, "resourceful")
    result = service.use_ability("mara", "resourceful", choice="arrows")
    assert result["ok"] is False
    assert result["error"] == "resource_not_found"
    assert service.store.read_character("mara").pools.get("resourceful", 1) == 1


def test_resourceful_refuses_an_already_full_resource_and_leaves_the_pool_unspent(
    service: GameService, roller
):
    _gifted_fighter(service, roller, "resourceful")
    untouched = service.store.read_character("mara")
    assert untouched.resources[0].die == untouched.resources[0].maximum == "d6"

    result = service.use_ability("mara", "resourceful", choice="rations")
    assert result["ok"] is False
    assert result["error"] == "resource_already_full"
    unspent = service.store.read_character("mara")
    assert unspent.resources[0].die == "d6"
    assert unspent.pools.get("resourceful", 1) == 1


def test_resourceful_refuses_a_legacy_depleted_resource_with_no_known_maximum(
    service: GameService, roller
):
    from bsh_mcp.models import UsageResource

    _gifted_fighter(service, roller, "resourceful")
    with service.store.transaction(
        "test_setup", actor_id="mara", reason="seed a legacy depleted resource"
    ) as transaction:
        character = transaction.character("mara")
        legacy = UsageResource(id="oil", name="Lamp oil", die=DEPLETED)
        assert legacy.maximum == ""  # never guessed for an already-depleted legacy resource
        character.resources = character.resources + [legacy]
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    result = service.use_ability("mara", "resourceful", choice="oil")
    assert result["ok"] is False
    assert result["error"] == "unknown_resource_maximum"
    unspent = service.store.read_character("mara")
    assert unspent.resources[1].die == DEPLETED  # still refused, still depleted
    assert unspent.pools.get("resourceful", 1) == 1


def test_resourceful_leaves_the_doom_die_byte_identical(service: GameService, roller):
    _gifted_fighter(service, roller, "resourceful")
    before = service.store.read_character("mara")
    doom_die_before, doom_max_before = before.doom_die, before.doom_max

    # The structural-exclusion witness: Doom is not reachable through `choice` at
    # all, checked first while the pool is still full so the refusal comes from
    # the handler's own lookup rather than the unrelated exhaustion guard.
    doom_attempt = service.use_ability("mara", "resourceful", choice="doom")
    assert doom_attempt["ok"] is False
    assert doom_attempt["error"] == "resource_not_found"

    roller.queue(1)
    service.usage_roll("mara", "rations", "eat on the road")
    result = service.use_ability("mara", "resourceful", choice="rations")
    assert result["ok"], result

    after = service.store.read_character("mara")
    assert after.doom_die == doom_die_before
    assert after.doom_max == doom_max_before


def test_resourceful_restores_a_d8_resource_to_d8_not_d6(service: GameService, roller):
    from bsh_mcp.models import UsageResource

    _gifted_fighter(service, roller, "resourceful")
    with service.store.transaction(
        "test_setup", actor_id="mara", reason="seed a d8 resource"
    ) as transaction:
        character = transaction.character("mara")
        oil = UsageResource(id="oil", name="Lamp oil", die="d8")
        assert oil.maximum == "d8"
        character.resources = character.resources + [oil]
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    for value in (1, 1):  # d8 -> d6 -> d4
        roller.queue(value)
        service.usage_roll("mara", "oil", "burn the lamp")
    assert service.store.read_character("mara").resources[1].die == "d4"

    result = service.use_ability("mara", "resourceful", choice="oil")
    assert result["ok"], result
    assert result["details"]["die"] == "d8"  # not d6, the shipped default
    restored = service.store.read_character("mara")
    assert restored.resources[1].die == "d8"


# -- Battle Hardened: widened critical-success band, in combat only ---------


#: survivor, chieftain, and storyteller are barbarian backgrounds that register no
#: attribute_test/attack/critical_damage/damage_dealt hooks in rules/effects.json.
#: FIGHTER_BACKGROUNDS cannot be reused here: it includes raider, whose raider_crit
#: hooks critical_damage with set_total_to_attribute, which would replace the
#: rolled critical total with the STR score and destroy the
#: maximum-base-plus-extra-die assertion these tests depend on.
BATTLE_HARDENED_BACKGROUNDS = ("survivor", "chieftain", "storyteller")


def _battle_hardened_fighter(service: GameService, roller: ScriptedRoller, **kwargs):
    """A level-3 barbarian holding only Battle Hardened."""
    kwargs.setdefault("backgrounds", BATTLE_HARDENED_BACKGROUNDS)
    make_character(service, roller, **kwargs)
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="battle-hardened")
    assert to_level_3["ok"], to_level_3
    return service.store.read_character("mara")


def test_battle_hardened_attack_crit_on_a_roll_of_three_deals_critical_damage_and_is_audited(
    service: GameService, roller
):
    _battle_hardened_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=4, band="close")
    before_hp = npc["npc"]["hp"]
    service.combat_begin_turn("mara")
    roller.queue(3, 5)  # d20=3, inside the widened band; one additional critical damage die
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["roll"]["dice"] == [3]
    assert result["roll"]["selected"] == 3
    assert result["applied_damage"] == 6 + 5  # max base d6 (default weapon) plus the extra die
    assert result["target_npc"]["hp"] == max(0, before_hp - 11)

    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    assert event["crit_success_max"] == 3
    assert event["roll"]["dice"] == [3]


def test_battle_hardened_natural_one_in_combat_carries_no_audit_key(
    service: GameService, roller
):
    """VO-10's paired negative: a natural 1 crits under the default band alone, so
    D7's audit key would be noise, not an explanation, and stays absent.

    A natural 1 is already a critical success with no Gift in play, so this
    result is unaffected by holding Battle Hardened -- unlike the die-3 case
    above, whose critical the Gift alone produces. The distinguishing evidence
    is the absence of crit_success_max in the audit event, not the outcome.
    """
    _battle_hardened_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=1, band="close")
    service.combat_begin_turn("mara")
    roller.queue(1, 5)  # a natural 1: critical under the default band, needing no widening
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["roll"]["dice"] == [1]
    assert result["applied_damage"] == 6 + 5  # max base d6 plus the extra critical die

    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    assert "crit_success_max" not in event


def test_battle_hardened_does_not_widen_a_plain_attribute_test(service: GameService, roller):
    """The guard that must fail if in_combat leaks outside combat_attack/combat_defend.

    The same character and the same die (3) that crits in combat_attack above must
    resolve ordinarily on a plain attribute_test: the in_combat signal is set only
    by the two combat tools' own internal logic, never derived or inherited. Proven,
    by a manual counterfactual during implementation (temporarily changing
    ``GameService._resolve_character_test``'s ``in_combat`` default to ``True``), to
    fail -- this test then reported ``critical_success`` instead of ``success`` --
    before the default was reverted to ``False``.
    """
    _battle_hardened_fighter(service, roller)
    roller.queue(3)
    result = service.attribute_test(
        "mara", "STR", "arm-wrestle a rival", "the rival concedes", "the rival scoffs"
    )
    assert result["ok"], result
    assert result["outcome"] == "success"  # resolved by total (3 < STR 14), not a critical
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "attribute_test"][-1]
    assert "crit_success_max" not in event


def test_battle_hardened_reads_the_unmodified_die_not_the_total_in_combat(
    service: GameService, roller
):
    """VO-3(b): the widened band reads the kept die, never the Threat-Level total.

    Two attacks land the same total (4) through opposite means -- a low die plus a
    Threat Level penalty, and a higher die with none -- and classify oppositely.
    Threat Level and Doom penalties are never negative in combat (see
    rules.threat_modifier and combat_attack/combat_defend's call_on_doom=False), so
    this same-total pair is the reachable in-combat counterpart of
    tests/test_rules.py's classify(4, 3, target, crit_success_max=3) case.
    """
    _battle_hardened_fighter(service, roller)

    npc_a = open_fight(service, roller, npc_level=4, band="close")  # Threat Level 1
    service.combat_begin_turn("mara")
    roller.queue(3, 5)  # d20=3 + Threat 1 -> total 4; inside the band
    attack_a = service.combat_attack("mara", npc_a["npc_id"], attack_type="melee")
    assert attack_a["ok"], attack_a
    assert attack_a["roll"]["total"] == 4
    assert attack_a["outcome"] == "critical_success"
    service.combat_end_turn("mara")
    closed = service.combat_close("the thug flees")
    assert closed["ok"], closed

    npc_b = open_fight(service, roller, npc_level=3, band="close")  # Threat Level 0
    service.combat_begin_turn("mara")
    roller.queue(4, 5)  # d20=4 + Threat 0 -> total 4; outside the band
    attack_b = service.combat_attack("mara", npc_b["npc_id"], attack_type="melee")
    assert attack_b["ok"], attack_b
    assert attack_b["roll"]["total"] == 4
    assert attack_b["outcome"] == "success"


def test_battle_hardened_defense_crit_on_a_roll_of_three_succeeds_with_no_damage(
    service: GameService, roller
):
    _battle_hardened_fighter(service, roller, shield=True)
    npc = open_fight(service, roller, npc_level=3, band="close")
    before = service.store.read_character("mara")
    roller.queue(3, 12)  # the shield's parry Advantage keeps the lower die, 3
    result = service.combat_defend("mara", attacker_id=npc["npc_id"], method="parry", shield=True)
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["applied_damage"] == 0
    assert result["absorbed"] == 0
    after = service.store.read_character("mara")
    assert after.hp == before.hp
    assert after.shield is True


def test_non_battle_hardened_attack_roll_of_three_is_unaffected(service: GameService, roller):
    make_character(service, roller, backgrounds=BATTLE_HARDENED_BACKGROUNDS)
    npc = open_fight(service, roller, npc_level=1, band="close")
    service.combat_begin_turn("mara")
    roller.queue(3, 5)  # d20=3, but this character holds no Battle Hardened
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "success"
    assert result["applied_damage"] == 5  # one rolled die, no critical bonus
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    assert "crit_success_max" not in event


#: street-urchin (civilised) plus two backgrounds that register no attack,
#: critical_damage, or damage_dealt hook -- bookworm and diplomat only touch
#: character_shape (languages) and a resource pool, so weapon damage stays the
#: default d6, letting a plain scripted comparison stand in for "matches a
#: background-neutral attacker."
STREET_URCHIN_BACKGROUNDS = ("street-urchin", "bookworm", "diplomat")


def test_street_urchin_attack_in_combat_is_undisturbed_by_the_widened_hook_guard(
    service: GameService, roller
):
    """VO-8/D6: the guard change from ``if category:`` to ``if category or
    in_combat:`` now fires the shared attribute_test hook on every combat_attack
    and combat_defend, for every character, not only a declared category.
    street_urchin_streetwise is the one other effect registered on that hook.
    Its primitive (_grant_category_advantage) returns early on a falsy
    ``ctx.category``, so a Street Urchin with no Battle Hardened Gift must see
    the exact same outcome and damage a hook-free background would, and no
    stray Advantage warning (``"Advantage on a  test."``, category="") in
    combat, where category is never declared.
    """
    make_character(service, roller, origin="civilised", backgrounds=STREET_URCHIN_BACKGROUNDS)
    npc = open_fight(service, roller, npc_level=1, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # same scripted roll test_melee_attack_tests_strength_and_applies_damage uses
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "success"
    assert result["applied_damage"] == 4  # one rolled die, matching a background-neutral attacker
    assert not any("advantage on a" in w.lower() for w in result["warnings"])
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    assert "crit_success_max" not in event


def test_battle_hardened_initiative_is_unchanged(service: GameService, roller):
    _battle_hardened_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    assert npc["ok"], npc
    roller.queue(3)  # WIS initiative test: 3 is not critical under the default band
    started = service.combat_start(
        pc_ids=["mara"],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="ambush on the plank walk",
    )
    assert started["ok"], started
    assert started["initiative"][0]["bucket"] == "before"
    assert started["initiative"][0]["first_turn_actions"] == 2  # an ordinary success, not 3
    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_start"][-1]
    assert "crit_success_max" not in event


def _bloodlust_fighter(service: GameService, roller: ScriptedRoller, **kwargs):
    """A level-3 barbarian holding only Bloodlust: weapon d6 -> d8, unarmed d4 -> d6.

    Backgrounds deliberately exclude Raider: its critical_damage hook replaces
    critical damage with the STR score, which would mask the stepped-die crit
    computation these tests exist to pin.
    """
    kwargs.setdefault("backgrounds", BATTLE_HARDENED_BACKGROUNDS)
    make_fighter(service, roller, **kwargs)
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="bloodlust")
    assert to_level_3["ok"], to_level_3
    return service.store.read_character("mara")


def test_bloodlust_hit_rolls_the_stepped_die_and_records_die_stepped(
    service: GameService, roller
):
    _bloodlust_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 7)  # d20=5 hits; the sheet's d6 rolls as 1d8, scripted face 7
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d8"
    assert result["applied_damage"] == 7
    assert result["damage"]["die_stepped"] == {"from": "d6", "to": "d8"}

    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    assert event["damage"]["die_stepped"] == {"from": "d6", "to": "d8"}


def test_bloodlust_steps_the_unarmed_die_too(service: GameService, roller):
    """VO-3: the step reaches whichever die the attack actually rolls, not only weapon_damage."""
    _bloodlust_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # d20=5 hits; the sheet's d4 rolls as 1d6, scripted face 4
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee", unarmed=True)
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d6"
    assert result["applied_damage"] == 4
    assert result["damage"]["die_stepped"] == {"from": "d4", "to": "d6"}


def test_bloodlust_critical_hit_computes_off_the_stepped_die(service: GameService, roller):
    _bloodlust_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=5)
    service.combat_begin_turn("mara")
    roller.queue(1, 5)  # a natural 1 crits; the one additional stepped d8 shows 5
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["damage"]["rolls"][0]["notation"] == "1d8"
    assert result["applied_damage"] == 8 + 5  # max base of the stepped d8, plus the stepped extra die
    assert result["damage"]["die_stepped"] == {"from": "d6", "to": "d8"}


def test_bloodlust_leaves_a_runic_strike_byte_identical(service: GameService, roller):
    """Attribute-fixed damage never rolls a die, so Bloodlust must not touch it.

    Two identically leveled and statted fighters, one holding Bloodlust and one
    holding an unrelated Gift (Armour of scars, which never touches combat_attack),
    resolve an identical runic strike against an identical NPC with identical
    scripted rolls; the persisted audit events -- not just the tool results --
    must match exactly.
    """
    _bloodlust_fighter(service, roller, name="Mara")

    make_fighter(service, roller, name="Ulf", backgrounds=BATTLE_HARDENED_BACKGROUNDS)
    _award_stories(service, "ulf", 3)
    ulf_to_level_2 = service.character_advance("ulf", attribute_increases=["STR"])
    assert ulf_to_level_2["ok"], ulf_to_level_2
    ulf_to_level_3 = service.character_advance("ulf", gift_id="armour-of-scars")
    assert ulf_to_level_3["ok"], ulf_to_level_3

    roller.queue(5, 6, 1)  # grant: weapon INT 12, then a successful session test
    service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    roller.queue(5, 6, 1)
    service.grant_runic_weapon("ulf", name="Sorrow", personality="brutal")

    npc_a = open_fight(service, roller, npc_level=3, pc_id="mara")
    service.combat_begin_turn("mara")
    roller.queue(2)
    result_a = service.combat_attack("mara", npc_a["npc_id"], runic=True)
    assert result_a["ok"], result_a
    # A killing strike ends the fight itself; only a survivor needs combat_close.
    if not result_a["combat_over"]:
        assert service.combat_close("fight a resolved")["ok"]

    npc_b = open_fight(service, roller, npc_level=3, pc_id="ulf")
    service.combat_begin_turn("ulf")
    roller.queue(2)
    result_b = service.combat_attack("ulf", npc_b["npc_id"], runic=True)
    assert result_b["ok"], result_b

    assert result_a["damage"] == result_b["damage"]
    assert result_a["applied_damage"] == result_b["applied_damage"]
    assert "die_stepped" not in result_a["damage"]

    event_a = [
        e for e in service.store.read_events(limit=20)
        if e["tool"] == "combat_attack" and e["actor_id"] == "mara"
    ][-1]
    event_b = [
        e for e in service.store.read_events(limit=20)
        if e["tool"] == "combat_attack" and e["actor_id"] == "ulf"
    ][-1]
    assert event_a["damage"] == event_b["damage"]
    assert event_a["applied_damage"] == event_b["applied_damage"]


def test_bloodlust_leaves_the_assassin_unaware_strike_override_byte_identical(
    service: GameService, roller
):
    """A second attribute-fixed damage path: the Assassin's unaware-strike override."""
    make_assassin(service, roller, name="Mara")
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="bloodlust")
    assert to_level_3["ok"], to_level_3

    # Same level, same attribute increase, an unrelated Gift that never touches
    # combat_attack -- isolates Bloodlust as the only difference between the two.
    make_assassin(service, roller, name="Ulf")
    _award_stories(service, "ulf", 3)
    ulf_to_level_2 = service.character_advance("ulf", attribute_increases=["STR"])
    assert ulf_to_level_2["ok"], ulf_to_level_2
    ulf_to_level_3 = service.character_advance("ulf", gift_id="armour-of-scars")
    assert ulf_to_level_3["ok"], ulf_to_level_3

    npc_a = open_fight(service, roller, npc_level=5, pc_id="mara")
    service.combat_begin_turn("mara")
    roller.queue(5)  # d20 only -- damage is the DEX score, no damage die is rolled
    result_a = service.combat_attack(
        "mara", npc_a["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result_a["ok"], result_a
    assert service.combat_close("fight a resolved")["ok"]

    npc_b = open_fight(service, roller, npc_level=5, pc_id="ulf")
    service.combat_begin_turn("ulf")
    roller.queue(5)
    result_b = service.combat_attack(
        "ulf", npc_b["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result_b["ok"], result_b

    assert result_a["damage"] == result_b["damage"]
    assert result_a["applied_damage"] == result_b["applied_damage"]
    assert "die_stepped" not in result_a["damage"]


def test_bloodlust_character_sheet_still_lists_the_base_dice(service: GameService, roller):
    _bloodlust_fighter(service, roller)
    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 7)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["die_stepped"] == {"from": "d6", "to": "d8"}

    sheet = service.character_sheet("mara")["sheet"]
    assert sheet["weapon_damage"] == "d6"
    assert sheet["unarmed_damage"] == "d4"
    persisted = service.store.read_character("mara")
    assert persisted.weapon_damage == "d6"
    assert persisted.unarmed_damage == "d4"


def _riddle_of_steel_fighter(service: GameService, roller: ScriptedRoller, **kwargs):
    """A level-3 barbarian holding only Riddle of Steel, with two starting weapons.

    Backgrounds deliberately exclude Raider, for the same reason
    ``_bloodlust_fighter`` does: its critical_damage hook would mask the
    critical-hit break tests below.
    """
    kwargs.setdefault("backgrounds", BATTLE_HARDENED_BACKGROUNDS)
    kwargs.setdefault("weapons", ("long knife", "hand axe"))
    make_fighter(service, roller, **kwargs)
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="riddle-of-steel")
    assert to_level_3["ok"], to_level_3
    return service.store.read_character("mara")


def _bloodlust_and_riddle_of_steel_fighter(service: GameService, roller: ScriptedRoller, **kwargs):
    """A level-5 barbarian holding both Bloodlust (3) and Riddle of Steel (5)."""
    kwargs.setdefault("backgrounds", BATTLE_HARDENED_BACKGROUNDS)
    kwargs.setdefault("weapons", ("long knife", "hand axe"))
    make_fighter(service, roller, **kwargs)
    _award_stories(service, "mara", 10)  # enough for level 5, two Gifts
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="bloodlust")
    assert to_level_3["ok"], to_level_3
    to_level_4 = service.character_advance("mara", attribute_increases=["STR", "DEX"])
    assert to_level_4["ok"], to_level_4
    to_level_5 = service.character_advance("mara", gift_id="riddle-of-steel")
    assert to_level_5["ok"], to_level_5
    return service.store.read_character("mara")


def test_riddle_of_steel_designation_refuses_a_weapon_not_held(service: GameService, roller):
    _riddle_of_steel_fighter(service, roller)
    result = service.use_ability("mara", "riddle_of_steel", choice="dagger")
    assert result["ok"] is False
    assert result["error"] == "invalid_intent_choice"
    assert service.store.read_character("mara").declared_choices == {}


def test_riddle_of_steel_designation_lists_weapons_as_the_legal_choices(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    sheet = service.character_sheet("mara")["sheet"]
    ability = next(a for a in sheet["abilities"] if a["ability_id"] == "riddle_of_steel")
    assert set(ability["choices"]) == {"long knife", "hand axe"}
    assert ability["declared"] is None

    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated
    sheet_after = service.character_sheet("mara")["sheet"]
    ability_after = next(a for a in sheet_after["abilities"] if a["ability_id"] == "riddle_of_steel")
    assert ability_after["declared"] == "long knife"


def test_riddle_of_steel_designated_attack_rolls_d12_and_survives_a_non_break_face(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 9)  # d20=5 hits; d12 damage, scripted face 9
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d12"
    assert result["applied_damage"] == 9

    after = service.store.read_character("mara")
    assert "long knife" in after.weapons
    assert after.declared_choices["riddle_of_steel"] == "long knife"


def test_riddle_of_steel_kept_face_of_one_applies_damage_then_breaks_the_weapon(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    before_hp = npc["npc"]["hp"]
    service.combat_begin_turn("mara")
    roller.queue(5, 1)  # d20=5 hits; d12 damage face 1 -- breaks the weapon
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["applied_damage"] == 1
    assert result["target_npc"]["hp"] == before_hp - 1

    after = service.store.read_character("mara")
    assert "long knife" not in after.weapons
    assert "riddle_of_steel" not in after.declared_choices
    assert any("breaks" in warning for warning in result["warnings"])

    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"][-1]
    hp_index = next(i for i, change in enumerate(event["changes"]) if "hit points" in change)
    break_index = next(i for i, change in enumerate(event["changes"]) if "breaks" in change)
    assert hp_index < break_index  # damage reaches the target before the weapon breaks


def test_riddle_of_steel_critical_extra_die_face_of_one_breaks_the_weapon(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=6)
    service.combat_begin_turn("mara")
    roller.queue(1, 1)  # a natural 1 crits; the one additional d12 shows 1
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["applied_damage"] == 12 + 1
    assert "long knife" not in service.store.read_character("mara").weapons


def test_riddle_of_steel_critical_extra_die_face_of_two_spares_the_weapon(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=6)
    service.combat_begin_turn("mara")
    roller.queue(1, 2)  # a natural 1 crits; the one additional d12 shows 2
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["outcome"] == "critical_success"
    assert result["applied_damage"] == 12 + 2
    assert "long knife" in service.store.read_character("mara").weapons


def test_riddle_of_steel_two_handed_kept_die_drives_the_break_not_a_discarded_one(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 1, 8)  # d20=5 hits; two-handed Advantage keeps the higher die, 8
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", two_handed=True
    )
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["dice"] == [1, 8]
    assert result["damage"]["rolls"][0]["selected"] == 8
    assert result["applied_damage"] == 8
    assert "long knife" in service.store.read_character("mara").weapons


def test_riddle_of_steel_withdrawal_via_deactivate_restores_the_ordinary_die(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    withdrawn = service.use_ability("mara", "riddle_of_steel", mode="deactivate")
    assert withdrawn["ok"], withdrawn
    assert "riddle_of_steel" not in service.store.read_character("mara").declared_choices

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # withdrawn: the ordinary d6 weapon die applies
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d6"
    assert result["applied_damage"] == 4
    assert "long knife" in service.store.read_character("mara").weapons


def test_riddle_of_steel_redesignation_onto_another_weapon_after_a_break_succeeds(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 1)  # breaks "long knife"
    broken = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert broken["ok"], broken
    assert "long knife" not in service.store.read_character("mara").weapons
    assert service.combat_close("the broken blade ends the bout")["ok"]

    redesignated = service.use_ability("mara", "riddle_of_steel", choice="hand axe")
    assert redesignated["ok"], redesignated
    assert service.store.read_character("mara").declared_choices["riddle_of_steel"] == "hand axe"

    npc2 = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 7)  # d12 damage on the newly designated hand axe
    result2 = service.combat_attack("mara", npc2["npc_id"], attack_type="melee")
    assert result2["ok"], result2
    assert result2["damage"]["rolls"][0]["notation"] == "1d12"
    assert result2["applied_damage"] == 7


def test_riddle_of_steel_weapon_removed_by_an_unrelated_path_silently_disarms_the_override(
    service: GameService, roller
):
    """Simulate a weapon lost through an unrelated path (e.g. Demon's Revenge face 2/3
    or the Helpless table's equipment loss) by mutating character.weapons directly,
    the same way _set_hp simulates an unrelated wound for other tests.
    """
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    with service.store.transaction(
        "test_setup", actor_id="mara", reason="simulate an unrelated weapon loss"
    ) as transaction:
        character = transaction.character("mara")
        character.weapons = [w for w in character.weapons if w != "long knife"]
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    # The stale designation is still on the mailbox; nothing swept it.
    assert service.store.read_character("mara").declared_choices.get("riddle_of_steel") == (
        "long knife"
    )

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # the override never fires; ordinary d6 weapon die applies
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d6"
    assert result["applied_damage"] == 4


def test_riddle_of_steel_unarmed_attack_ignores_a_standing_designation(
    service: GameService, roller
):
    _riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 1)  # unarmed d4; a scripted 1 cannot break a weapon never used
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", unarmed=True
    )
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d4"
    assert result["applied_damage"] == 1
    assert "long knife" in service.store.read_character("mara").weapons


def test_bloodlust_and_riddle_of_steel_together_still_cap_at_d12(
    service: GameService, roller
):
    """The ceiling interaction: needs both Gifts present on one character."""
    _bloodlust_and_riddle_of_steel_fighter(service, roller)
    designated = service.use_ability("mara", "riddle_of_steel", choice="long knife")
    assert designated["ok"], designated

    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(5, 9)  # d20=5 hits; d12 damage -- Bloodlust's step is a no-op at the ceiling
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["damage"]["rolls"][0]["notation"] == "1d12"
    assert result["applied_damage"] == 9
    assert "die_stepped" not in result["damage"]


def _survivors_luck_fighter(service: GameService, roller: ScriptedRoller, **kwargs):
    """A level-3 barbarian holding only Survivor's Luck, with two starting weapons."""
    kwargs.setdefault("backgrounds", BATTLE_HARDENED_BACKGROUNDS)
    kwargs.setdefault("weapons", ("long knife", "hand axe"))
    make_fighter(service, roller, **kwargs)
    _award_stories(service, "mara", 3)
    to_level_2 = service.character_advance("mara", attribute_increases=["STR"])
    assert to_level_2["ok"], to_level_2
    to_level_3 = service.character_advance("mara", gift_id="survivors-luck")
    assert to_level_3["ok"], to_level_3
    return service.store.read_character("mara")


def test_survivors_luck_declaration_arms_the_stance_and_stakes_a_weapon(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    result = service.use_ability("mara", "survivors_luck", choice="long knife")
    assert result["ok"], result
    assert result["details"]["armed"] is True
    assert result["details"]["choice"] == "long knife"
    assert result["details"]["remaining"] == 0

    mara = service.store.read_character("mara")
    assert any(
        c.effect_id == "survivors_luck" and c.scope == "scene" for c in mara.conditions
    )
    assert mara.declared_choices == {"survivors_luck": "long knife"}
    assert set(mara.weapons) == {"long knife", "hand axe"}  # arming alone costs no weapon

    sheet = service.character_sheet("mara")["sheet"]
    ability = next(a for a in sheet["abilities"] if a["ability_id"] == "survivors_luck")
    assert ability["remaining"] == 0
    assert any(c["effect_id"] == "survivors_luck" for c in sheet["conditions"])


def test_survivors_luck_a_failed_defence_nullifies_damage_and_spends_the_weapon(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")
    hp_before = service.store.read_character("mara").hp

    npc = open_fight(service, roller)
    roller.queue(19)  # a failed dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"], result
    assert result["outcome"] == "failure"
    assert result["applied_damage"] == 0

    mara = service.store.read_character("mara")
    assert mara.hp == hp_before
    assert mara.weapons == ["hand axe"]
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert mara.declared_choices == {}
    assert any("loses" in w and "long knife" in w for w in result["warnings"])

    event = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_defend"][-1]
    assert any("stance survivors_luck ended" in c for c in event["changes"])
    assert any("long knife is lost" in c for c in event["changes"])


def test_survivors_luck_a_successful_defence_leaves_the_stance_armed_for_the_next_failure(
    service: GameService, roller
):
    """First-*failed*-defence binding, not first-defence: a dodge that succeeds
    never even rolls the damage_incoming hook, so the stance survives it."""
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")

    npc = open_fight(service, roller)
    roller.queue(1)  # a successful dodge (a natural 1 crits, and a crit still succeeds)
    passed = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert passed["ok"] and passed["outcome"] in ("success", "critical_success")
    mara = service.store.read_character("mara")
    assert any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert mara.weapons == ["long knife", "hand axe"]

    roller.queue(19)  # now a failed dodge
    failed = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert failed["ok"] and failed["applied_damage"] == 0
    mara = service.store.read_character("mara")
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert mara.weapons == ["hand axe"]


def test_survivors_luck_ability_exhausted_then_session_close_refreshes_it(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")
    npc = open_fight(service, roller)
    roller.queue(19)
    service.combat_defend("mara", npc["npc_id"], method="dodge")  # consumes it

    exhausted = service.use_ability("mara", "survivors_luck", choice="hand axe")
    assert exhausted["ok"] is False and exhausted["error"] == "ability_exhausted"

    service.session_close(
        "Between adventures", "The party counts its scars.", accept_uncommitted=True
    )
    refreshed = service.use_ability("mara", "survivors_luck", choice="hand axe")
    assert refreshed["ok"], refreshed


def test_survivors_luck_already_armed_refuses_a_second_declaration_across_session_close(
    service: GameService, roller
):
    """The stance's scope="scene" condition outlives session_close's pool refresh
    (which only ever clears scope="session" conditions), so a straddling stance
    still refuses a second declaration."""
    _survivors_luck_fighter(service, roller)
    declared = service.use_ability("mara", "survivors_luck", choice="long knife")
    assert declared["ok"], declared

    service.session_close("Between adventures", "The party counts its scars.")
    sheet = service.character_sheet("mara")["sheet"]
    ability = next(a for a in sheet["abilities"] if a["ability_id"] == "survivors_luck")
    assert ability["remaining"] == 1  # session pool refreshed
    mara = service.store.read_character("mara")
    assert any(c.effect_id == "survivors_luck" for c in mara.conditions)  # stance still stands

    again = service.use_ability("mara", "survivors_luck", choice="hand axe")
    assert again["ok"] is False and again["error"] == "already_armed"

    after = service.character_sheet("mara")["sheet"]
    again_ability = next(a for a in after["abilities"] if a["ability_id"] == "survivors_luck")
    assert again_ability["remaining"] == 1  # the refusal spent nothing
    assert service.store.read_character("mara").declared_choices == {"survivors_luck": "long knife"}


def test_survivors_luck_missing_choice_refuses_before_any_state_change(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    result = service.use_ability("mara", "survivors_luck")
    assert result["ok"] is False and result["error"] == "missing_choice"

    mara = service.store.read_character("mara")
    assert mara.conditions == []
    assert mara.declared_choices == {}
    ability = next(
        a for a in service.character_sheet("mara")["sheet"]["abilities"]
        if a["ability_id"] == "survivors_luck"
    )
    assert ability["remaining"] == 1


def test_survivors_luck_weapon_not_held_refuses_before_any_state_change(
    service: GameService, roller
):
    """Test survivors luck weapon not held refuses before any state change.
    """
    _survivors_luck_fighter(service, roller)
    result = service.use_ability("mara", "survivors_luck", choice="dagger")
    assert result["ok"] is False and result["error"] == "weapon_not_held"

    mara = service.store.read_character("mara")
    assert mara.conditions == []
    assert mara.declared_choices == {}
    ability = next(
        a for a in service.character_sheet("mara")["sheet"]["abilities"]
        if a["ability_id"] == "survivors_luck"
    )
    assert ability["remaining"] == 1


def test_survivors_luck_critical_failure_still_rolls_doom_and_breaks_a_parrying_shield(
    service: GameService, roller
):
    """Consequences stand: the Gift ignores damage, not the roll's other costs."""
    _survivors_luck_fighter(service, roller, armour="heavy", shield=True)
    service.use_ability("mara", "survivors_luck", choice="long knife")

    npc = open_fight(service, roller, npc_level=5)
    roller.queue(20, 20, 1)  # a shield-Advantage critical failure, then a Doom roll of 1
    result = service.combat_defend("mara", npc["npc_id"], method="parry", shield=True)
    assert result["ok"], result
    assert result["outcome"] == "critical_failure"
    assert result["applied_damage"] == 0  # the Gift ignores the damage...
    assert result["shield_broken"] is True  # ...but not the shield loss...

    mara = service.store.read_character("mara")
    assert mara.shield is False
    assert mara.doom_die == "d4"  # ...nor the mandatory Doom roll
    assert mara.weapons == ["hand axe"]  # the staked knife is still lost
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)


def test_survivors_luck_bodyguard_warder_takes_nothing_when_the_stance_fires(
    service: GameService, roller
):
    make_character(
        service, roller, name="Kell", origin="civilised",
        backgrounds=("bodyguard", "sophist", "bookworm"),
    )
    _survivors_luck_fighter(service, roller)
    service.use_ability("kell", "bodyguard_ward", target_id="mara", mode="activate")
    service.use_ability("mara", "survivors_luck", choice="long knife")
    kell_before = service.store.read_character("kell").hp

    npc = open_fight(service, roller)
    roller.queue(19)  # a failed dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"], result
    assert result["applied_damage"] == 0
    assert result["absorbed"] == 0
    assert service.store.read_character("kell").hp == kell_before  # nothing to split


def test_survivors_luck_zero_applied_damage_never_causes_helpless(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")
    _set_hp(service, "mara", 1)

    npc = open_fight(service, roller)
    roller.queue(19)  # a failed dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"], result
    assert result["applied_damage"] == 0

    mara = service.store.read_character("mara")
    assert mara.hp == 1
    assert mara.status == "ok"
    assert not any(w for w in result["warnings"] if "Helpless" in w)


def test_survivors_luck_long_rest_clears_an_unconsumed_stance_without_weapon_loss(
    service: GameService, roller
):
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")

    rested = service.rest(["mara"], "long", safe_environment=True, reason="a lull in the fighting")
    assert rested["ok"], rested

    mara = service.store.read_character("mara")
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert set(mara.weapons) == {"long knife", "hand axe"}  # lapsing costs no weapon
    # The session pool is untouched by a rest -- only session_close refreshes it.
    ability = next(
        a for a in service.character_sheet("mara")["sheet"]["abilities"]
        if a["ability_id"] == "survivors_luck"
    )
    assert ability["remaining"] == 0
    # The stake declaration is left standing (inert) rather than swept; nothing
    # reads it once the condition is gone, and the next declaration overwrites it.
    assert mara.declared_choices == {"survivors_luck": "long knife"}


def test_survivors_luck_gate_extension_leaves_legacy_resource_abilities_unchanged(
    service: GameService, roller
):
    """VO-7 at the tool level: extending the stance gate to `resource` must not
    change the three pre-existing hookless resource Gifts' use_ability envelopes
    or exhaustion refusals."""
    _resource_holder(service, roller)
    for expected_left in (2, 1, 0):
        result = service.use_ability("sable", "legionnaire_reroll")
        assert result["ok"] and result["details"]["remaining"] == expected_left
        assert "armed" not in result["details"]
    exhausted = service.use_ability("sable", "legionnaire_reroll")
    assert exhausted["ok"] is False and exhausted["error"] == "ability_exhausted"

    sophist = service.use_ability("sable", "sophist_lie")
    assert sophist["ok"] and sophist["details"]["remaining"] == 0
    bookworm = service.use_ability("sable", "bookworm_substitution")
    assert bookworm["ok"] and bookworm["details"]["remaining"] == 0

    sable = service.store.read_character("sable")
    assert sable.conditions == []  # none of the three ever writes a Condition


def test_survivors_luck_a_harmless_failed_defence_does_not_spend_the_stance(
    service: GameService, roller
):
    """D6: a failed defence armour already absorbed entirely must not burn the
    stance or the staked weapon -- it attaches to the first attack that would
    actually hurt."""
    _survivors_luck_fighter(service, roller, armour="heavy")
    service.use_ability("mara", "survivors_luck", choice="long knife")

    npc = open_fight(service, roller)
    roller.queue(19)  # a failed dodge, but heavy armour (protection 3) absorbs it all
    harmless = service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=3)
    assert harmless["ok"], harmless
    assert harmless["outcome"] == "failure"
    assert harmless["applied_damage"] == 0
    mara = service.store.read_character("mara")
    assert any(c.effect_id == "survivors_luck" for c in mara.conditions)  # still armed
    assert set(mara.weapons) == {"long knife", "hand axe"}  # nothing lost

    roller.queue(19)  # a second failed dodge, this time with real damage
    real = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert real["ok"], real
    assert real["applied_damage"] == 0  # now the stance actually fires
    mara = service.store.read_character("mara")
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert mara.weapons == ["hand axe"]


def test_survivors_luck_a_stake_already_lost_by_another_path_still_nullifies(
    service: GameService, roller
):
    """D8: a stake removed by an unrelated path (a demon's theft, a Riddle of
    Steel break, the Helpless table) still ends the stance and nullifies the
    damage -- there is simply no weapon left to lose. Simulated the same way
    ``_riddle_of_steel_fighter``'s unrelated-path test does."""
    _survivors_luck_fighter(service, roller)
    service.use_ability("mara", "survivors_luck", choice="long knife")

    with service.store.transaction(
        "test_setup", actor_id="mara", reason="simulate an unrelated weapon loss"
    ) as transaction:
        character = transaction.character("mara")
        character.weapons = [w for w in character.weapons if w != "long knife"]
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})

    npc = open_fight(service, roller)
    roller.queue(19)  # a failed dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"], result
    assert result["applied_damage"] == 0

    mara = service.store.read_character("mara")
    assert mara.weapons == ["hand axe"]
    assert not any(c.effect_id == "survivors_luck" for c in mara.conditions)
    assert mara.declared_choices == {}
    assert any("no longer held" in w for w in result["warnings"])


def test_combat_tools_refuse_a_target_outside_the_open_fight(service: GameService, roller):
    """Bystander protection is the combat roster, not the narrator's text heuristics.
    """
    make_fighter(service, roller, backgrounds=("scout", "survivor", "raider"))
    bystander = service.npc_create(name="Orso Pell", level=1, motive="count barrels")
    assert bystander["ok"], bystander
    open_fight(service, roller, band="nearby")
    service.combat_begin_turn("mara")

    attack = service.combat_attack("mara", bystander["npc_id"], attack_type="ranged")
    assert attack["ok"] is False
    assert attack["error"] == "target_not_in_combat"

    move = service.combat_move("mara", bystander["npc_id"])
    assert move["ok"] is False
    assert move["error"] == "target_not_in_combat"

    # The refusals rolled nothing and spent nothing: the whole turn is still open.
    status = service.campaign_status()
    assert status["combat"]["active"] is True
    orso = service.store.read_state().npcs[bystander["npc_id"]]
    assert orso.hp == orso.hp_max


# -- Spirit alliance: Advantage invoking one declared spirit ------------------


def _grant_gift(service: GameService, character_id: str, gift_id: str) -> None:
    """Attach a Gift directly, bypassing the level-3/5/7/9 advancement gate these
    tests do not otherwise exercise -- the same test_setup pattern DEFERRED slices
    already use to force Doom-die/day-boundary state a tool call cannot reach."""
    with service.store.transaction(
        "test_setup", actor_id=character_id, reason=f"grant {gift_id}"
    ) as transaction:
        character = transaction.character(character_id)
        character.gifts = character.gifts + [gift_id]
        transaction.touch_character(character.id)
        transaction.commit({"outcome": "test_setup"})


def _shaman_with_alliance_gift(service: GameService, roller: ScriptedRoller) -> str:
    make_character(
        service, roller, name="Aya", origin="barbarian",
        backgrounds=("shaman", "hunter", "survivor"),
    )
    _grant_gift(service, "aya", "spirit-alliance")
    declared = service.use_ability("aya", "spirit_alliance", choice="ancestor_spirit")
    assert declared["ok"], declared
    return "aya"


def test_spirit_alliance_grants_advantage_on_the_declared_spirits_invocation(
    service: GameService, roller
):
    aya = _shaman_with_alliance_gift(service, roller)
    assert service.use_ability(aya, "spirit:ancestor_spirit", mode="bind")["ok"]
    # Advantage on the Doom die keeps the higher face away from 1 and 2: a 1 and a 6
    # keep the 6, so the die does not step. A single roll would consume only the 1.
    roller.queue(1, 6)
    inv = service.use_ability(aya, "spirit:ancestor_spirit", mode="invoke")
    doom = inv["details"]["invocation"]["doom"]
    assert doom["roll"]["edge"] == "advantage"
    assert doom["downgraded"] is False
    assert service.store.read_character(aya).doom_die == "d6"


def test_spirit_alliance_advantage_is_dormant_for_a_different_invoked_spirit(
    service: GameService, roller
):
    aya = _shaman_with_alliance_gift(service, roller)
    assert service.use_ability(aya, "spirit:fire_spirit", mode="bind")["ok"]
    roller.queue(1)  # a single roll: no Advantage on an undeclared spirit
    inv = service.use_ability(aya, "spirit:fire_spirit", mode="invoke")
    doom = inv["details"]["invocation"]["doom"]
    assert doom["roll"]["edge"] == "single"
    assert doom["downgraded"] is True
    assert service.store.read_character(aya).doom_die == "d4"


# -- Dark revelation / Dubious friendships: Advantage on a backlash table -----


def _forbidden_knowledge(service: GameService, roller: ScriptedRoller, name: str = "Kess") -> str:
    make_character(
        service, roller, name=name, origin="decadent",
        backgrounds=("forbidden-knowledge", "assassin", "snake-blood"),
    )
    return name.lower()


def test_a_sorcery_critical_failure_rolls_torn_veil_and_applies_its_consequence(
    service: GameService, roller
):
    kess = _forbidden_knowledge(service, roller)
    hp_before = service.store.read_character(kess).hp
    roller.queue(20, 4, 2, 3)  # crit failure, mandatory Doom, Torn Veil face 2, d6 HP loss
    result = service.attribute_test(
        kess, "INT", reason="unravel the ward's binding sigil",
        stakes_success="the ward opens without a sound",
        stakes_failure="the ward's backlash tears at her",
        category="sorcery",
    )
    assert result["ok"], result
    assert result["outcome"] == "critical_failure"
    assert result["torn_veil"]["face"] == 2
    assert result["torn_veil"]["hp_lost"] == 3
    assert service.store.read_character(kess).hp == hp_before - 3


def test_torn_veil_face_four_permanently_reduces_maximum_hit_points(
    service: GameService, roller
):
    kess = _forbidden_knowledge(service, roller)
    hp_max_before = service.store.read_character(kess).hp_max
    roller.queue(20, 4, 4)  # crit failure, mandatory Doom, Torn Veil face 4 (no sub-roll)
    result = service.attribute_test(
        kess, "INT", reason="hold the gate shut with will alone",
        stakes_success="the gate seals",
        stakes_failure="the price is paid in flesh",
        category="sorcery",
    )
    assert result["torn_veil"]["hp_max_lost"] == 1
    character = service.store.read_character(kess)
    assert character.hp_max == hp_max_before - 1
    assert character.hp <= character.hp_max


def test_torn_veil_face_five_permanently_reduces_int(service: GameService, roller):
    kess = _forbidden_knowledge(service, roller)
    int_before = service.store.read_character(kess).attributes.INT
    roller.queue(20, 4, 5)  # crit failure, mandatory Doom, Torn Veil face 5 (no sub-roll)
    result = service.attribute_test(
        kess, "INT", reason="read the unbound name aloud",
        stakes_success="the name is known",
        stakes_failure="the name unmakes something of her",
        category="sorcery",
    )
    assert result["torn_veil"]["attribute_lost"] == "INT"
    assert service.store.read_character(kess).attributes.INT == int_before - 1


def test_a_sorcery_success_never_touches_torn_veil(service: GameService, roller):
    """category alone must not roll Torn Veil; only an actual critical failure does."""
    kess = _forbidden_knowledge(service, roller)
    roller.queue(1)  # a natural 1 is a critical success, never critical_failure
    result = service.attribute_test(
        kess, "INT", reason="recall a half-remembered incantation",
        stakes_success="the incantation holds",
        stakes_failure="the words slip away",
        category="sorcery",
    )
    assert result["outcome"] == "critical_success"
    assert "torn_veil" not in result


def test_dark_revelation_rolls_torn_veil_with_advantage(service: GameService, roller):
    kess = _forbidden_knowledge(service, roller)
    _grant_gift(service, kess, "dark-revelation")
    # Advantage keeps the lower, safer face: a 1 and a 6 keep the 1 (no numeric
    # consequence), so no further sub-roll is needed for this assertion.
    roller.queue(20, 4, 1, 6)
    result = service.attribute_test(
        kess, "INT", reason="peer through the veil uninvited",
        stakes_success="the truth is hers",
        stakes_failure="the veil notices her",
        category="sorcery",
    )
    assert result["torn_veil"]["face"] == 1
    assert result["torn_veil"]["roll"]["edge"] == "advantage"
    assert result["torn_veil"]["roll"]["dice"] == [1, 6]


def test_dubious_friendships_rolls_demon_revenge_with_advantage(service: GameService, roller):
    _warlock(service, roller)
    _grant_gift(service, "grim", "dubious-friendships")
    service.use_ability("grim", "demon:wrath", mode="bind")
    roller.queue(1)  # Doom d6 -> d4
    service.use_ability("grim", "demon:wrath", mode="invoke")
    # 1 depletes the d4; the Revenge backlash pool rolls 6 and 1, Advantage keeps 1.
    roller.queue(1, 6, 1)
    inv = service.use_ability("grim", "demon:wrath", mode="invoke")
    revenge = inv["details"]["invocation"]["demon_revenge"]
    assert revenge["face"] == 1
    assert revenge["roll"]["edge"] == "advantage"
    assert revenge["roll"]["dice"] == [6, 1]
