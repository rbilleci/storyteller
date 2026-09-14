"""Pin the pluggable effect registry and its hook dispatch.

Slice 1 migrated the existing automation_tag consumers into the registry with
zero behaviour change; the full tool/rules suites are the behaviour golden. These
tests pin the dispatch mechanism directly: load-time validation, rejection of an
off-vocabulary primitive or hook, priority ordering, and each migrated hook.
"""
from __future__ import annotations

import dataclasses
import random

import pytest
from conftest import REPO_ROOT, ScriptedRoller

from bsh_mcp import effects
from bsh_mcp.data import RulesData, RulesDataError


@pytest.fixture
def data() -> RulesData:
    return RulesData.load(REPO_ROOT / "rules")


def test_effect_registry_loads_and_validates(data: RulesData):
    registry = data.effects["effects"]
    assert registry  # non-empty
    for effect_id, effect in registry.items():
        assert effect["source"], effect_id
        for hook in effect.get("hooks", {}):
            assert hook in effects.HOOKS


def test_unknown_primitive_is_rejected(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={"effects": {"x": {"source": "background:z",
                                   "hooks": {"character_shape": [{"op": "nonexistent"}]}}}},
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_unknown_hook_is_rejected(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={"effects": {"x": {"source": "background:z",
                                   "hooks": {"no_such_hook": [{"op": "grant_advantage"}]}}}},
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_character_shape_derives_dice_and_languages(data: RulesData):
    sources = effects.background_sources(["vicious", "pit-fighter", "diplomat"])
    ctx = effects.apply(
        "character_shape", effects.ShapeContext(languages=["Estuary Cant"]), data.effects, sources
    )
    # Vicious sets weapon d8/unarmed d6 (priority 10); Pit-fighter then matches
    # unarmed to the weapon die (priority 20), so a Vicious Pit-fighter punches d8.
    assert ctx.weapon_damage == "d8"
    assert ctx.unarmed_damage == "d8"
    # Diplomat adds two player-choice language slots.
    assert ctx.languages.count("additional language (player choice)") == 2


def test_priority_orders_vicious_before_pit_fighter(data: RulesData):
    """Reversing the order would leave unarmed at d6, so the golden proves ordering."""
    only_pit = effects.apply(
        "character_shape", effects.ShapeContext(), data.effects,
        effects.background_sources(["pit-fighter"]),
    )
    assert only_pit.unarmed_damage == "d6"  # matches the default weapon d6, not d8


def test_initiative_hook_grants_scout_advantage(data: RulesData):
    scout = effects.apply(
        "initiative", effects.InitiativeContext(), data.effects,
        effects.background_sources(["scout"]),
    )
    assert scout.advantage is True
    plain = effects.apply(
        "initiative", effects.InitiativeContext(), data.effects,
        effects.background_sources(["berserker"]),
    )
    assert plain.advantage is False


def test_rest_hook_allows_wildling_unsafe_long_rest(data: RulesData):
    wildling = effects.apply(
        "rest", effects.RestContext(), data.effects,
        effects.background_sources(["wildling"]),
    )
    assert wildling.allow_unsafe_long_rest is True
    plain = effects.apply(
        "rest", effects.RestContext(), data.effects,
        effects.background_sources(["scout"]),
    )
    assert plain.allow_unsafe_long_rest is False


def test_rest_hook_resolves_a_declared_herbalist_preparation(data: RulesData):
    ctx = effects.apply(
        "rest",
        effects.RestContext(
            environment_tags=["natural"],
            declared_choices={"herbalist_stock": "poison"},
        ),
        data.effects,
        effects.background_sources(["herbalist"]),
    )
    assert ctx.granted == [
        {"effect_id": "herbalist_stock", "dose_type": "poison", "quantity_die": "d6"}
    ]
    assert ctx.notes == []


def test_rest_hook_grants_nothing_without_a_declaration(data: RulesData):
    ctx = effects.apply(
        "rest",
        effects.RestContext(environment_tags=["natural"], declared_choices={}),
        data.effects,
        effects.background_sources(["herbalist"]),
    )
    assert ctx.granted == []
    assert ctx.notes == []


def test_rest_hook_nudges_when_the_scene_lacks_the_required_tag(data: RulesData):
    ctx = effects.apply(
        "rest",
        effects.RestContext(
            environment_tags=[], declared_choices={"herbalist_stock": "healing_balm"}
        ),
        data.effects,
        effects.background_sources(["herbalist"]),
    )
    assert ctx.granted == []
    assert any("natural" in note for note in ctx.notes)


def test_rest_hook_resolution_is_dormant_for_a_non_herbalist(data: RulesData):
    ctx = effects.apply(
        "rest",
        effects.RestContext(
            environment_tags=["natural"], declared_choices={"herbalist_stock": "poison"}
        ),
        data.effects,
        effects.background_sources(["scout"]),
    )
    assert ctx.granted == []
    assert ctx.notes == []


def test_resolve_declared_preparation_rejects_bad_op_params(data: RulesData):
    missing_types = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "rest": [
                            {"op": "resolve_declared_preparation", "quantity_die": "d6"}
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        missing_types._validate_effects()

    bad_die = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "rest": [
                            {
                                "op": "resolve_declared_preparation",
                                "types": ["a"],
                                "quantity_die": "not-a-die",
                            }
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad_die._validate_effects()

    blank_tag = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "rest": [
                            {
                                "op": "resolve_declared_preparation",
                                "types": ["a"],
                                "quantity_die": "d6",
                                "required_environment_tag": "",
                            }
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        blank_tag._validate_effects()


def test_intent_activation_requires_intent_types(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={"effects": {"x": {"source": "background:z", "activation": "intent"}}},
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_dose_activation_requires_a_legal_dose_effects_vocabulary(data: RulesData):
    missing = dataclasses.replace(
        data,
        effects={"effects": {"x": {"source": "background:z", "activation": "dose"}}},
    )
    with pytest.raises(RulesDataError):
        missing._validate_effects()

    bad_value = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "activation": "dose",
                    "dose_effects": {"tonic": "cures_everything"},
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad_value._validate_effects()


# -- Second Wind / Resourceful: the shared resource_effect dispatch --


def test_resource_activation_rejects_an_unknown_resource_effect(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "activation": "resource",
                    "pool": {"max": 1, "reset": "session"},
                    "resource_effect": "heal_everything",
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_resource_activation_allows_no_resource_effect(data: RulesData):
    """The three pre-existing resource entries declare none and must keep validating."""
    plain = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "activation": "resource",
                    "pool": {"max": 1, "reset": "session"},
                }
            }
        },
    )
    plain._validate_effects()  # does not raise


def test_second_wind_and_resourceful_entries_load_as_shipped(data: RulesData):
    registry = data.effects["effects"]

    second_wind = registry["second_wind"]
    assert second_wind["activation"] == "resource"
    assert second_wind["pool"] == {"max": 1, "reset": "day"}
    assert second_wind["resource_effect"] == "heal_level"
    assert "once per" in second_wind["rule"] and "day" in second_wind["rule"]
    assert "combat" in second_wind["rule"]

    resourceful = registry["resourceful"]
    assert resourceful["activation"] == "resource"
    assert resourceful["pool"] == {"max": 1, "reset": "session"}
    assert resourceful["resource_effect"] == "replenish_usage_die"
    assert "use_ability" in resourceful["rule"]
    assert "choice" in resourceful["rule"]
    assert "Doom" in resourceful["rule"]


def test_critical_damage_sets_total_to_str_for_raider(data: RulesData):
    from bsh_mcp.models import Attributes

    attrs = Attributes(STR=14, DEX=10, CON=10, INT=10, WIS=10, CHA=10)
    raider = effects.apply(
        "critical_damage", effects.DamageContext(total=9, attributes=attrs),
        data.effects, effects.background_sources(["raider"]),
    )
    assert raider.total == 14  # STR score replaces the rolled crit
    other = effects.apply(
        "critical_damage", effects.DamageContext(total=9, attributes=attrs),
        data.effects, effects.background_sources(["hunter"]),
    )
    assert other.total == 9  # untouched


def test_damage_incoming_reduces_only_when_unarmoured(data: RulesData):
    scars = {"gift:armour-of-scars"}
    bare = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=5, armour="none"),
        data.effects, scars,
    )
    assert bare.applied == 4  # -1 while unarmoured
    armoured = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=5, armour="light"),
        data.effects, scars,
    )
    assert armoured.applied == 5  # armour suppresses the effect


def test_damage_dealt_adds_rage_die_only_while_active(data: RulesData):
    src = effects.background_sources(["berserker"])
    active = frozenset({"berserker_rage"})
    added = effects.apply(
        "damage_dealt", effects.DamageDealtContext(total=4, roller=ScriptedRoller(rng=random.Random(0), script=[3])),
        data.effects, src, active_ids=active,
    )
    assert added.total == 7  # 4 + rolled 3
    assert added.end_effects == []

    ends = effects.apply(
        "damage_dealt", effects.DamageDealtContext(total=4, roller=ScriptedRoller(rng=random.Random(0), script=[1])),
        data.effects, src, active_ids=active,
    )
    assert ends.total == 5 and ends.end_effects == ["berserker_rage"]
    # Not raging: the toggle does not fire even though the background grants it.
    idle = effects.apply(
        "damage_dealt", effects.DamageDealtContext(total=4, roller=ScriptedRoller(rng=random.Random(0), script=[])),
        data.effects, src, active_ids=frozenset(),
    )
    assert idle.total == 4 and idle.added == []


def test_damage_incoming_halves_while_raging(data: RulesData):
    src = effects.background_sources(["berserker"])
    raging = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=5, armour="none"),
        data.effects, src, active_ids=frozenset({"berserker_rage"}),
    )
    assert raging.applied == 2  # floor(5 * 0.5)
    calm = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=5, armour="none"),
        data.effects, src, active_ids=frozenset(),
    )
    assert calm.applied == 5  # not raging, untouched


def test_helpless_hook_grants_advantage_and_level_hp(data: RulesData):
    both = effects.apply(
        "helpless", effects.HelplessContext(level=3),
        data.effects, {"gift:will-to-live", "gift:tough-as-nails"},
    )
    assert both.advantage is True  # Will to live
    assert both.hp_bonus == 3  # Tough as nails adds the level
    none = effects.apply(
        "helpless", effects.HelplessContext(level=3), data.effects, {"gift:paranoid"},
    )
    assert none.advantage is False and none.hp_bonus == 0


def test_attack_hook_grants_hunter_auto_hit_and_level_bonus(data: RulesData):
    src = effects.background_sources(["hunter"])
    first_shot = effects.apply(
        "attack",
        effects.AttackContext(attack_type="ranged", first_ranged_in_combat=True, level=3),
        data.effects, src,
    )
    assert first_shot.auto_hit is True
    assert first_shot.damage_bonus == 3

    later_shot = effects.apply(
        "attack",
        effects.AttackContext(attack_type="ranged", first_ranged_in_combat=False, level=3),
        data.effects, src,
    )
    assert later_shot.auto_hit is False
    assert later_shot.damage_bonus == 0

    other_background = effects.apply(
        "attack",
        effects.AttackContext(attack_type="ranged", first_ranged_in_combat=True, level=3),
        data.effects, effects.background_sources(["raider"]),
    )
    assert other_background.auto_hit is False
    assert other_background.damage_bonus == 0


def test_attack_hook_substitutes_dex_for_sword_master(data: RulesData):
    src = effects.background_sources(["sword-master"])
    bladed = effects.apply(
        "attack",
        effects.AttackContext(attribute="STR", one_handed_blade=True),
        data.effects, src,
    )
    assert bladed.attribute == "DEX"

    undeclared = effects.apply(
        "attack",
        effects.AttackContext(attribute="STR", one_handed_blade=False),
        data.effects, src,
    )
    assert undeclared.attribute == "STR"

    ranged = effects.apply(
        "attack",
        effects.AttackContext(attribute="DEX", one_handed_blade=True),
        data.effects, src,
    )
    assert ranged.attribute == "DEX"  # nothing to substitute; from STR never matched

    other_background = effects.apply(
        "attack",
        effects.AttackContext(attribute="STR", one_handed_blade=True),
        data.effects, effects.background_sources(["hunter"]),
    )
    assert other_background.attribute == "STR"


def test_attack_hook_overrides_damage_for_assassin_on_unaware_target(data: RulesData):
    src = effects.background_sources(["assassin"])
    qualifying = effects.apply(
        "attack",
        effects.AttackContext(target_unaware_verified=True, target_unaware_declared=True),
        data.effects, src,
    )
    assert qualifying.damage_override_attribute == "DEX"
    assert qualifying.damage_override_effect == "assassin_strike"
    assert qualifying.notes == []


def test_attack_hook_ignores_declaration_outside_the_verified_window(data: RulesData):
    src = effects.background_sources(["assassin"])
    out_of_window = effects.apply(
        "attack",
        effects.AttackContext(target_unaware_verified=False, target_unaware_declared=True),
        data.effects, src,
    )
    assert out_of_window.damage_override_attribute is None
    assert out_of_window.notes == []  # the window being closed carries no nudge


def test_attack_hook_nudges_an_undeclared_assassin_in_the_window(data: RulesData):
    src = effects.background_sources(["assassin"])
    undeclared = effects.apply(
        "attack",
        effects.AttackContext(target_unaware_verified=True, target_unaware_declared=False),
        data.effects, src,
    )
    assert undeclared.damage_override_attribute is None
    assert any("target_unaware" in note for note in undeclared.notes)


def test_attack_hook_override_is_dormant_for_a_non_assassin(data: RulesData):
    other_background = effects.apply(
        "attack",
        effects.AttackContext(target_unaware_verified=True, target_unaware_declared=True),
        data.effects, effects.background_sources(["hunter"]),
    )
    assert other_background.damage_override_attribute is None
    assert other_background.notes == []


def test_override_damage_with_attribute_rejects_an_unknown_attribute(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "attack": [
                            {"op": "override_damage_with_attribute", "attribute": "LUCK"}
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_attribute_test_hook_grants_street_urchin_category_advantage(data: RulesData):
    src = effects.background_sources(["street-urchin"])
    stealth = effects.apply(
        "attribute_test", effects.TestContext(attribute="DEX", category="stealth"),
        data.effects, src,
    )
    assert stealth.advantage is True

    off_category = effects.apply(
        "attribute_test", effects.TestContext(attribute="DEX", category="climbing"),
        data.effects, src,
    )
    assert off_category.advantage is False

    no_category = effects.apply(
        "attribute_test", effects.TestContext(attribute="DEX", category=""),
        data.effects, src,
    )
    assert no_category.advantage is False

    other_background = effects.apply(
        "attribute_test", effects.TestContext(attribute="DEX", category="stealth"),
        data.effects, effects.background_sources(["hunter"]),
    )
    assert other_background.advantage is False


def test_helpless_care_hook_declares_surgeon_int_test_and_d4(data: RulesData):
    surgeon = effects.apply(
        "helpless_care", effects.HelplessCareContext(), data.effects,
        effects.background_sources(["surgeon"]),
    )
    assert surgeon.test_attribute == "INT"
    assert surgeon.die_sides_on_success == 4

    other_background = effects.apply(
        "helpless_care", effects.HelplessCareContext(), data.effects,
        effects.background_sources(["hunter"]),
    )
    assert other_background.test_attribute == ""


def test_grant_category_advantage_rejects_a_category_outside_the_vocabulary(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={
            "test_categories": ["stealth"],
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "attribute_test": [
                            {"op": "grant_category_advantage", "categories": ["climbing"]}
                        ]
                    },
                }
            },
        },
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_tend_helpless_rejects_a_bad_attribute_or_die_sides(data: RulesData):
    bad_attribute = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "helpless_care": [
                            {"op": "tend_helpless", "test_attribute": "LUCK", "die_sides": 4}
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad_attribute._validate_effects()

    bad_sides = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "background:z",
                    "hooks": {
                        "helpless_care": [
                            {"op": "tend_helpless", "test_attribute": "INT", "die_sides": 8}
                        ]
                    },
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad_sides._validate_effects()


# -- Battle Hardened: widened critical-success band, in combat only ---------


def test_battle_hardened_entry_loads_as_shipped(data: RulesData):
    battle_hardened = data.effects["effects"]["battle_hardened"]
    assert battle_hardened["source"] == "gift:battle-hardened"
    assert battle_hardened["activation"] == "passive"
    operations = battle_hardened["hooks"]["attribute_test"]
    assert len(operations) == 1
    op = operations[0]
    assert op["op"] == "widen_critical_success"
    assert op["max_face"] == 3
    assert op["when_in_combat"] is True


def test_widen_critical_success_widens_only_in_combat_for_a_holder(data: RulesData):
    src = {"gift:battle-hardened"}

    in_combat = effects.apply(
        "attribute_test", effects.TestContext(in_combat=True), data.effects, src,
    )
    assert in_combat.crit_success_max == 3

    out_of_combat = effects.apply(
        "attribute_test", effects.TestContext(in_combat=False), data.effects, src,
    )
    assert out_of_combat.crit_success_max == 1

    non_holder = effects.apply(
        "attribute_test",
        effects.TestContext(in_combat=True),
        data.effects,
        effects.background_sources(["raider"]),
    )
    assert non_holder.crit_success_max == 1

    non_holder_gift = effects.apply(
        "attribute_test", effects.TestContext(in_combat=True), data.effects, {"gift:paranoid"},
    )
    assert non_holder_gift.crit_success_max == 1


def test_widen_critical_success_rejects_a_bad_max_face(data: RulesData):
    def with_op(op: dict) -> RulesData:
        return dataclasses.replace(
            data,
            effects={
                "effects": {
                    "x": {
                        "source": "gift:z",
                        "hooks": {"attribute_test": [op]},
                    }
                }
            },
        )

    missing_max_face = with_op({"op": "widen_critical_success", "when_in_combat": True})
    with pytest.raises(RulesDataError):
        missing_max_face._validate_effects()

    non_int_max_face = with_op(
        {"op": "widen_critical_success", "max_face": "three", "when_in_combat": True}
    )
    with pytest.raises(RulesDataError):
        non_int_max_face._validate_effects()

    max_face_reaches_twenty = with_op(
        {"op": "widen_critical_success", "max_face": 20, "when_in_combat": True}
    )
    with pytest.raises(RulesDataError):
        max_face_reaches_twenty._validate_effects()


# -- Bloodlust: permanent damage-die step-up --------------------------------


def test_bloodlust_entry_loads_as_shipped(data: RulesData):
    bloodlust = data.effects["effects"]["bloodlust"]
    assert bloodlust["source"] == "gift:bloodlust"
    assert bloodlust["activation"] == "passive"
    assert "pool" not in bloodlust
    assert "reset" not in bloodlust
    assert "resource_effect" not in bloodlust
    operations = bloodlust["hooks"]["attack"]
    assert len(operations) == 1
    op = operations[0]
    assert op["op"] == "step_up_damage_die"
    assert op["steps"] == 1


def test_step_up_damage_die_accumulates_for_a_holder_and_stays_dormant_otherwise(
    data: RulesData,
):
    holder = effects.apply(
        "attack", effects.AttackContext(), data.effects, {"gift:bloodlust"},
    )
    assert holder.damage_die_steps == 1

    non_holder = effects.apply(
        "attack", effects.AttackContext(), data.effects, {"gift:paranoid"},
    )
    assert non_holder.damage_die_steps == 0


def test_step_up_damage_die_rejects_a_bad_steps_value(data: RulesData):
    def with_op(op: dict) -> RulesData:
        return dataclasses.replace(
            data,
            effects={
                "effects": {
                    "x": {
                        "source": "gift:z",
                        "hooks": {"attack": [op]},
                    }
                }
            },
        )

    with pytest.raises(RulesDataError):
        with_op({"op": "step_up_damage_die", "steps": 0})._validate_effects()
    with pytest.raises(RulesDataError):
        with_op({"op": "step_up_damage_die", "steps": "one"})._validate_effects()
    with_op({"op": "step_up_damage_die"})._validate_effects()  # default 1: no raise
    with_op({"op": "step_up_damage_die", "steps": 2})._validate_effects()  # no raise


# -- Riddle of Steel: standing per-weapon damage-die override + break ------


def test_riddle_of_steel_entry_loads_as_shipped(data: RulesData):
    riddle = data.effects["effects"]["riddle_of_steel"]
    assert riddle["source"] == "gift:riddle-of-steel"
    assert riddle["activation"] == "intent"
    assert riddle["choices_from"] == "weapons"
    assert "pool" not in riddle
    assert "reset" not in riddle
    assert "resource_effect" not in riddle
    operations = riddle["hooks"]["attack"]
    assert len(operations) == 1
    op = operations[0]
    assert op["op"] == "override_damage_die"
    assert op["die"] == "d12"
    assert op["break_on_face"] == 1
    assert op["when_declared_weapon_held"] is True


def test_override_damage_die_fires_only_for_a_held_designated_weapon_on_an_armed_attack(
    data: RulesData,
):
    """The held / not-held / no-designation / unarmed quartet the acceptance names."""
    src = {"gift:riddle-of-steel"}

    held = effects.apply(
        "attack",
        effects.AttackContext(
            weapons_held=["dagger", "sword"],
            declared_choices={"riddle_of_steel": "sword"},
        ),
        data.effects, src,
    )
    assert held.damage_die_override == "d12"
    assert held.weapon_break_face == 1
    assert held.weapon_break_name == "sword"
    assert held.notes == []

    not_held = effects.apply(
        "attack",
        effects.AttackContext(
            weapons_held=["dagger"],
            declared_choices={"riddle_of_steel": "sword"},
        ),
        data.effects, src,
    )
    assert not_held.damage_die_override is None
    assert not_held.weapon_break_face is None
    assert not_held.weapon_break_name == ""

    no_designation = effects.apply(
        "attack",
        effects.AttackContext(weapons_held=["sword"], declared_choices={}),
        data.effects, src,
    )
    assert no_designation.damage_die_override is None
    assert no_designation.weapon_break_face is None
    assert no_designation.weapon_break_name == ""

    unarmed = effects.apply(
        "attack",
        effects.AttackContext(
            unarmed=True,
            weapons_held=["sword"],
            declared_choices={"riddle_of_steel": "sword"},
        ),
        data.effects, src,
    )
    assert unarmed.damage_die_override is None
    assert unarmed.weapon_break_face is None
    assert unarmed.weapon_break_name == ""


def test_override_damage_die_is_dormant_for_a_non_holder(data: RulesData):
    non_holder = effects.apply(
        "attack",
        effects.AttackContext(
            weapons_held=["sword"], declared_choices={"riddle_of_steel": "sword"},
        ),
        data.effects, {"gift:paranoid"},
    )
    assert non_holder.damage_die_override is None
    assert non_holder.weapon_break_name == ""


def test_intent_activation_requires_exactly_one_of_intent_types_or_choices_from(
    data: RulesData,
):
    both = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "gift:z",
                    "activation": "intent",
                    "intent_types": ["a"],
                    "choices_from": "weapons",
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        both._validate_effects()

    neither = dataclasses.replace(
        data,
        effects={"effects": {"x": {"source": "gift:z", "activation": "intent"}}},
    )
    with pytest.raises(RulesDataError):
        neither._validate_effects()


def test_choices_from_rejects_an_unknown_value(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "gift:z",
                    "activation": "intent",
                    "choices_from": "inventory",
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()


def test_override_damage_die_rejects_a_bad_die_or_break_on_face(data: RulesData):
    def with_op(op: dict) -> RulesData:
        return dataclasses.replace(
            data,
            effects={
                "effects": {
                    "x": {
                        "source": "gift:z",
                        "hooks": {"attack": [op]},
                    }
                }
            },
        )

    off_chain_die = with_op({"op": "override_damage_die", "die": "d20"})
    with pytest.raises(RulesDataError):
        off_chain_die._validate_effects()

    missing_die = with_op({"op": "override_damage_die", "break_on_face": 1})
    with pytest.raises(RulesDataError):
        missing_die._validate_effects()

    bad_break_face = with_op(
        {"op": "override_damage_die", "die": "d12", "break_on_face": 13}
    )
    with pytest.raises(RulesDataError):
        bad_break_face._validate_effects()

    valid = with_op({"op": "override_damage_die", "die": "d12", "break_on_face": 1})
    valid._validate_effects()  # does not raise


def test_survivors_luck_entry_loads_as_shipped(data: RulesData):
    luck = data.effects["effects"]["survivors_luck"]
    assert luck["source"] == "gift:survivors-luck"
    assert luck["activation"] == "resource"
    assert luck["pool"] == {"max": 1, "reset": "session"}
    assert luck["choices_from"] == "weapons"
    operations = luck["hooks"]["damage_incoming"]
    assert len(operations) == 1
    op = operations[0]
    assert op["op"] == "nullify_incoming"
    assert op["priority"] == 90
    assert "BEFORE" in luck["rule"]


def test_nullify_incoming_zeroes_damage_and_reports_itself_ended(data: RulesData):
    armed = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=7, armour="none"),
        data.effects, {"gift:survivors-luck"}, active_ids=frozenset({"survivors_luck"}),
    )
    assert armed.applied == 0
    assert armed.end_effects == ["survivors_luck"]


def test_nullify_incoming_is_dormant_until_armed(data: RulesData):
    """The gate's teeth: this fails if the stance-arming gate omits `resource`."""
    unarmed = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=7, armour="none"),
        data.effects, {"gift:survivors-luck"}, active_ids=frozenset(),
    )
    assert unarmed.applied == 7
    assert unarmed.end_effects == []


def test_nullify_incoming_spares_a_stance_when_no_damage_would_land(data: RulesData):
    armed_but_harmless = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=0, armour="none"),
        data.effects, {"gift:survivors-luck"}, active_ids=frozenset({"survivors_luck"}),
    )
    assert armed_but_harmless.applied == 0
    assert armed_but_harmless.end_effects == []  # not spent on a no-op


def test_nullify_incoming_is_dormant_for_a_non_holder(data: RulesData):
    non_holder = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=7, armour="none"),
        data.effects, {"gift:paranoid"}, active_ids=frozenset({"survivors_luck"}),
    )
    assert non_holder.applied == 7
    assert non_holder.end_effects == []


def test_nullify_incoming_wins_over_reduction_and_halving(data: RulesData):
    src = {"gift:armour-of-scars", "background:berserker", "gift:survivors-luck"}
    active = frozenset({"berserker_rage", "survivors_luck"})
    result = effects.apply(
        "damage_incoming", effects.DamageIncomingContext(applied=5, armour="none"),
        data.effects, src, active_ids=active,
    )
    assert result.applied == 0
    assert result.end_effects == ["survivors_luck"]


# -- Dark revelation / Dubious friendships: Advantage on a named backlash table --


def test_dark_revelation_and_dubious_friendships_entries_load_as_shipped(data: RulesData):
    dark_revelation = data.effects["effects"]["dark_revelation"]
    assert dark_revelation["source"] == "gift:dark-revelation"
    assert dark_revelation["activation"] == "passive"
    op = dark_revelation["hooks"]["backlash"][0]
    assert op == {"op": "grant_backlash_advantage", "tables": ["torn_veil"], "priority": 10}

    dubious_friendships = data.effects["effects"]["dubious_friendships"]
    assert dubious_friendships["source"] == "gift:dubious-friendships"
    op = dubious_friendships["hooks"]["backlash"][0]
    assert op == {"op": "grant_backlash_advantage", "tables": ["demon_revenge"], "priority": 10}


def test_grant_backlash_advantage_fires_only_for_the_named_table(data: RulesData):
    src = {"gift:dark-revelation"}
    matching = effects.apply(
        "backlash", effects.BacklashContext(table="torn_veil"), data.effects, src,
    )
    assert matching.advantage is True

    other_table = effects.apply(
        "backlash", effects.BacklashContext(table="demon_revenge"), data.effects, src,
    )
    assert other_table.advantage is False

    non_holder = effects.apply(
        "backlash", effects.BacklashContext(table="torn_veil"), data.effects, {"gift:paranoid"},
    )
    assert non_holder.advantage is False


def test_grant_backlash_advantage_rejects_a_table_outside_the_vocabulary(data: RulesData):
    bad = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "gift:z",
                    "hooks": {"backlash": [{"op": "grant_backlash_advantage", "tables": ["helpless"]}]},
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        bad._validate_effects()

    empty = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {
                    "source": "gift:z",
                    "hooks": {"backlash": [{"op": "grant_backlash_advantage", "tables": []}]},
                }
            }
        },
    )
    with pytest.raises(RulesDataError):
        empty._validate_effects()


# -- Spirit alliance: Advantage invoking one declared spirit ------------------


def test_spirit_alliance_entry_loads_as_shipped(data: RulesData):
    entry = data.effects["effects"]["spirit_alliance"]
    assert entry["source"] == "gift:spirit-alliance"
    assert entry["activation"] == "intent"
    assert entry["choices_from"] == "spirit_alliances"
    op = entry["hooks"]["invoke"][0]
    assert op == {"op": "grant_invocation_advantage", "priority": 10}


def test_choices_from_accepts_spirit_alliances(data: RulesData):
    ok = dataclasses.replace(
        data,
        effects={
            "effects": {
                "x": {"source": "gift:z", "activation": "intent", "choices_from": "spirit_alliances"}
            }
        },
    )
    ok._validate_effects()  # no raise


def test_grant_invocation_advantage_fires_only_for_the_declared_power(data: RulesData):
    src = {"gift:spirit-alliance"}
    declared = {"spirit_alliance": "ancestor_spirit"}

    matching = effects.apply(
        "invoke",
        effects.InvokeContext(power_id="ancestor_spirit", declared_choices=declared),
        data.effects, src,
    )
    assert matching.advantage is True

    other_spirit = effects.apply(
        "invoke",
        effects.InvokeContext(power_id="fire_spirit", declared_choices=declared),
        data.effects, src,
    )
    assert other_spirit.advantage is False

    undeclared = effects.apply(
        "invoke", effects.InvokeContext(power_id="ancestor_spirit"), data.effects, src,
    )
    assert undeclared.advantage is False

    non_holder = effects.apply(
        "invoke",
        effects.InvokeContext(power_id="ancestor_spirit", declared_choices=declared),
        data.effects, {"gift:paranoid"},
    )
    assert non_holder.advantage is False


_HOOK_CONTEXTS: dict[str, type] = {
    "character_shape": effects.ShapeContext,
    "initiative": effects.InitiativeContext,
    "rest": effects.RestContext,
    "damage_dealt": effects.DamageDealtContext,
    "critical_damage": effects.DamageContext,
    "damage_incoming": effects.DamageIncomingContext,
    "helpless": effects.HelplessContext,
    "attack": effects.AttackContext,
    "attribute_test": effects.TestContext,
    "helpless_care": effects.HelplessCareContext,
    "invoke": effects.InvokeContext,
    "backlash": effects.BacklashContext,
}


def test_the_stance_gate_leaves_hookless_resource_effects_unchanged(data: RulesData):
    """Extending the toggle/targeted gate to resource must not alter the three
    pre-existing resource entries, none of which declares hooks at all."""
    for effect_id in ("legionnaire_reroll", "sophist_lie", "bookworm_substitution"):
        entry = data.effects["effects"][effect_id]
        assert entry["activation"] == "resource"
        assert not entry.get("hooks")
        source = entry["source"]
        for hook, context_cls in _HOOK_CONTEXTS.items():
            unarmed = effects.apply(
                hook, context_cls(), data.effects, {source}, active_ids=frozenset(),
            )
            armed = effects.apply(
                hook, context_cls(), data.effects, {source},
                active_ids=frozenset({effect_id}),
            )
            assert unarmed == context_cls() == armed
