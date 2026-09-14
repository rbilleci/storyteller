"""NPC, combat, defence, turn order, combat close, and combat movement tool tests."""

from __future__ import annotations

from conftest import make_character
from tools_shared import (
    ASSASSIN_BACKGROUNDS,
    FIGHTER_BACKGROUNDS,
    _award_stories,
    make_assassin,
    make_fighter,
    open_fight,
)

from bsh_mcp.service import GameService

# -- NPCs and combat ---------------------------------------------------------


def test_npc_create_uses_the_level_table(service: GameService):
    result = service.npc_create(name="Salt Sergeant", level=3, armour="medium", motive="keep order")
    assert result["ok"], result
    assert result["npc"]["hp"] == 18
    assert result["npc"]["damage"] == 6


def test_combat_start_buckets_by_initiative(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller, initiative=5)
    status = service.campaign_status()
    assert status["combat"]["active"] is True
    assert status["combat"]["round"] == 1
    assert status["combat"]["order"][0] == "mara"
    assert npc["npc_id"] in status["combat"]["order"]


def test_critical_initiative_grants_three_actions(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(1)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    started = service.combat_start(["mara"], [npc["npc_id"]], {npc["npc_id"]: "close"})
    assert started["initiative"][0]["first_turn_actions"] == 3


def test_critical_initiative_failure_grants_one_action(service: GameService, roller):
    make_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    roller.queue(20, 4)
    started = service.combat_start(["mara"], [npc["npc_id"]], {npc["npc_id"]: "close"})
    assert started["initiative"][0]["first_turn_actions"] == 1
    assert started["initiative"][0]["bucket"] == "after"


def test_melee_attack_tests_strength_and_applies_damage(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["attribute"] == "STR"
    assert result["applied_damage"] == 4
    assert result["target_npc"]["hp"] == 1


def test_an_overkill_blow_audits_the_real_prior_hit_points(service: GameService, roller):
    """Test an overkill blow audits the real prior hit points.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 6)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["applied_damage"] == 6
    assert result["target_npc"]["hp"] == 0
    attack = next(
        e for e in service.store.read_events(limit=5) if e["tool"] == "combat_attack"
    )
    assert any(
        "hit points 5 -> 0" in change for change in attack["changes"]
    ), attack["changes"]


def test_ranged_attack_tests_dexterity(service: GameService, roller):
    # A Hunter's first ranged attack of a fight auto-hits without a d20 (see
    # test_hunter_first_ranged_attack_auto_hits_without_a_test), so this test pins
    # an ordinary rolled ranged attack with a hunter-free background set.
    make_fighter(service, roller, backgrounds=("scout", "survivor", "raider"))
    npc = open_fight(service, roller, band="nearby")
    service.combat_begin_turn("mara")
    roller.queue(5, 3)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="ranged")
    assert result["attribute"] == "DEX"
    assert result["applied_damage"] == 3
    assert result["auto_hit"] is False
    assert result["roll"]["dice"]


def test_hunter_first_ranged_attack_auto_hits_without_a_test(service: GameService, roller):
    """Test hunter first ranged attack auto hits without a test.
    """
    make_fighter(service, roller)  # FIGHTER_BACKGROUNDS includes hunter
    npc = open_fight(service, roller, band="nearby")
    service.combat_begin_turn("mara")
    roller.queue(3)  # damage die only -- no d20 is rolled for an auto-hit
    result = service.combat_attack("mara", npc["npc_id"], attack_type="ranged")
    assert result["ok"], result
    assert result["auto_hit"] is True
    assert result["outcome"] == "success"
    assert result["roll"]["dice"] == []
    assert result["damage"]["level_bonus"] == 1  # character level 1
    assert result["applied_damage"] == 4  # 3 rolled + 1 level bonus


def test_hunter_second_ranged_attack_of_a_fight_rolls_normally(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="nearby")
    service.combat_begin_turn("mara")
    roller.queue(3)
    first = service.combat_attack("mara", npc["npc_id"], attack_type="ranged")
    assert first["auto_hit"] is True
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    service.combat_end_turn(npc["npc_id"])
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    second = service.combat_attack("mara", npc["npc_id"], attack_type="ranged")
    assert second["auto_hit"] is False
    assert second["roll"]["dice"]
    assert second["applied_damage"] == 4  # no first-arrow bonus on the second shot


def test_hunter_melee_attack_does_not_consume_the_first_ranged_shot(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    melee = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert melee["auto_hit"] is False
    roller.queue(6, 5)  # repeat-action Doom die, then the auto-hit's damage die
    ranged = service.combat_attack("mara", npc["npc_id"], attack_type="ranged")
    assert ranged["auto_hit"] is True
    assert ranged["roll"]["dice"] == []
    assert ranged["repeat_action_doom"] is not None
    assert ranged["applied_damage"] == 6  # 5 rolled + 1 level bonus


def test_sword_master_one_handed_blade_tests_dex(service: GameService, roller):
    make_character(
        service, roller, name="Mara", origin="civilised",
        backgrounds=("sword-master", "bookworm", "diplomat"),
    )
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", one_handed_blade=True
    )
    assert result["ok"], result
    assert result["attribute"] == "DEX"


def test_sword_master_without_the_flag_still_tests_str(service: GameService, roller):
    make_character(
        service, roller, name="Mara", origin="civilised",
        backgrounds=("sword-master", "bookworm", "diplomat"),
    )
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["attribute"] == "STR"


def test_one_handed_blade_without_sword_master_buys_nothing(service: GameService, roller):
    make_fighter(service, roller)  # no sword-master
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", one_handed_blade=True
    )
    assert result["ok"], result
    assert result["attribute"] == "STR"


def test_inconsistent_weapon_declaration_is_refused_without_rolling(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    before = service.store.read_state().event_seq
    two_handed_conflict = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", one_handed_blade=True, two_handed=True
    )
    assert two_handed_conflict["ok"] is False
    assert two_handed_conflict["error"] == "inconsistent_weapon_declaration"
    unarmed_conflict = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", one_handed_blade=True, unarmed=True
    )
    assert unarmed_conflict["ok"] is False
    assert unarmed_conflict["error"] == "inconsistent_weapon_declaration"
    assert service.store.read_state().event_seq == before


def test_assassin_first_strike_on_unaware_target_deals_dex_score(service: GameService, roller):
    """Test assassin first strike on unaware target deals dex score.
    """
    make_assassin(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5)  # d20 only -- damage is the DEX score, no damage die is rolled
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result["ok"], result
    assert result["applied_damage"] == 14  # DEX 14 (13 + assassin's +1)
    assert result["damage"]["override"] == "assassin_strike"
    assert result["damage"]["attribute"] == "DEX"
    assert result["unaware_strike"] is True
    assert result["roll"]["dice"]  # the hit test still rolled normally; only damage is fixed


def test_assassin_second_attempt_this_fight_is_denied_even_on_the_same_fresh_target(
    service: GameService, roller
):
    """The spend is scoped to the attacker, not the target: a second declared strike by
    the SAME assassin against the SAME still-unreacted target is denied, because the
    attacker's own charge (not the target's awareness) is what ran out.
    """
    make_assassin(service, roller)
    npc = open_fight(service, roller, npc_level=4, band="close")  # 20 hp, survives one hit
    service.combat_begin_turn("mara")
    roller.queue(5)
    first = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert first["unaware_strike"] is True
    roller.queue(6, 5, 4)  # repeat-attack Doom d6, then a normal d20 + damage die
    second = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert second["unaware_strike"] is False
    assert "override" not in (second["damage"] or {})
    assert second["applied_damage"] == 4
    assert any("already spent their unaware strike" in w for w in second["warnings"])


def test_two_assassins_can_each_land_the_bonus_on_the_same_still_fresh_target(
    service: GameService, roller
):
    """Test two assassins can each land the bonus on the same still fresh target.
    """
    make_assassin(service, roller, name="Mara")
    make_character(
        service, roller, name="Ulf", origin="decadent", backgrounds=ASSASSIN_BACKGROUNDS,
    )
    npc = service.npc_create(name="Reed Thug", level=8, motive="stand watch")  # 40 hp, avoids morale noise
    assert npc["ok"], npc
    roller.queue(5, 5)  # both PCs' initiative tests
    started = service.combat_start(
        pc_ids=["mara", "ulf"],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="a two-Assassin ambush",
    )
    assert started["ok"], started

    service.combat_begin_turn("mara")
    roller.queue(5)
    first = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert first["unaware_strike"] is True
    assert first["applied_damage"] == 14
    service.combat_end_turn("mara")

    service.combat_begin_turn("ulf")
    roller.queue(5)
    second = service.combat_attack(
        "ulf", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert second["ok"], second
    assert second["unaware_strike"] is True  # still unaware from Ulf's perspective
    assert second["applied_damage"] == 14
    assert second["warnings"] == []


def test_a_non_assassin_allys_earlier_attack_does_not_burn_the_window(
    service: GameService, roller
):
    """Being struck at all does not count as the target reacting -- only the target's
    own turn opening, or the target itself attacking, closes the window. An ordinary
    ally's attack landing first must not cost the Assassin their strike.
    """
    make_character(service, roller, name="Mara", origin="barbarian", backgrounds=FIGHTER_BACKGROUNDS)
    make_character(
        service, roller, name="Ulf", origin="decadent", backgrounds=ASSASSIN_BACKGROUNDS,
    )
    npc = service.npc_create(name="Reed Thug", level=6, motive="stand watch")  # 30 hp, avoids morale noise
    assert npc["ok"], npc
    roller.queue(5, 5)
    started = service.combat_start(
        pc_ids=["mara", "ulf"],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="the fighter swings first",
    )
    assert started["ok"], started

    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    fighter_hit = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert fighter_hit["ok"], fighter_hit
    service.combat_end_turn("mara")

    service.combat_begin_turn("ulf")
    roller.queue(5)
    assassin_hit = service.combat_attack(
        "ulf", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert assassin_hit["ok"], assassin_hit
    assert assassin_hit["unaware_strike"] is True
    assert assassin_hit["applied_damage"] == 14
    assert assassin_hit["warnings"] == []


def test_assassin_without_the_declaration_rolls_normally_and_is_nudged(
    service: GameService, roller
):
    make_assassin(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result
    assert result["unaware_strike"] is False
    assert "override" not in (result["damage"] or {})
    assert any("target_unaware" in w for w in result["warnings"])


def test_assassin_declaration_after_the_target_acted_warns_and_rolls_normally(
    service: GameService, roller
):
    make_assassin(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    service.combat_end_turn(npc["npc_id"])
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result["ok"], result
    assert result["unaware_strike"] is False
    assert any("already reacted" in w for w in result["warnings"])


def test_npc_attack_via_combat_defend_closes_the_unaware_window(
    service: GameService, roller
):
    make_assassin(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    roller.queue(5)  # mara dodges the NPC's attack
    defend = service.combat_defend("mara", attacker_id=npc["npc_id"], method="dodge")
    assert defend["ok"], defend
    roller.queue(5, 4)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result["ok"], result
    assert result["unaware_strike"] is False
    assert any("already reacted" in w for w in result["warnings"])


def test_target_unaware_without_assassin_buys_nothing(service: GameService, roller):
    make_fighter(service, roller)  # hunter set, no assassin
    npc = open_fight(service, roller, npc_level=4, band="close")  # avoid morale noise
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result["ok"], result
    assert result["unaware_strike"] is False
    assert "override" not in (result["damage"] or {})
    assert not any("target_unaware" in w or "already" in w for w in result["warnings"])


def test_unaware_declaration_yields_to_a_runic_strike(service: GameService, roller):
    make_assassin(service, roller)
    roller.queue(5, 6, 3)  # 2d6 weapon INT, then the session test
    grant = service.grant_runic_weapon("mara", name="Sorrow", personality="brutal")
    assert grant["ok"], grant
    npc = open_fight(service, roller, npc_level=3, band="close")  # 15 hp, survives the 13-dmg hit
    service.combat_begin_turn("mara")
    roller.queue(5)  # d20 only -- runic damage is fixed too, no die rolled
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", runic=True, target_unaware=True
    )
    assert result["ok"], result
    assert result["damage"]["runic"] is True
    assert "override" not in result["damage"]
    assert result["unaware_strike"] is False  # runic determined the damage, not the override

    # The attempt still spent the charge: a same-fight non-runic follow-up gets nothing.
    roller.queue(6, 5, 4)
    second = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert second["unaware_strike"] is False
    assert any("already spent their unaware strike" in w for w in second["warnings"])


def test_the_unaware_window_reopens_in_a_new_fight(service: GameService, roller):
    make_assassin(service, roller)
    npc = open_fight(service, roller, npc_level=4, band="close")  # 20 hp
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # undeclared, ordinary attack: hp 20 -> 16
    undeclared = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert undeclared["unaware_strike"] is False
    service.combat_end_turn("mara")
    closed = service.combat_close("the thug is driven off")
    assert closed["ok"], closed

    roller.queue(5)  # fresh fight, fresh initiative
    reopened = service.combat_start(
        pc_ids=["mara"],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="the thug returns",
    )
    assert reopened["ok"], reopened
    service.combat_begin_turn("mara")
    roller.queue(5)
    result = service.combat_attack(
        "mara", npc["npc_id"], attack_type="melee", target_unaware=True
    )
    assert result["unaware_strike"] is True
    assert result["applied_damage"] == 14


def test_melee_against_a_distant_target_is_refused_without_rolling(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="far_away")
    service.combat_begin_turn("mara")
    before = service.store.read_state().event_seq
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"] is False
    assert result["error"] == "target_out_of_reach"
    assert service.store.read_state().event_seq == before


def test_repeating_the_attack_action_rolls_doom(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 2)
    service.combat_attack("mara", npc["npc_id"])
    roller.queue(2, 5, 2)
    second = service.combat_attack("mara", npc["npc_id"])
    assert second["repeat_action_doom"]["current_die"] == "d4"
    assert service.store.read_character("mara").doom_die == "d4"


def test_a_third_attack_is_refused_when_actions_run_out(service: GameService, roller):
    """Test a third attack is refused when actions run out.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 1)
    service.combat_attack("mara", npc["npc_id"])
    roller.queue(6, 5, 1)
    second = service.combat_attack("mara", npc["npc_id"])
    assert second["turn_advanced"] is True
    assert second["next_actor"] == npc["npc_id"]
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "turn_not_open"
    assert npc["npc_id"] in result["message"]


def test_raider_critical_hit_deals_str_score_damage(service: GameService, roller):
    """The Raider effect overrides a critical hit's damage with the STR score.

    make_fighter carries the raider background, and raider's +1 lifts STR to 14, so
    the registry's critical_damage hook replaces the rolled crit with 14. This is
    the Slice-2 combat hook firing through the real combat_attack path.
    """
    make_fighter(service, roller)  # backgrounds include raider; STR 14
    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(1, 4)
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["outcome"] == "critical_success"
    assert result["applied_damage"] == 14  # STR score, not the rolled 10


def test_critical_hit_deals_maximum_base_damage_plus_one_die(service: GameService, roller):
    """A non-Raider critical hit keeps the base mechanic: max weapon die plus one die."""
    make_character(
        service, roller, name="Mara", origin="barbarian",
        backgrounds=("hunter", "survivor", "scout"),
    )
    npc = open_fight(service, roller, npc_level=3)
    service.combat_begin_turn("mara")
    roller.queue(1, 4)
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["outcome"] == "critical_success"
    assert result["applied_damage"] == 10  # max d6 (6) + one rolled die (4)


def test_an_npc_at_zero_hit_points_is_dead(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 6)
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["target_npc"]["status"] == "dead"
    assert result["target_npc"]["hp"] == 0


def test_an_unimplemented_weapon_effect_is_refused_before_any_roll(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    before = service.store.read_state().event_seq
    result = service.combat_attack("mara", npc["npc_id"], weapon_effect="poison")
    assert result["ok"] is False
    assert result["error"] == "weapon_effect_not_implemented"
    assert "brutal" in result["allowed_next_steps"][0]
    assert service.store.read_state().event_seq == before


def test_shove_moves_the_target_one_band_and_deals_no_damage(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5)
    result = service.combat_attack("mara", npc["npc_id"], weapon_effect="shove")
    assert result["applied_damage"] == 0
    assert result["target_npc"]["hp"] == 5
    assert service.store.read_state().combat.ranges[npc["npc_id"]] == "nearby"


def test_disarm_reduces_the_npc_damage_output(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5)
    result = service.combat_attack("mara", npc["npc_id"], weapon_effect="disarm")
    assert "disarmed" in result["target_npc"]["flags"]
    assert result["target_npc"]["damage"] == 2


def test_entangle_flags_the_target_and_deals_no_damage(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5)
    result = service.combat_attack("mara", npc["npc_id"], weapon_effect="entangle")
    assert result["applied_damage"] == 0
    assert "entangled" in result["target_npc"]["flags"]


def _open_fight_with_two_npcs(
    service: GameService, roller, *, initiative: int = 5, band_a: str = "close",
    band_b: str = "close",
) -> tuple[dict, dict]:
    npc_a = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    npc_b = service.npc_create(name="Second Thug", level=1, motive="rob the party")
    assert npc_a["ok"] and npc_b["ok"]
    roller.queue(initiative)
    started = service.combat_start(
        pc_ids=["mara"],
        npc_ids=[npc_a["npc_id"], npc_b["npc_id"]],
        initial_ranges={npc_a["npc_id"]: band_a, npc_b["npc_id"]: band_b},
        reason="ambush by two thugs",
    )
    assert started["ok"], started
    return npc_a, npc_b


def test_cleave_hits_every_other_enemy_in_the_targets_band(service: GameService, roller):
    make_fighter(service, roller)
    npc_a, npc_b = _open_fight_with_two_npcs(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # to-hit, damage die
    result = service.combat_attack("mara", npc_a["npc_id"], weapon_effect="cleave")
    assert result["applied_damage"] == 4
    assert result["target_npc"]["hp"] == 1
    assert result["secondary_hits"] == [
        {"target_id": npc_b["npc_id"], "applied": 4, "dead": False}
    ]
    assert service.store.read_state().npcs[npc_b["npc_id"]].hp == 1


def test_cleave_does_not_reach_an_enemy_in_a_different_band(service: GameService, roller):
    make_fighter(service, roller)
    npc_a, npc_b = _open_fight_with_two_npcs(service, roller, band_a="close", band_b="nearby")
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack("mara", npc_a["npc_id"], weapon_effect="cleave")
    assert result["secondary_hits"] == []
    assert service.store.read_state().npcs[npc_b["npc_id"]].hp == 5


def test_impale_carries_through_to_one_more_enemy_only_on_a_kill(service: GameService, roller):
    make_fighter(service, roller)
    npc_a, npc_b = _open_fight_with_two_npcs(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 6)  # to-hit, damage die -- 6 damage fells the 5-hp target
    result = service.combat_attack("mara", npc_a["npc_id"], weapon_effect="impale")
    assert result["applied_damage"] == 6
    assert result["target_npc"]["hp"] == 0
    assert result["secondary_hits"] == [
        {"target_id": npc_b["npc_id"], "applied": 6, "dead": True}
    ]
    assert service.store.read_state().npcs[npc_b["npc_id"]].status == "dead"


def test_impale_carries_through_nothing_when_the_target_survives(service: GameService, roller):
    make_fighter(service, roller)
    npc_a, npc_b = _open_fight_with_two_npcs(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)  # 4 damage leaves the 5-hp target alive
    result = service.combat_attack("mara", npc_a["npc_id"], weapon_effect="impale")
    assert result["secondary_hits"] == []
    assert service.store.read_state().npcs[npc_b["npc_id"]].hp == 5


# -- defence -----------------------------------------------------------------


def test_a_failed_dodge_applies_damage_after_armour(service: GameService, roller):
    make_fighter(service, roller, armour="medium")
    npc = open_fight(service, roller)
    roller.queue(18)
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["outcome"] == "failure"
    assert result["incoming_damage"] == 4
    assert result["applied_damage"] == 2
    assert result["absorbed"] == 2


def test_a_critical_defence_failure_ignores_armour_and_breaks_the_shield(
    service: GameService, roller
):
    make_fighter(service, roller, armour="heavy", shield=True)
    npc = open_fight(service, roller, npc_level=5)
    roller.queue(20, 20)
    result = service.combat_defend(
        "mara", npc["npc_id"], method="parry", shield=True
    )
    assert result["outcome"] == "critical_failure"
    assert result["applied_damage"] == 8
    assert result["absorbed"] == 0
    assert result["shield_broken"] is True
    assert service.store.read_character("mara").shield is False


def test_armour_of_scars_reduces_unarmoured_damage(service: GameService, roller):
    """The Slice-2 damage_incoming hook fires end-to-end through combat_defend.

    Armour of scars reduces incoming damage by 1 while unarmoured; a level-1 NPC's
    4 damage lands as 3 on a bare-skinned character carrying the Gift.
    """
    make_fighter(service, roller)  # armour none by default
    _award_stories(service, "mara", 3)
    service.character_advance("mara", attribute_increases=["STR"])  # to level 2
    service.character_advance("mara", gift_id="armour-of-scars")  # to level 3, takes the Gift
    npc = open_fight(service, roller)
    roller.queue(18)  # a failed dodge
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["outcome"] == "failure"
    assert result["incoming_damage"] == 4
    assert result["applied_damage"] == 3  # 4 base (no armour) minus 1 from Armour of scars


def test_a_shield_grants_advantage_when_parrying(service: GameService, roller):
    make_fighter(service, roller, shield=True)
    npc = open_fight(service, roller)
    roller.queue(19, 2)
    result = service.combat_defend("mara", npc["npc_id"], method="parry", shield=True)
    assert result["roll"]["dice"] == [19, 2]
    assert result["roll"]["selected"] == 2
    assert result["outcome"] == "success"


def test_a_ranged_attack_cannot_be_parried(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    result = service.combat_defend("mara", npc["npc_id"], method="parry", ranged=True)
    assert result["ok"] is False
    assert result["error"] == "parry_against_ranged"


def test_parry_requires_something_to_hold(service: GameService, roller):
    make_character(
        service,
        roller,
        name="Bare",
        origin="barbarian",
        backgrounds=FIGHTER_BACKGROUNDS,
        weapons=(),
    )
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    result = service.combat_defend("bare", npc["npc_id"], method="parry")
    assert result["ok"] is False
    assert result["error"] == "nothing_to_parry_with"


def test_defence_does_not_consume_actions(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(3)
    service.combat_defend("mara", npc["npc_id"], method="dodge")
    combat = service.store.read_state().combat
    assert combat.actors["mara"].actions_used == 0


def test_a_player_character_at_zero_hit_points_becomes_helpless(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(19)
    result = service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=99)
    assert result["defender"]["status"] == "helpless"
    assert result["defender"]["hp"] == 0
    assert any("Helpless" in warning for warning in result["warnings"])


def test_a_helpless_character_takes_no_further_actions(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(19)
    service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=99)
    result = service.attribute_test("mara", "STR", "crawl away", "the way opens", "the cost lands")
    assert result["ok"] is False
    assert result["error"] == "character_helpless"


# -- turn order --------------------------------------------------------------


def test_combat_ends_when_the_last_enemy_falls(service: GameService, roller):
    """The killing blow closes the fight itself: no combat_end_turn call the
    narrator must remember stands between the last enemy dropping and the fight
    being over."""
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 6)
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["target_npc"]["status"] == "dead"
    assert result["combat_over"] is True
    assert "Combat is over" in result["next_step"]
    assert service.store.read_state().combat.active is False


def test_turn_order_advances_and_increments_the_round(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    ended = service.combat_end_turn("mara")
    assert ended["next_actor"] == npc["npc_id"]
    assert ended["round"] == 1
    service.combat_begin_turn(npc["npc_id"])
    wrapped = service.combat_end_turn(npc["npc_id"])
    assert wrapped["next_actor"] == "mara"
    assert wrapped["round"] == 2


def test_every_combat_result_names_the_next_call(service: GameService, roller):
    """A narrator that follows next_step never loses the turn order."""
    make_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    roller.queue(5)
    started = service.combat_start(["mara"], [npc["npc_id"]], {npc["npc_id"]: "close"})
    # The winner's turn is already open; next_step says how to spend it.
    assert "mara" in started["next_step"]
    assert "already" in started["next_step"]

    opened = service.combat_begin_turn("mara")
    assert "combat_attack" in opened["next_step"]

    roller.queue(5, 2)
    attacked = service.combat_attack("mara", npc["npc_id"])
    assert "action(s) left" in attacked["next_step"]

    roller.queue(5, 2, 2)
    exhausted = service.combat_attack("mara", npc["npc_id"])
    # The last action advanced the turn itself; next_step hands play to the enemy.
    assert exhausted["turn_advanced"] is True
    assert npc["npc_id"] in exhausted["next_step"]
    assert "combat_defend" in exhausted["next_step"]

    roller.queue(3)
    defended = service.combat_defend("mara", npc["npc_id"], method="dodge")
    # The defence spent the reed thug's only action, so this call closes its turn
    # itself and hands play back to mara, whose turn is already open.
    assert "mara" in defended["next_step"]
    assert "already open" in defended["next_step"]


def test_only_the_active_actor_may_begin_a_turn(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    result = service.combat_begin_turn(npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "not_active_actor"


def test_combat_begin_turn_never_refills_an_open_turns_actions(service: GameService, roller):
    """Test combat begin turn never refills an open turns actions.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(5, 2)
    service.combat_attack("mara", npc["npc_id"])
    reopened = service.combat_begin_turn("mara")
    assert reopened["ok"], reopened
    assert reopened["outcome"] == "turn_already_open"
    assert reopened["actions_used"] == 1
    assert service.store.read_state().combat.actors["mara"].actions_used == 1


def test_combat_defend_spends_the_npcs_one_action_when_its_turn_is_open(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    roller.queue(5)
    defended = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert defended["ok"], defended
    actor = service.store.read_state().combat.actors[npc["npc_id"]]
    assert actor.actions_used == 1
    assert actor.actions_taken == ["attack"]


def test_combat_defend_refuses_a_second_npc_attack_within_one_open_turn(
    service: GameService, roller
):
    """Test combat defend refuses a second npc attack within one open turn.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    roller.queue(5)
    first = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert first["ok"], first
    second = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert second["ok"] is False
    assert second["error"] == "npc_no_actions_remaining"


def test_a_reach_refusal_states_that_nothing_was_rolled_or_spent(
    service: GameService, roller
):
    """Test a reach refusal states that nothing was rolled or spent.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="nearby")
    service.combat_begin_turn("mara")
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"] is False
    assert result["error"] == "target_out_of_reach"
    assert "No dice were rolled and no action was spent" in result["message"]
    assert "2 of 2 action(s)" in result["message"]
    assert service.store.read_state().combat.actors["mara"].actions_used == 0


def test_a_melee_attacker_at_nearby_closes_to_close_as_part_of_its_strike(
    service: GameService, roller
):
    """A live fight ran two full melee 'paddle strikes' from NEARBY range while the
    player's own melee was refused for reach at the same distance. An enemy gets a
    move and an action each turn, so its melee strike from nearby closes the band
    -- recorded, so the player's own counter-attack now reaches too."""
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="nearby", initiative=18)  # enemy first
    roller.queue(18)
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"], result
    assert any("closes from nearby to close range" in w for w in result["warnings"])
    state = service.store.read_state()
    assert state.combat.ranges[npc["npc_id"]] == "close"
    # The turn advanced to mara (auto-opened), and melee now reaches.
    roller.queue(5, 2)
    counter = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert counter["ok"], counter


def test_a_melee_attacker_beyond_nearby_cannot_strike_at_all(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="far_away", initiative=18)  # enemy first
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["ok"] is False
    assert result["error"] == "attacker_out_of_reach"
    assert "No dice were rolled" in result["message"]
    # Nothing was consumed by the refusal: the enemy's action budget is intact.
    assert service.store.read_state().combat.actors[npc["npc_id"]].actions_used == 0
    # A missile from the same distance still resolves, as a dodge.
    roller.queue(18)
    ranged = service.combat_defend(
        "mara", npc["npc_id"], method="dodge", ranged=True
    )
    assert ranged["ok"], ranged


def test_an_attackerless_defence_on_the_players_own_turn_names_the_real_problem(
    service: GameService, roller
):
    """Test an attackerless defence on the players own turn names the real problem.
    """
    make_fighter(service, roller)
    open_fight(service, roller)  # mara wins initiative and holds the open turn
    result = service.combat_defend("mara", method="dodge")
    assert result["ok"] is False
    assert result["error"] == "incoming_damage_required"
    assert "No dice were rolled" in result["message"]
    steps = " ".join(result["allowed_next_steps"])
    assert "'mara'" in steps and "no enemy attack is pending" in steps


def test_a_defence_with_no_named_attacker_defaults_to_the_active_enemy(
    service: GameService, roller
):
    """Test a defence with no named attacker defaults to the active enemy.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller, initiative=18)  # enemy holds the open turn
    roller.queue(5)
    result = service.combat_defend("mara", method="dodge")
    assert result["ok"], result
    assert result["attacker_id"] == npc["npc_id"]
    assert result["turn_advanced"] is True
    assert result["next_actor"] == "mara"
    state = service.store.read_state()
    assert state.combat.actors[npc["npc_id"]].actions_used == 1
    assert state.combat.active_actor == "mara"


def test_a_defence_outside_an_enemy_turn_still_requires_a_threat(
    service: GameService, roller
):
    """On the player's own turn there is no active enemy to default to, so an
    attacker-less defence still needs the incoming damage named."""
    make_fighter(service, roller)
    open_fight(service, roller)  # mara holds the open turn
    result = service.combat_defend("mara", method="dodge")
    assert result["ok"] is False
    assert result["error"] == "incoming_damage_required"


def test_combat_defend_advances_the_active_actor_when_it_spends_the_last_action(
    service: GameService, roller
):
    """Test combat defend advances the active actor when it spends the last action.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    roller.queue(5)
    defended = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert defended["ok"], defended

    state = service.store.read_state()
    assert state.combat.active_actor == "mara"
    assert state.combat.actors[npc["npc_id"]].turn_open is False

    opened = service.combat_begin_turn("mara")
    assert opened["ok"], opened


def test_combat_defend_ending_the_fight_needs_no_further_combat_end_turn(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    service.combat_end_turn("mara")
    service.combat_begin_turn(npc["npc_id"])
    roller.queue(19)
    defended = service.combat_defend(
        "mara", npc["npc_id"], method="dodge", incoming_damage=99
    )
    assert defended["ok"], defended
    assert defended["defender"]["status"] == "helpless"
    assert "Combat is over" in defended["next_step"]

    state = service.store.read_state()
    assert state.combat.active is False
    assert state.combat.active_actor is None


def test_combat_start_opens_the_first_turn_itself(service: GameService, roller):
    """Winner or loser, somebody's turn is open the moment the fight starts."""
    make_fighter(service, roller)
    open_fight(service, roller)  # a passed WIS test: mara first
    state = service.store.read_state()
    assert state.combat.active_actor == "mara"
    assert state.combat.actors["mara"].turn_open is True
    assert state.combat.actors["mara"].actions_max == 2


def test_combat_start_with_a_lost_initiative_opens_the_enemys_turn(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller, initiative=18)  # a failed WIS test: enemy first
    state = service.store.read_state()
    assert state.combat.active_actor == npc["npc_id"]
    assert state.combat.actors[npc["npc_id"]].turn_open is True


def test_the_transcript_wedge_round_trip_no_bookkeeping_calls(
    service: GameService, roller
):
    """Test the transcript wedge round trip no bookkeeping calls.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller, initiative=18)  # enemy acts first

    # Round 1, enemy turn (opened by combat_start): the player defends, which
    # spends the enemy's one action and hands the open turn to the player.
    roller.queue(18)
    defended = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert defended["ok"], defended
    assert defended["next_actor"] == "mara"
    state = service.store.read_state()
    assert state.combat.actors["mara"].turn_open is True

    # Round 1, player turn: two attacks spend both actions; the round advances
    # itself and the enemy's round-2 turn opens itself.
    roller.queue(5, 1)
    first = service.combat_attack("mara", npc["npc_id"])
    assert first["ok"], first
    roller.queue(6, 15, 1)  # repeat-action Doom d6, then a miss
    second = service.combat_attack("mara", npc["npc_id"])
    assert second["ok"], second
    assert second["turn_advanced"] is True
    assert second["next_actor"] == npc["npc_id"]
    assert second["round"] == 2


    roller.queue(5)
    again = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert again["ok"], again
    assert again["next_actor"] == "mara"
    assert service.store.read_state().combat.round == 2


def test_an_attribute_test_on_the_actors_own_turn_spends_a_combat_action(
    service: GameService, roller
):
    """Test an attribute test on the actors own turn spends a combat action.
    """
    make_fighter(service, roller)
    open_fight(service, roller)
    roller.queue(18)
    result = service.attribute_test(
        "mara", "STR", "trip the thug",
        "the thug falls into the mud",
        "mara stumbles and loses her footing",
    )
    assert result["ok"], result
    assert result["combat_action"]["actions_used"] == 1
    assert result["combat_action"]["turn_advanced"] is False
    assert service.store.read_state().combat.actors["mara"].actions_used == 1
    assert "test" in service.store.read_state().combat.actors["mara"].actions_taken


def test_an_attribute_test_spending_the_last_action_advances_the_turn(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(5, 1)
    service.combat_attack("mara", npc["npc_id"])
    roller.queue(18)
    result = service.attribute_test(
        "mara", "STR", "trip the thug",
        "the thug falls into the mud",
        "mara stumbles and loses her footing",
    )
    assert result["ok"], result
    assert result["combat_action"]["turn_advanced"] is True
    assert result["combat_action"]["next_actor"] == npc["npc_id"]
    assert npc["npc_id"] in result["next_step"]
    assert service.store.read_state().combat.active_actor == npc["npc_id"]


def test_an_attribute_test_outside_the_actors_turn_spends_nothing(
    service: GameService, roller
):
    """A reactive save on the enemy's turn is not the character's combat action."""
    make_fighter(service, roller)
    open_fight(service, roller, initiative=18)  # enemy holds the open turn
    roller.queue(5)
    result = service.attribute_test(
        "mara", "WIS", "keep footing on the slick planks",
        "mara keeps her feet", "mara slips",
    )
    assert result["ok"], result
    assert "combat_action" not in result
    assert service.store.read_state().combat.actors["mara"].actions_used == 0


def test_a_test_envelope_names_the_rolling_character(service: GameService, roller):
    """Test a test envelope names the rolling character.
    """
    make_fighter(service, roller)
    result = service.attribute_test(
        "mara", "STR", "force the door", "the door gives", "the bar holds",
    )
    assert result["character_id"] == "mara"


def test_attacking_out_of_turn_is_refused_and_names_the_active_actor(
    service: GameService, roller
):
    """Test attacking out of turn is refused and names the active actor.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller, initiative=18)  # a failed WIS test: enemy first
    result = service.combat_attack("mara", npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "turn_not_open"
    assert npc["npc_id"] in result["message"]
    assert "combat_defend" in " ".join(result["allowed_next_steps"])


# -- combat close --------------------------------------------------------------


def test_combat_close_ends_a_fight_and_clears_the_active_actor(service: GameService, roller):
    make_fighter(service, roller)
    open_fight(service, roller)
    result = service.combat_close("fled")
    assert result["ok"], result
    assert result["outcome"] == "combat_closed"
    assert result["combat_active"] is False
    assert result["reason"] == "fled"
    state = service.store.read_state()
    assert state.combat.active is False
    assert state.combat.active_actor is None


def test_combat_close_refuses_without_an_active_fight(service: GameService, roller):
    make_fighter(service, roller)
    result = service.combat_close("fled")
    assert result["ok"] is False
    assert result["error"] == "combat_not_active"


def test_combat_close_requires_a_reason(service: GameService, roller):
    make_fighter(service, roller)
    open_fight(service, roller)
    result = service.combat_close("")
    assert result["ok"] is False
    assert result["error"] == "empty_close_reason"
    # A rejected close leaves the fight exactly as it was.
    assert service.store.read_state().combat.active is True


def test_combat_close_records_the_reason_in_the_audit_log(service: GameService, roller):
    make_fighter(service, roller)
    open_fight(service, roller)
    result = service.combat_close("surrendered")
    assert result["ok"], result
    event = [
        event for event in service.store.read_events(limit=10) if event["tool"] == "combat_close"
    ][-1]
    assert event["reason"] == "surrendered"
    assert event["outcome"] == "combat_closed"


def test_combat_close_after_a_close_call_leaves_the_actor_free_to_act_elsewhere(
    service: GameService, roller
):
    """combat_close closes the fight before either side is eliminated -- disengagement,
    not annihilation -- so the standing player character keeps acting outside combat."""
    make_fighter(service, roller)
    open_fight(service, roller)
    closed = service.combat_close("fled")
    assert closed["ok"], closed
    sheet = service.character_sheet("mara")
    assert sheet["ok"], sheet
    assert sheet["sheet"]["status"] == "ok"


# -- combat movement ---------------------------------------------------------


def test_a_default_fight_can_close_range_and_land_a_melee_hit(service: GameService, roller):
    """Test a default fight can close range and land a melee hit.
    """
    make_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    roller.queue(5)  # Mara wins initiative
    started = service.combat_start(
        pc_ids=["mara"], npc_ids=[npc["npc_id"]], reason="ambush"
    )
    assert started["ok"], started
    service.combat_begin_turn("mara")

    # A default combat_start places the opponent at nearby, so melee is out of reach.
    blocked = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert blocked["ok"] is False
    assert blocked["error"] == "target_out_of_reach"
    assert "combat_move" in " ".join(blocked["allowed_next_steps"])

    moved = service.combat_move("mara", npc["npc_id"])
    assert moved["ok"], moved
    assert moved["from_band"] == "nearby"
    assert moved["to_band"] == "close"
    assert moved["actions_remaining"] == 1

    roller.queue(3, 6, 6)  # a hit and its damage
    landed = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert landed["ok"], landed
    assert landed["roll"]  # a die actually rolled, unlike every refused attack
    # The blow killed the only enemy, so the fight closed itself.
    assert landed["combat_over"] is True
    third = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert third["ok"] is False
    assert third["error"] == "combat_not_active"


def test_combat_move_spends_one_action_and_records_the_band_change(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="far_away")
    service.combat_begin_turn("mara")
    before = service.store.read_state().event_seq

    moved = service.combat_move("mara", npc["npc_id"])
    assert moved["ok"], moved
    assert (moved["from_band"], moved["to_band"]) == ("far_away", "nearby")
    assert service.store.read_state().event_seq == before + 1
    actor = service.store.read_state().combat.actors["mara"]
    assert actor.actions_used == 1
    assert "move" in actor.actions_taken


def test_combat_move_refuses_when_already_at_close_range(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="close")
    service.combat_begin_turn("mara")
    result = service.combat_move("mara", npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "already_closest"


def test_combat_move_refuses_without_an_open_turn(service: GameService, roller):
    # A failed initiative puts the opposition first, so mara's turn is not open.
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="nearby", initiative=18)
    result = service.combat_move("mara", npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "turn_not_open"
    assert npc["npc_id"] in result["message"]


def test_combat_move_refuses_when_no_actions_remain(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller, band="distant")
    service.combat_begin_turn("mara")
    service.combat_move("mara", npc["npc_id"])  # distant -> far_away
    second = service.combat_move("mara", npc["npc_id"])  # far_away -> nearby
    # The second move spent the last action, so the turn advanced itself and a
    # third move is refused as out of turn.
    assert second["turn_advanced"] is True
    result = service.combat_move("mara", npc["npc_id"])
    assert result["ok"] is False
    assert result["error"] == "turn_not_open"


def test_combat_attack_never_returns_a_dict_where_a_roll_target_is_read(
    service: GameService, roller
):
    """Every other rolling tool in this service sends the roll's own target number under
    ``target``: ``attribute_test`` through ``_test_envelope``, and ``group_test`` in
    each per-participant entry, both integers. ``combat_attack`` alone sent an NPC
    descriptor object there, while already carrying ``target_id`` two fields away, so a
    narrator reading ``target`` for the number to print in
    ``<Name> rolls <ATTRIBUTE>: <total> vs <target>, <outcome>`` found a dict and had
    nothing numeric to announce.

    The invariant is stated over the whole payload rather than over the one renamed
    field, so a future tool that reintroduces the same overload fails here too.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 4)
    result = service.combat_attack("mara", npc["npc_id"], attack_type="melee")
    assert result["ok"], result

    # The descriptor is still available, under a name no one reads as a number.
    assert isinstance(result["target_npc"], dict)
    assert result["target_npc"]["id"] == npc["npc_id"]
    # The identifier reference is untouched.
    assert result["target_id"] == npc["npc_id"]
    # And the key a roll's target is conventionally read from is never a dict.
    assert not isinstance(result.get("target"), dict), result.get("target")
    # The roll's real target number is where every roll keeps it, and it is an integer.
    assert isinstance(result["roll"]["target"], int)

    # The same invariant across the two tools that do send a numeric ``target``, so
    # this test states one rule for the surface rather than one exception for one tool.
    roller.queue(8)
    single = service.attribute_test(
        "mara", "INT", "reads the ledger", "the entry is clear", "the ink has run"
    )
    assert isinstance(single["target"], int)
    assert not isinstance(single.get("target"), dict)
