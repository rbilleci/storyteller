"""Possessions, recovery, and helpless-condition tool tests."""

from __future__ import annotations

from conftest import make_character
from tools_shared import drop_to_helpless, make_fighter, open_fight

from bsh_mcp.dice import DEPLETED
from bsh_mcp.service import GameService

# -- possessions ----------------------------------------------------------------


def test_inventory_update_moves_coins_and_items_atomically(service: GameService, roller):
    """The live gap: a narrator told a player they looted 8 coins and a charm, and
    the character file still held the old coins and an empty equipment list, because
    no tool could move either. Possessions now change like hit points do: through
    one audited call."""
    make_fighter(service, roller)
    before = service.store.read_character("mara").coins
    result = service.inventory_update(
        "mara", "looted the reed thug's pouch",
        coins_delta=8,
        add_equipment=["carved bone charm"],
        add_weapons=["gutting knife"],
    )
    assert result["ok"], result
    assert result["coins"] == before + 8
    character = service.store.read_character("mara")
    assert character.coins == before + 8
    assert "carved bone charm" in character.equipment
    assert "gutting knife" in character.weapons
    event = [e for e in service.store.read_events(limit=5) if e["tool"] == "inventory_update"][-1]
    assert event["outcome"] == "inventory_updated"
    assert event["coins_delta"] == 8
    assert sorted(event["added"]) == ["carved bone charm", "gutting knife"]


def test_inventory_update_refuses_an_overdraw_before_any_write(service: GameService, roller):
    make_fighter(service, roller)
    before = service.store.read_character("mara").coins
    result = service.inventory_update(
        "mara", "bribes the constable", coins_delta=-(before + 1)
    )
    assert result["ok"] is False
    assert result["error"] == "insufficient_coins"
    assert service.store.read_character("mara").coins == before


def test_inventory_update_refuses_removing_an_item_not_held(service: GameService, roller):
    """The refusal aborts the whole change: a coin delta passed alongside the bad
    removal must not land either."""
    make_fighter(service, roller)
    before = service.store.read_character("mara").coins
    result = service.inventory_update(
        "mara", "drops the rope", coins_delta=3, remove_equipment=["rope"]
    )
    assert result["ok"] is False
    assert result["error"] == "item_not_held"
    assert service.store.read_character("mara").coins == before


def test_inventory_update_removal_matches_a_held_name_case_insensitively(
    service: GameService, roller
):
    make_fighter(service, roller)
    assert service.inventory_update("mara", "coils the rope", add_equipment=["Salt-crusted Rope"])["ok"]
    result = service.inventory_update("mara", "hands the rope over", remove_equipment=["salt-crusted rope"])
    assert result["ok"], result
    assert service.store.read_character("mara").equipment == []


def test_inventory_update_requires_a_reason_and_a_change(service: GameService, roller):
    make_fighter(service, roller)
    no_reason = service.inventory_update("mara", "", coins_delta=1)
    assert no_reason["ok"] is False and no_reason["error"] == "empty_inventory_reason"
    no_change = service.inventory_update("mara", "nothing happens")
    assert no_change["ok"] is False and no_change["error"] == "empty_inventory_update"


def test_scene_commit_deduplicates_repeated_facts(service: GameService, roller):
    """A live scene record accumulated "Rade is dead" and "Rade is dead." as two
    visible facts (the ratifying commit and the settle step's own commit each
    recorded the same truth), and every duplicate rides the canon digest into every
    later prompt. Equal facts -- case, spacing, and trailing punctuation aside --
    are recorded once; genuinely different facts still both land."""
    make_fighter(service, roller)
    service.scene_commit("the fight ends", visible_changes=["Rade is dead"])
    service.scene_commit(
        "the market reacts",
        visible_changes=["Rade is dead.", "rade is dead", "The market falls silent"],
    )
    facts = service.store.read_state().scene.visible_facts
    dead = [f for f in facts if f.rstrip(" .").casefold() == "rade is dead"]
    assert dead == ["Rade is dead"]
    assert "The market falls silent" in facts


# -- recovery ----------------------------------------------------------------


def test_short_rest_restores_half_constitution_once_per_day(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(19)
    service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=10)
    assert service.store.read_character("mara").hp == 4

    result = service.rest(["mara"], "short", reason="catch breath")
    assert result["ok"], result
    assert result["characters"][0]["healed"] == 7
    assert service.store.read_character("mara").hp == 11

    again = service.rest(["mara"], "short", reason="catch breath again")
    assert again["ok"] is False
    assert again["error"] == "short_rest_already_used"


def test_a_long_rest_needs_a_safe_environment(service: GameService, roller):
    make_fighter(service, roller)
    result = service.rest(["mara"], "long", safe_environment=False, reason="camp in the reeds")
    assert result["ok"] is False
    assert result["error"] == "unsafe_environment"


def test_a_wildling_may_take_a_long_rest_anywhere(service: GameService, roller):
    make_character(
        service,
        roller,
        name="Keth",
        origin="barbarian",
        backgrounds=("wildling", "survivor", "hunter"),
        weapons=("boar spear",),
    )
    result = service.rest(["keth"], "long", safe_environment=False, reason="sleep in the reeds")
    assert result["ok"], result


def test_a_safe_long_rest_restores_hit_points_doom_and_conditions(service: GameService, roller):
    make_fighter(service, roller)
    for _ in range(2):
        roller.queue(1)
        service.doom_roll("mara", "spend it", mode="call_on_doom")
    npc = open_fight(service, roller)
    roller.queue(19, 19)
    service.combat_defend("mara", npc["npc_id"], method="dodge", incoming_damage=10)

    character = service.store.read_character("mara")
    assert character.doom_die == DEPLETED
    assert any(condition.id == "doomed" for condition in character.conditions)

    result = service.rest(["mara"], "long", safe_environment=True, reason="sleep in the tower")
    assert result["ok"], result
    restored = service.store.read_character("mara")
    assert restored.hp == restored.hp_max
    assert restored.doom_die == "d6"
    assert restored.conditions == []


def test_rest_advances_in_game_time(service: GameService, roller):
    make_fighter(service, roller)
    service.rest(["mara"], "short", reason="breathe")
    assert service.store.read_state().in_game_minutes == 60
    service.rest(["mara"], "long", safe_environment=True, reason="sleep")
    assert service.store.read_state().in_game_minutes == 420


# -- helpless ----------------------------------------------------------------


def test_helpless_scratched_restores_hit_points(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(1, 3)
    result = service.helpless_roll("mara")
    assert result["result"]["name"] == "Scratched"
    assert result["hp"] == 3
    assert result["status"] == "ok"
    assert service.store.read_character("mara").scars


def test_helpless_injured_adds_a_session_condition(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(4, 2)
    result = service.helpless_roll("mara")
    assert result["result"]["condition"]["effect"] == "disadvantage_all"
    character = service.store.read_character("mara")
    assert any(condition.id == "injured" for condition in character.conditions)


def test_helpless_butchered_removes_an_attribute_point(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    before = service.store.read_character("mara").attributes.STR
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(5, 2, 1)
    result = service.helpless_roll("mara")
    assert result["result"]["attribute_lost"] == "STR"
    assert service.store.read_character("mara").attributes.STR == before - 1


def test_helpless_killed_marks_the_character_dead(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(6)
    result = service.helpless_roll("mara")
    assert result["status"] == "dead"
    assert service.store.read_character("mara").status == "dead"


def test_helpless_roll_requires_a_helpless_character(service: GameService, roller):
    make_fighter(service, roller)
    result = service.helpless_roll("mara")
    assert result["ok"] is False
    assert result["error"] == "character_not_helpless"


def test_surgeon_care_reduces_the_helpless_die_to_d4(service: GameService, roller):
    """Test surgeon care reduces the helpless die to d4.
    """
    make_fighter(service, roller)
    make_character(
        service, roller, name="Ulf", origin="civilised",
        backgrounds=("surgeon", "bookworm", "diplomat"),
    )
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(5, 4, 3)  # Ulf's INT test succeeds, Helpless face 4 (d4), recovered hp 3
    result = service.helpless_roll("mara", carer_id="ulf")
    assert result["ok"], result
    assert result["care"]["carer_id"] == "ulf"
    assert result["care"]["outcome"] == "success"
    assert result["roll"]["notation"] == "1d4"
    assert result["result"]["name"] == "Injured"  # face 5/6 (Butchered/Killed) unreachable on a d4


def test_surgeon_failed_int_test_keeps_the_helpless_die_at_d6(service: GameService, roller):
    make_fighter(service, roller)
    make_character(
        service, roller, name="Ulf", origin="civilised",
        backgrounds=("surgeon", "bookworm", "diplomat"),
    )
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(19, 6)  # Ulf's INT test fails, Helpless face 6 (d6): Killed
    result = service.helpless_roll("mara", carer_id="ulf")
    assert result["ok"], result
    assert result["care"]["outcome"] == "failure"
    assert result["roll"]["notation"] == "1d6"
    assert result["status"] == "dead"


def test_carer_without_a_tending_effect_is_refused_without_rolling(
    service: GameService, roller
):
    make_fighter(service, roller)
    make_character(
        service, roller, name="Ulf", origin="civilised",
        backgrounds=("bookworm", "diplomat", "sophist"),
    )
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    before = service.store.read_state().event_seq
    result = service.helpless_roll("mara", carer_id="ulf")
    assert result["ok"] is False
    assert result["error"] == "carer_cannot_help"
    assert service.store.read_state().event_seq == before
