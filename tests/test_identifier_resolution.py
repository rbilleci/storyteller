"""Identifier-resolution tool tests."""

from __future__ import annotations

from conftest import make_character
from tools_shared import make_fighter, open_fight

from bsh_mcp.service import GameService

# -- identifier resolution -----------------------------------------------------


def test_display_name_resolves_to_the_character_id(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    result = service.attribute_test("Mara", "DEX", "slip past the clerk unseen", "the way opens", "the cost lands")
    assert result["ok"], result
    assert result["outcome"] == "success"
    assert any("resolved character id 'Mara' to 'mara'" in w for w in result["warnings"])


def test_case_folded_id_resolves_on_a_read(service: GameService, roller):
    make_fighter(service, roller)
    result = service.character_sheet("MARA")
    assert result["ok"], result
    assert result["sheet"]["id"] == "mara"
    assert any("'mara'" in w for w in result["warnings"])


def test_unknown_character_error_lists_valid_ids(service: GameService, roller):
    make_fighter(service, roller)
    result = service.attribute_test("Bogus", "DEX", "no such person", "the way opens", "the cost lands")
    assert result["ok"] is False
    assert result["error"] == "character_not_found"
    assert any("mara" in step for step in result["allowed_next_steps"])


def test_ambiguous_display_name_is_refused(service: GameService, roller):
    make_character(service, roller, name="The Kid")
    make_character(service, roller, name="The Kid")
    result = service.attribute_test("The Kid", "DEX", "which kid?", "the way opens", "the cost lands")
    assert result["ok"] is False
    assert result["error"] == "character_ambiguous"
    assert any("the-kid" in step for step in result["allowed_next_steps"])


def test_group_advantage_list_accepts_display_names(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(15, 2)
    result = service.group_test(["Mara"], "DEX", "cross the mud", "the group slips through", "the group is marked", advantage_character_ids=["MARA"])
    assert result["ok"], result
    assert roller.script == []  # two d20s consumed proves Advantage matched
    assert result["individual"][0]["character_id"] == "mara"
    assert result["individual"][0]["succeeded"] is True


def test_combat_calls_resolve_display_names(service: GameService, roller):
    make_fighter(service, roller)
    open_fight(service, roller)
    begin = service.combat_begin_turn("MARA")
    assert begin["ok"], begin
    assert begin["actor_id"] == "mara"
    roller.queue(5, 4)
    attack = service.combat_attack("Mara", "Reed Thug")
    assert attack["ok"], attack
    assert attack["target_id"] == "reed-thug"
    assert any("resolved NPC id 'Reed Thug' to 'reed-thug'" in w for w in attack["warnings"])


def test_rest_with_a_display_name_writes_the_canonical_sheet(service: GameService, roller):
    make_fighter(service, roller)
    result = service.rest(["Mara"], "short")
    assert result["ok"], result
    assert result["characters"][0]["character_id"] == "mara"
    assert service.store.character_ids() == ["mara"]  # no stray 'Mara' sheet file


def test_scene_commit_resolves_present_npc_names(service: GameService, roller):
    service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    result = service.scene_commit(
        public_summary="The thug lingers by the fish stalls.",
        present_npcs=["Reed Thug"],
    )
    assert result["ok"], result
    assert service.store.read_state().scene.present_npcs == ["reed-thug"]


def test_helpless_roll_with_a_display_name_commits_canonically(service: GameService, roller):
    make_fighter(service, roller)
    # Reduce Mara to 0 hit points through a failed defence against heavy damage.
    roller.queue(19)  # dodge roll 19 vs DEX 14: failure
    result = service.combat_defend("mara", method="dodge", incoming_damage=20)
    assert result["ok"], result
    sheet = service.store.read_character("mara")
    assert sheet.hp == 0 and sheet.status == "helpless"
    roller.queue(3, 4)  # Helpless face 3, then the survivor hit-point die
    outcome = service.helpless_roll("MARA", "the fight is over")
    assert outcome["ok"], outcome
    assert service.store.read_character("mara").status in ("ok", "dead")
    assert service.store.character_ids() == ["mara"]


def test_session_close_awards_stories_by_display_name(service: GameService, roller):
    make_fighter(service, roller)
    result = service.session_close(
        session_title="The Silent Bell",
        public_summary="The party learned why the bell stopped.",
        character_stories_awarded={"Mara": 2},
    )
    assert result["ok"], result
    assert service.store.read_character("mara").stories == 2
    assert service.store.character_ids() == ["mara"]
