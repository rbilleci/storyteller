"""``narrator.status`` reads campaign state for a status line and never raises."""

from __future__ import annotations

import json
from pathlib import Path

from narrator import status


def _write_state(campaign_root: Path, payload: dict) -> None:
    (campaign_root / "campaign").mkdir(parents=True, exist_ok=True)
    (campaign_root / "campaign" / "state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_character(campaign_root: Path, character_id: str, payload: dict) -> None:
    characters = campaign_root / "campaign" / "characters"
    characters.mkdir(parents=True, exist_ok=True)
    (characters / f"{character_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_a_campaign_with_no_state_file_yields_an_empty_but_valid_snapshot(tmp_path: Path):
    result = status.snapshot(tmp_path)

    assert result["scene_title"] == ""
    assert result["location_id"] == ""
    assert result["day"] is None
    assert result["combat_active"] is False
    assert result["combat_round"] is None
    assert result["combat_active_actor"] is None
    assert result["characters"] == {}


def test_scene_day_and_inactive_combat_read_from_state(tmp_path: Path):
    _write_state(
        tmp_path,
        {
            "scene": {"title": "The Ashen Bell", "location_id": "ashenport"},
            "in_game_minutes": 1500,
            "combat": {"active": False, "round": 3, "active_actor": "rill"},
        },
    )

    result = status.snapshot(tmp_path)

    assert result["scene_title"] == "The Ashen Bell"
    assert result["location_id"] == "ashenport"
    # 1500 minutes is one day and 60 minutes in, so this is the second day, and the
    # display convention (matching ``BSHService._time_string``) is 1-indexed.
    assert result["day"] == 2
    # Combat fields are dropped once ``active`` is false, even though the state file
    # still carries a stale round/actor from the fight that just ended.
    assert result["combat_active"] is False
    assert result["combat_round"] is None
    assert result["combat_active_actor"] is None


def test_active_combat_reports_its_round_and_active_actor(tmp_path: Path):
    _write_state(
        tmp_path,
        {"combat": {"active": True, "round": 2, "active_actor": "rill"}},
    )

    result = status.snapshot(tmp_path)

    assert result["combat_active"] is True
    assert result["combat_round"] == 2
    assert result["combat_active_actor"] == "rill"


def test_character_vitals_are_keyed_by_id_and_read_from_their_own_file(tmp_path: Path):
    _write_character(
        tmp_path,
        "rill",
        {
            "id": "rill",
            "name": "Rill",
            "status": "helpless",
            "hp": 2,
            "hp_max": 10,
            "doom_die": "d4",
            "conditions": [{"label": "bleeding"}, {"label": ""}, "not-a-dict"],
        },
    )

    result = status.snapshot(tmp_path)

    entry = result["characters"]["rill"]
    # The sheet fields nest under their own key (asserted in the sheet tests below)
    # so the status line's flat vitals contract stays exactly what it always was.
    sheet = entry.pop("sheet")
    assert isinstance(sheet, dict)
    assert result["characters"] == {
        "rill": {
            "name": "Rill",
            "status": "helpless",
            "hp": 2,
            "hp_max": 10,
            "doom_die": "d4",
            "conditions": ["bleeding"],
        }
    }


def test_the_sheet_fields_ride_each_character_entry_for_the_character_command(
    tmp_path: Path,
):
    """``/character`` renders from the snapshot, so the sheet's own fields -- the ones
    the printed Black Sword Hack sheet holds beyond the status line's vitals -- must
    ride each character entry, normalized."""
    _write_character(
        tmp_path,
        "kara",
        {
            "id": "kara",
            "name": "Kara",
            "origin": "barbarian",
            "backgrounds": ["berserker", "hunter"],
            "level": 3,
            "stories": 1,
            "attributes": {"STR": 15, "DEX": 12, "CON": 11, "INT": 9, "WIS": 10, "CHA": 8},
            "hp": 11,
            "hp_max": 12,
            "doom_die": "d4",
            "doom_max": "d6",
            "weapon_damage": "d8",
            "unarmed_damage": "d4",
            "armour": "light",
            "shield": True,
            "weapons": ["claymore (two-handed)"],
            "equipment": ["rope", "flint"],
            "languages": ["Thyrenian", "Estuary Cant"],
            "coins": 25,
            "resources": [{"id": "rations", "name": "Rations", "die": "d6"}],
            "scars": ["split brow"],
            "gifts": ["second-wind"],
            "spells": ["The Crimson Mist"],
            "powers": ["demon:abyss"],
            "doses": {"healing_balm": 2},
            "runic_weapon": {"name": "Mourner", "personality": "patient", "weapon_int": 12},
            "notes": "Berserker: rage adds a d6 to damage dealt.",
        },
    )

    sheet = status.snapshot(tmp_path)["characters"]["kara"]["sheet"]

    assert sheet == {
        "origin": "barbarian",
        "level": 3,
        "stories": 1,
        "backgrounds": ["berserker", "hunter"],
        "attributes": {"STR": 15, "DEX": 12, "CON": 11, "INT": 9, "WIS": 10, "CHA": 8},
        "doom_max": "d6",
        "weapon_damage": "d8",
        "unarmed_damage": "d4",
        "armour": "light",
        "shield": True,
        "weapons": ["claymore (two-handed)"],
        "equipment": ["rope", "flint"],
        "languages": ["Thyrenian", "Estuary Cant"],
        "coins": 25,
        "resources": [{"name": "Rations", "die": "d6"}],
        "scars": ["split brow"],
        "gifts": ["second-wind"],
        "spells": ["The Crimson Mist"],
        "powers": ["demon:abyss"],
        "doses": {"healing_balm": 2},
        "runic_weapon": "Mourner",
        "notes": "Berserker: rage adds a d6 to damage dealt.",
    }


def test_malformed_sheet_values_contribute_their_empty_shapes_not_errors(tmp_path: Path):
    """One corrupt field never costs the rest of the sheet: every mis-typed value
    normalizes to its empty shape, the same fail-open rule the vitals follow."""
    _write_character(
        tmp_path,
        "rill",
        {
            "id": "rill",
            "name": "Rill",
            "hp": 5,
            "hp_max": 10,
            "origin": 7,
            "level": "three",
            "stories": True,
            "backgrounds": ["hunter", 4, ""],
            "attributes": {"STR": 15, "DEX": "high", "CON": True},
            "shield": "yes",
            "weapons": "a knife",
            "coins": 2.5,
            "resources": [{"die": "d6"}, "rations", {"name": "Rations", "die": 6}],
            "doses": {"healing_balm": 0, "antidote": "one", "salve": 2},
            "runic_weapon": "Mourner",
            "notes": ["not", "text"],
        },
    )

    sheet = status.snapshot(tmp_path)["characters"]["rill"]["sheet"]

    assert sheet["origin"] == ""
    assert sheet["level"] is None
    assert sheet["stories"] is None
    assert sheet["backgrounds"] == ["hunter"]
    assert sheet["attributes"] == {"STR": 15}
    # A truthy non-bool still reads as "carries a shield" -- display data fails open.
    assert sheet["shield"] is True
    assert sheet["weapons"] == []
    assert sheet["coins"] is None
    assert sheet["resources"] == [{"name": "Rations", "die": ""}]
    assert sheet["doses"] == {"salve": 2}
    assert sheet["runic_weapon"] == ""
    assert sheet["notes"] == ""


def test_a_malformed_character_file_is_dropped_not_raised(tmp_path: Path):
    _write_character(tmp_path, "rill", {"id": "rill", "name": "Rill", "hp": 5, "hp_max": 10})
    characters = tmp_path / "campaign" / "characters"
    (characters / "ossa.json").write_text("not json", encoding="utf-8")

    result = status.snapshot(tmp_path)

    assert set(result["characters"]) == {"rill"}


def test_an_unreadable_or_non_object_state_file_yields_the_empty_campaign_fields(
    tmp_path: Path,
):
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "state.json").write_text("[1, 2, 3]", encoding="utf-8")

    result = status.snapshot(tmp_path)

    assert result["scene_title"] == ""
    assert result["day"] is None


def test_player_links_ride_the_snapshot_and_fail_open(tmp_path: Path):
    """The links let a channel adapter bind its display to its own player's
    character without hardcoding one — the session-zero handshake. Display data
    only: an absent or broken players.yaml yields an empty map, never an error,
    and malformed entries are dropped individually."""
    absent = status.snapshot(tmp_path)
    assert absent["players"] == {}

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "players.yaml").write_text(
        "players:\n"
        "- discord_user_id: terminal-player\n"
        "  character_id: kara\n"
        "  display_name: Kara\n"
        "- discord_user_id: '12345'\n"
        "  character_id: ossa\n"
        "- not-a-mapping\n"
        # An unquoted numeric id parses as an int; the fail-closed player directory
        # rejects that shape, and this display reader drops it the same way.
        "- discord_user_id: 99\n"
        "  character_id: broken\n",
        encoding="utf-8",
    )
    result = status.snapshot(tmp_path)
    assert result["players"] == {
        "terminal-player": {"character_id": "kara", "display_name": "Kara"},
        "12345": {"character_id": "ossa", "display_name": ""},
    }

    (campaign / "players.yaml").write_text("players: {broken\n", encoding="utf-8")
    assert status.snapshot(tmp_path)["players"] == {}
