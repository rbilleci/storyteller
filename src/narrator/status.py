"""Read-only status snapshot: scene, day, combat, per-character vitals and sheets.

Mirrors ``narrator.engine``'s ``_character_resource_values``/``_clock_fills``: a direct,
dependency-free read of ``campaign/`` for display, never a ``bsh_mcp`` import and never a
transaction. Unlike ``narrator.ledger`` this fails OPEN throughout -- an absent or
malformed file contributes nothing rather than raising, because a status line is
cosmetic and must never stall or block a turn the way the fiction-debt gate does.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The campaign-wide keys ``snapshot`` always carries, so a caller can index them
#: unconditionally rather than defaulting through ``dict.get`` at every use site.
_EMPTY_CAMPAIGN_FIELDS: dict = {
    "scene_title": "",
    "location_id": "",
    "day": None,
    "combat_active": False,
    "combat_round": None,
    "combat_active_actor": None,
}


def _campaign_fields(campaign_root: Path | str) -> dict:
    """Scene, in-game day, and combat fields read from ``campaign/state.json``."""
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_EMPTY_CAMPAIGN_FIELDS)
    if not isinstance(payload, dict):
        return dict(_EMPTY_CAMPAIGN_FIELDS)
    scene = payload.get("scene")
    scene = scene if isinstance(scene, dict) else {}
    combat = payload.get("combat")
    combat = combat if isinstance(combat, dict) else {}
    minutes = payload.get("in_game_minutes")
    day = None
    if isinstance(minutes, int) and not isinstance(minutes, bool):
        # Matches ``BSHService._time_string``'s display convention (src/bsh_mcp/service.py):
        # the first day of the campaign reads "Day 1", though ``CampaignState.day`` itself
        # (``in_game_minutes // 1440``) is 0-indexed.
        day = minutes // 1440 + 1
    active = bool(combat.get("active"))
    combat_round = combat.get("round")
    active_actor = combat.get("active_actor")
    return {
        "scene_title": str(scene.get("title") or ""),
        "location_id": str(scene.get("location_id") or ""),
        "day": day,
        "combat_active": active,
        "combat_round": (
            combat_round
            if active and isinstance(combat_round, int) and not isinstance(combat_round, bool)
            else None
        ),
        "combat_active_actor": (
            str(active_actor) if active and isinstance(active_actor, str) else None
        ),
    }


def _sheet_fields(payload: dict) -> dict:
    """The character-sheet fields the terminal's ``/character`` command renders.

    Everything here is durable mechanical state a player's own printed Black Sword
    Hack sheet holds (origin, backgrounds, attributes, arms, belongings, advancement)
    plus this engine's own sheet-worthy extensions (usage-die resources, doses,
    powers, a runic weapon). Normalized fail-open like every other field in this
    module: a missing or mis-typed value contributes its empty shape rather than
    raising, so one corrupt field never costs the rest of the sheet.
    """

    def text(key: str) -> str:
        value = payload.get(key)
        return value if isinstance(value, str) else ""

    def integer(key: str):
        value = payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def strings(key: str) -> list[str]:
        value = payload.get(key)
        return [
            entry for entry in (value if isinstance(value, list) else [])
            if isinstance(entry, str) and entry
        ]

    attributes = payload.get("attributes")
    attributes = {
        name: value
        for name, value in (attributes if isinstance(attributes, dict) else {}).items()
        if isinstance(name, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
    }
    resources = [
        {"name": str(entry.get("name")), "die": _text_or_empty(entry.get("die"))}
        for entry in payload.get("resources") or []
        if isinstance(entry, dict) and entry.get("name")
    ]
    doses = payload.get("doses")
    doses = {
        str(kind): count
        for kind, count in (doses if isinstance(doses, dict) else {}).items()
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }
    runic = payload.get("runic_weapon")
    return {
        "origin": text("origin"),
        "level": integer("level"),
        "stories": integer("stories"),
        "backgrounds": strings("backgrounds"),
        "attributes": attributes,
        "doom_max": text("doom_max"),
        "weapon_damage": text("weapon_damage"),
        "unarmed_damage": text("unarmed_damage"),
        "armour": text("armour"),
        "shield": bool(payload.get("shield")),
        "weapons": strings("weapons"),
        "equipment": strings("equipment"),
        "languages": strings("languages"),
        "coins": integer("coins"),
        "resources": resources,
        "scars": strings("scars"),
        "gifts": strings("gifts"),
        "spells": strings("spells"),
        "powers": strings("powers"),
        "doses": doses,
        "runic_weapon": (
            str(runic.get("name") or "") if isinstance(runic, dict) else ""
        ),
        "notes": text("notes"),
    }


def _text_or_empty(value) -> str:
    """One value as display text, or "" for the non-string shapes fail-open drops."""
    return value if isinstance(value, str) else ""


def _character_fields(campaign_root: Path | str) -> dict[str, dict]:
    """Every character's vitals plus their ``sheet`` fields, keyed by character id.

    Reads the same ``campaign/characters/*.json`` files
    ``narrator.engine._character_resource_values`` reads, independently and for a
    different purpose: that helper feeds the sweep guard, this feeds a status line
    and the terminal's ``/character`` sheet. The vitals stay top-level (they are the
    status line's contract); the sheet-only fields nest under ``"sheet"`` so the two
    consumers stay visibly separate.
    """
    characters_dir = Path(campaign_root) / "campaign" / "characters"
    characters: dict[str, dict] = {}
    if not characters_dir.is_dir():
        return characters
    for path in sorted(characters_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        character_id = str(payload.get("id") or path.stem)
        hp = payload.get("hp")
        hp_max = payload.get("hp_max")
        doom_die = payload.get("doom_die")
        conditions = [
            str(entry.get("label"))
            for entry in payload.get("conditions") or []
            if isinstance(entry, dict) and entry.get("label")
        ]
        characters[character_id] = {
            "name": str(payload.get("name") or character_id),
            "status": str(payload.get("status") or "ok"),
            "hp": hp if isinstance(hp, int) and not isinstance(hp, bool) else None,
            "hp_max": (
                hp_max if isinstance(hp_max, int) and not isinstance(hp_max, bool) else None
            ),
            "doom_die": str(doom_die) if isinstance(doom_die, str) else None,
            "conditions": conditions,
            "sheet": _sheet_fields(payload),
        }
    return characters


def _player_links(campaign_root: Path | str) -> dict[str, dict]:
    """Account-to-character links from ``campaign/players.yaml``, keyed by account id.

    This is what lets a channel adapter bind its display to *its own* player's
    character without hardcoding one: a campaign that starts at session zero has no
    characters at all, and the link only appears once ``character_create`` writes it.
    Unlike ``narrator.player_directory`` this fails open — a status line is cosmetic
    — and it carries no authorization weight: decisions still authorize through the
    directory's own fail-closed read.
    """
    players_path = Path(campaign_root) / "campaign" / "players.yaml"
    links: dict[str, dict] = {}
    try:
        import yaml

        payload = yaml.safe_load(players_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - display data fails open
        return links
    entries = payload.get("players") if isinstance(payload, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        account_id = entry.get("discord_user_id")
        character_id = entry.get("character_id")
        if not isinstance(account_id, str) or not isinstance(character_id, str):
            continue
        display_name = entry.get("display_name")
        links[account_id] = {
            "character_id": character_id,
            "display_name": display_name if isinstance(display_name, str) else "",
        }
    return links


def snapshot(campaign_root: Path | str) -> dict:
    """One status-line snapshot: campaign-wide fields plus every character's vitals.

    Fails open at every field: an unreadable or malformed source contributes nothing
    rather than raising.
    """
    fields = _campaign_fields(campaign_root)
    fields["characters"] = _character_fields(campaign_root)
    fields["players"] = _player_links(campaign_root)
    return fields
