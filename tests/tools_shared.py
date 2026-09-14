"""Character/combat builders and other cross-file test helpers shared across the
tests/test_*.py files split from what used to be tests/test_tools.py."""

from __future__ import annotations

from conftest import ScriptedRoller, make_character

from bsh_mcp.service import GameService

FIGHTER_BACKGROUNDS = ("hunter", "survivor", "raider")

ASSASSIN_BACKGROUNDS = ("assassin", "snake-blood", "pit-fighter")


def make_fighter(service: GameService, roller: ScriptedRoller, **kwargs) -> dict:
    """A barbarian with STR 14, DEX 14, CON 14, INT 13, WIS 13, CHA 13."""
    kwargs.setdefault("backgrounds", FIGHTER_BACKGROUNDS)
    return make_character(service, roller, **kwargs)


def make_assassin(service: GameService, roller: ScriptedRoller, **kwargs) -> dict:
    """A decadent with DEX 14 (assassin's +1), STR/CON/INT/WIS/CHA 13."""
    kwargs.setdefault("origin", "decadent")
    kwargs.setdefault("backgrounds", ASSASSIN_BACKGROUNDS)
    return make_character(service, roller, **kwargs)


def open_fight(
    service: GameService,
    roller: ScriptedRoller,
    *,
    initiative: int = 5,
    npc_level: int = 1,
    band: str = "close",
    pc_id: str = "mara",
) -> dict:
    npc = service.npc_create(name="Reed Thug", level=npc_level, motive="rob the party")
    assert npc["ok"], npc
    roller.queue(initiative)
    started = service.combat_start(
        pc_ids=[pc_id],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: band},
        reason="ambush on the plank walk",
    )
    assert started["ok"], started
    return npc


def _award_stories(service: GameService, character_id: str, count: int) -> None:
    """Award Stories through the real session_close flow."""
    result = service.session_close(
        session_title="Between adventures",
        public_summary="The party rested and counted their scars.",
        character_stories_awarded={character_id: count},
    )
    assert result["ok"], result


def drop_to_helpless(service: GameService, roller, npc_id: str) -> None:
    roller.queue(19)
    service.combat_defend("mara", npc_id, method="dodge", incoming_damage=99)
