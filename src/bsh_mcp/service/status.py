"""Read-only campaign, character, and NPC status, and the MCP resource bodies."""

from __future__ import annotations

import json

from .. import results, rules
from ..dice import DEPLETED
from ..models import ARMOUR_CATEGORIES, NPC, ORIGINS, Character
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from .common import guard

# -- tool result envelopes ------------------------------------------------
#
# One TypedDict per MCP-tool method on GameService (the 26 methods server.py
# registers with @mcp.tool()), each extending ToolEnvelopeSuccess with the exact
# extra keys that method's own results.success() call sites add. A field is
# NotRequired only where at least one success branch omits it; a field every
# branch sets is required even when its value can be None (e.g. next_actor).
# Every one of these methods is also decorated with @guard, so its actual return
# annotation unions in ToolEnvelopeFailure too: guard turns CampaignError,
# RulesDataError, ValidationError, ValueError, and KeyError into that shape.
# `dict`/`list[dict]` fields below are payloads built by other typed models'
# own `.model_dump()`/`.as_dict()` (Roll, DoomOutcome, StakesDeclaration, the
# campaign models, ...); typing those nested shapes is out of scope here.


class CampaignStatusResult(ToolEnvelopeSuccess):
    campaign: dict
    scene: dict
    party: list[dict]
    npcs: list[dict]
    clocks: list[dict]
    combat: dict
    valid_ids: dict
    uncommitted_fiction: list[dict]
    pending_rulings: list[dict]
    recent_events: list[dict]


class CharacterSheetResult(ToolEnvelopeSuccess):
    sheet: dict


class CharacterOptionsResult(ToolEnvelopeSuccess):
    origins: dict
    armour_categories: list[str]
    constraints: list[str]


class StatusMixin:
    """Read-only campaign, character, and NPC status, and MCP resource bodies."""

    def _character_summary(self, character: Character) -> dict:
        return {
            "id": character.id,
            "name": character.name,
            "player_discord_id": character.discord_user_id,
            "level": character.level,
            "hp": character.hp,
            "hp_max": character.hp_max,
            "doom_die": character.doom_die,
            "status": character.status,
            "armour": character.armour,
            "shield": character.shield,
            "conditions": [condition.label for condition in character.conditions],
            "attributes": character.attributes.model_dump(),
        }

    def _npc_summary(self, npc: NPC) -> dict:
        return {
            "id": npc.id,
            "name": npc.name,
            "level": npc.level,
            "hp": npc.hp,
            "hp_max": npc.hp_max,
            "damage": self._npc_damage(npc),
            "status": npc.status,
            "motive": npc.motive,
            "actions": npc.actions,
            "flags": npc.flags,
        }

    def _npc_damage(self, npc: NPC) -> int:
        damage = npc.damage
        if "disarmed" in npc.flags:
            penalty = int(
                (self.data.effect_rule("disarm") or {}).get("npc_damage_penalty", 0)
            )
            damage = max(1, damage - penalty)
        return damage

    # -- read-only tools -----------------------------------------------------

    @guard
    def campaign_status(self) -> CampaignStatusResult | ToolEnvelopeFailure:
        """Return the full readable campaign state."""
        state = self.store.read_state()
        manifest = self.store.read_manifest()
        players = self.store.read_players()
        characters = self.store.list_characters()
        link_by_character = {link.character_id: link for link in players.players}

        party = []
        for character in characters:
            summary = self._character_summary(character)
            link = link_by_character.get(character.id)
            summary["player_display_name"] = link.display_name if link else ""
            party.append(summary)

        combat = state.combat
        combat_payload = {"active": combat.active}
        if combat.active:
            combat_payload.update(
                {
                    "round": combat.round,
                    "order": combat.order,
                    "active_actor": combat.active_actor,
                    "ranges": combat.ranges,
                    "actors": {
                        actor_id: {
                            "side": actor.side,
                            "bucket": actor.bucket,
                            "actions_used": actor.actions_used,
                            "actions_max": actor.actions_max,
                            "actions_taken": actor.actions_taken,
                            "turn_open": actor.turn_open,
                        }
                        for actor_id, actor in combat.actors.items()
                    },
                }
            )

        return results.success(
            f"{manifest.title}: session {state.session}, {self._time_string(state.in_game_minutes)}.",
            campaign={
                "title": manifest.title,
                "rules_version": manifest.rules_version,
                "session": state.session,
                "in_game_minutes": state.in_game_minutes,
                "in_game_time": self._time_string(state.in_game_minutes),
                "day": state.day,
                "event_seq": state.event_seq,
            },
            scene=state.scene.model_dump(),
            party=party,
            npcs=[self._npc_summary(npc) for npc in state.npcs.values()],
            clocks=[clock.model_dump() for clock in state.clocks],
            combat=combat_payload,
            valid_ids={
                "characters": [character.id for character in characters],
                "npcs": sorted(state.npcs.keys()),
                "locations": self.store.location_ids(),
                "clocks": [clock.id for clock in state.clocks],
            },
            uncommitted_fiction=[debt.model_dump() for debt in state.fiction_debt],
            pending_rulings=[ruling.model_dump() for ruling in state.pending_rulings],
            recent_events=[
                {
                    "seq": event.get("seq"),
                    "tool": event.get("tool"),
                    "actor_id": event.get("actor_id"),
                    "outcome": event.get("outcome"),
                    "reason": event.get("reason"),
                    # A waived stake must stay tool-visible, not only in the raw log.
                    # An audit found a ledger_settle clearing a stake-bearing debt
                    # whose realized text then left every surface a resuming session
                    # reads; this projection previously dropped the waived array.
                    **(
                        {"waived": event["waived"]}
                        if event.get("tool") == "ledger_settle" and event.get("waived")
                        else {}
                    ),
                }
                for event in self.store.read_events(limit=10)
            ],
            narration_facts=[],
        )

    @guard
    def character_sheet(self, character_id: str) -> CharacterSheetResult | ToolEnvelopeFailure:
        """Return one authoritative character sheet."""
        resolved = self.store.resolve_character_id(character_id)
        warnings = (
            [f"resolved character id {character_id!r} to {resolved!r}; "
             f"use {resolved!r} in later calls."]
            if resolved != character_id
            else []
        )
        character = self.store.read_character(resolved)
        sheet = character.model_dump()
        sheet["armour_protection"] = character.armour_protection()
        sheet["doomed"] = character.doom_die == DEPLETED
        sheet["abilities"] = self.abilities_block(character, day=self.store.read_state().day)
        sheet["eligible_level"] = rules.eligible_level(character.stories)
        sheet["stories_for_next_level"] = rules.stories_required_for_level(
            min(10, character.level + 1)
        )
        return results.success(
            f"{character.name}: {character.hp}/{character.hp_max} hit points, Doom {character.doom_die}.",
            sheet=sheet,
            narration_facts=[],
            warnings=warnings,
        )

    # -- character creation --------------------------------------------------

    @guard
    def character_options(self) -> CharacterOptionsResult | ToolEnvelopeFailure:
        """Return the complete legal character-creation menu. Reads nothing mutable.

        This is the narrator's only grounded source for the origin and background
        lists: the rules tables live on the server side of the protocol, and the
        narrator reads no files and no resources. Every option ``character_create``
        would accept appears here, so a guided session zero offers real choices
        instead of recalled ones.
        """
        origins: dict[str, dict] = {}
        for origin_id, block in self.data.backgrounds["origins"].items():
            origins[origin_id] = {
                "name": block.get("name", origin_id),
                "starting_coins": self.data.starting_coins(origin_id),
                "languages": self.data.origin_languages(origin_id),
                "weapon_table": self.data.starting_weapons(origin_id),
                "backgrounds": [
                    {
                        "id": entry["id"],
                        "name": entry.get("name", entry["id"]),
                        "unique": bool(entry.get("unique")),
                        "attribute_bonus": dict(entry.get("attribute_bonus") or {}),
                        "feature": entry.get("feature", ""),
                    }
                    for entry in block["backgrounds"]
                ],
            }
        return results.success(
            "Character creation menu: "
            + ", ".join(
                f"{origin_id} ({len(block['backgrounds'])} backgrounds)"
                for origin_id, block in origins.items()
            )
            + ".",
            origins=origins,
            armour_categories=list(ARMOUR_CATEGORIES),
            constraints=[
                "Choose one origin: " + ", ".join(ORIGINS) + ".",
                "Choose exactly three backgrounds, at least two from the chosen "
                "origin, at most one marked unique, and no background twice.",
                "character_create rolls the attributes itself (2d6-derived per "
                "attribute), applies background increases, sets hit points equal "
                "to CON, and starts the Doom die at d6.",
                "Weapons are the player's choice; the origin weapon_table lists "
                "the traditional starting draws. Armour must be one armour "
                "category.",
            ],
            narration_facts=[],
        )

    # -- resources -----------------------------------------------------------

    def resource_quick_reference(self) -> str:
        return self.store.quick_reference()

    def resource_subsystems(self) -> str:
        return self.store.subsystems_reference()

    def resource_effects(self) -> str:
        return self.store.effects_reference()

    def resource_status(self) -> str:
        return json.dumps(self.campaign_status(), indent=2, ensure_ascii=False)

    def resource_current_scene(self) -> str:
        state = self.store.read_state()
        return self.store.render_scene_markdown(state)

    def resource_characters(self) -> str:
        payload = [self._character_summary(c) for c in self.store.list_characters()]
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def resource_character(self, character_id: str) -> str:
        return json.dumps(self.character_sheet(character_id), indent=2, ensure_ascii=False)

    def resource_location(self, location_id: str) -> str:
        return json.dumps(self.store.read_location(location_id), indent=2, ensure_ascii=False)
