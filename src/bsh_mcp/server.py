"""Black Sword Hack Model Context Protocol (MCP) server.

Runs over stdio. Every diagnostic goes to stderr; stdout carries only the MCP
protocol stream.

Environment variables:

``BSH_CAMPAIGN_ROOT``
    Project root that holds ``campaign/``, ``rules/``, and ``world/``.
    Defaults to the repository that contains this package.

``BSH_SEED``
    Optional integer seed. Set it only for reproducible tests or demonstrations.

``BSH_DEBUG``
    Set to ``1`` to raise the stderr log level. It never changes tool output.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    # Support ``python src/bsh_mcp/server.py`` as well as ``python -m bsh_mcp.server``.
    # PEP 366 resolves the relative imports below once the package is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "bsh_mcp"

from mcp.server import MCPServer

from .service import GameService, build_service
from .store import discover_root

INSTRUCTIONS = (
    "Deterministic Black Sword Hack mechanics and controlled campaign state. "
    "Call tools for all dice and mechanical changes. Narrate only returned facts. "
    "Call campaign_status at the start of a consequential turn or after any error. "
    "Every tool returns ok=true with narration_facts, or ok=false with an error code "
    "and allowed_next_steps."
)

logger = logging.getLogger("bsh_mcp")

#: Every tool argument whose schema expects an array or an object.
#:
#: Smaller narrator models routinely send a bare string where the schema wants a
#: one-item array, or an empty array where it wants an empty object. Both mistakes
#: fail schema validation before any tool code runs, which costs the table a turn.
#: The middleware below repairs exactly those two shapes and changes nothing else.
#: ``tests/test_mcp_wiring.py`` asserts this map matches the generated schemas, so it
#: cannot drift away from the tool signatures.
ARGUMENT_SHAPES: dict[str, dict[str, str]] = {
    "character_create": {"backgrounds": "array", "weapons": "array"},
    "character_advance": {"attribute_increases": "array"},
    "group_test": {
        "character_ids": "array",
        "advantage_character_ids": "array",
        "disadvantage_character_ids": "array",
    },
    "inventory_update": {
        "add_equipment": "array",
        "remove_equipment": "array",
        "add_weapons": "array",
        "remove_weapons": "array",
    },
    "npc_create": {"actions": "array"},
    "combat_start": {"pc_ids": "array", "npc_ids": "array", "initial_ranges": "object"},
    "rest": {"character_ids": "array"},
    "scene_commit": {
        "visible_changes": "array",
        "hidden_changes": "array",
        "new_clocks": "array",
        "clock_updates": "object",
        "resolved_hooks": "array",
        "new_hooks": "array",
        "present_npcs": "array",
        "exits": "array",
        "environment_tags": "array",
        "persons": "array",
        "departed_persons": "array",
        "refs": "array",
        "retire_facts": "array",
        "party_secrets": "array",
    },
    "session_close": {"character_stories_awarded": "object", "open_hooks": "array"},
}


def repair_arguments(tool_name: str, arguments: dict) -> tuple[dict, list[str]]:
    """Coerce the two argument shapes narrator models get wrong. Pure function."""
    shapes = ARGUMENT_SHAPES.get(tool_name)
    if not shapes or not isinstance(arguments, dict):
        return arguments, []

    repaired = dict(arguments)
    notes: list[str] = []
    for key, expected in shapes.items():
        if key not in repaired:
            continue
        value = repaired[key]
        if expected == "array" and isinstance(value, (str, int, float, bool)):
            repaired[key] = [value]
            notes.append(f"{key}: wrapped a single value in a list")
        elif expected == "object" and isinstance(value, list) and not value:
            repaired[key] = {}
            notes.append(f"{key}: replaced an empty list with an empty object")
    return repaired, notes


async def argument_repair_middleware(ctx, call_next):
    """Repair malformed tool arguments before the protocol validates them."""
    if ctx.method == "tools/call" and isinstance(ctx.params, dict):
        arguments = ctx.params.get("arguments")
        if isinstance(arguments, dict):
            repaired, notes = repair_arguments(ctx.params.get("name", ""), arguments)
            if notes:
                logger.info("repaired %s arguments: %s", ctx.params.get("name"), "; ".join(notes))
                ctx = replace(ctx, params={**ctx.params, "arguments": repaired})
    return await call_next(ctx)


def configure_logging() -> None:
    level = logging.DEBUG if os.environ.get("BSH_DEBUG") == "1" else logging.INFO
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_server(root: Path | str | None = None, seed: int | None = None) -> MCPServer:
    """Build the MCP server bound to one campaign root."""
    resolved_root = Path(root) if root is not None else discover_root()
    if seed is None and os.environ.get("BSH_SEED"):
        seed = int(os.environ["BSH_SEED"])
    service: GameService = build_service(resolved_root, seed=seed)

    mcp = MCPServer(
        "black-sword-hack",
        title="Black Sword Hack campaign engine",
        instructions=INSTRUCTIONS,
        version="0.1.0",
        middleware=[argument_repair_middleware],
    )

    # -- resources -----------------------------------------------------------

    @mcp.resource("bsh://rules/quick-reference", mime_type="text/markdown")
    def rules_quick_reference() -> str:
        """Compact Black Sword Hack rules reference for adjudication."""
        return service.resource_quick_reference()

    @mcp.resource("bsh://rules/subsystems", mime_type="application/json")
    def rules_subsystems() -> str:
        """Dark-pacts subsystem rules: demons, spirits, sorcery, faerie, science, runic."""
        return service.resource_subsystems()

    @mcp.resource("bsh://rules/effects", mime_type="application/json")
    def rules_effects() -> str:
        """The pluggable effect registry: background, Gift, and subsystem mechanics."""
        return service.resource_effects()

    @mcp.resource("bsh://campaign/status", mime_type="application/json")
    def campaign_status_resource() -> str:
        """Current scene, party, NPCs, clocks, combat state, and valid identifiers."""
        return service.resource_status()

    @mcp.resource("bsh://campaign/current-scene", mime_type="text/markdown")
    def current_scene_resource() -> str:
        """The active scene, including game-master-only facts. Never quote hidden facts."""
        return service.resource_current_scene()

    @mcp.resource("bsh://campaign/characters", mime_type="application/json")
    def characters_resource() -> str:
        """Summary of every player character."""
        return service.resource_characters()

    @mcp.resource("bsh://campaign/character/{character_id}", mime_type="application/json")
    def character_resource(character_id: str) -> str:
        """One authoritative character sheet."""
        return service.resource_character(character_id)

    @mcp.resource("bsh://world/location/{location_id}", mime_type="application/json")
    def location_resource(location_id: str) -> str:
        """One authored location, including hidden truths. Never quote hidden truths."""
        return service.resource_location(location_id)

    # -- read-only tools -----------------------------------------------------

    @mcp.tool()
    def campaign_status() -> dict:
        """Read the authoritative campaign state.

        Returns the active scene, the party with hit points and Doom dice, live NPCs,
        clocks, combat state, and every valid identifier. Changes nothing.
        Call this at the start of a consequential turn and after any tool error.
        """
        return service.campaign_status()

    @mcp.tool()
    def character_sheet(character_id: str) -> dict:
        """Read one authoritative character sheet. Changes nothing."""
        return service.character_sheet(character_id)

    # -- character creation --------------------------------------------------

    @mcp.tool()
    def character_options() -> dict:
        """Read the legal character-creation menu. Changes nothing.

        Returns every origin with its name, starting coins, languages,
        starting-weapon table, and complete background list (id, name, attribute
        bonus, feature, unique flag), plus the armour categories and the selection
        constraints character_create enforces. Call this before guiding a player
        through character creation and offer only options it lists; never recite
        origins or backgrounds from memory.
        """
        return service.character_options()

    @mcp.tool()
    def character_create(
        discord_user_id: str,
        name: str,
        origin: str,
        backgrounds: list[str],
        weapons: list[str] | None = None,
        language: str = "",
        armour: str = "none",
        shield: bool = False,
    ) -> dict:
        """Create one player character and write the sheet.

        Rolls 2d6 per attribute, applies background increases, sets hit points equal to
        CON, sets the Doom die to d6, assigns starting coins, and links the Discord user
        to the character. Rejects an illegal background selection without writing.

        origin: barbarian, civilised, or decadent.
        backgrounds: exactly three background ids, at least two from the chosen origin,
        at most one marked unique.
        armour: none, light, medium, or heavy.
        """
        return service.character_create(
            discord_user_id=discord_user_id,
            name=name,
            origin=origin,
            backgrounds=backgrounds,
            weapons=weapons,
            language=language,
            armour=armour,
            shield=shield,
        )

    # -- advancement ---------------------------------------------------------

    @mcp.tool()
    def character_advance(
        character_id: str,
        attribute_increases: list[str] | None = None,
        gift_id: str = "",
    ) -> dict:
        """Advance one player character by exactly one level.

        Call this when the table levels a character up. The tool refuses unless the
        character has collected Stories equal to their current level, so check
        eligibility with character_sheet or session_close first. Advancing two levels
        needs two calls, each with that level's own choices.

        The tool adds the level's hit point, applies the player's chosen attribute
        increases and Gift, and upgrades the Doom die to d8 at level 10. It rejects a
        choice that does not match the level's benefits and writes nothing. It also
        refuses while combat is active: advance between encounters, not mid-fight.

        attribute_increases: the attributes the player raises this level, from STR, DEX,
        CON, INT, WIS, CHA. Levels 2 and 6 raise one; levels 4 and 8 raise two distinct
        attributes. Each raised attribute must sit below 18. Other levels raise none.
        gift_id: at levels 3, 5, 7, and 9, one Gift id from rules/gifts.json the
        character does not already hold. Leave it empty at other levels.
        """
        return service.character_advance(
            character_id=character_id,
            attribute_increases=attribute_increases,
            gift_id=gift_id,
        )

    # -- tests ---------------------------------------------------------------

    @mcp.tool()
    def attribute_test(
        character_id: str,
        attribute: str,
        reason: str,
        stakes_success: str,
        stakes_failure: str,
        stakes_hidden: str = "",
        advantage: bool = False,
        disadvantage: bool = False,
        opponent_level: int | None = None,
        call_on_doom: bool = False,
        failure_mode: str = "gm_choice",
        category: str = "",
    ) -> dict:
        """Roll one d20 roll-under attribute test with declared stakes.

        Call this whenever an action has a meaningful chance and consequence of failure.
        Do not call it for an ordinary active search that simply succeeds. Inside a
        fight, a declaration about the actor's own position or footing on their own
        turn -- circling behind, a trip, a climb -- rolls here.

        Declare the stakes before rolling: stakes_success is what durably changes if
        the dice favour, stakes_failure what durably changes if they do not, in one
        sentence each. You already must know both to justify the roll. The server
        records your words verbatim, tags the branch the dice realize, and holds it
        as an unratified outcome until scene_commit. stakes_hidden is optional and
        never shown to players.

        The tool applies Advantage, Disadvantage, conditions, Threat Level, and Doom.
        It decides dice and outcome. It does not decide the fictional cost.

        attribute: STR, DEX, CON, INT, WIS, or CHA.
        opponent_level: the opposing creature's level, when one opposes the action.
        call_on_doom: true only when the player chooses to call on Doom.
        failure_mode: fail, success_at_cost, or gm_choice. Metadata for the narrator.
        category: optional structured tag for the kind of test — stealth, pickpocketing,
            eavesdropping, streetwise, or sorcery. Some backgrounds grant Advantage on a
            category; an unknown category is refused with the legal list. Pass sorcery
            (attribute INT) for a Forbidden-knowledge spell casting: a critical failure
            then rolls the Torn Veil table itself and applies its consequence, returned
            under torn_veil, exactly as a demonic invocation's Doom depletion rolls
            Demon's Revenge through use_ability. Do not roll Torn Veil yourself in
            fiction; it is not a separate tool call.

        A test rolled by the active combatant on their own turn is that character's
        combat action and spends one action exactly as an attack would. The result's
        combat_action block reports the count, and the turn advances itself when the
        last action is spent. Stakes in combat may only promise what the failing
        character suffers or fails to do; never promise that an enemy attacks, because
        an enemy acts only on its own turn through combat_defend.
        """
        return service.attribute_test(
            character_id=character_id,
            attribute=attribute,
            reason=reason,
            stakes_success=stakes_success,
            stakes_failure=stakes_failure,
            stakes_hidden=stakes_hidden,
            advantage=advantage,
            disadvantage=disadvantage,
            opponent_level=opponent_level,
            call_on_doom=call_on_doom,
            failure_mode=failure_mode,
            category=category,
        )

    @mcp.tool()
    def group_test(
        character_ids: list[str],
        attribute: str,
        reason: str,
        stakes_success: str,
        stakes_failure: str,
        stakes_hidden: str = "",
        advantage_character_ids: list[str] | None = None,
        disadvantage_character_ids: list[str] | None = None,
        opponent_level: int | None = None,
        category: str = "",
    ) -> dict:
        """Roll one attribute test per character and return the group result.

        The group succeeds when at least half the participants succeed.
        Use this for coordinated actions such as a whole party crossing unseen.

        Declare the group's stakes before rolling, one sentence per branch:
        stakes_success for what durably changes if the group passes,
        stakes_failure for what durably changes if it does not. stakes_hidden
        is optional and never shown to players.

        category: optional structured tag (see attribute_test) applied to every
        participant's test; a background that grants Advantage on it edges only that
        participant's roll.
        """
        return service.group_test(
            character_ids=character_ids,
            attribute=attribute,
            reason=reason,
            stakes_success=stakes_success,
            stakes_failure=stakes_failure,
            stakes_hidden=stakes_hidden,
            advantage_character_ids=advantage_character_ids,
            disadvantage_character_ids=disadvantage_character_ids,
            opponent_level=opponent_level,
            category=category,
        )

    @mcp.tool()
    def usage_roll(
        owner_id: str,
        resource_id: str,
        reason: str,
        advantage: bool = False,
        disadvantage: bool = False,
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> dict:
        """Roll one Usage Die for a tracked resource.

        A result of 1 or 2 steps the die down one grade along d20, d12, d10, d8, d6, d4.
        A d4 that steps down is depleted. Mutates the named resource on the sheet.

        Optional stakes record what depletion durably means (stakes_failure) and
        what holding means (stakes_success). A downgrade always enters the
        unratified-outcomes ledger, with or without stakes.
        """
        return service.usage_roll(
            owner_id=owner_id,
            resource_id=resource_id,
            reason=reason,
            advantage=advantage,
            disadvantage=disadvantage,
            stakes_success=stakes_success,
            stakes_failure=stakes_failure,
            stakes_hidden=stakes_hidden,
        )

    @mcp.tool()
    def doom_roll(
        character_id: str,
        reason: str,
        mode: str = "roll",
        disadvantage: bool = False,
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> dict:
        """Roll, call on, or restore a Doom die.

        mode 'roll' rolls the Doom die; 1 or 2 steps it down. A depleted Doom die leaves
        the character Doomed, testing everything at Disadvantage until a long rest.
        mode 'call_on_doom' rolls the die, always steps it down, and returns a value to
        subtract from an attribute test.
        mode 'restore' returns the die to its maximum.

        attribute_test already rolls Doom on a critical failure, and combat_attack already
        rolls Doom for a repeated action. Call this tool only for other triggers.
        """
        return service.doom_roll(
            character_id=character_id,
            reason=reason,
            mode=mode,
            disadvantage=disadvantage,
            stakes_success=stakes_success,
            stakes_failure=stakes_failure,
            stakes_hidden=stakes_hidden,
        )

    # -- NPCs ----------------------------------------------------------------

    @mcp.tool()
    def npc_create(
        name: str,
        level: int,
        armour: str = "none",
        motive: str = "",
        actions: list[str] | None = None,
        location_id: str = "",
        present_in_scene: bool = True,
    ) -> dict:
        """Create one quick NPC from the level table.

        Level 1 to 10 sets hit points and damage. Armour adds hit points rather than
        subtracting damage. Supply one motive and one or two named actions.
        """
        return service.npc_create(
            name=name,
            level=level,
            armour=armour,
            motive=motive,
            actions=actions,
            location_id=location_id,
            present_in_scene=present_in_scene,
        )

    # -- combat --------------------------------------------------------------

    @mcp.tool()
    def combat_start(
        pc_ids: list[str],
        npc_ids: list[str],
        initial_ranges: dict[str, str] | None = None,
        reason: str = "",
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> dict:
        """Open combat and roll initiative.

        Each player character makes a WIS test. Success acts before the opposition,
        failure after. A critical success grants three actions on the first turn; a
        critical failure grants one.

        The first actor's turn is already open when this returns; follow next_step.
        When the opposition acts first, narrate the enemy action and resolve it with
        combat_defend -- a player's declared attack then resolves on their own turn,
        after the opposition acts, and must not be rolled or narrated before that.

        initial_ranges maps an NPC id to close, nearby, far_away, or distant.
        Create every NPC with npc_create first.
        """
        return service.combat_start(
            pc_ids=pc_ids,
            npc_ids=npc_ids,
            initial_ranges=initial_ranges,
            reason=reason,
            stakes_success=stakes_success,
            stakes_failure=stakes_failure,
            stakes_hidden=stakes_hidden,
        )

    @mcp.tool()
    def combat_begin_turn(actor_id: str) -> dict:
        """Confirm the active actor's turn is open (turns now open themselves).

        combat_start opens the first turn and every resolved action that ends a turn
        opens the next one, so this call is normally unnecessary. Calling it for the
        active actor anyway succeeds and reports the actions left without resetting
        anything; calling it for anyone else is refused, because only the active
        actor may act.
        """
        return service.combat_begin_turn(actor_id=actor_id)

    @mcp.tool()
    def combat_move(character_id: str, target_id: str) -> dict:
        """Spend one action to close one range band toward a target NPC.

        Melee needs close range. A fight often opens at nearby, so a character spends a
        move action to close the distance before a melee strike lands. One move changes
        one band. Ranges are per opponent, so this closes the band toward one NPC.
        """
        return service.combat_move(character_id=character_id, target_id=target_id)

    @mcp.tool()
    def combat_attack(
        attacker_id: str,
        target_id: str,
        attack_type: str = "melee",
        advantage: bool = False,
        disadvantage: bool = False,
        two_handed: bool = False,
        weapon_effect: str = "none",
        unarmed: bool = False,
        runic: bool = False,
        one_handed_blade: bool = False,
        target_unaware: bool = False,
        poisoned: bool = False,
    ) -> dict:
        """Resolve one player attack against one NPC.

        Melee tests STR and needs close range. Ranged tests DEX. The tool applies Threat
        Level, rolls damage on a success, applies a critical as maximum base damage plus
        one extra die, updates hit points, and marks a dead NPC.

        Repeating the attack action in the same turn rolls Doom automatically.

        Spending the last action advances the turn itself, and dropping the last enemy
        ends the fight itself: read turn_advanced, next_actor, combat_over, and
        next_step from the result rather than calling combat_end_turn. Only the active
        actor may attack; an attack out of turn is refused with the actor whose turn it
        actually is.

        weapon_effect accepts none, brutal, disarm, pin_down, shove, entangle, cleave, or
        impale. Disarm, pin_down, and entangle deal no damage and instead flag the
        target (reduced damage output, or blocked from changing range band until it
        spends an action to break free). Shove deals no damage and pushes the target
        one band away. Cleave and impale both strike beyond the named target on the
        same swing: cleave also hits every other enemy sharing the target's range
        band, and impale, only when the blow fells the target, carries through for
        the same damage to one more enemy in that band, chosen at random. Any other
        value is rejected before dice are rolled: adjudicate it in fiction and record
        durable results with scene_commit.

        Set runic true to strike with a granted runic weapon: damage equals the attribute
        the weapon's personality fixes, and a kill rolls the on-kill effect. A runic strike
        takes no weapon_effect and is not unarmed.

        Set one_handed_blade true when the weapon in hand is a one-handed blade; a Sword
        master then tests DEX instead of STR on a melee attack. A Hunter's first ranged
        attack of a fight hits automatically without a test and adds their level to damage.

        Set target_unaware true when the fiction holds the target unaware of the attacker;
        an Assassin's strike then deals damage equal to their DEX score in place of the
        rolled weapon damage. The engine only honours this while the target has not itself
        reacted this fight (its own turn opened, or it attacked someone) and this character
        has not already spent their own unaware strike this fight; being struck does not
        itself count as reacting, so a second Assassin can still find the same target
        unaware. Outside that window the flag is ignored and normal damage applies.

        Set poisoned true to coat the strike with one prepared Herbalist poison dose.
        Refused before any dice roll if no dose is in stock, or alongside runic, unarmed,
        or a non-damaging weapon_effect (disarm, pin_down, shove, entangle). The dose is
        spent on the swing whether it hits or misses, and a damaging hit adds d6 poison
        damage.

        A positioning declaration -- circling behind, backing off, working for an
        angle -- is not an attack; resolve it with attribute_test on the actor's own
        turn instead.
        """
        return service.combat_attack(
            attacker_id=attacker_id,
            target_id=target_id,
            attack_type=attack_type,
            advantage=advantage,
            disadvantage=disadvantage,
            two_handed=two_handed,
            weapon_effect=weapon_effect,
            unarmed=unarmed,
            runic=runic,
            one_handed_blade=one_handed_blade,
            target_unaware=target_unaware,
            poisoned=poisoned,
        )

    @mcp.tool()
    def combat_defend(
        defender_id: str,
        attacker_id: str = "",
        method: str = "dodge",
        incoming_damage: int | None = None,
        ranged: bool = False,
        shield: bool = False,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> dict:
        """Resolve one player defence against an incoming attack.

        Enemy attacks resolve through this tool, so the player always rolls. Defending
        never consumes the defender's own actions, but it spends the attacking enemy's
        one action for its turn -- and when that exhausts the enemy's actions the turn
        advances itself: read next_actor and next_step from the result. An enemy whose
        turn is closed cannot attack again until the order reaches it in the next
        round; do not narrate an enemy attack outside its own turn.

        Parry tests STR and needs a held object. Dodge tests DEX and is the only option
        against a ranged attack. A shield grants Advantage when parrying and breaks on a
        critical failure. A critical failure ignores armour.

        Omit incoming_damage to use the named NPC's recorded damage.
        """
        return service.combat_defend(
            defender_id=defender_id,
            attacker_id=attacker_id,
            method=method,
            incoming_damage=incoming_damage,
            ranged=ranged,
            shield=shield,
            advantage=advantage,
            disadvantage=disadvantage,
        )

    @mcp.tool()
    def combat_end_turn(actor_id: str = "") -> dict:
        """End the active actor's turn early, before its actions are all spent.

        Turns advance themselves when the last action is spent, so this is only for a
        combatant who stops acting with actions left -- holding back, hesitating, or
        an enemy that does something other than attack. Increments the round after the
        last actor. Ends combat when no enemy is alive or no player character is
        standing. A finished fight rolls each participant's held reusable marvel
        Usage Die.
        """
        return service.combat_end_turn(actor_id=actor_id)

    @mcp.tool()
    def combat_close(reason: str) -> dict:
        """Close an active combat for any reason other than elimination.

        `combat_end_turn` only ends a fight when one side is empty. Every other way a
        fight ends -- fleeing, surrendering, a truce, or any other mutual disengagement
        -- closes through this tool instead, so a fight can never stay active once the
        fiction has moved on. Clears the active actor and rolls the same post-combat
        bookkeeping a decisive `combat_end_turn` would. Refuses when no combat is active.
        """
        return service.combat_close(reason=reason)

    # -- possessions ----------------------------------------------------------

    @mcp.tool()
    def inventory_update(
        character_id: str,
        reason: str,
        coins_delta: int = 0,
        add_equipment: list[str] | None = None,
        remove_equipment: list[str] | None = None,
        add_weapons: list[str] | None = None,
        remove_weapons: list[str] | None = None,
    ) -> dict:
        """Apply one audited change to a character's coins, equipment, or weapons.

        Coins and carried items are mechanical state exactly like hit points: they
        change only through this tool. Call it whenever the fiction moves a
        possession -- looting a body or a stall, picking something up, dropping or
        handing over an item, paying or receiving coins outside a negotiated trade.
        Pass a signed coins_delta and exact item names; the tool refuses before any
        write when coins would go below zero or a removed item is not held. A runic
        weapon is put down the same way: name it in remove_weapons, and the weapon
        leaves with its kill verdict.

        Never use this for a negotiated merchant purchase: the trade confirmation
        flow owns those, and only its authenticated confirmation moves the coins.

        Answer inventory questions from character_sheet or this tool's own result,
        never from memory or the scene digest.
        """
        return service.inventory_update(
            character_id=character_id,
            reason=reason,
            coins_delta=coins_delta,
            add_equipment=add_equipment,
            remove_equipment=remove_equipment,
            add_weapons=add_weapons,
            remove_weapons=remove_weapons,
        )

    # -- recovery ------------------------------------------------------------

    @mcp.tool()
    def rest(
        character_ids: list[str],
        rest_type: str,
        safe_environment: bool = False,
        reason: str = "",
    ) -> dict:
        """Resolve a party's short- or long-rest declaration.

        Call this for every rest declaration the party makes, short or long --
        including a repeat later in the same session you expect to already be
        blocked. Whether a rest is available now depends on campaign state that
        moves between turns, so an earlier result or refusal in this conversation
        says nothing about this one: the tool's own response, never your memory of
        a previous answer, is what settles a rest. Restating an earlier answer
        without a fresh call states a mechanical fact no tool returned this turn
        -- the same failure as never calling at all.

        The rest limits are the tool's to enforce, not yours to predict or
        pre-refuse: a declaration reaches this tool directly -- do not refuse it
        yourself or invent a confirmation step first -- and you narrate only what
        it returns, what was restored or why it refused.

        character_ids names everyone resting. rest_type is short or long.

        safe_environment matters only to a long rest and is trusted, never
        verified: the tool takes your word for the fiction. Set it true only when
        the party has actually secured a safe place -- a wilderness camp is not
        safe without a specific fictional reason. Setting it true unearned grants
        a durable recovery the fiction never earned, a worse defect than any
        refusal.

        Whatever comes back, restored or refused, deliver it in fiction: the
        world answers the party, never a tool, a system, or an out-of-character
        aside.
        """
        return service.rest(
            character_ids=character_ids,
            rest_type=rest_type,
            safe_environment=safe_environment,
            reason=reason,
        )

    @mcp.tool()
    def helpless_roll(character_id: str, reason: str = "", carer_id: str = "") -> dict:
        """Roll the Helpless table for a player character at 0 hit points.

        Call this once the fight ends or the character reaches safety. On a d6, results
        1 to 5 restore d4 hit points; result 6 kills the character.

        Pass carer_id when an ally with a tending effect (a Surgeon) treats the downed
        character first: the carer's gating test resolves in the same transaction and,
        on success, the table rolls a d4 instead of a d6, putting the worst results out
        of reach.
        """
        return service.helpless_roll(
            character_id=character_id, reason=reason, carer_id=carer_id
        )

    @mcp.tool()
    def grant_runic_weapon(character_id: str, name: str, personality: str) -> dict:
        """Grant a sentient runic weapon a character finds in play (a rare event).

        personality is one of brutal, vicious, patient, cunning, judgemental, or prideful,
        and fixes the weapon's damage to STR, DEX, CON, INT, WIS, or CHA in turn. The tool
        rolls the weapon's INT with 2d6 and makes its session test at once; that test is
        announced to the table for you, like any roll. Attack with it by calling
        combat_attack with runic true; put it down with inventory_update remove_weapons.
        A character holds one at a time: the tool refuses while one is held.
        """
        return service.grant_runic_weapon(
            character_id=character_id,
            name=name,
            personality=personality,
        )

    @mcp.tool()
    def use_ability(
        character_id: str,
        ability_id: str,
        target_id: str = "",
        mode: str = "use",
        choice: str = "",
    ) -> dict:
        """Activate, deactivate, or spend a character's activatable ability.

        Read the character sheet's 'abilities' block for the legal ability_id values.
        Use mode 'use' to spend a bounded resource (Legionnaire, Sophist, Bookworm),
        mode 'activate' to enter a toggle stance (Berserker rage) and 'deactivate' to
        drop it. Passive effects apply on their own and need no call.

        choice names a value for two other kinds of ability, both listed on the sheet
        alongside their legal choices: an 'intent' ability (e.g. herbalist_stock)
        records a standing declaration for later -- set it any time, independent of
        when it resolves; a 'dose' ability (e.g. herbalist_dose) spends one already-
        prepared dose of the named type, applying its effect immediately (target_id
        names who receives it; omit it to affect the caller).
        """
        return service.use_ability(
            character_id=character_id,
            ability_id=ability_id,
            target_id=target_id,
            mode=mode,
            choice=choice,
        )

    # -- narrative state -----------------------------------------------------

    @mcp.tool()
    def scene_commit(
        public_summary: str,
        location_id: str = "",
        in_game_time_delta_minutes: int = 0,
        visible_changes: list[str] | None = None,
        hidden_changes: list[str] | None = None,
        new_clocks: list[dict] | None = None,
        clock_updates: dict[str, int] | None = None,
        resolved_hooks: list[str] | None = None,
        new_hooks: list[str] | None = None,
        scene_title: str = "",
        present_npcs: list[str] | None = None,
        exits: list[str] | None = None,
        environment_tags: list[str] | None = None,
        persons: list[dict] | None = None,
        departed_persons: list[str] | None = None,
        refs: list[str] | None = None,
        retire_facts: list[str] | None = None,
        party_secrets: list[str] | None = None,
    ) -> dict:
        """Commit a durable fictional change to the campaign files.

        Call this after the party discovers an entrance, alarms a faction, moves to a new
        location, learns a secret, or changes the situation in a way that must survive a
        restart.

        visible_changes records what the party knows, including the party's own secret
        actions: a hiding place the party made is the party's knowledge, not a
        game-master secret, even when the scene hides it from others. hidden_changes
        records only facts the party does not know. Filing a party action as hidden
        makes the game master unable to answer the party about their own deed.

        This tool never changes hit points, Doom, inventory, or any other mechanic that
        belongs to a dedicated tool.

        new_clocks entries take the form {"id": "tide", "name": "The tide", "segments": 6}.
        clock_updates maps a clock id to a signed segment change.

        environment_tags replaces the scene's open-vocabulary descriptors (e.g. "natural",
        "urban") when passed; omit it to leave the current tags unchanged. Some
        backgrounds' rest-time mechanics read these -- e.g. a Herbalist's stock only
        replenishes on a long rest whose scene carries "natural". Tag the scene here
        before that rest happens, not as an argument to rest.

        persons records people present who have no NPC record yet -- a clerk, a
        trader, someone the party only talks to. Each entry takes the form
        {"name": "Salt Magistrate clerk", "role": "counts barrels for the Magistrates"};
        role is optional. The server allocates the id from the name, so the same
        name later names the same person, and npc_create on that name promotes the
        person to a full NPC when a fight needs one. Pass present_npcs, not persons,
        for anyone npc_create already created. departed_persons names persons who
        have left the scene, by the same name; a move to a new location_id already
        takes every person and every located NPC of the old scene with it.

        refs optionally lists the recorded identifiers this commit's changes are
        about (characters, NPCs, persons, objects); unknown identifiers are dropped.
        retire_facts removes a fact that is no longer true from the visible record:
        quote the line exactly as the Visible facts list states it. The fact stays
        in the campaign's history; it simply stops being current.

        party_secrets records knowledge the party holds that the world's people do
        not -- a hiding place the party made, a plan they whispered. It is the
        party's own knowledge: answer the party about it freely, and never treat it
        as a game-master secret. hidden_changes remains only for facts the party
        does not know.
        """
        return service.scene_commit(
            public_summary=public_summary,
            location_id=location_id,
            in_game_time_delta_minutes=in_game_time_delta_minutes,
            visible_changes=visible_changes,
            hidden_changes=hidden_changes,
            new_clocks=new_clocks,
            clock_updates=clock_updates,
            resolved_hooks=resolved_hooks,
            new_hooks=new_hooks,
            scene_title=scene_title,
            present_npcs=present_npcs,
            exits=exits,
            environment_tags=environment_tags,
            persons=persons,
            departed_persons=departed_persons,
            refs=refs,
            retire_facts=retire_facts,
            party_secrets=party_secrets,
        )

    @mcp.tool()
    def ledger_settle(reason: str) -> dict:
        """Clear the unratified-outcome ledger without writing a scene entry.

        RUNTIME-ONLY. The narrator engine calls this during its settle step; it is
        excluded from the narrator model's tool surface by `src/narrator/policy.py`,
        and a game-master prompt must never be able to reach it. The waived outcomes
        stay in campaign state history and the audit log with their realized stakes;
        only the pending scene entry is skipped.

        reason: one or two sentences on why no scene entry is needed.
        """
        return service.ledger_settle(reason=reason)

    @mcp.tool()
    def ability_apply_ruling(ruling_id: str, choice: str = "", source: str = "model") -> dict:
        """Apply a fictional choice to an open pending ruling. RUNTIME-ONLY.

        The narrator engine's adjudicate step calls this once per turn to resolve a
        pending_ruling (e.g. which possession a demon stole) from the narration. It
        is excluded from the narrator model's tool surface by `src/narrator/policy.py`.
        An out-of-options choice falls to the ruling's default strategy.
        """
        return service.ability_apply_ruling(ruling_id=ruling_id, choice=choice, source=source)

    @mcp.tool()
    def session_close(
        session_title: str,
        public_summary: str,
        character_stories_awarded: dict[str, int] | None = None,
        open_hooks: list[str] | None = None,
        next_intention: str = "",
        accept_uncommitted: bool = False,
    ) -> dict:
        """Close the session.

        Writes a dated summary, awards Stories, reports advancement eligibility, clears
        session-length conditions, re-rolls every runic weapon's session INT test for the
        next session (announced for you), creates a backup archive, and opens the next
        session log.

        Refuses once if any rolled outcome is still unratified (call scene_commit first);
        pass accept_uncommitted=true to close anyway. The debt is never silently dropped
        either way -- an accepted close still leaves it recorded and settleable later.
        """
        return service.session_close(
            session_title=session_title,
            public_summary=public_summary,
            character_stories_awarded=character_stories_awarded,
            open_hooks=open_hooks,
            next_intention=next_intention,
            accept_uncommitted=accept_uncommitted,
        )

    logger.info("black-sword-hack MCP server bound to %s", resolved_root)
    return mcp


def main() -> None:
    configure_logging()
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
