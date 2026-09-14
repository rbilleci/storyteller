# Black Sword Hack Quick Reference

This reference summarises the Black Sword Hack — Ultimate Chaos Edition System Reference Document (SRD) v1.0.2 for live adjudication. See ../../../rules/attribution.md for the licence. Where this summary and the SRD disagree, the SRD governs.

The Model Context Protocol (MCP) server implements every rule marked "tool" here. The narrator never computes those results by hand.

## Attribute Tests

A character tests an attribute when an action carries a meaningful chance of failure and a meaningful consequence. Roll one twenty-sided die (d20). A total strictly below the attribute score succeeds. A total equal to or above the score fails, or succeeds at a cost the narrator names. An unmodified 1 is a critical success. An unmodified 20 is a critical failure and forces a Doom roll.

Criticals read the unmodified selected die. Threat Level and Doom never create or erase a critical.

Tool: `attribute_test`. Attributes are STR, DEX, CON, INT, WIS, and CHA.

## Advantage and Disadvantage

Advantage rolls two d20 and keeps the favourable result. Disadvantage rolls two d20 and keeps the unfavourable result. For a roll-under test, Advantage keeps the lower die and Disadvantage keeps the higher die. For a damage roll, Advantage keeps the higher die. Advantage and Disadvantage cancel when both apply, leaving one die.

## Threat Level

When an opponent's level exceeds the character's level, add the difference to the character's d20. A lower-level opponent grants no bonus. The modifier is never negative.

Tool: every combat tool applies this automatically. Pass `opponent_level` to `attribute_test` outside combat.

## Group Tests

Each participating character tests the relevant attribute. The group succeeds when at least half the participants succeed.

Tool: `group_test`.

## Perception

A character who actively searches a place they can reasonably search finds what is there. Call for an INT test only when time, pressure, or uncertainty makes the search consequential. Habitual perception checks are wrong at this table.

## Usage Dice

A Usage Die tracks a consumable without counting individual items. The chain runs d20, d12, d10, d8, d6, d4. Roll the current die when the resource sees use. A result of 1 or 2 steps the die down one grade. A d4 that steps down is depleted.

Tool: `usage_roll`.

## Doom

Every character starts with a d6 Doom die. Doom rolls follow Usage Die mechanics: a 1 or 2 steps the die down.

A Doom roll triggers on a critical failure, on repeating the same action inside one combat turn, and on certain Gifts and powers. A player may also call on Doom: roll the die, subtract the result from an attribute test, and step the die down regardless of the result.

A depleted Doom die leaves the character Doomed. A Doomed character tests every attribute and rolls every damage die at Disadvantage until a long rest. A long rest restores Doom to its maximum.

Tools: `doom_roll` with mode `roll`, `call_on_doom`, or `restore`. The `attribute_test` and `combat_attack` tools roll Doom automatically on their own triggers.

## Turns and Actions

Outside combat, one turn covers whatever span the narrator chooses, and normally permits movement plus one action.

Inside combat, each player character normally takes two actions. Movement, an attack, item use, and magic each consume one action. Taking the same action twice requires a Doom roll before the second action. Enemies take one action and one move each turn and cannot repeat an action.

Defending against an enemy action costs the defender nothing from their own two actions.

Tools: `combat_attack`, `combat_defend`, and `combat_move` resolve everything; turns open and advance themselves. An `attribute_test` rolled by the active character on their own turn also spends one action. `combat_end_turn` only ends a turn early; `combat_begin_turn` only re-reads an open turn.

## Possessions

Coins, equipment, and weapons change only through `inventory_update` -- loot, pickups, drops, handovers, and payments outside a negotiated trade. A negotiated purchase resolves only through the trade confirmation flow. Answer inventory questions from `character_sheet`.

## Distances

Four abstract bands describe every distance: close at roughly 1.5 metres, nearby at roughly 10 metres, far away at roughly 20 metres, and distant beyond 20 metres. One move changes one band. The campaign files store bands, never coordinates.

Melee attacks need close range. Ranged attacks reach far away.

## Initiative

At the start of combat each player character makes a WIS test. Success acts before the opposition. Failure acts after. A critical success grants three actions on the first turn. A critical failure grants one action on the first turn.

Tool: `combat_start`.

## Attacking

A melee attack tests STR. A ranged attack tests DEX. A success rolls the character's weapon damage. A critical success deals maximum base damage plus one additional damage die.

Default weapon damage is a six-sided die (d6). Default unarmed damage is a four-sided die (d4). A two-handed weapon grants Advantage on the damage roll.

Tool: `combat_attack`.

## Defending

The player always rolls. Parrying tests STR and requires a held object. Dodging tests DEX and is the only defence against a ranged attack. A shield grants Advantage when parrying and breaks on a critical failure.

A failed defence applies the attacker's damage. A critical failure ignores armour entirely.

Tool: `combat_defend`.

## Damage and Armour

Armour subtracts its protection rating from incoming damage, to a minimum of zero. Light armour protects 1, medium armour protects 2, and heavy armour protects 3.

Quick NPCs handle armour differently: armour adds hit points instead of subtracting damage. Light armour adds 2 hit points, medium adds 3, and heavy adds 4.

## Zero Hit Points

An NPC or monster at 0 hit points is dead. A player character at 0 hit points becomes Helpless. Once the fight ends or the character reaches safety, roll a six-sided die (d6) on the Helpless table.

| d6 | Result | Effect |
|---:|---|---|
| 1 | Scratched | A new scar and nothing worse. |
| 2 | Missed | One random piece of equipment is destroyed. |
| 3 | Impaired | Disadvantage on DEX tests for the rest of the session. |
| 4 | Injured | Disadvantage on all attribute tests for the rest of the session. |
| 5 | Butchered | Permanently lose 1 STR on a d6 of 1 to 3, or 1 DEX on a d6 of 4 to 6. |
| 6 | Killed | The character dies. Create a replacement of the same level. |

A survivor on results 1 to 5 immediately regains d4 hit points.

Tool: `helpless_roll`.

## Recovery

A short rest lasts one hour, happens at most once per day, and restores half CON, rounded down. A long rest lasts six hours, requires a genuinely safe environment, restores every hit point, restores the Doom die, and refreshes features that need a long rest.

A wilderness camp is not safe without a specific fictional reason.

Tool: `rest`.

## Character Creation

Roll 2d6 for each of the six attributes in order and map the total to a score.

| 2d6 | Score |
|---:|---:|
| 2-3 | 8 |
| 4-5 | 9 |
| 6-7 | 10 |
| 8-9 | 11 |
| 10-11 | 12 |
| 12 | 13 |

Choose one origin: Barbarian, Civilised, or Decadent. Choose three backgrounds, at least two from the chosen origin, and no more than one marked unique. Apply each background's attribute increase.

Hit points equal the CON score. The Doom die starts at d6. Weapon damage starts at d6 and unarmed damage at d4. Starting coins are 25 for Barbarian, 50 for Civilised, and 100 for Decadent.

Tool: `character_create`. The background lists live in ../../../rules/backgrounds.json.

## Quick NPCs

| Level | HP | Damage |
|---:|---:|---:|
| 1 | 5 | 4 |
| 2 | 10 | 5 |
| 3 | 15 | 6 |
| 4 | 20 | 7 |
| 5 | 25 | 8 |
| 6 | 30 | 9 |
| 7 | 35 | 10 |
| 8 | 40 | 11 |
| 9 | 45 | 12 |
| 10 | 50 | 13 |

Humanoid adversaries surrender or flee below 3 hit points, or after losing half their force. Only fanatics fight to the death.

Tool: `npc_create`.

## Advancement

A character advances after collecting Stories equal to their current level. Reaching level 2 needs 1 Story. Reaching level 3 needs 3 Stories in total. Each level adds 1 hit point. Levels 2, 4, 6, and 8 raise attributes. Levels 3, 5, 7, and 9 grant a Gift. Level 10 upgrades the Doom die to d8.

Tool: `session_close` records Stories and reports eligibility. A human applies the level-up choices.

## Dark-pacts Subsystems

The `bsh://rules/subsystems` resource carries the six supernatural subsystems: demonic pacts, spirit alliances, sorcery, faerie ties, twisted science, and runic weapons. Read it to adjudicate from canon rather than memory. The SRD source sits at ../../../docs/srd/dark-pacts.md.

Many parts are deterministic. Sorcery casting rolls the d100 spell table, and a Forbidden Knowledge character draws four spells at creation. A demonic invocation that depletes the Doom die rolls the six-face Demon's Revenge table. A sorcery critical failure rolls the six-face Torn Veil table. Demon and spirit powers invoke through `use_ability` mode `invoke`, which rolls the Doom die. Faerie ties invoke through the same tool but roll the Doom die only for the SRD-marked ties: Barrow wisdom, Doomed to greatness, True faith, and Elfin secret. Twisted-science marvels build with mode `bind`, which spends coins and workshop time, and fire with mode `invoke`, which spends a Usage Die or one single use. A runic weapon comes from `grant_runic_weapon`; strike with it by calling `combat_attack` with `runic` true; put it down by naming it in `inventory_update`'s `remove_weapons`. Each subsystem's individual power effect text still stays a narrator ruling: adjudicate it and record durable outcomes with `scene_commit`. Demon pacts, spirit alliances, faerie ties, and twisted-science marvels each cap a character at two. See ../../../DEFERRED.md.
