# Rules attribution and adaptations

This project uses Black Sword Hack — Ultimate Chaos Edition SRD v1.0.2,
by Alexandre "Kobayashi" Jeannette, published by The Merry Mushmen.
The [SRD](https://blackswordhack.github.io/) is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
See [NOTICE](../NOTICE) for complete credits, file scope, and licensing.
Only SRD material is used; other Black Sword Hack products require separate permission.

The SRD has been selected, transcribed into JSON, paraphrased into short operational
descriptions, and adapted for automation. Background benefits and Gift descriptions
are paraphrases unless identified as quotations. The source is authoritative about
the tabletop rules; the adaptations below describe how this engine differs.
Earlier paraphrases were corrected to restrict Battle Hardened to unmodified combat
rolls and to model Herbalist preparations as discrete doses, not Usage Dice.

The Black Estuary setting under `world/` is original project content.
`weapon-effects.json` defines project-authored weapon qualities; the SRD names
the two-handed designation, not the additional brutal, disarm, pin-down, shove,
entangle, cleave, and impale qualities used here. The origin weapon tables in
`backgrounds.json` are transcribed from the SRD in die order.

## Automation choices and house rules

| Mechanic | Engine interpretation |
| --- | --- |
| Legionnaire | Predeclared Advantage for an ally replaces the SRD's reactive reroll. Committed defence damage is not rewound. Success probabilities on an otherwise unedged roll align, but critical selection differs. |
| Assassin | The narrator declares an unaware target; the engine also requires an open per-attacker strike window before the target has reacted. The attempt spends the window. This is not a full simulation of awareness. |
| Herbalist | A standing preparation choice replenishes discrete doses after a long rest in a scene tagged `natural`. Replenishment replaces that type's stock. The tag operationalizes the SRD's proximity-to-nature condition. |
| Poison | A coated-weapon dose can be spent on an attack, including a miss. The SRD does not specify a delivery procedure. Runic damage takes precedence, but the dose is still consumed. |
| Demon's Revenge | An ally is another living character in the campaign roster. Eligible weapons form a ruling the engine resolves from narration, with a random default if no choice is implied. |
| Second Wind | Once per in-game day, only for the spender. Full-HP and dead-character attempts are refused; healing a Helpless character above zero restores normal status. |
| Resourceful | Restores a recorded resource maximum, excluding Doom. An unknown maximum or an already-full resource is refused. |
| Bloodlust | Permanently steps the damage die up, for armed and unarmed attacks, capped at d12. Attribute-fixed damage is unaffected. |
| Riddle of Steel | A standing held-weapon designation supplies d12 on armed attacks. The selected damage die's 1 breaks it after damage; discarded dice do not. Redesignation is allowed. |
| Spirit Alliance | Advantage applies to the declared specific spirit's invocation. |
| Dark Revelation / Dubious Friendships | Advantage on the relevant backlash table keeps the lower of two faces, treating higher table results as worse. |
| Cleave / Impale | Project-authored secondary-hit effects reuse primary damage and select only eligible enemies in the same fight/range. Impale selects one eligible enemy randomly. |
| Survivor's Luck | Predeclared protection stakes a held weapon and remains armed until an attack would deal damage. It cancels that damage and destroys the staked weapon, without reversing an earlier transaction. |
| Runic weapons | The engine exposes explicit grant, strike, and removal operations; consult `effects.json` and the tool descriptions for their deterministic conditions. |

The machine-readable contracts are in [effects.json](effects.json),
[subsystems.json](subsystems.json), and [weapon-effects.json](weapon-effects.json).
Fictional preconditions and many power consequences still require a narrator ruling.
The retained [dark-pacts](../docs/srd/dark-pacts.md) and
[experience](../docs/srd/experience.md) references let readers compare the data
with the SRD source.
