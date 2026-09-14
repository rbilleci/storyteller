# SRD Experience rules (retained artifact)

Source: Black Sword Hack — Ultimate Chaos Edition SRD, Experience page,
https://blackswordhack.github.io/5experience.html. Fetched August 5, 2026.

This file retains the advancement rules that `rules/advancement.json` and
`rules/gifts.json` transcribe, so a later reader can verify the encoded figures
against the source without a network fetch. The SRD site holds the authoritative
wording; this file records what the fetch returned on the date above.

## Experience and levelling

A character who survives an adventure gains one Story: the player records the
adventure's title on the sheet. A character advances one level once their Stories
equal their present level. The maximum level is 10.

## Levels and benefits

| Level | Benefit |
|---|---|
| 2 | +1 to an attribute (maximum score is 18), +1 HP |
| 3 | Gain a Gift, +1 HP |
| 4 | +1 to two attributes (maximum score is 18), +1 HP |
| 5 | Gain a Gift, +1 HP |
| 6 | +1 to an attribute (maximum score is 18), +1 HP |
| 7 | Gain a Gift, +1 HP |
| 8 | +1 to two attributes (maximum score is 18), +1 HP |
| 9 | Gain a Gift, +1 HP |
| 10 | The Doom die becomes a d8 |

Two facts govern the engine. The attribute maximum reachable by advancement is 18,
below the general attribute ceiling of 20 the character model permits. Level 10
grants the Doom die upgrade and no hit point, so a level-10 advancement adds 0 HP.

## Gifts

A character gains a Gift at levels 3, 5, 7, and 9, chosen from the Balance, Chaos,
or Law categories. Each Gift can be taken only once. `rules/gifts.json` records all
15 Gifts, 5 per category, each carrying an `automation` field: `effect` when the
effect registry (`rules/effects.json`) drives the Gift's mechanics, `manual` when
the server stores the Gift and the narrator applies its effect as a ruling.
