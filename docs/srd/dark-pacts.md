# SRD Dark Pacts rules (retained artifact)

Source: Black Sword Hack — Ultimate Chaos Edition SRD, "Dark pacts & other
vileness" page, https://blackswordhack.github.io/6darkpact.html. Fetched
August 5, 2026. The SRD publishes under Creative Commons Attribution 4.0, so
this project reproduces the mechanical rules with attribution to the source URL.

This file retains the six supernatural subsystems that `rules/subsystems.json`
and `rules/weapon-effects.json` transcribe, so a later reader can verify the
encoded figures against the source without a network fetch. The SRD site holds
the authoritative wording; this file records the mechanical facts the fetch
returned on the date above. The August 5, 2026 fetch reached the page through a
markdown-conversion reader, so this artifact records rules data — names, dice,
table rows, and numbers — rather than the source's verbatim prose. Counts that
the reader first returned inconsistently were reconciled by a second enumerated
fetch on the same date: 13 demons, 9 spirits, 12 faerie ties, 15 marvels, and
the d100 spell list.

## Common invocation pattern

Demonic pacts and spirit alliances both invoke through the Doom die. The caller
rolls the Doom die, with Disadvantage when invoking the same power more than once
in a day; the day resets after a long rest. A rolled 1 triggers the power's own
side effect, which each entry records. When the Doom die depletes during a
demonic invocation, the caller rolls on the Demon's Revenge table. Sorcery
invokes through an INT test instead of the Doom die, and a critical failure sends
the caster to the Torn Veil table. Faerie ties, twisted science, and runic
weapons each carry their own trigger, recorded in their sections.

## Demonic Pacts

A Warlock invokes a demon by name and rolls the Doom die. Disadvantage applies
when invoking the same demon more than once per day. A rolled 1 triggers that
demon's side effect against the caster. When the Doom die depletes during the
invocation, the caller rolls on the Demon's Revenge table. Forming a new pact is
the result of a dangerous adventure or quest rather than a priced transaction;
the SRD provides a d20 table of locations where a demon may be found.

The 13 demons and their effects, with the roll-of-1 side effect:

| Demon | Effect (side effect on a rolled 1) |
|---|---|
| Abyss | Target becomes monstrous or disfigured for d6 hours; permanent on a 1. |
| Envy | Target tries to seize a chosen object by force; the caster's own equipment becomes the target on a 1. |
| Fear | The caster learns the target's deepest fear; the effect is mutual on a 1. |
| Greed | Creates 4d6 fake coins lasting one hour; steals 4d6 of the caster's real coins on a 1. |
| Hate | Target verbally abuses a chosen person for d6 minutes; the caster is abused on a 1. |
| Isolation | Target is unseen and unheard for d6 hours; the caster cannot see the target on a 1. |
| Gluttony | Target attacks randomly chosen people and consumes 2d6 HP worth; it wants to eat the caster on a 1. |
| Nightmare | Target loses sleep, granting Advantage on the next day's tests against them; Disadvantage on a 1. |
| Oblivion | Target is forgotten for d6 hours; the demon steals one Background on a 1, recoverable after a long rest. |
| Ruin | Breaks equipment up to cart size; the caster's own gear breaks on a 1. |
| Sloth | Target falls asleep; never wakes on a 1. |
| Suspicion | Target suspects a chosen person; the caster becomes the suspected one on a 1. |
| Wrath | Target goes berserk and attacks everyone; the caster is affected on a 1. |

### Demon's Revenge table (d6)

| Roll | Outcome |
|---|---|
| 1 | Cannot invoke this demon until the next sunrise. |
| 2 | The demon steals one possession. |
| 3 | The demon destroys an ally's weapon. |
| 4 | The demon takes payment in blood: lose d6 HP. |
| 5 | Broken pact: this demon can never be invoked again. |
| 6 | The demon manifests, inflicts 3d6 damage, and the pact breaks. |

## Spirit Alliances

A Shaman summons a bound spirit by rolling the Doom die, with Disadvantage when
it is not the first summon of the day; the day resets after a long rest. The SRD
records no Doom-depletion table for spirits, unlike demons and sorcery. Each
spirit records its own effect and its roll-of-1 consequence. A shaman may
sacrifice an alliance for one large effect — a firestorm, a hurricane, or a beast
lord summoned in the flesh — after which spirits of that type shun the shaman and
deny further alliances. Binding a new spirit always carries a price; the SRD
provides a d20 table of prices, from permanently losing 1 HP through paying
d6×100 coins to hunting and killing a corrupted shaman.

The 9 spirits and their effects, with the roll-of-1 consequence:

| Spirit | Effect (consequence on a rolled 1) |
|---|---|
| Ancestor spirit | Grants Advantage on one attribute test, requiring an object belonging to the ancestor; gives Disadvantage instead on a 1. |
| Animal lord spirit | Summons animal subjects to help or to cease attacking; the animals flee or attack on a 1. |
| Disease spirit | Renders a target bedridden with a non-lethal disease until cured; the disease becomes contagious on a 1. |
| Fire spirit | Manipulates an existing fire — double its size, shape it, move it — and inflicts d6 damage plus d6 ongoing; a rolled 20 burns the caster and a 1 extinguishes the fire. |
| Forest spirit | Finds a way, food, water, or shelter in the wilderness, and summons plants granting Advantage against enemies for one turn; denies rest for two days on a 1. |
| Hunger spirit | Drives a target to search for food or water until fed a full meal; affects the shaman too on a 1. |
| Pain spirit | Prevents a target from acting for one Turn; afflicts the shaman at the same time on a 1. |
| River spirit | Manipulates present water — double it, shape it, move it; the water disappears on a 1. |
| Wind spirit | Extinguishes flames, dissipates fumes, or forces humanoids down; the effect reverses on a 1. |

## Sorcery

A sorcerer casts a spell by making an INT test. Disadvantage applies when the
spell was already cast that day. A failure prevents casting until a long rest. A
critical failure sends the caster to the Torn Veil table. A character with the
Forbidden Knowledge background begins play with four spells rolled on the d100
table. The maximum number of spells a character can know equals their INT score.
Learning a new spell requires a formula, a mystical item bond, or a mentor, then
an INT roll; a failure means finding another version of the spell, and a critical
failure means never learning that specific spell.

### Torn Veil table (d6)

| Roll | Outcome |
|---|---|
| 1 | Cannot cast spells until the next sunrise. |
| 2 | Energy ravages the body: lose d6 HP. |
| 3 | Arcane explosion: d6 damage to all Nearby. |
| 4 | The body pays the price: permanently lose 1 HP. |
| 5 | The mind shatters: permanently lose 1 INT. |
| 6 | A tentacled monstrosity appears and the caster is transported to another plane. |

### Spell list (d100)

Each range is the die result that yields the spell, so a Forbidden Knowledge
character rolls d100 four times and reads the spell whose range contains each
result.

| d100 | Spell | Effect |
|---|---|---|
| 1–2 | Acid blood | Convert 3 HP to acid: d6 damage, or dissolves a small item. |
| 3–5 | Animate mirror | A reflection attacks from a mirror for d6 damage. |
| 6–7 | Blood mark | Mark a possession and always know its location; costs 1 HP permanently. |
| 8–10 | Call the Id | Summon an invisible anger creature to remove an obstacle or deal 2d6 damage. |
| 11–12 | Curse of the mute | Target cannot speak; permanent on a 1. |
| 13–14 | Darkness | d6 targets are blinded for d6 minutes. |
| 15–17 | Dead man's map | The blood of a corpse reveals the murderer's location. |
| 18–20 | Deafening scream | Paralyze all who hear it; children and animals die. |
| 21–22 | Demon's breath | Extinguish light sources and start a fire elsewhere. |
| 23–24 | Dream guardian | Animate an object to guard the caster while sleeping. |
| 25–27 | Dream message | Send a message through dreams with no line of sight. |
| 28–29 | Fading memories | Target forgets d6 hours of interaction, or years on a 1. |
| 30–31 | Feather crash | Survive a deadly fall; all equipment is destroyed. |
| 32–34 | Feeding the fire | An existing flame bursts for d6 damage Nearby. |
| 35–36 | Fireflies | Summon a torch-light swarm that disappears if violence occurs. |
| 37–38 | Fleabag | Transform into a dog until sunset or sunrise, keeping HP and abilities, biting for unarmed damage. |
| 39–41 | Ghost pains | The victim feels a loss and believes the caster can restore it. |
| 42–43 | Gloomy lullaby | Target loses consciousness; permanent on a 1. |
| 44–46 | Greedy hand | An item the target holds flies to the caster; it is destroyed on a casting roll of 1. |
| 47–48 | Guiding rat | Underground, summon a rat to the nearest exit; costs d4 HP in blood. |
| 49–51 | Hellhound | Transform a dog into a killer: Attack 11, Dodge 11, d6 damage, 10 HP; it dies after the fight. |
| 52–53 | Impotent arrows | Projectiles do no damage next turn; allies' projectiles too on a 1. |
| 54–56 | Inquisition | A restrained target endures pain and answers d4 questions; it dies if asked more. |
| 57–58 | Iron ghost | Create an invisible weapon, invisible until used; costs d6 HP if it does not injure. |
| 59–61 | Murmurs | Target hears voices revealing their darkest secrets. |
| 62–63 | Never-ending music | Target hears a tune, granting Advantage on tests against them. |
| 64–66 | Poisonous projectile | Make a projectile lethal; the victim dies in d6 minutes. |
| 67–68 | Portal | Create a door to a known location; arrive naked, and followers roll d6, disappearing on a 6. |
| 69–70 | Red trap | A blood pool costs 2 HP and stops movement. |
| 71–72 | Rotten fumes | Advantage against Nearby targets for d6 minutes. |
| 73–75 | Serpent bones | Boneless form for d6 minutes to escape bonds or squeeze through gaps. |
| 76–77 | Sharing the pain | Transfer wound HP loss to a companion. |
| 78–79 | Soundkiller | Muffle all nearby sounds for 2d6 minutes. |
| 80–82 | Soul-eater | Chomp a target's soul: unconsciousness, or death on a 1. |
| 83–85 | Spontaneous combustion | Target bursts into flames for d4 continuous damage; all burn on a 1. |
| 86–87 | Steal life | Take d6 HP from a touched target, not exceeding the caster's maximum. |
| 88–90 | Tongue thief | Speak six words through a target's mouth. |
| 91–92 | Unnatural speed | Target moves a Far distance per turn; random equipment is destroyed on a 1. |
| 93–95 | War drums | Target experiences war horrors; d6: 1–4 panic, 5–6 berserk. |
| 96–97 | Wine of death | Drinkers seek fights and kill each other on a 1. |
| 98–100 | Withering | Target has the strength and vitality of a 90-year-old for d6 hours. |

## Faerie Ties

A character with faerie ties calls on a bound tie for its effect. The SRD records
no shared Doom-die pattern for this section; two ties note their own Doom roll,
recorded below. New ties form at the GM's discretion during play, with no formal
mechanic.

The 12 faerie ties and their effects:

| Faerie tie | Effect |
|---|---|
| Barrow wisdom | Speak with the dead; a WIS test compels an answer, decreasing the Doom die. |
| Cauldron of gold | A magic coin worth 100 returns after 1d8 dawns. |
| Changeling knowledge | Advantage on tests in two chosen areas of expertise. |
| Cold iron weapon | A legendary blade dealing +d6 damage against faerie. |
| Doomed to greatness | Once per day, roll the Doom die with Advantage; recovery requires 1d3 long rests. |
| Dwarf deceit | Advantage when lying or sneaking to cause harm. |
| Elfin secret | One of six options — speak with birds, command plants, invisibility, enthral, divine desire, frighten mortals — with a Doom roll if used more than once per day. |
| Silversmith sorcerer | Bind two rolled spells to jewelry, costing 1d6×100 coins and 3d6 days of work; the wearer casts them as a sorcerer. |
| Skinwalker | Transform into an animal through a prepared pelt; own at most two pelts. |
| Trollish ruggedness | +1 armour protection. |
| True faith | Roll the Doom die to dispel one fey magical effect. |
| Witchsight | See fey beings that are always visible; a WIS roll detects magical auras and illusions. |

## Twisted Science

An Inventor builds marvels in a well-equipped workshop. Invention points equal
the INT score per week. Each marvel's bracketed number is its cost in invention
points. Materials cost 20 coins per invention point required. Maintaining a
marvel subtracts half its cost from the weekly points and needs no workshop.
Non-single-use marvels carry a d6 Usage Die (Ud6), rolled after each use or
fight. The SRD records no malfunction table; single-use marvels and critical
failures note their own effects.

The 15 marvels and their invention-point cost:

| Marvel | Cost | Effect |
|---|---|---|
| Acid spray | 2 | An enemy weapon deals −2 damage. |
| Blood-tinted spyglass | 4 | See living beings through obstacles and darkness. |
| Bomb | 4, single-use | d8 damage Nearby; a critical failure explodes near the user. |
| Firelance | 6 | A ranged fire weapon adding Ud4 ongoing damage. |
| Freezing warhammer | 6 | Freezes non-living matter; a rolled 20 deals d6 frost damage and destroys the target. |
| Gas mask | 2 | Standard protection against gas. |
| Hallucinogenic gas | 4, single-use | Indoors, roll d6 per person: 1–2 relaxed, 3–5 dreams, 6 panic. |
| Image crystal | 4 | Projects a chosen human-sized image. |
| Metal Owl | 4 | An automaton following six-word orders; 4 HP; distracts only. |
| Prosthetic limb | 10 | Functions as the original; sacrifice it to negate one attack. |
| Resurrection shot | 4, single-use | Roll d20 to avoid the Helpless table; a 20 means death. |
| Sleep box | 4 | A music box trance; a Ud6 gives the duration in minutes. |
| Targeting monocle | 4 | Ignore ranged-attack penalties. |
| Terror gas grenade | 2, single-use | Roll d6: 1–4 flee, 5–6 berserk with +d4 damage. |
| Truth serum | 4, single-use | The target speaks the truth; roll d20, and 15–20 means death before speaking. |

## Runic Weapons

Runic weapons are extremely rare — the SRD suggests no more than two in a whole
campaign world — and sentient, communicating by telepathy. The GM makes an INT
test for the weapon at the start of each session. On a success, the weapon kills
its wielder if the wielder ever becomes helpless during that session. The weapon
inflicts damage equal to one of the wielder's six attributes, fixed by the
weapon's personality: brutal (STR), vicious (DEX), patient (CON), cunning (INT),
judgemental (WIS), or prideful (CHA). The weapon carries its own INT score,
rolled the way a character's is. On killing an enemy, the weapon triggers one of
six effects, one of which grants the wielder d6 HP; the others cover soul
devouring, memory absorption, body transformation, tattoo manifestation, and
explosive conversion. The SRD provides a d20 table of locations where such a
weapon may be found.
