---
name: bsh-worldsmith
description: "Author Black Sword Hack world content in the developer profile: premises, locations with exits and secrets, factions, NPC motives, clocks, and world-file linting. Never load in the player-facing runtime."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  storyteller:
    tags: [worldbuilding, tabletop, black-sword-hack, authoring]
    related_skills: [bsh-gm]
---

# Black Sword Hack worldsmith

## Overview

You author setting files for a Black Sword Hack campaign. This skill writes files. It belongs in the trusted developer profile only.

Never load this skill in the player-facing narrator runtime. That profile has no file tools by design, and a skill that assumes them will fail confusingly or, worse, encourage a workaround.

## Scope

Author content under `world/`. Never write under `campaign/`. Campaign state belongs to the Model Context Protocol (MCP) server, and hand-edited state breaks the audit log.

## Authoring a premise

A doomed-world premise needs one sentence stating what is ending, one region, one settlement, and one immediate situation that puts a clock on the party. Write the ending as a mechanism, not a mood: something specific is failing, on a schedule, for a reason somebody could in principle address.

Write three factions aligned to Law, Chaos, and Balance. Give each a count of members, a concrete want, and a price they will pay. A faction that only has an ideology is not usable at the table.

## Authoring a location

Every location file lives at `world/locations/<id>.md`. The identifier uses lower-case letters, digits, and hyphens only, and it must match the frontmatter `id` and the file stem.

Frontmatter carries `id`, `name`, `exits`, `visible_entities`, `hidden_entities`, and `tags`. Every exit must name another location file that exists; the validator rejects a dangling exit.

The body carries seven sections in this order: public description, immediate danger, useful details, hidden truths, NPC motives, discoverable clues, and consequences.

Write the public description in sensory specifics a narrator can read aloud. Write the immediate danger as a thing that acts, or state plainly that none stands. Write discoverable clues as findings, not as tests: say what an active search turns up, and name the pressure that would justify a roll instead.

Write consequences as branches the world takes, each tied to a party action. Two or three per location is enough.

## Authoring NPCs

Give each NPC an identifier, a level between 1 and 10, an armour category, a role, a faction, a location, one motive, one or two named actions, an attitude, and a voice note of under ten words. The level drives hit points and damage through the quick NPC table; the narrator never invents those numbers.

A motive must be actionable this session. "Wants power" is not a motive. "Wants to know who is asking about the bell-keeper before the Magistrates do" is.

## Authoring names and complications

Give twelve given names, twelve family names, and twelve places in one shared idiom. Give a d12 complication table where every entry is physical and concrete, and where no entry erases a player's success.

## Linting before handoff

Run `uv run python scripts/validate_campaign.py` from the project root. It checks that every location frontmatter identifier matches its filename, that every exit resolves, that `world/locations/index.yaml` matches the files on disk, and that the rules tables load.

Fix every reported error before the content reaches a live session. Warnings are advisory.

## Boundaries

Do not copy text from any Black Sword Hack product other than the System Reference Document. Do not write mechanics that contradict `rules/quick-reference.md`. When a location needs a subsystem the minimum viable product defers, write the fiction and note the deferral rather than inventing a rule the MCP server cannot enforce.
