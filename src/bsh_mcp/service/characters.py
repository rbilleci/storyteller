"""Character creation and advancement, and quick NPC creation."""

from __future__ import annotations

from .. import effects, rules
from ..dice import DEPLETED
from ..models import (
    ARMOUR_CATEGORIES,
    NPC,
    ORIGINS,
    Character,
    PlayerLink,
    StakesDeclaration,
    UsageResource,
)
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError
from .common import guard, success_after_commit
from .scene import _clean_entries


class CharacterCreateResult(ToolEnvelopeSuccess):
    character_id: str
    sheet: dict
    attribute_rolls: dict[str, dict]
    background_features: list[str]
    spell_draws: list[dict]


class CharacterAdvanceResult(ToolEnvelopeSuccess):
    character_id: str
    from_level: int
    to_level: int
    benefits_applied: dict
    gift_feature: str
    further_advancement_available: bool
    eligible_level: int
    sheet: dict


class NpcCreateResult(ToolEnvelopeSuccess):
    npc_id: str
    npc: dict


class CharactersMixin:
    """Character creation and advancement, and quick NPC creation."""

    @guard
    def character_create(
        self,
        discord_user_id: str,
        name: str,
        origin: str,
        backgrounds: list[str],
        weapons: list[str] | None = None,
        language: str = "",
        armour: str = "none",
        shield: bool = False,
    ) -> CharacterCreateResult | ToolEnvelopeFailure:
        """Roll and persist one new player character."""
        origin_key = origin.strip().lower()
        if origin_key not in ORIGINS:
            raise CampaignError(
                "invalid_origin",
                f"{origin!r} is not an origin.",
                [f"Use one of {', '.join(ORIGINS)}."],
            )
        if armour not in ARMOUR_CATEGORIES:
            raise CampaignError(
                "invalid_armour",
                f"{armour!r} is not an armour category.",
                [f"Use one of {', '.join(ARMOUR_CATEGORIES)}."],
            )

        background_ids = [b.strip().lower() for b in backgrounds]
        violations = rules.validate_backgrounds(origin_key, background_ids, self.data)
        if violations:
            raise CampaignError(
                "invalid_backgrounds",
                "; ".join(violations),
                [
                    "Call character_options for the legal background lists.",
                    "Select three backgrounds, at least two from the chosen origin, "
                    "with at most one unique background.",
                ],
            )

        generated = rules.generate_character(self.roller, origin_key, background_ids, self.data)

        languages = list(generated.languages)
        language = language.strip()
        if language and language not in languages:
            languages.append(language)

        shape = effects.apply(
            "character_shape",
            effects.ShapeContext(languages=languages),
            self.data.effects,
            effects.background_sources(background_ids),
        )
        weapon_damage = shape.weapon_damage
        unarmed_damage = shape.unarmed_damage
        languages = shape.languages

        chosen_weapons = (
            _clean_entries(weapons)
            if weapons is not None
            else rules.roll_starting_weapons(self.roller, origin_key, self.data)
        )

        def build(character_id: str) -> Character:
            return Character(
                id=character_id,
                name=name.strip(),
                discord_user_id=str(discord_user_id),
                origin=origin_key,
                backgrounds=background_ids,
                attributes=generated.attributes,
                hp=generated.hp,
                hp_max=generated.hp,
                weapon_damage=weapon_damage,
                unarmed_damage=unarmed_damage,
                armour=armour,
                shield=shield,
                weapons=chosen_weapons,
                languages=languages,
                coins=generated.coins,
                spells=list(generated.spells),
                resources=[UsageResource(id="rations", name="Rations", die="d6")],
                notes="\n".join(generated.background_features),
            )

        with self.store.transaction(
            "character_create", actor_id="", reason=f"create {name}"
        ) as transaction:
            # Allocate the identifier under the campaign lock. Reading the
            # directory before the lock lets two processes pick the same id and
            # lets the second overwrite the first authoritative sheet.
            character_id = self._unique_id(name, set(self.store.character_ids()))
            character = build(character_id)
            transaction.actor_id = character_id
            transaction.add_character(character)
            if discord_user_id:
                players = transaction.players
                players.players = [
                    link
                    for link in players.players
                    if link.discord_user_id != str(discord_user_id)
                    and link.character_id != character_id
                ] + [
                    PlayerLink(
                        discord_user_id=str(discord_user_id),
                        character_id=character_id,
                        display_name=name.strip(),
                    )
                ]
                transaction.touch_players()
            transaction.record(f"created character {character_id}")
            for warning in generated.warnings:
                transaction.warn(warning)
            commit_detail = {
                "outcome": "created",
                "attribute_rolls": generated.attribute_rolls,
                "origin": origin_key,
                "backgrounds": background_ids,
            }
            if generated.spell_draws:
                commit_detail["spell_draws"] = generated.spell_draws
            sequence = transaction.commit(commit_detail)

        return success_after_commit(
            transaction,
            f"{character.name} joins the party: {character.hp} hit points, Doom d6.",
            sequence=sequence,
            outcome="created",
            narration_facts=[
                f"{character.name} is a {origin_key} character with "
                f"{character.hp} hit points and a d6 Doom die."
            ],
            character_id=character_id,
            sheet=character.model_dump(),
            attribute_rolls=generated.attribute_rolls,
            background_features=generated.background_features,
            spell_draws=generated.spell_draws,
        )

    # -- advancement ---------------------------------------------------------

    @guard
    def character_advance(
        self,
        character_id: str,
        attribute_increases: list[str] | None = None,
        gift_id: str = "",
    ) -> CharacterAdvanceResult | ToolEnvelopeFailure:
        """Raise one character by one level, applying that level's SRD benefits.

        The tool advances the character exactly one level per call, and only when
        their Stories reach the next level. It adds the hit point, applies the
        chosen attribute increases and Gift, and upgrades the Doom die at level 10.
        It refuses while combat is active, because a level-up changes state the
        fight is using.
        """
        gift_id = gift_id.strip().lower()
        chosen_attributes = [name.strip().upper() for name in (attribute_increases or []) if name.strip()]

        with self.store.transaction(
            "character_advance", actor_id="", reason=f"advance {character_id}"
        ) as transaction:
            character = transaction.character(character_id)
            transaction.actor_id = character.id

            if transaction.state.combat.active:
                raise CampaignError(
                    "combat_active",
                    f"{character.name} cannot advance during combat. A level-up changes "
                    "hit points, attributes, Gifts, and the Doom die the fight is using.",
                    ["Resolve the current fight, then advance between encounters."],
                )

            max_level = int(self.data.advancement["max_level"])
            if character.level >= max_level:
                raise CampaignError(
                    "already_max_level",
                    f"{character.name} is at level {character.level}, the maximum. "
                    "No further advancement exists.",
                    ["Award Stories for the record; the sheet does not change."],
                )

            target_level = character.level + 1
            reachable = rules.eligible_level(character.stories)
            if reachable < target_level:
                needed = rules.stories_required_for_level(target_level)
                raise CampaignError(
                    "not_eligible",
                    f"{character.name} has {character.stories} Stories and needs "
                    f"{needed} to reach level {target_level}.",
                    [
                        f"Award Stories with session_close until {character.name} holds "
                        f"{needed}, then advance.",
                    ],
                )

            benefits = rules.advancement_benefits(target_level, self.data)
            gift_index = self.data.gift_index()
            violations = rules.validate_advancement_choices(
                character, benefits, chosen_attributes, gift_id, gift_index
            )
            if violations:
                raise CampaignError(
                    "invalid_advancement_choices",
                    "; ".join(violations),
                    self._advancement_next_steps(benefits),
                )

            applied: list[str] = [f"level {character.level} to {target_level}"]
            character.level = target_level

            for name in chosen_attributes:
                character.attributes.adjust(name, 1)
                applied.append(f"{name} to {character.attributes.get(name)}")

            if benefits["hit_point_gain"]:
                character.hp_max += benefits["hit_point_gain"]
                character.hp += benefits["hit_point_gain"]
                applied.append(f"hit points to {character.hp}/{character.hp_max}")

            gift_feature = ""
            if benefits["grants_gift"] and gift_id:
                gift = gift_index[gift_id]
                character.gifts = character.gifts + [gift_id]
                gift_feature = gift.get("feature", "")
                applied.append(f"Gift {gift['name']}")
                subsystem = gift.get("subsystem")
                if subsystem:
                    readable = subsystem.replace("_", " ")
                    transaction.warn(
                        f"Gift {gift['name']}: {gift_feature} The narrator adjudicates this "
                        f"{readable} Gift from bsh://rules/subsystems."
                    )
                else:
                    transaction.warn(
                        f"Gift {gift['name']}: {gift_feature} The narrator applies this as a ruling."
                    )

            if benefits["doom_die"]:
                character.doom_max = benefits["doom_die"]
                if character.doom_die != DEPLETED:
                    character.doom_die = benefits["doom_die"]
                applied.append(f"Doom die to {benefits['doom_die']}")

            transaction.touch_character(character.id)
            transaction.record(f"{character.id}: advanced to level {target_level}")

            still_eligible = rules.eligible_level(character.stories) > character.level
            if still_eligible:
                transaction.warn(
                    f"{character.name} holds enough Stories to advance again; call "
                    "character_advance once more with the next level's choices."
                )

            sequence = transaction.commit(
                {
                    "outcome": "advanced",
                    "from_level": target_level - 1,
                    "to_level": target_level,
                    "attribute_increases": chosen_attributes,
                    "gift_id": gift_id if benefits["grants_gift"] else "",
                }
            )

        narration = f"{character.name} advances to level {target_level}."
        return success_after_commit(
            transaction,
            narration,
            sequence=sequence,
            outcome="advanced",
            state_changes=[f"advanced {character.id}: " + ", ".join(applied)],
            narration_facts=[narration],
            character_id=character.id,
            from_level=target_level - 1,
            to_level=target_level,
            benefits_applied=benefits,
            gift_feature=gift_feature,
            further_advancement_available=still_eligible,
            eligible_level=rules.eligible_level(character.stories),
            sheet=character.model_dump(),
        )

    @staticmethod
    def _advancement_next_steps(benefits: dict) -> list[str]:
        """Return actionable guidance naming exactly what this level requires."""
        steps: list[str] = []
        count = int(benefits["attribute_increases"])
        if count:
            noun = "attribute" if count == 1 else "distinct attributes"
            steps.append(
                f"Pass attribute_increases with {count} {noun} from "
                f"STR, DEX, CON, INT, WIS, CHA, each below {benefits['attribute_max']}."
            )
        else:
            steps.append("Leave attribute_increases empty for this level.")
        if benefits["grants_gift"]:
            steps.append("Pass gift_id with one Gift id from rules/gifts.json the character lacks.")
        else:
            steps.append("Leave gift_id empty for this level.")
        return steps

    # -- NPCs ----------------------------------------------------------------

    @guard
    def npc_create(
        self,
        name: str,
        level: int,
        armour: str = "none",
        motive: str = "",
        actions: list[str] | None = None,
        location_id: str = "",
        present_in_scene: bool = True,
    ) -> NpcCreateResult | ToolEnvelopeFailure:
        """Create one quick NPC from the level table."""
        if armour not in ARMOUR_CATEGORIES:
            raise CampaignError(
                "invalid_armour",
                f"{armour!r} is not an armour category.",
                [f"Use one of {', '.join(ARMOUR_CATEGORIES)}."],
            )
        location_id = location_id.strip()
        known_locations = set(self.store.location_ids())
        if not location_id and present_in_scene:


            current = self.store.read_state().scene.location_id
            if current in known_locations:
                location_id = current
        if location_id and known_locations and location_id not in known_locations:
            raise CampaignError(
                "location_not_found",
                f"No location file exists for id {location_id!r}.",
                [
                    f"Known locations: {', '.join(sorted(known_locations))}.",
                    "Omit location_id when the NPC is not tied to an authored location.",
                ],
            )
        hp, damage = rules.npc_stats(int(level), armour, self.data)

        with self.store.transaction("npc_create", actor_id="", reason=f"create {name}") as transaction:
            npc_id = self._unique_id(name, set(transaction.state.npcs.keys()))


            person = transaction.state.scene.persons.get(npc_id)
            npc = NPC(
                id=npc_id,
                name=name.strip(),
                level=int(level),
                hp=hp,
                hp_max=hp,
                damage=damage,
                armour=armour,
                motive=motive.strip(),
                actions=_clean_entries(actions),
                location_id=location_id,
                notes=person.role if person is not None and not motive.strip() else "",
            )
            npcs = dict(transaction.state.npcs)
            npcs[npc_id] = npc
            transaction.state.npcs = npcs
            if present_in_scene and npc_id not in transaction.state.scene.present_npcs:
                scene = transaction.state.scene
                scene.present_npcs = scene.present_npcs + [npc_id]
                transaction.state.scene = scene
                transaction.scene_dirty = True
            if person is not None:
                scene = transaction.state.scene
                scene.persons = {
                    pid: recorded for pid, recorded in scene.persons.items() if pid != npc_id
                }
                transaction.state.scene = scene
                transaction.scene_dirty = True
                transaction.record(f"promoted person {npc_id} to NPC")
            transaction.record(f"created NPC {npc_id} at level {level}")
            stake = f"{npc.name} is present at {npc.location_id or 'the scene'}"
            if npc.motive:
                stake += f" ({npc.motive})"
            transaction.add_fiction_debt(
                tool="npc_create",
                actor_id=npc_id,
                reason=transaction.reason or f"{npc.name} enters play",
                outcome="created",
                stakes=StakesDeclaration(success=stake),
                realized="success",
            )
            sequence = transaction.commit(
                {
                    "outcome": "created",
                    "npc": npc.model_dump(),
                    "promoted_from_person": person is not None,
                }
            )

        return success_after_commit(
            transaction,
            f"{npc.name} enters play: level {npc.level}, {npc.hp} hit points, damage {npc.damage}.",
            sequence=sequence,
            outcome="created",
            narration_facts=[f"{npc.name} is present. Motive: {motive or 'undeclared'}."],
            npc_id=npc_id,
            npc=npc.model_dump(),
        )
