"""Short and long rests, inventory changes, and the Helpless table."""

from __future__ import annotations

from .. import effects, rules
from ..models import Character, Condition
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError
from .common import guard, success_after_commit
from .scene import _clean_entries


class InventoryUpdateResult(ToolEnvelopeSuccess):
    character_id: str
    coins: int
    coins_delta: int
    equipment: list[str]
    weapons: list[str]
    runic_weapon_relinquished: str | None
    next_step: str


class RestResult(ToolEnvelopeSuccess):
    rest_type: str
    characters: list[dict]
    minutes: int
    in_game_time: str


class HelplessRollResult(ToolEnvelopeSuccess):
    character_id: str
    result: dict
    status: str
    hp: int
    care: dict | None


class RecoveryMixin:
    """Short and long rests, inventory changes, and the Helpless table."""

    # -- recovery ------------------------------------------------------------

    @guard
    def rest(
        self,
        character_ids: list[str],
        rest_type: str,
        safe_environment: bool = False,
        reason: str = "",
    ) -> RestResult | ToolEnvelopeFailure:
        """Apply a short or long rest to the named characters."""
        if rest_type not in ("short", "long"):
            raise CampaignError(
                "invalid_rest_type",
                f"{rest_type!r} is not a rest type.",
                ["Use short or long."],
            )
        if not character_ids:
            raise CampaignError("no_participants", "a rest needs at least one character id.")

        applied: list[dict] = []
        with self.store.transaction(
            "rest", actor_id=",".join(character_ids), reason=reason
        ) as transaction:
            day = transaction.state.day

            # rest_ctx is computed once per character here, for every long rest
            # (not only an unsafe one), so the per-character loop below can reuse
            # it for dose-style resolution without a second effects.apply pass.
            # This never inspects who grants what: it just hands every resting
            # character the ambient facts (scene tags, their own standing
            # declarations) and lets whatever their own backgrounds register
            # decide what, if anything, happens.
            rest_ctx_by_character: dict[str, effects.RestContext] = {}
            if rest_type == "long":
                environment_tags = list(transaction.state.scene.environment_tags)
                unsafe_blocked = []
                for character_id in character_ids:
                    character = transaction.character(character_id)
                    rest_ctx = effects.apply(
                        "rest",
                        effects.RestContext(
                            environment_tags=environment_tags,
                            declared_choices=character.declared_choices,
                            level=character.level,
                        ),
                        self.data.effects,
                        self._effect_sources(character),
                    )
                    rest_ctx_by_character[character.id] = rest_ctx
                    if not safe_environment and not rest_ctx.allow_unsafe_long_rest:
                        unsafe_blocked.append(character.name)
                if unsafe_blocked:
                    raise CampaignError(
                        "unsafe_environment",
                        "a long rest requires a safe environment: "
                        f"{', '.join(unsafe_blocked)} cannot rest here.",
                        [
                            "Find or create a genuinely safe place in fiction, then call rest again "
                            "with safe_environment true.",
                            "A wilderness camp is not safe without a specific fictional reason.",
                        ],
                    )

            resolved_resters: list[str] = []
            for character_id in character_ids:
                character = transaction.character(character_id)
                resolved_resters.append(character.id)
                if character.status == "dead":
                    raise CampaignError(
                        "character_dead",
                        f"{character.name} is dead and cannot rest.",
                    )
                entry = {"character_id": character.id, "name": character.name}

                if rest_type == "short":
                    if character.last_short_rest_day == day:
                        raise CampaignError(
                            "short_rest_already_used",
                            f"{character.name} already took a short rest on day {day + 1}.",
                            [
                                "Take a long rest in a safe place, or advance the in-game day "
                                "through scene_commit.",
                            ],
                        )
                    healed = rules.short_rest_healing(character.attributes.CON)
                    previous = character.hp
                    character.hp = min(character.hp_max, character.hp + healed)
                    character.last_short_rest_day = day
                    entry.update(
                        {
                            "healed": character.hp - previous,
                            "hp": character.hp,
                            "hp_max": character.hp_max,
                        }
                    )
                    transaction.record(
                        f"{character.id}: short rest, hit points {previous} -> {character.hp}"
                    )
                else:
                    previous = character.hp
                    character.hp = character.hp_max
                    character.doom_die = character.doom_max
                    character.last_short_rest_day = None
                    kept = [
                        condition
                        for condition in character.conditions
                        if condition.scope not in ("until_long_rest", "scene")
                    ]
                    cleared = len(character.conditions) - len(kept)
                    character.conditions = kept
                    self._reset_pools(character, "long_rest")
                    faerie_recovery = self._advance_faerie_recovery(character)
                    entry.update(
                        {
                            "healed": character.hp - previous,
                            "hp": character.hp,
                            "hp_max": character.hp_max,
                            "doom_die": character.doom_die,
                            "conditions_cleared": cleared,
                        }
                    )
                    if faerie_recovery:
                        entry["faerie_recovery_rests_remaining"] = faerie_recovery
                        transaction.record(
                            f"{character.id}: advanced faerie recovery after a long rest"
                        )
                    transaction.record(
                        f"{character.id}: long rest, hit points {previous} -> {character.hp}, "
                        f"Doom restored to {character.doom_max}"
                    )

                    # Class-neutral: rest() never asks which background this is.
                    # rest_ctx.granted is whatever the character's own effects
                    # decided to hand back; roll and write it generically.
                    long_rest_ctx = rest_ctx_by_character[character.id]
                    for note in long_rest_ctx.notes:
                        transaction.warn(note)
                    prepared: list[dict] = []
                    for grant in long_rest_ctx.granted:
                        rolled = self.roller.notation(grant["quantity_die"])
                        character.doses = {**character.doses, grant["dose_type"]: rolled}
                        character.declared_choices = {
                            k: v for k, v in character.declared_choices.items()
                            if k != grant["effect_id"]
                        }
                        transaction.record(
                            f"{character.id}: prepared {rolled} dose(s) of {grant['dose_type']}"
                        )
                        prepared.append({"dose_type": grant["dose_type"], "count": rolled})
                    if prepared:
                        entry["doses_prepared"] = prepared

                if character.status == "helpless" and character.hp > 0:
                    character.status = "ok"
                    transaction.record(f"{character.id}: no longer Helpless")

                transaction.touch_character(character.id)
                applied.append(entry)

            transaction.actor_id = ",".join(resolved_resters)
            minutes = 60 if rest_type == "short" else 360
            transaction.state.in_game_minutes += minutes
            transaction.add_fiction_debt(
                tool="rest",
                actor_id=transaction.actor_id,
                reason=transaction.reason,
                outcome=f"{rest_type}_rest",
            )
            sequence = transaction.commit(
                {
                    "outcome": f"{rest_type}_rest",
                    "safe_environment": safe_environment,
                    "characters": applied,
                    "minutes": minutes,
                }
            )

        return success_after_commit(
            transaction,
            f"The party takes a {rest_type} rest. {minutes} minutes pass.",
            sequence=sequence,
            outcome=f"{rest_type}_rest",
            narration_facts=[
                f"{entry['name']} recovers {entry.get('healed', 0)} hit points "
                f"({entry['hp']}/{entry['hp_max']})."
                for entry in applied
            ]
            + [
                f"{entry['name']} prepares {prep['count']} dose(s) of "
                f"{prep['dose_type'].replace('_', ' ')}."
                for entry in applied
                for prep in entry.get("doses_prepared", [])
            ],
            rest_type=rest_type,
            characters=applied,
            minutes=minutes,
            in_game_time=self._time_string(transaction.state.in_game_minutes),
        )

    @guard
    def inventory_update(
        self,
        character_id: str,
        reason: str,
        coins_delta: int = 0,
        add_equipment: list[str] | None = None,
        remove_equipment: list[str] | None = None,
        add_weapons: list[str] | None = None,
        remove_weapons: list[str] | None = None,
    ) -> InventoryUpdateResult | ToolEnvelopeFailure:
        """Apply one audited change to a character's coins, equipment, or weapons.

        Deliberately narrow: a signed coin delta and exact add/remove lists, refused
        before any write when the result would be impossible (negative coins, removing
        something not held). It never rolls dice, never touches hit points or Doom,
        and never substitutes for a negotiated merchant purchase -- the trade
        confirmation flow owns those, and its authenticated confirmation is the
        boundary a prompt-injected \"purchase\" must not step around.

        A runic weapon (``grant_runic_weapon``) is held in its own typed field, not in
        ``weapons``, and this is the one path that lets it go: naming it in
        ``remove_weapons`` clears the field and its per-session kill verdict with it.
        Nothing else ever did, so a player told the blade was armed to kill them that
        session had no way to put it down.
        """
        reason = str(reason or "").strip()
        if not reason:
            raise CampaignError(
                "empty_inventory_reason",
                "an inventory change needs a reason; the audit event is the record of "
                "why possessions moved.",
                ["State in a few words what happened (looted Rade's pouch, dropped "
                 "the rope, paid the ferryman), then call again."],
            )
        add_equipment = _clean_entries(add_equipment)
        remove_equipment = _clean_entries(remove_equipment)
        add_weapons = _clean_entries(add_weapons)
        remove_weapons = _clean_entries(remove_weapons)
        coins_delta = int(coins_delta)
        if not (coins_delta or add_equipment or remove_equipment or add_weapons or remove_weapons):
            raise CampaignError(
                "empty_inventory_update",
                "no change was requested: coins_delta is 0 and every list is empty.",
                ["Pass the actual change, or skip the call for a no-op."],
            )

        def _take(held: list[str], wanted: str, kind: str) -> list[str]:
            for index, entry in enumerate(held):
                if entry.strip().casefold() == wanted.strip().casefold():
                    return held[:index] + held[index + 1:]
            raise CampaignError(
                "item_not_held",
                f"{wanted!r} is not among the character's {kind}: "
                f"{', '.join(held) or '(none)'}. Nothing was changed.",
                [f"Remove an exact held {kind[:-1]} name, or add it first."],
            )

        def _names_runic(character: Character, wanted: str) -> bool:
            weapon = character.runic_weapon
            return (
                weapon is not None
                and weapon.name.strip().casefold() == wanted.strip().casefold()
            )

        with self.store.transaction(
            "inventory_update", actor_id=character_id, reason=reason
        ) as transaction:
            character = transaction.character(character_id)
            transaction.actor_id = character.id
            coins_before = character.coins
            if coins_before + coins_delta < 0:
                raise CampaignError(
                    "insufficient_coins",
                    f"{character.name} holds {coins_before} coin(s); a change of "
                    f"{coins_delta} would go below zero. Nothing was changed.",
                    ["Charge at most what the character holds."],
                )
            character.coins = coins_before + coins_delta
            equipment = list(character.equipment)
            weapons = list(character.weapons)
            # A runic weapon is held in ``runic_weapon`` (a typed record carrying its
            # own INT and per-session verdict), never as an entry in ``weapons``, so
            # naming it here is the one way a character ever puts one down. The
            # verdict leaves with it: ``_become_helpless`` reads the field, not the
            # name. Dropping it is a possession change like any other; a weapon
            # that argues about being dropped is the narrator's fiction, not a
            # refusal this tool issues.
            relinquished: str | None = None
            for entry in remove_equipment:
                equipment = _take(equipment, entry, "equipment")
            for entry in remove_weapons:
                if _names_runic(character, entry):
                    relinquished = character.runic_weapon.name
                    character.runic_weapon = None
                    # Also clear a stray list entry under the same name, if the
                    # narrator ever added one by hand; its absence is no error.
                    weapons = [
                        held for held in weapons
                        if held.strip().casefold() != entry.strip().casefold()
                    ]
                    continue
                try:
                    weapons = _take(weapons, entry, "weapons")
                except CampaignError as error:
                    if character.runic_weapon is None:
                        raise
                    raise CampaignError(
                        "item_not_held",
                        f"{entry!r} is not among the character's weapons: "
                        f"{', '.join(weapons) or '(none)'}; the runic weapon "
                        f"{character.runic_weapon.name!r} is held separately. "
                        "Nothing was changed.",
                        [
                            "Remove an exact held weapon name, or add it first.",
                            f"Name {character.runic_weapon.name!r} exactly to relinquish "
                            "the runic weapon.",
                        ],
                    ) from error
            equipment.extend(add_equipment)
            weapons.extend(add_weapons)
            character.equipment = equipment
            character.weapons = weapons
            transaction.touch_character(character.id)
            if relinquished is not None:
                transaction.record(
                    f"{character.id}: relinquished runic weapon {relinquished}"
                )
            if coins_delta:
                transaction.record(
                    f"{character.id}: coins {coins_before} -> {character.coins} "
                    f"({coins_delta:+d})"
                )
            for entry in add_equipment + add_weapons:
                transaction.record(f"{character.id}: gained {entry}")
            for entry in remove_equipment + remove_weapons:
                transaction.record(f"{character.id}: lost {entry}")
            sequence = transaction.commit(
                {
                    "outcome": "inventory_updated",
                    "coins_delta": coins_delta,
                    "coins": character.coins,
                    "added": add_equipment + add_weapons,
                    "removed": remove_equipment + remove_weapons,
                    "runic_weapon_relinquished": relinquished,
                }
            )

        gained = ", ".join(add_equipment + add_weapons)
        lost = ", ".join(
            remove_equipment
            + [
                f"{relinquished} (runic weapon)"
                if relinquished is not None
                and entry.strip().casefold() == relinquished.strip().casefold()
                else entry
                for entry in remove_weapons
            ]
        )
        pieces = []
        if coins_delta:
            pieces.append(f"{character.name} now holds {character.coins} coin(s)")
        if gained:
            pieces.append(f"gains {gained}")
        if lost:
            pieces.append(f"loses {lost}")
        summary = "; ".join(pieces) + "."
        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            outcome="inventory_updated",
            narration_facts=[summary],
            character_id=character.id,
            coins=character.coins,
            coins_delta=coins_delta,
            equipment=list(character.equipment),
            weapons=list(character.weapons),
            runic_weapon_relinquished=relinquished,
            next_step=(
                "State the change in the fiction. Answer any inventory question from "
                "this result or character_sheet, never from memory."
            ),
        )

    @guard
    def helpless_roll(
        self, character_id: str, reason: str = "", carer_id: str = ""
    ) -> HelplessRollResult | ToolEnvelopeFailure:
        """Roll the Helpless table for a player character at 0 hit points.

        Pass carer_id when an ally with a tending effect (a Surgeon) treats the downed
        character first: the carer's gating test resolves in the same transaction and,
        on success, the table rolls a smaller die (a Surgeon's success rolls a d4).
        """
        with self.store.transaction(
            "helpless_roll", actor_id=character_id, reason=reason
        ) as transaction:
            character = transaction.character(character_id)
            transaction.actor_id = character.id
            if character.status != "helpless":
                raise CampaignError(
                    "character_not_helpless",
                    f"{character.name} is not Helpless.",
                    ["Only a player character reduced to 0 hit points rolls on this table."],
                )

            sides = 6
            care = None
            if carer_id:
                carer = transaction.character(carer_id)
                if carer.id == character.id:
                    raise CampaignError(
                        "invalid_carer",
                        f"{character.name} cannot tend themselves.",
                        ["Name a different character as carer_id, or omit it."],
                    )
                self._require_actable(carer)
                care_ctx = effects.apply(
                    "helpless_care",
                    effects.HelplessCareContext(),
                    self.data.effects,
                    self._effect_sources(carer),
                )
                if not care_ctx.test_attribute:
                    raise CampaignError(
                        "carer_cannot_help",
                        f"{carer.name} holds no effect that changes a Helpless roll.",
                        ["Omit carer_id, or name a character with a tending background."],
                    )
                care_payload = self._resolve_character_test(
                    transaction,
                    carer,
                    care_ctx.test_attribute,
                    advantage=False,
                    disadvantage=False,
                    opponent_level=None,
                    call_on_doom=False,
                    trigger="critical failure tending a helpless ally",
                )
                care_test = care_payload["test"]
                if care_test.succeeded:
                    sides = care_ctx.die_sides_on_success
                care = {
                    "carer_id": carer.id,
                    "attribute": care_ctx.test_attribute,
                    "roll": care_test.roll.as_dict(),
                    "outcome": care_test.outcome,
                    "die_sides": sides,
                }
                transaction.touch_character(carer.id)
                transaction.record(
                    f"{carer.id}: tends {character.id}, {care_ctx.test_attribute} test "
                    f"{care_test.outcome}, Helpless die d{sides}"
                )

            helpless_ctx = effects.apply(
                "helpless",
                effects.HelplessContext(level=character.level),
                self.data.effects,
                self._effect_sources(character),
            )
            # The Helpless table runs high-is-worse (6 kills, 1 scars), so Advantage
            # keeps the lower of two dice.
            roll = rules.roll_backlash(self.roller, sides, advantage=helpless_ctx.advantage)
            face = roll.selected
            entry = self.data.helpless_result(face)
            effect = entry["effect"]
            details: dict = {
                "face": face, "name": entry["name"], "text": entry["text"], "roll": roll.as_dict(),
            }
            recovered = 0

            if effect == "death":
                character.status = "dead"
                character.hp = 0
                transaction.record(f"{character.id}: killed by the Helpless table")
            else:
                recovered = self.roller.notation(
                    self.data.helpless_table.get("survivor_hp_die", "d4")
                )
                character.hp = min(character.hp_max, max(1, recovered) + helpless_ctx.hp_bonus)
                character.status = "ok"
                details["recovered_hp"] = character.hp
                transaction.record(
                    f"{character.id}: survives Helpless with {character.hp} hit points"
                )

                if effect == "scar":
                    scar = f"scar from session {transaction.state.session}"
                    character.scars = character.scars + [scar]
                elif effect == "destroy_equipment":
                    pool = list(character.weapons) + list(character.equipment)
                    if pool:
                        lost = self.roller.choice(pool)
                        if lost in character.weapons:
                            character.weapons = [w for w in character.weapons if w != lost]
                        else:
                            character.equipment = [e for e in character.equipment if e != lost]
                        details["destroyed"] = lost
                        transaction.record(f"{character.id}: lost {lost}")
                    else:
                        details["destroyed"] = None
                        transaction.warn("The character carried nothing left to destroy.")
                elif effect == "condition":
                    condition = Condition(**entry["condition"])
                    if not any(existing.id == condition.id for existing in character.conditions):
                        character.conditions = character.conditions + [condition]
                    details["condition"] = condition.model_dump()
                    transaction.record(f"{character.id}: gained condition {condition.label}")
                elif effect == "permanent_attribute_loss":
                    sub = entry["sub_roll"]
                    sub_face = self.roller.notation(sub["die"])
                    attribute = next(
                        band["attribute"]
                        for band in sub["ranges"]
                        if band["min"] <= sub_face <= band["max"]
                    )
                    character.attributes.adjust(attribute, -1)
                    details["sub_roll"] = {"die": sub["die"], "result": sub_face}
                    details["attribute_lost"] = attribute
                    if attribute == "CON":
                        character.hp_max = max(1, character.attributes.CON)
                        character.hp = min(character.hp, character.hp_max)
                    transaction.record(
                        f"{character.id}: permanently lost 1 point of {attribute}"
                    )

            transaction.touch_character(character.id)
            transaction.add_fiction_debt(
                tool="helpless_roll",
                actor_id=character.id,
                reason=transaction.reason,
                outcome=entry["name"].lower(),
            )
            roll_dict = {
                "notation": f"1d{sides}", "dice": [face], "selected": face, "total": face,
            }
            commit_payload = {
                "roll": roll_dict,
                "outcome": entry["name"].lower(),
                "helpless": details,
            }
            if care is not None:
                commit_payload["care"] = care
            sequence = transaction.commit(commit_payload)

        summary = f"{character.name}: {entry['name']}. {entry['text']}"
        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=roll_dict,
            outcome=entry["name"].lower(),
            narration_facts=[summary],
            character_id=character_id,
            result=details,
            status=character.status,
            hp=character.hp,
            care=care,
        )
