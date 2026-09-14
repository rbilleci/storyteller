"""Subsystem powers (demonic pacts, faerie ties, twisted-science marvels), the
registry-effect activation engine (``use_ability``), and runic weapons.
"""

from __future__ import annotations

from .. import effects, rules
from ..dice import DEPLETED
from ..models import Character, Condition, PendingRuling, RunicWeapon
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError, Transaction
from .common import guard, success_after_commit


class UseAbilityResult(ToolEnvelopeSuccess):
    ability_id: str
    details: dict


class AbilityApplyRulingResult(ToolEnvelopeSuccess):
    details: dict


class GrantRunicWeaponResult(ToolEnvelopeSuccess):
    character_id: str
    details: dict


class AbilitiesMixin:
    """Subsystem powers, registry-effect activation, and runic weapons."""

    def _intent_choices(self, character: Character, entry: dict) -> list[str]:
        """Return the legal choice values for an ``intent`` ability's own vocabulary.

        ``choices_from`` names a source to draw the vocabulary from live state
        (a character's held weapons, or the subsystems.json spirit list) instead
        of a static ``intent_types`` list; both ``use_ability`` and the sheet's
        ``abilities_block`` read it through here so the two never diverge.
        """
        choices_from = entry.get("choices_from")
        if choices_from == "weapons":
            return list(character.weapons)
        if choices_from == "spirit_alliances":
            return [s["id"] for s in self.data.subsystem("spirit_alliances")["spirits"]]
        return list(entry.get("intent_types", []))

    def _subsystem_capacity(self, character: Character, subsystem_key: str) -> int:
        """Total power slots a character's backgrounds grant for one subsystem."""
        index = self.data.background_index()
        cap = 0
        for bid in character.backgrounds:
            record = index.get(bid, {})
            if record.get("subsystem") == subsystem_key:
                cap += int(record.get("subsystem_slots", 0))
        return cap

    def _invoke_power(self, transaction, character: Character, power_id: str, power: dict,
                      target_id: str, mode: str) -> tuple[str, dict]:
        """Bind, invoke, or use one subsystem power. Returns (message, details).

        Each subsystem carries its own trigger, so this method dispatches on
        ``subsystem_key``. Demonic pacts and spirit alliances roll the Doom die on
        invoke, and a demonic invocation that depletes the die rolls Demon's Revenge.
        Faerie ties roll no Doom die by default; only the ties the SRD marks with a
        ``doom_trigger`` roll or step it. Twisted-science marvels branch entirely to
        ``_marvel_action``: binding builds a marvel and invoking spends its Usage Die.
        Every power's own effect text stays a narrator ruling.
        """
        subsystem_key = power["subsystem"]
        entry = power["entry"]
        name = entry["name"]
        details: dict = {"power_id": power_id, "subsystem": subsystem_key, "mode": mode}

        if subsystem_key == "twisted_science":
            return self._marvel_action(
                transaction, character, power_id, entry, name, mode, details
            )

        if mode == "bind":
            cap = self._subsystem_capacity(character, subsystem_key)
            readable = subsystem_key.replace("_", " ")
            if cap <= 0:
                raise CampaignError(
                    "no_subsystem_access",
                    f"{character.name} has no {readable} and cannot bind {name}.",
                    ["Only a background granting this subsystem binds its powers."],
                )
            held = [p for p in character.powers if p.startswith(power["prefix"] + ":")]
            if power_id not in character.powers:
                if len(held) >= cap:
                    raise CampaignError(
                        "subsystem_full",
                        f"{character.name} already holds {len(held)} of {cap} "
                        f"{subsystem_key.split('_')[-1]}.",
                        ["Release one before binding another (mode 'release')."],
                    )
                character.powers = character.powers + [power_id]
                transaction.touch_character(character.id)
                transaction.record(f"{character.id}: bound {power_id}")
            details["bound"] = True
            return f"{character.name} binds {name}.", details

        if mode == "release":
            character.powers = [p for p in character.powers if p != power_id]
            transaction.touch_character(character.id)
            transaction.record(f"{character.id}: released {power_id}")
            details["bound"] = False
            return f"{character.name} releases {name}.", details

        # invoke (default)
        if power_id not in character.powers:
            raise CampaignError(
                "power_not_bound",
                f"{character.name} has not bound {name}.",
                ["Bind it first with mode 'bind'."],
            )
        if subsystem_key == "faerie_ties":
            return self._invoke_faerie(
                transaction, character, power_id, entry, name, target_id, details
            )
        if character.doom_die == DEPLETED:
            raise CampaignError(
                "doom_depleted",
                f"{character.name}'s Doom die is spent; the invocation cannot roll.",
                ["Restore the Doom die on a long rest."],
            )
        invoke_ctx = effects.apply(
            "invoke",
            effects.InvokeContext(
                power_id=entry.get("id", ""), declared_choices=character.declared_choices,
            ),
            self.data.effects,
            self._effect_sources(character),
            active_ids=effects.active_ids(character),
        )
        doom = rules.roll_doom(self.roller, character.doom_die, advantage=invoke_ctx.advantage)
        character.doom_die = doom.current_die
        result = {
            "power": name,
            "effect": entry["effect"],
            "doom": doom.as_dict(),
            "target_id": target_id,
            "side_effect": doom.roll.selected == 1,
        }
        if subsystem_key == "demonic_pacts" and doom.depleted:
            backlash_ctx = effects.apply(
                "backlash",
                effects.BacklashContext(table="demon_revenge"),
                self.data.effects,
                self._effect_sources(character),
                active_ids=effects.active_ids(character),
            )
            face_roll = rules.roll_backlash(self.roller, 6, advantage=backlash_ctx.advantage)
            face = face_roll.selected
            revenge = {
                "face": face, "text": self.data.demon_revenge(face), "roll": face_roll.as_dict(),
            }
            if face == 4:
                loss = self.roller.notation("d6")
                character.hp = max(0, character.hp - loss)
                revenge["hp_lost"] = loss
            elif face == 6:
                loss = sum(self.roller.pool(3, 6))
                character.hp = max(0, character.hp - loss)
                character.powers = [p for p in character.powers if p != power_id]
                revenge["hp_lost"] = loss
                revenge["pact_broken"] = True
            elif face == 5:
                character.powers = [p for p in character.powers if p != power_id]
                revenge["pact_broken"] = True
            elif face == 2:
                # "Steals one possession" is a fictional choice, so it becomes a
                # pending ruling the adjudicate step resolves from the narration.
                possessions = list(character.weapons) + list(character.equipment)
                if possessions:
                    ruling_id = f"ruling-{character.id}-{transaction.state.event_seq + 1}"
                    transaction.state.pending_rulings = transaction.state.pending_rulings + [
                        PendingRuling(
                            id=ruling_id, actor_id=character.id, kind="demon_revenge_steal",
                            question=f"Which of {character.name}'s possessions does {name} steal?",
                            options=possessions, default_strategy="random",
                        )
                    ]
                    revenge["ruling_id"] = ruling_id
                    details["adjudication_needed"] = True
            elif face == 3:
                # "Destroys an ally's weapon" is a cross-character fictional choice,
                # so each option carries its owner: "<ally-id>: <weapon>". Identifier
                # forbids ':' in a character id, so partition(": ") splits it back.
                options: list[str] = []
                for ally_id in self.store.character_ids():
                    if ally_id == character.id:
                        continue
                    ally = transaction.character(ally_id)
                    if ally.status == "dead":
                        continue
                    options.extend(f"{ally.id}: {weapon}" for weapon in ally.weapons)
                if options:
                    ruling_id = f"ruling-{character.id}-{transaction.state.event_seq + 1}"
                    transaction.state.pending_rulings = transaction.state.pending_rulings + [
                        PendingRuling(
                            id=ruling_id, actor_id=character.id,
                            kind="demon_revenge_destroy_ally_weapon",
                            question=(
                                f"Which ally's weapon does {name} destroy? "
                                "Each option names the owner, then the weapon."
                            ),
                            options=options, default_strategy="random",
                        )
                    ]
                    revenge["ruling_id"] = ruling_id
                    details["adjudication_needed"] = True
                else:
                    revenge["no_ally_weapons"] = True
            # Face 1 names a fictional outcome the text carries to the narrator.
            result["demon_revenge"] = revenge
            if character.hp == 0 and character.status == "ok":
                if self._become_helpless(
                    transaction, character, "the demon's revenge"
                ) == "helpless":
                    transaction.warn(f"{character.name} is Helpless after the demon's revenge.")
        transaction.touch_character(character.id)
        transaction.record(f"{character.id}: invoked {power_id}")
        details["invocation"] = result
        return f"{character.name} invokes {name}.", details

    def _resolve_torn_veil(self, transaction: Transaction, character: Character) -> dict:
        """Roll the Torn Veil table for a sorcery-casting critical failure.

        Mirrors ``_invoke_power``'s Demon's Revenge dispatch: a face with an
        unambiguous single-character numeric consequence applies directly. Faces
        1, 3, and 6 carry only their SRD text for the narrator to realize -- face
        1 and 6 bind no target the engine can resolve, and face 3's blast area is
        fiction the narrator places, the same boundary Demon's Revenge's own
        face 1 already draws.
        """
        backlash_ctx = effects.apply(
            "backlash",
            effects.BacklashContext(table="torn_veil"),
            self.data.effects,
            self._effect_sources(character),
            active_ids=effects.active_ids(character),
        )
        roll = rules.roll_backlash(self.roller, 6, advantage=backlash_ctx.advantage)
        face = roll.selected
        outcome: dict = {"face": face, "text": self.data.torn_veil(face), "roll": roll.as_dict()}
        if face == 2:
            loss = self.roller.notation("d6")
            character.hp = max(0, character.hp - loss)
            outcome["hp_lost"] = loss
        elif face == 4:
            character.hp_max = max(1, character.hp_max - 1)
            character.hp = min(character.hp, character.hp_max)
            outcome["hp_max_lost"] = 1
        elif face == 5:
            character.attributes.adjust("INT", -1)
            outcome["attribute_lost"] = "INT"
        transaction.touch_character(character.id)
        transaction.record(f"{character.id}: Torn Veil face {face}")
        if character.hp == 0 and character.status == "ok":
            if self._become_helpless(transaction, character, "the torn veil") == "helpless":
                transaction.warn(f"{character.name} is Helpless after the torn veil.")
        return outcome

    def _invoke_faerie(self, transaction: Transaction, character: Character, power_id: str,
                       entry: dict, name: str, target_id: str,
                       details: dict) -> tuple[str, dict]:
        """Invoke one faerie tie. Roll the Doom die only for a tie the SRD marks.

        The SRD gives faerie ties no shared Doom-die pattern, so a tie without a
        ``doom_trigger`` field returns its effect text with no roll. Four ties carry a
        trigger: ``step_down`` decreases the Doom die (Barrow wisdom), ``roll`` rolls it
        (True faith), ``roll_advantage`` rolls it with Advantage (Doomed to greatness),
        and ``roll_on_repeat`` rolls it only on a second use in one day (Elfin secret).
        """
        trigger = entry.get("doom_trigger")
        result = {
            "power": name,
            "effect": entry["effect"],
            "target_id": target_id,
            "doom_trigger": trigger,
        }
        if trigger:
            doom_payload = self._faerie_doom(transaction, character, power_id, entry, trigger)
            if doom_payload is not None:
                result["doom"] = doom_payload
            transaction.touch_character(character.id)
        transaction.record(f"{character.id}: invoked {power_id}")
        details["invocation"] = result
        return f"{character.name} invokes {name}.", details

    def _faerie_doom(
        self,
        transaction: Transaction,
        character: Character,
        power_id: str,
        entry: dict,
        trigger: str,
    ) -> dict | None:
        """Roll or step the Doom die for one faerie tie's ``doom_trigger``.

        Returns the Doom outcome as a dict, or None when the trigger rolls nothing:
        a first ``roll_on_repeat`` use in the day, or a spent Doom die.
        """
        if trigger == "roll_advantage":
            recovery_rests = character.faerie_recovery_rests.get(power_id, 0)
            if recovery_rests:
                raise CampaignError(
                    "faerie_recovery_pending",
                    f"{character.name} needs {recovery_rests} more long rest(s) before "
                    "invoking Doomed to greatness.",
                    ["Complete the recorded long rests, then use an unused in-game day."],
                )
            daily_limit = int(entry["daily_limit"])
            if (
                character.faerie_uses_day == transaction.state.day
                and character.faerie_uses_today.get(power_id, 0) >= daily_limit
            ):
                raise CampaignError(
                    "faerie_daily_limit",
                    f"{character.name} already invoked Doomed to greatness on day "
                    f"{transaction.state.day + 1}.",
                    ["Advance the in-game day before invoking Doomed to greatness again."],
                )
            if character.doom_die == DEPLETED:
                raise CampaignError(
                    "doom_depleted",
                    f"{character.name}'s Doom die is spent; Doomed to greatness cannot roll.",
                    ["Restore the Doom die with a long rest before invoking the tie."],
                )
            self._first_faerie_use_today(character, power_id, transaction.state.day)
        elif trigger == "roll_on_repeat" and self._first_faerie_use_today(
            character, power_id, transaction.state.day
        ):
            return None
        if character.doom_die == DEPLETED:
            transaction.warn(
                f"{character.name}'s Doom die is spent; the faerie tie rolls nothing."
            )
            return None
        if trigger == "step_down":
            outcome = rules.call_on_doom(self.roller, character.doom_die)
        else:
            outcome = rules.roll_doom(
                self.roller, character.doom_die, advantage=trigger == "roll_advantage"
            )
        if outcome.downgraded:
            self._apply_doom_step(transaction, character, outcome.current_die)
            transaction.record(
                f"{character.id}: Doom {outcome.previous_die} -> {outcome.current_die} "
                f"(faerie {power_id})"
            )
        payload = outcome.as_dict()
        if trigger == "roll_advantage":
            recovery_rests = self.roller.notation(entry["recovery_long_rests"])
            character.faerie_recovery_rests = {
                **character.faerie_recovery_rests, power_id: recovery_rests
            }
            payload["recovery_long_rests"] = recovery_rests
        return payload

    def _first_faerie_use_today(self, character: Character, power_id: str, day: int) -> bool:
        """Record one faerie-tie use on ``day`` and report whether it is the day's first.

        The per-character counter resets when the in-game day changes, so a tie the
        SRD gates to once per day rolls the Doom die only on the day's later uses.
        """
        if character.faerie_uses_day != day:
            character.faerie_uses_day = day
            character.faerie_uses_today = {}
        count = character.faerie_uses_today.get(power_id, 0)
        character.faerie_uses_today = {**character.faerie_uses_today, power_id: count + 1}
        return count == 0

    def _advance_faerie_recovery(self, character: Character) -> dict[str, int]:
        """Apply one completed long rest to every recovering faerie tie."""
        remaining: dict[str, int] = {}
        progressed: dict[str, int] = {}
        for power_id, before in character.faerie_recovery_rests.items():
            after = max(0, before - 1)
            progressed[power_id] = after
            if after:
                remaining[power_id] = after
        character.faerie_recovery_rests = remaining
        return progressed

    def _marvel_maintenance_load(self, character: Character) -> int:
        """Sum the weekly maintenance points every reusable marvel the character holds draws.

        Single-use marvels carry no Usage Die and need no maintenance, so they add zero.
        """
        load = 0
        for held_id in character.powers:
            if not held_id.startswith("marvel:"):
                continue
            resolved = self.data.subsystem_power(held_id)
            if resolved and not resolved["entry"].get("single_use"):
                load += self.data.marvel_maintenance_points(int(resolved["entry"]["cost"]))
        return load

    def _drop_marvel(self, character: Character, power_id: str) -> None:
        character.powers = [p for p in character.powers if p != power_id]
        character.marvel_usage = {
            k: v for k, v in character.marvel_usage.items() if k != power_id
        }

    def _roll_marvel_usage(self, character: Character, power_id: str) -> dict:
        """Roll one reusable marvel's Usage Die and persist its new state."""
        die = character.marvel_usage.get(power_id, self.data.marvel_usage_die())
        outcome = rules.roll_usage(self.roller, die)
        result = {"usage": outcome.as_dict()}
        if outcome.depleted:
            self._drop_marvel(character, power_id)
            result["broken"] = True
        else:
            character.marvel_usage = {
                **character.marvel_usage, power_id: outcome.current_die
            }
        return result

    def _marvel_action(self, transaction: Transaction, character: Character, power_id: str,
                       entry: dict, name: str, mode: str, details: dict) -> tuple[str, dict]:
        """Build, use, or discard one twisted-science marvel.

        Binding a marvel builds it. The Inventor pays materials and workshop time.
        Invoking a single-use marvel consumes it. A reusable marvel rolls its Usage Die.
        A depleted Usage Die breaks the marvel. The narrator realizes each effect.
        """
        single_use = bool(entry.get("single_use"))
        cost = int(entry["cost"])

        if mode == "bind":
            if power_id in character.powers:
                details["built"] = True
                return f"{character.name} already holds {name}.", details
            cap = self._subsystem_capacity(character, "twisted_science")
            if cap <= 0:
                raise CampaignError(
                    "no_subsystem_access",
                    f"{character.name} is no Inventor and cannot build {name}.",
                    ["Only the Inventor background builds marvels."],
                )
            held = [p for p in character.powers if p.startswith("marvel:")]
            if len(held) >= cap:
                raise CampaignError(
                    "subsystem_full",
                    f"{character.name} already holds {len(held)} of {cap} marvels.",
                    ["Discard one before building another (mode 'release')."],
                )
            materials = self.data.marvel_materials_cost(cost)
            if character.coins < materials:
                raise CampaignError(
                    "insufficient_materials",
                    f"building {name} needs {materials} coins of materials; "
                    f"{character.name} holds {character.coins}.",
                    ["Earn or recover coins, then build again."],
                )
            weeks = self.data.marvel_build_weeks(
                cost, character.attributes.INT, self._marvel_maintenance_load(character)
            )
            character.coins -= materials
            transaction.state.in_game_minutes += weeks * 7 * 1440
            character.powers = character.powers + [power_id]
            if not single_use:
                character.marvel_usage = {
                    **character.marvel_usage, power_id: self.data.marvel_usage_die()
                }
            transaction.touch_character(character.id)
            transaction.record(
                f"{character.id}: built {power_id} ({materials} coins, {weeks} week(s))"
            )
            transaction.warn(
                f"{character.name} builds {name}; confirm a well-equipped workshop in fiction."
            )
            details.update(
                {
                    "built": True,
                    "materials_coins": materials,
                    "build_weeks": weeks,
                    "single_use": single_use,
                }
            )
            return f"{character.name} builds {name}.", details

        if mode == "release":
            self._drop_marvel(character, power_id)
            transaction.touch_character(character.id)
            transaction.record(f"{character.id}: discarded {power_id}")
            details["built"] = False
            return f"{character.name} discards {name}.", details

        # invoke = use
        if power_id not in character.powers:
            raise CampaignError(
                "power_not_bound",
                f"{character.name} has not built {name}.",
                ["Build it first with mode 'bind'."],
            )
        result: dict = {"marvel": name, "effect": entry["effect"], "single_use": single_use}
        if single_use:
            self._drop_marvel(character, power_id)
            result["consumed"] = True
        else:
            result.update(self._roll_marvel_usage(character, power_id))
        transaction.touch_character(character.id)
        transaction.record(f"{character.id}: used {power_id}")
        details["invocation"] = result
        return f"{character.name} uses {name}.", details

    def _reset_pools(self, character: Character, period: str) -> None:
        """Refresh every resource pool that resets on ``period`` by dropping its key.

        A missing key reads as a full pool, so dropping it restores the maximum.
        """
        kept = {}
        for ability_id, remaining in character.pools.items():
            entry = effects.effect_entry(self.data.effects, ability_id)
            if entry and entry.get("pool", {}).get("reset") == period:
                continue
            kept[ability_id] = remaining
        character.pools = kept

    def _refresh_day_pools(self, character: Character, day: int) -> None:
        """Lazily refresh every ``reset: "day"`` pool on the first spend of a new day.

        No tool marks an in-game day rollover. The day is derived state
        (``CampaignState.day`` is ``in_game_minutes // 1440``) that ``scene_commit``,
        ``rest``, and the marvel workshop-weeks path each advance as a side effect
        of their own purpose, and ``scene_commit``'s own contract forbids it from
        touching a character's hit points, Doom, or inventory -- a day-reset pool
        is exactly that kind of state, so an eager sweep triggered from any of
        those tools is not available. Instead this runs at the top of the
        ``resource`` branch, on every spend attempt, and compares the character's
        own ``pools_day`` marker against the day the spend is happening on --
        mirroring ``_first_faerie_use_today``'s identical lazy shape for the
        faerie daily-limit counters.
        """
        if character.pools_day != day:
            self._reset_pools(character, "day")
            character.pools_day = day

    def _resource_effect_heal_level(
        self, transaction: Transaction, character: Character
    ) -> dict:
        """Second Wind: heal the spender by their level, capped at ``hp_max``.

        Refuses before any state changes (``character_dead``, ``hp_already_full``),
        matching the project's "refuse before any die roll or write" convention --
        there is no die here, but the day's single use is just as irreplaceable.
        No target: "Regain" names the spender, and the balm (``heal_d6_plus_level``,
        the ``dose`` branch above) remains the only healing path that takes a
        ``target_id``.
        """
        if character.status == "dead":
            raise CampaignError(
                "character_dead",
                f"{character.name} is dead and cannot be healed.",
            )
        if character.hp >= character.hp_max:
            raise CampaignError(
                "hp_already_full",
                f"{character.name} is already at full hit points.",
                ["Second Wind is wasted on a character with no hit points to regain."],
            )
        previous_hp = character.hp
        character.hp = min(character.hp_max, character.hp + character.level)
        if character.status == "helpless" and character.hp > 0:
            character.status = "ok"
            transaction.record(f"{character.id}: no longer Helpless")
        transaction.touch_character(character.id)
        transaction.record(
            f"{character.id}: caught a second wind, hit points {previous_hp} -> {character.hp}"
        )
        return {
            "healed": character.hp - previous_hp,
            "hp": character.hp,
            "hp_max": character.hp_max,
        }

    def _resource_effect_replenish_usage_die(
        self, transaction: Transaction, character: Character, choice: str
    ) -> dict:
        """Resourceful: restore one tracked Usage Die to its own recorded maximum.

        Restores to ``UsageResource.maximum`` -- the die the resource was recorded
        with -- never a literal ``d6`` and never a one-step upgrade; see
        ``rules/attribution.md`` for why both alternatives were rejected. The Doom
        die cannot be named: ``doom_die``/``doom_max`` are separate ``Character``
        fields, not members of ``character.resources``, so no ``choice`` value can
        reach them.
        """
        resource = next(
            (item for item in character.resources if item.id == choice), None
        )
        if resource is None:
            raise CampaignError(
                "resource_not_found",
                f"{character.name} carries no resource with id {choice!r}.",
                [
                    "Call character_sheet to list the character's tracked resources.",
                    "Record a new resource through scene_commit before rolling it.",
                ],
            )
        if not resource.maximum:
            raise CampaignError(
                "unknown_resource_maximum",
                f"{resource.name} was recorded before its maximum die was tracked "
                "and is already depleted, so its full value is unknown.",
                ["The resource cannot be replenished until its maximum is known."],
            )
        if resource.die == resource.maximum:
            raise CampaignError(
                "resource_already_full",
                f"{resource.name} is already at its maximum, {resource.maximum}.",
                ["Resourceful is wasted on a resource with nothing to restore."],
            )
        previous_die = resource.die
        resource.die = resource.maximum
        character.resources = list(character.resources)
        transaction.touch_character(character.id)
        transaction.record(
            f"{character.id}: {resource.id} replenished {previous_die} -> {resource.die}"
        )
        return {"resource_id": resource.id, "die": resource.die}

    @guard
    def use_ability(
        self,
        character_id: str,
        ability_id: str,
        target_id: str = "",
        mode: str = "use",
        choice: str = "",
    ) -> UseAbilityResult | ToolEnvelopeFailure:
        """Activate, deactivate, or spend one of a character's registry abilities.

        The model calls this for a toggle stance (Berserker rage), a bounded
        resource (Legionnaire, Sophist, Bookworm), a standing declaration set ahead
        of the moment it matters (Herbalist's next preparation), or a discrete
        consumable spend (a prepared Herbalist dose). Every argument is a scalar,
        which the small model drives reliably. Passive effects apply automatically
        and are refused here. The legal ability ids appear in the sheet's
        ``abilities`` block. ``choice`` names the declared value for an ``intent``
        ability or the dose type for a ``dose`` ability; both list their legal
        values in the sheet.
        """
        with self.store.transaction(
            "use_ability", actor_id=character_id, reason=f"{mode} {ability_id}"
        ) as transaction:
            character = transaction.character(character_id)
            transaction.actor_id = character.id
            receipt = f"{character.id}:{ability_id}"
            if mode == "use" and receipt in transaction.state.social_ability_receipts:
                transaction.state.social_ability_receipts.pop(receipt, None)
                transaction.record("acknowledged reserved social ability")
                sequence = transaction.commit(
                    {"outcome": "social_ability_acknowledged", "ability_id": ability_id}
                )
                return success_after_commit(
                    transaction,
                    f"{character.name} uses the reserved social ability.",
                    sequence=sequence,
                    ability_id=ability_id,
                    details={"reserved_for_social_test": True},
                )

            power = self.data.subsystem_power(ability_id)
            if power is not None:
                message, details = self._invoke_power(
                    transaction, character, ability_id, power, target_id, mode
                )
                sequence = transaction.commit({"outcome": mode, **details})
                return success_after_commit(
                    transaction, message, sequence=sequence, ability_id=ability_id, details=details
                )

            entry = effects.effect_entry(self.data.effects, ability_id)
            if entry is None or entry.get("source") not in effects.source_ids(character):
                raise CampaignError(
                    "unknown_ability",
                    f"{ability_id!r} is not an ability {character.name} holds.",
                    ["Read the character sheet's 'abilities' block for the legal ability ids."],
                )
            activation = entry.get("activation", "passive")
            label = entry.get("rule", ability_id)[:60]
            details: dict = {"ability_id": ability_id, "activation": activation, "mode": mode}

            if activation == "toggle":
                if mode == "deactivate":
                    character.conditions = [
                        c for c in character.conditions if c.effect_id != ability_id
                    ]
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: stance {ability_id} deactivated")
                    details["active"] = False
                    message = f"{character.name} drops the stance."
                else:
                    if not any(c.effect_id == ability_id for c in character.conditions):
                        character.conditions = character.conditions + [
                            Condition(
                                id=ability_id, label=label, effect_id=ability_id, scope="scene"
                            )
                        ]
                        transaction.touch_character(character.id)
                        transaction.record(f"{character.id}: stance {ability_id} activated")
                    details["active"] = True
                    message = f"{character.name} takes the stance."

            elif activation == "targeted":
                if mode == "deactivate":
                    character.conditions = [
                        c for c in character.conditions if c.effect_id != ability_id
                    ]
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: {ability_id} dropped")
                    details["active"] = False
                    message = f"{character.name} drops the ward."
                else:
                    if not target_id:
                        raise CampaignError(
                            "missing_target",
                            f"{ability_id!r} must ward an ally; pass their id as target_id.",
                            ["Pass an ally character id as target_id."],
                        )
                    ally = transaction.character(target_id)  # resolves and validates existence
                    if ally.id == character.id:
                        raise CampaignError(
                            "invalid_target",
                            f"{character.name} cannot ward themselves.",
                            ["Ward a different ally."],
                        )
                    character.conditions = [
                        c for c in character.conditions if c.effect_id != ability_id
                    ] + [
                        Condition(
                            id=ability_id, label=label, effect_id=ability_id,
                            target_id=ally.id, scope="scene",
                        )
                    ]
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: {ability_id} warding {ally.id}")
                    details["active"] = True
                    details["target_id"] = ally.id
                    message = f"{character.name} guards {ally.name}."

            elif activation == "resource":
                self._refresh_day_pools(character, transaction.state.day)
                pool = entry["pool"]
                remaining = character.pools.get(ability_id, int(pool["max"]))
                reset = str(pool["reset"]).replace("_", " ")
                if remaining <= 0:
                    raise CampaignError(
                        "ability_exhausted",
                        f"{character.name} has no uses of that ability left until the next {reset}.",
                        [f"It refreshes on a {reset}."],
                    )
                # The resource_effect handler runs -- and raises its own refusals --
                # before the pool decrement below, mirroring the project's "refuse
                # before any die roll" convention: a day's or session's single use
                # is irreplaceable, so a refusal must never spend it. This is a
                # deliberate divergence from the `dose` branch's own post-decrement
                # `character_dead` check, toward that stated convention rather than
                # away from it.
                resource_effect = entry.get("resource_effect")
                if resource_effect == "heal_level":
                    details.update(self._resource_effect_heal_level(transaction, character))
                elif resource_effect == "replenish_usage_die":
                    details.update(
                        self._resource_effect_replenish_usage_die(transaction, character, choice)
                    )
                # A resource entry that declares hooks arms a one-shot stance: the
                # spend writes a scene-scoped Condition whose effect id the hook
                # sites read through effects.active_ids, and the hook that consumes
                # it clears it. Registry-driven, so no ability id appears here.
                # Every refusal below runs before the decrement below's pool write,
                # per this branch's stated check-then-spend convention.
                if entry.get("hooks"):
                    if any(c.effect_id == ability_id for c in character.conditions):
                        raise CampaignError(
                            "already_armed",
                            f"{character.name} already has that stance standing.",
                            [
                                "It is spent by the next attack it applies to, or lapses on "
                                "a long rest.",
                            ],
                        )
                    if entry.get("choices_from") == "weapons":
                        if not choice:
                            raise CampaignError(
                                "missing_choice",
                                f"{ability_id!r} must name the weapon you stake; pass it as choice.",
                                [
                                    "Read the character sheet's 'weapons' list and pass one "
                                    "name as choice.",
                                ],
                            )
                        if choice not in character.weapons:
                            raise CampaignError(
                                "weapon_not_held",
                                f"{character.name} does not hold {choice!r}.",
                                [f"Held weapons: {', '.join(character.weapons) or 'none'}."],
                            )
                        character.declared_choices = {
                            **character.declared_choices, ability_id: choice
                        }
                        details["choice"] = choice
                    character.conditions = character.conditions + [
                        Condition(id=ability_id, label=label, effect_id=ability_id, scope="scene")
                    ]
                    transaction.record(f"{character.id}: stance {ability_id} armed")
                    details["armed"] = True
                remaining -= 1
                character.pools = {**character.pools, ability_id: remaining}
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: spent {ability_id} ({remaining}/{pool['max']} left)"
                )
                details["remaining"] = remaining
                details["max"] = int(pool["max"])
                message = (
                    f"{character.name} braces for the blow; {remaining} of {pool['max']} left."
                    if details.get("armed")
                    else f"{character.name} uses the ability; {remaining} of {pool['max']} left."
                )

            elif activation == "intent":
                if mode == "deactivate":
                    character.declared_choices = {
                        k: v for k, v in character.declared_choices.items() if k != ability_id
                    }
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: withdrew the declared {ability_id}")
                    details["choice"] = None
                    message = f"{character.name} withdraws the declaration."
                else:
                    choices_from_weapons = entry.get("choices_from") == "weapons"
                    legal = self._intent_choices(character, entry)
                    if choice not in legal:
                        if choices_from_weapons and not legal:
                            hint = f"{character.name} holds no weapon to designate."
                        else:
                            hint = f"Use one of: {', '.join(sorted(legal))}."
                        raise CampaignError(
                            "invalid_intent_choice",
                            f"{choice!r} is not a legal choice for {ability_id!r}.",
                            [hint],
                        )
                    character.declared_choices = {**character.declared_choices, ability_id: choice}
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: declared {choice} for {ability_id}")
                    details["choice"] = choice
                    message = f"{character.name} declares {choice.replace('_', ' ')}."

            elif activation == "dose":
                dose_effects = entry.get("dose_effects", {})
                if choice not in dose_effects:
                    raise CampaignError(
                        "invalid_dose_type",
                        f"{choice!r} is not a dose type.",
                        [f"Legal types: {', '.join(sorted(dose_effects))}."],
                    )
                held = character.doses.get(choice, 0)
                if held < 1:
                    raise CampaignError(
                        "no_doses",
                        f"{character.name} holds no {choice.replace('_', ' ')} doses.",
                        [
                            "Declare a preparation with use_ability, then replenish on a "
                            "qualifying long rest.",
                        ],
                    )
                remaining_doses = held - 1
                if remaining_doses:
                    character.doses = {**character.doses, choice: remaining_doses}
                else:
                    character.doses = {k: v for k, v in character.doses.items() if k != choice}
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: spent one {choice} dose ({remaining_doses} left)"
                )
                details["dose_type"] = choice
                details["remaining"] = remaining_doses

                if dose_effects[choice] == "heal_d6_plus_level":
                    recipient = transaction.character(target_id) if target_id else character
                    if recipient.status == "dead":
                        raise CampaignError(
                            "character_dead",
                            f"{recipient.name} is dead and cannot be healed.",
                        )
                    rolled = self.roller.notation("d6")
                    previous_hp = recipient.hp
                    recipient.hp = min(recipient.hp_max, recipient.hp + rolled + character.level)
                    if recipient.status == "helpless" and recipient.hp > 0:
                        recipient.status = "ok"
                        transaction.record(f"{recipient.id}: no longer Helpless")
                    transaction.touch_character(recipient.id)
                    transaction.record(
                        f"{recipient.id}: tended with a balm, hit points {previous_hp} -> "
                        f"{recipient.hp}"
                    )
                    details.update(
                        {
                            "roll": rolled,
                            "level_bonus": character.level,
                            "healed": recipient.hp - previous_hp,
                            "hp": recipient.hp,
                            "hp_max": recipient.hp_max,
                            "target_id": recipient.id,
                        }
                    )
                    message = (
                        f"{character.name} tends {recipient.name} with a healing balm; "
                        f"{recipient.hp - previous_hp} hit points restored."
                    )
                else:
                    details["adjudicated"] = True
                    message = f"{character.name} spends a {choice.replace('_', ' ')} dose."

            else:
                raise CampaignError(
                    "not_activatable",
                    f"{ability_id!r} is a {activation} effect and applies automatically.",
                    ["Passive effects need no call; invocation powers ship in a later slice."],
                )

            sequence = transaction.commit({"outcome": mode, **details})

        return success_after_commit(
            transaction, message, sequence=sequence, ability_id=ability_id, details=details
        )

    def abilities_block(self, character: Character, day: int | None = None) -> list[dict]:
        """The character's activatable abilities, for the sheet and campaign status.

        Lists every non-passive effect the character's backgrounds and Gifts grant,
        with its activation, whether a stance is live, and remaining resource uses,
        so the model reads its legal ``use_ability`` targets from authoritative state.

        ``day`` is optional and, when supplied, scoped to ``reset: "day"`` pools
        only: a day-reset pool this character has not yet spent against on ``day``
        (``character.pools_day != day``) reports its full ``max`` rather than a
        stale ``pools`` entry from an earlier day, so the sheet shows the true
        remaining count before the lazy sweep in ``use_ability`` has run. Scoping
        to ``reset: "day"`` is deliberate: applying the same rule to a session or
        long-rest pool would over-report an already-spent one as full whenever
        this character's ``pools_day`` happens to lag the supplied day.
        """
        sources = effects.source_ids(character)
        live = effects.active_ids(character)
        block: list[dict] = []
        for effect_id, effect in self.data.effects.get("effects", {}).items():
            if effect.get("source") not in sources:
                continue
            activation = effect.get("activation", "passive")
            if activation == "passive":
                continue
            record = {"ability_id": effect_id, "activation": activation, "rule": effect.get("rule", "")}
            if activation in ("toggle", "targeted"):
                record["active"] = effect_id in live
            if activation == "resource":
                pool = effect["pool"]
                if (
                    pool.get("reset") == "day"
                    and day is not None
                    and character.pools_day != day
                ):
                    record["remaining"] = int(pool["max"])
                else:
                    record["remaining"] = character.pools.get(effect_id, int(pool["max"]))
                record["max"] = int(pool["max"])
                record["reset"] = pool["reset"]
            if activation == "intent":
                record["declared"] = character.declared_choices.get(effect_id)
                record["choices"] = self._intent_choices(character, effect)
            if activation == "dose":
                record["doses"] = {
                    dose_type: character.doses.get(dose_type, 0)
                    for dose_type in effect.get("dose_effects", {})
                }
            block.append(record)
        for power_id in character.powers:
            power = self.data.subsystem_power(power_id)
            if power is not None:
                block.append({
                    "ability_id": power_id, "activation": "invoke",
                    "power": power["entry"]["name"], "effect": power["entry"]["effect"],
                })
        return block

    @guard
    def ability_apply_ruling(
        self, ruling_id: str, choice: str = "", source: str = "model"
    ) -> AbilityApplyRulingResult | ToolEnvelopeFailure:
        """Apply a fictional choice to an open pending ruling. Engine-only.

        The narrator engine's adjudicate step calls this with the choice the
        narration states. An out-of-options choice, or ``source`` ``default``,
        selects by the ruling's declared strategy, so a failed adjudication still
        resolves. Idempotent: a ruling already resolved returns a structured error
        rather than mutating twice.
        """
        with self.store.transaction(
            "ability_apply_ruling", actor_id="", reason=f"resolve {ruling_id}"
        ) as transaction:
            ruling = next(
                (r for r in transaction.state.pending_rulings if r.id == ruling_id), None
            )
            if ruling is None:
                raise CampaignError(
                    "unknown_ruling",
                    f"{ruling_id!r} is not an open ruling; it may already be resolved.",
                    ["Read campaign_status for the open pending_rulings."],
                )
            transaction.actor_id = ruling.actor_id
            chosen = choice if choice in ruling.options else ""
            if not chosen and ruling.options:
                chosen = (
                    self.roller.choice(ruling.options)
                    if ruling.default_strategy == "random"
                    else ruling.options[0]
                )
                source = "default"
            details = {"ruling_id": ruling_id, "kind": ruling.kind, "choice": chosen, "source": source}
            if ruling.kind == "demon_revenge_steal" and chosen:
                character = transaction.character(ruling.actor_id)
                if chosen in character.weapons:
                    character.weapons = [w for w in character.weapons if w != chosen]
                elif chosen in character.equipment:
                    character.equipment = [e for e in character.equipment if e != chosen]
                transaction.touch_character(character.id)
                transaction.record(f"{character.id}: the demon stole {chosen}")
            if ruling.kind == "demon_revenge_destroy_ally_weapon" and chosen:
                owner_id, _, weapon = chosen.partition(": ")
                owner = transaction.character(owner_id)
                if weapon in owner.weapons:
                    owner.weapons = [w for w in owner.weapons if w != weapon]
                transaction.touch_character(owner.id)
                transaction.record(f"{owner.id}: the demon destroyed {weapon}")
            transaction.state.pending_rulings = [
                r for r in transaction.state.pending_rulings if r.id != ruling_id
            ]
            sequence = transaction.commit({"outcome": "ruling_applied", **details})
        return success_after_commit(
            transaction, f"Resolved: {chosen or 'nothing'}.", sequence=sequence, details=details
        )

    @guard
    def grant_runic_weapon(
        self,
        character_id: str,
        name: str,
        personality: str,
    ) -> GrantRunicWeaponResult | ToolEnvelopeFailure:
        """Grant a sentient runic weapon to a character, a rare in-play discovery.

        The personality fixes the weapon's damage to one of the wielder's six attributes.
        The weapon rolls its own INT with 2d6, using the character score table. This rolls
        the current session's INT test at once. ``session_close`` rolls it each session.
        A character holds one at a time: a second grant is refused before any die is
        rolled, and ``inventory_update`` (``remove_weapons``) is the way to clear the slot.
        """
        clean_name = str(name).strip()
        if not clean_name:
            raise CampaignError("empty_name", "a runic weapon needs a name.")
        personality = str(personality).strip().lower()
        if personality not in self.data.runic_personalities():
            raise CampaignError(
                "invalid_personality",
                f"{personality!r} is not a runic personality.",
                [f"Use one of {', '.join(sorted(self.data.runic_personalities()))}."],
            )
        with self.store.transaction(
            "grant_runic_weapon", actor_id=character_id, reason=clean_name
        ) as transaction:
            character = transaction.character(character_id)
            transaction.actor_id = character.id
            if character.runic_weapon is not None:
                # One slot. A second grant would roll a fresh INT and a fresh session
                # verdict over the first with nothing to stop it -- the path a
                # withheld granting turn and a ``/retry`` reaches. Refuse before any
                # die is rolled, like every impossible possession change.
                held = character.runic_weapon
                raise CampaignError(
                    "runic_weapon_held",
                    f"{character.name} already wields the runic weapon {held.name!r}; "
                    "a character holds one at a time. Nothing was changed.",
                    [
                        f"Narrate from the weapon already held ({held.name!r}); its "
                        "session test was rolled when it was granted.",
                        f"To replace it, relinquish it first: inventory_update with "
                        f"{held.name!r} in remove_weapons, then grant the new one.",
                    ],
                )
            attribute_dice = self.roller.pool(2, 6)
            attribute_total = sum(attribute_dice)
            score = rules.score_from_2d6(attribute_total)
            session_test = rules.resolve_test(self.roller, target=score)
            weapon = RunicWeapon(
                name=clean_name,
                personality=personality,
                weapon_int=score,
                kills_helpless=session_test.succeeded,
            )
            character.runic_weapon = weapon
            transaction.touch_character(character.id)
            transaction.record(
                f"{character.id}: granted runic weapon {clean_name} "
                f"({personality}, INT {score}, kills_helpless={weapon.kills_helpless})"
            )
            details = {
                "name": clean_name,
                "personality": personality,
                "damage_attribute": self.data.runic_damage_attribute(personality),
                "weapon_int": score,
                "weapon_int_roll": {
                    "notation": "2d6",
                    "dice": attribute_dice,
                    "total": attribute_total,
                    "score": score,
                },
                "kills_helpless": weapon.kills_helpless,
                "session_test": session_test.roll.as_dict(),
            }
            sequence = transaction.commit({"outcome": "runic_weapon_granted", **details})
        return success_after_commit(
            transaction,
            f"{character.name} takes up the runic weapon {clean_name}.",
            sequence=sequence,
            character_id=character.id,
            details=details,
        )
