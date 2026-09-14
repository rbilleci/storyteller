"""Loader for the JSON rules tables under ``rules/``.

Tables live in data files rather than in code so an operator can correct or
extend them without editing the rules engine. Every table validates on load.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class RulesDataError(RuntimeError):
    """Raised when a rules table is missing or structurally invalid."""


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise RulesDataError(f"missing rules table: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise RulesDataError(f"invalid JSON in {path}: {error}") from error


@dataclass(frozen=True)
class RulesData:
    """Every rules table the deterministic engine consults."""

    backgrounds: dict
    gifts: dict
    npc_table: dict
    helpless_table: dict
    weapon_effects: dict
    advancement: dict
    subsystems: dict
    effects: dict

    @classmethod
    def load(cls, rules_dir: Path) -> RulesData:
        data = cls(
            backgrounds=_load_json(rules_dir / "backgrounds.json"),
            gifts=_load_json(rules_dir / "gifts.json"),
            npc_table=_load_json(rules_dir / "npc-table.json"),
            helpless_table=_load_json(rules_dir / "helpless-table.json"),
            weapon_effects=_load_json(rules_dir / "weapon-effects.json"),
            advancement=_load_json(rules_dir / "advancement.json"),
            subsystems=_load_json(rules_dir / "subsystems.json"),
            effects=_load_json(rules_dir / "effects.json"),
        )
        data.validate()
        return data

    def validate(self) -> None:
        for origin in ("barbarian", "civilised", "decadent"):
            if origin not in self.backgrounds.get("origins", {}):
                raise RulesDataError(f"backgrounds.json omits origin {origin!r}")
            origin_block = self.backgrounds["origins"][origin]
            if not isinstance(origin_block.get("backgrounds"), list):
                raise RulesDataError(f"backgrounds.json origin {origin!r} lists no backgrounds")
            if not isinstance(origin_block.get("starting_coins"), int):
                raise RulesDataError(f"backgrounds.json origin {origin!r} omits starting_coins")

        levels = self.npc_table.get("levels", {})
        for level in range(1, 11):
            entry = levels.get(str(level))
            if not entry or "hp" not in entry or "damage" not in entry:
                raise RulesDataError(f"npc-table.json omits level {level}")
        if not isinstance(self.npc_table.get("armour_hp_bonus"), dict):
            raise RulesDataError("npc-table.json omits armour_hp_bonus")
        morale = self.npc_table.get("morale")
        if not isinstance(morale, dict) or "hp_threshold" not in morale or "force_loss_fraction" not in morale:
            raise RulesDataError("npc-table.json omits morale")

        results = self.helpless_table.get("results", {})
        for face in range(1, 7):
            if str(face) not in results:
                raise RulesDataError(f"helpless-table.json omits d6 result {face}")

        if not isinstance(self.weapon_effects.get("implemented"), dict):
            raise RulesDataError("weapon-effects.json omits the implemented block")

        if not isinstance(self.advancement.get("stories_per_level"), str):
            raise RulesDataError("advancement.json omits stories_per_level")
        if not isinstance(self.advancement.get("max_level"), int):
            raise RulesDataError("advancement.json omits an integer max_level")
        if not isinstance(self.advancement.get("attribute_increase_levels"), dict):
            raise RulesDataError("advancement.json omits attribute_increase_levels")
        if not isinstance(self.advancement.get("gift_levels"), list):
            raise RulesDataError("advancement.json omits gift_levels")
        if not isinstance(self.advancement.get("attribute_max"), int):
            raise RulesDataError("advancement.json omits an integer attribute_max")

        gifts = self.gifts.get("gifts")
        if not isinstance(gifts, list) or not gifts:
            raise RulesDataError("gifts.json omits the gifts list")
        gift_ids = [entry.get("id") for entry in gifts]
        if len(set(gift_ids)) != len(gift_ids):
            raise RulesDataError("gifts.json repeats a gift id")

        self._validate_subsystems()
        self._validate_effects()

    def _validate_effects(self) -> None:
        from .dice import DiceError
        from .dice import die_sides as _die_sides
        from .effects import HOOKS, PRIMITIVES
        from .models import ATTRIBUTE_NAMES

        registry = self.effects.get("effects")
        if not isinstance(registry, dict):
            raise RulesDataError("effects.json omits the effects block")
        test_categories = self.effects.get("test_categories", [])
        if not isinstance(test_categories, list) or not all(
            isinstance(category, str) and category for category in test_categories
        ):
            raise RulesDataError("effects.json test_categories must be a list of non-empty strings")
        backlash_tables = self.effects.get("backlash_tables", [])
        if not isinstance(backlash_tables, list) or not all(
            isinstance(table, str) and table for table in backlash_tables
        ):
            raise RulesDataError("effects.json backlash_tables must be a list of non-empty strings")
        activations = {"passive", "toggle", "targeted", "resource", "invoke", "intent", "dose"}
        resets = {"session", "long_rest", "day"}
        dose_effect_kinds = {"heal_d6_plus_level", "adjudicated"}
        resource_effect_kinds = {"heal_level", "replenish_usage_die"}
        #: Declarative choice sources an "intent" entry may draw its legal choice
        #: set from, other than a static intent_types list. Validated for every
        #: entry regardless of activation: Riddle of Steel uses it for its intent
        #: designation, and Survivor's Luck reuses it for its resource-activation
        #: stake, and Spirit alliance draws its one named spirit from the
        #: subsystems.json spirit list.
        choice_sources = {"weapons", "spirit_alliances"}
        for effect_id, effect in registry.items():
            if not effect.get("source"):
                raise RulesDataError(f"effects.json effect {effect_id!r} omits a source")
            activation = effect.get("activation", "passive")
            if activation not in activations:
                raise RulesDataError(
                    f"effects.json effect {effect_id!r} has unknown activation {activation!r}"
                )
            if "choices_from" in effect and effect["choices_from"] not in choice_sources:
                raise RulesDataError(
                    f"effects.json effect {effect_id!r} names unknown choices_from "
                    f"{effect['choices_from']!r}; legal values are {sorted(choice_sources)}"
                )
            if activation == "resource":
                pool = effect.get("pool", {})
                if not isinstance(pool.get("max"), int) or pool.get("reset") not in resets:
                    raise RulesDataError(
                        f"effects.json resource effect {effect_id!r} needs an integer pool.max "
                        f"and a reset in {sorted(resets)}"
                    )
                resource_effect = effect.get("resource_effect")
                if resource_effect is not None and resource_effect not in resource_effect_kinds:
                    raise RulesDataError(
                        f"effects.json resource effect {effect_id!r} names unknown "
                        f"resource_effect {resource_effect!r}; legal values are "
                        f"{sorted(resource_effect_kinds)}"
                    )
            if activation == "intent":
                intent_types = effect.get("intent_types")
                has_intent_types = isinstance(intent_types, list) and bool(intent_types) and all(
                    isinstance(t, str) and t for t in intent_types
                )
                choices_from = effect.get("choices_from")
                has_choices_from = isinstance(choices_from, str) and bool(choices_from)
                if has_intent_types == has_choices_from:
                    raise RulesDataError(
                        f"effects.json intent effect {effect_id!r} needs exactly one of a "
                        "non-empty intent_types list of non-empty strings, or a non-empty "
                        "choices_from string"
                    )
            if activation == "dose":
                dose_effects = effect.get("dose_effects")
                if not isinstance(dose_effects, dict) or not dose_effects or not all(
                    isinstance(k, str) and k and v in dose_effect_kinds
                    for k, v in dose_effects.items()
                ):
                    raise RulesDataError(
                        f"effects.json dose effect {effect_id!r} needs a non-empty "
                        f"dose_effects mapping whose values are in {sorted(dose_effect_kinds)}"
                    )
            hooks = effect.get("hooks", {})
            if not isinstance(hooks, dict):
                raise RulesDataError(f"effects.json effect {effect_id!r} has a non-object hooks")
            for hook, operations in hooks.items():
                if hook not in HOOKS:
                    raise RulesDataError(
                        f"effects.json effect {effect_id!r} names unknown hook {hook!r}"
                    )
                for op in operations:
                    name = op.get("op")
                    if name not in PRIMITIVES:
                        raise RulesDataError(
                            f"effects.json effect {effect_id!r} names unknown primitive {name!r}"
                        )
                    if name == "override_damage_with_attribute":
                        if op.get("attribute") not in ATTRIBUTE_NAMES:
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op "
                                f"override_damage_with_attribute needs an attribute in "
                                f"{ATTRIBUTE_NAMES}"
                            )
                    if name == "grant_category_advantage":
                        categories = op.get("categories", [])
                        if not categories or not set(categories) <= set(test_categories):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op grant_category_advantage "
                                f"names a category outside test_categories {sorted(test_categories)}"
                            )
                    if name == "grant_backlash_advantage":
                        tables = op.get("tables", [])
                        if not tables or not set(tables) <= set(backlash_tables):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op grant_backlash_advantage "
                                f"names a table outside backlash_tables {sorted(backlash_tables)}"
                            )
                    if name == "resolve_declared_preparation":
                        types = op.get("types")
                        if not isinstance(types, list) or not all(
                            isinstance(t, str) and t for t in types
                        ):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op "
                                "resolve_declared_preparation needs a non-empty types list "
                                "of non-empty strings"
                            )
                        try:
                            _die_sides(str(op.get("quantity_die", "")))
                        except DiceError as error:
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op "
                                f"resolve_declared_preparation needs a valid quantity_die: {error}"
                            ) from error
                        tag = op.get("required_environment_tag")
                        if tag is not None and (not isinstance(tag, str) or not tag):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op "
                                "resolve_declared_preparation's required_environment_tag must be "
                                "a non-empty string when present"
                            )
                    if name == "tend_helpless":
                        if op.get("test_attribute") not in ATTRIBUTE_NAMES:
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op tend_helpless needs a "
                                f"test_attribute in {ATTRIBUTE_NAMES}"
                            )
                        die_sides = op.get("die_sides")
                        if not isinstance(die_sides, int) or not (2 <= die_sides <= 6):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op tend_helpless needs an "
                                "integer die_sides between 2 and 6"
                            )
                    if name == "widen_critical_success":
                        max_face = op.get("max_face")
                        if not isinstance(max_face, int) or not (2 <= max_face <= 19):
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op widen_critical_success "
                                "needs an integer max_face between 2 and 19"
                            )
                    if name == "step_up_damage_die":
                        steps = op.get("steps", 1)
                        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op step_up_damage_die needs "
                                "an integer steps of at least 1 when present"
                            )
                    if name == "override_damage_die":
                        from .dice import DAMAGE_CHAIN

                        die = op.get("die")
                        if die not in DAMAGE_CHAIN:
                            raise RulesDataError(
                                f"effects.json effect {effect_id!r} op override_damage_die "
                                f"needs a die in {DAMAGE_CHAIN}"
                            )
                        if "break_on_face" in op:
                            break_on_face = op["break_on_face"]
                            if not isinstance(break_on_face, int) or isinstance(
                                break_on_face, bool
                            ) or not (1 <= break_on_face <= _die_sides(die)):
                                raise RulesDataError(
                                    f"effects.json effect {effect_id!r} op override_damage_die "
                                    f"needs an integer break_on_face between 1 and "
                                    f"{_die_sides(die)} when present"
                                )

    def _validate_subsystems(self) -> None:
        subs = self.subsystems
        for face_table, path in (
            (subs.get("demonic_pacts", {}).get("demon_revenge", {}),
             "subsystems.json demonic_pacts.demon_revenge"),
            (subs.get("sorcery", {}).get("torn_veil", {}),
             "subsystems.json sorcery.torn_veil"),
            (subs.get("runic_weapons", {}).get("on_kill", {}),
             "subsystems.json runic_weapons.on_kill"),
        ):
            results = face_table.get("results", {})
            for face in range(1, 7):
                if str(face) not in results:
                    raise RulesDataError(f"{path} omits d6 result {face}")

        spells = subs.get("sorcery", {}).get("spell_table", {}).get("spells")
        if not isinstance(spells, list) or not spells:
            raise RulesDataError("subsystems.json omits the sorcery spell_table")
        seen: set[int] = set()
        for spell in spells:
            low, high = spell.get("low"), spell.get("high")
            if not isinstance(low, int) or not isinstance(high, int) or low > high:
                raise RulesDataError(f"subsystems.json spell {spell.get('id')!r} has a bad range")
            for face in range(low, high + 1):
                if face in seen:
                    raise RulesDataError(f"subsystems.json spell table overlaps at d100 {face}")
                seen.add(face)
        if seen != set(range(1, 101)):
            missing = sorted(set(range(1, 101)) - seen)
            raise RulesDataError(f"subsystems.json spell table omits d100 result(s) {missing}")

        ties = subs.get("faerie_ties", {}).get("ties", [])
        doomed_to_greatness = next(
            (tie for tie in ties if tie.get("id") == "doomed_to_greatness"), None
        )
        if not isinstance(doomed_to_greatness, dict):
            raise RulesDataError("subsystems.json omits the Doomed to greatness faerie tie")
        daily_limit = doomed_to_greatness.get("daily_limit")
        if type(daily_limit) is not int or daily_limit != 1:
            raise RulesDataError("Doomed to greatness needs a daily_limit of 1")
        if doomed_to_greatness.get("recovery_long_rests") != "d3":
            raise RulesDataError("Doomed to greatness needs d3 long-rest recovery")

    # -- lookups -------------------------------------------------------------

    def background_index(self) -> dict[str, dict]:
        """Return every background keyed by its identifier."""
        index: dict[str, dict] = {}
        for origin, block in self.backgrounds["origins"].items():
            for entry in block["backgrounds"]:
                record = dict(entry)
                record["origin"] = origin
                index[entry["id"]] = record
        return index

    def starting_coins(self, origin: str) -> int:
        return int(self.backgrounds["origins"][origin]["starting_coins"])

    def origin_languages(self, origin: str) -> list[str]:
        common = list(self.backgrounds.get("common_languages", []))
        return common + list(self.backgrounds["origins"][origin].get("languages", []))

    def starting_weapons(self, origin: str) -> list[str]:
        return list(self.backgrounds["origins"][origin].get("weapon_table", []))

    def npc_stats(self, level: int, armour: str) -> tuple[int, int]:
        entry = self.npc_table["levels"][str(level)]
        bonus = int(self.npc_table["armour_hp_bonus"].get(armour, 0))
        return int(entry["hp"]) + bonus, int(entry["damage"])

    def helpless_result(self, face: int) -> dict:
        return dict(self.helpless_table["results"][str(face)])

    def implemented_effects(self) -> list[str]:
        return sorted(self.weapon_effects["implemented"].keys())

    def effect_rule(self, effect: str) -> dict | None:
        return self.weapon_effects["implemented"].get(effect)

    def gift_index(self) -> dict[str, dict]:
        """Return every Gift keyed by its identifier."""
        return {entry["id"]: dict(entry) for entry in self.gifts.get("gifts", [])}

    # -- subsystem lookups ---------------------------------------------------

    def subsystem(self, name: str) -> dict:
        """Return one subsystem block (e.g. ``sorcery``) from subsystems.json."""
        block = self.subsystems.get(name)
        if not isinstance(block, dict):
            raise RulesDataError(f"subsystems.json omits subsystem {name!r}")
        return block

    def spell_for_roll(self, roll: int) -> dict:
        """Return the sorcery spell whose d100 range contains ``roll``."""
        for spell in self.subsystems["sorcery"]["spell_table"]["spells"]:
            if spell["low"] <= roll <= spell["high"]:
                return dict(spell)
        raise RulesDataError(f"no sorcery spell covers d100 result {roll}")

    def demon_revenge(self, face: int) -> str:
        """Return the Demon's Revenge outcome text for a d6 ``face``."""
        return self.subsystems["demonic_pacts"]["demon_revenge"]["results"][str(face)]

    def torn_veil(self, face: int) -> str:
        """Return the Torn Veil outcome text for a d6 ``face``."""
        return self.subsystems["sorcery"]["torn_veil"]["results"][str(face)]

    #: Maps a power-id prefix to its subsystem block and list key.
    _POWER_SUBSYSTEMS = {
        "demon": ("demonic_pacts", "demons"),
        "spirit": ("spirit_alliances", "spirits"),
        "faerie": ("faerie_ties", "ties"),
        "marvel": ("twisted_science", "marvels"),
    }

    def subsystem_power(self, power_id: str) -> dict | None:
        """Resolve a ``prefix:id`` power (e.g. ``demon:abyss``) to its subsystem entry.

        Returns ``{prefix, subsystem, entry}`` or None when the id names no power.
        """
        if ":" not in power_id:
            return None
        prefix, pid = power_id.split(":", 1)
        mapping = self._POWER_SUBSYSTEMS.get(prefix)
        if mapping is None:
            return None
        subsystem_key, list_key = mapping
        for entry in self.subsystems.get(subsystem_key, {}).get(list_key, []):
            if entry.get("id") == pid:
                return {"prefix": prefix, "subsystem": subsystem_key, "entry": dict(entry)}
        return None

    def runic_personalities(self) -> dict[str, str]:
        """Return the runic personality-to-attribute map (brutal -> STR, ...)."""
        return dict(self.subsystems["runic_weapons"]["personalities"])

    def runic_damage_attribute(self, personality: str) -> str:
        """Return the attribute a runic weapon's ``personality`` sets damage to."""
        table = self.subsystems["runic_weapons"]["personalities"]
        if personality not in table:
            raise RulesDataError(f"runic_weapons omits personality {personality!r}")
        return table[personality]

    def runic_on_kill(self, face: int) -> str:
        """Return the runic weapon's on-kill effect text for a d6 ``face``."""
        return self.subsystems["runic_weapons"]["on_kill"]["results"][str(face)]

    def runic_on_kill_heal_face(self) -> int:
        """Return the d6 face on which the runic weapon heals its wielder."""
        return int(self.subsystems["runic_weapons"]["on_kill"]["heal_face"])

    def runic_on_kill_heal_die(self) -> str:
        """Return the die the runic weapon's heal face restores (d6)."""
        return self.subsystems["runic_weapons"]["on_kill"]["heal_die"]

    def marvel_usage_die(self) -> str:
        """Return the Usage Die a reusable twisted-science marvel carries (Ud6)."""
        return self.subsystems["twisted_science"]["usage_die"]

    def marvel_materials_cost(self, cost: int) -> int:
        """Return the coins of materials a marvel needs: 20 per invention point."""
        per_point = int(self.subsystems["twisted_science"]["materials_coins_per_point"])
        return per_point * int(cost)

    def marvel_maintenance_points(self, cost: int) -> int:
        """Return a reusable marvel's weekly maintenance draw: half its cost, rounded up."""
        return math.ceil(int(cost) / 2)

    def marvel_build_weeks(self, cost: int, int_score: int, maintenance_load: int) -> int:
        """Return the workshop weeks to build a marvel.

        An Inventor produces ``int_score`` invention points per week, less the points
        every reusable marvel already held draws for maintenance. Building a marvel
        costs its invention-point number, so the weeks equal that cost divided by the
        net weekly rate, rounded up. The rate never drops below one point per week, so
        a heavy maintenance load slows a build rather than stalling it forever.
        """
        rate = max(1, int(int_score) - int(maintenance_load))
        return math.ceil(int(cost) / rate)


@lru_cache(maxsize=8)
def load_rules_data(rules_dir: str) -> RulesData:
    """Load and cache the rules tables for one ``rules/`` directory."""
    return RulesData.load(Path(rules_dir))
