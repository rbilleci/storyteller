"""Pure Black Sword Hack mechanics.

Every function here is deterministic given a :class:`~bsh_mcp.dice.Roller`.
No function touches the filesystem, campaign state, or Model Context Protocol
(MCP) plumbing. The store layer applies the results these functions return.

Reference: Black Sword Hack - Ultimate Chaos Edition SRD v1.0.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from .data import RulesData
from .dice import DEPLETED, Roll, Roller, die_sides, resolve_edge, step_down
from .models import (
    ARMOUR_CATEGORIES,
    ARMOUR_PROTECTION,
    ATTRIBUTE_NAMES,
    RANGE_BANDS,
    Attributes,
    Character,
)

Outcome = Literal["critical_success", "success", "failure", "critical_failure"]

#: 2d6 to attribute score, SRD "Attributes".
ATTRIBUTE_SCORE_TABLE: dict[int, int] = {
    2: 8,
    3: 8,
    4: 9,
    5: 9,
    6: 10,
    7: 10,
    8: 11,
    9: 11,
    10: 12,
    11: 12,
    12: 13,
}


# ---------------------------------------------------------------------------
# Attribute tests
# ---------------------------------------------------------------------------


@dataclass
class DoomOutcome:
    """Result of touching a character's Doom die."""

    mode: Literal["roll", "call_on_doom", "restore", "none"]
    previous_die: str
    current_die: str
    roll: Roll | None = None
    downgraded: bool = False
    depleted: bool = False
    restored: bool = False

    def as_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "previous_die": self.previous_die,
            "current_die": self.current_die,
            "downgraded": self.downgraded,
            "depleted": self.depleted,
        }
        if self.roll is not None:
            payload["roll"] = self.roll.as_dict()
        if self.restored:
            payload["restored"] = True
        return payload


@dataclass
class TestResult:
    """One resolved roll-under attribute test."""

    roll: Roll
    outcome: Outcome
    target: int
    threat_modifier: int = 0
    doom: DoomOutcome | None = None
    doom_penalty: int = 0

    @property
    def succeeded(self) -> bool:
        return self.outcome in ("critical_success", "success")

    def as_dict(self) -> dict:
        payload = {
            "roll": self.roll.as_dict(),
            "outcome": self.outcome,
            "target": self.target,
            "threat_modifier": self.threat_modifier,
        }
        if self.doom is not None:
            payload["doom"] = self.doom.as_dict()
        if self.doom_penalty:
            payload["doom_penalty"] = self.doom_penalty
        return payload


def threat_modifier(actor_level: int, opponent_level: int | None) -> int:
    """Return the Threat Level penalty added to the actor's d20.

    A higher-level opponent adds the level difference. A lower-level opponent
    grants no bonus, so the result is never negative.
    """
    if opponent_level is None:
        return 0
    return max(0, int(opponent_level) - int(actor_level))


def classify(selected: int, total: int, target: int, *, crit_success_max: int = 1) -> Outcome:
    """Classify a roll-under result.

    Criticals read the unmodified selected die, so Threat Level and Doom never
    create or erase a critical. ``crit_success_max`` widens the critical-success
    band to any unmodified roll at or below it (default 1, i.e. current
    behavior); it is clamped below 20 so a natural 20 always classifies as a
    critical failure regardless of how wide the band is.
    """
    if selected <= min(crit_success_max, 19):
        return "critical_success"
    if selected == 20:
        return "critical_failure"
    return "success" if total < target else "failure"


def resolve_test(
    roller: Roller,
    *,
    target: int,
    advantage: bool = False,
    disadvantage: bool = False,
    modifier: int = 0,
    crit_success_max: int = 1,
) -> TestResult:
    """Roll one roll-under d20 attribute test."""
    edge = resolve_edge(advantage, disadvantage)
    roll = roller.d20(edge=edge, modifier=modifier, target=target)
    outcome = classify(roll.selected, roll.total, target, crit_success_max=crit_success_max)
    return TestResult(roll=roll, outcome=outcome, target=target, threat_modifier=modifier)


def character_edges(
    character: Character,
    attribute: str,
    advantage: bool,
    disadvantage: bool,
) -> tuple[bool, bool, list[str]]:
    """Fold conditions into the fiction-supplied Advantage and Disadvantage.

    A Doomed or Injured character tests everything at Disadvantage. An Impaired
    character tests DEX at Disadvantage.
    """
    notes: list[str] = []
    for condition in character.conditions:
        if condition.effect == "disadvantage_all":
            disadvantage = True
            notes.append(f"{condition.label} imposes Disadvantage on all tests.")
        elif condition.effect == "disadvantage_dex" and attribute.upper() == "DEX":
            disadvantage = True
            notes.append(f"{condition.label} imposes Disadvantage on DEX tests.")
    return advantage, disadvantage, notes


def damage_edges(character: Character, two_handed: bool) -> tuple[bool, bool, list[str]]:
    """Return Advantage and Disadvantage for a damage roll."""
    notes: list[str] = []
    disadvantage = False
    for condition in character.conditions:
        if condition.effect in ("disadvantage_all", "disadvantage_damage"):
            disadvantage = True
            notes.append(f"{condition.label} imposes Disadvantage on damage rolls.")
    return two_handed, disadvantage, notes


# ---------------------------------------------------------------------------
# Usage Dice and Doom
# ---------------------------------------------------------------------------


@dataclass
class UsageOutcome:
    roll: Roll
    previous_die: str
    current_die: str
    downgraded: bool
    depleted: bool

    def as_dict(self) -> dict:
        return {
            "roll": self.roll.as_dict(),
            "previous_die": self.previous_die,
            "current_die": self.current_die,
            "downgraded": self.downgraded,
            "depleted": self.depleted,
        }


def roll_usage(
    roller: Roller, die: str, *, advantage: bool = False, disadvantage: bool = False
) -> UsageOutcome:
    """Roll a Usage Die. A result of 1 or 2 steps the die down one grade."""
    if die == DEPLETED:
        raise ValueError("a depleted Usage Die cannot be rolled")
    edge = resolve_edge(advantage, disadvantage)
    roll = roller.usage(die, edge=edge)
    downgraded = roll.selected <= 2
    current = step_down(die) if downgraded else die
    return UsageOutcome(
        roll=roll,
        previous_die=die,
        current_die=current,
        downgraded=downgraded,
        depleted=current == DEPLETED,
    )


def roll_doom(
    roller: Roller, die: str, *, disadvantage: bool = False, advantage: bool = False
) -> DoomOutcome:
    """Roll the Doom die. 1 or 2 steps it down; a spent d4 leaves the character Doomed."""
    usage = roll_usage(roller, die, advantage=advantage, disadvantage=disadvantage)
    return DoomOutcome(
        mode="roll",
        previous_die=usage.previous_die,
        current_die=usage.current_die,
        roll=usage.roll,
        downgraded=usage.downgraded,
        depleted=usage.depleted,
    )


def roll_backlash(roller: Roller, sides: int, *, advantage: bool = False) -> Roll:
    """Roll a high-is-worse consequence table face (Helpless, Demon's Revenge,
    Torn Veil). Advantage keeps the lower, safer face."""
    return roller.backlash(sides, edge=resolve_edge(advantage, False))


def call_on_doom(roller: Roller, die: str, *, disadvantage: bool = False) -> DoomOutcome:
    """Call on Doom: roll it, subtract the result from a test, always step down."""
    if die == DEPLETED:
        raise ValueError("a depleted Doom die cannot be called upon")
    edge = resolve_edge(False, disadvantage)
    roll = roller.usage(die, edge=edge)
    current = step_down(die)
    return DoomOutcome(
        mode="call_on_doom",
        previous_die=die,
        current_die=current,
        roll=roll,
        downgraded=True,
        depleted=current == DEPLETED,
    )


def restore_doom(maximum: str) -> DoomOutcome:
    return DoomOutcome(
        mode="restore",
        previous_die=maximum,
        current_die=maximum,
        downgraded=False,
        depleted=False,
        restored=True,
    )


# ---------------------------------------------------------------------------
# Damage
# ---------------------------------------------------------------------------


@dataclass
class DamageOutcome:
    total: int
    rolls: list[Roll] = field(default_factory=list)
    critical: bool = False
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "rolls": [roll.as_dict() for roll in self.rolls],
            "critical": self.critical,
            "detail": self.detail,
        }


def roll_damage(
    roller: Roller,
    die: str,
    *,
    two_handed: bool = False,
    disadvantage: bool = False,
    critical: bool = False,
    brutal: bool = False,
) -> DamageOutcome:
    """Roll weapon damage.

    A critical hit deals maximum base damage plus one additional damage die.
    Two-handed weapons grant Advantage on the rolled dice.
    """
    edge = resolve_edge(two_handed, disadvantage)
    sides = die_sides(die)
    rolls: list[Roll] = []

    if critical:
        base = sides
        extra = replace(roller.damage(die, edge=edge), note="additional critical damage die")
        rolls.append(extra)
        total = base + extra.selected
        detail = f"maximum base damage {base} plus additional die {extra.selected}"
    else:
        base_roll = roller.damage(die, edge=edge)
        if brutal and base_roll.selected == 1:
            rolls.append(replace(base_roll, note="brutal reroll discarded"))
            base_roll = replace(roller.damage(die, edge=edge), note="brutal reroll")
        rolls.append(base_roll)
        total = base_roll.selected
        detail = f"base damage {total}"

    return DamageOutcome(total=total, rolls=rolls, critical=critical, detail=detail)


def apply_armour(damage: int, protection: int, *, ignore_armour: bool = False) -> tuple[int, int]:
    """Return ``(applied_damage, absorbed)``. Damage never drops below zero."""
    if ignore_armour or protection <= 0:
        return max(0, damage), 0
    applied = max(0, damage - protection)
    return applied, damage - applied


def armour_protection(category: str) -> int:
    if category not in ARMOUR_CATEGORIES:
        raise ValueError(f"unknown armour category: {category!r}")
    return ARMOUR_PROTECTION[category]


# ---------------------------------------------------------------------------
# Character creation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedCharacter:
    attributes: Attributes
    attribute_rolls: dict[str, dict]
    hp: int
    coins: int
    languages: list[str]
    background_features: list[str]
    warnings: list[str] = field(default_factory=list)
    spells: list[str] = field(default_factory=list)
    spell_draws: list[dict] = field(default_factory=list)


def score_from_2d6(total: int) -> int:
    """Map a 2d6 total to an attribute score between 8 and 13."""
    if total not in ATTRIBUTE_SCORE_TABLE:
        raise ValueError(f"2d6 total out of range: {total}")
    return ATTRIBUTE_SCORE_TABLE[total]


def validate_backgrounds(
    origin: str, background_ids: list[str], data: RulesData
) -> list[str]:
    """Return a list of rule violations. An empty list means the selection is legal."""
    errors: list[str] = []
    index = data.background_index()

    if len(background_ids) != 3:
        errors.append(f"select exactly 3 backgrounds, received {len(background_ids)}")
    if len(set(background_ids)) != len(background_ids):
        errors.append("the same background cannot be selected twice")

    unknown = [bid for bid in background_ids if bid not in index]
    if unknown:
        errors.append(f"unknown background ids: {', '.join(sorted(unknown))}")
        return errors

    from_origin = [bid for bid in background_ids if index[bid]["origin"] == origin]
    if len(from_origin) < 2:
        errors.append(
            f"at least 2 backgrounds must come from origin {origin!r}, received {len(from_origin)}"
        )

    unique_selected = [bid for bid in background_ids if index[bid].get("unique")]
    if len(unique_selected) > 1:
        errors.append(
            "no more than 1 unique background is allowed, received "
            f"{', '.join(sorted(unique_selected))}"
        )

    return errors


def generate_character(
    roller: Roller, origin: str, background_ids: list[str], data: RulesData
) -> GeneratedCharacter:
    """Roll attributes, apply background increases, and derive starting values."""
    from .models import ATTRIBUTE_NAMES

    index = data.background_index()
    scores: dict[str, int] = {}
    attribute_rolls: dict[str, dict] = {}

    for name in ATTRIBUTE_NAMES:
        dice = roller.pool(2, 6)
        total = sum(dice)
        scores[name] = score_from_2d6(total)
        attribute_rolls[name] = {"dice": dice, "total": total, "score": scores[name]}

    warnings: list[str] = []
    features: list[str] = []
    spells: list[str] = []
    spell_draws: list[dict] = []
    for background_id in background_ids:
        record = index[background_id]
        for attribute, delta in (record.get("attribute_bonus") or {}).items():
            key = attribute.upper()
            if key not in ATTRIBUTE_NAMES:
                warnings.append(f"background {background_id!r} names unknown attribute {attribute!r}")
                continue
            scores[key] = min(20, scores[key] + int(delta))
        feature = record.get("feature")
        if feature:
            features.append(f"{record['name']}: {feature}")
        if record.get("automation_tag") == "draw_spells_4":
            spell_draws = draw_starting_spells(roller, data)
            spells = [draw["name"] for draw in spell_draws]
            features.append(
                f"{record['name']} spells: {', '.join(spells)} (see bsh://rules/subsystems)."
            )
        if record.get("automation") == "manual":
            warnings.append(_manual_background_warning(record))

    attributes = Attributes(**scores)
    return GeneratedCharacter(
        attributes=attributes,
        attribute_rolls=attribute_rolls,
        hp=attributes.CON,
        coins=data.starting_coins(origin),
        languages=data.origin_languages(origin),
        background_features=features,
        warnings=warnings,
        spells=spells,
        spell_draws=spell_draws,
    )


def _manual_background_warning(record: dict) -> str:
    """One warning line for a manual background, subsystem-aware when it links one.

    A subsystem background points the narrator at the encoded canon in
    ``bsh://rules/subsystems`` and names any capacity it grants, so the narrator
    adjudicates from the rules rather than from memory.
    """
    subsystem = record.get("subsystem")
    if subsystem:
        readable = subsystem.replace("_", " ")
        slots = record.get("subsystem_slots")
        capacity = f" up to {slots}" if isinstance(slots, int) else ""
        return (
            f"background {record['name']!r} grants{capacity} {readable}; the narrator "
            "adjudicates it from bsh://rules/subsystems and records outcomes with scene_commit"
        )
    return (
        f"background {record['name']!r} has narrative-only effects; "
        "the narrator applies them as rulings"
    )


def draw_starting_spells(roller: Roller, data: RulesData, count: int | None = None) -> list[dict]:
    """Roll the Forbidden Knowledge starting spells on the d100 sorcery table.

    The SRD grants a Forbidden Knowledge character four spells rolled on d100.
    Each draw is independent, so a repeated d100 result yields the same spell
    twice; the narrator resolves a duplicate as a ruling. Returns one record per
    draw carrying the die result, the spell id, name, and effect text.
    """
    sorcery = data.subsystem("sorcery")
    if count is None:
        count = int(sorcery.get("starting_spells", 4))
    draws: list[dict] = []
    for _ in range(count):
        roll = roller.die(100)
        spell = data.spell_for_roll(roll)
        draws.append({"roll": roll, "id": spell["id"], "name": spell["name"],
                      "effect": spell["effect"]})
    return draws


def roll_starting_weapons(roller: Roller, origin: str, data: RulesData) -> list[str]:
    """Roll two d10 weapons: one from the origin table, one from any table.

    A result of 10 yields no weapon, matching the SRD's ``0/10`` empty entry.
    """
    second_origin = roller.choice(["barbarian", "civilised", "decadent"])
    weapons: list[str] = []
    for table_origin in (origin, second_origin):
        table = data.starting_weapons(table_origin)
        if not table:
            continue
        index = roller.die(10)
        if index > len(table):
            continue
        weapons.append(table[index - 1])
    return weapons


# ---------------------------------------------------------------------------
# Group tests
# ---------------------------------------------------------------------------


def group_succeeds(successes: int, participants: int) -> bool:
    """The group succeeds when at least half of the participants succeed."""
    if participants <= 0:
        return False
    return successes * 2 >= participants


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def short_rest_healing(constitution: int) -> int:
    """A short rest restores half CON, rounded down."""
    return max(0, constitution // 2)


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def band_index(band: str) -> int:
    if band not in RANGE_BANDS:
        raise ValueError(f"unknown range band: {band!r}")
    return RANGE_BANDS.index(band)


def move_band(band: str, direction: Literal["closer", "away"]) -> str:
    """Move one range band. One move changes one band."""
    index = band_index(band)
    if direction == "closer":
        return RANGE_BANDS[max(0, index - 1)]
    return RANGE_BANDS[min(len(RANGE_BANDS) - 1, index + 1)]


# ---------------------------------------------------------------------------
# NPCs
# ---------------------------------------------------------------------------


def npc_stats(level: int, armour: str, data: RulesData) -> tuple[int, int]:
    if not 1 <= level <= 10:
        raise ValueError(f"NPC level must fall between 1 and 10, received {level}")
    if armour not in ARMOUR_CATEGORIES:
        raise ValueError(f"unknown armour category: {armour!r}")
    return data.npc_stats(level, armour)


def npc_morale_warning(hp: int, side_losses_fraction: float, data: RulesData) -> str | None:
    """Return a morale prompt when the SRD's flight or surrender condition holds."""
    morale = data.npc_table["morale"]
    if hp < morale["hp_threshold"]:
        return "Fewer than 3 HP remain: humanoid enemies may surrender or flee."
    if side_losses_fraction >= morale["force_loss_fraction"]:
        return "Half the enemy force is down: survivors may surrender or flee."
    return None


# ---------------------------------------------------------------------------
# Advancement
# ---------------------------------------------------------------------------


def stories_required_for_level(level: int) -> int:
    """Return the cumulative Stories needed to reach ``level``.

    A character advances after collecting Stories equal to their current level,
    so the cumulative requirement is the triangular number of ``level - 1``.
    """
    if level <= 1:
        return 0
    return (level - 1) * level // 2


def eligible_level(stories: int) -> int:
    level = 1
    while level < 10 and stories >= stories_required_for_level(level + 1):
        level += 1
    return level


def advancement_benefits(target_level: int, data: RulesData) -> dict:
    """Return the SRD benefits a character gains on reaching ``target_level``.

    ``target_level`` is the level the character advances into, so the lowest
    value is 2. The dictionary states the hit-point gain, the number of attribute
    increases the player must choose, whether the level grants a Gift, and the
    Doom die the level sets, if any. Every figure derives from advancement.json.
    """
    advancement = data.advancement
    max_level = int(advancement["max_level"])
    if not 2 <= target_level <= max_level:
        raise ValueError(
            f"advancement target level must fall between 2 and {max_level}, "
            f"received {target_level}"
        )
    attribute_levels = advancement.get("attribute_increase_levels", {})
    gift_levels = {int(level) for level in advancement.get("gift_levels", [])}
    # The SRD grants +1 hit point at levels 2 through 9 and the Doom die upgrade
    # at level 10, so the top level adds no hit point.
    hit_point_gain = int(advancement.get("hp_per_level", 1)) if target_level < max_level else 0
    return {
        "target_level": target_level,
        "hit_point_gain": hit_point_gain,
        "attribute_increases": int(attribute_levels.get(str(target_level), 0)),
        "grants_gift": target_level in gift_levels,
        "doom_die": str(advancement["level_10_doom_die"]) if target_level == max_level else "",
        "attribute_max": int(advancement["attribute_max"]),
    }


def validate_advancement_choices(
    character: Character,
    benefits: dict,
    attribute_increases: list[str],
    gift_id: str,
    gift_index: dict[str, dict],
) -> list[str]:
    """Return a list of rule violations. An empty list means the choices are legal.

    ``gift_index`` maps a Gift id to its record, as ``RulesData.gift_index``
    returns. The caller builds it once and reuses it to apply the chosen Gift, so
    the validation and the write share one lookup table.

    The tool applies these choices only after this returns empty, so a rejected
    call writes nothing.
    """
    errors: list[str] = []

    required = int(benefits["attribute_increases"])
    if len(attribute_increases) != required:
        errors.append(
            f"level {benefits['target_level']} raises exactly {required} "
            f"attribute(s), received {len(attribute_increases)}"
        )
    unknown = [name for name in attribute_increases if name not in ATTRIBUTE_NAMES]
    if unknown:
        errors.append(f"unknown attribute names: {', '.join(unknown)}")
    if required >= 2 and len(set(attribute_increases)) != len(attribute_increases):
        # The SRD wording "+1 to two attributes" raises two distinct attributes.
        errors.append("each attribute may be raised once per level; choose distinct attributes")
    attribute_max = int(benefits["attribute_max"])
    for name in attribute_increases:
        if name in ATTRIBUTE_NAMES and character.attributes.get(name) >= attribute_max:
            errors.append(
                f"{name} is already at the advancement maximum of {attribute_max}; "
                "choose another attribute"
            )

    if benefits["grants_gift"]:
        if not gift_id:
            errors.append(f"level {benefits['target_level']} grants a Gift; name one gift_id")
        else:
            if gift_id not in gift_index:
                errors.append(f"unknown gift id {gift_id!r}")
            elif gift_id in character.gifts:
                errors.append(f"{character.name} already has the Gift {gift_id!r}; each Gift is taken once")
    elif gift_id:
        errors.append(
            f"level {benefits['target_level']} grants no Gift; leave gift_id empty"
        )

    return errors
