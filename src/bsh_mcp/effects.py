"""Pluggable effect registry: hook-point dispatch for character effects.

Effects live as data in ``rules/effects.json``, keyed by a namespaced id. Each
entry declares its ``source`` (the background, Gift, or subsystem that grants it),
an activation kind, an optional resource ``pool``, and a ``hooks`` map. A hook
value is an ordered list of operations; each operation names a primitive in
:data:`PRIMITIVES` and carries that primitive's parameters.

The engine calls :func:`apply` at each hook site instead of branching on
individual effects. Adding an effect is therefore a JSON entry naming existing
primitives, and only a genuinely novel mechanic costs one new primitive function
plus its test. Neither case edits the combat engine. This generalizes the
data-driven interpreter ``rules/weapon-effects.json`` already uses, giving it
named hook points across the whole service.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

#: Every hook point a primitive may target. A hook absent here fails validation,
#: so a typo cannot silently never fire.
HOOKS: frozenset[str] = frozenset(
    {
        "character_shape",  # character_create: derive weapon/unarmed dice and languages
        "initiative",  # combat_start: initiative-test edges
        "rest",  # rest: long-rest safety gate and resource resets
        "damage_dealt",  # combat_attack: add to the total on any successful hit
        "critical_damage",  # combat_attack: override the total on a critical hit
        "damage_incoming",  # combat_defend: modify damage applied to the defender
        "helpless",  # helpless_roll: roll edge and regained-HP bonus
        "attack",  # combat_attack: pre-test shaping (auto-hit, attribute substitution, flat damage bonus)
        "attribute_test",  # attribute_test/group_test: category-tagged test edges
        "helpless_care",  # helpless_roll: a carer's tending, gated on the carer's own test
        "invoke",  # use_ability: a subsystem power's own invocation-roll edges
        "backlash",  # a named consequence table's own roll (Demon's Revenge, Torn Veil)
    }
)


# ---------------------------------------------------------------------------
# Hook contexts. Each hook passes one mutable dataclass to its primitives.
# ---------------------------------------------------------------------------


@dataclass
class ShapeContext:
    """``character_shape``: the derived combat dice and languages at creation."""

    weapon_damage: str = "d6"
    unarmed_damage: str = "d4"
    languages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class InitiativeContext:
    """``initiative``: whether the initiative test rolls with Advantage."""

    advantage: bool = False


@dataclass
class RestContext:
    """``rest``: the long-rest safety gate, plus generic resolution of standing
    per-character declarations (e.g. Herbalist's dose preparation).

    ``environment_tags`` and ``declared_choices`` are ambient facts rest() hands to
    every resting character's hooks; rest() itself never inspects them or knows
    which background, if any, cares. ``granted`` is the output: rest() rolls and
    applies whatever a primitive appended here, generically, with no per-background
    branching of its own.
    """

    allow_unsafe_long_rest: bool = False
    environment_tags: list[str] = field(default_factory=list)
    declared_choices: dict[str, str] = field(default_factory=dict)
    level: int = 1
    #: Output: one entry per satisfied declaration -- {"effect_id", "dose_type",
    #: "quantity_die"} -- for rest() to roll and write generically.
    granted: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DamageContext:
    """``critical_damage``: the dealt-damage total, overridable on a critical hit."""

    total: int = 0
    attributes: object = None  # the attacker's Attributes, for attribute-scaled damage
    notes: list[str] = field(default_factory=list)


@dataclass
class DamageDealtContext:
    """``damage_dealt``: add to the total on a successful hit (e.g. Berserker rage).

    ``roller`` rolls added dice. ``end_effects`` collects the effect ids whose stance
    ends this hit (a rage die showing its end face), which the caller deactivates.
    """

    total: int = 0
    roller: object = None
    added: list[dict] = field(default_factory=list)
    end_effects: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DamageIncomingContext:
    """``damage_incoming``: the damage applied to a defender after armour.

    ``end_effects`` collects the effect ids whose stance ends on this hit (a
    one-shot defensive stance that spent itself), which the caller deactivates --
    the same output ``DamageDealtContext`` carries for the dealt side.
    """

    applied: int = 0
    armour: str = "none"
    end_effects: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class HelplessContext:
    """``helpless``: the Helpless-table roll edge and the regained-HP bonus."""

    level: int = 1
    advantage: bool = False
    hp_bonus: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class AttackContext:
    """``attack``: pre-test attack shaping in combat_attack."""

    attack_type: str = "melee"
    attribute: str = "STR"
    first_ranged_in_combat: bool = False
    one_handed_blade: bool = False
    level: int = 1
    auto_hit: bool = False
    damage_bonus: int = 0
    #: Engine fact: the target has not reacted AND this attacker has not yet
    #: spent their unaware-strike this fight. See CombatActor.reacted and
    #: CombatActor.unaware_strike_used.
    target_unaware_verified: bool = False
    #: The narrator's combat_attack target_unaware declaration.
    target_unaware_declared: bool = False
    #: Engine fact: True when this attack is unarmed (combat_attack's own
    #: ``unarmed`` parameter), False for an armed attack.
    unarmed: bool = False
    #: Engine fact: the attacker's character.weapons at attack time.
    weapons_held: list[str] = field(default_factory=list)
    #: Engine fact: the attacker's character.declared_choices at attack time
    #: (the same standing-mailbox pattern rest()'s RestContext already reads).
    declared_choices: dict[str, str] = field(default_factory=dict)
    #: Output: the attribute whose score replaces the rolled damage, or None.
    damage_override_attribute: str | None = None
    #: Output: the effect id that set damage_override_attribute, or None.
    damage_override_effect: str | None = None
    #: Output: cumulative DAMAGE_CHAIN steps to apply to whichever damage die
    #: this attack rolls (weapon or unarmed). 0 means no step (Bloodlust).
    damage_die_steps: int = 0
    #: Output: a damage die notation (e.g. "d12") that replaces the character's
    #: normal weapon/unarmed die for this attack, or None (Riddle of Steel).
    damage_die_override: str | None = None
    #: Output: the face that breaks the designated weapon when it is the kept
    #: base-component die's selected result, or None when no break is possible.
    weapon_break_face: int | None = None
    #: Output: the weapon name to remove from character.weapons on a break, or "".
    weapon_break_name: str = ""
    #: Output: the effect id that set weapon_break_name, so the service can
    #: clear the matching declared_choices entry without naming a Gift itself.
    weapon_break_effect: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class TestContext:
    """``attribute_test``: category-tagged test edges."""

    attribute: str = ""
    category: str = ""
    #: Engine fact: set only by combat_attack's to-hit roll and combat_defend's
    #: defense roll, never by a caller-supplied or narrator-declared parameter.
    in_combat: bool = False
    advantage: bool = False
    #: Output: the unmodified-roll critical-success band, widened only by an
    #: effect gated on in_combat (e.g. Battle Hardened). Default 1 is current
    #: behavior (only a natural 1 crits).
    crit_success_max: int = 1
    notes: list[str] = field(default_factory=list)


@dataclass
class HelplessCareContext:
    """``helpless_care``: what a carer's tending does, applied against the carer's sources."""

    test_attribute: str = ""
    die_sides_on_success: int = 6
    notes: list[str] = field(default_factory=list)


@dataclass
class InvokeContext:
    """``invoke``: whether a subsystem power's own invocation roll gets Advantage.

    ``power_id`` is the bare (unprefixed) power id being invoked -- e.g.
    ``ancestor_spirit``, not ``spirit:ancestor_spirit`` -- so it compares directly
    against a standing declared choice of the same shape (Spirit alliance names
    one spirit id via ``use_ability``'s ``choice``).
    """

    power_id: str = ""
    declared_choices: dict[str, str] = field(default_factory=dict)
    advantage: bool = False


@dataclass
class BacklashContext:
    """``backlash``: whether a named consequence table's own roll (Torn Veil,
    Demon's Revenge) rolls with Advantage."""

    table: str = ""
    advantage: bool = False


# ---------------------------------------------------------------------------
# Primitives. Each mutates its hook's context given the operation parameters.
# ---------------------------------------------------------------------------


def _set_weapon_damage(ctx: ShapeContext, op: dict, effect_id: str) -> None:
    ctx.weapon_damage = op["die"]


def _set_unarmed_damage(ctx: ShapeContext, op: dict, effect_id: str) -> None:
    ctx.unarmed_damage = op["die"]


def _set_unarmed_to_weapon(ctx: ShapeContext, op: dict, effect_id: str) -> None:
    ctx.unarmed_damage = ctx.weapon_damage


def _add_languages(ctx: ShapeContext, op: dict, effect_id: str) -> None:
    ctx.languages.extend([op["label"]] * int(op["count"]))


def _grant_advantage(ctx: InitiativeContext, op: dict, effect_id: str) -> None:
    ctx.advantage = True


def _allow_unsafe_long_rest(ctx: RestContext, op: dict, effect_id: str) -> None:
    ctx.allow_unsafe_long_rest = True


def _resolve_declared_preparation(ctx: RestContext, op: dict, effect_id: str) -> None:
    choice = ctx.declared_choices.get(effect_id, "")
    if not choice or choice not in op["types"]:
        return
    required_tag = op.get("required_environment_tag")
    if required_tag and required_tag not in ctx.environment_tags:
        ctx.notes.append(
            f"a declared {choice.replace('_', ' ')} preparation needs a scene tagged "
            f"{required_tag!r}; nothing was prepared this rest"
        )
        return
    ctx.granted.append(
        {"effect_id": effect_id, "dose_type": choice, "quantity_die": op["quantity_die"]}
    )


def _set_total_to_attribute(ctx: DamageContext, op: dict, effect_id: str) -> None:
    ctx.total = ctx.attributes.get(op["attribute"])


def _add_damage_die(ctx: DamageDealtContext, op: dict, effect_id: str) -> None:
    rolled = ctx.roller.notation(op["die"])
    ctx.total += rolled
    ctx.added.append({"effect": effect_id, "die": op["die"], "rolled": rolled})
    if "end_on_face" in op and rolled == int(op["end_on_face"]):
        ctx.end_effects.append(effect_id)


def _flat_reduce_incoming(ctx: DamageIncomingContext, op: dict, effect_id: str) -> None:
    if op.get("when_no_armour") and ctx.armour != "none":
        return
    ctx.applied = max(0, ctx.applied - int(op["amount"]))


def _scale_incoming(ctx: DamageIncomingContext, op: dict, effect_id: str) -> None:
    scaled = ctx.applied * float(op["factor"])
    ctx.applied = int(scaled)  # floor; halving 5 lands at 2


def _nullify_incoming(ctx: DamageIncomingContext, op: dict, effect_id: str) -> None:
    """A one-shot defensive stance: ignore this attack's damage entirely and spend
    itself doing it. A no-op when nothing would be applied, so a stance is never
    burned on an attack that was already going to do nothing."""
    if ctx.applied <= 0:
        return
    ctx.applied = 0
    ctx.end_effects.append(effect_id)


def _grant_helpless_advantage(ctx: HelplessContext, op: dict, effect_id: str) -> None:
    ctx.advantage = True


def _add_level_to_regained_hp(ctx: HelplessContext, op: dict, effect_id: str) -> None:
    ctx.hp_bonus += ctx.level


def _grant_auto_hit(ctx: AttackContext, op: dict, effect_id: str) -> None:
    if op.get("when_first_ranged_in_combat") and not ctx.first_ranged_in_combat:
        return
    ctx.auto_hit = True


def _add_level_to_damage(ctx: AttackContext, op: dict, effect_id: str) -> None:
    if op.get("when_first_ranged_in_combat") and not ctx.first_ranged_in_combat:
        return
    ctx.damage_bonus += ctx.level


def _substitute_attack_attribute(ctx: AttackContext, op: dict, effect_id: str) -> None:
    if op.get("when_one_handed_blade") and not ctx.one_handed_blade:
        return
    if ctx.attribute == op["from"]:
        ctx.attribute = op["to"]


def _override_damage_with_attribute(ctx: AttackContext, op: dict, effect_id: str) -> None:
    if op.get("when_target_unaware"):
        if not ctx.target_unaware_verified:
            return
        if not ctx.target_unaware_declared:
            ctx.notes.append(
                "the target has not reacted this fight and you have not yet spent your "
                "unaware strike; if the fiction holds them unaware, declare target_unaware "
                "on this attack"
            )
            return
    ctx.damage_override_attribute = op["attribute"]
    ctx.damage_override_effect = effect_id


def _step_up_damage_die(ctx: AttackContext, op: dict, effect_id: str) -> None:
    """Bloodlust: accumulate a permanent damage-die step. Fires for every attack
    the holder makes (armed or unarmed); combat_attack decides which die to step."""
    ctx.damage_die_steps += int(op.get("steps", 1))


def _override_damage_die(ctx: AttackContext, op: dict, effect_id: str) -> None:
    """Riddle of Steel: override the damage die on a currently-held designated
    weapon. A no-op when unarmed, or when the effect's own standing declaration
    does not name a weapon the character currently holds."""
    if ctx.unarmed:
        return
    if op.get("when_declared_weapon_held"):
        declared = ctx.declared_choices.get(effect_id, "")
        if not declared or declared not in ctx.weapons_held:
            return
        ctx.weapon_break_name = declared
        ctx.weapon_break_effect = effect_id
    ctx.damage_die_override = op["die"]
    if "break_on_face" in op:
        ctx.weapon_break_face = int(op["break_on_face"])


def _grant_category_advantage(ctx: TestContext, op: dict, effect_id: str) -> None:
    if ctx.category and ctx.category in op.get("categories", []):
        ctx.advantage = True


def _widen_critical_success(ctx: TestContext, op: dict, effect_id: str) -> None:
    if op.get("when_in_combat") and not ctx.in_combat:
        return
    ctx.crit_success_max = max(ctx.crit_success_max, int(op["max_face"]))


def _tend_helpless(ctx: HelplessCareContext, op: dict, effect_id: str) -> None:
    ctx.test_attribute = op["test_attribute"]
    ctx.die_sides_on_success = int(op["die_sides"])


def _grant_invocation_advantage(ctx: InvokeContext, op: dict, effect_id: str) -> None:
    """Spirit alliance: Advantage on invoking the one power this effect's own
    standing declaration names. A no-op while undeclared, or while invoking any
    other power."""
    if ctx.declared_choices.get(effect_id) == ctx.power_id:
        ctx.advantage = True


def _grant_backlash_advantage(ctx: BacklashContext, op: dict, effect_id: str) -> None:
    if ctx.table and ctx.table in op.get("tables", []):
        ctx.advantage = True


#: The primitive dispatch table. An operation names one of these; the loader
#: rejects any effect naming a primitive absent here.
PRIMITIVES: dict[str, Callable[[object, dict, str], None]] = {
    "set_weapon_damage": _set_weapon_damage,
    "set_unarmed_damage": _set_unarmed_damage,
    "set_unarmed_to_weapon": _set_unarmed_to_weapon,
    "add_languages": _add_languages,
    "grant_advantage": _grant_advantage,
    "allow_unsafe_long_rest": _allow_unsafe_long_rest,
    "resolve_declared_preparation": _resolve_declared_preparation,
    "set_total_to_attribute": _set_total_to_attribute,
    "add_damage_die": _add_damage_die,
    "flat_reduce_incoming": _flat_reduce_incoming,
    "scale_incoming": _scale_incoming,
    "nullify_incoming": _nullify_incoming,
    "grant_helpless_advantage": _grant_helpless_advantage,
    "add_level_to_regained_hp": _add_level_to_regained_hp,
    "grant_auto_hit": _grant_auto_hit,
    "add_level_to_damage": _add_level_to_damage,
    "substitute_attack_attribute": _substitute_attack_attribute,
    "override_damage_with_attribute": _override_damage_with_attribute,
    "step_up_damage_die": _step_up_damage_die,
    "override_damage_die": _override_damage_die,
    "grant_category_advantage": _grant_category_advantage,
    "widen_critical_success": _widen_critical_success,
    "tend_helpless": _tend_helpless,
    "grant_invocation_advantage": _grant_invocation_advantage,
    "grant_backlash_advantage": _grant_backlash_advantage,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def source_ids(character) -> set[str]:
    """Return the effect-source keys a character's backgrounds and Gifts grant."""
    sources = {f"background:{bid}" for bid in getattr(character, "backgrounds", [])}
    sources |= {f"gift:{gid}" for gid in getattr(character, "gifts", [])}
    return sources


def background_sources(background_ids: list[str]) -> set[str]:
    """Return the effect-source keys for a list of background ids (pre-creation)."""
    return {f"background:{bid}" for bid in background_ids}


def apply(
    hook: str, ctx: object, effects: dict, sources: set[str],
    active_ids: frozenset[str] = frozenset(),
) -> object:
    """Run every operation the active effects register for ``hook`` against ``ctx``.

    An effect fires when the character's ``sources`` grant it. A ``toggle``,
    ``targeted``, or ``resource`` effect fires only while its stance is active, so
    its id must also appear in ``active_ids`` (the effect ids of the character's
    live conditions) -- a ``resource`` effect that declares no ``hooks`` is
    unaffected, since it registers no operations at any hook. A ``passive`` effect
    ignores ``active_ids``. Operations run in ``(priority, effect_id)`` order,
    priority defaulting to 100, so composition is declared by number rather than
    dictionary insertion order. Returns ``ctx``.
    """
    registry = effects.get("effects", {})
    pending: list[tuple[int, str, dict]] = []
    for effect_id, effect in registry.items():
        if effect.get("source") not in sources:
            continue
        if (
            effect.get("activation") in ("toggle", "targeted", "resource")
            and effect_id not in active_ids
        ):
            continue
        for op in effect.get("hooks", {}).get(hook, []):
            pending.append((int(op.get("priority", 100)), effect_id, op))
    pending.sort(key=lambda item: (item[0], item[1]))
    for _, effect_id, op in pending:
        PRIMITIVES[op["op"]](ctx, op, effect_id)
    return ctx


def active_ids(character) -> frozenset[str]:
    """Return the effect ids of a character's live conditions (activated stances)."""
    return frozenset(c.effect_id for c in getattr(character, "conditions", []) if c.effect_id)


def effect_entry(effects: dict, effect_id: str) -> dict | None:
    """Return one effect's registry entry, or None when it is not registered."""
    return effects.get("effects", {}).get(effect_id)
