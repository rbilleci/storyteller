"""Deterministic dice primitives for Black Sword Hack.

Every random draw in this project passes through :class:`Roller`. Tools inject a
seeded :class:`random.Random` during tests, so every rule is reproducible.

Reference: Black Sword Hack - Ultimate Chaos Edition SRD v1.0.2, "Rules".
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

#: Usage Die chain. A Ud4 that downgrades becomes :data:`DEPLETED`.
USAGE_CHAIN: tuple[str, ...] = ("d20", "d12", "d10", "d8", "d6", "d4")

#: Sentinel stored in campaign files when a Usage Die is used up.
DEPLETED = "depleted"

#: Selection mode produced by combining Advantage and Disadvantage.
Edge = Literal["single", "advantage", "disadvantage"]

_DIE_PATTERN = re.compile(r"^d(\d+)$")


class DiceError(ValueError):
    """Raised when a die notation is not a supported Black Sword Hack die."""


def die_sides(die: str) -> int:
    """Return the face count of a ``dN`` notation string."""
    match = _DIE_PATTERN.match(die.strip().lower())
    if match is None:
        raise DiceError(f"unsupported die notation: {die!r}")
    sides = int(match.group(1))
    if sides < 2:
        raise DiceError(f"unsupported die notation: {die!r}")
    return sides


def step_down(die: str) -> str:
    """Return the next Usage Die below ``die``.

    ``d4`` steps down to :data:`DEPLETED`. :data:`DEPLETED` stays depleted.
    """
    if die == DEPLETED:
        return DEPLETED
    if die not in USAGE_CHAIN:
        raise DiceError(f"{die!r} is not a Usage Die: expected one of {USAGE_CHAIN}")
    index = USAGE_CHAIN.index(die)
    if index == len(USAGE_CHAIN) - 1:
        return DEPLETED
    return USAGE_CHAIN[index + 1]


#: Damage die step-up chain. Distinct from :data:`USAGE_CHAIN`: it runs the
#: opposite direction (upgrade, not deplete), has different membership (no
#: ``d20``), and never depletes — it clamps at its top member instead.
DAMAGE_CHAIN: tuple[str, ...] = ("d4", "d6", "d8", "d10", "d12")


def step_up(die: str, steps: int = 1) -> str:
    """Return ``die`` stepped up ``steps`` places along :data:`DAMAGE_CHAIN`.

    Clamps at ``d12``, the chain's top member, rather than raising or wrapping.
    Raises :class:`DiceError` when ``die`` is not a member of
    :data:`DAMAGE_CHAIN` (this is a step-up chain for damage dice, not the
    Usage Die depletion chain — a ``d20`` Usage Die is not a valid input here).
    """
    if die not in DAMAGE_CHAIN:
        raise DiceError(f"{die!r} is not a damage die: expected one of {DAMAGE_CHAIN}")
    index = DAMAGE_CHAIN.index(die)
    return DAMAGE_CHAIN[min(index + steps, len(DAMAGE_CHAIN) - 1)]


def resolve_edge(advantage: bool, disadvantage: bool) -> Edge:
    """Combine Advantage and Disadvantage. Both together cancel to one die."""
    if advantage and disadvantage:
        return "single"
    if advantage:
        return "advantage"
    if disadvantage:
        return "disadvantage"
    return "single"


def select_die(dice: list[int], edge: Edge, prefer_low: bool) -> int:
    """Pick the die the rules keep.

    ``prefer_low`` is ``True`` for roll-under attribute tests, where a low
    number is favourable. It is ``False`` for damage and Usage Die rolls, where
    a high number is favourable.
    """
    if not dice:
        raise DiceError("cannot select from an empty dice pool")
    if edge == "single":
        return dice[0]
    favourable = min if prefer_low else max
    unfavourable = max if prefer_low else min
    chooser = favourable if edge == "advantage" else unfavourable
    return chooser(dice)


@dataclass(frozen=True)
class Roll:
    """One recorded roll. Serialised verbatim into tool results and audit events."""

    notation: str
    dice: list[int]
    selected: int
    modifier: int = 0
    total: int = 0
    target: int | None = None
    edge: Edge = "single"
    note: str = ""

    def as_dict(self) -> dict:
        payload = {
            "notation": self.notation,
            "dice": list(self.dice),
            "selected": self.selected,
            "modifier": self.modifier,
            "total": self.total,
            "edge": self.edge,
        }
        if self.target is not None:
            payload["target"] = self.target
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class Roller:
    """Injectable random source.

    Every mechanics function receives a ``Roller`` rather than calling the
    :mod:`random` module directly, so tests replay exact sequences.
    """

    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def seeded(cls, seed: int) -> Roller:
        return cls(rng=random.Random(seed))

    def die(self, sides: int) -> int:
        if sides < 2:
            raise DiceError(f"a die needs at least 2 faces, got {sides}")
        return self.rng.randint(1, sides)

    def pool(self, count: int, sides: int) -> list[int]:
        return [self.die(sides) for _ in range(count)]

    def notation(self, die: str) -> int:
        """Roll one die given ``dN`` notation."""
        return self.die(die_sides(die))

    def choice(self, items: list):
        return self.rng.choice(items)

    def d20(self, edge: Edge = "single", modifier: int = 0, target: int | None = None) -> Roll:
        """Roll the roll-under d20 test pool. Advantage keeps the lower die."""
        count = 1 if edge == "single" else 2
        dice = self.pool(count, 20)
        selected = select_die(dice, edge, prefer_low=True)
        return Roll(
            notation=f"{count}d20",
            dice=dice,
            selected=selected,
            modifier=modifier,
            total=selected + modifier,
            target=target,
            edge=edge,
        )

    def damage(self, die: str, edge: Edge = "single") -> Roll:
        """Roll damage. Advantage keeps the higher die (two-handed weapons)."""
        sides = die_sides(die)
        count = 1 if edge == "single" else 2
        dice = self.pool(count, sides)
        selected = select_die(dice, edge, prefer_low=False)
        return Roll(
            notation=f"{count}{die}",
            dice=dice,
            selected=selected,
            total=selected,
            edge=edge,
        )

    def usage(self, die: str, edge: Edge = "single") -> Roll:
        """Roll a Usage Die. Advantage keeps the higher die, away from 1 and 2."""
        sides = die_sides(die)
        count = 1 if edge == "single" else 2
        dice = self.pool(count, sides)
        selected = select_die(dice, edge, prefer_low=False)
        return Roll(
            notation=f"{count}{die}",
            dice=dice,
            selected=selected,
            total=selected,
            edge=edge,
        )

    def backlash(self, sides: int, edge: Edge = "single") -> Roll:
        """Roll a high-is-worse consequence table (Helpless, Demon's Revenge,
        Torn Veil). Advantage keeps the lower, safer die."""
        count = 1 if edge == "single" else 2
        dice = self.pool(count, sides)
        selected = select_die(dice, edge, prefer_low=True)
        return Roll(
            notation=f"{count}d{sides}",
            dice=dice,
            selected=selected,
            total=selected,
            edge=edge,
        )
