"""Shared test fixtures.

Every test runs against a temporary campaign root so no test ever touches the
repository's live campaign files.
"""

from __future__ import annotations

import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
try:


    import bsh_mcp  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))


sys.path.insert(0, str(REPO_ROOT / "tests_narrator"))

from bsh_mcp.dice import Roller  # noqa: E402
from bsh_mcp.service import GameService  # noqa: E402


@dataclass
class ScriptedRoller(Roller):
    """A roller that returns queued die results before falling back to chance."""

    script: list[int] = field(default_factory=list)

    def die(self, sides: int) -> int:
        if self.script:
            value = self.script.pop(0)
            if not 1 <= value <= sides:
                raise AssertionError(f"scripted value {value} cannot appear on a d{sides}")
            return value
        return super().die(sides)

    def queue(self, *values: int) -> ScriptedRoller:
        self.script.extend(values)
        return self


@pytest.fixture
def campaign_root(tmp_path: Path) -> Path:
    """A temporary project root holding a copy of rules/ and world/."""
    shutil.copytree(REPO_ROOT / "rules", tmp_path / "rules")
    world_source = REPO_ROOT / "world"
    if world_source.is_dir():
        shutil.copytree(world_source, tmp_path / "world")
    else:  # pragma: no cover - the repository always ships world content
        (tmp_path / "world" / "locations").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def roller() -> ScriptedRoller:
    return ScriptedRoller(rng=random.Random(1234))


@pytest.fixture
def service(campaign_root: Path, roller: ScriptedRoller) -> GameService:
    game = GameService(campaign_root, roller=roller)
    game.store.initialize(title="Test Campaign")
    return game


def make_character(
    service: GameService,
    roller: ScriptedRoller,
    name: str = "Mara",
    origin: str = "barbarian",
    backgrounds: tuple[str, ...] = ("scout", "hunter", "survivor"),
    *,
    attribute_totals: tuple[int, ...] = (12, 12, 12, 12, 12, 12),
    armour: str = "none",
    shield: bool = False,
    weapons: tuple[str, ...] = ("long knife",),
) -> dict:
    """Create a character with fully scripted attribute rolls.

    ``attribute_totals`` holds one 2d6 total per attribute, in STR, DEX, CON, INT,
    WIS, CHA order. A total of 12 maps to a score of 13; a total of 2 maps to 8.
    """
    for total in attribute_totals:
        first = min(6, total - 1)
        roller.queue(first, total - first)
    result = service.character_create(
        discord_user_id=f"discord-{name.lower()}",
        name=name,
        origin=origin,
        backgrounds=list(backgrounds),
        weapons=list(weapons),
        armour=armour,
        shield=shield,
    )
    assert result["ok"], result
    return result
