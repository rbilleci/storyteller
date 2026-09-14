"""Game service: every Black Sword Hack tool behind the Model Context Protocol.

Each public method mirrors one Model Context Protocol (MCP) tool exactly and
returns the standard envelope defined in :mod:`bsh_mcp.results`. The server
module wires these methods to MCP; the methods themselves never touch MCP, so
tests call them directly.

``GameService`` is composed from one mixin per concern (trade, abilities,
attribute tests, characters, combat, recovery, scene, status, session) plus
``CommonMixin`` for the cross-cutting helpers every other mixin calls through
``self``. Every mixin method is called as ``self.method_name(...)``, so
Python's MRO resolves a cross-mixin call (e.g. ``reserve_social_ability`` in
``TradeMixin`` calling ``self.use_ability(...)``, defined in
``AbilitiesMixin``) with no special handling.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..data import RulesData, load_rules_data
from ..dice import Roller
from ..store import CampaignStore
from .abilities import AbilitiesMixin
from .attribute_tests import AttributeTestsMixin
from .characters import CharactersMixin
from .combat import CombatMixin
from .common import CommonMixin
from .recovery import RecoveryMixin
from .scene import SceneMixin
from .scene import resolve_person_label as resolve_person_label
from .session import SessionMixin
from .status import StatusMixin
from .trade import TradeMixin


class GameService(
    TradeMixin,
    AbilitiesMixin,
    AttributeTestsMixin,
    CharactersMixin,
    CombatMixin,
    RecoveryMixin,
    SceneMixin,
    StatusMixin,
    SessionMixin,
    CommonMixin,
):
    """Deterministic Black Sword Hack mechanics over one campaign directory."""

    def __init__(
        self,
        root: Path | str,
        roller: Roller | None = None,
        clock: Callable[[], datetime] | None = None,
        data_loader: Callable[[str], RulesData] = load_rules_data,
    ):
        self.store = CampaignStore(root, clock=clock)
        self.roller = roller or Roller()
        self._purchase_capability = object()
        #: Defaults to the process-wide ``lru_cache``d loader, so a `rules/*.json`
        #: edit is invisible to a running process until restart -- true before this
        #: parameter existed and still true by default. A caller that needs a fresh
        #: read (a test proving a rules edit takes effect, a hot-reload tool)
        #: injects an uncached one, e.g. ``lambda rules_dir: RulesData.load(Path(rules_dir))``.
        self._data_loader = data_loader

    # -- helpers -------------------------------------------------------------

    @property
    def data(self) -> RulesData:
        return self._data_loader(str(self.store.rules_dir))


def build_service(root: Path | str, seed: int | None = None) -> GameService:
    roller = Roller.seeded(seed) if seed is not None else Roller()
    return GameService(root, roller=roller)
