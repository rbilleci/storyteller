"""The one identifier-resolution precedence every resolver in this package shares.

``CampaignStore.resolve_character_id``, ``CommonMixin._resolve_npc_id``, and
``CommonMixin._resolve_combat_actor`` each resolve a narrator-supplied identifier
against a different collection (characters, NPCs, combat actors), but by the same
rule: exact id, then a unique case-folded id, then a unique case-folded display
name. This module holds that rule once, as a pure function with no knowledge of
campaign state, transactions, or error shapes -- each caller decides what a
zero-match or multi-match result means for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class IdMatch:
    """One resolution attempt's outcome.

    ``resolved`` is set only when exactly one candidate matched, by exact id or,
    failing that, by a unique case-folded id or display name. ``matches`` holds
    every candidate the fuzzy stage found (0, 1, or many) so a caller can build
    its own not-found or ambiguous error; it is empty on an exact-id match, which
    never needs one.
    """

    resolved: str | None
    matches: tuple[str, ...]


def match_id(
    requested: str, candidates: Iterable[str], display_name: Callable[[str], str]
) -> IdMatch:
    """Resolve ``requested`` against ``candidates`` by exact id, unique case-folded
    id, or unique case-folded display name.
    """
    ids = list(candidates)
    if requested in ids:
        return IdMatch(resolved=requested, matches=())
    folded = str(requested).strip().casefold()
    matches = tuple(cid for cid in ids if cid.casefold() == folded)
    if not matches:
        matches = tuple(cid for cid in ids if display_name(cid).strip().casefold() == folded)
    resolved = matches[0] if len(matches) == 1 else None
    return IdMatch(resolved=resolved, matches=matches)
