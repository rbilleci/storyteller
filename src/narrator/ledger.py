"""Read the fiction-debt ledger the Model Context Protocol server persists.

Requirement ``NR-DELIVERY-GATE`` needs one fact before any turn reaches a player: does
``campaign/state.json`` still hold unratified outcomes? This module answers that and
nothing else.

The gate fails CLOSED. An unreadable or absent ledger withholds the turn rather than
delivering it. An earlier version returned an empty list on a read failure, which made a
missing state file indistinguishable from a settled one and would have delivered every
turn ungated for a whole run.

The cost is one stalled turn on a transient read failure. The alternative cost is
unratified fiction reaching the table, which is the outcome this barrier exists to
prevent, so the trade is not close.
"""

from __future__ import annotations

import json
from pathlib import Path


class LedgerUnreadable(RuntimeError):  # noqa: N818 -- renaming ripples through every `except LedgerUnreadable`
    """The ledger could not be read, so ratification cannot be established."""


def read_fiction_debt(campaign_root: Path | str) -> list[dict]:
    """Return the unratified entries. Raises ``LedgerUnreadable`` if the file is not readable.

    An earlier version returned an empty list here, which made an absent state file
    indistinguishable from a settled one and delivered every turn ungated for a whole
    run. Absence is now loud, because the barrier must fail closed.
    """
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        # ValueError covers json.JSONDecodeError and UnicodeDecodeError. Catching only
        # the former let an invalid-UTF-8 state file escape and abort the service.
        raise LedgerUnreadable(f"{state_path}: {error}") from error
    if not isinstance(payload, dict):
        # A JSON array parses cleanly and then raises AttributeError on .get, which
        # also escaped and aborted the service.
        raise LedgerUnreadable(f"{state_path}: expected an object, found {type(payload).__name__}")
    debt = payload.get("fiction_debt")
    return debt if isinstance(debt, list) else []


def is_ratified(campaign_root: Path | str) -> bool:
    """True only when the ledger is readable and empty.

    An unreadable ledger returns False, so the turn is withheld. Withholding on a read
    failure costs one stalled turn; delivering on one costs unratified fiction reaching
    the table, which is the outcome the whole barrier exists to prevent.
    """
    try:
        return not read_fiction_debt(campaign_root)
    except LedgerUnreadable:
        return False


def read_pending_rulings(campaign_root: Path | str) -> list[dict]:
    """Return the open mechanical rulings. Raises ``LedgerUnreadable`` on a bad file."""
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LedgerUnreadable(f"{state_path}: {error}") from error
    if not isinstance(payload, dict):
        raise LedgerUnreadable(f"{state_path}: expected an object, found {type(payload).__name__}")
    rulings = payload.get("pending_rulings")
    return rulings if isinstance(rulings, list) else []


def has_open_rulings(campaign_root: Path | str) -> bool:
    """True when a mechanical ruling is unresolved, or the state is unreadable.

    Fails closed on a read error, matching the ratification gate: an unreadable state
    file withholds the turn rather than delivering an unresolved obligation.
    """
    try:
        return bool(read_pending_rulings(campaign_root))
    except LedgerUnreadable:
        return True
