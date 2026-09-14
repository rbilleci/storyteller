"""Read-only player-to-character resolution for structured decisions.

Campaign links remain the authority for identity.  This module intentionally exposes
only character IDs and display names to the planner, while the coordinator keeps the
adapter subject binding private.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from narrator.decisions import DecisionValidationError, semantic_fingerprint
from narrator.identifiers import IDENTIFIER


class PlayerDirectoryError(RuntimeError):
    """The campaign player directory cannot safely authorize a decision."""


@dataclass(frozen=True)
class DirectoryPlayer:
    """One private authorization binding and its planner-safe character projection."""

    adapter_name: str
    subject_id: str
    character_id: str
    character_name: str


@dataclass(frozen=True)
class PlayerDirectorySnapshot:
    """An immutable point-in-time player directory projection."""

    players: tuple[DirectoryPlayer, ...]
    fingerprint: str

    @property
    def eligible_characters(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.character_id, item.character_name) for item in self.players)

    def for_principal(self, adapter_name: str, subject_id: str) -> DirectoryPlayer | None:
        for item in self.players:
            if item.subject_id != subject_id:
                continue
            if item.adapter_name == "terminal" and adapter_name != "terminal":
                return None
            if item.adapter_name == "campaign":
                return replace(item, adapter_name=adapter_name)
            if item.adapter_name == adapter_name:
                return item
        return None

    def for_character(self, character_id: str, adapter_name: str) -> DirectoryPlayer | None:
        for item in self.players:
            if item.character_id == character_id:
                if item.adapter_name == "terminal" and adapter_name != "terminal":
                    return None
                return (
                    replace(item, adapter_name=adapter_name)
                    if item.adapter_name == "campaign"
                    else item
                )
        return None


class PlayerDirectory:
    """Load ``campaign/players.yaml`` without mutating campaign files or locks."""

    def __init__(self, campaign_root: Path | str) -> None:
        self.campaign_root = Path(campaign_root)

    @property
    def _players_path(self) -> Path:
        return self.campaign_root / "campaign" / "players.yaml"

    def snapshot(self) -> PlayerDirectorySnapshot:
        """Read one complete directory snapshot and reject duplicate or malformed links."""
        try:
            raw = self._players_path.read_bytes()
        except OSError as error:
            raise PlayerDirectoryError("player directory is unreadable") from error
        try:
            import yaml

            payload = yaml.safe_load(raw) or {}
        except Exception as error:  # noqa: BLE001 - YAML errors must fail authorization
            raise PlayerDirectoryError("player directory is invalid") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("players", []), list):
            raise PlayerDirectoryError("player directory is invalid")
        players: list[DirectoryPlayer] = []
        account_ids: set[str] = set()
        character_ids: set[str] = set()
        for link in payload["players"]:
            if not isinstance(link, dict):
                raise PlayerDirectoryError("player directory is invalid")
            account_id = link.get("discord_user_id")
            character_id = link.get("character_id")
            display_name = link.get("display_name")
            if not isinstance(account_id, str) or not isinstance(character_id, str):
                raise PlayerDirectoryError("player directory is invalid")
            try:
                if not IDENTIFIER.fullmatch(character_id):
                    raise DecisionValidationError("invalid character identifier")
                if not account_id.strip() or len(account_id) > 256:
                    raise DecisionValidationError("invalid account identifier")
                character_id = str(character_id)
                if not display_name:
                    display_name = self._character_name(character_id)
                if not isinstance(display_name, str):
                    raise DecisionValidationError("invalid display name")
                if not display_name.strip() or len(display_name) > 240:
                    raise DecisionValidationError("invalid display name")
                binding = DirectoryPlayer(
                    adapter_name="terminal" if account_id == "terminal-player" else "campaign",
                    subject_id=account_id,
                    character_id=character_id,
                    character_name=display_name.strip() or character_id,
                )
            except (DecisionValidationError, OSError, ValueError) as error:
                raise PlayerDirectoryError("player directory is invalid") from error
            if account_id in account_ids or character_id in character_ids:
                raise PlayerDirectoryError("player directory repeats a link")
            account_ids.add(account_id)
            character_ids.add(character_id)
            players.append(binding)
        fingerprint = sha256(raw).hexdigest()
        return PlayerDirectorySnapshot(players=tuple(players), fingerprint=fingerprint)

    def _character_name(self, character_id: str) -> str:
        path = self.campaign_root / "campaign" / "characters" / f"{character_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return character_id
        name = payload.get("name") if isinstance(payload, dict) else None
        return name if isinstance(name, str) and name.strip() else character_id

    def context_fingerprint(
        self,
        *,
        snapshot: PlayerDirectorySnapshot,
        original_turn: str,
        prior_resolutions: tuple,
    ) -> str:
        """Fingerprint canon, links, original turn, resolutions, and event sequence."""
        scene_path = self.campaign_root / "campaign" / "scene.md"
        state_path = self.campaign_root / "campaign" / "state.json"
        try:
            scene_digest = sha256(scene_path.read_bytes()).hexdigest()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            event_seq = state.get("event_seq") if isinstance(state, dict) else None
            if not isinstance(event_seq, int) or isinstance(event_seq, bool) or event_seq < 0:
                raise ValueError("event sequence")
        except (OSError, ValueError, UnicodeError) as error:
            raise PlayerDirectoryError("campaign context is unreadable") from error
        return semantic_fingerprint(
            {
                "canon": scene_digest,
                "player_directory": snapshot.fingerprint,
                "original_turn": original_turn,
                "prior_resolutions": [
                    resolution.model_dump() for resolution in prior_resolutions
                ],
                "campaign_event_sequence": event_seq,
            }
        )
