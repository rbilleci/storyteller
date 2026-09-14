"""Campaign file storage: path confinement, one lock, atomic writes, audit log.

The Model Context Protocol (MCP) server is the only component permitted to
mutate mechanical campaign state. Every mutation passes through
:meth:`CampaignStore.transaction`, which acquires one campaign lock, loads and
validates current data, applies the caller's changes, revalidates, writes each
file atomically, and appends one audit event.

If any step raises, the transaction writes nothing at all.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from filelock import FileLock, Timeout
from pydantic import ValidationError

from .models import (
    CampaignState,
    Character,
    FictionDebt,
    Manifest,
    PlayersFile,
    StakesDeclaration,
    is_valid_id,
)
from .resolution import match_id

if TYPE_CHECKING:
    from .models import ScenePerson

LOCK_TIMEOUT_SECONDS = 20.0

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


class CampaignError(Exception):
    """A recoverable, structured failure returned to the narrator."""

    def __init__(self, code: str, message: str, allowed_next_steps: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.allowed_next_steps = allowed_next_steps or []

    def as_dict(self) -> dict:
        return {
            "ok": False,
            "error": self.code,
            "message": self.message,
            "allowed_next_steps": self.allowed_next_steps,
        }


def resolve_or_raise(
    requested: str,
    candidates: Iterable[str],
    display_name: Callable[[str], str],
    *,
    entity: str,
    entity_plural: str,
    code_prefix: str,
    extra_steps: list[str] | None = None,
) -> str:
    """:func:`bsh_mcp.resolution.match_id`, turning a zero- or multi-match result
    into a ``CampaignError`` naming ``entity``/``entity_plural`` -- the "valid ids"
    error shape ``resolve_character_id`` and ``CommonMixin._resolve_npc_id`` share.
    ``_resolve_combat_actor`` does not: an unmatched combat actor is "not part of
    this combat", a different concept from "no such id" with a different next
    step (the turn order, not a full id list), so it calls ``match_id`` directly
    instead of this wrapper.
    """
    ids = sorted(candidates)
    match = match_id(requested, ids, display_name)
    if match.resolved is not None:
        return match.resolved
    valid = ", ".join(ids) or "none yet"
    if match.matches:
        raise CampaignError(
            f"{code_prefix}_ambiguous",
            f"{requested!r} matches {len(match.matches)} {entity_plural}: "
            f"{', '.join(sorted(match.matches))}.",
            [f"Use one exact id. Valid {entity_plural} ids: {valid}."] + (extra_steps or []),
        )
    raise CampaignError(
        f"{code_prefix}_not_found",
        f"No {entity} has id {requested!r}.",
        [f"Valid {entity_plural} ids: {valid}."] + (extra_steps or []),
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` so a reader never observes a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _append_line(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Transaction:
    """Mutable working copy of campaign state held under the campaign lock."""

    def __init__(self, store: CampaignStore, tool: str, actor_id: str, reason: str):
        self.store = store
        self.tool = tool
        self.actor_id = actor_id
        # Strip once here so every tool's audit reason is normalised in one place.
        self.reason = str(reason).strip()
        self.state: CampaignState = store.read_state()
        self._characters: dict[str, Character] = {}
        self._dirty_characters: set[str] = set()
        self._players: PlayersFile | None = None
        self._players_dirty = False
        self._manifest: Manifest | None = None
        self._manifest_dirty = False
        self.changes: list[str] = []
        self.warnings: list[str] = []
        self.extra_events: list[dict] = []
        self.scene_dirty = False
        # The opening combat state, frozen for comparison at commit. No combat tool
        # sets ``scene_dirty``, so without this snapshot the rendered Combat section
        # would report the fight that ran when some unrelated tool last touched the
        # scene. ``grep -n "scene_dirty" src/bsh_mcp/service.py`` names every writer.
        self._initial_combat = self.state.combat.model_dump_json()
        self.event_seq: int | None = None
        self._staged_files: list[tuple[Path, str, bool]] = []
        self._pending_debt: list[dict] = []

    # -- accessors -----------------------------------------------------------

    def character(self, character_id: str) -> Character:
        """Load a character for mutation, resolving the id first.

        The exact-id cache check runs before resolution so a character staged
        through ``add_character`` stays reachable before its file exists. The
        cache keys by canonical id, so one transaction that names a character
        two ways still mutates one object.
        """
        if character_id in self._characters:
            return self._characters[character_id]
        resolved = self.store.resolve_character_id(character_id)
        if resolved != character_id:
            note = (
                f"resolved character id {character_id!r} to {resolved!r}; "
                f"use {resolved!r} in later calls."
            )
            if note not in self.warnings:
                self.warn(note)
        if resolved in self._characters:
            return self._characters[resolved]
        character = self.store.read_character(resolved)
        self._characters[resolved] = character
        return character

    def touch_character(self, character_id: str) -> None:
        self._dirty_characters.add(character_id)

    def add_character(self, character: Character) -> None:
        """Register a freshly created character for writing at commit time."""
        self._characters[character.id] = character
        self._dirty_characters.add(character.id)

    def loaded_characters(self) -> dict[str, Character]:
        return dict(self._characters)

    @property
    def players(self) -> PlayersFile:
        if self._players is None:
            self._players = self.store.read_players()
        return self._players

    def touch_players(self) -> None:
        self._players_dirty = True

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = self.store.read_manifest()
        return self._manifest

    def touch_manifest(self) -> None:
        self._manifest_dirty = True

    def stage_file(self, path: Path, content: str, *, only_if_absent: bool = False) -> None:
        """Queue an advisory file write for commit time.

        A tool must never write a file before the transaction commits. A later
        step can still raise, and a summary left on disk claims work that never
        happened.
        """
        self._staged_files.append((Path(path), content, only_if_absent))

    def add_fiction_debt(
        self,
        *,
        tool: str,
        actor_id: str = "",
        reason: str = "",
        outcome: str = "",
        stakes: StakesDeclaration | None = None,
        realized: str = "none",
    ) -> None:
        """Queue one unratified-fiction entry for this transaction's event.

        commit() stamps the entry with the event sequence it belongs to, so
        the ledger and the audit log always agree on which roll produced
        which stake.
        """
        self._pending_debt.append(
            {
                "tool": tool,
                "actor_id": actor_id,
                "reason": reason,
                "outcome": outcome,
                "stakes": stakes or StakesDeclaration(),
                "realized": realized,
            }
        )

    def record(self, change: str) -> None:
        self.changes.append(change)

    def warn(self, warning: str) -> None:
        self.warnings.append(warning)

    # -- commit --------------------------------------------------------------

    def commit(self, event_payload: dict) -> int:
        """Validate everything, then write every dirty file, then append the audit event.

        Two passes on purpose: the first only validates and serializes, touching no
        file; the second only writes, from data the first pass already proved valid.
        A raise during validation (a model rejecting a value, a render building bad
        markdown) now genuinely writes nothing, matching this module's own documented
        invariant -- collapsing the two passes into one meant a state-validation
        failure could leave already-written character/player/manifest files on disk
        with no matching state.json update and no audit event to explain them.
        """
        self.state.event_seq += 1
        sequence = self.state.event_seq
        self.event_seq = sequence
        # One clock read for the whole commit, not one per timestamp below: every
        # file this commit touches should carry the same instant, not four
        # independently-read moments a slow validation pass could let drift apart.
        timestamp = self.store.clock().isoformat(timespec="seconds")
        self.state.updated_at = timestamp

        if self._pending_debt:
            # The rendered scene must track the ledger: a stakes-carrying roll
            # with no scene_commit is exactly the flow the Unratified outcomes
            # section exists to expose, so the file re-renders with the entry.
            self.scene_dirty = True
        for pending in self._pending_debt:
            self.state.fiction_debt = self.state.fiction_debt + [
                FictionDebt(seq=sequence, **pending)
            ]
        self._pending_debt = []

        # -- pass 1: validate and serialize; write nothing yet -------------------

        writes: list[tuple[Path, str]] = []

        for character_id in sorted(self._dirty_characters):
            character = self._characters[character_id]
            character.updated_at = timestamp
            validated = Character.model_validate(character.model_dump())
            writes.append((
                self.store.character_path(character_id),
                json.dumps(validated.model_dump(), indent=2, ensure_ascii=False) + "\n",
            ))

        if self._players_dirty and self._players is not None:
            validated_players = PlayersFile.model_validate(self._players.model_dump())
            writes.append((
                self.store.players_path,
                yaml.safe_dump(validated_players.model_dump(), sort_keys=False, allow_unicode=True),
            ))

        if self._manifest_dirty and self._manifest is not None:
            self._manifest.updated_at = timestamp
            validated_manifest = Manifest.model_validate(self._manifest.model_dump())
            writes.append((
                self.store.manifest_path,
                yaml.safe_dump(validated_manifest.model_dump(), sort_keys=False, allow_unicode=True),
            ))

        validated_state = CampaignState.model_validate(self.state.model_dump())
        if validated_state.combat.model_dump_json() != self._initial_combat:
            # The rendered scene must track the fight for the same reason it tracks the
            # ledger: the narrator and the turn classifier read the render, never
            # ``state.json``, so an unrefreshed section reports a stale round.
            self.scene_dirty = True
        writes.append((
            self.store.state_path,
            json.dumps(validated_state.model_dump(), indent=2, ensure_ascii=False) + "\n",
        ))

        if self.scene_dirty:
            writes.append((
                self.store.scene_path, self.store.render_scene_markdown(validated_state)
            ))

        for path, content, only_if_absent in self._staged_files:
            if only_if_absent and path.exists():
                continue
            writes.append((path, content))

        event = {
            "seq": sequence,
            "at": timestamp,
            "tool": self.tool,
            "actor_id": self.actor_id,
            "reason": self.reason,
            "changes": list(self.changes),
        }
        event.update(event_payload)
        event_lines = [json.dumps(event, ensure_ascii=False)]
        event_lines += [json.dumps(extra, ensure_ascii=False) for extra in self.extra_events]

        # -- pass 2: every validation above succeeded; now write -----------------

        for path, content in writes:
            _atomic_write_text(path, content)
        for line in event_lines:
            _append_line(self.store.events_path, line)

        return sequence


#: How many ``Scene.persons`` the render lists, newest mention first. The digest's
#: scene block keeps this section resident at every depth, so it must stay small.
PERSONS_RENDER_CAP = 8


class CampaignStore:
    """File-backed campaign root. Holds no state between calls."""

    def __init__(self, root: Path | str, clock: Callable[[], datetime] | None = None):
        #: The one time source every audit/backup timestamp reads. A test
        #: substitutes a fixed instant; production leaves the default, one fresh
        #: UTC reading per call. No wrapping class: unlike ``Roller``, which
        #: threads a stateful ``random.Random`` stream, "what instant is it" holds
        #: no state between calls, so a bare callable is the right-sized seam.
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self.root = Path(root).expanduser().resolve()
        self.campaign_dir = self.root / "campaign"
        self.characters_dir = self.campaign_dir / "characters"
        self.logs_dir = self.campaign_dir / "logs"
        self.summaries_dir = self.campaign_dir / "summaries"
        self.backups_dir = self.campaign_dir / "backups"
        self.rules_dir = self.root / "rules"
        self.world_dir = self.root / "world"
        self.state_path = self.campaign_dir / "state.json"
        self.players_path = self.campaign_dir / "players.yaml"
        self.manifest_path = self.campaign_dir / "manifest.yaml"
        self.scene_path = self.campaign_dir / "scene.md"
        self.events_path = self.logs_dir / "events.jsonl"
        self.lock_path = self.campaign_dir / ".campaign.lock"

    # -- paths ---------------------------------------------------------------

    def character_path(self, character_id: str) -> Path:
        """Return the sheet path for a validated identifier.

        The identifier grammar excludes separators and dots, so a caller cannot
        traverse outside ``campaign/characters/``. The resolved path is checked
        a second time.
        """
        if not is_valid_id(character_id):
            raise CampaignError(
                "invalid_character_id",
                f"{character_id!r} is not a valid character id. "
                "Use lower-case letters, digits, hyphen, and underscore.",
                ["Call campaign_status to list valid character ids."],
            )
        path = (self.characters_dir / f"{character_id}.json").resolve()
        if not str(path).startswith(str(self.characters_dir.resolve()) + os.sep):
            raise CampaignError(
                "path_outside_campaign",
                f"resolved path for {character_id!r} escapes the campaign directory",
            )
        return path

    def location_path(self, location_id: str) -> Path:
        if not is_valid_id(location_id):
            raise CampaignError(
                "invalid_location_id",
                f"{location_id!r} is not a valid location id.",
                ["Call campaign_status to list known location ids."],
            )
        locations_dir = (self.world_dir / "locations").resolve()
        path = (locations_dir / f"{location_id}.md").resolve()
        if not str(path).startswith(str(locations_dir) + os.sep):
            raise CampaignError(
                "path_outside_campaign",
                f"resolved path for {location_id!r} escapes the world directory",
            )
        return path

    # -- reads ---------------------------------------------------------------

    def _read_json(self, path: Path, model, code: str):
        if not path.is_file():
            raise CampaignError(
                "campaign_not_initialised",
                f"missing campaign file {path.name}. The campaign directory is not initialised.",
                ["Run scripts/new_campaign.py to create the campaign files."],
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise CampaignError(code, f"{path.name} contains invalid JSON: {error}") from error
        try:
            return model.model_validate(payload)
        except ValidationError as error:
            raise CampaignError(code, f"{path.name} failed validation: {error}") from error

    def _read_yaml(self, path: Path, model, code: str):
        if not path.is_file():
            raise CampaignError(
                "campaign_not_initialised",
                f"missing campaign file {path.name}. The campaign directory is not initialised.",
                ["Run scripts/new_campaign.py to create the campaign files."],
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except yaml.YAMLError as error:
            raise CampaignError(code, f"{path.name} contains invalid YAML: {error}") from error
        try:
            return model.model_validate(payload)
        except ValidationError as error:
            raise CampaignError(code, f"{path.name} failed validation: {error}") from error

    def read_state(self) -> CampaignState:
        return self._read_json(self.state_path, CampaignState, "invalid_state_file")

    def read_manifest(self) -> Manifest:
        return self._read_yaml(self.manifest_path, Manifest, "invalid_manifest_file")

    def read_players(self) -> PlayersFile:
        return self._read_yaml(self.players_path, PlayersFile, "invalid_players_file")

    def read_character(self, character_id: str) -> Character:
        path = self.character_path(character_id)
        if not path.is_file():
            valid = ", ".join(self.character_ids()) or "none yet"
            raise CampaignError(
                "character_not_found",
                f"No character has id {character_id!r}.",
                [f"Valid character ids: {valid}."],
            )
        return self._read_json(path, Character, "invalid_character_file")

    def resolve_character_id(self, requested: str) -> str:
        """Resolve a narrator-supplied identifier to a canonical character id.

        Precedence: exact id, unique case-folded id, unique case-folded sheet
        name. A model that sends a display name such as 'Ossa' for the id
        'ossa' resolves deterministically. An unknown or ambiguous identifier
        raises, so a wrong guess never lands on another character.
        """
        return resolve_or_raise(
            requested,
            self.character_ids(),
            lambda cid: self.read_character(cid).name,
            entity="character",
            entity_plural="characters",
            code_prefix="character",
        )

    def character_ids(self) -> list[str]:
        if not self.characters_dir.is_dir():
            return []
        return sorted(path.stem for path in self.characters_dir.glob("*.json"))

    def list_characters(self) -> list[Character]:
        return [self.read_character(cid) for cid in self.character_ids()]

    def location_ids(self) -> list[str]:
        locations_dir = self.world_dir / "locations"
        if not locations_dir.is_dir():
            return []
        return sorted(path.stem for path in locations_dir.glob("*.md"))

    def starting_location(self) -> str:
        """The authored world's declared starting location id, or "".

        Reads ``world/locations/index.yaml``'s ``starting_location`` key. Fails
        open to the empty string — a world without the declaration simply has no
        default, and a display-and-defaulting convenience must never raise.
        """
        index_path = self.world_dir / "locations" / "index.yaml"
        try:
            payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - an absent or broken index means no default
            return ""
        value = payload.get("starting_location") if isinstance(payload, dict) else None
        return value.strip() if isinstance(value, str) else ""

    def npc_index_entries(self) -> dict[str, dict]:
        """``world/npcs/index.yaml`` as id to entry, fail-open to empty.
        """
        index_path = self.world_dir / "npcs" / "index.yaml"
        try:
            payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - an absent or broken index means no authored names
            return {}
        entries = payload.get("npcs") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return {}
        return {
            str(entry["id"]): entry
            for entry in entries
            if isinstance(entry, dict) and is_valid_id(str(entry.get("id", "")))
        }

    def authored_persons_for(self, location_id: str) -> dict[str, ScenePerson]:
        """The ``ScenePerson`` entries a location's ``visible_entities`` declare.
        """
        from .models import ScenePerson

        try:
            meta = self.read_location(location_id).get("meta") or {}
        except CampaignError:
            return {}
        visible = meta.get("visible_entities") if isinstance(meta, dict) else None
        if not isinstance(visible, list):
            return {}
        entries = self.npc_index_entries()
        persons: dict[str, ScenePerson] = {}
        for entity_id in visible:
            entity_id = str(entity_id).strip()
            entry = entries.get(entity_id)
            if entry is None:
                continue
            persons[entity_id] = ScenePerson(
                id=entity_id,
                name=str(entry.get("name") or entity_id).strip(),
                role=str(entry.get("role") or "").strip(),
                source="authored",
            )
        return persons

    def read_location(self, location_id: str) -> dict:
        """Return frontmatter, public body, and game-master-only body."""
        path = self.location_path(location_id)
        if not path.is_file():
            raise CampaignError(
                "location_not_found",
                f"No location file exists for id {location_id!r}.",
                ["Call campaign_status to list known location ids."],
            )
        raw = path.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(raw)
        if match is None:
            return {"id": location_id, "meta": {}, "body": raw}
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as error:
            raise CampaignError(
                "invalid_location_file", f"{path.name} has invalid frontmatter: {error}"
            ) from error
        return {"id": location_id, "meta": meta, "body": match.group(2)}

    def read_events(self, limit: int = 20) -> list[dict]:
        if not self.events_path.is_file():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        events: list[dict] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def quick_reference(self) -> str:
        path = self.rules_dir / "quick-reference.md"
        if not path.is_file():
            raise CampaignError("missing_rules_file", "rules/quick-reference.md is absent.")
        return path.read_text(encoding="utf-8")

    def subsystems_reference(self) -> str:
        path = self.rules_dir / "subsystems.json"
        if not path.is_file():
            raise CampaignError("missing_rules_file", "rules/subsystems.json is absent.")
        return path.read_text(encoding="utf-8")

    def effects_reference(self) -> str:
        path = self.rules_dir / "effects.json"
        if not path.is_file():
            raise CampaignError("missing_rules_file", "rules/effects.json is absent.")
        return path.read_text(encoding="utf-8")

    # -- rendering -----------------------------------------------------------

    @staticmethod
    def _combat_section(state: CampaignState) -> list[str]:
        """Render an open fight, and render nothing at all while none runs.

        A closed fight emits zero lines, so every campaign outside combat renders the
        same bytes this method's introduction found. That keeps the digest cost at zero
        for the turns that need no combat facts.

        The line grammar is machine-readable on purpose.
        ``narrator.interactions.read_combat_snapshot`` parses this exact section, so the
        classifier and the model read one authoritative source rather than two.
        """
        combat = state.combat
        if not combat.active:
            return []
        lines = ["", "## Combat", ""]
        lines.append(f"- round: {combat.round}")
        lines.append(f"- active actor: {combat.active_actor or 'none'}")
        lines.append(f"- order: {', '.join(combat.order) or 'none'}")
        for actor_id in combat.order:
            actor = combat.actors.get(actor_id)
            if actor is None:
                continue
            remaining = max(actor.actions_max - actor.actions_used, 0)
            # Range is omitted rather than defaulted. ``combat_start`` records a band
            # for each opponent and none for a player character, so any default here
            # would publish a distance the engine never measured.
            distance = combat.ranges.get(actor_id, "")
            band = f" range={distance}" if distance else ""
            lines.append(
                f"- combatant: {actor_id} side={actor.side}{band} "
                f"actions={remaining}/{actor.actions_max}"
            )
        return lines

    def render_scene_markdown(self, state: CampaignState) -> str:
        scene = state.scene
        lines = [
            "---",
            f"location_id: {scene.location_id or 'unknown'}",
            f"session: {state.session}",
            f"in_game_minutes: {state.in_game_minutes}",
            f"updated_at: {scene.updated_at}",
            "---",
            "",
            f"# {scene.title or 'Active scene'}",
            "",
            "## Summary",
            "",
            scene.summary or "No summary recorded.",
            "",
            "## Visible facts",
            "",
        ]
        lines += [f"- {fact}" for fact in scene.visible_facts] or ["- None recorded."]
        lines += ["", "## Exits", ""]
        lines += [f"- {exit_id}" for exit_id in scene.exits] or ["- None recorded."]
        lines += ["", "## Objects", ""]
        lines += [
            f"- {obj.id} ({obj.state}) {obj.note}".rstrip() for obj in scene.objects.values()
        ] or ["- None recorded."]
        lines += ["", "## Present NPCs", ""]


        lines += [
            f"- {npc_id}"
            if (npc := state.npcs.get(npc_id)) is None or npc.status == "alive"
            else f"- {npc_id} ({npc.status})"
            for npc_id in scene.present_npcs
        ] or ["- None recorded."]


        persons = sorted(scene.persons.values(), key=lambda person: (-person.last_event, person.id))
        lines += ["", "## Other persons present", ""]
        lines += [
            f"- {person.id}: {person.name}" for person in persons[:PERSONS_RENDER_CAP]
        ] or ["- None recorded."]
        lines += self._combat_section(state)
        lines += ["", "## Open hooks", ""]
        lines += [f"- {hook}" for hook in scene.hooks] or ["- None recorded."]
        lines += ["", "## Clocks", ""]
        lines += [
            f"- {clock.name} ({clock.filled}/{clock.segments}) {clock.note}".rstrip()
            for clock in state.clocks
        ] or ["- None recorded."]
        lines += ["", "## Unratified outcomes", ""]
        public_debt = []
        for debt in state.fiction_debt:
            realized_text = debt.realized_public_text()
            if realized_text:
                public_debt.append(f"- [{debt.seq}] {realized_text} ({debt.tool}: {debt.reason})")
            elif debt.reason or debt.outcome:
                public_debt.append(
                    f"- [{debt.seq}] {debt.tool} {debt.outcome}: {debt.reason}".rstrip(": ")
                )
        lines += public_debt or ["- None recorded."]
        lines += [
            "",
            "## Game-master-only facts",
            "",
            "<!-- Never quote this section to players. -->",
            "",
        ]
        lines += [f"- {fact}" for fact in scene.hidden_facts] or ["- None recorded."]
        lines += ["", "### Unratified GM-only outcomes", ""]
        hidden_debt = [
            f"- [{debt.seq}] {debt.stakes.hidden} ({debt.tool}: {debt.reason})"
            for debt in state.fiction_debt
            if debt.stakes.hidden
        ]
        lines += hidden_debt or ["- None recorded."]
        lines += [""]
        # Trailing whitespace in an authoritative rendered file is a defect: it
        # survives every later diff and signals a blank entry upstream.
        return "\n".join(line.rstrip() for line in lines)

    # -- transactions --------------------------------------------------------

    @contextmanager
    def transaction(self, tool: str, actor_id: str = "", reason: str = "") -> Iterator[Transaction]:
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.lock_path), timeout=LOCK_TIMEOUT_SECONDS)
        try:
            lock.acquire()
        except Timeout as error:
            raise CampaignError(
                "campaign_locked",
                "another campaign write is in progress; the lock did not release within "
                f"{LOCK_TIMEOUT_SECONDS:.0f} seconds",
                ["Wait for the running tool to finish, then retry once."],
            ) from error
        try:
            yield Transaction(self, tool=tool, actor_id=actor_id, reason=reason)
        finally:
            lock.release()

    # -- initialisation and maintenance --------------------------------------

    def is_initialised(self) -> bool:
        return self.state_path.is_file() and self.manifest_path.is_file()

    def initialize(self, title: str = "Untitled Campaign", force: bool = False) -> None:
        """Create an empty, valid campaign directory."""
        if self.is_initialised() and not force:
            raise CampaignError(
                "campaign_exists",
                f"a campaign already exists at {self.campaign_dir}",
                ["Pass force to overwrite, or choose another project root."],
            )
        for directory in (
            self.campaign_dir,
            self.characters_dir,
            self.logs_dir,
            self.summaries_dir,
            self.backups_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for directory in (self.characters_dir, self.summaries_dir, self.backups_dir):
            keep = directory / ".gitkeep"
            if not keep.exists():
                keep.write_text("", encoding="utf-8")

        manifest = Manifest(title=title)
        state = CampaignState()
        _atomic_write_text(
            self.manifest_path,
            yaml.safe_dump(manifest.model_dump(), sort_keys=False, allow_unicode=True),
        )
        _atomic_write_text(
            self.players_path,
            yaml.safe_dump(PlayersFile().model_dump(), sort_keys=False, allow_unicode=True),
        )
        _atomic_write_text(
            self.state_path, json.dumps(state.model_dump(), indent=2, ensure_ascii=False) + "\n"
        )
        _atomic_write_text(self.scene_path, self.render_scene_markdown(state))
        if not self.events_path.exists():
            _atomic_write_text(self.events_path, "")
        session_log = self.logs_dir / f"session-{state.session:03d}.md"
        if not session_log.exists():
            _atomic_write_text(
                session_log,
                f"# Session {state.session:03d}\n\nNo entries recorded yet.\n",
            )

# link_player was removed. It wrote players.yaml outside the campaign lock and
# had zero callers. A lock-bypassing writer in a store whose whole purpose is
# lock discipline is a hazard waiting for its first caller. character_create
# links a player inside its transaction instead.

    def backup(self, label: str = "") -> Path:
        """Archive campaign, world, and rules directories into one tar.gz."""
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.clock().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        archive = self.backups_dir / f"campaign-{stamp}{suffix}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for directory in (self.campaign_dir, self.world_dir, self.rules_dir):
                if not directory.is_dir():
                    continue
                handle.add(
                    directory,
                    arcname=directory.name,
                    filter=lambda info: None
                    if "/backups/" in info.name or info.name.endswith(".campaign.lock")
                    else info,
                )
        return archive


def discover_root(explicit: str | None = None) -> Path:
    """Resolve the project root.

    Precedence: an explicit argument, then the ``BSH_CAMPAIGN_ROOT`` environment
    variable, then the package's own repository root.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    from_env = os.environ.get("BSH_CAMPAIGN_ROOT")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]
