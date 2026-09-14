"""Scene identity resolution, ``scene_commit`` and its per-concern helpers, and
the fiction-debt ledger's waive path (``ledger_settle``).
"""

from __future__ import annotations

import re

from ..models import (
    OBJECT_STATES,
    TRAVERSAL_CLOCK_PREFIX,
    Clock,
    SceneObject,
    ScenePerson,
    Statement,
    live_statements,
    slugify,
)
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError
from .common import guard, success_after_commit


def _realized_branch(outcome: str) -> str:
    """Map a test outcome to the stakes branch the dice realized.

    Criticals map to their base branch; the audit event's outcome field
    already distinguishes them.
    """
    return "success" if outcome in ("critical_success", "success") else "failure"


_NAME_BOUNDARY = r"(?<!\w){name}(?!\w)"


def _label_names(label: str, name: str) -> bool:
    r"""Whether ``name`` occurs in ``label`` as a whole word, case-folded.

    "Sera Vane, the bell-keeper" names Sera Vane; "Rade, the fishmonger" names Rade;
    "Orso" does not name "Orso Pell". Unicode-aware word boundaries (``\w``), not
    ``\b``, so a Japanese or Cyrillic name resolves the same way.
    """
    name = name.strip()
    if len(name) < 2:
        return False
    return re.search(_NAME_BOUNDARY.format(name=re.escape(name)), label, re.IGNORECASE) is not None


#: Leading articles dropped from an introduced label before it becomes a person's
#: name or id, so "a Salt Magistrate clerk" and "the Salt Magistrate clerk, frowning"
#: converge on one ``salt-magistrate-clerk``. An English list, deliberately small: it
#: is identifier hygiene for labels the sweep copies from English narration, not a
#: guard -- in another language the article simply stays part of the name and two
#: spellings converge less often, which costs a duplicate person, never canon.
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def person_display_name(label: str) -> str:
    """The label minus a leading article, whitespace-collapsed."""
    return " ".join(_LEADING_ARTICLE.sub("", label.strip()).split())


def resolve_person_label(
    label: str,
    *,
    npcs: dict[str, str],
    characters: dict[str, str],
    persons: dict[str, str],
    index: dict[str, str],
) -> tuple[str, str]:
    """Which recorded or authored entity an introduced label names, if any.
    """
    label = person_display_name(label)
    slug = slugify(label)
    order = (("npc", npcs), ("character", characters), ("index", index), ("person", persons))
    for kind, known in order:
        if slug in known:
            return kind, slug
    for kind, known in order:
        for entity_id, name in known.items():
            if _label_names(label, name):
                return kind, entity_id
    return "new", slug


def _normalized_fact(entry: str) -> str:
    """One scene fact reduced to its comparable core: case, spacing, and trailing
    sentence punctuation stripped, so "Rade is dead." and "Rade is dead" are the
    same fact rather than two."""
    return " ".join(str(entry).split()).casefold().rstrip(" .!?")


def _clean_entries(values: list[str] | None) -> list[str]:
    """Drop blank and whitespace-only entries from a narrator-supplied list.

    A narrator model routinely pads a list with empty strings. Writing one into
    an authoritative file corrupts canon: `scene.md` renders it as an empty
    bullet and every later reader treats it as a real fact.
    """
    return [entry.strip() for entry in (values or []) if str(entry).strip()]


class SceneCommitResult(ToolEnvelopeSuccess):
    scene: dict
    in_game_time: str


class SceneMixin:
    """Scene identity, ``scene_commit`` and its helpers, and the ledger waive path."""

    # -- narrative state -----------------------------------------------------

    def _location_display_name(self, location_id: str) -> str:
        """The authored location's display name, else the slug in Title Case.

        Reads the location file's ``name`` frontmatter — the same field
        ``world/locations/index.yaml`` publishes — and falls back to a slug-derived
        title for an authored file that omits it. Callers pass ids they have
        already validated against ``location_ids``; an unreadable file still
        degrades to the slug form rather than raising, because a display name must
        never fail a commit.
        """
        try:
            name = str(
                (self.store.read_location(location_id).get("meta") or {}).get("name") or ""
            )
        except CampaignError:
            name = ""
        return name.strip() or location_id.replace("-", " ").replace("_", " ").strip().title()

    @guard
    def scene_commit(
        self,
        public_summary: str,
        location_id: str = "",
        in_game_time_delta_minutes: int = 0,
        visible_changes: list[str] | None = None,
        hidden_changes: list[str] | None = None,
        new_clocks: list[dict] | None = None,
        clock_updates: dict[str, int] | None = None,
        resolved_hooks: list[str] | None = None,
        new_hooks: list[str] | None = None,
        scene_title: str = "",
        present_npcs: list[str] | None = None,
        exits: list[str] | None = None,
        object_updates: dict[str, dict] | None = None,
        environment_tags: list[str] | None = None,
        persons: list[dict] | None = None,
        departed_persons: list[str] | None = None,
        refs: list[str] | None = None,
        retire_facts: list[str] | None = None,
        party_secrets: list[str] | None = None,
    ) -> SceneCommitResult | ToolEnvelopeFailure:
        """Commit a durable fictional change. Never changes hit points, Doom, or inventory.

        environment_tags replaces the scene's open-vocabulary descriptors (e.g.
        "natural", "urban") when passed; omit it to leave the current tags
        unchanged. Some backgrounds' rest-time mechanics read these tags -- e.g. a
        Herbalist's stock only replenishes on a long rest whose scene carries
        "natural". Tag the scene here before that rest happens, not as a rest
        argument.
        """
        if in_game_time_delta_minutes < 0:
            raise CampaignError(
                "invalid_time_delta",
                "in-game time cannot run backwards.",
                ["Pass a non-negative number of minutes."],
            )
        public_summary = public_summary.strip()
        if not public_summary:
            raise CampaignError(
                "empty_public_summary",
                "a scene commit needs a public summary. A blank summary would erase the "
                "committed one.",
                ["State in one or two sentences what durably changed, then call again."],
            )
        scene_title = scene_title.strip()

        with self.store.transaction(
            "scene_commit", actor_id="", reason=public_summary
        ) as transaction:
            scene = transaction.state.scene
            moved = self._commit_location(transaction, scene, location_id)
            self._commit_title(transaction, scene, scene_title)
            scene.summary = public_summary
            visible_changes = _clean_entries(visible_changes)
            hidden_changes = _clean_entries(hidden_changes)
            party_secrets = _clean_entries(party_secrets)
            validated_refs, retired_texts = self._commit_statements(
                transaction, scene,
                visible_changes=visible_changes, hidden_changes=hidden_changes,
                party_secrets=party_secrets, refs=refs, retire_facts=retire_facts,
            )
            if exits is not None:
                scene.exits = _clean_entries(exits)
            if environment_tags is not None:
                scene.environment_tags = [tag.lower() for tag in _clean_entries(environment_tags)]
                transaction.record(
                    f"scene environment tags set to {scene.environment_tags or '(none)'}"
                )
            objects_applied = self._commit_objects(transaction, scene, object_updates)
            if present_npcs is not None:
                present_npcs = [
                    self._resolve_npc_id(
                        transaction,
                        npc_id,
                        ["Call npc_create before placing an NPC in the scene."],
                    )
                    for npc_id in _clean_entries(present_npcs)
                ]
                scene.present_npcs = present_npcs
            persons_added = self._commit_persons(transaction, scene, persons)
            persons_departed = self._commit_departures(transaction, scene, departed_persons)
            hooks_resolved, hooks_added = self._commit_hooks(
                transaction, scene, resolved_hooks, new_hooks
            )
            scene.updated_at = self.store.clock().isoformat(timespec="seconds")
            transaction.state.scene = scene
            clocks_created = self._commit_clocks(transaction, scene, new_clocks, clock_updates)

            if in_game_time_delta_minutes:
                transaction.state.in_game_minutes += int(in_game_time_delta_minutes)
                transaction.record(f"time advanced {in_game_time_delta_minutes} minutes")

            transaction.scene_dirty = True
            # Ratify: every debt entry present now has seq <= this event's
            # sequence by construction, so a scene commit clears the ledger.
            cleared = [debt.seq for debt in transaction.state.fiction_debt]
            if cleared:
                transaction.state.fiction_debt = []
                transaction.record(
                    f"ratified {len(cleared)} unratified outcome(s): "
                    + ", ".join(str(seq) for seq in cleared)
                )
            sequence = transaction.commit(
                {
                    "outcome": "scene_committed",
                    "public_summary": public_summary,
                    "visible_changes": visible_changes,
                    "hidden_changes": hidden_changes,
                    "location_id": location_id,
                    "in_game_time_delta_minutes": int(in_game_time_delta_minutes),
                    "new_clocks": clocks_created,
                    "clock_updates": dict(clock_updates or {}),
                    "resolved_hooks": hooks_resolved,
                    "new_hooks": hooks_added,
                    "object_updates": objects_applied,
                    "persons_added": persons_added,
                    "refs": list(validated_refs),
                    "retired_facts": retired_texts,
                    "party_secrets": party_secrets,
                    "persons_departed": persons_departed,
                    **moved,
                    "ratified_seqs": cleared,
                }
            )

        return success_after_commit(
            transaction,
            public_summary,
            sequence=sequence,
            outcome="scene_committed",
            narration_facts=[public_summary] + list(visible_changes or []),
            scene=transaction.state.scene.model_dump(),
            in_game_time=self._time_string(transaction.state.in_game_minutes),
        )

    # -- scene_commit's per-concern steps ------------------------------------
    #
    # scene_commit accreted to 615 lines as each measured slice went inline; these
    # helpers are its mechanical decomposition. Every one runs inside the caller's
    # transaction, mutates the shared ``scene``, records and warns through the
    # transaction, and returns only its audit-payload fragment. Order is the
    # caller's; behavior is byte-for-byte the inline original's.

    def _commit_location(self, transaction, scene, location_id: str) -> dict:
        """Move or anchor the scene; persons and located NPCs travel with it."""
        persons_cleared: list[str] = []
        persons_seeded: list[str] = []
        npcs_left: list[str] = []
        npcs_joined: list[str] = []
        known_locations = set(self.store.location_ids())
        if location_id:
            if known_locations and location_id not in known_locations:
                raise CampaignError(
                    "location_not_found",
                    f"No location file exists for id {location_id!r}.",
                    [
                        f"Known locations: {', '.join(sorted(known_locations))}.",
                        "Author the location file before moving the party there.",
                    ],
                )
            if scene.location_id != location_id:
                # A fight left active across a location change would stay
                # ``combat.active`` while the scene it was fought in is gone,
                # blocking every later combat_start and contradicting the fiction.
                # Refuse the move instead of carrying it along silently; the
                # narrator must resolve the fight in the same scene it happened.
                if transaction.state.combat.active:
                    raise CampaignError(
                        "combat_blocks_location_change",
                        "combat is still active; moving the party now would leave "
                        "the fight open in a scene the party already left.",
                        [
                            "Call combat_close with a reason (fled, surrendered, a "
                            "truce, etc.) to end the fight, then call scene_commit "
                            "again with the new location."
                        ],
                    )
                transaction.record(f"scene moved to {location_id}")


                arriving = self.store.authored_persons_for(location_id)
                kept = {
                    pid: person for pid, person in scene.persons.items()
                    if pid in arriving
                }
                cleared = sorted(pid for pid in scene.persons if pid not in kept)
                seeded = sorted(pid for pid in arriving if pid not in kept and pid not in transaction.state.npcs)
                for pid in seeded:
                    kept[pid] = arriving[pid]
                scene.persons = kept
                if cleared:
                    transaction.record(f"persons left with the scene: {', '.join(cleared)}")
                if seeded:
                    transaction.record(f"authored persons present at {location_id}: {', '.join(seeded)}")
                persons_cleared, persons_seeded = cleared, seeded
                # NPCs leave with the scene too, by their own recorded location: one
                # whose location_id is the scene being left goes, one recorded at
                # the destination joins (a corpse stays where it fell, so dead
                # counts; fled and captured are elsewhere by definition), and one
                # with no location -- a companion, or an NPC created absent -- is
                # left exactly as it was, because the record cannot say.
                old_location = scene.location_id
                npcs = transaction.state.npcs
                left = sorted(
                    nid for nid in scene.present_npcs
                    if nid in npcs and old_location and npcs[nid].location_id == old_location
                    and npcs[nid].location_id != location_id
                )
                joined = sorted(
                    nid for nid, npc in npcs.items()
                    if nid not in scene.present_npcs
                    and npc.location_id == location_id
                    and npc.status in ("alive", "dead")
                )
                scene.present_npcs = [nid for nid in scene.present_npcs if nid not in left] + joined
                if left:
                    transaction.record(f"NPCs left with the scene: {', '.join(left)}")
                if joined:
                    transaction.record(f"NPCs present at {location_id}: {', '.join(joined)}")
                npcs_left, npcs_joined = left, joined
            scene.location_id = location_id
        elif scene.location_id not in known_locations:


            start = self.store.starting_location()
            if start and start in known_locations:
                scene.location_id = start
                transaction.record(
                    f"scene located at authored starting location {start}"
                )

        return {
            "persons_cleared": persons_cleared,
            "persons_seeded": persons_seeded,
            "npcs_left": npcs_left,
            "npcs_joined": npcs_joined,
        }

    def _commit_title(self, transaction, scene, scene_title: str) -> None:
        """An explicit title wins; a located, untitled scene takes its location's name."""
        known_locations = set(self.store.location_ids())
        if scene_title:
            scene.title = scene_title
        elif not scene.title and scene.location_id in known_locations:


            scene.title = self._location_display_name(scene.location_id)
            transaction.record(
                f"scene titled {scene.title!r} from location {scene.location_id}"
            )

    def _commit_statements(
        self, transaction, scene, *, visible_changes, hidden_changes, party_secrets,
        refs, retire_facts,
    ) -> tuple[tuple[str, ...], list[str]]:
        """Retire quoted facts, then add this commit's statements per scope."""


        next_seq = transaction.state.event_seq + 1
        characters = self.store.list_characters()
        known_refs = (
            set(transaction.state.npcs)
            | set(scene.persons)
            | set(scene.objects)
            | {character.id for character in characters}
        )
        validated_refs = tuple(
            ref for ref in _clean_entries(refs) if ref in known_refs
        )


        names_by_id: dict[str, tuple[str, ...]] = {}
        for npc_id, npc in transaction.state.npcs.items():
            names_by_id[npc_id] = (npc.name,)
        for person_id, person in scene.persons.items():
            names_by_id.setdefault(person_id, (person.name,))
        for object_id in scene.objects:
            names_by_id.setdefault(object_id, (object_id.replace("-", " "),))
        for character in characters:
            names_by_id.setdefault(character.id, (character.name,))

        def _refs_for(text: str) -> tuple[str, ...]:
            resolved = tuple(
                entity_id
                for entity_id, names in names_by_id.items()
                if entity_id not in validated_refs
                and any(_label_names(text, name) for name in names)
            )
            return validated_refs + resolved
        for ref in _clean_entries(refs):
            if ref not in known_refs:
                transaction.warn(f"ref {ref!r} names nothing recorded; dropped.")
        retired_texts: list[str] = []
        statements = list(scene.statements)
        if retire_facts:
            live_by_norm = {
                _normalized_fact(statement.text): index
                for index, statement in enumerate(statements)
                if not statement.superseded_at
            }
            for quote in _clean_entries(retire_facts):
                index = live_by_norm.get(_normalized_fact(quote))
                if index is None:
                    transaction.warn(
                        f"retire_facts entry {quote!r} matches no live fact; "
                        "quote the line exactly as the record states it."
                    )
                    continue
                match = statements[index]
                # Copy-and-replace, the record model's own mutation style.
                statements[index] = match.model_copy(update={"superseded_at": next_seq})
                retired_texts.append(match.text)
                transaction.record(f"fact retired: {match.text}")
        # B2: a party secret is knowledge the party holds that the world's
        # people do not -- the arm-3 bucket semantics, typed. It renders with the
        # visible facts (the party knows it) and never in the game-master-only
        # section, which is exactly the placement the tool-contract prose used
        # to carry.
        for scope, additions in (
            ("public", visible_changes),
            ("world_hidden", hidden_changes),
            ("party_secret", party_secrets),
        ):
            live_norms = {
                _normalized_fact(statement.text)
                for statement in live_statements(statements, (scope,))
            }
            for text in additions or []:
                normalized = _normalized_fact(text)
                if not normalized or normalized in live_norms:
                    continue  # the exact-duplicate skip _merge_facts performed
                live_norms.add(normalized)
                statements.append(
                    Statement(
                        seq=next_seq, text=text, scope=scope,
                        refs=_refs_for(text), source="model",
                    )
                )
        scene.statements = statements
        return validated_refs, retired_texts

    def _commit_objects(self, transaction, scene, object_updates) -> dict[str, dict]:
        """Upsert typed barrier and prop state; a new object needs an explicit state."""
        objects = dict(scene.objects)
        objects_applied: dict[str, dict] = {}
        for object_id, changes in (object_updates or {}).items():
            if not isinstance(changes, dict):
                raise CampaignError(
                    "invalid_object_update",
                    f"the update for object {object_id!r} must be a mapping of "
                    "state and note.",
                    ['Pass {"state": "locked", "note": "..."} for each object id.'],
                )
            existing = objects.get(object_id)
            state = changes.get("state", existing.state if existing else None)
            if state is None:
                raise CampaignError(
                    "empty_object_state",
                    f"object {object_id!r} needs a state; it has no prior state to "
                    "keep.",
                    [f"Use one of {', '.join(OBJECT_STATES)}."],
                )
            if state not in OBJECT_STATES:
                raise CampaignError(
                    "invalid_object_state",
                    f"{state!r} is not a recognized object state.",
                    [f"Use one of {', '.join(OBJECT_STATES)}."],
                )
            note = changes.get("note", existing.note if existing else "")


            if "traversal_segments" in changes:
                segments = changes["traversal_segments"]
            elif existing is not None:
                segments = existing.traversal_segments
            else:
                segments = None
            obj = SceneObject(
                id=object_id, state=state, note=str(note).strip(),
                traversal_segments=segments,
            )
            objects[object_id] = obj
            objects_applied[object_id] = obj.model_dump()
            if existing is None:
                transaction.record(f"new object {object_id}: {obj.state}")
            elif existing.state != obj.state:
                transaction.record(
                    f"object {object_id}: {existing.state} -> {obj.state}"
                )
            else:
                transaction.record(f"object {object_id}: note updated")
        scene.objects = objects
        return objects_applied

    def _commit_persons(self, transaction, scene, persons) -> list[str]:
        """Introduce or update scene persons, converging identity on known names."""
        persons_added: list[str] = []
        if persons:
            if not isinstance(persons, list):
                raise CampaignError(
                    "invalid_persons",
                    "persons must be a list of {name, role} mappings.",
                    ['Pass [{"name": "Salt Magistrate clerk", "role": "..."}].'],
                )
            recorded_persons = dict(scene.persons)
            next_seq = transaction.state.event_seq + 1
            npc_names = {nid: npc.name for nid, npc in transaction.state.npcs.items()}
            character_names = {
                character.id: character.name for character in self.store.list_characters()
            }
            index_entries = self.store.npc_index_entries()
            index_names = {iid: str(entry.get("name") or iid) for iid, entry in index_entries.items()}
            for entry in persons:
                if isinstance(entry, str):
                    entry = {"name": entry}
                if not isinstance(entry, dict):
                    raise CampaignError(
                        "invalid_person",
                        "a persons entry must be a mapping of name and role.",
                        ['Pass {"name": "...", "role": "..."} for each person.'],
                    )
                name = person_display_name(str(entry.get("name") or ""))
                if not name:
                    continue
                role = str(entry.get("role") or "").strip()
                # Second iteration, change 1: identity converges on names. The
                # label is resolved against every recorded or authored name
                # before an id is allocated, so an epithet ("Sera Vane, the
                # bell-keeper") lands on the index's own id.
                kind, person_id = resolve_person_label(
                    name,
                    npcs=npc_names,
                    characters=character_names,
                    persons={pid: person.name for pid, person in recorded_persons.items()},
                    index=index_names,
                )
                if kind == "npc":
                    transaction.warn(
                        f"{name!r} names the recorded NPC {person_id!r}; pass present_npcs "
                        "for an NPC, not persons."
                    )
                    continue
                if kind == "character":
                    transaction.warn(
                        f"{name!r} names the party member {person_id!r}; a character is "
                        "never a scene person."
                    )
                    continue
                folded_from: dict | None = None
                if kind == "index":
                    authored = index_entries[person_id]
                    name = str(authored.get("name") or person_id).strip()
                    authored_role = str(authored.get("role") or "").strip()
                    role = role or authored_role
                    # The iteration-2 soak left "the bell-keeper" recorded as the
                    # person ``bell-keeper`` -- a role-only label carries no name to
                    # converge on -- which a later "Sera Vane, the bell-keeper" would
                    # then sit beside. At the moment the name is learned, a
                    # model-sourced person whose recorded name is a whole word of
                    # the authored role is folded into the authored identity,
                    # carrying its first_event: a merge into one person, the same
                    # move promotion makes, not a removal.
                    for other_id, other in list(recorded_persons.items()):
                        if (
                            other_id != person_id
                            and other.source == "model"
                            and authored_role
                            and _label_names(authored_role, other.name)
                        ):
                            folded_from = {"id": other_id, "first_event": other.first_event}
                            del recorded_persons[other_id]
                            transaction.record(f"person {other_id} folded into {person_id}")
                            break
                existing = recorded_persons.get(person_id)
                if existing is None:
                    recorded_persons[person_id] = ScenePerson(
                        id=person_id,
                        name=name,
                        role=role,
                        source="model",
                        first_event=folded_from["first_event"] if folded_from else next_seq,
                        last_event=next_seq,
                    )
                    persons_added.append(person_id)
                    transaction.record(f"new person {person_id}: {name}")
                else:
                    updates: dict = {"last_event": next_seq}
                    if role and not existing.role:
                        updates["role"] = role
                        transaction.record(f"person {person_id}: role recorded")
                    recorded_persons[person_id] = existing.model_copy(update=updates)
            scene.persons = recorded_persons

        return persons_added

    def _commit_departures(self, transaction, scene, departed_persons) -> list[str]:
        """Let a named person leave the scene; an unknown name only warns."""
        persons_departed: list[str] = []
        if departed_persons:
            remaining = dict(scene.persons)
            for entry in _clean_entries(departed_persons):
                kind, person_id = resolve_person_label(
                    entry,
                    npcs={}, characters={},
                    persons={pid: person.name for pid, person in remaining.items()},
                    index={},
                )
                if kind != "person":
                    # A departure is usually the short form of a recorded name
                    # ("Vell" for "Vell the eel-seller"). The introduction rule
                    # deliberately does not match a short label inside a longer
                    # known name -- "Orso" must not become Orso Pell -- but here the
                    # candidates are only the persons present, so the reverse
                    # match resolves a departure and can create nothing.
                    short = person_display_name(entry)
                    matches = [
                        pid for pid, person in remaining.items()
                        if len(short) >= 3 and _label_names(person.name, short)
                    ]
                    if len(matches) == 1:
                        kind, person_id = "person", matches[0]
                if kind == "person":
                    del remaining[person_id]
                    persons_departed.append(person_id)
                    transaction.record(f"person {person_id} left the scene")
                else:
                    transaction.warn(f"{entry!r} names no person present; nothing left the scene.")
            scene.persons = remaining

        return persons_departed

    def _commit_hooks(self, transaction, scene, resolved_hooks, new_hooks) -> tuple[list[str], list[str]]:
        """Resolve and add open hooks."""
        hooks = list(scene.hooks)
        hooks_resolved: list[str] = []
        hooks_added: list[str] = []
        for hook in _clean_entries(resolved_hooks):
            if hook in hooks:
                hooks.remove(hook)
                hooks_resolved.append(hook)
                transaction.record(f"resolved hook: {hook}")
            else:
                transaction.warn(f"hook not found, nothing removed: {hook}")
        for hook in _clean_entries(new_hooks):
            if hook not in hooks:
                hooks.append(hook)
                hooks_added.append(hook)
                transaction.record(f"new hook: {hook}")
        scene.hooks = hooks
        return hooks_resolved, hooks_added

    def _commit_clocks(self, transaction, scene, new_clocks, clock_updates) -> list[dict]:
        """Create and advance clocks, with the traversal-clock rules intact."""
        clocks = {clock.id: clock for clock in transaction.state.clocks}
        clocks_created: list[dict] = []
        for definition in new_clocks or []:
            clock = Clock(**definition)
            clock.name = clock.name.strip()
            clock.note = clock.note.strip()
            if not clock.name:
                raise CampaignError(
                    "empty_clock_name",
                    f"clock {clock.id!r} needs a name.",
                    ["Name the clock after the pressure it tracks."],
                )
            if clock.id in clocks:
                existing_clock = clocks[clock.id]


                if (
                    clock.id.startswith(TRAVERSAL_CLOCK_PREFIX)
                    and existing_clock.filled >= existing_clock.segments
                ):
                    existing_clock.filled = 0
                    transaction.record(
                        f"clock {clock.id} reopened for a fresh crossing "
                        f"(was {existing_clock.segments}/{existing_clock.segments})"
                    )
                else:
                    transaction.warn(
                        f"clock {clock.id} already exists; leaving it unchanged"
                    )
                continue


            typed_exit = scene.objects.get(clock.id)
            if typed_exit is not None and typed_exit.traversal_segments is not None:
                if clock.segments != typed_exit.traversal_segments:
                    transaction.record(
                        f"clock {clock.id} sized to the typed exit's own "
                        f"{typed_exit.traversal_segments}-segment count "
                        f"(requested {clock.segments})"
                    )
                    clock.segments = typed_exit.traversal_segments
                    clock.filled = min(clock.filled, clock.segments)
            clocks[clock.id] = clock
            clocks_created.append(clock.model_dump())
            transaction.record(f"new clock {clock.id} ({clock.filled}/{clock.segments})")
        for clock_id, delta in (clock_updates or {}).items():
            clock = clocks.get(clock_id)
            if clock is None:
                raise CampaignError(
                    "clock_not_found",
                    f"No clock has id {clock_id!r}.",
                    ["Call campaign_status to list clock ids."],
                )
            previous = clock.filled


            base = (
                0
                if clock.id.startswith(TRAVERSAL_CLOCK_PREFIX)
                and clock.filled >= clock.segments
                else clock.filled
            )
            clock.filled = max(0, min(clock.segments, base + int(delta)))
            transaction.record(
                f"clock {clock_id}: {previous}/{clock.segments} -> "
                f"{clock.filled}/{clock.segments}"
            )
            if clock.filled >= clock.segments:
                transaction.warn(f"clock {clock_id} is full: the consequence lands now")
        transaction.state.clocks = list(clocks.values())
        return clocks_created

    @guard
    def ledger_settle(self, reason: str) -> ToolEnvelopeSuccess | ToolEnvelopeFailure:
        """Clear the fiction-debt ledger without writing a scene entry, recording why.

        Nothing mechanical is lost by a waive. Each debt entry was written inside the
        same transaction as its mechanics, and the audit event this method appends
        preserves every waived entry with its realized stake text and the stated
        reason. Only the pending scene entry is skipped.
        """
        reason = reason.strip()
        if not reason:
            raise CampaignError(
                "empty_waive_reason",
                "a ledger settle needs a reason. The audit event is the only record of "
                "why these outcomes never entered the scene.",
                ["State in one or two sentences why no scene entry is needed."],
            )
        with self.store.transaction(
            "ledger_settle", actor_id="", reason=reason
        ) as transaction:
            debts = list(transaction.state.fiction_debt)
            if not debts:
                raise CampaignError(
                    "ledger_already_clear",
                    "the fiction-debt ledger holds nothing to settle.",
                    ["Call this only while campaign_status reports uncommitted fiction."],
                )
            waived = [
                {
                    "seq": debt.seq,
                    "tool": debt.tool,
                    "outcome": debt.outcome,
                    "realized_public_text": debt.realized_public_text(),
                }
                for debt in debts
            ]
            transaction.state.fiction_debt = []
            transaction.record(
                f"waived {len(waived)} unratified outcome(s) without a scene entry: "
                + ", ".join(str(entry["seq"]) for entry in waived)
            )
            sequence = transaction.commit(
                {
                    "outcome": "ledger_waived",
                    "reason": reason,
                    "waived": waived,
                }
            )

        return success_after_commit(
            transaction,
            f"Ledger settled: {len(waived)} outcome(s) waived without a scene entry.",
            sequence=sequence,
            outcome="ledger_waived",
            narration_facts=[],
        )
