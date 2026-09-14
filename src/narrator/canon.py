"""Deliberate canon delivery: the digest injected into every turn's prompt.

The fix is construction, not persuasion: the engine prepends this digest to every
turn's prompt, so canon is in context before the model decides anything. Retrieval by
recall or by tool call is exactly the model-discretionary step the measurements show
failing (16 voluntary ``campaign_status`` calls in 420 instrumented turns, none on a
memory question).

Three sources, each keyed by pointers the scene render itself carries: the scene render
``campaign/scene.md``, the current location's authored file
``world/locations/{location_id}.md`` keyed off the render's front matter, and the
present NPCs' entries from ``world/npcs/index.yaml`` keyed off the render's Present
NPCs section. Movement needs no machinery: a ``scene_commit`` that changes the location
changes the next turn's injected file through the key.

Every block carries a hard character bound. Guided-decoding work in this repository
established the pattern: an unbounded field is an invitation. Bounds derive from
measurement — the five authored location files run 2,106 to 2,494 bytes and the scene
render measured 1,496 after one session — and truncation is counted per block, because
routine truncation is the recorded trigger for the stage-two lexical search.

This module is read-only in the exact mold of ``narrator/ledger.py``: the server owns
every file, the engine reads. It imports nothing outside the standard library, so the
main test environment pins it without Strands. Failure is open for context and closed
for canon: an unreadable source yields an empty block and the turn proceeds — reading
less is the status quo this module improves on, and the delivery gate, not the digest,
protects correctness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Appended where a block was cut at its bound. It tells the model more exists rather
#: than letting a truncated sentence read as complete canon.
TRUNCATION_MARKER = "\n[canon truncated at the block bound; more exists in the campaign files]"


DIGEST_HEADING = "# Scene canon, reference from the campaign files"


@dataclass
class CanonDigest:
    """The rendered digest plus the measurements the soak report records."""

    text: str = ""
    stats: dict = field(default_factory=dict)


def _read(path: Path) -> str:
    """One source, fail-open: unreadable or undecodable means absent, never an exception."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _bounded(text: str, bound: int) -> tuple[str, bool, int]:
    """Return the bounded block, whether it was cut, and how many source characters survived.
    """
    if len(text) <= bound:
        return text, False, len(text)
    kept = max(0, bound - len(TRUNCATION_MARKER))
    return text[:kept] + TRUNCATION_MARKER, True, kept


def _bounded_lines(text: str, bound: int) -> tuple[str, bool, int]:
    """Bound whole-line entries, dropping the last partial line rather than cutting it.

    Built for the Party resources block, whose lines carry exact coin and hit-point
    figures. ``_bounded``'s raw byte-position cut can land mid-digit inside one of
    those figures, showing a wrong partial number immediately before the truncation
    marker -- a plausible-looking but incorrect value is worse than showing nothing at
    all, for a block whose own instruction to the model is "copy this figure exactly".
    Advancing one whole line at a time, the same way ``_compose_scene_block`` trims
    the fact list, means a character's entry is either fully present or fully absent:
    never partially rendered.
    """
    if len(text) <= bound:
        return text, False, len(text)
    budget = max(0, bound - len(TRUNCATION_MARKER))
    kept = 0
    for line in text.split("\n"):
        cost = len(line) + 1
        if kept + cost > budget:
            break
        kept += cost
    kept = min(kept, len(text))
    return text[:kept].rstrip("\n") + TRUNCATION_MARKER, True, kept


_PROTECTED_SECTIONS = (
    "frontmatter", "Summary", "Objects", "Present NPCs", "Other persons present", "Combat",
    "Open hooks", "Clocks",
)


_FACTS_SECTION = "Visible facts"


def _section_spans(text: str) -> list[dict]:
    """The render's sections as half-open character spans, in document order."""
    spans: list[dict] = [{"name": "frontmatter", "start": 0}]
    offset = 0
    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            spans.append({"name": line.lstrip("#").strip(), "start": offset})
        offset += len(line) + 1
    for index, span in enumerate(spans):
        span["end"] = spans[index + 1]["start"] if index + 1 < len(spans) else len(text)
    return spans


def _items(chunk: str) -> list[str]:
    """The list entries in one section, excluding the render's empty-list placeholder."""
    return [
        line
        for line in chunk.split("\n")
        if line.startswith("- ") and line.strip() != "- None recorded."
    ]


def _fact_priorities(campaign: Path) -> dict[str, bool] | None:
    """Map each live visible statement's text to whether its refs hit a present entity.
    """
    try:
        state = json.loads(
            (campaign / "campaign" / "state.json").read_text(encoding="utf-8")
        )
        scene = state.get("scene") or {}
        statements = scene.get("statements") or []
        if not isinstance(statements, list) or not statements:
            return None
        present = set(scene.get("persons") or {})
        present.update(scene.get("present_npcs") or [])
        present.update(scene.get("objects") or {})
        characters_dir = campaign / "campaign" / "characters"
        if characters_dir.is_dir():
            present.update(path.stem for path in characters_dir.glob("*.json"))
        priorities: dict[str, bool] = {}
        any_refs = False
        for entry in statements:
            if not isinstance(entry, dict) or entry.get("superseded_at"):
                continue
            if entry.get("scope", "public") == "world_hidden":
                continue
            refs = entry.get("refs") or []
            if refs:
                any_refs = True
            priorities[str(entry.get("text", ""))] = any(ref in present for ref in refs)
        return priorities if any_refs else None
    except Exception:  # noqa: BLE001 - reading less is the status quo this improves on
        return None


def _rescue_referenced_facts(
    lines: list[str], boundary: int, budget: int, fact_priority: dict[str, bool]
) -> tuple[str, int]:
    """Trade the head's newest unreferenced facts for cut facts about present entities.

    Selection, not reordering: the emitted lines keep the render's own order, so
    every line is a verbatim render line and the document order the model reads is
    the order the record wrote. ``boundary`` is the first line the legacy prefix
    walk could not afford. Rescued facts (refs hitting a present entity, sitting at
    or beyond the boundary) enter oldest-first; room is made by evicting the
    *newest* unreferenced facts from the kept head, never a referenced one and never
    a non-fact line -- the head's measured 8-of-8 recall is the strongest result and
    the eviction takes its cheapest end. With nothing to rescue the emission equals
    the legacy prefix byte for byte.
    """

    def _is_fact(line: str) -> bool:
        return line.startswith("- ") and line.strip() != "- None recorded."

    def _hits(line: str) -> bool:
        return _is_fact(line) and fact_priority.get(line[2:], False)

    selected = list(range(boundary))
    rescued_indices = [i for i in range(boundary, len(lines)) if _hits(lines[i])]
    if not rescued_indices:
        return "".join(lines[i] + "\n" for i in selected), 0

    def _cost(index: int) -> int:
        return len(lines[index]) + 1

    used = sum(_cost(i) for i in selected)
    evictable = [
        i for i in sorted(selected, reverse=True) if _is_fact(lines[i]) and not _hits(lines[i])
    ]
    rescued = 0
    for index in rescued_indices:
        need = _cost(index)
        while used + need > budget and evictable:
            evicted = evictable.pop(0)
            selected.remove(evicted)
            used -= _cost(evicted)
        if used + need > budget:
            break
        selected.append(index)
        used += need
        rescued += 1
    return "".join(lines[i] + "\n" for i in sorted(selected)), rescued


def _compose_scene_block(
    text: str, bound: int, fact_priority: dict[str, bool] | None = None
) -> tuple[str, bool, list[dict], str]:
    """Compose the scene block by value, and report what each section kept.

    Head retention alone discards the render's tail by byte position, which takes Open
    hooks, Clocks, Unratified outcomes, and both game-master sections before it touches
    a single fact. This function keeps the protected sections resident at every depth
    and lets the fact list's tail and the low-value sections absorb the cut instead.

    Three properties hold by construction. A block that fits under the bound returns
    unchanged, so a non-truncating turn produces the bytes head retention produced.
    Every emitted region is a verbatim substring of the render, including the fact
    list, which the budget trims at line boundaries rather than rebuilding from its
    lines. An unrecognized render shape falls back to head retention, because reading
    less is the status quo this module improves on.
    """
    if len(text) <= bound:
        rows = [
            {
                "name": span["name"],
                "state": "full",
                "items": len(_items(text[span["start"]:span["end"]])),
                "items_retained": len(_items(text[span["start"]:span["end"]])),
            }
            for span in _section_spans(text)
        ]
        return text, False, rows, "whole"

    spans = _section_spans(text)
    budget = bound - len(TRUNCATION_MARKER)
    reserved = sum(
        span["end"] - span["start"] for span in spans if span["name"] in _PROTECTED_SECTIONS
    )
    unknown_shape = not any(span["name"] == _FACTS_SECTION for span in spans)
    if unknown_shape or reserved > budget:
        block, cut, kept = _bounded(text, bound)
        return block, cut, _scene_sections(text, kept), "head"

    remaining = budget - reserved
    pieces: list[str] = []
    rows: list[dict] = []
    dropped = False
    policy_override = ""

    for span in spans:
        chunk = text[span["start"]:span["end"]]
        entries = _items(chunk)
        if span["name"] in _PROTECTED_SECTIONS:
            pieces.append(chunk)
            rows.append(
                {
                    "name": span["name"],
                    "state": "full",
                    "items": len(entries),
                    "items_retained": len(entries),
                }
            )
            continue
        if span["name"] == _FACTS_SECTION:
            # The budget advances one whole line at a time, so ``kept`` is always a
            # line boundary inside the chunk. Emitting that byte range keeps the
            # section verbatim; rejoining the lines would append a newline the render
            # does not carry whenever every line survives.
            lines = chunk.split("\n")
            kept = 0
            boundary = None
            for index, line in enumerate(lines):
                cost = len(line) + 1
                if kept + cost > remaining:
                    dropped = True
                    boundary = index
                    break
                kept += cost
            kept = min(kept, len(chunk))
            emitted = chunk[:kept]
            rescued = 0
            if boundary is not None and fact_priority is not None:
                # B3: the cut is where reference-keyed selection earns its keep. With
                # no refs-hitting line beyond the boundary this returns the legacy
                # prefix byte for byte.
                emitted, rescued = _rescue_referenced_facts(
                    lines, boundary, remaining, fact_priority
                )
                kept = len(emitted)
            if rescued:
                policy_override = "refs"
            remaining -= kept
            pieces.append(emitted)
            # State follows what the block carries, not what the counters imply. A
            # section whose heading itself did not fit emitted nothing, and reporting
            # that as partial would credit the digest with a section it never sent.
            if emitted == chunk:
                state = "full"
            elif not emitted:
                state = "absent"
            else:
                state = "partial"
            row = {
                "name": span["name"],
                "state": state,
                "items": len(entries),
                "items_retained": len(_items(emitted)),
            }
            if fact_priority is not None:
                row["items_rescued"] = rescued
            rows.append(row)
            continue
        # A section the budget cannot hold ends the tail. Skipping it to reach a
        # smaller later section would emit the record out of document order, which
        # reads as canon the render never wrote.
        if not dropped and len(chunk) <= remaining:
            pieces.append(chunk)
            remaining -= len(chunk)
            rows.append(
                {
                    "name": span["name"],
                    "state": "full",
                    "items": len(entries),
                    "items_retained": len(entries),
                }
            )
            continue
        dropped = True
        rows.append(
            {
                "name": span["name"],
                "state": "absent",
                "items": len(entries),
                "items_retained": 0,
            }
        )

    block = "".join(pieces).rstrip("\n")
    if dropped:
        block += TRUNCATION_MARKER
    return block, dropped, rows, policy_override or "priority"


def _scene_sections(text: str, kept: int) -> list[dict]:
    """Per-section survival of the scene render into the bounded block.

    The render at ``bsh_mcp.store.render_scene_markdown`` writes frontmatter, Summary,
    Visible facts, Exits, Objects, Present NPCs, Open hooks, Clocks, Unratified
    outcomes, and the two game-master sections in that fixed document order. ``_bounded``
    keeps a prefix, so every section's fate follows from its span against ``kept``: a
    section ending at or before ``kept`` survives whole, a section starting at or after
    it disappears, and one straddling the boundary loses its tail.

    A section's list entries carry the same measurement one level down. The visible
    facts list accrues across a session, so counting retained entries against total
    entries states how far the cut ate into the newest facts. The measurement reads
    heading text out of the render itself, so a future render change renames a row
    here instead of silently dropping one.

    Every value derives from the text and the bound alone, so two runs over the same
    record produce identical rows.
    """
    sections: list[dict] = [
        {"name": "frontmatter", "start": 0, "state": "full", "items": 0, "items_retained": 0}
    ]
    offset = 0
    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            sections.append(
                {
                    "name": line.lstrip("#").strip(),
                    "start": offset,
                    "state": "full",
                    "items": 0,
                    "items_retained": 0,
                }
            )
        elif line.startswith("- ") and line.strip() != "- None recorded.":
            sections[-1]["items"] += 1
            if offset + len(line) <= kept:
                sections[-1]["items_retained"] += 1
        offset += len(line) + 1

    for index, section in enumerate(sections):
        end = sections[index + 1]["start"] if index + 1 < len(sections) else len(text)
        if end <= kept:
            section["state"] = "full"
        elif section["start"] >= kept:
            section["state"] = "absent"
        else:
            section["state"] = "partial"
        del section["start"]
    return sections


def _scene_location_id(scene_text: str) -> str:
    """The front-matter ``location_id`` line, the key for the authored block."""
    for line in scene_text.splitlines()[:10]:
        if line.startswith("location_id:"):
            return line.split(":", 1)[1].strip()
    return ""


def _present_npc_ids(scene_text: str) -> list[str]:
    """The ids under the render's Present NPCs heading, written as ``- {npc_id}``
    or, for a non-alive NPC, ``- {npc_id} ({status})`` -- the annotation
    ``CampaignStore.render_scene_markdown`` adds is stripped back to the bare id."""
    ids: list[str] = []
    in_section = False
    for line in scene_text.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Present NPCs"
            continue
        if in_section and line.startswith("- ") and line.strip() != "- None recorded.":
            ids.append(re.sub(r"\s*\([^)]*\)$", "", line[2:].strip()))
    return ids


def _scene_person_ids(scene_text: str) -> list[str]:
    """Scene person ids.
    """
    ids: list[str] = []
    in_section = False
    for line in scene_text.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Other persons present"
            continue
        if in_section and line.startswith("- ") and line.strip() != "- None recorded.":
            ids.append(line[2:].split(":", 1)[0].strip())
    return ids


def _npc_entries(index_text: str, npc_ids: list[str], *, public: bool = False) -> str:
    """The index entries for the present NPCs.

    The index is a simple list of mappings, so a line-level split on the entry
    delimiter suffices; parsing it with a YAML library would add a dependency the
    narrator environment does not carry, for structure this function only needs to
    slice, not interpret.
    """
    if not npc_ids:
        return ""
    chunks: list[str] = []
    current: list[str] = []
    keep = False
    for line in index_text.splitlines():
        if line.lstrip().startswith("- id:"):
            if keep and current:
                chunks.append("\n".join(current))
            entry_id = line.split(":", 1)[1].strip()
            keep = entry_id in npc_ids
            current = [line]
            continue
        if keep and (not public or line.lstrip().startswith("name:")):
            current.append(line)
    if keep and current:
        chunks.append("\n".join(current))
    return "\n".join(chunks)


def _scene_exit_ids(scene_text: str) -> list[str]:
    """The ids under the render's Exits heading, written as ``- {location_id}``."""
    ids: list[str] = []
    in_section = False
    for line in scene_text.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Exits"
            continue
        if in_section and line.startswith("- ") and line.strip() != "- None recorded.":
            ids.append(line[2:].strip())
    return ids


def _exit_entries(index_text: str, exit_ids: list[str]) -> str:
    """Each present exit's ``id``/``name`` pair from ``world/locations/index.yaml``.
    """
    if not exit_ids:
        return ""
    chunks: list[str] = []
    current: list[str] = []
    keep = False
    for line in index_text.splitlines():
        if line.lstrip().startswith("- id:"):
            if keep and current:
                chunks.append("\n".join(current))
            entry_id = line.split(":", 1)[1].strip()
            keep = entry_id in exit_ids
            current = [line]
            continue
        if keep and line.lstrip().startswith("name:"):
            current.append(line)
    if keep and current:
        chunks.append("\n".join(current))
    return "\n".join(chunks)


def _character_summary(text: str) -> str:
    """One bullet line for a character sheet's coins, hit points, and carried items.

    Reads the same raw JSON ``bsh_mcp.models.Character`` writes, without importing
    that model: a parse failure or a missing field yields no line rather than an
    exception, matching this module's fail-open contract for every other source it
    reads. A field the schema has not populated yet (an older campaign, a stub
    fixture) reads as its type's zero value instead of dropping the character.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    name = str(payload.get("name") or payload.get("id") or "")
    if not name:
        return ""
    status = str(payload.get("status") or "ok")
    coins = payload.get("coins", 0)
    hp = payload.get("hp", 0)
    hp_max = payload.get("hp_max", 0)
    weapons = [str(item) for item in payload.get("weapons") or [] if str(item)]
    equipment = [str(item) for item in payload.get("equipment") or [] if str(item)]
    resources = [
        f"{entry.get('id') or entry.get('name') or '?'} ({entry.get('die') or '?'})"
        for entry in payload.get("resources") or []
        if isinstance(entry, dict)
    ]
    weapons_text = ", ".join(weapons) if weapons else "none"
    equipment_text = ", ".join(equipment) if equipment else "none"
    resources_text = ", ".join(resources) if resources else "none"
    return (
        f"- {name} ({status}): {coins} coins, HP {hp}/{hp_max}, "
        f"weapons: {weapons_text}, equipment: {equipment_text}, "
        f"resources: {resources_text}"
    )


def _session_zero_cue(characters_dir: Path) -> str:
    """The digest's opening block while the campaign holds no characters at all.

    A campaign that begins at session zero renders every other digest block empty —
    no scene, no location, no party — so without this the model receives no signal
    that the disclosed ``bsh-session-zero`` procedure applies, and the measured
    default for an unprompted small model is to improvise. The cue names the
    procedure and the two tools that ground it. Unlike the fail-open blocks around
    it, this one fails ABSENT: it renders only when an initialised characters
    directory positively holds zero sheets, because telling an established campaign
    to run session zero on a transient read error would be worse than a missing cue.
    The block disappears on the first ``character_create``.
    """
    try:
        if not characters_dir.is_dir():
            return ""
        if any(characters_dir.glob("*.json")):
            return ""
    except OSError:
        return ""
    return (
        "## Session zero (this campaign has no characters yet)\n\n"
        "No player characters exist, so ordinary play cannot begin. Run the "
        "session-zero procedure: load the bsh-session-zero skill with the skills "
        "tool and follow it in order. Ground every choice in the engine — "
        "character_options lists the legal origins, backgrounds, weapon tables, "
        "and armour categories, and character_create rolls and writes the sheet. "
        "Never recite an origin or background list from memory, and never invent "
        "rolled numbers: read back exactly what the tools return."
    )


def _party_resources_text(characters_dir: Path) -> str:
    """One bullet per readable character file, sorted by file stem for determinism.

    Sorting by stem rather than by the sheet's own ``name`` keeps this deterministic
    across a rename and matches ``bsh_mcp.store.CampaignStore.character_ids``, the
    engine's own listing order.
    """
    if not characters_dir.is_dir():
        return ""
    lines = [_character_summary(_read(path)) for path in sorted(characters_dir.glob("*.json"))]
    return "\n".join(line for line in lines if line)


#: The location-file sections a player present at the scene could perceive. The
#: public digest keeps only these for the planner. It is an allowlist, so an
#: authored section the world adds later stays out of the planner by default, which
#: fails closed for disclosure. The narrator receives every section, gated by the
#: delivery boundary. World files carry these headings; see ../../world/locations/.
_PUBLIC_LOCATION_SECTIONS = frozenset(
    {"Public description", "Immediate danger", "Useful details"}
)

#: The scene-render section that opens the game-master-only tail. Everything from
#: this heading onward — the hidden facts and the unratified GM-only outcomes — is
#: authored for the narrator alone. ``src/bsh_mcp/store.py`` renders it under this
#: exact heading with the comment that it must never reach players.
_GM_ONLY_SCENE_SECTION = "Game-master-only facts"


def _public_scene(scene_text: str) -> str:
    """Drop the game-master-only tail from the scene render for the planner.

    The narrator reads the whole record and the delivery gate protects players. The
    planner's output becomes player-visible text, so it must never read the hidden
    facts or the unratified GM-only outcomes the tail carries.
    """
    for span in _section_spans(scene_text):
        if span["name"] == _GM_ONLY_SCENE_SECTION:
            return scene_text[: span["start"]].rstrip()
    return scene_text


def _public_location(location_text: str) -> str:
    """Keep only the player-perceivable location sections for the planner.

    This drops the frontmatter — which carries ``hidden_entities`` — and every
    section outside the allowlist, including Hidden truths, NPC motives, Discoverable
    clues, and Consequences. The planner reads what a person standing there sees.
    """
    if not location_text:
        return ""
    spans = _section_spans(location_text)
    kept = [
        location_text[span["start"] : span["end"]]
        for span in spans
        if span["name"] in _PUBLIC_LOCATION_SECTIONS
    ]
    return "\n".join(chunk.strip() for chunk in kept).strip()


def render_digest(
    campaign_root: Path | str,
    repo_root: Path | str,
    scene_max_chars: int,
    location_max_chars: int,
    npcs_max_chars: int,
    resources_max_chars: int = 900,
    exits_max_chars: int = 400,
    *,
    public: bool = False,
) -> CanonDigest:
    """Build the five-block digest and its measurements. Never raises.

    ``public`` produces the planner scope: the scene render minus its game-master-only
    tail, the location file reduced to its player-perceivable sections, and each NPC
    reduced to id and name. The narrator receives the full digest, because the delivery
    gate stands between it and a player; the planner receives the public one, because
    its output becomes player-visible decision text. Party resources carry no hidden
    content — a coin total and a hit-point figure are the character's own knowledge —
    so that block renders unfiltered in both scopes.

    ``resources_max_chars`` and ``exits_max_chars`` both default rather than joining
    the earlier bounds as required positionals, so every existing call site keeps
    rendering both blocks without a signature update. 900 characters holds a full
    six-character party at this block's per-character line length with margin; a
    campaign that grows past that still renders every character up to the bound and
    flags the cut. Unlike the location, npc, and scene blocks, the resources block
    bounds on whole-line entries (``_bounded_lines``, not ``_bounded``): a raw
    byte-position cut can land mid-digit inside a coin or hit-point figure, and a wrong
    partial number is worse than an absent one for a block whose only purpose is an
    exact figure to copy. A character whose full entry does not fit the remaining
    budget is dropped entirely rather than shown partially. 400 characters for the
    exit block holds every one of ``world/locations/index.yaml``'s five authored
    locations' own id/name pair (the whole index file is 652 bytes; no scene names
    more than three exits at once) with room to grow.
    """
    campaign = Path(campaign_root)
    repo = Path(repo_root)

    scene_text = _read(campaign / "campaign" / "scene.md")
    location_id = _scene_location_id(scene_text)
    npc_ids = _present_npc_ids(scene_text)
    npc_ids += [pid for pid in _scene_person_ids(scene_text) if pid not in npc_ids]
    exit_ids = _scene_exit_ids(scene_text)
    session_zero_block = _session_zero_cue(campaign / "campaign" / "characters")

    location_text = (
        _read(repo / "world" / "locations" / f"{location_id}.md") if location_id else ""
    )
    npc_text = _npc_entries(
        _read(repo / "world" / "npcs" / "index.yaml"), npc_ids, public=public
    )
    exit_text = _exit_entries(
        _read(repo / "world" / "locations" / "index.yaml"), exit_ids
    )
    resources_text = _party_resources_text(campaign / "campaign" / "characters")

    if public:
        scene_text = _public_scene(scene_text)
        location_text = _public_location(location_text)

    scene_stripped = scene_text.strip()
    fact_priority = _fact_priorities(campaign)
    scene_block, scene_cut, scene_rows, scene_policy = _compose_scene_block(
        scene_stripped, scene_max_chars, fact_priority
    )
    location_block, location_cut, _ = _bounded(location_text.strip(), location_max_chars)
    npc_block, npc_cut, _ = _bounded(npc_text.strip(), npcs_max_chars)
    exit_block, exit_cut, _ = _bounded(exit_text.strip(), exits_max_chars)
    resources_block, resources_cut, _ = _bounded_lines(resources_text.strip(), resources_max_chars)

    parts: list[str] = []
    if session_zero_block:
        parts.append(session_zero_block)
    if location_block:
        parts.append(
            "## Authored location canon (world file for the current location)\n\n"
            + location_block
        )
    if npc_block:
        parts.append("## Present NPC canon (world index entries)\n\n" + npc_block)
    if exit_block:
        parts.append("## Exit canon (world index entries)\n\n" + exit_block)
    if scene_block:
        parts.append("## Current scene record (campaign/scene.md)\n\n" + scene_block)


    if resources_block:
        parts.append(
            "## Party resources (campaign/characters, current totals)\n\n" + resources_block
        )

    text = ""
    if parts:


        ownership = (
            ""
            if public
            else (
                "Knowledge has owners. Hidden truths in the authored location canon are "
                "the world's secrets: reveal them only when play uncovers them. Facts in "
                "the scene record were established in play; any fact the party created or "
                "witnessed — their own actions, discoveries, and hiding places, wherever "
                "the record files them — is the party's knowledge, and you answer the "
                "party about it plainly when they ask.\n\n"
            )
        )


        resource_instruction = (
            "" if not resources_block else (
                "A coin, item, or hit-point figure is never estimated, recalled, or "
                "computed: the Party resources block below is the character sheet's "
                "own current numbers, so your fiction stays consistent with them, but "
                "it never licenses stating one. State a coin, item, or hit-point "
                "figure only when a tool result this turn returned it, and call "
                "`campaign_status` or `character_sheet` when you want to state one "
                "and no tool has -- except for a pending short- or long-rest "
                "declaration, where only `rest` supplies the figure for what the rest "
                "restored: `campaign_status` and `character_sheet` return the pre-rest "
                "total, not the rest's own effect, and stating that total as what the "
                "rest restored is wrong even though it is a real, current number.\n\n"
            )
        )
        text = (
            f"{DIGEST_HEADING}\n\n"
            "This is background reference, not narration to copy. The one invariant "
            "stands: every die roll and every durable state change passes through "
            "the tools — canon never substitutes for a roll. Adjudicate declared "
            "actions with the tools first, then narrate the outcome consistently "
            "with this canon.\n\n"
            + ownership
            + resource_instruction
            + "This digest is a view of the record, not the recorder. The record "
            "advances only through the tools: seeing a fact here never means a new "
            "one is saved. Commit durable changes with scene_commit as they happen, "
            "exactly as if this digest were absent.\n\n"
            + "\n\n".join(parts)
        )

    return CanonDigest(
        text=text,
        stats={
            "location_id": location_id,
            "present_npcs": npc_ids,
            "exits": exit_ids,
            "session_zero": bool(session_zero_block),
            "scene_chars": len(scene_block),
            "location_chars": len(location_block),
            "npcs_chars": len(npc_block),
            "exits_chars": len(exit_block),
            "resources_chars": len(resources_block),
            "total_chars": len(text),
            "truncated": {
                "scene": scene_cut,
                "location": location_cut,
                "npcs": npc_cut,
                "exits": exit_cut,
                "resources": resources_cut,
            },
            "scene_sections": scene_rows,
            "scene_policy": scene_policy,
        },
    )
