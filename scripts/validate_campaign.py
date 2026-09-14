#!/usr/bin/env python3
"""Validate every campaign, rules, and world file without changing state.

Usage::

    uv run python scripts/validate_campaign.py
    uv run python scripts/validate_campaign.py --root /srv/black-bell --json

Exit code 0 means every check passed. Exit code 1 means at least one error.
Warnings never change the exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.data import RulesData, RulesDataError  # noqa: E402
from bsh_mcp.store import CampaignError, CampaignStore  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def ok(self, message: str) -> None:
        self.checks.append(message)


def check_rules(store: CampaignStore, report: Report) -> RulesData | None:
    try:
        data = RulesData.load(store.rules_dir)
    except RulesDataError as error:
        report.error(f"rules tables: {error}")
        return None
    report.ok(f"rules tables loaded from {store.rules_dir}")

    index = data.background_index()
    report.ok(f"{len(index)} backgrounds across 3 origins")
    if not (store.rules_dir / "quick-reference.md").is_file():
        report.error("rules/quick-reference.md is missing")
    if not (store.rules_dir / "attribution.md").is_file():
        report.error("rules/attribution.md is missing; the SRD licence requires attribution")
    return data


def check_world(store: CampaignStore, report: Report) -> set[str]:
    location_ids = set(store.location_ids())
    if not location_ids:
        report.warn("world/locations holds no location files")
    for location_id in sorted(location_ids):
        try:
            location = store.read_location(location_id)
        except CampaignError as error:
            report.error(f"world location {location_id}: {error.message}")
            continue
        meta = location.get("meta") or {}
        if meta.get("id") != location_id:
            report.error(
                f"world location {location_id}: frontmatter id is {meta.get('id')!r}"
            )
        for exit_id in meta.get("exits") or []:
            if exit_id not in location_ids:
                report.error(f"world location {location_id}: exit {exit_id!r} has no file")
    if location_ids:
        report.ok(f"{len(location_ids)} world locations parsed with consistent exits")

    index_path = store.world_dir / "locations" / "index.yaml"
    if index_path.is_file():
        listed = {
            entry.get("id")
            for entry in (yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}).get(
                "locations", []
            )
        }
        missing = location_ids - listed
        extra = listed - location_ids
        for location_id in sorted(missing):
            report.warn(f"world/locations/index.yaml omits {location_id}")
        for location_id in sorted(extra):
            report.error(f"world/locations/index.yaml lists {location_id} with no file")
    return location_ids


def check_campaign(store: CampaignStore, report: Report, location_ids: set[str]) -> None:
    if not store.is_initialised():
        report.error(
            f"campaign is not initialised at {store.campaign_dir}; "
            "run scripts/new_campaign.py"
        )
        return

    try:
        state = store.read_state()
        manifest = store.read_manifest()
        players = store.read_players()
    except CampaignError as error:
        report.error(f"campaign files: {error.message}")
        return
    report.ok("campaign/state.json, manifest.yaml, and players.yaml validate")

    if manifest.session != state.session:
        report.error(
            f"manifest session {manifest.session} does not match state session {state.session}"
        )

    character_ids = set(store.character_ids())
    for character_id in sorted(character_ids):
        try:
            store.read_character(character_id)
        except CampaignError as error:
            report.error(f"character {character_id}: {error.message}")
    if character_ids:
        report.ok(f"{len(character_ids)} character sheets validate")

    for link in players.players:
        if link.character_id not in character_ids:
            report.error(
                f"players.yaml links Discord user {link.discord_user_id} to unknown character "
                f"{link.character_id}"
            )
    unlinked = character_ids - {link.character_id for link in players.players}
    for character_id in sorted(unlinked):
        report.warn(f"character {character_id} has no Discord user link in players.yaml")

    if state.scene.location_id and location_ids and state.scene.location_id not in location_ids:
        report.error(f"scene location {state.scene.location_id!r} has no world file")

    for npc_id in state.scene.present_npcs:
        if npc_id not in state.npcs:
            report.error(f"scene lists present NPC {npc_id!r} with no record in state.npcs")

    combat = state.combat
    if combat.active:
        for actor_id in combat.order:
            if actor_id not in combat.actors:
                report.error(f"combat order names {actor_id!r} with no actor record")
            elif combat.actors[actor_id].side == "pc" and actor_id not in character_ids:
                report.error(f"combat includes unknown character {actor_id!r}")
            elif combat.actors[actor_id].side == "npc" and actor_id not in state.npcs:
                report.error(f"combat includes unknown NPC {actor_id!r}")
        report.ok(f"combat is active in round {combat.round}")

    events = []
    if store.events_path.is_file():
        for number, line in enumerate(
            store.events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                report.error(f"campaign/logs/events.jsonl line {number}: {error}")

    sequences = [event.get("seq") for event in events if isinstance(event.get("seq"), int)]
    if sequences != sorted(sequences):
        report.error("events.jsonl sequence numbers are not monotonic")
    if len(set(sequences)) != len(sequences):
        report.error("events.jsonl contains duplicate sequence numbers")
    if sequences and sequences[-1] > state.event_seq:
        report.error(
            f"events.jsonl reached sequence {sequences[-1]} but state.json records "
            f"{state.event_seq}"
        )
    if events:
        report.ok(f"{len(events)} audit events with consistent sequence numbers")

    leftovers = sorted(str(path) for path in store.campaign_dir.rglob(".*.tmp-*"))
    for path in leftovers:
        report.error(f"an interrupted write left a temporary file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT), help="project root directory")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    arguments = parser.parse_args()

    store = CampaignStore(Path(arguments.root))
    report = Report()

    check_rules(store, report)
    location_ids = check_world(store, report)
    check_campaign(store, report, location_ids)

    if arguments.json:
        print(
            json.dumps(
                {
                    "root": str(store.root),
                    "ok": not report.errors,
                    "checks": report.checks,
                    "warnings": report.warnings,
                    "errors": report.errors,
                },
                indent=2,
            )
        )
    else:
        print(f"campaign root: {store.root}")
        for message in report.checks:
            print(f"  ok      {message}")
        for message in report.warnings:
            print(f"  warning {message}")
        for message in report.errors:
            print(f"  ERROR   {message}")
        print(
            f"\n{len(report.checks)} checks passed, {len(report.warnings)} warnings, "
            f"{len(report.errors)} errors"
        )

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
