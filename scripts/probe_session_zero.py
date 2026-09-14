#!/usr/bin/env python3
"""Probe: a scripted session zero against the live endpoint, once per origin.

Runs under the narrator interpreter (it drives the real ``NarratorService``), with
``BSH_SERVER_PYTHON`` naming the project interpreter for the MCP server child::

    export BSH_SERVER_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
    .narrator-venv/bin/python scripts/probe_session_zero.py \
        --report /tmp/session-zero-probe.json

Each run bootstraps a fresh campaign with no characters, no player links, and no
seeded scene — the exact state ``scripts/play-session-zero.sh`` hands a player —
then replays one scripted table through the ceremony: opening, boundaries, an
options question, an explicit character request, bonds, and the first scene. The
replay channel stamps the authenticated ``player`` principal, so the engine-side
``character_create`` account binding is on the path exactly as it is in the
terminal.

The checks afterwards read the campaign files directly, never the model's prose:

- exactly one character exists, with the requested origin and background trio;
- the sheet's account link and ``players.yaml`` both name the replay principal;
- the audit log carries the ``character_create`` event (the sheet is audited);
- ``character_options`` was called before creation (the menu grounded the offer;
  read from the engine's own per-turn tool log, because a read-only tool
  deliberately writes no audit event);
- the scene record was committed and the state carries a scene title.

The last two are the model's discipline rather than the engine's guarantee; a run
that fails them proves a prompt or skill gap, not a mechanics gap, and the report
says which.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.service import GameService  # noqa: E402
from narrator.channels.replay import TranscriptReplayAdapter  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.interactions import read_trusted_scope  # noqa: E402
from narrator.service import NarratorService  # noqa: E402

#: One scripted table per origin. Backgrounds deliberately differ from the
#: ``play_terminal`` premades, so a pass proves guided creation follows the
#: player's own request rather than any recorded default. Every trio is legal:
#: three ids, all from the origin, none unique.
SCENARIOS: dict[str, dict] = {
    "barbarian": {
        "name": "Ketil",
        "backgrounds": ["chieftain", "raider", "storyteller"],
        "weapons": "a warhammer",
    },
    "civilised": {
        "name": "Isra",
        "backgrounds": ["bookworm", "surgeon", "street-urchin"],
        "weapons": "a duelling sword",
    },
    "decadent": {
        "name": "Nereza",
        "backgrounds": ["warlock", "changeling", "pit-fighter"],
        "weapons": "a jagged dagger",
    },
}


def transcript_for(origin: str, scenario: dict) -> str:
    """The scripted table: six mentions walking the whole session-zero order."""
    name = scenario["name"]
    backgrounds = ", ".join(scenario["backgrounds"])
    return f"""
@GM Hello! We are ready to start the campaign. Please run session zero for us.
@GM No content boundaries from me — nothing needs to stay off screen. Keep romance disabled.
@GM I want to play a {origin} character. What backgrounds can I choose from?
@GM Create my character now. Name: {name}. Origin: {origin}. Backgrounds: {backgrounds}. For weapons give me {scenario['weapons']}. No armour, no shield.
@GM My bond: {name} owes the bell-keeper Sera Vane for a debt paid in silence. That is also our reason to care that the evening bell stopped.
@GM Please open the first scene now and tell me what {name} sees.
"""


def bootstrap(campaign_root: Path, server_python: str) -> None:
    """The session-zero start state, built exactly as the terminal launcher builds it."""
    shutil.copytree(REPO_ROOT / "rules", campaign_root / "rules")
    shutil.copytree(REPO_ROOT / "world", campaign_root / "world")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            server_python,
            str(REPO_ROOT / "scripts" / "new_campaign.py"),
            "--root",
            str(campaign_root),
        ],
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
    )


def inspect_campaign(
    campaign_root: Path,
    origin: str,
    scenario: dict,
    model_tool_calls: list[str] | None = None,
) -> dict:
    """Read the durable record and score every check from files, never prose.

    ``model_tool_calls`` is the engine's flattened per-turn tool log — the only
    place a read-only ``character_options`` call is visible, since it opens no
    transaction and therefore writes no audit event.
    """
    campaign = campaign_root / "campaign"
    characters = sorted(
        path for path in (campaign / "characters").glob("*.json")
    )
    sheets = [json.loads(path.read_text(encoding="utf-8")) for path in characters]

    events = []
    events_path = campaign / "logs" / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    tool_sequence = [event.get("tool") for event in events]

    import yaml

    players_payload = yaml.safe_load((campaign / "players.yaml").read_text(encoding="utf-8")) or {}
    players = players_payload.get("players") or []

    state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
    scene = state.get("scene") or {}

    model_calls = list(model_tool_calls or [])
    sheet = sheets[0] if len(sheets) == 1 else {}
    checks = {
        "one_character_created": len(sheets) == 1,
        "origin_matches": sheet.get("origin") == origin,
        "backgrounds_match": sorted(sheet.get("backgrounds") or [])
        == sorted(scenario["backgrounds"]),
        "name_matches": sheet.get("name") == scenario["name"],
        "hp_equals_con": bool(sheet)
        and sheet.get("hp") == sheet.get("hp_max") == (sheet.get("attributes") or {}).get("CON"),
        "doom_d6": sheet.get("doom_die") == "d6",
        "account_bound_to_principal": sheet.get("discord_user_id") == "player",
        "player_link_written": any(
            link.get("discord_user_id") == "player"
            and link.get("character_id") == sheet.get("id")
            for link in players
            if isinstance(link, dict)
        ),
        "creation_audited": "character_create" in tool_sequence,
        "options_grounded_the_offer": "character_options" in model_calls
        and (
            "character_create" not in model_calls
            or model_calls.index("character_options")
            < model_calls.index("character_create")
        ),
        "scene_committed": "scene_commit" in tool_sequence,
        "scene_located": str(scene.get("location_id") or "") not in ("", "unknown"),
        "scene_title_set": bool(str(scene.get("title") or "").strip()),
    }
    return {
        "checks": checks,
        "scene": {
            "title": scene.get("title"),
            "location_id": scene.get("location_id"),
            "summary_chars": len(str(scene.get("summary") or "")),
        },
        "sheet": {
            key: sheet.get(key)
            for key in ("id", "name", "origin", "backgrounds", "hp", "doom_die", "weapons")
        },
        "tool_sequence": tool_sequence,
        "model_tool_calls": model_calls,
        "events": len(events),
    }


async def run_origin(origin: str, arguments: argparse.Namespace) -> dict:
    scenario = SCENARIOS[origin]
    with tempfile.TemporaryDirectory(prefix=f"sz-probe-{origin}-") as tmp:
        campaign_root = Path(tmp)
        bootstrap(campaign_root, os.environ["BSH_SERVER_PYTHON"])

        transcript = campaign_root / "table.txt"
        transcript.write_text(transcript_for(origin, scenario), encoding="utf-8")

        config = NarratorConfig(
            campaign_root=campaign_root,
            repo_root=REPO_ROOT,
            base_url=arguments.base_url,
            model_id=arguments.model,
        )
        adapter = TranscriptReplayAdapter(
            transcript, backfill_limit=config.backfill_limit, echo=arguments.echo
        )
        game = GameService(campaign_root)
        game.provision_scene_merchants(read_trusted_scope(campaign_root).present_npc_ids)
        service = NarratorService(
            config, adapter, purchase_executor=game.narrator_purchase_executor()
        )
        report = await service.run()

        # The engine's per-turn tool log, flattened in turn order: the soak
        # harness reads it through the same window, and it is the only visible
        # trace of read-only tool calls.
        model_tool_calls = [
            name
            for turn_calls in getattr(service.engine, "_tool_log", [])
            for name in turn_calls
        ]
        result = inspect_campaign(campaign_root, origin, scenario, model_tool_calls)
        result["turns"] = report.turns
        result["delivered"] = report.delivered
        result["withheld"] = report.withheld
        result["errors"] = list(report.errors)
        result["posted_chars"] = sum(len(text) for text in adapter.posted)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin",
        choices=[*sorted(SCENARIOS), "all"],
        default="all",
        help="which origin's table to run (default: all three)",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--report", default="", help="write the JSON report to this path")
    parser.add_argument(
        "--echo", action="store_true", help="print delivered narration while running"
    )
    arguments = parser.parse_args()

    if not os.environ.get("BSH_SERVER_PYTHON"):
        print("error: BSH_SERVER_PYTHON must name the project interpreter", file=sys.stderr)
        return 2

    origins = sorted(SCENARIOS) if arguments.origin == "all" else [arguments.origin]
    report: dict = {"origins": {}}
    for origin in origins:
        print(f"== session-zero probe: {origin} ==", flush=True)
        result = asyncio.run(run_origin(origin, arguments))
        report["origins"][origin] = result
        for check, passed in result["checks"].items():
            print(f"  {'ok  ' if passed else 'FAIL'} {check}", flush=True)

    all_checks = [
        passed
        for origin_result in report["origins"].values()
        for passed in origin_result["checks"].values()
    ]
    report["passed"] = sum(all_checks)
    report["failed"] = len(all_checks) - sum(all_checks)
    if arguments.report:
        Path(arguments.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"report: {arguments.report}", flush=True)
    print(f"{report['passed']} checks passed, {report['failed']} failed", flush=True)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
