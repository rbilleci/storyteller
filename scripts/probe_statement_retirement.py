#!/usr/bin/env python3
"""Whether the narrator uses ``scene_commit``'s ``retire_facts`` -- quoting a stale
Visible facts line so it stops being current -- is model behavior; the offline pins
prove only the mechanism. Two scenarios, each a full service run in a sandbox
(the ``probe_runic_weapon.py`` harness pattern):

- ``voluntary``: the campaign record states Rill is sprawled in the mud; the player
  simply stands up. Does the model retire the stale line on its own?
- ``instructed``: the same, followed by a player turn pointing at the stale line and
  asking for it to be retired.

The scorer reads the campaign record and audit log, never the narration: a run
scores ``retired`` when the stale statement's ``superseded_at`` is set. The
``voluntary`` rate is reported, not gated -- commit discipline history predicts it
low; the ``instructed`` scenario is the capability gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.service import GameService  # noqa: E402
from bsh_mcp.testing import bootstrap_probe_campaign  # noqa: E402

_LOCATION = "the-eel-market"
_STALE = "Rill is sprawled in the mud on the road."

_SCENARIOS = {
    "voluntary": [
        "Rill plants a hand in the cold mud, pushes himself up, and stands, brushing "
        "the worst of it from his coat.",
    ],
    "instructed": [
        "Rill plants a hand in the cold mud, pushes himself up, and stands, brushing "
        "the worst of it from his coat.",
        "The scene record still says \"Rill is sprawled in the mud on the road.\" "
        "That is no longer true now that he is standing. Retire that line from the "
        "record and note that he is on his feet.",
    ],
}


def _models(endpoint: str) -> str:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as response:
        payload = json.loads(response.read())
    return str(payload["data"][0]["id"])


def _prepare_campaign(root: Path) -> str:
    game, character_id = bootstrap_probe_campaign(
        root, title="Retirement probe", weapons=["long knife"]
    )
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    seeded = game.scene_commit(
        "Rill slips off the plank walk into the mud.",
        location_id=_LOCATION,
        visible_changes=[_STALE],
    )
    if not seeded.get("ok"):
        raise ValueError(f"probe seed commit was unavailable: {seeded}")
    return character_id


def _score(root: Path) -> dict:
    state = GameService(root).store.read_state()
    stale = next((s for s in state.scene.statements if s.text == _STALE), None)
    events = GameService(root).store.read_events(limit=50)
    retire_events = [
        event for event in events if event.get("retired_facts")
    ]
    return {
        "retired": bool(stale is not None and stale.superseded_at),
        "stale_present": stale is not None,
        "retire_events": [
            {"seq": event.get("seq"), "retired_facts": event.get("retired_facts")}
            for event in retire_events
        ],
        "live_visible_facts": state.scene.visible_facts,
    }


async def _runtime_flow(scenario: str, root: Path, endpoint: str, model_id: str) -> dict:
    from narrator.channels.base import (
        ChannelCapabilities,
        ChannelMessage,
        ChannelPrincipal,
        DecisionDeliveryReceipt,
        InboundTurn,
    )
    from narrator.config import NarratorConfig
    from narrator.decisions import DecisionSubmission
    from narrator.service_assembly import build_narrator_service

    _prepare_campaign(root)
    declarations = _SCENARIOS[scenario]

    class _Adapter:
        name = "terminal"
        decision_capabilities = ChannelCapabilities(
            structured_decisions=True, atomic_decision_delivery=True
        )

        def __init__(self) -> None:
            self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
            self.posts: list[str] = []

        async def turns(self):
            for text in declarations:
                yield InboundTurn(_LOCATION, ChannelMessage("Rill", text, self._principal))

        async def post(self, channel_id: str, text: str) -> None:
            del channel_id
            self.posts.append(text)

        async def close(self) -> None:
            return None

        async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
            rendered = tuple(views)
            return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

        async def collect_decision(self, views):
            view = tuple(views)[0]
            selection = "confirm" if view.kind == "confirmation" else view.options[0].id
            return self._principal, DecisionSubmission(
                presentation_token=view.presentation_token, selection_id=selection
            )

        async def acknowledge_decision(self, result) -> None:
            del result

    adapter = _Adapter()
    config = NarratorConfig(
        campaign_root=root, base_url=endpoint, model_id=model_id, max_decision_rounds=2
    )
    service = build_narrator_service(config, adapter)
    report = await service.run()
    checks = _score(root)
    checks["service_completed"] = report.turns == len(declarations) and not report.errors
    checks["errors"] = [str(error) for error in report.errors]
    checks["posts"] = len(adapter.posts)
    return checks


def _run_attempts(scenario: str, attempts: int, endpoint: str, model_id: str, work: Path) -> dict:
    environment = {
        **os.environ,
        "BSH_SERVER_PYTHON": os.environ.get("BSH_SERVER_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    results = []
    for attempt in range(attempts):
        root = work / f"{scenario}-{attempt}"
        result = subprocess.run(
            [
                sys.executable, "scripts/probe_statement_retirement.py", "--runtime",
                "--scenario", scenario, "--campaign-root", str(root),
                "--endpoint", endpoint, "--model-id", model_id,
            ],
            cwd=REPO_ROOT, env=environment, text=True, capture_output=True,
            timeout=900, check=False,
        )
        marker = next(
            (line.partition("=")[2] for line in result.stdout.splitlines()
             if line.startswith("RETIRE_RUNTIME=")),
            "",
        )
        try:
            checks = json.loads(marker)
        except json.JSONDecodeError:
            checks = {"retired": False, "stderr_tail": result.stderr[-2000:]}
        results.append({"attempt": attempt, "returncode": result.returncode, "checks": checks})
        print(
            f"{scenario} attempt {attempt}: retired={checks.get('retired')} "
            f"completed={checks.get('service_completed')}",
            file=sys.stderr, flush=True,
        )
    return {
        "attempts": results,
        "retired": sum(1 for r in results if r["checks"].get("retired")),
        "completed": sum(1 for r in results if r["checks"].get("service_completed")),
        "total": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--scenario", choices=sorted(_SCENARIOS), action="append")
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--report", default="/tmp/retire-probe.json")
    parser.add_argument("--work", default="/tmp/retire-probe-work")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if args.runtime:
        checks = asyncio.run(
            _runtime_flow(args.scenario[0], Path(args.campaign_root), endpoint, args.model_id)
        )
        print(f"RETIRE_RUNTIME={json.dumps(checks, sort_keys=True)}")
        return 0 if checks.get("service_completed") else 75
    model_id = _models(endpoint)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report = {"model_id": model_id, "endpoint": endpoint, "attempts_per_scenario": args.attempts}
    for scenario in args.scenario or sorted(_SCENARIOS):
        report[scenario] = _run_attempts(scenario, args.attempts, endpoint, model_id, work)
        print(
            f"{scenario}: retired {report[scenario]['retired']}/{args.attempts} "
            f"(completed {report[scenario]['completed']})",
            file=sys.stderr, flush=True,
        )
    Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"report_path={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
