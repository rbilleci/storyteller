#!/usr/bin/env python3
"""Write redacted evidence that a bare answer to a question stays in conversation.

`narrator.interactions.classify_turn` decides the route deterministically, and unit tests
pin that decision. They cannot pin the exchange the decision exists to serve. In that
exchange the model asks the player something and the player answers in bare words. The
answer must then reach the narrator as conversation rather than the planner.

The scenario runs two turns against the live endpoint. The first greets an interlocutor.
The second sends a bare reply. The probe reports whether the first narration actually
ended by asking the player something, because that arming step belongs to the model rather
than to this slice. A run whose narration asked nothing reports a failed precondition
rather than a pass, so a green report never rests on an exchange that never happened.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsh_mcp.models import NPC
from bsh_mcp.testing import bootstrap_probe_campaign, candidate_tree, resolve_model_and_digest
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission
from narrator.interactions import narration_invites_reply
from narrator.service_assembly import build_narrator_service

GREETING = "hello orso pell"
REPLY = "Rill"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _container_image() -> dict:
    command = ["docker", "inspect", "--format", "{{.Config.Image}} {{.Image}}", "vllm-gemma"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    image, digest = result.stdout.strip().split(maxsplit=1)
    return {"image": image, "image_digest": digest}


def _prepare_campaign(root: Path) -> str:
    """Create one disposable campaign holding a present, named interlocutor."""
    game, character_id = bootstrap_probe_campaign(root, title="Probe", weapons=["long knife"])
    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.npcs["orso-pell"] = NPC(
            id="orso-pell", name="Orso Pell", level=1, hp=4, hp_max=4, damage=1,
            location_id="market",
        )
        transaction.state.scene.location_id = "market"
        transaction.state.scene.title = "Market"
        transaction.state.scene.present_npcs = ["orso-pell"]
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    return character_id


class _RuntimeProbeAdapter:
    """Send a greeting then a bare reply, retaining narration only for the arming test."""

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self.decision_kinds: list[str] = []
        self.posted: list[str] = []

    async def turns(self):
        for text in (GREETING, REPLY):
            yield InboundTurn("market", ChannelMessage("Rill", text, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        # The first narration's closing line is the arming signal this probe must judge,
        # so the text is retained in process and never written to the report.
        self.posted.append(text)

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


async def _runtime_flow(root: Path, endpoint: str, model_id: str) -> dict:
    """Run two real turns and report whether the reply stayed in conversation."""
    _prepare_campaign(root)
    adapter = _RuntimeProbeAdapter()
    config = NarratorConfig(
        campaign_root=root, base_url=endpoint, model_id=model_id, max_decision_rounds=2
    )
    service = build_narrator_service(config, adapter)

    # Counting planner entries is the only direct read of the routing decision available
    # from outside the service, which owns the policy and never returns it.
    planner_calls = 0
    original_plan = service.engine.plan_turn

    async def counting_plan(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return await original_plan(*args, **kwargs)

    service.engine.plan_turn = counting_plan
    report = await service.run()
    first = adapter.posted[0] if adapter.posted else ""
    return {
        "service_completed": report.turns == 2 and not report.errors,
        "both_turns_delivered": report.delivered == 2 and report.withheld == 0,
        "first_narration_invited_reply": narration_invites_reply(first),
        "planner_never_invoked": planner_calls == 0,
        "no_decision_presented": adapter.decision_kinds == [],
    }


def _run_runtime(root: Path, endpoint: str, model_id: str) -> int:
    checks = asyncio.run(_runtime_flow(root, endpoint, model_id))
    print(f"SOCIAL_REPLY_RUNTIME={json.dumps(checks, sort_keys=True)}")
    return 0 if all(checks.values()) else 75


def _live_scenario(run_directory: Path, endpoint: str, model_id: str) -> dict:
    """Retry the exchange, because the arming question belongs to the model.

    A retry never inherits a campaign. The returned checks come from the last attempt, so
    a run that never armed reports the failed precondition rather than hiding it.
    """
    narrator_python = os.environ.get(
        "NARRATOR_PYTHON",
        str(Path(__file__).resolve().parents[1] / ".narrator-venv" / "bin" / "python"),
    )
    environment = {
        **os.environ,
        "BSH_SERVER_PYTHON": os.environ.get("BSH_SERVER_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    repository = Path(__file__).resolve().parents[1]
    checks: dict = {}
    for attempt in range(3):
        campaign_root = run_directory / f"social-reply-{attempt}"
        result = subprocess.run(
            [
                narrator_python, "scripts/probe_social_reply.py", "--runtime",
                "--campaign-root", str(campaign_root),
                "--endpoint", endpoint,
                "--model-id", model_id,
            ],
            cwd=repository, env=environment, text=True, capture_output=True,
            timeout=300, check=False,
        )
        marker = next(
            (
                line.partition("=")[2]
                for line in result.stdout.splitlines()
                if line.startswith("SOCIAL_REPLY_RUNTIME=")
            ),
            "",
        )
        try:
            checks = json.loads(marker)
        except json.JSONDecodeError:
            checks = {}
        if result.returncode == 0 and checks and all(checks.values()):
            return {"attempts": attempt + 1, "checks": checks}
    if not checks:
        raise ValueError("the social reply scenario produced no marker")
    return {"attempts": 3, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.runtime:
        if not args.campaign_root or not args.model_id:
            print("runtime requires --campaign-root and --model-id", file=sys.stderr)
            return 2
        return _run_runtime(Path(args.campaign_root), args.endpoint.rstrip("/"), args.model_id)
    run_directory = os.environ.get("BSH_PROBE_RUN_DIRECTORY", "")
    if not run_directory:
        print("HELD: BSH_PROBE_RUN_DIRECTORY is required.", file=sys.stderr)
        return 75
    destination = Path(run_directory) / "social-reply-probe.json"
    if destination.exists():
        print("HELD: the run-bound report already exists.", file=sys.stderr)
        return 75
    report = {
        "status": "PASS",
        "candidate_tree": candidate_tree(Path(run_directory)),
        "command_sha256": _sha256(" ".join(sys.argv).encode()),
        "launcher_sha256": (_sha256(Path(os.environ["BSH_MODEL_LAUNCHER"]).read_bytes())
                            if os.environ.get("BSH_MODEL_LAUNCHER") else None),
        "checks": {},
        "outputs": [],
        "offline": args.offline,
    }
    if not args.offline:
        try:
            endpoint = args.endpoint.rstrip("/")
            model = resolve_model_and_digest(endpoint)
            report.update(model)
            report.update(_container_image())
            result = _live_scenario(Path(run_directory), endpoint, model["model_id"])
            report["checks"] = {"social_reply": result}
            failed = [name for name, value in result["checks"].items() if value is not True]
            report["failed_checks"] = failed
            report["status"] = "PASS" if not failed else "FAIL"
        except (OSError, ValueError, urllib.error.URLError, subprocess.SubprocessError) as error:
            print(
                f"HELD: production endpoint evidence is unavailable: {type(error).__name__}",
                file=sys.stderr,
            )
            return 75
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"report_path={destination}")
    print(f"report_sha256={_sha256(destination.read_bytes())}")
    return 0 if report["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
