#!/usr/bin/env python3
"""Three things changed that day, and each is model-dependent in a way the offline
suites cannot settle: the engine now announces a runic weapon's start-of-session INT
test under the weapon's name (``narrator.engine.ANNOUNCED_ROLL_TOOLS`` gained
``grant_runic_weapon`` and ``session_close``), ``inventory_update`` lets a character
put a runic weapon down, and three tool descriptions gained a sentence each telling
the model so. What the offline tests prove is that *if* the tool is called, the line
is built and the field is cleared. What only the endpoint can show is whether the
live narrator calls the tool at all when a player asks, and whether the injected
line survives ``narrator.delivery`` to reach the adapter.

Three scenarios, each run ``--attempts`` times on a fresh disposable campaign
(``tests_narrator`` measured two repeats inverting twice in one session, so ten is
the floor), each scored from the audit log beside the retained narration, the same
discipline ``scripts/probe_world_state.py`` established:

``grant``
    The player asks the GM to grant a named runic weapon now. Pass: the audit log
    holds a ``grant_runic_weapon`` event, and the posted narration carries exactly
    one announcement whose attribute, total, target and outcome match that event's
    own ``session_test``, named for the weapon.

``close``
    The weapon is granted directly, then the player asks the GM to close the session
    with every argument supplied. Pass: a ``session_close`` event whose
    ``runic_session_tests`` entry matches one announcement in the posted narration.

Nothing here starts or restarts the endpoint. Usage::

    export BSH_SERVER_PYTHON=\"$(uv run python -c 'import sys; print(sys.executable)')\"
    PYTHONDONTWRITEBYTECODE=1 .narrator-venv/bin/python \\
        scripts/probe_runic_weapon.py --attempts 10 --report /tmp/runic-probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bsh_mcp.service import GameService  # noqa: E402
from bsh_mcp.testing import bootstrap_probe_campaign, resolve_model_id  # noqa: E402

_WEAPON = "Sorrow"
_PERSONALITY = "brutal"
_LOCATION = "the-road-shrine"

_SCENARIOS = {
    "grant": [
        "GM, a direct instruction: the black blade in the shrine's reliquary is a "
        f"runic weapon -- {_PERSONALITY}, named {_WEAPON}. Rill takes it up. Grant it "
        "to her now with the tool, in this reply.",
    ],
    "drop": [
        f"I won't carry this thing. I set {_WEAPON} down on the shrine's altar and "
        "leave it there for good.",
        f"GM, straight answer: is {_WEAPON} still in Rill's hands, or on the altar? "
        "Record whichever it is.",
    ],
    "close": [
        "GM: we stop here. Close the session now with the tool -- title 'The Black "
        f"Blade', public summary 'Rill took up the runic blade {_WEAPON} at the road "
        "shrine.', no Stories awarded, no open hooks.",
    ],
}


_FALL_WORDS = re.compile(
    r"\b(helpless|falls?|fallen|falling|goes? down|brought low|laid low|0 hit points|"
    r"zero hit points|at (?:its|her|his) mercy|should (?:she|he|rill) (?:fall|drop))\b",
    re.IGNORECASE,
)
_FATE_WORDS = re.compile(
    r"\b(kill|kills|killed|claim|claims|die|dies|death|devour|devours|spare|spares|spared|"
    r"safe|no claim|no hold|let (?:her|him) live|stay(?:s|ed)? (?:its|her|his) hand|"
    r"will not (?:kill|claim|take)|won't (?:kill|claim|take)|dormant|sleeps|slumbers)\b",
    re.IGNORECASE,
)
_ANNOUNCEMENT_LINE = re.compile(r"^.*\brolls INT:.*$", re.MULTILINE)
#: The engine's own stake line (``narrator.engine.runic_stake_line``, English
#: catalog), excluded from the model-prose score the same way the announcement is.
_STAKE_LINE = re.compile(r"^.*\btaken the measure of\b.*$", re.MULTILINE)


def stake_stated(post: str) -> bool:
    """Whether the *model's* prose states the stake, engine lines excluded."""
    body = _STAKE_LINE.sub("", _ANNOUNCEMENT_LINE.sub("", post))
    return bool(_FALL_WORDS.search(body) and _FATE_WORDS.search(body))


def stake_line_present(post: str, armed: bool) -> bool:
    """Whether the engine's stake line for this verdict reached the adapter."""
    line = _STAKE_LINE.search(post)
    if line is None:
        return False
    return ("has not taken" in line.group(0)) is (not armed)


def _prepare_campaign(root: Path, scenario: str) -> str:
    game, character_id = bootstrap_probe_campaign(root, title="Runic probe", weapons=["long knife"])
    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.scene.location_id = _LOCATION
        transaction.state.scene.title = "The Road Shrine"
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    if scenario in ("drop", "close"):
        granted = game.grant_runic_weapon(character_id, name=_WEAPON, personality=_PERSONALITY)
        if not granted.get("ok"):
            raise ValueError(f"probe grant was unavailable: {granted}")
    return character_id


def _score(scenario: str, root: Path, posts: list[str], character_id: str) -> dict:
    from narrator.engine import announced_rolls

    game = GameService(root)
    events = [e for e in game.store.read_events(limit=200) if isinstance(e, dict)]
    character = game.store.read_character(character_id)
    all_lines = [line for post in posts for line in announced_rolls(post)]
    checks: dict = {"posts": posts, "announced": all_lines}

    def _matches(roll: dict, outcome: str) -> list[dict]:
        return [
            line for line in all_lines
            if line["attribute"] == "INT"
            and line["total"] == roll.get("total")
            and line["target"] == roll.get("target")
            and line["outcome"] == outcome
        ]

    if scenario == "grant":
        grant = next((e for e in events if e.get("tool") == "grant_runic_weapon"), None)
        checks["tool_called"] = grant is not None
        if grant is not None:
            from narrator.engine import _classify_roll_under

            roll = grant["session_test"]
            outcome = _classify_roll_under(roll["selected"], roll["total"], roll["target"])
            matched = _matches(roll, outcome)
            checks["session_test"] = {**roll, "outcome": outcome}
            checks["announced_once"] = len(matched) == 1


            checks["extra_announcements"] = len(
                _ANNOUNCEMENT_LINE.findall("\n".join(posts))
            ) - len(matched)
            checks["named_for_weapon"] = bool(matched) and matched[0]["character"] == grant["name"]
            checks["weapon_name"] = grant["name"]
            checks["stake_stated"] = any(stake_stated(post) for post in posts)
            checks["stake_line_present"] = any(
                stake_line_present(post, bool(grant.get("kills_helpless"))) for post in posts
            )
            checks["passed"] = (
                checks["announced_once"] and checks["named_for_weapon"]
                and checks["stake_line_present"] and checks["extra_announcements"] == 0
            )
        else:
            checks["passed"] = False
    elif scenario == "drop":
        updates = [e for e in events if e.get("tool") == "inventory_update"]
        relinquished = [e for e in updates if e.get("runic_weapon_relinquished")]
        checks["inventory_update_calls"] = len(updates)
        checks["relinquished_in_log"] = bool(relinquished)
        checks["weapon_cleared"] = character.runic_weapon is None
        checks["refusals"] = [
            e.get("message") for e in events if e.get("error") == "item_not_held"
        ]
        checks["passed"] = checks["relinquished_in_log"] and checks["weapon_cleared"]
    elif scenario == "close":
        closed = next((e for e in events if e.get("tool") == "session_close"), None)
        checks["tool_called"] = closed is not None
        if closed is not None:
            tests = closed.get("runic_session_tests") or []
            checks["re_rolls"] = tests
            if tests:
                entry = tests[0]
                matched = _matches(entry["roll"], entry["outcome"])
                checks["announced_once"] = len(matched) == 1
                checks["named_for_weapon"] = (
                    bool(matched) and matched[0]["character"] == entry["weapon_name"]
                )
                checks["stake_line_present"] = any(
                    stake_line_present(post, bool(entry.get("kills_helpless"))) for post in posts
                )
                checks["passed"] = (
                    checks["announced_once"] and checks["named_for_weapon"]
                    and checks["stake_line_present"]
                )
            else:
                checks["passed"] = False
        else:
            checks["passed"] = False
    return checks


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

    character_id = _prepare_campaign(root, scenario)
    declarations = _SCENARIOS[scenario]

    class _Adapter:
        name = "terminal"
        decision_capabilities = ChannelCapabilities(
            structured_decisions=True, atomic_decision_delivery=True
        )

        def __init__(self) -> None:
            self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
            self.posts: list[str] = []
            self.cleared_after_turn: list[bool] = []
            self.decision_kinds: list[str] = []
            self.decision_views: list[dict] = []

        async def turns(self):
            for text in declarations:
                yield InboundTurn(_LOCATION, ChannelMessage("Rill", text, self._principal))

        async def post(self, channel_id: str, text: str) -> None:
            del channel_id
            self.posts.append(text)
            # Read the record beside each retained post, so the drop scenario can
            # say which turn cleared the field.
            self.cleared_after_turn.append(
                GameService(root).store.read_character(character_id).runic_weapon is None
            )

        async def close(self) -> None:
            return None

        async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
            rendered = tuple(views)
            self.decision_kinds.extend(view.kind for view in rendered)
            # Retained so a withheld turn can be read beside the decision that
            # withheld it: the planner's own question and options, never the
            # routing token.
            self.decision_views.extend(
                {
                    "kind": view.kind,
                    "question": view.question,
                    "context": view.context,
                    "options": [
                        {"id": option.id, "label": option.label,
                         "description": getattr(option, "description", "")}
                        for option in view.options
                    ],
                }
                for view in rendered
            )
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
    checks = _score(scenario, root, adapter.posts, character_id)
    # The engine's own verdict record per turn: how many model-written
    # announcements contradicted the facts and were cut before delivery.
    checks["verdict_log"] = [
        {k: entry.get(k) for k in ("triggered", "retried", "resolved", "kinds",
                                   "announcements_scrubbed")}
        for entry in getattr(service.engine, "_verdict_log", [])
    ]
    checks["landed_on_turn"] = next(
        (index + 1 for index, cleared in enumerate(adapter.cleared_after_turn) if cleared),
        None,
    )
    checks["service_completed"] = report.turns == len(declarations) and not report.errors
    checks["errors"] = [str(e) for e in report.errors]
    checks["decision_kinds"] = adapter.decision_kinds
    checks["decision_views"] = adapter.decision_views
    checks["thinking_level"] = config.turn_thinking_level
    return checks


def _run_attempts(scenario: str, attempts: int, endpoint: str, model_id: str, work: Path) -> dict:
    narrator_python = sys.executable
    environment = {
        **os.environ,
        "BSH_SERVER_PYTHON": os.environ.get("BSH_SERVER_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    repository = Path(__file__).resolve().parents[1]
    results = []
    for attempt in range(attempts):
        root = work / f"{scenario}-{attempt}"
        result = subprocess.run(
            [
                narrator_python, "scripts/probe_runic_weapon.py", "--runtime",
                "--scenario", scenario, "--campaign-root", str(root),
                "--endpoint", endpoint, "--model-id", model_id,
            ],
            cwd=repository, env=environment, text=True, capture_output=True,
            timeout=900, check=False,
        )
        marker = next(
            (line.partition("=")[2] for line in result.stdout.splitlines()
             if line.startswith("RUNIC_RUNTIME=")),
            "",
        )
        try:
            checks = json.loads(marker)
        except json.JSONDecodeError:
            checks = {"passed": False, "stderr_tail": result.stderr[-2000:]}
        results.append({"attempt": attempt, "returncode": result.returncode, "checks": checks})
        print(
            f"{scenario} attempt {attempt}: passed={checks.get('passed')} "
            f"completed={checks.get('service_completed')}",
            file=sys.stderr, flush=True,
        )
    passed = sum(1 for r in results if r["checks"].get("passed"))
    completed = sum(1 for r in results if r["checks"].get("service_completed"))
    stake = sum(1 for r in results if r["checks"].get("stake_stated"))
    extra = sum(1 for r in results if r["checks"].get("extra_announcements"))
    scrubbed = sum(
        1 for r in results
        if any(e.get("announcements_scrubbed") for e in r["checks"].get("verdict_log", []))
    )
    retried = sum(
        1 for r in results
        if any(e.get("retried") for e in r["checks"].get("verdict_log", []))
    )
    return {
        "attempts": results, "passed": passed, "total": attempts,
        "completed": completed, "stake_stated": stake,
        "turns_with_extra_announcements": extra,
        "turns_with_verdict_retry": retried,
        "turns_with_scrubbed_announcements": scrubbed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--scenario", choices=sorted(_SCENARIOS), action="append")
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--report", default="/tmp/runic-probe.json")
    parser.add_argument("--work", default="/tmp/runic-probe-work")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if args.runtime:
        checks = asyncio.run(
            _runtime_flow(args.scenario[0], Path(args.campaign_root), endpoint, args.model_id)
        )
        print(f"RUNIC_RUNTIME={json.dumps(checks, sort_keys=True)}")
        return 0 if checks.get("service_completed") else 75
    model_id = resolve_model_id(endpoint)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scenarios = args.scenario or sorted(_SCENARIOS)
    report = {"model_id": model_id, "endpoint": endpoint, "attempts_per_scenario": args.attempts}
    for scenario in scenarios:
        report[scenario] = _run_attempts(scenario, args.attempts, endpoint, model_id, work)
        print(
            f"{scenario}: {report[scenario]['passed']}/{args.attempts} "
            f"(completed {report[scenario]['completed']}, stake stated "
            f"{report[scenario]['stake_stated']}, extra announcements in "
            f"{report[scenario]['turns_with_extra_announcements']}, verdict retries "
            f"{report[scenario]['turns_with_verdict_retry']}, scrubbed "
            f"{report[scenario]['turns_with_scrubbed_announcements']})",
            file=sys.stderr, flush=True,
        )
    Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"report_path={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
