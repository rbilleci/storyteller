#!/usr/bin/env python3
"""Live probe: a full multi-round fight, checked for turn-order faults and false narration.

Four checks, none of which existed before this file:

* ``no_turn_integrity_faults`` -- none of this run's own tool calls hit
  ``not_active_actor``, ``turn_not_open``, ``turn_already_open``,
  ``npc_no_actions_remaining``, or ``no_actions_remaining``. The engine guard makes
  the underlying state safe either way; this asks the sharper question of whether the
  narrator still needed the guard at all, which is the usability half of the defect.
* ``rounds_monotonic`` and ``combat_progressed`` -- read from the audit log and the
  final campaign state directly, never from the adapter, matching the read-the-record
  discipline every scenario in ``probe_combat_decisions.py`` already uses.
* ``hp_claim_parsed`` and ``every_hp_claim_matches_state`` -- the script asks, out of
  character, how hurt the enemy is, twice across the fight. Each reply is reduced in
  place to a typed claim (an explicit fraction, or an unambiguous \"at full health\"),
  compared against ``campaign/state.json``'s own NPC record at that exact moment, and
  discarded; no reply prose survives past ``post()``. A run whose replies never state
  anything parseable reports that honestly (``hp_claim_parsed`` false) rather than
  passing on an untested premise.
* ``every_announced_verdict_matches`` / ``every_announced_target_matches`` /
  ``no_unbacked_announcement`` -- the same dice-truth comparator
  (``narrator.engine.verdict_mismatches``) the ``mechanical-truth`` scenario uses,
  run across every turn of the fight instead of two.

Run directly under the narrator interpreter, against the shared vLLM endpoint this
project already runs (see CONTRIBUTING.md -- never started or restarted by this script):

    .narrator-venv/bin/python scripts/probe_combat_integrity.py

Exits 0 when every check passes, 1 otherwise, and prints the full result as JSON on
stdout. No campaign or model prose is retained in that JSON beyond the typed fields
the docstring above names.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_combat_decisions import _announcement_text, _recorded_roll_facts

from bsh_mcp.models import NPC
from bsh_mcp.store import CampaignStore
from bsh_mcp.testing import bootstrap_probe_campaign, resolve_model_and_digest
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission
from narrator.engine import announced_rolls, verdict_mismatches
from narrator.service_assembly import build_narrator_service

_NPC_ID = "reed-thug"
_NPC_NAME = "Reed Thug"

#: Tool-call error codes that fire only when the narrator has lost track of whose
#: turn it is or how many actions an actor has left this turn. A clean run shows
#: none of them. ``turn_already_open`` and ``npc_no_actions_remaining`` are the two
#: this project's own combat engine did not check at all before the fix this probe
#: guards; ``not_active_actor``, ``turn_not_open``, and ``no_actions_remaining``
#: were already enforced and are included for completeness, matching every error
#: code ``combat_begin_turn``, ``combat_attack``, and ``combat_defend`` can raise
#: for this exact reason.
_TURN_INTEGRITY_ERRORS = frozenset(
    {
        "not_active_actor",
        "turn_not_open",
        "turn_already_open",
        "npc_no_actions_remaining",
        "no_actions_remaining",
    }
)

#: One scripted line per turn. Odd-numbered fiction turns keep the fight moving;
#: the two out-of-character turns are the live half of the false-health check,
#: asked once after the fight's first exchange and once more after a second, so a
#: narrator that recovers a correct number once but drifts on the second attempt is
#: still caught. Nothing here is a mechanic declaration a tool must obey -- the
#: probe measures how the narrator actually handles an ordinary fight, not a single
#: pinned obligation the way ``probe_combat_decisions.py``'s scenarios do.
_DECLARATIONS: tuple[str, ...] = (
    "I press the attack and stab the reed thug with my long knife.",
    "I brace myself and watch for its counter, ready to dodge or parry.",
    "@GM how badly hurt is the reed thug right now?",
    "I press the attack again, going for another strike.",
    "I brace myself and watch for its counter, ready to dodge or parry.",
    "@GM how badly hurt is the reed thug right now, exactly?",
)

_HP_SLASH_RE = re.compile(r"(\d{1,3})\s*/\s*(\d{1,3})\s*(?:hit points?|hp\b)", re.IGNORECASE)


_HP_OF_RE = re.compile(
    r"(\d{1,3})\s*(?:hit points?|hp)?\s*(?:remaining\s*)?(?:of|out of)\s*(\d{1,3})\s*"
    r"(?:hit points?|hp)?",
    re.IGNORECASE,
)
_FULL_HEALTH_RE = re.compile(
    r"\b(?:full health|unharmed|unwounded|uninjured|unscathed|at full strength)\b",
    re.IGNORECASE,
)
#: The fallback shape: a current figure with no stated maximum ("has 5 hit points
#: remaining."). Tried last, after every two-number pattern above, so it never
#: swallows just the first half of a fraction those already parse in full.
_HP_CURRENT_ONLY_RE = re.compile(
    r"\b(\d{1,3})\s*(?:hit points?|hp)\s*(?:remaining|left)?\b", re.IGNORECASE
)


def _parse_hp_claim(text: str) -> dict | None:
    """Reduce one delivered reply to a typed hit-point claim, or ``None``.

    Deliberately narrow: a claim is counted only for an explicit number against a
    real hit-point phrase, an unambiguous "at full health" phrase, or -- as a last
    resort -- a bare current figure with no stated maximum. A reply this cannot
    parse is not assumed correct -- it is a distinct, honest "no claim parsed"
    outcome (``hp_claim_parsed`` in the returned checks), never silently dropped
    into a pass the way an unconditional ``True`` default would.
    """
    match = _HP_SLASH_RE.search(text) or _HP_OF_RE.search(text)
    if match:
        return {"kind": "fraction", "current": int(match.group(1)), "max": int(match.group(2))}
    if _FULL_HEALTH_RE.search(text):
        return {"kind": "full"}
    match = _HP_CURRENT_ONLY_RE.search(text)
    if match:
        return {"kind": "current_only", "current": int(match.group(1))}
    return None


def _prepare_campaign(root: Path) -> str:
    """One player character and one NPC, close range, combat already open.

    Hit points sit well above what one exchange resolves (Black Sword Hack weapon
    damage is a few points a hit), so the fight survives long enough for both
    out-of-character health checks to land against a real, changing number rather
    than a fight that ended before the second question.
    """
    game, character_id = bootstrap_probe_campaign(
        root, title="Combat Integrity Probe", weapons=["long knife"]
    )
    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.npcs[_NPC_ID] = NPC(
            id=_NPC_ID, name=_NPC_NAME, level=2, hp=12, hp_max=12, damage=3,
            location_id="docks",
        )
        transaction.state.scene.present_npcs = [_NPC_ID]
        transaction.state.scene.location_id = "docks"
        transaction.state.scene.title = "Docks"
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    opened = game.combat_start(
        pc_ids=[character_id],
        npc_ids=[_NPC_ID],
        initial_ranges={_NPC_ID: "close"},
        reason="probe",
    )
    if not opened.get("ok"):
        raise ValueError("probe fight did not open")
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    return character_id


class _CombatIntegrityAdapter:
    """Drive the scripted fight, retaining only typed, reduced signals.

    No raw model prose survives a ``post`` call: each reply is reduced, in place, to
    an announced-roll list, an engine-notice classification against the production
    ``NarratorConfig`` strings, and -- only on a turn whose text carries a
    hit-point claim -- one typed claim already compared against
    ``campaign/state.json``'s own NPC record at that exact moment. The local
    ``text`` variable is never assigned to an attribute, matching the redaction
    discipline every adapter in ``probe_combat_decisions.py`` keeps.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self, root: Path, npc_id: str) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._root = root
        self._npc_id = npc_id
        self.decision_kinds: list[str] = []


        self.output_categories: list[str] = []
        #: Every attempted tool call this run, successful or refused.
        self.tool_calls: list[dict] = []
        self.announced: list[dict] = []
        #: One entry per reply whose text carried a parseable hit-point claim.
        self.hp_checks: list[dict] = []
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call":
            self.tool_calls.append(
                {
                    "tool": str(fields.get("tool", "")),
                    "ok": bool(fields.get("ok")),
                    "error": str(fields.get("error", "")),
                }
            )

    async def turns(self):
        for text in _DECLARATIONS:
            yield InboundTurn("docks", ChannelMessage("Rill", text, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.output_categories.append(None)
        self.announced.extend(announced_rolls(text))
        claim = _parse_hp_claim(text)
        if claim is not None:
            npc = CampaignStore(self._root).read_state().npcs.get(self._npc_id)
            actual_hp = npc.hp if npc is not None else None
            actual_hp_max = npc.hp_max if npc is not None else None
            if claim["kind"] == "fraction":
                matches = claim["current"] == actual_hp and claim["max"] == actual_hp_max
            elif claim["kind"] == "current_only":
                matches = actual_hp is not None and claim["current"] == actual_hp
            else:
                matches = actual_hp is not None and actual_hp == actual_hp_max
            self.hp_checks.append(
                {
                    "claim": claim,
                    "actual_hp": actual_hp,
                    "actual_hp_max": actual_hp_max,
                    "matches": matches,
                }
            )

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


def _round_sequence(root: Path) -> list[int]:
    """The ``round`` field from every begin/end-turn audit event, in event order.

    Read from the record rather than the adapter, the way the disengage and
    no-phantom-scene scenarios already do: it is immune to a narrator reaching a
    correct state through a call shape this probe did not anticipate.
    """
    rounds: list[int] = []
    for event in CampaignStore(root).read_events(limit=500):
        if isinstance(event, dict) and event.get("tool") in (
            "combat_begin_turn",
            "combat_end_turn",
        ):
            value = event.get("round")
            if isinstance(value, int):
                rounds.append(value)
    return rounds


def _longest_notice_run(categories: list[str]) -> int:
    """The longest consecutive run of engine notices in a reply sequence.
    """
    longest = current = 0
    for category in categories:
        current = current + 1 if category == "notice" else 0
        longest = max(longest, current)
    return longest


async def _runtime_flow(root: Path, endpoint: str, model_id: str) -> dict:
    character_id = _prepare_campaign(root)
    del character_id
    config = NarratorConfig(campaign_root=root, base_url=endpoint, model_id=model_id)
    initial_hp = CampaignStore(root).read_state().npcs[_NPC_ID].hp

    adapter = _CombatIntegrityAdapter(root, _NPC_ID)
    service = build_narrator_service(config, adapter)
    report = await service.run()
    if len(service.post_log) != len(adapter.output_categories):
        raise RuntimeError(
            f"post log carries {len(service.post_log)} posts for "
            f"{len(adapter.output_categories)} recorded replies; notice attribution is broken"
        )
    adapter.output_categories = [
        "notice" if post.kind == "notice" else "narration" for post in service.post_log
    ]

    final_state = CampaignStore(root).read_state()
    final_npc = final_state.npcs.get(_NPC_ID)
    combat_progressed = (
        not final_state.combat.active
        or final_npc is None
        or final_npc.hp < initial_hp
    )

    rounds = _round_sequence(root)
    rounds_monotonic = all(a <= b for a, b in zip(rounds, rounds[1:]))

    turn_integrity_faults = [
        call
        for call in adapter.tool_calls
        if not call["ok"] and call["error"] in _TURN_INTEGRITY_ERRORS
    ]

    recorded = _recorded_roll_facts(root)
    mismatches = verdict_mismatches(_announcement_text(adapter.announced), recorded)
    mismatch_kinds = [entry["kind"] for entry in mismatches]

    hp_mismatches = [entry for entry in adapter.hp_checks if not entry["matches"]]

    checks = {
        "service_completed": report.turns == len(_DECLARATIONS),
        "no_turn_integrity_faults": len(turn_integrity_faults) == 0,
        "rounds_monotonic": rounds_monotonic,
        "combat_progressed": combat_progressed,
        "no_excessive_withheld_run": _longest_notice_run(adapter.output_categories) < 3,
        "hp_claim_parsed": len(adapter.hp_checks) >= 1,
        "every_hp_claim_matches_state": len(hp_mismatches) == 0,
        "every_announced_verdict_matches": mismatch_kinds.count("verdict") == 0,
        "every_announced_target_matches": mismatch_kinds.count("target") == 0,
        "no_unbacked_announcement": mismatch_kinds.count("unbacked") == 0,
    }
    return {
        "checks": checks,
        "turn_integrity_faults": turn_integrity_faults,
        "rounds": rounds,
        "hp_checks": adapter.hp_checks,
        "initial_hp": initial_hp,
        "final_hp": final_npc.hp if final_npc is not None else None,
        "delivered": report.delivered,
        "withheld": report.withheld,
        "turns": report.turns,
        "output_categories": adapter.output_categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    try:
        model_id = args.model_id or resolve_model_and_digest(endpoint)["model_id"]
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"HELD: endpoint is unavailable: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    owns_root = args.campaign_root is None
    root = Path(args.campaign_root) if args.campaign_root else Path(
        tempfile.mkdtemp(prefix="combat-integrity-probe-")
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(
            asyncio.wait_for(_runtime_flow(root, endpoint, model_id), timeout=args.timeout)
        )
    finally:
        if owns_root:
            shutil.rmtree(root, ignore_errors=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
