"""Live acceptance for the durable-fiction Phase 2 refusal, against the served model.

The refusal only fires when a tool call opens fiction debt and ``session_close`` is
called before the engine's end-of-turn settle step runs -- in the recorded turn
lifecycle (``NarratorEngine.run_turn``) that step always runs after the model's own
tool-calling loop finishes, so the two must land inside the *same* turn to reproduce
the refusal at all. This probe seeds that precondition directly through ``GameService``
(the same \"bypass the model, then measure one live turn\" pattern
``tests_narrator/test_policy_live.py`` uses for its combat-recovery probe) before
sending one natural close message, rather than hoping a free-form turn happens to roll
and close in the same breath on its own.

What is measured: whether the model recovers cleanly -- either by resolving the named
debt (``scene_commit``) and retrying, or by passing ``accept_uncommitted=true`` -- and
whether the session actually closes, without looping on ``session_close`` and without
narrating the refusal's raw error code to the player (a violation of the skill's \"never
explain engine state to the player\" rule that nothing mechanical enforces since
``leaked_tool_name`` was removed; see DEFERRED.md's note on that removal).
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.channels.base import ChannelMessage, InboundTurn  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine  # noqa: E402

REPEATS = 10


@pytest.fixture
def engine(tmp_path):
    """A started engine bound to a throwaway campaign root, mirroring the fixture in
    ``test_policy_live.py``: the server child needs an interpreter holding mcp 2.x."""
    if not os.environ.get("BSH_SERVER_PYTHON"):
        pytest.skip("BSH_SERVER_PYTHON must name an interpreter holding mcp 2.x")

    import shutil
    import subprocess

    for directory in ("rules", "world"):
        source = REPO_ROOT / directory
        if source.is_dir():
            shutil.copytree(source, tmp_path / directory)

    subprocess.run(
        [os.environ["BSH_SERVER_PYTHON"], str(REPO_ROOT / "scripts" / "new_campaign.py"),
         "--root", str(tmp_path), "--title", "Narrator test", "--seed-scene"],
        check=True, capture_output=True, cwd=str(REPO_ROOT),
    )

    config = NarratorConfig(campaign_root=tmp_path, repo_root=REPO_ROOT)
    engine = NarratorEngine(config)
    engine.start()
    yield engine
    engine.stop()


@pytest.mark.live
@pytest.mark.asyncio
async def test_the_live_model_recovers_from_a_session_close_debt_refusal(engine):
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    from bsh_mcp.service import GameService

    root = engine.config.campaign_root
    game = GameService(root)
    created = game.character_create(
        discord_user_id="live-session-close-debt-test",
        name="Rill",
        origin="barbarian",
        backgrounds=["scout", "hunter", "survivor"],
        weapons=["long knife"],
    )
    assert created["ok"], created
    character_id = created["character_id"]
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: live-session-close-debt-test\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )

    results = []
    for i in range(REPEATS):
        # Each repeat runs on its own channel: a shared channel would carry prior
        # "let's end the session" exchanges into every later repeat's context,
        # contaminating the very thing being measured.
        channel_id = f"live-close-probe-{i}"

        rolled = game.attribute_test(
            character_id, "STR", f"probe roll {i}",
            "the crossing holds", "the crossing gives way",
        )
        assert rolled["ok"], rolled
        debt_seq = game.store.read_state().fiction_debt[-1].seq

        before_session = game.store.read_state().session
        turn = InboundTurn(
            channel_id=channel_id,
            mention=ChannelMessage(
                "Rill",
                "Let's end tonight's session here at the ferry landing. "
                f"Title it 'The Ferry Crossing {i}.'",
            ),
        )
        outcome = await engine.run_turn(turn)
        after_session = game.store.read_state().session

        close_events = [e for e in outcome.tool_events if e.get("tool") == "session_close"]
        results.append({
            "closed": after_session == before_session + 1,
            "close_attempts": len(close_events),
            "close_events": close_events,
            "tools_called": [e.get("tool") for e in outcome.tool_events],
            "error": outcome.error,
            "withheld": outcome.withheld,
            "narration": outcome.narration,
            "debt_survived": any(
                debt.seq == debt_seq for debt in game.store.read_state().fiction_debt
            ),
        })
        # Whatever this repeat did to the debt (ratified, waived, or left open and
        # accepted), the next repeat still expects to seed exactly one fresh entry.
        if game.store.read_state().fiction_debt:
            game.ledger_settle(reason="probe: clearing a straggler before the next repeat")

    closed = [r for r in results if r["closed"]]
    faulted = [r for r in results if r["error"]]
    looped = [r for r in results if r["close_attempts"] > 2]
    leaked_code = [
        r for r in results
        if "uncommitted_fiction_debt" in r["narration"] or "CampaignError" in r["narration"]
    ]
    # A closed-session claim with no session_close call behind it at all.
    phantom_claims = [
        r for r in results
        if not r["closed"] and "session is closed" in r["narration"].casefold()
    ]

    summary = (
        f"{len(closed)}/{REPEATS} closed, {len(faulted)} faulted, "
        f"{len(looped)} looped (>2 session_close attempts), "
        f"{len(leaked_code)} leaked the raw error code, "
        f"{len(phantom_claims)} phantom-claimed closure with no session_close call, "
        f"attempts per repeat: {[r['close_attempts'] for r in results]}"
    )

    import json
    Path("/tmp/session-close-debt-probe.json").write_text(json.dumps(results, indent=2))

    assert not phantom_claims, f"{summary}\nphantom-claim repeats: {phantom_claims}"
    assert not faulted, f"{summary}\nfaulted repeats: {faulted}"
    assert not leaked_code, f"{summary}\nleaked repeats: {leaked_code}"
    assert not looped, f"{summary}\nlooped repeats: {looped}"
    assert len(closed) >= REPEATS - 1, summary
