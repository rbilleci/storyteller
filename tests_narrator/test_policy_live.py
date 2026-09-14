"""Assert the narrator's real tool surface against a live Model Context Protocol server.

These tests need `strands-agents`, which requires `mcp<2.0.0`, while the project pins
`mcp[cli]>=2,<3`. They therefore run in a second environment and cannot live under
`tests/`, where collection in the main environment would fail on import.

The check that matters is set equality against a server the test actually spawns. A
mock would pin our idea of the tool surface; this pins the real one. It is the
replacement for the Hermes profile's allowlist and its two declined ledgers, and unlike
those it fails closed: a tool that appears without being named breaks the test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator import policy  # noqa: E402
from narrator.channels.base import ChannelMessage, ChannelPrincipal, InboundTurn  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import DISCLOSED_SKILLS, NarratorEngine  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    """A started engine bound to a throwaway campaign root.

    The server child needs an interpreter holding mcp 2.x, which `BSH_SERVER_PYTHON`
    supplies. The acceptance environment is that interpreter.
    """
    if not os.environ.get("BSH_SERVER_PYTHON"):
        pytest.skip("BSH_SERVER_PYTHON must name an interpreter holding mcp 2.x")

    import shutil
    import subprocess

    for directory in ("rules", "world"):
        source = REPO_ROOT / directory
        if source.is_dir():
            shutil.copytree(source, tmp_path / directory)

    # The server binds to an initialised campaign, so seed one the same way
    # ``scripts/demo_sandbox.sh`` does. Without it the child exits and the client
    # reports only "Connection closed".
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


def test_the_live_tool_surface_equals_the_frozen_manifest(engine):
    """The server's real tools: the model-facing set plus the two engine-only tools."""
    names = {
        getattr(tool, "tool_name", None) or getattr(tool, "name", "")
        for tool in engine._tools
    }
    assert names == policy.MCP_SERVED_TOOLS
    assert len(names) == len(policy.MCP_TOOLS) + len(policy.ENGINE_ONLY_TOOLS)
    assert {"ledger_settle", "ability_apply_ruling"} <= names


def test_no_forbidden_tool_reaches_the_agent(engine):
    """Strands ships bash, file_editor and http_request. None may be registered."""
    agent = engine._agent_for("test-channel")
    registered = set(agent.tool_names)
    assert not registered & policy.FORBIDDEN_TOOLS
    # The engine-only settle tool is served but must never register on the model's
    # agent: a prompt-injected waive would erase pending outcomes on request.
    assert "ledger_settle" not in registered
    policy.assert_tool_surface(registered)


def test_start_refuses_a_violating_surface_before_any_channel_opens(tmp_path, engine):
    """The ordering guarantee, proved rather than asserted in a docstring.

    An audit found the surface check running lazily on a channel's first turn, so a
    channel that yielded zero turns never ran it. This test breaks the manifest and
    requires ``start`` itself to raise, with zero agents cached and the session closed.
    """
    config = NarratorConfig(campaign_root=engine.config.campaign_root, repo_root=REPO_ROOT)
    victim = NarratorEngine(config)

    original = policy.PLAYER_FACING_TOOLS
    policy.PLAYER_FACING_TOOLS = original | {"bash"}
    try:
        with pytest.raises(policy.ToolSurfaceError):
            victim.start()
    finally:
        policy.PLAYER_FACING_TOOLS = original
        victim.stop()

    assert victim._agents == {}
    assert victim._client is None


def test_only_the_two_episode_skills_are_disclosed(engine):
    """bsh-gm is concatenated, not disclosed; bsh-worldsmith never reaches players."""
    assert DISCLOSED_SKILLS == (
        "skills/bsh-session-zero",
        "skills/bsh-session-close",
    )
    assert not any("worldsmith" in path for path in DISCLOSED_SKILLS)
    assert not any("bsh-gm" in path for path in DISCLOSED_SKILLS)


def test_character_create_binds_to_the_turns_authenticated_principal(engine):
    """The account id a new character answers to comes from the engine, not the model.

    The model's copy of an account identifier is hearsay -- it can only repeat what a
    prompt showed it, and a prompt-injected turn could type someone else's. The
    ``BeforeToolCallEvent`` hook must therefore overwrite ``discord_user_id`` with the
    authenticated subject of the turn being run, leave every other tool alone, and
    leave the model's value alone when no principal exists (the pre-principal replay
    grammar). Exercised through the agent's real hook registry, so the wiring inside
    ``_agent_for`` is what this proves, not a helper called in isolation.
    """
    from strands.hooks import BeforeToolCallEvent

    agent = engine._agent_for("bind-channel")
    engine._turn_principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
    try:
        creation = {
            "toolUseId": "bind-1",
            "name": "character_create",
            "input": {"discord_user_id": "someone-else", "name": "Kara"},
        }
        agent.hooks.invoke_callbacks(
            BeforeToolCallEvent(
                agent=agent, selected_tool=None, tool_use=creation, invocation_state={}
            )
        )
        assert creation["input"]["discord_user_id"] == "terminal-player"

        # An omitted identifier is injected rather than left to fail validation.
        omitted = {"toolUseId": "bind-2", "name": "character_create", "input": {"name": "Kara"}}
        agent.hooks.invoke_callbacks(
            BeforeToolCallEvent(
                agent=agent, selected_tool=None, tool_use=omitted, invocation_state={}
            )
        )
        assert omitted["input"]["discord_user_id"] == "terminal-player"

        other = {
            "toolUseId": "bind-3",
            "name": "npc_create",
            "input": {"discord_user_id": "someone-else"},
        }
        agent.hooks.invoke_callbacks(
            BeforeToolCallEvent(
                agent=agent, selected_tool=None, tool_use=other, invocation_state={}
            )
        )
        assert other["input"]["discord_user_id"] == "someone-else"

        engine._turn_principal = None
        unprincipaled = {
            "toolUseId": "bind-4",
            "name": "character_create",
            "input": {"discord_user_id": "scripted-player"},
        }
        agent.hooks.invoke_callbacks(
            BeforeToolCallEvent(
                agent=agent, selected_tool=None, tool_use=unprincipaled, invocation_state={}
            )
        )
        assert unprincipaled["input"]["discord_user_id"] == "scripted-player"
    finally:
        engine._turn_principal = None
        engine._calls_this_turn = 0
        engine._tool_names_this_turn = []


def test_an_unregistered_tool_name_is_cancelled_with_a_pointer_to_the_real_tools(engine):
    """A hallucinated tool name gets a corrective refusal, not Strands' bare message.
    """
    from strands.hooks import BeforeToolCallEvent

    agent = engine._agent_for("unknown-tool-channel")
    try:
        call = {"toolUseId": "unknown-1", "name": "attribute_status", "input": {}}
        event = BeforeToolCallEvent(agent=agent, selected_tool=None, tool_use=call, invocation_state={})
        agent.hooks.invoke_callbacks(event)
        assert event.cancel_tool
        assert "attribute_status" in event.cancel_tool
        assert "character_sheet" in event.cancel_tool

        # A real, served tool name is untouched by this guard.
        real_call = {"toolUseId": "unknown-2", "name": "character_sheet", "input": {}}
        real_event = BeforeToolCallEvent(
            agent=agent, selected_tool=None, tool_use=real_call, invocation_state={}
        )
        agent.hooks.invoke_callbacks(real_event)
        assert not real_event.cancel_tool
    finally:
        engine._calls_this_turn = 0
        engine._tool_names_this_turn = []


def test_the_agent_carries_the_pinned_output_budget(engine):
    """8192 was pinned in the Hermes profile and must survive the migration."""
    agent = engine._agent_for("budget-channel")
    params = agent.model.get_config().get("params") or {}
    assert params.get("max_tokens") == 8192
    assert params.get("temperature") == 0.4


def test_the_session_is_shared_per_channel_never_per_user(engine):
    """One narrator per channel: every player talks to the same game master."""
    first = engine._agent_for("channel-a")
    second = engine._agent_for("channel-a")
    other = engine._agent_for("channel-b")
    assert first is second
    assert first is not other


@pytest.mark.live
@pytest.mark.asyncio
async def test_a_defence_that_closes_the_attackers_turn_needs_no_manual_recovery(engine):
    """Test a defence that closes the attackers turn needs no manual recovery.
    """
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    from bsh_mcp.service import GameService

    root = engine.config.campaign_root
    game = GameService(root)
    created = game.character_create(
        discord_user_id="live-defence-test",
        name="Rill",
        origin="barbarian",
        backgrounds=["scout", "hunter", "survivor"],
        weapons=["long knife"],
    )
    assert created["ok"], created
    character_id = created["character_id"]
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: live-defence-test\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    npc = game.npc_create(name="Reed Thug", level=1, motive="rob the party")
    assert npc["ok"], npc
    started = game.combat_start(
        pc_ids=[character_id],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="ambush",
    )
    assert started["ok"], started
    # combat_start rolls real initiative here (no scripted roller, unlike the offline
    # tests in tests/test_tools.py), so either side may go first. If the PC won it,
    # open and close that turn first to advance to the enemy, exactly as a real
    # narrator would.
    if game.store.read_state().combat.active_actor == character_id:
        opened_pc = game.combat_begin_turn(character_id)
        assert opened_pc["ok"], opened_pc
        closed_pc = game.combat_end_turn(character_id)
        assert closed_pc["ok"], closed_pc
    opened = game.combat_begin_turn(npc["npc_id"])
    assert opened["ok"], opened
    defended = game.combat_defend(character_id, npc["npc_id"], method="dodge")
    assert defended["ok"], defended
    state = game.store.read_state()
    assert state.combat.active_actor == character_id

    turn = InboundTurn(
        channel_id="live",
        mention=ChannelMessage("Rill", "Rill attacks the reed thug with her long knife."),
    )
    outcome = await engine.run_turn(turn)

    assert outcome.error == "", outcome.error
    if not outcome.withheld:
        assert outcome.narration.strip()
    for leaked in (
        "turn_not_open",
        "not_active_actor",
        "combat_begin_turn",
        "combat_attack",
        "combat_defend",
        "combat_end_turn",
    ):
        assert leaked not in outcome.narration, outcome.narration


@pytest.mark.live
@pytest.mark.asyncio
async def test_a_reported_roll_is_acknowledged_rather_than_repeated(tmp_path):
    """Sampled repeatedly because this is model prose, not a deterministic branch, and the
    fix does not make the planner stop asking a follow-up question -- a genuinely
    failed, noisy attempt reasonably invites \"what now?\" -- it makes that follow-up
    informed.

    The first landed wording measured 4/4 the day it was built, which this test then
    asserted as \"a majority of 4.\" A later, larger same-day recheck (prompted by an
    unrelated live test failing during an unconnected change) found that number was a
    small-sample fluke: 8/20 (40%) across three fresh batches, against 0/8 with the
    sentence removed -- a real, nonzero effect, but nowhere near reliable. The wording
    below replaces the original: the roll-repeat instruction moved out of the same
    paragraph as the unrelated \"treat this as fact\" sentence into its own paragraph,
    with a concrete worked example matching this exact scenario. Two fresh batches
    against the same endpoint measured 20/20 (8 then 12) with the new wording, against
    the same day's 0/8 and 40% baselines -- still not proof of literal determinism from
    one day's sample, but a clearly different, much higher rate than the original
    wording ever showed. The sample size below is widened from the original 4 to 8, and
    the tolerance kept at \"all but one,\" reflecting more statistical power at a
    similar cost rather than asserting the literal 100% one day's sample cannot prove.
    """
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    from narrator.decisions import RequestDecisionPlan

    narration = (
        'The blade remains wedged in the crate after the attempt.\n\nRill rolls STR: 13 vs 12, failure.'
    )
    samples = 8
    acknowledged = 0
    for _ in range(samples):
        engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
        # ``plan_turn`` classifies before every hazard decision, and an engine with no
        # client answers ``None`` -- the fail-closed value, which routes every
        # declaration to the uncategorised confirmation regardless of what the planner
        # would have said. This test measures the planner against the served endpoint,
        # so the classifier has to reach it too; the assignment is the same started-
        # engine seam ``test_probe_classifier`` uses.
        engine._client = object()  # noqa: SLF001 - started-engine seam; only the endpoint is used
        turn = InboundTurn("sample", ChannelMessage("Rill", "I take the cleaver"))
        outcome = await engine.plan_turn(
            turn, (("rill", "Rill"),), recent_narration=narration
        )
        plan = outcome.plan
        if not isinstance(plan, RequestDecisionPlan):
            acknowledged += 1  # proceeding outright is at least as informed as asking
            continue
        question = getattr(plan.decision, "question", "").casefold()
        if "already" in question or "fail" in question:
            acknowledged += 1

    assert acknowledged >= samples - 1, f"only {acknowledged}/{samples} acknowledged the reported roll"


@pytest.mark.live
@pytest.mark.asyncio
async def test_a_real_turn_runs_end_to_end_against_the_live_endpoint(engine):
    """Acceptance criterion 1: a turn runs on Strands against the real server.

    Skipped when the endpoint is absent, so the suite stays runnable without a model.
    Starting one to satisfy a test is not authorized.

    Marked ``live``: this is the one test in this module that actually reaches the
    served model rather than only a local Model Context Protocol subprocess.
    ``package.json``'s ``check:unit:*`` narrator suite excludes ``-m live`` so an
    offline validate run never makes a GPU call, regardless of whether an operator's
    shell happens to export ``BSH_SERVER_PYTHON`` or a live endpoint happens to answer.
    """
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    turn = InboundTurn(
        channel_id="live",
        mention=ChannelMessage(
            "Rill", 'Ossa questions the dock workers about a missing courier.'
        ),
    )
    outcome = await engine.run_turn(turn)

    assert outcome.error == "", outcome.error
    # "failed" means the settler broke twice against a live endpoint; that is an
    # infrastructure defect this test exists to catch, not model taste.
    assert outcome.settle in ("none", "commit", "waive"), outcome.settle
    # A turn either delivers narration or the barrier withholds it. Both are correct
    # outcomes; an empty narration on a delivered turn is not.
    if not outcome.withheld:
        assert outcome.narration.strip()
    assert "<|" not in outcome.narration
    assert "<tool_call" not in outcome.narration
