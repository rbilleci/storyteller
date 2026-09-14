"""Pin the M9 fix: a same-turn guard withhold must settle its own debt.

This file needs no live model and no Strands agent: the fake agent below stands in
for one, in the mold ``tests/test_narrator_units.py`` and
``tests_narrator/test_engine_repetition.py`` already use (``_StubAgent``,
``_seed_ledger`` -- the latter shared via ``tests_narrator/engine_fixtures.py``,
not imported across the ``tests/``/``tests_narrator/`` split itself -- the two
trees still import nothing from each other anywhere in this repository).
"""

from __future__ import annotations

from pathlib import Path

from engine_fixtures import _seed_ledger

from narrator import ledger
from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.decisions import DecisionResolution, HazardResolutionDirective
from narrator.engine import NarratorEngine


class _StubResult:
    """The shape ``Agent.invoke_async`` returns: an object carrying ``.message``."""

    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """A plain reply, no tool calls at all -- the ordinary-turn half of this file."""

    def __init__(self, reply: str = "Narration.") -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": self.reply}]})
        return _StubResult(self.reply)


class _ConfirmedAttackSetupAgent:
    """Reproduces the live sequence's tool shape: setup runs, the roll never does.
    """

    def __init__(self, engine: NarratorEngine, campaign_root: Path, reply: str) -> None:
        self._engine = engine
        self._campaign_root = campaign_root
        self.reply = reply
        self.calls: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": self.reply}]})
        self._engine._tool_events_this_turn.extend(  # noqa: SLF001 - test seam
            [
                {"tool": "npc_create", "ok": True, "error": "", "event_id": 1},
                {"tool": "combat_start", "ok": True, "error": "", "event_id": 2},
                {"tool": "combat_begin_turn", "ok": True, "error": "", "event_id": 3},
            ]
        )
        _seed_ledger(
            self._campaign_root,
            [
                {
                    "seq": 1,
                    "tool": "npc_create",
                    "actor_id": "fishmonger-rade",
                    "reason": "rob the party",
                    "outcome": "created",
                }
            ],
        )
        return _StubResult(self.reply)


def _engine(campaign_root: Path) -> NarratorEngine:
    return NarratorEngine(NarratorConfig(campaign_root=campaign_root))


def _confirmed_attack_resolution(character_id: str = "rill") -> DecisionResolution:
    """The resolution ``resolution_from_submission`` builds for a confirmed violence
    decision -- see ``tests_narrator/test_decisions.py``'s
    ``test_a_confirmed_violence_action_yields_a_hazard_directive``."""
    return DecisionResolution(
        character_id=character_id,
        kind="confirmation",
        answer_kind="confirm",
        selection_id="confirm",
        directive=HazardResolutionDirective(
            kind="hazard_resolution", character_id=character_id, category="violence"
        ),
    )


async def test_a_same_turn_guard_withhold_settles_the_debt_its_own_tools_opened(tmp_path):
    """Against the code before this slice's fix, this test fails: the guard-error
    branch returned ``TurnOutcome(ratified=ledger.is_ratified(...), ...)`` computed
    the instant the guard fired, before anything ever called ``_settle`` -- so
    ``settle_calls`` stays empty, ``outcome.ratified`` reads False, and
    ``ledger.is_ratified(tmp_path)`` stays False after the call returns. The fix
    settles before returning, so all three now read the closed state.
    """
    _seed_ledger(tmp_path, [])
    engine = _engine(tmp_path)
    stub = _ConfirmedAttackSetupAgent(
        engine, tmp_path, reply="Rill draws steel and closes on the fishmonger."
    )
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam

    settle_calls: list[str] = []
    withheld_settles: list[str] = []

    async def stub_settle(narration: str):
        settle_calls.append(narration)
        _seed_ledger(tmp_path, [])
        return ("waive", 1, "")

    async def stub_settle_withheld():
        # ledger_settle really clears campaign/state.json on a successful waive;
        # see tests/test_narrator_units.py's
        # test_the_settle_waive_path_clears_the_ledger_on_the_record for the real path.
        withheld_settles.append("called")
        _seed_ledger(tmp_path, [])
        return ("waive", 1, "")

    engine._settle = stub_settle  # noqa: SLF001 - test seam
    engine._settle_withheld = stub_settle_withheld  # noqa: SLF001 - test seam

    turn = InboundTurn(
        channel_id="c1",
        mention=ChannelMessage("Rill", "I draw my knife and attack the fishmonger"),
    )
    resolution = _confirmed_attack_resolution()

    outcome = await engine.run_turn(
        turn, decision_resolutions=(resolution,), decision_action_fingerprint=""
    )

    # The guard verdict itself is untouched by the fix: the turn still withholds,
    # still carries no narration, and still names the same missing mechanic.
    assert outcome.decision_recovery is True
    assert outcome.withheld is True
    assert outcome.narration == ""
    assert outcome.error == "the confirmed action rolled no resolving mechanic"


    assert [event["tool"] for event in outcome.tool_events] == [
        "npc_create", "combat_start", "combat_begin_turn",
    ]
    assert all(event["ok"] for event in outcome.tool_events)


    assert settle_calls == []
    assert len(withheld_settles) == 1
    assert outcome.settle == "waive"
    assert outcome.settle_attempts == 1
    assert outcome.ratified is True
    assert ledger.is_ratified(tmp_path) is True


async def test_a_guard_withhold_with_no_debt_never_calls_settle(tmp_path):
    """The common case stays cheap: an already-clean ledger never invokes the settler."""
    _seed_ledger(tmp_path, [])
    engine = _engine(tmp_path)
    stub = _StubAgent("Rill hesitates, blade half-drawn.")
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001

    settle_calls: list[str] = []

    async def stub_settle(narration: str):
        settle_calls.append(narration)
        return ("none", 0, "")

    async def stub_settle_withheld():
        settle_calls.append("withheld")
        return ("none", 0, "")

    engine._settle = stub_settle  # noqa: SLF001
    engine._settle_withheld = stub_settle_withheld  # noqa: SLF001

    resolution = _confirmed_attack_resolution()
    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))

    outcome = await engine.run_turn(turn, decision_resolutions=(resolution,))

    assert outcome.decision_recovery is True
    assert outcome.withheld is True
    assert settle_calls == []
    assert outcome.settle == "none"
    assert outcome.ratified is True


async def test_a_framework_fault_mid_tool_loop_also_settles_debt_it_already_opened(tmp_path):
    """The same defect class, one frame up: the agent call itself raises after a
    tool already committed. ``run_turn``'s except-Exception branch used to hardcode
    ``ratified=False`` and never settle either; it now settles the same way."""
    _seed_ledger(tmp_path, [])
    engine = _engine(tmp_path)

    class _RaisingAfterATooCallAgent:
        messages: list[dict] = []

        async def invoke_async(self, text: str):
            engine._tool_events_this_turn.append(  # noqa: SLF001 - test seam
                {"tool": "npc_create", "ok": True, "error": "", "event_id": 1}
            )
            _seed_ledger(tmp_path, [{"seq": 1, "tool": "npc_create"}])
            raise RuntimeError("framework fault mid tool-loop")

    engine._agent_for = lambda channel_id: _RaisingAfterATooCallAgent()  # noqa: SLF001

    settle_calls: list[str] = []
    withheld_settles: list[str] = []

    async def stub_settle(narration: str):
        settle_calls.append(narration)
        _seed_ledger(tmp_path, [])
        return ("waive", 1, "")

    async def stub_settle_withheld():
        withheld_settles.append("called")
        _seed_ledger(tmp_path, [])
        return ("waive", 1, "")

    engine._settle = stub_settle  # noqa: SLF001
    engine._settle_withheld = stub_settle_withheld  # noqa: SLF001

    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "attack the fishmonger"))
    outcome = await engine.run_turn(turn)

    assert outcome.withheld is True
    assert outcome.narration == ""
    assert "framework fault" in outcome.error


    assert settle_calls == []
    assert len(withheld_settles) == 1
    assert outcome.ratified is True
    assert ledger.is_ratified(tmp_path) is True


async def test_stale_debt_from_an_earlier_turn_recovers_at_the_very_next_turn(tmp_path):
    """Test stale debt from an earlier turn recovers at the very next turn.
    """
    _seed_ledger(tmp_path, [{"seq": 9, "tool": "attribute_test", "reason": "stale"}])
    assert ledger.is_ratified(tmp_path) is False

    engine = _engine(tmp_path)
    stub = _StubAgent("The market goes about its business; the fight rages on nearby.")
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001

    async def stub_settle(narration: str):
        _seed_ledger(tmp_path, [])
        return ("waive", 1, "")

    engine._settle = stub_settle  # noqa: SLF001

    turn = InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "I look around the market"))
    outcome = await engine.run_turn(turn)

    assert outcome.withheld is False
    assert outcome.ratified is True
    assert ledger.is_ratified(tmp_path) is True


class _CommittingSettler:
    """A settler that always chooses ``commit``, the branch M10 must make unreachable.
    """

    class _Outcome:
        kind = "commit"
        public_summary = "Rill drew her knife and cut the fishmonger down."
        visible_changes = ("the fishmonger is dead",)
        in_game_time_delta_minutes = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str):
        self.prompts.append(prompt)
        return type("_Settled", (), {"outcome": self._Outcome()})()


async def test_a_withheld_turn_commits_no_scene_record_naming_the_action_it_withheld(tmp_path):
    """Against the code before this slice this test fails on its first assertion:
    ``_settle_once`` is reached, the committing settler returns, and ``_call_tool``
    records ``scene_commit`` carrying the withheld action's summary. After the fix the
    guard branch takes ``_settle_withheld``, which has no commit branch and no model
    call, so the settler is never consulted and the only tool call is ``ledger_settle``.
    """
    _seed_ledger(tmp_path, [])
    engine = _engine(tmp_path)
    stub = _ConfirmedAttackSetupAgent(
        engine, tmp_path, reply="Rill draws steel and closes on the fishmonger."
    )
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam

    settler = _CommittingSettler()
    engine._settle_once = settler  # noqa: SLF001 - test seam

    tool_calls: list[tuple[str, dict]] = []

    def stub_call_tool(name: str, arguments: dict, origin: str = "settle") -> str:
        tool_calls.append((name, arguments))
        _seed_ledger(tmp_path, [])
        return "success"

    engine._call_tool = stub_call_tool  # noqa: SLF001 - test seam

    outcome = await engine.run_turn(
        turn := InboundTurn(
            channel_id="c1",
            mention=ChannelMessage("Rill", "I draw my knife and attack the fishmonger"),
        ),
        decision_resolutions=(_confirmed_attack_resolution(),),
        decision_action_fingerprint="",
    )
    del turn

    # The claim itself: no scene record was written for a turn nobody saw.
    assert [name for name, _ in tool_calls] == ["ledger_settle"]
    assert all(name != "scene_commit" for name, _ in tool_calls)
    # The settler was never even asked, so no model reply could have carried the
    # withheld action into a summary.
    assert settler.prompts == []
    # The waive states only what is true of every withheld turn and never names the
    # action, which is the whole point of an engine-authored reason here.
    reason = tool_calls[0][1]["reason"]
    assert reason == NarratorEngine.WITHHELD_SETTLE_REASON
    for word in ("knife", "fishmonger", "attack", "steel"):
        assert word not in reason.casefold()

    # M9's guarantees are unchanged: the turn still withholds and the debt still closes
    # inside the same turn.
    assert outcome.withheld is True
    assert outcome.decision_recovery is True
    assert outcome.settle == "waive"
    assert outcome.ratified is True
    assert ledger.is_ratified(tmp_path) is True
