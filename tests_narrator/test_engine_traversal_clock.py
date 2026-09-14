"""This file needs no live model and no Strands agent: ``_StubAgent`` stands in for
one, in the exact mold ``tests_narrator/test_engine_repetition.py`` already
reproduces from ``tests/test_narrator_units.py`` -- duplicated locally rather than
imported across files, the same discipline that mold's own docstring already
states (the two test trees import nothing from each other anywhere in this
repository, and this repository's own test files do not import one another
either). ``_StubAgent`` never calls a tool itself -- the harness has no tool loop
at all -- so every test here already exercises \"a fake model that never advances
the clock itself\" by construction; ``_StubClient`` stands in for the one thing
that does: the engine's own ``scene_commit`` calls through ``self._client``.
"""

from __future__ import annotations

import json
from pathlib import Path

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.engine import NarratorEngine


class _StubResult:
    """The shape ``Agent.invoke_async`` returns: an object carrying ``.message``."""

    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """Stubagent.
    """

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": self._reply}]})
        return _StubResult(self._reply)


#: Duplicated from ``bsh_mcp.models.TRAVERSAL_CLOCK_PREFIX`` for the same reason
#: ``engine.py`` itself duplicates it (see that module's own constant): this tree
#: installs ``mcp`` 1.29, incompatible with ``bsh_mcp``'s own ``mcp``>=2 requirement.
_TRAVERSAL_CLOCK_PREFIX = "travel-"


class _StubClient:
    """Applies a ``scene_commit`` call's ``clock_updates`` directly to
    ``state.json``, the only tool call ``_advance_stalled_traversal`` (or
    ``_settle``/``_sweep``, unused here) ever issues through ``self._client``.

    Implements the real restart-then-apply formula ``src/bsh_mcp/service.py``'s
    own ``clock_updates`` handling uses, not a naive clamp-add: a traversal clock
    (``_TRAVERSAL_CLOCK_PREFIX``) already at or above its own segment count
    restarts at zero before the delta applies. An independent audit
    (result-1.json, blocker AUD-1) found the original, naive-clamp-add version of
    this stub could not see the interaction between that restart branch and
    ``_advance_stalled_traversal`` at all, which is exactly the gap that let a
    real defect (AUD-1) through undetected offline. Every test in this file now
    exercises the same formula the engine's own fallback calls run against,
    whether the "model's own call" is simulated through this client directly
    (the adversarial AUD-1 regression tests below) or fired by the fallback
    itself.
    """

    def __init__(self, campaign_root: Path) -> None:
        self._state_path = campaign_root / "campaign" / "state.json"

    def call_tool_sync(self, tool_use_id: str, name: str, arguments: dict):
        assert name == "scene_commit", name
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        clocks = payload.setdefault("clocks", [])
        for clock_id, delta in (arguments.get("clock_updates") or {}).items():
            for entry in clocks:
                if entry.get("id") == clock_id:
                    base = (
                        0
                        if clock_id.startswith(_TRAVERSAL_CLOCK_PREFIX)
                        and entry["filled"] >= entry["segments"]
                        else entry["filled"]
                    )
                    entry["filled"] = max(0, min(entry["segments"], base + int(delta)))
                    break
        self._state_path.write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success"}


def _seed_state(campaign_root: Path, *, clocks: list[dict], fiction_debt=()) -> None:
    campaign = campaign_root / "campaign"
    campaign.mkdir(exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps({"fiction_debt": list(fiction_debt), "clocks": clocks}),
        encoding="utf-8",
    )


def _engine_with_stub(campaign_root: Path, stub: _StubAgent) -> NarratorEngine:
    """An engine whose agent construction and MCP client are both replaced, so
    no framework and no live server are needed."""
    engine = NarratorEngine(NarratorConfig(campaign_root=campaign_root))
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam
    engine._client = _StubClient(campaign_root)  # noqa: SLF001 - test seam
    return engine


def _turn(channel_id: str, text: str) -> InboundTurn:
    return InboundTurn(channel_id=channel_id, mention=ChannelMessage("Rill", text))


def _clock_filled(campaign_root: Path, clock_id: str) -> int:
    payload = json.loads((campaign_root / "campaign" / "state.json").read_text())
    return next(entry["filled"] for entry in payload["clocks"] if entry["id"] == clock_id)


_CLOCK_ID = "travel-the-north-camp-the-south-camp"
_DECLARATION = "I keep pressing on across the causeway toward the south camp."


async def test_a_repeated_identical_declaration_advances_a_stalled_clock_every_time(
    tmp_path,
):
    """Test a repeated identical declaration advances a stalled clock every time.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""}],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("You keep to the raised stones."))

    # First call on this channel: nothing precedes it, so it is not yet a
    # "repeated" declaration and the stalled clock is correctly left alone.
    await engine.run_turn(_turn("c1", _DECLARATION))
    assert _clock_filled(tmp_path, _CLOCK_ID) == 1
    assert engine._traversal_log[-1]["declaration_repeat"] is False  # noqa: SLF001
    assert engine._traversal_log[-1]["advanced"] == []  # noqa: SLF001

    # Every later identical declaration is now a repeat, and the model (the
    # stub) never once calls scene_commit, so only the engine's own fallback can
    # be moving the clock forward.
    fills: list[int] = []
    for _ in range(3):
        await engine.run_turn(_turn("c1", _DECLARATION))
        fills.append(_clock_filled(tmp_path, _CLOCK_ID))

    assert fills == [2, 3, 4], "monotonic, one segment per repeated declaration"
    assert engine._traversal_log[-1]["advanced"] == [_CLOCK_ID]  # noqa: SLF001
    # The clock reached its own segment count: nothing further to nudge, so a
    # fourth repeat must not push it past its own cap or record a fifth advance.
    await engine.run_turn(_turn("c1", _DECLARATION))
    assert _clock_filled(tmp_path, _CLOCK_ID) == 4
    assert engine._traversal_log[-1]["checked"] == []  # noqa: SLF001 - already complete


async def test_a_freshly_opened_clock_at_zero_fill_earns_its_own_first_segment(
    tmp_path,
):
    """The turn that opens a traversal clock already earns progress, with no
    separate repeat check needed: opening only ever happens on a genuine
    incomplete movement declaration.
    """
    _seed_state(tmp_path, clocks=[])

    class _OpeningStubAgent(_StubAgent):
        """Simulates the model's own ``new_clocks`` call landing mid-turn, at
        zero fill -- the shape a model that forgot the same-call
        ``clock_updates`` nudge from skills/bsh-gm/SKILL.md would produce."""

        async def invoke_async(self, text: str):
            payload = json.loads((tmp_path / "campaign" / "state.json").read_text())
            payload["clocks"] = [
                {"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 0, "note": ""}
            ]
            (tmp_path / "campaign" / "state.json").write_text(json.dumps(payload))
            return await super().invoke_async(text)

    engine = _engine_with_stub(tmp_path, _OpeningStubAgent("The causeway stretches ahead."))
    await engine.run_turn(_turn("c1", _DECLARATION))

    assert _clock_filled(tmp_path, _CLOCK_ID) == 1
    assert engine._traversal_log[-1]["advanced"] == [_CLOCK_ID]  # noqa: SLF001
    assert engine._traversal_log[-1]["declaration_repeat"] is False  # noqa: SLF001


async def test_a_non_repeated_declaration_leaves_a_stalled_clock_alone(tmp_path):
    """A stalled open clock must not silently fill itself on an unrelated turn."""
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""}],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("You keep to the raised stones."))
    await engine.run_turn(_turn("c1", _DECLARATION))
    assert _clock_filled(tmp_path, _CLOCK_ID) == 1

    stub2 = _StubAgent("You pause to look over the market stalls instead.")
    engine._agent_for = lambda channel_id: stub2  # noqa: SLF001
    await engine.run_turn(_turn("c1", "I stop to look over the market stalls instead."))

    assert _clock_filled(tmp_path, _CLOCK_ID) == 1, (
        "a genuinely different declaration must never advance a stalled clock"
    )
    assert engine._traversal_log[-1]["declaration_repeat"] is False  # noqa: SLF001
    assert engine._traversal_log[-1]["advanced"] == []  # noqa: SLF001


async def test_the_fallback_never_touches_an_unrelated_clock(tmp_path):
    """Adversarial case: a non-traversal clock sitting beside an open traversal
    clock must never be advanced, prefix-gated by ``TRAVERSAL_CLOCK_PREFIX``.
    """
    _seed_state(
        tmp_path,
        clocks=[
            {"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""},
            {"id": "alarm", "name": "The garrison's suspicion", "segments": 5, "filled": 2, "note": ""},
        ],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("You keep to the raised stones."))
    await engine.run_turn(_turn("c1", _DECLARATION))
    await engine.run_turn(_turn("c1", _DECLARATION))

    assert _clock_filled(tmp_path, _CLOCK_ID) == 2
    assert _clock_filled(tmp_path, "alarm") == 2, "an unrelated clock must never move"


async def test_a_clock_the_model_already_advanced_is_never_double_counted(tmp_path):
    """A model that already called ``clock_updates`` itself must not be
    second-guessed: the fallback only ever nudges a clock that made zero
    progress, never one that already moved.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""}],
    )

    class _SelfAdvancingStubAgent(_StubAgent):
        """Simulates the model's own compliant ``clock_updates`` call."""

        async def invoke_async(self, text: str):
            payload = json.loads((tmp_path / "campaign" / "state.json").read_text())
            for entry in payload["clocks"]:
                if entry["id"] == _CLOCK_ID:
                    entry["filled"] += 2
            (tmp_path / "campaign" / "state.json").write_text(json.dumps(payload))
            return await super().invoke_async(text)

    engine = _engine_with_stub(
        tmp_path, _SelfAdvancingStubAgent("You cross two more spans of the causeway.")
    )
    await engine.run_turn(_turn("c1", _DECLARATION))

    assert _clock_filled(tmp_path, _CLOCK_ID) == 3, "the model's own +2, untouched by the fallback"
    assert engine._traversal_log[-1]["advanced"] == []  # noqa: SLF001


async def test_a_withheld_turn_never_advances_a_traversal_clock(tmp_path):
    """The same "nobody saw it" discipline the sweep already applies: a turn
    withheld behind an open ledger must not silently record crossing progress.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""}],
        fiction_debt=[{"seq": 1, "tool": "attribute_test"}],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("You keep to the raised stones."))

    async def failing_settle(narration):
        return ("failed", 2, "settler stub broke twice")

    engine._settle = failing_settle  # noqa: SLF001 - test seam
    outcome = await engine.run_turn(_turn("c1", _DECLARATION))

    assert outcome.withheld is True
    assert _clock_filled(tmp_path, _CLOCK_ID) == 1
    assert engine._traversal_log[-1] == {  # noqa: SLF001
        "checked": [],
        "advanced": [],
        "declaration_repeat": False,
    }


async def test_channels_are_compared_independently_for_declaration_repeats(tmp_path):
    """The same declaration delivered fresh on a different channel is not a
    repeat there, matching how ``_avoid_repeat`` already isolates channels for
    narration.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 4, "filled": 1, "note": ""}],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("You keep to the raised stones."))
    await engine.run_turn(_turn("channel-a", _DECLARATION))

    stub_b = _StubAgent("You keep to the raised stones.")
    engine._agent_for = lambda channel_id: stub_b  # noqa: SLF001
    await engine.run_turn(_turn("channel-b", _DECLARATION))

    assert _clock_filled(tmp_path, _CLOCK_ID) == 1
    assert engine._traversal_log[-1]["declaration_repeat"] is False  # noqa: SLF001


# -- AUD-1 regressions (result-1.json): the reopening/restart interaction ----


async def test_a_genuine_same_turn_restart_is_never_double_credited(tmp_path):
    """Test a genuine same turn restart is never double credited.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 3, "filled": 3, "note": ""}],
    )
    declaration = "I start back across the causeway toward the north camp."
    engine = _engine_with_stub(tmp_path, _StubAgent("Rill turns to start back."))
    client = engine._client  # noqa: SLF001 - shared so the model's own simulated

    # First turn: establishes the prior declaration on this channel. The stub
    # agent never touches the clock, so it stays full going into the second turn.
    await engine.run_turn(_turn("c1", declaration))
    assert _clock_filled(tmp_path, _CLOCK_ID) == 3

    class _RestartingStubAgent(_StubAgent):
        """The model's own turn: a genuine restart-then-progress call, routed
        through the real client so the restart-then-apply formula applies
        exactly as src/bsh_mcp/service.py implements it."""

        async def invoke_async(self, text: str):
            client.call_tool_sync("model-sim", "scene_commit", {"clock_updates": {_CLOCK_ID: 2}})
            return await super().invoke_async(text)

    engine._agent_for = lambda channel_id: _RestartingStubAgent(  # noqa: SLF001
        "Rill makes real progress back across the causeway."
    )
    await engine.run_turn(_turn("c1", declaration))

    assert engine._traversal_log[-1]["declaration_repeat"] is True, (  # noqa: SLF001
        "the test must actually exercise the repeat-detected path the live bug depended on"
    )
    assert _clock_filled(tmp_path, _CLOCK_ID) == 2, (
        "the model's own genuine +2 restart must stand alone: no unearned extra +1"
    )
    assert engine._traversal_log[-1]["advanced"] == []  # noqa: SLF001


async def test_a_reopened_clock_at_zero_earns_credit_even_without_a_detected_repeat(
    tmp_path,
):
    """The mirror-image gap AUD-1 also found: before the fix, a clock reset to
    zero by the model's own reopen call (with no same-call progress of its own)
    got no fallback help at all unless this turn's declaration happened to score
    as a repeat of whatever preceded it -- unlike a genuinely brand-new clock,
    which always earns its first segment regardless. A reopening declaration
    (a different direction, worded differently from the leg that just finished)
    must not depend on textual similarity to earn its own first segment either.
    """
    _seed_state(
        tmp_path,
        clocks=[{"id": _CLOCK_ID, "name": "Crossing the causeway", "segments": 3, "filled": 3, "note": ""}],
    )
    engine = _engine_with_stub(tmp_path, _StubAgent("Rill finishes the outbound leg."))
    client = engine._client  # noqa: SLF001

    await engine.run_turn(_turn("c1", "I keep pressing on across the causeway toward the south camp."))
    assert _clock_filled(tmp_path, _CLOCK_ID) == 3

    class _BareResetStubAgent(_StubAgent):
        """The model's own reopen call with no same-call progress -- the shape
        a half-followed SKILL.md reopen instruction would produce."""

        async def invoke_async(self, text: str):
            client.call_tool_sync("model-sim", "scene_commit", {"clock_updates": {_CLOCK_ID: -3}})
            return await super().invoke_async(text)

    engine._agent_for = lambda channel_id: _BareResetStubAgent(  # noqa: SLF001
        "Rill turns and starts the trek back toward the north camp."
    )
    # Deliberately dissimilar to the prior declaration (score 0.48, well under
    # REPETITION_SIMILARITY_THRESHOLD's 0.85) -- unlike the live condition that
    # accidentally masked this bug, this turn must NOT read as a repeat.
    await engine.run_turn(
        _turn("c1", "I turn and start the long walk back toward the distant hills.")
    )

    assert engine._traversal_log[-1]["declaration_repeat"] is False  # noqa: SLF001
    assert _clock_filled(tmp_path, _CLOCK_ID) == 1, (
        "a reopened clock earns its own first segment the same way a brand-new one does"
    )
    assert engine._traversal_log[-1]["advanced"] == [_CLOCK_ID]  # noqa: SLF001
