"""Pin ``NarratorEngine.run_turn``'s wiring of ``delivery.strip_leaked_notices``.

This file needs no live model and no Strands agent: ``_StubAgent`` stands in for
one, reproduced locally in the mold ``tests_narrator/test_engine_repetition.py``
and ``tests/test_narrator_units.py`` already use.
"""

from __future__ import annotations

from engine_fixtures import _engine_with_stub, _seed_ledger

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.service import PreparedNarrationTurn


class _StubResult:
    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": self.reply}]})
        return _StubResult(self.reply)


async def test_a_declared_action_never_delivers_a_leaked_engine_notice(tmp_path):
    _seed_ledger(tmp_path, [])
    romance_notice = NarratorConfig(campaign_root=tmp_path).romance_boundary_notice
    stub = _StubAgent(f"{romance_notice}\n\nRade lunges with his knife.")
    engine = _engine_with_stub(tmp_path, stub)
    turn = PreparedNarrationTurn(
        source=InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "I kick rade")),
        interaction_cue=None,
        turn_framing="",  # a declared action, exactly "kick rade"'s own shape
    )

    outcome = await engine.run_turn(turn)

    assert outcome.narration == "Rade lunges with his knife."
    assert romance_notice not in outcome.narration
    assert outcome.leaks_scrubbed >= 1


async def test_a_question_shaped_turn_keeps_a_restated_notice(tmp_path):
    """Test a question shaped turn keeps a restated notice.
    """
    _seed_ledger(tmp_path, [])
    romance_notice = NarratorConfig(campaign_root=tmp_path).romance_boundary_notice
    stub = _StubAgent(f"I am asking because {romance_notice}")
    engine = _engine_with_stub(tmp_path, stub)
    turn = PreparedNarrationTurn(
        source=InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "GM, why?")),
        interaction_cue=None,
        turn_framing="gm_discussion",
    )

    outcome = await engine.run_turn(turn)

    assert romance_notice in outcome.narration
