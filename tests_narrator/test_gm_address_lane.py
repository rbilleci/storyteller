"""The game-master address lane: the designator routes, vocabulary never does.
"""

from __future__ import annotations

from pathlib import Path

from fake_classifier import ClassifyingEngine
from lexical_double import classify_turn

from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.delivery import TurnOutcome
from narrator.policy_types import TrustedScope
from narrator.service import NarratorService

COMPLAINT = "i had attacked and killed him. look at the transcript"


class _Adapter:
    """One addressed turn; every decision surface booby-trapped."""

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self, text: str):
        self.text = text
        self.posted = []

    async def turns(self):
        yield InboundTurn(
            "terminal",
            ChannelMessage(
                "Rill", self.text, ChannelPrincipal("terminal", "terminal-player", "Rill")
            ),
        )

    async def post(self, channel_id, text):
        self.posted.append((channel_id, text))

    async def close(self):
        return None

    async def present_decision(self, view):
        raise AssertionError("a gm-addressed turn must present no decision")

    async def deliver_decision_views(self, views):
        raise AssertionError("a gm-addressed turn must present no decision")

    async def collect_decision(self, views):
        raise AssertionError("a gm-addressed turn must collect no decision")

    async def acknowledge_decision(self, result):
        raise AssertionError("a gm-addressed turn must acknowledge no decision")


class _Engine(ClassifyingEngine):
    """Record what reaches the narrator; refuse to plan anything.

    The classifier comes from ``ClassifyingEngine`` because ``NarratorService``
    withholds every turn from an engine without one. On this lane it should stay
    unused -- the designator decides the route before any classification could -- and
    ``classify_calls`` is what lets a test say so.
    """

    def __init__(self):
        super().__init__()
        self.turns = []

    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    async def plan_turn(self, turn, eligible, resolutions):
        raise AssertionError("a gm-addressed turn must never reach the planner")

    async def run_turn(self, turn, decision_resolutions=()):
        self.turns.append(turn)
        return TurnOutcome("Answered from the record.", ratified=True, withheld=False)


async def test_the_live_complaint_reaches_the_narrator_as_gm_discussion(tmp_path: Path):
    """The exact words that drew a kill confirmation live, now addressed: no
    decision, no planner, the designator stripped, the framing and flag set."""
    adapter = _Adapter(f"@GM {COMPLAINT}")
    engine = _Engine()
    await NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine).run()

    assert adapter.posted == [("terminal", "Answered from the record.")]
    assert len(engine.turns) == 1
    executed = engine.turns[0]
    assert executed.mention.text == COMPLAINT
    assert executed.turn_framing == "gm_discussion"
    assert executed.meta is True
    # Pure syntax, as the design freezes it: the classifier is never consulted, so no
    # verdict about the words can reach this lane's routing decision.
    assert engine.classify_calls == []


async def test_the_same_words_without_the_designator_stay_on_the_risk_route():
    """Test the same words without the designator stay on the risk route.
    """
    verdict = classify_turn(COMPLAINT, scope=TrustedScope("session-1", "market", ()))
    assert verdict.route == "risk"
    assert verdict.risk_category == "violence"


async def test_the_designator_is_configurable_and_multilingual(tmp_path: Path):
    """A French table's ``@MJ``, French content: the lane never reads a word."""
    adapter = _Adapter("@MJ j'avais déjà tué Rade, regarde le journal")
    engine = _Engine()
    config = NarratorConfig(campaign_root=tmp_path, gm_address="MJ")
    await NarratorService(config, adapter, engine).run()

    assert adapter.posted == [("terminal", "Answered from the record.")]
    executed = engine.turns[0]
    assert executed.mention.text == "j'avais déjà tué Rade, regarde le journal"
    assert executed.turn_framing == "gm_discussion"
    assert executed.meta is True
