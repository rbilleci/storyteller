"""The hazard assessor's typed contract, and the engine seams that consume it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.assess import (  # noqa: E402
    HazardAssessment,
    assessment_downgrades,
    assessment_prompt,
)
from narrator.channels.base import ChannelMessage, InboundTurn  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.decisions import RequestDecisionPlan  # noqa: E402
from narrator.engine import NarratorEngine  # noqa: E402
from narrator.policy_types import CombatSnapshot, TrustedScope  # noqa: E402


def test_only_a_valid_non_declaration_verdict_downgrades():
    assert assessment_downgrades(None) is False
    assert assessment_downgrades(HazardAssessment(kind="declaration", reason="acts now")) is False
    for kind in ("report", "question", "table_talk"):
        assert assessment_downgrades(HazardAssessment(kind=kind, reason="not an act")) is True


def test_the_prompt_carries_typed_scene_facts_and_the_full_json_shape():
    scope = TrustedScope(
        "session-1", "market", ("rade", "orso-pell"), ("maren",),
        (("rade", "dead"), ("orso-pell", "alive")),
    )
    fight = CombatSnapshot(
        active=True, round=1, active_actor="maren",
        order=("maren", "reed-thug"), sides=(("maren", "pc"), ("reed-thug", "npc")),
    )
    prompt = assessment_prompt("i attacked him", scope, fight)
    assert "i attacked him" in prompt
    assert "rade (dead)" in prompt and "orso-pell (alive)" in prompt
    assert "reed-thug" in prompt
    # Mirrors settle_prompt's contract note: the decoder enforces the schema, but the
    # model only ever sees this prompt, so the full shape must be stated.
    assert '"kind"' in prompt and "declaration|report|question|table_talk" in prompt

    bare = assessment_prompt("i attacked him")
    assert "No fight is currently running." in bare


async def test_engine_risk_floor_defers_to_a_typed_non_declaration_verdict(tmp_path: Path):
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))
    engine._client = object()  # noqa: SLF001 - started-engine seam; no live call is made

    async def _report(prompt):
        return HazardAssessment(kind="report", reason="disputes past events")

    engine._assess_once = _report  # noqa: SLF001
    outcome = await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "i attacked him")), (), ()
    )
    assert outcome.plan.kind == "proceed"

    async def _declaration(prompt):
        return HazardAssessment(kind="declaration", reason="the character acts now")

    engine._assess_once = _declaration  # noqa: SLF001
    outcome = await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "Punch a trader")), (), ()
    )
    assert isinstance(outcome.plan, RequestDecisionPlan)
    assert outcome.plan.decision.kind == "confirmation"

    async def _broken(prompt):
        raise RuntimeError("endpoint down")

    engine._assess_once = _broken  # noqa: SLF001
    outcome = await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "Punch a trader")), (), ()
    )
    assert isinstance(outcome.plan, RequestDecisionPlan)  # every fault keeps the floor


async def test_an_unstarted_engine_never_builds_an_assessor_request(tmp_path: Path):
    """The offline suites drive ``plan_turn`` on engines that never called
    ``start()``; the suite contract says they reach no live endpoint, so the
    assessor must not fire on an unstarted engine at all."""
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, max_decision_rounds=1))

    async def _unexpected(prompt):
        raise AssertionError("an unstarted engine must not assess")

    engine._assess_once = _unexpected  # noqa: SLF001
    outcome = await engine.plan_turn(
        InboundTurn("terminal", ChannelMessage("Rill", "Punch a trader")), (), ()
    )
    assert isinstance(outcome.plan, RequestDecisionPlan)
    assert outcome.plan.decision.kind == "confirmation"
