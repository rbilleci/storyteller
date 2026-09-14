"""Engine plan turn failure. Synthetic fixtures exercise this contract."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fake_classifier import fake_classify_intent

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.decisions import (
    ConfirmationDraft,
    CurrentAudience,
    PlannerPolicyError,
    PlanOutcome,
    ProceedPlan,
    RequestDecisionPlan,
)
from narrator.engine import (
    PLANNER_FAILURE_REASONS,
    DecisionPlanningError,
    NarratorEngine,
    _planner_failure_reason,
)

#: Routes ``planner`` through ``tests_narrator/lexical_double.py``: no hazard, so the
#: floor this falls back to answers ``proceed``.
_BENIGN = "I try to force the door open."
#: Routes ``risk``/``violence``: the floor answers with the engine-authored confirmation.
_VIOLENT = "I attack rade"


def _engine(tmp_path: Path) -> NarratorEngine:
    """An unstarted engine that can still route a turn offline.

    Mirrors ``test_engine_promised_check_detection._engine``: an unstarted engine's own
    ``classify_intent`` answers ``None``, the fail-closed value, so without the double
    every declaration here would draw the uncategorised confirmation instead of the
    verdict under test.
    """
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine.classify_intent = fake_classify_intent
    return engine


def _plan(engine: NarratorEngine, declaration: str, *, retry: bool):
    turn = InboundTurn("probe", ChannelMessage("Rill", declaration))
    return asyncio.run(
        engine.plan_turn(
            turn, eligible_characters=(("rill", "Rill"),), prior_resolutions=(), retry=retry,
        )
    )


def _always_failing(engine: NarratorEngine, monkeypatch, error: BaseException) -> list[bool]:
    """Stub ``_plan_once`` to raise on both attempts; returns the recorded repair flags."""
    attempts: list[bool] = []

    async def fake_plan_once(prompt, *, repair=False):
        attempts.append(repair)
        raise error

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)
    return attempts


# -- _planner_failure_reason ---------------------------------------------------------


def test_every_reason_is_one_of_the_declared_vocabulary():
    assert _planner_failure_reason(TimeoutError()) == "planner_timeout"
    assert _planner_failure_reason(PlannerPolicyError("refused")) == "planner_policy"
    assert _planner_failure_reason(RuntimeError("something else")) == "planner"
    assert {
        _planner_failure_reason(error)
        for error in (TimeoutError(), PlannerPolicyError("x"), RuntimeError("y"))
    } <= set(PLANNER_FAILURE_REASONS)


def test_a_schema_rejection_reports_the_schema_reason(tmp_path: Path, monkeypatch):
    """The model answered, but not in a shape ``PlanOutcome`` accepts. Driven through
    ``plan_turn`` rather than by hand-raising a ``ValidationError``, so the branch that
    actually classifies it is the one measured."""
    engine = _engine(tmp_path)

    async def fake_plan_once(prompt, *, repair=False):
        return {"plan": {"kind": "not_a_plan_kind"}}

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)

    with pytest.raises(DecisionPlanningError) as raised:
        _plan(engine, _BENIGN, retry=False)

    assert raised.value.reason == "planner_schema"
    assert engine._plan_log[-1]["reason"] == "planner_schema"


def test_no_proposal_at_all_reports_the_policy_reason(tmp_path: Path, monkeypatch):
    """``_plan_once`` returning ``None`` raises ``PlannerPolicyError`` inside the loop."""
    engine = _engine(tmp_path)

    async def fake_plan_once(prompt, *, repair=False):
        return None

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)

    with pytest.raises(DecisionPlanningError) as raised:
        _plan(engine, _BENIGN, retry=False)

    assert raised.value.reason == "planner_policy"


def test_a_timeout_reports_the_timeout_reason(tmp_path: Path, monkeypatch):
    engine = _engine(tmp_path)
    _always_failing(engine, monkeypatch, TimeoutError())

    with pytest.raises(DecisionPlanningError) as raised:
        _plan(engine, _BENIGN, retry=False)

    assert raised.value.reason == "planner_timeout"


def test_unreadable_prompt_inputs_report_the_inputs_reason(tmp_path: Path, monkeypatch):
    """The digest/prompt build failing is a different failure from the model call
    failing, and it now says so instead of sharing one category with it."""
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        "narrator.canon.render_digest",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("campaign root is gone")),
    )

    with pytest.raises(DecisionPlanningError) as raised:
        _plan(engine, _BENIGN, retry=False)

    assert raised.value.reason == "planner_inputs"
    assert engine._plan_log[-1] == {
        "kind": "failed", "promised_unbound_check": False, "escalation": "not_triggered",
        "final_kind": "failed", "reason": "planner_inputs", "disposition": "fault",
    }


# -- _plan_log records a failure ------------------------------------------------------


def test_a_failed_planning_attempt_is_no_longer_silent(tmp_path: Path, monkeypatch):
    """``_plan_log`` recorded only successes, so the one thing an operator most needs to
    explain -- a session whose planner kept failing -- was the one thing the instrument
    said nothing about. ``kind`` is ``failed``, outside the decision vocabulary
    ``narrator.soak_harness`` counts, so a failure can never read as a decision.
    """
    engine = _engine(tmp_path)
    _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))

    with pytest.raises(DecisionPlanningError):
        _plan(engine, _BENIGN, retry=False)

    assert len(engine._plan_log) == 1
    assert engine._plan_log[-1]["kind"] == "failed"
    assert engine._plan_log[-1]["promised_unbound_check"] is False
    assert engine._plan_log[-1]["reason"] == "planner"
    assert engine._plan_log[-1]["disposition"] == "fault"


# -- the bounded retry floor ----------------------------------------------------------


def test_the_first_non_retry_failure_still_faults(tmp_path: Path, monkeypatch):
    """The scope boundary. Ordinary play must not silently downgrade to the floor: a
    planner that fails on a freshly typed declaration still stops the turn, which is
    what makes the fallback below a recovery step rather than a new default.
    """
    engine = _engine(tmp_path)
    attempts = _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))

    with pytest.raises(DecisionPlanningError):
        _plan(engine, _BENIGN, retry=False)

    assert attempts == [False, True]


def test_a_retry_that_fails_twice_falls_back_to_the_floor(tmp_path: Path, monkeypatch):
    """Both attempts spent, so the narrowed re-ask has already had its chance. Instead
    of a second identical fault at a player who typed exactly what the notice
    recommended, the engine answers with its own floor -- ``proceed`` for a declaration
    carrying no hazard.
    """
    engine = _engine(tmp_path)
    attempts = _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))

    result = _plan(engine, _BENIGN, retry=True)

    assert attempts == [False, True]
    assert isinstance(result.plan, ProceedPlan)
    assert engine._plan_log[-1]["kind"] == "failed"
    assert engine._plan_log[-1]["disposition"] == "floor"


def test_the_retry_floor_goes_through_verify_plan_and_surfaces_what_it_authors(
    tmp_path: Path, monkeypatch
):
    """The safety claim, and the reason it is stated against the call rather than the
    verdict. Today a risk-routed declaration is answered by the branch *above* the
    planner loop and never reaches this fallback at all, so no live declaration can
    currently drive the floor here to a confirmation -- asserting on the returned kind
    would pass whether or not the fallback existed. What must hold is the composition:
    the fallback returns ``verify_plan(proceed, ...)``, on this turn's own declaration,
    session, fight, scene and verdict, and hands back whatever that authors. A fallback
    that returned a bare ``ProceedPlan`` instead would turn any future route change into
    a silent consent bypass.
    """
    engine = _engine(tmp_path)
    _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))
    calls: list[tuple] = []
    floored = PlanOutcome(
        plan=RequestDecisionPlan(
            kind="request",
            decision=ConfirmationDraft(
                kind="confirmation", question="Really?", audience=CurrentAudience(kind="current")
            ),
        )
    )

    async def recording_verify_plan(proposal, declaration, session, **kwargs):
        calls.append((proposal, declaration, session, kwargs))
        return floored

    monkeypatch.setattr("narrator.engine.verify_plan", recording_verify_plan)

    result = _plan(engine, _BENIGN, retry=True)

    assert result is floored
    assert len(calls) == 1
    proposal, declaration, session, kwargs = calls[0]
    assert isinstance(proposal.plan, ProceedPlan)
    assert declaration == _BENIGN
    assert session.effective_action == _BENIGN
    assert set(kwargs) == {"combat", "scope", "classify", "policy"}
    assert kwargs["classify"] is engine.classify_intent
    assert kwargs["policy"].route == "planner"


def test_a_violent_declaration_never_reaches_the_planner_loop_at_all(
    tmp_path: Path, monkeypatch
):
    """Why the test above states its claim against the call rather than the verdict.

    ``requires_risk_confirmation`` intercepts a risk-routed declaration above the loop,
    so the fallback's own floor is only ever reached by declarations the floor lets
    proceed. That is the standing arrangement, not a property of the fallback, and it is
    pinned here so a future route change that lands violence in the loop is caught by a
    failing test rather than by a table that was never asked.
    """
    engine = _engine(tmp_path)
    attempts = _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))

    result = _plan(engine, _VIOLENT, retry=True)

    assert attempts == []
    assert isinstance(result.plan, RequestDecisionPlan)
    assert isinstance(result.plan.decision, ConfirmationDraft)


def test_a_retry_whose_floor_also_raises_does_not_return_a_bare_proceed(
    tmp_path: Path, monkeypatch
):
    """The fallback is bounded at one extra call. A floor that itself fails propagates
    to the service's generic planner fault rather than being caught and downgraded --
    an unbounded ladder of fallbacks is how a consent gate quietly disappears.
    """
    engine = _engine(tmp_path)
    _always_failing(engine, monkeypatch, RuntimeError("endpoint hiccup"))

    async def exploding_verify_plan(*args, **kwargs):
        raise RuntimeError("the floor is unreachable too")

    monkeypatch.setattr("narrator.engine.verify_plan", exploding_verify_plan)

    with pytest.raises(RuntimeError, match="the floor is unreachable too"):
        _plan(engine, _BENIGN, retry=True)
